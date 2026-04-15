from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from exp.shared.common import (
    CACHE_DIR,
    CORRUPTION_FAMILIES,
    FIGURES_DIR,
    RESULTS_DIR,
    SEEDS,
    SEVERITIES,
    append_jsonl,
    ensure_dir,
    experiment_artifacts,
    gpu_memory_gb,
    load_json,
    now,
    save_json,
    save_system_info,
    stage_logger,
    write_skipped,
)
from exp.shared.data import (
    CIFAR10_CLASSNAMES,
    CIFAR100_CLASSNAMES,
    FeatureBundle,
    load_feature_bundle,
    prepare_all_features,
)
from exp.shared.methods import (
    build_text_bank,
    family_posterior,
    load_openclip,
    score_clean_ensemble,
    score_family_residual,
    score_generic_residual,
    score_naive,
    score_zero_shot,
)
from exp.shared.metrics import bootstrap_confidence_interval, cosine_margin, expected_calibration_error, summarize, top1_accuracy


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bundle_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pt"


def _load_clean_and_corruptions(dataset_name: str) -> dict:
    clean_name = f"{dataset_name}_test_clean"
    clean = load_feature_bundle(_bundle_path(clean_name))
    corruptions = {}
    for family in CORRUPTION_FAMILIES:
        for severity in SEVERITIES:
            key = f"{dataset_name}_{family}_severity_{severity}"
            corruptions[(family, severity)] = load_feature_bundle(_bundle_path(key))
    return {"clean": clean, "corruptions": corruptions}


def _load_pilot_seed(seed: int) -> dict:
    split = load_json(Path("data") / "proxy_splits" / f"seed_{seed}.json")
    clean = load_feature_bundle(CACHE_DIR / "pilot" / f"seed_{seed}_clean.pt")
    corrupted = load_feature_bundle(CACHE_DIR / "pilot" / f"seed_{seed}_corrupted.pt")
    return {"split": split, "clean": clean, "corrupted": corrupted}


def _group_pilot_corruptions(bundle: FeatureBundle) -> dict[str, FeatureBundle]:
    groups = {}
    meta = bundle.metadata
    for family in CORRUPTION_FAMILIES:
        idx = [i for i, item in enumerate(meta) if item["family"] == family]
        groups[family] = FeatureBundle(
            features=bundle.features[idx],
            labels=bundle.labels[idx],
            indices=bundle.indices[idx],
            metadata=[meta[i] for i in idx],
        )
    return groups


def _slice_clean_indices(bundle: FeatureBundle, source_indices: list[int]) -> FeatureBundle:
    wanted = {int(x): i for i, x in enumerate(source_indices)}
    idx = [i for i, src in enumerate(bundle.indices.tolist()) if int(src) in wanted]
    metadata = bundle.metadata if len(bundle.metadata) == len(bundle.indices) else [{"family": "clean"} for _ in range(len(bundle.indices))]
    return FeatureBundle(
        features=bundle.features[idx],
        labels=bundle.labels[idx],
        indices=bundle.indices[idx],
        metadata=[metadata[i] for i in idx],
    )


def _slice_corrupted_indices(grouped: dict[str, FeatureBundle], source_indices: list[int]) -> dict[str, FeatureBundle]:
    wanted = set(int(x) for x in source_indices)
    output = {}
    for family, bundle in grouped.items():
        idx = [i for i, src in enumerate(bundle.indices.tolist()) if int(src) in wanted]
        output[family] = FeatureBundle(
            features=bundle.features[idx],
            labels=bundle.labels[idx],
            indices=bundle.indices[idx],
            metadata=[bundle.metadata[i] for i in idx],
        )
    return output


def _predict_family(features: torch.Tensor, q_vectors: torch.Tensor, beta: float) -> torch.Tensor:
    return family_posterior(features, q_vectors, beta=beta).argmax(dim=1)


def _proxy_objective(
    base_scores_clean: torch.Tensor,
    method_scores_clean: torch.Tensor,
    method_scores_corr: dict[str, torch.Tensor],
    base_scores_corr: dict[str, torch.Tensor],
) -> float:
    pseudo_labels = base_scores_clean.argmax(dim=1)
    clean_consistency = (method_scores_clean.argmax(dim=1) == pseudo_labels).float().mean().item()
    gains = []
    for family in CORRUPTION_FAMILIES:
        base_probs = torch.softmax(base_scores_corr[family], dim=1)
        method_probs = torch.softmax(method_scores_corr[family], dim=1)
        gains.append(
            (
                method_probs[torch.arange(len(pseudo_labels)), pseudo_labels]
                - base_probs[torch.arange(len(pseudo_labels)), pseudo_labels]
            ).mean().item()
        )
    return float(np.mean(gains) + 0.5 * clean_consistency)


def _seed_frozen_configuration(seed: int, proxy_grid: list[dict], global_frozen: dict) -> dict:
    seed_rows = [row for row in proxy_grid if row["seed"] == seed]
    if not seed_rows:
        return dict(global_frozen)
    ranked = sorted(
        seed_rows,
        key=lambda row: (
            row["proxy_objective"],
            row["family_posterior_accuracy"],
            row["variant"] == "strong",
        ),
        reverse=True,
    )
    selected = ranked[0]
    return {
        "seed": seed,
        "variant": selected["variant"],
        "beta": selected["beta"],
        "alpha": selected["alpha"],
        "lambda": selected["lambda"],
        "proxy_objective": selected["proxy_objective"],
        "family_posterior_accuracy": selected["family_posterior_accuracy"],
        "global_proxy_objective_mean": global_frozen["proxy_objective_mean"],
    }


def run_prepare_data() -> None:
    artifacts = experiment_artifacts("prepare_data")
    logger = stage_logger("prepare_data")
    device = _device()
    try:
        env_checks = []
        for command in ["nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "nproc", "free -h"]:
            env_checks.append(logger.log_command(command, cwd=Path.cwd()))
        bundle = load_openclip(device)
        start = now()
        manifest = prepare_all_features(bundle.model, bundle.preprocess, device)
        runtime = now() - start
        pilot_counts = {}
        for seed in SEEDS:
            split = load_json(Path("data") / "proxy_splits" / f"seed_{seed}.json")
            pilot_counts[str(seed)] = {
                "pilot_holdout": len(split["pilot_holdout_indices"]),
                "proxy": len(split["proxy_indices"]),
            }
        logger.log(f"Prepared cache entries={len(manifest)} proxy_counts={pilot_counts}")
        payload = {
            "experiment": "prepare_data",
            "model": {"name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
            "num_cached_artifacts": len(manifest),
            "runtime_seconds": runtime,
            "gpu_memory_gb": gpu_memory_gb(),
            "cache_manifest": manifest,
            "environment_checks": env_checks,
            "proxy_split_counts": pilot_counts,
        }
        save_json(artifacts.config_path, {"device": str(device), "cache_dir": str(CACHE_DIR)})
        save_json(artifacts.results_path, payload)
        save_system_info()
    finally:
        logger.close()


def run_pilot_gate() -> None:
    artifacts = experiment_artifacts("pilot_gate")
    logger = stage_logger("pilot_gate")
    device = _device()
    try:
        bundle = load_openclip(device)
        text_bank = {
            "cifar10": build_text_bank(bundle, CIFAR10_CLASSNAMES),
        }
        search_records = []
        pilot_seed_results = []
        proxy_grid = []
        beta_values = [2.0, 5.0, 10.0]
        alpha_values = [0.05, 0.10, 0.20]
        lambda_values = [0.25, 0.50, 1.00]
        variants = ["base", "strong"]
        global_scores = defaultdict(list)

        for seed in SEEDS:
            seed_data = _load_pilot_seed(seed)
            logger.log(
                f"Seed {seed} pilot_counts clean={len(seed_data['clean'].labels)} corrupted={len(seed_data['corrupted'].labels)} "
                f"pilot_holdout={len(seed_data['split']['pilot_holdout_indices'])} proxy={len(seed_data['split']['proxy_indices'])}"
            )
            grouped = _group_pilot_corruptions(seed_data["corrupted"])
            pilot_clean = _slice_clean_indices(seed_data["clean"], seed_data["split"]["pilot_holdout_indices"])
            proxy_clean = _slice_clean_indices(seed_data["clean"], seed_data["split"]["proxy_indices"])
            pilot_corrupted = _slice_corrupted_indices(grouped, seed_data["split"]["pilot_holdout_indices"])
            proxy_corrupted = _slice_corrupted_indices(grouped, seed_data["split"]["proxy_indices"])

            delta_by_family = {}
            for family in CORRUPTION_FAMILIES:
                delta_by_family[family] = pilot_corrupted[family].features.mean(dim=0) - pilot_clean.features.mean(dim=0)

            best_seed = None
            for variant in variants:
                q_vectors = text_bank["cifar10"]["q_bank"][variant]
                residuals = text_bank["cifar10"]["residual_bank"][variant]
                generic_residual = text_bank["cifar10"]["generic_residual"]
                for beta in beta_values:
                    family_preds = []
                    family_labels = []
                    for family_index, family in enumerate(CORRUPTION_FAMILIES):
                        preds = _predict_family(pilot_corrupted[family].features, q_vectors, beta=beta)
                        family_preds.append(preds)
                        family_labels.append(torch.full_like(preds, family_index))
                    family_preds = torch.cat(family_preds)
                    family_labels = torch.cat(family_labels)
                    family_acc = float((family_preds == family_labels).float().mean().item() * 100.0)

                    align_main = cosine_margin(delta_by_family, residuals)
                    align_generic = cosine_margin(
                        delta_by_family,
                        {family: generic_residual for family in CORRUPTION_FAMILIES},
                    )
                    permutation = np.random.default_rng(seed).permutation(CORRUPTION_FAMILIES).tolist()
                    align_random = cosine_margin(
                        delta_by_family,
                        {family: residuals[perm] for family, perm in zip(CORRUPTION_FAMILIES, permutation)},
                    )
                    for alpha in alpha_values:
                        for lam in lambda_values:
                            q_proxy = {family: family_posterior(proxy_corrupted[family].features, q_vectors, beta=beta) for family in CORRUPTION_FAMILIES}
                            zero_clean_proxy = score_zero_shot(proxy_clean.features, text_bank["cifar10"]["clean_single"])
                            zero_corr_proxy = {
                                family: score_zero_shot(proxy_corrupted[family].features, text_bank["cifar10"]["clean_single"])
                                for family in CORRUPTION_FAMILIES
                            }
                            method_clean = score_family_residual(
                                proxy_clean.features,
                                family_posterior(proxy_clean.features, q_vectors, beta=beta),
                                text_bank["cifar10"]["clean_single"],
                                residuals,
                                alpha=alpha,
                                lam=lam,
                            )
                            method_corr = {
                                family: score_family_residual(
                                    proxy_corrupted[family].features,
                                    q_proxy[family],
                                    text_bank["cifar10"]["clean_single"],
                                    residuals,
                                    alpha=alpha,
                                    lam=lam,
                                )
                                for family in CORRUPTION_FAMILIES
                            }
                            objective = _proxy_objective(zero_clean_proxy, method_clean, method_corr, zero_corr_proxy)
                            config_key = f"{variant}|beta={beta}|alpha={alpha}|lambda={lam}"
                            global_scores[config_key].append(objective)
                            proxy_grid.append(
                                {
                                    "seed": seed,
                                    "variant": variant,
                                    "beta": beta,
                                    "alpha": alpha,
                                    "lambda": lam,
                                    "proxy_objective": objective,
                                    "family_posterior_accuracy": family_acc,
                                }
                            )

                            q_pilot = {family: family_posterior(pilot_corrupted[family].features, q_vectors, beta=beta) for family in CORRUPTION_FAMILIES}
                            permutation = np.random.default_rng(seed).permutation(CORRUPTION_FAMILIES).tolist()
                            method_corr_acc = np.mean(
                                [
                                    top1_accuracy(
                                        score_family_residual(
                                            pilot_corrupted[family].features,
                                            q_pilot[family],
                                            text_bank["cifar10"]["clean_single"],
                                            residuals,
                                            alpha=alpha,
                                            lam=lam,
                                        ),
                                        pilot_corrupted[family].labels,
                                    )
                                    for family in CORRUPTION_FAMILIES
                                ]
                            )
                            generic_corr_acc = np.mean(
                                [
                                    top1_accuracy(
                                        score_generic_residual(
                                            pilot_corrupted[family].features,
                                            text_bank["cifar10"]["clean_single"],
                                            generic_residual,
                                            alpha=alpha,
                                            lam=lam,
                                        ),
                                        pilot_corrupted[family].labels,
                                    )
                                    for family in CORRUPTION_FAMILIES
                                ]
                            )
                            random_corr_acc = np.mean(
                                [
                                    top1_accuracy(
                                        score_family_residual(
                                            pilot_corrupted[family].features,
                                            q_pilot[family],
                                            text_bank["cifar10"]["clean_single"],
                                            residuals,
                                            alpha=alpha,
                                            lam=lam,
                                            family_order=permutation,
                                        ),
                                        pilot_corrupted[family].labels,
                                    )
                                    for family in CORRUPTION_FAMILIES
                                ]
                            )
                            clean_zero = top1_accuracy(
                                score_zero_shot(pilot_clean.features, text_bank["cifar10"]["clean_single"]),
                                pilot_clean.labels,
                            )
                            clean_method = top1_accuracy(
                                score_family_residual(
                                    pilot_clean.features,
                                    family_posterior(pilot_clean.features, q_vectors, beta=beta),
                                    text_bank["cifar10"]["clean_single"],
                                    residuals,
                                    alpha=alpha,
                                    lam=lam,
                                ),
                                pilot_clean.labels,
                            )
                            record = {
                                "seed": seed,
                                "variant": variant,
                                "beta": beta,
                                "alpha": alpha,
                                "lambda": lam,
                                "family_posterior_accuracy": family_acc,
                                "main_alignment_margin": align_main["margin"],
                                "generic_alignment_margin": align_generic["margin"],
                                "random_alignment_margin": align_random["margin"],
                                "main_alignment_matrix": align_main["matrix"],
                                "generic_alignment_matrix": align_generic["matrix"],
                                "random_alignment_matrix": align_random["matrix"],
                                "alignment_families": align_main["families"],
                                "pilot_method_accuracy": float(method_corr_acc),
                                "pilot_generic_accuracy": float(generic_corr_acc),
                                "pilot_random_accuracy": float(random_corr_acc),
                                "pilot_clean_zero_shot": clean_zero,
                                "pilot_clean_method": clean_method,
                                "proxy_objective": objective,
                            }
                            search_records.append(record)
                            if best_seed is None or record["proxy_objective"] > best_seed["proxy_objective"]:
                                best_seed = record
            logger.log(
                f"Seed {seed} best variant={best_seed['variant']} beta={best_seed['beta']} alpha={best_seed['alpha']} "
                f"lambda={best_seed['lambda']} proxy_objective={best_seed['proxy_objective']:.6f} "
                f"pilot_method={best_seed['pilot_method_accuracy']:.3f} generic={best_seed['pilot_generic_accuracy']:.3f} "
                f"random={best_seed['pilot_random_accuracy']:.3f}"
            )
            pilot_seed_results.append(best_seed)

        frozen_candidates = []
        for config_key, scores in global_scores.items():
            variant, beta_str, alpha_str, lambda_str = config_key.split("|")
            frozen_candidates.append(
                {
                    "variant": variant,
                    "beta": float(beta_str.split("=")[1]),
                    "alpha": float(alpha_str.split("=")[1]),
                    "lambda": float(lambda_str.split("=")[1]),
                    "proxy_objective_mean": float(np.mean(scores)),
                    "proxy_objective_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                }
            )
        frozen_candidates = sorted(frozen_candidates, key=lambda item: item["proxy_objective_mean"], reverse=True)
        frozen = frozen_candidates[0]
        per_seed_frozen = {str(seed): _seed_frozen_configuration(seed, proxy_grid, frozen) for seed in SEEDS}

        success_votes = 0
        for result in pilot_seed_results:
            wins_accuracy = (
                result["pilot_method_accuracy"] >= result["pilot_generic_accuracy"] + 0.5
                and result["pilot_method_accuracy"] >= result["pilot_random_accuracy"] + 0.5
            )
            wins_alignment = (
                result["main_alignment_margin"] >= result["generic_alignment_margin"] + 0.03
                and result["main_alignment_margin"] >= result["random_alignment_margin"] + 0.03
            )
            clean_ok = result["pilot_clean_method"] >= result["pilot_clean_zero_shot"] - 0.5
            if clean_ok and wins_accuracy:
                success_votes += 1
        pilot_success = success_votes >= 2
        logger.log(
            f"Pilot gate outcome success={pilot_success} votes={success_votes} "
            f"global_frozen={frozen} per_seed={per_seed_frozen}"
        )

        save_json(RESULTS_DIR / "frozen_recipe.json", {"global": frozen, "per_seed": per_seed_frozen})
        payload = {
            "experiment": "pilot_gate",
            "pilot_success": pilot_success,
            "success_votes": success_votes,
            "pilot_interpretation": "negative_result_baseline_study" if not pilot_success else "provisional_positive_pilot_only",
            "frozen_configuration": frozen,
            "per_seed_frozen_configuration": per_seed_frozen,
            "best_per_seed": pilot_seed_results,
            "search_records": search_records,
            "proxy_grid": proxy_grid,
        }
        save_json(artifacts.config_path, {"grid": {"beta": beta_values, "alpha": alpha_values, "lambda": lambda_values, "variant": variants}})
        save_json(artifacts.results_path, payload)
    finally:
        logger.close()


def _method_scores(method_name: str, features: torch.Tensor, text_bank: dict, cfg: dict, seed: int, q_override: torch.Tensor | None = None) -> torch.Tensor:
    q_vectors = text_bank["q_bank"][cfg["variant"]]
    q = q_override if q_override is not None else family_posterior(features, q_vectors, beta=cfg["beta"])
    if method_name == "zero_shot":
        return score_zero_shot(features, text_bank["clean_single"])
    if method_name == "clean_prompt_ensemble":
        return score_clean_ensemble(features, text_bank["clean_ensemble"])
    if method_name == "naive_corruption_prompt":
        return score_naive(features, q, text_bank["naive"])
    if method_name == "generic_residual_control":
        return score_generic_residual(features, text_bank["clean_single"], text_bank["generic_residual"], cfg["alpha"], cfg["lambda"])
    if method_name == "family_residual":
        return score_family_residual(features, q, text_bank["clean_single"], text_bank["residual_bank"][cfg["variant"]], cfg["alpha"], cfg["lambda"])
    if method_name == "random_family_reassignment":
        permutation = np.random.default_rng(seed).permutation(CORRUPTION_FAMILIES).tolist()
        return score_family_residual(
            features,
            q,
            text_bank["clean_single"],
            text_bank["residual_bank"][cfg["variant"]],
            cfg["alpha"],
            cfg["lambda"],
            family_order=permutation,
        )
    if method_name == "uniform_family_weighting":
        uniform_q = torch.full((len(features), len(CORRUPTION_FAMILIES)), 1.0 / len(CORRUPTION_FAMILIES))
        return score_family_residual(features, uniform_q, text_bank["clean_single"], text_bank["residual_bank"][cfg["variant"]], cfg["alpha"], cfg["lambda"])
    raise ValueError(f"Unsupported method: {method_name}")


def _evaluate_dataset(method_name: str, dataset_name: str, seed: int | None, cfg: dict, text_bank: dict, dataset_cache: dict) -> dict:
    runtime_start = time.perf_counter()
    clean_bundle = dataset_cache["clean"]
    if seed is None:
        eval_seed = 0
    else:
        eval_seed = seed
    clean_scores = _method_scores(method_name, clean_bundle.features, text_bank, cfg, eval_seed)
    clean_accuracy = top1_accuracy(clean_scores, clean_bundle.labels)
    clean_ece = expected_calibration_error(clean_scores, clean_bundle.labels)
    records = []
    family_breakdown = defaultdict(list)
    severity_breakdown = defaultdict(list)
    for family in CORRUPTION_FAMILIES:
        for severity in SEVERITIES:
            bundle = dataset_cache["corruptions"][(family, severity)]
            scores = _method_scores(method_name, bundle.features, text_bank, cfg, eval_seed)
            accuracy = top1_accuracy(scores, bundle.labels)
            ece = expected_calibration_error(scores, bundle.labels)
            family_breakdown[family].append(accuracy)
            severity_breakdown[str(severity)].append(accuracy)
            records.append(
                {
                    "dataset": dataset_name,
                    "corruption_family": family,
                    "severity": severity,
                    "accuracy": accuracy,
                    "ece": ece,
                }
            )
    wall_seconds = time.perf_counter() - runtime_start
    corruption_mean = float(np.mean([item["accuracy"] for item in records]))
    runtime_per_10k = wall_seconds / ((len(records) * 10000) / 10000.0)
    return {
        "method": method_name,
        "seed": seed,
        "dataset": dataset_name,
        "clean_accuracy": clean_accuracy,
        "clean_ece": clean_ece,
        "corruption_mean_accuracy": corruption_mean,
        "per_family_accuracy": {family: float(np.mean(values)) for family, values in family_breakdown.items()},
        "per_severity_accuracy": {severity: float(np.mean(values)) for severity, values in severity_breakdown.items()},
        "records": records,
        "runtime_per_10k_images_seconds": float(runtime_per_10k),
    }


def run_final_evaluation() -> None:
    artifacts = experiment_artifacts("final_evaluation")
    logger = stage_logger("final_evaluation")
    ensure_dir(RESULTS_DIR / "main")
    frozen_payload = load_json(RESULTS_DIR / "frozen_recipe.json")
    device = _device()
    try:
        ledger_path = RESULTS_DIR / "ledger.jsonl"
        if ledger_path.exists():
            ledger_path.unlink()
        bundle = load_openclip(device)
        text_banks = {
            "cifar10": build_text_bank(bundle, CIFAR10_CLASSNAMES),
            "cifar100": build_text_bank(bundle, CIFAR100_CLASSNAMES),
        }
        caches = {
            "cifar10": _load_clean_and_corruptions("cifar10"),
            "cifar100": _load_clean_and_corruptions("cifar100"),
        }
        for dataset_name, cache in caches.items():
            logger.log(
                f"Dataset {dataset_name} counts clean={len(cache['clean'].labels)} "
                f"corruptions={len(cache['corruptions'])} per_condition={len(next(iter(cache['corruptions'].values())).labels)}"
            )
        deterministic_methods = ["zero_shot", "clean_prompt_ensemble", "naive_corruption_prompt", "generic_residual_control"]
        seeded_methods = ["family_residual", "random_family_reassignment"]
        all_results = []

        for dataset_name in ["cifar10", "cifar100"]:
            for method_name in deterministic_methods:
                result = _evaluate_dataset(method_name, dataset_name, None, frozen_payload["global"], text_banks[dataset_name], caches[dataset_name])
                all_results.append(result)
                save_json(RESULTS_DIR / "main" / f"{method_name}_{dataset_name}.json", result)
                logger.log(
                    f"Final {dataset_name} {method_name} clean={result['clean_accuracy']:.3f} "
                    f"corruption_mean={result['corruption_mean_accuracy']:.3f}"
                )
                for record in result["records"]:
                    append_jsonl(
                        RESULTS_DIR / "ledger.jsonl",
                        {
                            "method": method_name,
                            "seed": None,
                            "dataset": dataset_name,
                            "corruption_family": record["corruption_family"],
                            "severity": record["severity"],
                            "split": "test",
                            "wall_clock_seconds": result["runtime_per_10k_images_seconds"] * 20,
                            "gpu_memory_gb": gpu_memory_gb(),
                            "clean_accuracy": result["clean_accuracy"],
                            "corruption_accuracy": record["accuracy"],
                            "output_path": str(RESULTS_DIR / "main" / f"{method_name}_{dataset_name}.json"),
                        },
                    )
            for method_name in seeded_methods:
                for seed in SEEDS:
                    cfg = frozen_payload["per_seed"][str(seed)]
                    result = _evaluate_dataset(method_name, dataset_name, seed, cfg, text_banks[dataset_name], caches[dataset_name])
                    result["selected_configuration"] = cfg
                    all_results.append(result)
                    save_json(RESULTS_DIR / "main" / f"{method_name}_{dataset_name}_seed_{seed}.json", result)
                    logger.log(
                        f"Final {dataset_name} {method_name} seed={seed} variant={cfg['variant']} beta={cfg['beta']} "
                        f"alpha={cfg['alpha']} lambda={cfg['lambda']} clean={result['clean_accuracy']:.3f} "
                        f"corruption_mean={result['corruption_mean_accuracy']:.3f}"
                    )
                    for record in result["records"]:
                        append_jsonl(
                            RESULTS_DIR / "ledger.jsonl",
                            {
                                "method": method_name,
                                "seed": seed,
                                "dataset": dataset_name,
                                "corruption_family": record["corruption_family"],
                                "severity": record["severity"],
                                "split": "test",
                                "wall_clock_seconds": result["runtime_per_10k_images_seconds"] * 20,
                                "gpu_memory_gb": gpu_memory_gb(),
                                "clean_accuracy": result["clean_accuracy"],
                                "corruption_accuracy": record["accuracy"],
                                "output_path": str(RESULTS_DIR / "main" / f"{method_name}_{dataset_name}_seed_{seed}.json"),
                            },
                        )

        save_json(artifacts.config_path, frozen_payload)
        save_json(artifacts.results_path, {"experiment": "final_evaluation", "runs": all_results})
    finally:
        logger.close()


def run_ablations() -> None:
    artifacts = experiment_artifacts("ablations")
    logger = stage_logger("ablations")
    frozen_payload = load_json(RESULTS_DIR / "frozen_recipe.json")
    device = _device()
    try:
        bundle = load_openclip(device)
        text_banks = {
            "cifar10": build_text_bank(bundle, CIFAR10_CLASSNAMES),
            "cifar100": build_text_bank(bundle, CIFAR100_CLASSNAMES),
        }
        caches = {
            "cifar10": _load_clean_and_corruptions("cifar10"),
            "cifar100": _load_clean_and_corruptions("cifar100"),
        }
        ablation_methods = {
            "ablation_remove_family_residuals": "naive_corruption_prompt",
            "ablation_uniform_family_weighting": "uniform_family_weighting",
            "ablation_generic_vs_family": "generic_residual_control",
            "ablation_random_identity": "random_family_reassignment",
        }
        all_results = []
        for dataset_name in ["cifar10", "cifar100"]:
            for ablation_name, method_name in ablation_methods.items():
                for seed in SEEDS:
                    cfg = frozen_payload["per_seed"][str(seed)]
                    result = _evaluate_dataset(method_name, dataset_name, seed, cfg, text_banks[dataset_name], caches[dataset_name])
                    result["ablation_name"] = ablation_name
                    result["compare_against"] = "family_residual"
                    result["selected_configuration"] = cfg
                    all_results.append(result)
                    suffix = f"_seed_{seed}"
                    save_json(RESULTS_DIR / "ablations" / f"{ablation_name}_{dataset_name}{suffix}.json", result)
                    logger.log(
                        f"Ablation {ablation_name} dataset={dataset_name} seed={seed} method={method_name} "
                        f"corruption_mean={result['corruption_mean_accuracy']:.3f}"
                    )

        used_strong = any(cfg["variant"] == "strong" for cfg in frozen_payload["per_seed"].values())
        if used_strong:
            for seed in SEEDS:
                strong_cfg = dict(frozen_payload["per_seed"][str(seed)])
                strong_cfg["variant"] = "strong"
                base_cfg = dict(frozen_payload["per_seed"][str(seed)])
                base_cfg["variant"] = "base"
                result_strong = _evaluate_dataset("family_residual", "cifar10", seed, strong_cfg, text_banks["cifar10"], caches["cifar10"])
                result_base = _evaluate_dataset("family_residual", "cifar10", seed, base_cfg, text_banks["cifar10"], caches["cifar10"])
                all_results.append(
                    {
                        "ablation_name": "ablation_prompt_bank_sensitivity",
                        "seed": seed,
                        "dataset": "cifar10",
                        "strong_variant": result_strong,
                        "base_variant": result_base,
                    }
                )
                logger.log(
                    f"Ablation prompt_bank_sensitivity seed={seed} strong={result_strong['corruption_mean_accuracy']:.3f} "
                    f"base={result_base['corruption_mean_accuracy']:.3f}"
                )
        save_json(artifacts.config_path, frozen_payload)
        save_json(artifacts.results_path, {"experiment": "ablations", "runs": all_results})
    finally:
        logger.close()


def run_visualization() -> None:
    artifacts = experiment_artifacts("visualization")
    logger = stage_logger("visualization")
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR / "tables")
    try:
        main_dir = RESULTS_DIR / "main"
        ablation_dir = RESULTS_DIR / "ablations"
        pilot = load_json(experiment_artifacts("pilot_gate").results_path)

        rows = []
        for path in sorted(main_dir.glob("*.json")):
            payload = load_json(path)
            rows.append(
                {
                    "method": payload["method"],
                    "seed": payload["seed"],
                    "dataset": payload["dataset"],
                    "clean_accuracy": payload["clean_accuracy"],
                    "corruption_mean_accuracy": payload["corruption_mean_accuracy"],
                    "runtime_per_10k_images_seconds": payload["runtime_per_10k_images_seconds"],
                }
            )
        main_df = pd.DataFrame(rows)
        table_rows = []
        for method in sorted(main_df["method"].unique()):
            row = {"method": method}
            for dataset in ["cifar10", "cifar100"]:
                subset = main_df[(main_df["method"] == method) & (main_df["dataset"] == dataset)]
                row[f"{dataset}_clean"] = summarize(subset["clean_accuracy"].tolist())["mean"]
                row[f"{dataset}_corruption_mean"] = summarize(subset["corruption_mean_accuracy"].tolist())["mean"]
                row[f"{dataset}_runtime"] = summarize(subset["runtime_per_10k_images_seconds"].tolist())["mean"]
            table_rows.append(row)
        table_df = pd.DataFrame(table_rows).sort_values("method")
        table_df.to_csv(RESULTS_DIR / "tables" / "main_results_table.csv", index=False)

        latex_lines = []
        latex_lines.append("Method & CIFAR-10 Clean & CIFAR-100 Clean & CIFAR-10-C Mean & CIFAR-100-C Mean & Runtime \\\\")
        for _, row in table_df.iterrows():
            latex_lines.append(
                f"{row['method']} & {row['cifar10_clean']:.2f} & {row['cifar100_clean']:.2f} & "
                f"{row['cifar10_corruption_mean']:.2f} & {row['cifar100_corruption_mean']:.2f} & "
                f"{row['cifar10_runtime']:.3f}/{row['cifar100_runtime']:.3f} \\\\"
            )
        with (RESULTS_DIR / "tables" / "main_results_table.tex").open("w") as handle:
            handle.write("\n".join(latex_lines) + "\n")

        alignment_records = pd.DataFrame(pilot["search_records"])
        best_alignment = alignment_records.sort_values("proxy_objective", ascending=False).iloc[0]
        family_labels = best_alignment["alignment_families"]
        matrix = np.array(best_alignment["main_alignment_matrix"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, xticklabels=family_labels, yticklabels=family_labels, cmap="mako")
        plt.title("Pilot Alignment Heatmap")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pilot_alignment_heatmap.png", dpi=200)
        plt.savefig(FIGURES_DIR / "pilot_alignment_heatmap.pdf")
        plt.close()

        detail_rows = []
        for path in sorted(main_dir.glob("*.json")):
            payload = load_json(path)
            for family, acc in payload["per_family_accuracy"].items():
                detail_rows.append(
                    {
                        "method": payload["method"],
                        "seed": payload["seed"],
                        "dataset": payload["dataset"],
                        "family": family,
                        "accuracy": acc,
                    }
                )
        detail_df = pd.DataFrame(detail_rows)
        selected_methods = [
            "zero_shot",
            "clean_prompt_ensemble",
            "naive_corruption_prompt",
            "generic_residual_control",
            "random_family_reassignment",
            "family_residual",
        ]
        for dataset in ["cifar10", "cifar100"]:
            plot_df = detail_df[(detail_df["dataset"] == dataset) & (detail_df["method"].isin(selected_methods))]
            plt.figure(figsize=(9, 5))
            sns.barplot(data=plot_df, x="family", y="accuracy", hue="method", errorbar="sd")
            plt.title(f"{dataset.upper()}-C Per-Family Accuracy")
            plt.ylabel("Top-1 Accuracy")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{dataset}_per_family_bar.png", dpi=200)
            plt.savefig(FIGURES_DIR / f"{dataset}_per_family_bar.pdf")
            plt.close()

        severity_rows = []
        for path in sorted(main_dir.glob("*.json")):
            payload = load_json(path)
            for severity, acc in payload["per_severity_accuracy"].items():
                severity_rows.append(
                    {
                        "method": payload["method"],
                        "seed": payload["seed"],
                        "dataset": payload["dataset"],
                        "severity": int(severity),
                        "accuracy": acc,
                    }
                )
        severity_df = pd.DataFrame(severity_rows)
        severity_df.to_csv(RESULTS_DIR / "tables" / "severity_curves.csv", index=False)
        for dataset in ["cifar10", "cifar100"]:
            plot_df = severity_df[(severity_df["dataset"] == dataset) & (severity_df["method"].isin(selected_methods))]
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=plot_df, x="severity", y="accuracy", hue="method", style="method", markers=True, dashes=False, errorbar="sd")
            plt.title(f"{dataset.upper()}-C Severity Curves")
            plt.ylabel("Top-1 Accuracy")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"{dataset}_severity_curves.png", dpi=200)
            plt.savefig(FIGURES_DIR / f"{dataset}_severity_curves.pdf")
            plt.close()

        ablation_payload = {
            "num_ablation_files": len(list(ablation_dir.glob("*.json"))),
            "table_path": str(RESULTS_DIR / "tables" / "main_results_table.csv"),
        }
        logger.log(f"Generated figures and tables with methods={selected_methods}")
        save_json(artifacts.config_path, {"selected_methods": selected_methods})
        save_json(artifacts.results_path, {"experiment": "visualization", **ablation_payload})
    finally:
        logger.close()


def aggregate_results() -> None:
    frozen_payload = load_json(RESULTS_DIR / "frozen_recipe.json")
    pilot = load_json(experiment_artifacts("pilot_gate").results_path)
    main = load_json(experiment_artifacts("final_evaluation").results_path)
    ablations = load_json(experiment_artifacts("ablations").results_path)
    summary_rows = defaultdict(lambda: defaultdict(list))
    condition_rows = defaultdict(lambda: defaultdict(list))
    for run in main["runs"]:
        summary_rows[run["method"]][f"{run['dataset']}_clean"].append(run["clean_accuracy"])
        summary_rows[run["method"]][f"{run['dataset']}_corruption"].append(run["corruption_mean_accuracy"])
        for record in run["records"]:
            condition_rows[run["method"]][run["dataset"]].append(record["accuracy"])
    methods_summary = {}
    for method, metrics in summary_rows.items():
        methods_summary[method] = {metric: summarize(values) for metric, values in metrics.items()}
        for dataset in ["cifar10", "cifar100"]:
            dataset_key = f"{dataset}_corruption_bootstrap_ci"
            if dataset in condition_rows[method]:
                methods_summary[method][dataset_key] = bootstrap_confidence_interval(
                    condition_rows[method][dataset],
                    seed=sum(ord(ch) for ch in f"{method}-{dataset}"),
                )
    family_vs_generic = methods_summary["family_residual"]["cifar10_corruption"]["mean"] > methods_summary["generic_residual_control"]["cifar10_corruption"]["mean"] and methods_summary["family_residual"]["cifar100_corruption"]["mean"] > methods_summary["generic_residual_control"]["cifar100_corruption"]["mean"]
    family_vs_random = methods_summary["family_residual"]["cifar10_corruption"]["mean"] > methods_summary["random_family_reassignment"]["cifar10_corruption"]["mean"] and methods_summary["family_residual"]["cifar100_corruption"]["mean"] > methods_summary["random_family_reassignment"]["cifar100_corruption"]["mean"]
    downstream_success = family_vs_generic and family_vs_random
    study_conclusion = (
        "negative_result_baseline_study"
        if not downstream_success
        else "family_specific_signal_supported"
    )
    root_payload = {
        "title": "Do Corruption-Family Text Residuals Help Zero-Shot CLIP? A Controlled Baseline Study",
        "frozen_configuration": frozen_payload["global"],
        "per_seed_frozen_configuration": frozen_payload["per_seed"],
        "pilot_success": pilot["pilot_success"],
        "pilot_success_votes": pilot["success_votes"],
        "pilot_interpretation": pilot.get("pilot_interpretation"),
        "downstream_success": downstream_success,
        "study_conclusion": study_conclusion,
        "claim_assessment": {
            "family_residual_beats_generic_on_both_datasets": family_vs_generic,
            "family_residual_beats_random_on_both_datasets": family_vs_random,
            "negative_result_reason": (
                "Family residuals do not beat both the generic residual control and the random family reassignment control on final downstream CIFAR-C comparisons."
                if not downstream_success
                else None
            ),
        },
        "method_summary": methods_summary,
        "num_final_runs": len(main["runs"]),
        "num_ablation_runs": len(ablations["runs"]),
        "artifacts": {
            "prepare_data": str(experiment_artifacts("prepare_data").results_path),
            "pilot_gate": str(experiment_artifacts("pilot_gate").results_path),
            "final_evaluation": str(experiment_artifacts("final_evaluation").results_path),
            "ablations": str(experiment_artifacts("ablations").results_path),
            "visualization": str(experiment_artifacts("visualization").results_path),
            "ledger": str(RESULTS_DIR / "ledger.jsonl"),
            "tables": str(RESULTS_DIR / "tables" / "main_results_table.csv"),
        },
    }
    save_json(Path("results.json"), root_payload)


def run_all() -> None:
    run_prepare_data()
    run_pilot_gate()
    run_final_evaluation()
    run_ablations()
    run_visualization()
    write_skipped(
        "autoclip",
        "Skipped by design: the optional AutoCLIP comparator is only permitted after a clearly positive pilot, full required evaluations, required ablations, and at least 90 minutes of remaining budget.",
    )
    aggregate_results()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare_data", "pilot_gate", "final_evaluation", "ablations", "visualization", "aggregate", "all"])
    args = parser.parse_args()
    if args.stage == "prepare_data":
        run_prepare_data()
    elif args.stage == "pilot_gate":
        run_pilot_gate()
    elif args.stage == "final_evaluation":
        run_final_evaluation()
    elif args.stage == "ablations":
        run_ablations()
    elif args.stage == "visualization":
        run_visualization()
    elif args.stage == "aggregate":
        aggregate_results()
    elif args.stage == "all":
        run_all()


if __name__ == "__main__":
    main()
