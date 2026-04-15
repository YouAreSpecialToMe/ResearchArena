from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared import config
from exp.shared.data import prepare_dataset_split
from exp.shared.metrics import mean_perturbed_reference_pearson, nearest_centroid_metrics, rmse
from exp.shared.models import fit_mlp, fit_ridge
from exp.shared.pipeline import _leave_one_out_retrieval, _weighted_neighbor_summary
from exp.shared.utils import (
    Timer,
    append_jsonl,
    capture_config_snapshot,
    capture_environment_metadata,
    max_rss_mb,
    save_json,
    set_global_seed,
    slugify,
)


def compute_metrics(split, pred, true, labels):
    pearson, _ = mean_perturbed_reference_pearson(pred, true, split.mu_pert_train)
    rmse_value, _ = rmse(pred, true)
    top1, median_rank = nearest_centroid_metrics(pred, true, labels)
    return {
        "perturbed_reference_pearson": pearson,
        "rmse": rmse_value,
        "top1_accuracy": top1,
        "median_rank": median_rank,
    }


def fit_protocol_mlp(split, x_train, y_train, x_val, decode_fn, seed, log_path):
    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_val_t = torch.from_numpy(x_val.astype(np.float32)).to(device)

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            pred_val_latent = model(x_val_t).cpu().numpy()
        pred_val = decode_fn(pred_val_latent)
        metrics = compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
        return metrics["perturbed_reference_pearson"], metrics["rmse"]

    def log_fn(row):
        append_jsonl(log_path, {"event": "epoch", **row})

    return fit_mlp(
        x_train=x_train,
        y_train=y_train,
        seed=seed,
        device=device,
        eval_fn=eval_fn,
        log_fn=log_fn,
    )


def run_embedding_mlp(split, seed: int, target_mode: str, log_path: Path) -> tuple[dict[str, float], dict[str, float | int]]:
    x_train = split.descriptor_train
    x_val = split.descriptor_val
    x_test = split.descriptor_test
    if target_mode == "full":
        y_train = split.full_train_pca
        decode_fn = lambda pred_latent: split.full_pca.inverse_transform(pred_latent)
    elif target_mode == "residual":
        y_train = split.residual_train_pca
        decode_fn = lambda pred_latent: split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_latent)
    else:
        raise ValueError(target_mode)

    model, info = fit_protocol_mlp(split, x_train, y_train, x_val, decode_fn, seed, log_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        pred_latent = model(torch.from_numpy(x_test.astype(np.float32)).to(device)).cpu().numpy()
    pred = decode_fn(pred_latent)
    return compute_metrics(split, pred, split.test_matrix, split.test_perts), info


def run_resrp_no_retrieval(split, seed: int, log_path: Path) -> tuple[dict[str, float], dict[str, float | int]]:
    candidates = []
    for k in config.RETRIEVAL_K_GRID:
        zeros_train = np.zeros((len(split.train_perts), config.RESIDUAL_PCA_DIM), dtype=np.float32)
        zeros_val = np.zeros((len(split.val_perts), config.RESIDUAL_PCA_DIM), dtype=np.float32)
        zeros_test = np.zeros((len(split.test_perts), config.RESIDUAL_PCA_DIM), dtype=np.float32)
        zero_max_train = np.zeros((len(split.train_perts), 1), dtype=np.float32)
        zero_max_val = np.zeros((len(split.val_perts), 1), dtype=np.float32)
        zero_max_test = np.zeros((len(split.test_perts), 1), dtype=np.float32)

        # Match ReSRP feature layout exactly; retrieval channels are zeroed by construction.
        _leave_one_out_retrieval(split.descriptor_train, split.residual_train_pca, k)
        _weighted_neighbor_summary(
            split.retrieval_cache_val["similarities"],
            split.retrieval_cache_val["indices"],
            split.residual_train_pca,
            k,
        )
        x_train = np.concatenate([split.descriptor_train, zeros_train, zero_max_train], axis=1)
        x_val = np.concatenate([split.descriptor_val, zeros_val, zero_max_val], axis=1)
        x_test = np.concatenate([split.descriptor_test, zeros_test, zero_max_test], axis=1)
        model, info = fit_protocol_mlp(
            split,
            x_train,
            split.residual_train_pca,
            x_val,
            lambda pred_latent: split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_latent),
            seed,
            log_path,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        with torch.no_grad():
            pred_val_latent = model(torch.from_numpy(x_val.astype(np.float32)).to(device)).cpu().numpy()
            pred_test_latent = model(torch.from_numpy(x_test.astype(np.float32)).to(device)).cpu().numpy()
        pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_val_latent)
        metrics = compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
        pred_test = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(pred_test_latent)
        candidates.append((metrics, pred_test, {"k": k, "input_dim": x_train.shape[1], **info}))

    best_metrics, best_pred, best_info = max(
        candidates,
        key=lambda item: (item[0]["perturbed_reference_pearson"], -item[0]["rmse"]),
    )
    return compute_metrics(split, best_pred, split.test_matrix, split.test_perts), best_info


def run_descriptor_contingency(split, seed: int, log_path: Path) -> list[dict[str, object]]:
    variants = {
        "true_string": (split.descriptor_train, split.descriptor_val, split.descriptor_test),
        "degree_only": (
            split.descriptor_degree_train,
            split.descriptor_degree_val,
            split.descriptor_degree_test,
        ),
    }
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(split.train_perts))
    variants["permuted_string"] = (
        split.descriptor_train[perm],
        split.descriptor_val,
        split.descriptor_test,
    )
    rows = []
    for name, (x_train, x_val, x_test) in variants.items():
        best = None
        for alpha in config.RIDGE_ALPHA_GRID:
            model = fit_ridge(x_train, split.residual_train_pca, alpha)
            pred_val = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(model.predict(x_val))
            metrics = compute_metrics(split, pred_val, split.val_matrix, split.val_perts)
            append_jsonl(log_path, {"event": "descriptor_candidate", "variant": name, "alpha": alpha, "metrics": metrics})
            score = (metrics["perturbed_reference_pearson"], -metrics["rmse"])
            if best is None or score > best[0]:
                best = (score, alpha, model)
        _, alpha, model = best
        pred = split.mu_pert_train[None, :] + split.residual_pca.inverse_transform(model.predict(x_test))
        rows.append(
            {
                "dataset": split.dataset,
                "seed": seed,
                "descriptor_variant": name,
                "alpha": alpha,
                **compute_metrics(split, pred, split.test_matrix, split.test_perts),
            }
        )
    return rows


def main() -> None:
    out_dir = ROOT / "exp" / "ablations"
    log_dir = out_dir / "logs"
    hp_dir = out_dir / "hyperparams"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    hp_dir.mkdir(parents=True, exist_ok=True)

    environment = capture_environment_metadata()
    config_snapshot = capture_config_snapshot(config)
    preprocess_results = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "seed": int(seed),
                "preprocess_runtime_minutes": audit["runtime_minutes"],
            }
            for dataset, seeds in json.load((ROOT / "exp" / "preprocess" / "results.json").open("r", encoding="utf-8"))[
                "datasets"
            ].items()
            for seed, audit in seeds.items()
        ]
    )

    rows = []
    for dataset in config.DATASETS:
        for seed in config.SEEDS:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            split = prepare_dataset_split(dataset, seed)
            prep_runtime = float(
                preprocess_results[
                    (preprocess_results["dataset"] == dataset) & (preprocess_results["seed"] == seed)
                ]["preprocess_runtime_minutes"].iloc[0]
            )

            for ablation_name, runner in [
                ("compact_mlp_full_target", lambda log_path: run_embedding_mlp(split, seed, "full", log_path)),
                ("compact_mlp_residual_target", lambda log_path: run_embedding_mlp(split, seed, "residual", log_path)),
                ("resrp_mlp_no_retrieval", lambda log_path: run_resrp_no_retrieval(split, seed, log_path)),
            ]:
                log_path = log_dir / f"{dataset}_seed{seed}_{slugify(ablation_name)}.jsonl"
                with Timer() as timer:
                    metrics, info = runner(log_path)
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "ablation": ablation_name,
                        **metrics,
                        "runtime_minutes": prep_runtime + timer.minutes,
                        "peak_memory_mb": max_rss_mb(),
                        "peak_gpu_memory_mb": (
                            float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None
                        ),
                        **{f"info_{k}": v for k, v in info.items()},
                    }
                )
                save_json(
                    hp_dir / f"{dataset}_seed{seed}_{slugify(ablation_name)}.json",
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "ablation": ablation_name,
                        "hyperparams": info,
                        "metrics": metrics,
                        "runtime_minutes": prep_runtime + timer.minutes,
                        "peak_memory_mb": max_rss_mb(),
                        "peak_gpu_memory_mb": (
                            float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None
                        ),
                        "environment": environment,
                        "config": config_snapshot,
                    },
                )

            contingency_log = log_dir / f"{dataset}_seed{seed}_descriptor_contingency.jsonl"
            with Timer() as timer:
                contingency_rows = run_descriptor_contingency(split, seed, contingency_log)
            for row in contingency_rows:
                row["runtime_minutes"] = prep_runtime + timer.minutes / max(1, len(contingency_rows))
                row["peak_memory_mb"] = max_rss_mb()
                row["peak_gpu_memory_mb"] = (
                    float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else None
                )
            rows.extend(contingency_rows)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    ablation_rows = df[df["ablation"].notna()].copy()
    contingency_rows = df[df["descriptor_variant"].notna()].copy()
    ablation_summary = (
        ablation_rows.groupby(["dataset", "ablation"], as_index=False)
        .agg(
            perturbed_reference_pearson_mean=("perturbed_reference_pearson", "mean"),
            perturbed_reference_pearson_std=("perturbed_reference_pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            top1_accuracy_mean=("top1_accuracy", "mean"),
            top1_accuracy_std=("top1_accuracy", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
        )
    )
    contingency_summary = (
        contingency_rows.groupby(["dataset", "descriptor_variant"], as_index=False)
        .agg(
            perturbed_reference_pearson_mean=("perturbed_reference_pearson", "mean"),
            perturbed_reference_pearson_std=("perturbed_reference_pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
    )
    save_json(
        out_dir / "results.json",
        {
            "experiment": "ablations",
            "environment": environment,
            "config": config_snapshot,
            "per_seed_rows": df.to_dict(orient="records"),
            "ablation_summary": ablation_summary.to_dict(orient="records"),
            "descriptor_contingency_summary": contingency_summary.to_dict(orient="records"),
        },
    )
    skip_path = out_dir / "SKIPPED.md"
    skip_path.write_text(
        "Optional control-referenced residual sensitivity was not run because the required benchmark path and aggregation consumed the planned budget.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
