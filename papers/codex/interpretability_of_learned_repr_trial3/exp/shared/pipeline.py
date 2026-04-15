from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from .config import BACKBONES, DATASETS, ENV_ROOT, FEATURE_ROOT, META_ROOT, PILOT_GRID, RESULT_ROOT, SEEDS, SENSITIVITY_SEED, SPLIT_SEED
from .data import load_dataset, prepare_splits, read_json
from .features import extract_all_clean_features, extract_split_features
from .probes import fit_probe
from .trainer import RunConfig, build_anchor_bank, construct_pseudo_blocks_from_pairs, evaluate_method, load_clean_features, reorder_latents, train_sae, unit_block_payload_to_spec
from .models import LinearSAE
from .utils import append_registry_row, collect_environment_snapshot, ensure_dir, select_device, write_json


def run_env_setup() -> None:
    ensure_dir(ENV_ROOT)
    snapshot = collect_environment_snapshot()
    write_json(ENV_ROOT / "environment_snapshot.json", snapshot)
    requirements = "\n".join(f"{k}=={v}" for k, v in snapshot.items() if k.startswith("pkg_"))
    (ENV_ROOT / "requirements_lock.txt").write_text(requirements + "\n")


def run_data_prep() -> None:
    for dataset in DATASETS:
        prepare_splits(dataset)


def _nuisance_pair_payload(dataset: str, split: str) -> list[dict]:
    payload = read_json(Path("pairs") / dataset / split / "pairs.json")
    return payload["nuisance_pairs"]


def run_feature_cache() -> None:
    smoke_records = []
    for dataset in DATASETS:
        for backbone in BACKBONES:
            if BACKBONES[backbone]["sensitivity_only"] and dataset != "dsprites":
                continue
            extract_all_clean_features(dataset, backbone)
            for split in ["train", "val", "test"]:
                nuisance_pairs = _nuisance_pair_payload(dataset, split)
                pair_indices = np.asarray([entry["index"] for entry in nuisance_pairs], dtype=np.int64)
                view1 = [entry["view1"] for entry in nuisance_pairs]
                view2 = [entry["view2"] for entry in nuisance_pairs]
                extract_split_features(dataset, backbone, split, nuisance_views=view1, tag="nuisance_view1", pair_indices=pair_indices)
                extract_split_features(dataset, backbone, split, nuisance_views=view2, tag="nuisance_view2", pair_indices=pair_indices)
            sample = np.load(FEATURE_ROOT / dataset / backbone / "all_clean.npy")[:128]
            smoke_records.append({"dataset": dataset, "backbone": backbone, "samples": len(sample), "feature_dim": int(sample.shape[1])})
    write_json(Path("artifacts") / "metadata" / "smoke_test.json", smoke_records)


def run_probes() -> None:
    registry_path = META_ROOT / "run_registry.csv"
    for dataset, meta in DATASETS.items():
        bundle = load_dataset(dataset)
        split_indices = {split: np.asarray(read_json(Path("data/processed") / dataset / f"{split}_tuples.json")["indices"], dtype=np.int64) for split in ["train", "val", "test"]}
        for backbone in BACKBONES:
            if BACKBONES[backbone]["sensitivity_only"] and dataset != "dsprites":
                continue
            all_x = np.load(FEATURE_ROOT / dataset / backbone / "all_clean.npy")
            train_x = all_x[split_indices["train"]]
            val_x = all_x[split_indices["val"]]
            test_x = all_x[split_indices["test"]]
            results = []
            for factor_idx, factor_name in enumerate(meta["factors"]):
                train_y = bundle.factors[split_indices["train"], factor_idx]
                val_y = bundle.factors[split_indices["val"], factor_idx]
                test_y = bundle.factors[split_indices["test"], factor_idx]
                probe = fit_probe(train_x, train_y, val_x, val_y, test_x, test_y, factor_name, meta["factor_sizes"][factor_idx])
                results.append(probe.__dict__)
                append_registry_row(registry_path, {"stage": "probe", "dataset": dataset, "backbone": backbone, "factor": factor_name, "seed": SPLIT_SEED, "test_metric": probe.test_metric})
            results = sorted(results, key=lambda item: item["validation_metric"], reverse=True)
            write_json(RESULT_ROOT / dataset / backbone / "probes.json", results)
            write_json(Path("exp/feature_cache") / f"probes_{dataset}_{backbone}.json", results)


def get_probe_context(dataset: str, backbone: str) -> tuple[list[str], list[str]]:
    probes = read_json(RESULT_ROOT / dataset / backbone / "probes.json")
    factor_order = [item["factor"] for item in probes]
    admissible = [item["factor"] for item in probes if item["admissible"]]
    return factor_order, admissible


def run_pilot() -> None:
    dataset = "dsprites"
    backbone = "dinov2_vits14"
    factor_order, admissible = get_probe_context(dataset, backbone)
    pilot_results = []
    for lambda_nuis in PILOT_GRID["lambda_nuis"]:
        cache_tag = f"orbit_sae_ln{str(lambda_nuis).replace('.', 'p')}"
        result = train_sae(RunConfig(dataset, backbone, "orbit_sae", 11, lambda_nuis=lambda_nuis, cache_tag=cache_tag), DATASETS[dataset]["factors"], admissible, factor_order)
        pilot_results.append(result)
    for lambda_cf in PILOT_GRID["lambda_cf"]:
        cache_tag = f"fb_sae_lc{str(lambda_cf).replace('.', 'p')}"
        result = train_sae(RunConfig(dataset, backbone, "fb_sae", 11, lambda_cf=lambda_cf, cache_tag=cache_tag), DATASETS[dataset]["factors"], admissible, factor_order)
        pilot_results.append(result)
        cache_tag = f"fb_osae_ln{str(PILOT_GRID['lambda_nuis'][0]).replace('.', 'p')}_lc{str(lambda_cf).replace('.', 'p')}"
        result = train_sae(RunConfig(dataset, backbone, "fb_osae", 11, lambda_nuis=PILOT_GRID["lambda_nuis"][0], lambda_cf=lambda_cf, cache_tag=cache_tag), DATASETS[dataset]["factors"], admissible, factor_order)
        pilot_results.append(result)
    for ra_relax in PILOT_GRID["ra_relax"]:
        cache_tag = f"ra_sae_rr{str(ra_relax).replace('.', 'p')}"
        result = train_sae(RunConfig(dataset, backbone, "ra_sae", 11, ra_relax=ra_relax, cache_tag=cache_tag), DATASETS[dataset]["factors"], admissible, factor_order)
        pilot_results.append(result)
    write_json(Path("exp/pilot/results.json"), pilot_results)


def select_pilot_hparams() -> dict[str, float]:
    pilot = read_json(Path("exp/pilot/results.json"))
    selected = {"lambda_nuis": 0.1, "lambda_cf": 0.3, "ra_relax": 0.01}
    families = {}
    for entry in pilot:
        families.setdefault(entry["experiment"], []).append(entry)
    for method, entries in families.items():
        best_ve = max(item["selection_metrics"]["val_variance_explained"] for item in entries)
        eligible = [item for item in entries if item["selection_metrics"]["val_variance_explained"] >= 0.95 * best_ve]
        eligible.sort(
            key=lambda item: (
                item["selection_metrics"]["val_tfcc"],
                -item["metrics"]["mean_l0"],
                -item["metrics"]["reconstruction_mse"],
            ),
            reverse=True,
        )
        chosen = eligible[0]
        if method == "orbit_sae":
            selected["lambda_nuis"] = chosen["config"]["lambda_nuis"]
        if method == "fb_sae":
            selected["lambda_cf"] = chosen["config"]["lambda_cf"]
        if method == "ra_sae":
            selected["ra_relax"] = chosen["config"]["ra_relax"]
    return selected


def run_main_matrix() -> None:
    hparams = select_pilot_hparams()
    registry_path = META_ROOT / "run_registry.csv"
    primary_methods = ["topk_sae", "ra_sae", "orbit_sae", "fb_sae", "fb_osae"]
    for dataset in DATASETS:
        factor_order, admissible = get_probe_context(dataset, "dinov2_vits14")
        for method, seed in itertools.product(primary_methods, SEEDS):
            cfg = RunConfig(dataset, "dinov2_vits14", method, seed, lambda_nuis=hparams["lambda_nuis"], lambda_cf=hparams["lambda_cf"], ra_relax=hparams["ra_relax"])
            result = train_sae(cfg, DATASETS[dataset]["factors"], admissible, factor_order)
            append_registry_row(registry_path, {"stage": "train", "dataset": dataset, "backbone": "dinov2_vits14", "method": method, "seed": seed, "tfcc": result["metrics"]["tfcc"]})
    factor_order, admissible = get_probe_context("dsprites", "openclip_vit_b32")
    for method in primary_methods:
        cfg = RunConfig("dsprites", "openclip_vit_b32", method, SENSITIVITY_SEED, lambda_nuis=hparams["lambda_nuis"], lambda_cf=hparams["lambda_cf"], ra_relax=hparams["ra_relax"])
        result = train_sae(cfg, DATASETS["dsprites"]["factors"], admissible, factor_order)
        append_registry_row(registry_path, {"stage": "train", "dataset": "dsprites", "backbone": "openclip_vit_b32", "method": method, "seed": SENSITIVITY_SEED, "tfcc": result["metrics"]["tfcc"]})


def _load_model(dataset: str, backbone: str, method: str, seed: int) -> LinearSAE:
    train_x, _ = load_clean_features(dataset, backbone, "train")
    input_dim = train_x.shape[1]
    latent_dim = input_dim * 4
    topk = 32 if input_dim <= 512 else 64
    anchor_bank = build_anchor_bank(train_x, latent_dim) if method == "ra_sae" else None
    model = LinearSAE(input_dim, latent_dim, topk, method, anchor_bank=anchor_bank)
    ckpt = torch.load(Path("checkpoints") / dataset / backbone / method / str(seed) / "model.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(select_device())
    model.eval()
    return model


def _make_reordered_model(dataset: str, backbone: str, method: str, seed: int, pseudo_spec: dict) -> LinearSAE:
    model = _load_model(dataset, backbone, method, seed)
    original_forward = model.forward

    def reordered_forward(x):
        output = original_forward(x)
        output["z"] = reorder_latents(output["z"], pseudo_spec["unit_order"])
        return output

    model.forward = reordered_forward
    return model


def _per_seed_rows() -> pd.DataFrame:
    rows = []
    for path in RESULT_ROOT.glob("*/*/*/*/results.json"):
        payload = read_json(path)
        rows.append(
            {
                "dataset": payload["dataset"],
                "backbone": payload["backbone"],
                "method": payload["experiment"],
                "seed": payload["seed"],
                **payload["metrics"],
            }
        )
    return pd.DataFrame(rows)


def _save_permutation_controls() -> list[dict]:
    rng = np.random.default_rng(0)
    outputs = []
    for dataset in DATASETS:
        factor_names = DATASETS[dataset]["factors"]
        for method in ["fb_sae", "fb_osae"]:
            for seed in SEEDS:
                result_dir = RESULT_ROOT / dataset / "dinov2_vits14" / method / str(seed)
                out_path = result_dir / "permutation_control.json"
                if out_path.exists():
                    outputs.append(read_json(out_path))
                    continue
                _, admissible = get_probe_context(dataset, "dinov2_vits14")
                model = _load_model(dataset, "dinov2_vits14", method, seed)
                permutations = []
                for trial in range(20):
                    perm = rng.permutation(len(factor_names)).tolist()
                    eval_payload = evaluate_method(model, dataset, "dinov2_vits14", "test", factor_names, admissible, permute=perm)
                    permutations.append({"trial": trial, "perm": perm, "tfcc": eval_payload["tfcc_mean"], "tba": eval_payload["tba_mean"]})
                payload = {
                    "dataset": dataset,
                    "backbone": "dinov2_vits14",
                    "method": method,
                    "seed": seed,
                    "chance_tba": 1.0 / len(factor_names),
                    "permutations": permutations,
                    "mean_tfcc": float(np.mean([p["tfcc"] for p in permutations])),
                    "mean_tba": float(np.mean([p["tba"] for p in permutations])),
                }
                write_json(out_path, payload)
                outputs.append(payload)
    return outputs


def _save_pseudoblock_robustness() -> list[dict]:
    outputs = []
    rng = np.random.default_rng(0)
    for dataset in DATASETS:
        factor_names = DATASETS[dataset]["factors"]
        factor_order, admissible = get_probe_context(dataset, "dinov2_vits14")
        train_pairs = read_json(Path("pairs") / dataset / "train" / "pairs.json")
        half_factor_pairs = {factor_name: pairs[: max(1, len(pairs) // 2)] for factor_name, pairs in train_pairs["factor_pairs"].items()}
        nuisance_pairs = train_pairs["nuisance_pairs"]
        half_nuisance_pairs = nuisance_pairs[: max(1, len(nuisance_pairs) // 2)]
        for method in ["topk_sae", "ra_sae", "orbit_sae"]:
            for seed in SEEDS:
                result_dir = RESULT_ROOT / dataset / "dinov2_vits14" / method / str(seed)
                out_path = result_dir / "pseudo_block_robustness.json"
                if out_path.exists():
                    outputs.append(read_json(out_path))
                    continue
                base_spec = read_json(result_dir / "pseudo_blocks.json")
                base_eval = read_json(result_dir / "results.json")["metrics"]
                model = _load_model(dataset, "dinov2_vits14", method, seed)
                half_payload = construct_pseudo_blocks_from_pairs(
                    model, dataset, "dinov2_vits14", factor_names, factor_order, half_factor_pairs, half_nuisance_pairs
                )
                half_spec = unit_block_payload_to_spec(half_payload)
                reordered_model = _make_reordered_model(dataset, "dinov2_vits14", method, seed, half_spec)
                half_eval = evaluate_method(reordered_model, dataset, "dinov2_vits14", "test", factor_names, admissible, pseudo_blocks=half_spec)
                payload = {
                    "dataset": dataset,
                    "backbone": "dinov2_vits14",
                    "method": method,
                    "seed": seed,
                    "base_tfcc": base_eval["tfcc"],
                    "base_tba": base_eval["tba"],
                    "half_train_tfcc": half_eval["tfcc_mean"],
                    "half_train_tba": half_eval["tba_mean"],
                    "delta_tfcc": half_eval["tfcc_mean"] - base_eval["tfcc"],
                    "delta_tba": half_eval["tba_mean"] - base_eval["tba"],
                    "base_pseudo_blocks": base_spec,
                    "half_pseudo_blocks": half_spec,
                }
                write_json(out_path, payload)
                outputs.append(payload)
    return outputs


def _holm_correct(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    corrected = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * p_values[idx]
        running = max(running, adjusted)
        corrected[idx] = min(1.0, running)
    return corrected


def _paired_tests(df: pd.DataFrame) -> list[dict]:
    tests = []
    comparisons = [
        ("fb_sae", "topk_sae"),
        ("fb_sae", "ra_sae"),
        ("fb_sae", "orbit_sae"),
        ("fb_osae", "fb_sae"),
    ]
    raw_p = []
    pending = []
    primary = df[df["backbone"] == "dinov2_vits14"]
    for dataset in DATASETS:
        subset = primary[primary["dataset"] == dataset]
        for left, right in comparisons:
            lhs = subset[subset["method"] == left].sort_values("seed")
            rhs = subset[subset["method"] == right].sort_values("seed")
            merged = lhs[["seed", "tfcc", "tba"]].merge(rhs[["seed", "tfcc", "tba"]], on="seed", suffixes=("_lhs", "_rhs"))
            diff = merged["tfcc_lhs"] - merged["tfcc_rhs"]
            if len(diff) >= 2 and not np.allclose(diff.values, diff.values[0]):
                stat = stats.ttest_rel(merged["tfcc_lhs"], merged["tfcc_rhs"])
                p_value = float(stat.pvalue)
            else:
                p_value = 1.0
            raw_p.append(p_value)
            pending.append(
                {
                    "dataset": dataset,
                    "metric": "tfcc",
                    "left": left,
                    "right": right,
                    "seed_deltas": diff.tolist(),
                    "mean_delta": float(diff.mean()),
                    "raw_p": p_value,
                }
            )
    corrected = _holm_correct(raw_p)
    for item, p_corr in zip(pending, corrected):
        item["holm_p"] = p_corr
        item["significant_0.05"] = p_corr < 0.05
        tests.append(item)
    return tests


def aggregate_results() -> None:
    df = _per_seed_rows()
    permutation_controls = _save_permutation_controls()
    robustness_outputs = _save_pseudoblock_robustness()
    summary = []
    factor_rows = []
    for (dataset, backbone, method), group in df.groupby(["dataset", "backbone", "method"]):
        summary.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "method": method,
                "tfcc_mean": float(group["tfcc"].mean()),
                "tfcc_std": float(group["tfcc"].std(ddof=0)),
                "tba_mean": float(group["tba"].mean()),
                "tba_std": float(group["tba"].std(ddof=0)),
                "nuisance_inv_mean": float(group["nuisance_inv"].mean()),
                "nuisance_inv_std": float(group["nuisance_inv"].std(ddof=0)),
                "variance_explained_mean": float(group["variance_explained"].mean()),
                "variance_explained_std": float(group["variance_explained"].std(ddof=0)),
                "mean_l0_mean": float(group["mean_l0"].mean()),
                "mean_l0_std": float(group["mean_l0"].std(ddof=0)),
                "reconstruction_mse_mean": float(group["reconstruction_mse"].mean()),
                "reconstruction_mse_std": float(group["reconstruction_mse"].std(ddof=0)),
            }
        )
    for path in RESULT_ROOT.glob("*/*/*/*/results.json"):
        payload = read_json(path)
        for factor_name, tfcc in payload["metrics"]["tfcc_by_factor"].items():
            factor_rows.append(
                {
                    "dataset": payload["dataset"],
                    "backbone": payload["backbone"],
                    "method": payload["experiment"],
                    "seed": payload["seed"],
                    "factor": factor_name,
                    "tfcc": tfcc,
                    "tba": payload["metrics"]["tba_by_factor"].get(factor_name),
                }
            )
    probe_rows = []
    for path in RESULT_ROOT.glob("*/*/probes.json"):
        for item in read_json(path):
            probe_rows.append({"dataset": path.parts[-3], "backbone": path.parts[-2], **item})
    paired_tests = _paired_tests(df)
    ablations = {
        "fb_osae_vs_fb_sae": [row for row in paired_tests if row["left"] == "fb_osae" and row["right"] == "fb_sae"],
        "fb_sae_vs_unpartitioned": [row for row in paired_tests if row["left"] == "fb_sae" and row["right"] in {"topk_sae", "ra_sae", "orbit_sae"}],
        "permutation_control": permutation_controls,
        "pseudo_block_robustness": robustness_outputs,
    }
    backbone_rankings = []
    for backbone in ["dinov2_vits14", "openclip_vit_b32"]:
        subset = [row for row in summary if row["dataset"] == "dsprites" and row["backbone"] == backbone]
        ranking = sorted(subset, key=lambda row: row["tfcc_mean"], reverse=True)
        backbone_rankings.append({"dataset": "dsprites", "backbone": backbone, "ranking": [row["method"] for row in ranking]})
    output = {
        "summary": summary,
        "factor_summary": factor_rows,
        "probes": probe_rows,
        "paired_tests": paired_tests,
        "ablations": ablations,
        "backbone_rankings": backbone_rankings,
    }
    write_json(Path("results.json"), output)
    df.to_csv(Path("results/all_runs.csv"), index=False)
    pd.DataFrame(summary).to_csv(Path("results/summary.csv"), index=False)
    pd.DataFrame(factor_rows).to_csv(Path("results/factor_summary.csv"), index=False)
    write_json(Path("results/paired_tests.json"), paired_tests)
    write_json(Path("results/ablation_summary.json"), ablations)
    write_json(Path("exp/eval/results.json"), output)


def run_figures() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    ensure_dir(Path("figures"))
    summary = pd.read_csv(Path("results/summary.csv"))
    factor_summary = pd.read_csv(Path("results/factor_summary.csv"))
    results_payload = read_json(Path("results.json"))
    probes = pd.DataFrame(results_payload["probes"])

    summary["label"] = summary["dataset"] + "\n" + summary["backbone"]
    table = summary.pivot(index="method", columns="label", values="tfcc_mean").sort_index()
    plt.figure(figsize=(10, 4.5))
    sns.heatmap(table, annot=True, fmt=".3f", cmap="crest", cbar_kws={"label": "Mean TFCC"})
    plt.title("Main Results Table: TFCC")
    plt.tight_layout()
    plt.savefig("figures/main_results_table.png")
    plt.savefig("figures/main_results_table.pdf")
    plt.close()

    admissible_lookup = {
        (row["dataset"], row["backbone"], row["factor"]): bool(row["admissible"])
        for row in results_payload["probes"]
    }
    factor_summary = factor_summary[
        factor_summary.apply(lambda row: admissible_lookup.get((row["dataset"], row["backbone"], row["factor"]), False), axis=1)
    ]
    for dataset in ["dsprites", "shapes3d"]:
        subset = factor_summary[factor_summary["dataset"] == dataset]
        if subset.empty:
            continue
        plt.figure(figsize=(12, 5))
        sns.barplot(data=subset[subset["backbone"] == "dinov2_vits14"], x="factor", y="tfcc", hue="method", errorbar="sd")
        plt.title(f"{dataset} DINOv2 per-factor TFCC")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"figures/{dataset}_per_factor_tfcc.png")
        plt.savefig(f"figures/{dataset}_per_factor_tfcc.pdf")
        plt.close()

    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=summary, x="nuisance_inv_mean", y="tfcc_mean", hue="method", style="dataset")
    plt.xlabel("Nuisance Leakage")
    plt.ylabel("Mean TFCC")
    plt.tight_layout()
    plt.savefig("figures/nuisance_vs_tfcc.png")
    plt.savefig("figures/nuisance_vs_tfcc.pdf")
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=probes, x="factor", y="test_metric", hue="backbone")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("figures/probe_admissibility.png")
    plt.savefig("figures/probe_admissibility.pdf")
    plt.close()

    heatmap_rows = []
    for method in ["topk_sae", "ra_sae", "orbit_sae", "fb_sae", "fb_osae"]:
        path = RESULT_ROOT / "dsprites" / "dinov2_vits14" / method / "11" / "results.json"
        if not path.exists():
            continue
        payload = read_json(path)
        for factor_name, block_data in payload["selection_metrics"]["val_block_changes_by_factor"].items():
            heatmap_rows.extend(
                [
                    {"method": method, "factor": factor_name, "block": "inv", "value": block_data["inv"]},
                    {"method": method, "factor": factor_name, "block": "target", "value": block_data["target"]},
                    {"method": method, "factor": factor_name, "block": "off_target_mean", "value": block_data["off_target_mean"]},
                    {"method": method, "factor": factor_name, "block": "residual", "value": block_data["residual"]},
                ]
            )
    if heatmap_rows:
        heatmap_df = pd.DataFrame(heatmap_rows)
        for method, group in heatmap_df.groupby("method"):
            table = group.pivot(index="factor", columns="block", values="value")
            plt.figure(figsize=(6, 4))
            sns.heatmap(table, annot=True, fmt=".3f", cmap="mako")
            plt.title(f"{method} block-change diagnostics")
            plt.tight_layout()
            plt.savefig(f"figures/{method}_block_change_heatmap.png")
            plt.savefig(f"figures/{method}_block_change_heatmap.pdf")
            plt.close()

    ablation_rows = []
    for row in results_payload["ablations"]["fb_osae_vs_fb_sae"]:
        ablation_rows.append({"dataset": row["dataset"], "comparison": "fb_osae_vs_fb_sae", "metric": "tfcc_delta", "value": row["mean_delta"]})
    for row in results_payload["ablations"]["fb_sae_vs_unpartitioned"]:
        ablation_rows.append({"dataset": row["dataset"], "comparison": f"fb_sae_vs_{row['right']}", "metric": "tfcc_delta", "value": row["mean_delta"]})
    for row in results_payload["ablations"]["permutation_control"]:
        ablation_rows.append({"dataset": row["dataset"], "comparison": row["method"], "metric": "permuted_tba", "value": row["mean_tba"]})
    for row in results_payload["ablations"]["pseudo_block_robustness"]:
        ablation_rows.append({"dataset": row["dataset"], "comparison": row["method"], "metric": "pseudo_block_delta_tfcc", "value": row["delta_tfcc"]})
    if ablation_rows:
        ablation_df = pd.DataFrame(ablation_rows)
        plt.figure(figsize=(12, 5))
        sns.barplot(data=ablation_df, x="comparison", y="value", hue="metric")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig("figures/ablation_summary.png")
        plt.savefig("figures/ablation_summary.pdf")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["env", "data", "features", "probes", "pilot", "main", "aggregate", "figures", "all"])
    args = parser.parse_args()
    if args.stage in {"env", "all"}:
        run_env_setup()
    if args.stage in {"data", "all"}:
        run_data_prep()
    if args.stage in {"features", "all"}:
        run_feature_cache()
    if args.stage in {"probes", "all"}:
        run_probes()
    if args.stage in {"pilot", "all"}:
        run_pilot()
    if args.stage in {"main", "all"}:
        run_main_matrix()
    if args.stage in {"aggregate", "all"}:
        aggregate_results()
    if args.stage in {"figures", "all"}:
        run_figures()


if __name__ == "__main__":
    main()
