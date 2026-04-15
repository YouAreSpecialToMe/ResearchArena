from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from .metrics import bootstrap_cluster_rows
from .utils import FIGURES_DIR, mean_std, write_json


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_model_seed_metrics(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "q0_acc",
        "mean_semantic_variant_acc",
        "pir",
        "bfa",
        "css",
        "fca",
        "exact_q0_acc",
        "exact_mean_semantic_variant_acc",
        "exact_pir",
        "exact_bfa",
        "ablation_exact_string_css",
        "exact_fca",
    ]
    ablation_names = [
        "ablation_remove_paraphrase",
        "ablation_remove_flip",
        "ablation_include_q4",
        "ablation_exact_string_css",
    ]
    aggregate = {"metrics": {}, "ablations": {}, "success_checks": {}, "seed_count": len(seed_metrics)}
    for metric in metric_names:
        aggregate["metrics"][metric] = mean_std([item["overall"][metric] for item in seed_metrics])
    for metric in ablation_names:
        aggregate["ablations"][metric] = mean_std([item["overall"][metric] for item in seed_metrics])
    q0_gap = [item["overall"]["q0_acc"] - item["overall"]["css"] for item in seed_metrics]
    uncoupled_gap = [item["overall"]["mean_semantic_variant_acc"] - item["overall"]["css"] for item in seed_metrics]
    aggregate["success_checks"] = {
        "q0_minus_css": mean_std(q0_gap),
        "uncoupled_minus_css": mean_std(uncoupled_gap),
    }
    timing_rows = [item.get("timing", {}) for item in seed_metrics if item.get("timing")]
    if timing_rows:
        aggregate["timing"] = {
            "mean_latency_seconds": mean_std([row["mean_latency_seconds"] for row in timing_rows]),
            "p95_latency_seconds": mean_std([row["p95_latency_seconds"] for row in timing_rows]),
            "peak_vram_mb": mean_std([row["peak_vram_mb"] for row in timing_rows]),
        }
    combined_rows = []
    for seed_metric in seed_metrics:
        for row in seed_metric["cluster_rows"]:
            copied = row.copy()
            copied["cluster_id"] = f"s{seed_metric['seed']}::{row['cluster_id']}"
            combined_rows.append(copied)
    aggregate["bootstrap_summary"] = bootstrap_cluster_rows(combined_rows, seed=20260325)
    aggregate["split_metrics"] = {}
    aggregate["split_bootstrap_summary"] = {}
    for split in sorted({row["construction_split"] for row in combined_rows}):
        split_rows = [row for row in combined_rows if row["construction_split"] == split]
        aggregate["split_metrics"][split] = {
            "css": float(np.mean([row["css"] for row in split_rows])),
            "q0_minus_css": float(np.mean([row["q0_acc"] - row["css"] for row in split_rows])),
            "ablation_remove_paraphrase": float(np.mean([row["ablation_remove_paraphrase"] for row in split_rows])),
            "ablation_remove_flip": float(np.mean([row["ablation_remove_flip"] for row in split_rows])),
            "ablation_include_q4": float(np.mean([row["ablation_include_q4"] for row in split_rows])),
            "ablation_exact_string_css": float(np.mean([row["ablation_exact_string_css"] for row in split_rows])),
            "bfa": float(np.mean([row["bfa"] for row in split_rows])),
        }
        split_bootstrap = bootstrap_cluster_rows(split_rows, seed=20260325)
        aggregate["split_bootstrap_summary"][split] = {
            "css": split_bootstrap["metrics"]["css"],
            "q0_minus_css": split_bootstrap["paired_deltas"]["q0_minus_css"],
        }
    return aggregate


def combine_cluster_rows(seed_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined_rows = []
    for seed_metric in seed_metrics:
        for row in seed_metric["cluster_rows"]:
            copied = row.copy()
            copied["seed"] = seed_metric["seed"]
            copied["cluster_id"] = f"s{seed_metric['seed']}::{row['cluster_id']}"
            combined_rows.append(copied)
    return combined_rows


def paired_bootstrap_delta(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_metric: str,
    right_metric: str,
    *,
    pair_by_key: bool,
    seed: int = 20260325,
    n: int = 1000,
) -> dict[str, float]:
    if not left_rows or not right_rows:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    if pair_by_key:
        left_map = {row["cluster_id"]: row for row in left_rows}
        right_map = {row["cluster_id"]: row for row in right_rows}
        shared = sorted(set(left_map) & set(right_map))
        pairs = [(left_map[key], right_map[key]) for key in shared]
    else:
        left_sorted = sorted(left_rows, key=lambda row: (row["seed"], row["family"], row["construction_split"], row["cluster_id"]))
        right_sorted = sorted(right_rows, key=lambda row: (row["seed"], row["family"], row["construction_split"], row["cluster_id"]))
        pairs = list(zip(left_sorted, right_sorted))

    if not pairs:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    rng = np.random.default_rng(seed)
    pair_arr = np.array(pairs, dtype=object)
    draws = []
    for _ in range(n):
        sample = pair_arr[rng.integers(0, len(pair_arr), len(pair_arr))]
        left_mean = float(np.mean([left[left_metric] for left, _ in sample]))
        right_mean = float(np.mean([right[right_metric] for _, right in sample]))
        draws.append(left_mean - right_mean)
    values = np.array(draws, dtype=float)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def split_replication_from_rows(
    baseline_rows: list[dict[str, Any]],
    ablated_rows: list[dict[str, Any]],
    baseline_metric: str,
    ablated_metric: str,
) -> dict[str, Any]:
    baseline_splits = {row["construction_split"] for row in baseline_rows}
    ablated_splits = {row["construction_split"] for row in ablated_rows}
    shared_splits = sorted(split for split in baseline_splits & ablated_splits if split in {"A", "B"})
    rows = []
    for split in shared_splits:
        base_split = [row for row in baseline_rows if row["construction_split"] == split]
        alt_split = [row for row in ablated_rows if row["construction_split"] == split]
        base_value = float(np.mean([row[baseline_metric] for row in base_split]))
        alt_value = float(np.mean([row[ablated_metric] for row in alt_split]))
        rows.append({"split": split, "baseline_css": base_value, "ablated_score": alt_value, "delta_vs_css": alt_value - base_value})
    same_direction = False
    if len(rows) >= 2:
        deltas = [row["delta_vs_css"] for row in rows]
        if all(delta == 0 for delta in deltas):
            same_direction = True
        elif all(delta >= 0 for delta in deltas) or all(delta <= 0 for delta in deltas):
            same_direction = True
    return {"by_split": rows, "same_direction_across_splits": same_direction}


def generate_figures(
    model_aggregates: dict[str, dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    construction_rows: list[dict[str, Any]],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _figure1(model_aggregates)
    _figure2(model_aggregates)
    _figure3(failure_rows)
    _figure4(construction_rows)


def _savefig(stem: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{stem}.png", dpi=200)
    plt.savefig(FIGURES_DIR / f"{stem}.pdf")
    plt.close()


def _figure1(model_aggregates: dict[str, dict[str, Any]]) -> None:
    labels = list(model_aggregates.keys())
    metrics = ["q0_acc", "mean_semantic_variant_acc", "pir", "bfa", "css"]
    x = np.arange(len(labels))
    width = 0.15
    plt.figure(figsize=(11, 5))
    for idx, metric in enumerate(metrics):
        vals = [model_aggregates[label]["metrics"][metric]["mean"] for label in labels]
        yerr = np.array(
            [
                [
                    vals[label_idx] - model_aggregates[label]["bootstrap_summary"]["metrics"][metric]["ci_low"],
                    model_aggregates[label]["bootstrap_summary"]["metrics"][metric]["ci_high"] - vals[label_idx],
                ]
                for label_idx, label in enumerate(labels)
            ]
        ).T
        plt.bar(x + (idx - 2) * width, vals, width=width, label=metric, yerr=yerr, capsize=3)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.title("Procedural-core pilot metrics by model")
    plt.legend()
    _savefig("figure1_model_metrics")


def _figure2(model_aggregates: dict[str, dict[str, Any]]) -> None:
    models = list(model_aggregates.keys())
    x = np.arange(len(models))
    width = 0.35
    plt.figure(figsize=(10, 5))
    for idx, split in enumerate(["A", "B"]):
        vals = [model_aggregates[model]["split_metrics"][split]["q0_minus_css"] for model in models]
        yerr = np.array(
            [
                [
                    vals[model_idx] - model_aggregates[model]["split_bootstrap_summary"][split]["q0_minus_css"]["ci_low"],
                    model_aggregates[model]["split_bootstrap_summary"][split]["q0_minus_css"]["ci_high"] - vals[model_idx],
                ]
                for model_idx, model in enumerate(models)
            ]
        ).T
        plt.bar(x + (idx - 0.5) * width, vals, width=width, label=f"Split {split}", yerr=yerr, capsize=3)
    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylabel("q0 - CSS")
    plt.title("Split-wise q0 minus CSS gaps")
    plt.legend()
    _savefig("figure2_split_gap")


def _figure3(failure_rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10, 5))
    models = sorted({row["model"] for row in failure_rows})
    categories = ["paraphrase_failure", "flip_update_failure", "combined_failure", "format_only_failure"]
    bottoms = np.zeros(len(models))
    for category in categories:
        vals = []
        for model in models:
            row = next(
                (item for item in failure_rows if item["model"] == model and item["failure_mode"] == category),
                {"fraction": 0.0},
            )
            vals.append(row["fraction"])
        plt.bar(models, vals, bottom=bottoms, label=category)
        bottoms += np.array(vals)
    plt.ylabel("Fraction of clusters")
    plt.title("Failure mode breakdown")
    plt.legend()
    _savefig("figure3_failure_modes")


def _figure4(construction_rows: list[dict[str, Any]]) -> None:
    families = sorted({row["family"] for row in construction_rows})
    candidates = [float(np.mean([row["candidate_count"] for row in construction_rows if row["family"] == family])) for family in families]
    kept = [float(np.mean([row["kept_count"] for row in construction_rows if row["family"] == family])) for family in families]
    audited = [float(np.mean([row["audited_pass_count"] for row in construction_rows if row["family"] == family])) for family in families]
    rejected = [candidate - audited_pass for candidate, audited_pass in zip(candidates, audited)]
    plt.figure(figsize=(9, 5))
    plt.bar(families, candidates, label="candidates")
    plt.bar(families, audited, label="audited pass")
    plt.bar(families, kept, label="kept")
    plt.bar(families, rejected, bottom=audited, label="rejected before audit pass")
    plt.ylabel("Clusters")
    plt.title("Candidate-to-kept counts by family")
    plt.legend()
    _savefig("figure4_construction_waterfall")


def summarize_success(model_aggregates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    avg_q0_gap = float(np.mean([row["success_checks"]["q0_minus_css"]["mean"] for row in model_aggregates.values()]))
    avg_uncoupled_gap = float(np.mean([row["success_checks"]["uncoupled_minus_css"]["mean"] for row in model_aggregates.values()]))
    split_css = {
        split: {model: agg["split_metrics"][split]["css"] for model, agg in model_aggregates.items()}
        for split in ["A", "B"]
    }
    corr = None
    corr_value = spearmanr(
        [split_css["A"][model] for model in split_css["A"]],
        [split_css["B"][model] for model in split_css["A"]],
    ).statistic
    if not np.isnan(corr_value):
        corr = float(corr_value)
    gap_direction_count = sum(
        int(agg["split_metrics"]["A"]["q0_minus_css"] > 0 and agg["split_metrics"]["B"]["q0_minus_css"] > 0)
        for agg in model_aggregates.values()
    )
    return {
        "primary_avg_q0_minus_css": avg_q0_gap,
        "secondary_avg_uncoupled_minus_css": avg_uncoupled_gap,
        "split_css_spearman": corr,
        "split_gap_same_direction_models": gap_direction_count,
        "per_model_css_mean": {name: agg["metrics"]["css"]["mean"] for name, agg in model_aggregates.items()},
    }


def dump_appendix(path: Path, appendix: dict[str, Any]) -> None:
    write_json(path, appendix)
