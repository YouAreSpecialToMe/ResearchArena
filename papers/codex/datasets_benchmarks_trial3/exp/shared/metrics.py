from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _group_predictions(predictions: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        grouped[row["cluster_id"]][row["question_id"]] = row
    return grouped


def _strict_equal(row: dict[str, Any]) -> float:
    return float(row["prediction_final"].strip() == row["gold"].strip())


def _failure_mode(q0: bool, q1: bool, q2: bool, q3: bool, q4: bool, same_paraphrase: bool, different_flip: bool) -> str:
    pir = q0 and q1 and q2 and same_paraphrase
    if pir and q3 and different_flip and not q4:
        return "format_only_failure"
    if not pir and q3:
        return "paraphrase_failure"
    if pir and (not q3 or not different_flip):
        return "flip_update_failure"
    if not pir and not q3:
        return "combined_failure"
    return "pass"


def _aggregate(cluster_rows: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    result = {"overall": {}, "by_family": {}, "by_split": {}, "by_flip_template": {}}
    if not cluster_rows:
        return result
    for metric in metrics:
        result["overall"][metric] = float(np.mean([row[metric] for row in cluster_rows]))
    for key, bucket in [("family", "by_family"), ("construction_split", "by_split"), ("flip_template", "by_flip_template")]:
        labels = sorted({row[key] for row in cluster_rows})
        result[bucket] = {
            label: {metric: float(np.mean([row[metric] for row in cluster_rows if row[key] == label])) for metric in metrics}
            for label in labels
        }
    counts = defaultdict(int)
    for row in cluster_rows:
        counts[row["failure_mode"]] += 1
    result["failure_mode_counts"] = dict(counts)
    return result


def _interval(values: list[float]) -> dict[str, float]:
    values_np = np.array(values, dtype=float)
    return {
        "mean": float(values_np.mean()),
        "ci_low": float(np.quantile(values_np, 0.025)),
        "ci_high": float(np.quantile(values_np, 0.975)),
    }


def bootstrap_cluster_rows(cluster_rows: list[dict[str, Any]], seed: int, n: int = 1000) -> dict[str, Any]:
    metric_names = [
        "q0_acc",
        "mean_semantic_variant_acc",
        "pir",
        "bfa",
        "css",
        "fca",
        "ablation_remove_paraphrase",
        "ablation_remove_flip",
        "ablation_include_q4",
        "ablation_exact_string_css",
    ]
    delta_names = {
        "q0_minus_css": ("q0_acc", "css"),
        "uncoupled_minus_css": ("mean_semantic_variant_acc", "css"),
        "pir_minus_css": ("pir", "css"),
        "bfa_minus_css": ("bfa", "css"),
        "remove_paraphrase_minus_css": ("ablation_remove_paraphrase", "css"),
        "remove_flip_minus_css": ("ablation_remove_flip", "css"),
        "include_q4_minus_css": ("ablation_include_q4", "css"),
        "exact_string_minus_css": ("ablation_exact_string_css", "css"),
    }
    if not cluster_rows:
        return {"metrics": {}, "paired_deltas": {}, "minimum_detectable_effect_css": 0.0}
    rng = np.random.default_rng(seed)
    arr = np.array(cluster_rows, dtype=object)
    draws = {metric: [] for metric in metric_names}
    deltas = {name: [] for name in delta_names}
    for _ in range(n):
        sample = arr[rng.integers(0, len(arr), len(arr))]
        means = {metric: float(np.mean([row[metric] for row in sample])) for metric in metric_names}
        for metric in metric_names:
            draws[metric].append(means[metric])
        for name, (left, right) in delta_names.items():
            deltas[name].append(means[left] - means[right])
    return {
        "metrics": {metric: _interval(values) for metric, values in draws.items()},
        "paired_deltas": {metric: _interval(values) for metric, values in deltas.items()},
        "minimum_detectable_effect_css": float(2 * np.std(draws["css"])),
    }


def compute_metrics(predictions: list[dict[str, Any]], bootstrap_seed: int = 20260325) -> dict[str, Any]:
    grouped = _group_predictions(predictions)
    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, items in grouped.items():
        q0 = items["q0"]["correct"]
        q1 = items["q1"]["correct"]
        q2 = items["q2"]["correct"]
        q3 = items["q3"]["correct"]
        q4 = items["q4"]["correct"]
        same_paraphrase = (
            items["q0"]["prediction_normalized"]
            == items["q1"]["prediction_normalized"]
            == items["q2"]["prediction_normalized"]
        )
        different_flip = items["q3"]["prediction_normalized"] != items["q0"]["prediction_normalized"]
        pir = q0 and q1 and q2 and same_paraphrase
        css = pir and q3 and different_flip
        exact_q0 = _strict_equal(items["q0"])
        exact_q1 = _strict_equal(items["q1"])
        exact_q2 = _strict_equal(items["q2"])
        exact_q3 = _strict_equal(items["q3"])
        exact_q4 = _strict_equal(items["q4"])
        exact_same_paraphrase = (
            items["q0"]["prediction_final"].strip()
            == items["q1"]["prediction_final"].strip()
            == items["q2"]["prediction_final"].strip()
        )
        exact_different_flip = items["q3"]["prediction_final"].strip() != items["q0"]["prediction_final"].strip()
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "family": items["q0"]["family"],
                "construction_split": items["q0"]["construction_split"],
                "flip_template": items["q0"]["flip_template"],
                "q0_acc": float(q0),
                "mean_semantic_variant_acc": float(np.mean([q0, q1, q2, q3])),
                "pir": float(pir),
                "bfa": float(q3),
                "css": float(css),
                "fca": float(q4),
                "ablation_remove_paraphrase": float(q0 and q3 and different_flip),
                "ablation_remove_flip": float(pir),
                "ablation_include_q4": float(css and q4),
                "exact_q0_acc": exact_q0,
                "exact_mean_semantic_variant_acc": float(np.mean([exact_q0, exact_q1, exact_q2, exact_q3])),
                "exact_pir": float(exact_q0 and exact_q1 and exact_q2 and exact_same_paraphrase),
                "exact_bfa": exact_q3,
                "ablation_exact_string_css": float(
                    exact_q0 and exact_q1 and exact_q2 and exact_q3 and exact_same_paraphrase and exact_different_flip
                ),
                "exact_fca": exact_q4,
                "failure_mode": _failure_mode(q0, q1, q2, q3, q4, same_paraphrase, different_flip),
            }
        )

    metric_names = [
        "q0_acc",
        "mean_semantic_variant_acc",
        "pir",
        "bfa",
        "css",
        "fca",
        "ablation_remove_paraphrase",
        "ablation_remove_flip",
        "ablation_include_q4",
        "exact_q0_acc",
        "exact_mean_semantic_variant_acc",
        "exact_pir",
        "exact_bfa",
        "ablation_exact_string_css",
        "exact_fca",
    ]
    result = _aggregate(cluster_rows, metric_names)
    result["bootstrap"] = bootstrap_cluster_rows(cluster_rows, bootstrap_seed)
    result["bootstrap_by_split"] = {
        split: bootstrap_cluster_rows([row for row in cluster_rows if row["construction_split"] == split], bootstrap_seed)
        for split in sorted({row["construction_split"] for row in cluster_rows})
    }
    result["cluster_rows"] = cluster_rows
    return result
