from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def directed_metrics(pred_adj: np.ndarray, true_adj: np.ndarray) -> dict[str, float]:
    pred_edges = {(i, j) for i, j in zip(*np.where(pred_adj == 1))}
    true_edges = {(i, j) for i, j in zip(*np.where(true_adj == 1))}
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    skeleton_pred = {(min(i, j), max(i, j)) for i, j in pred_edges}
    skeleton_true = {(min(i, j), max(i, j)) for i, j in true_edges}
    shd = len(skeleton_pred ^ skeleton_true) + len(skeleton_pred & skeleton_true) - tp
    unresolved_true = len([1 for i, j in skeleton_true if pred_adj[i, j] == 0 and pred_adj[j, i] == 0])
    return {
        "shd": float(shd),
        "directed_precision": float(precision),
        "directed_recall": float(recall),
        "directed_f1": float(f1),
        "correct_oriented_edges": float(tp),
        "unresolved_true_adjacencies": float(unresolved_true),
    }


def auc_over_samples(samples: list[int], values: list[float], max_budget: int) -> float:
    if len(samples) < 2:
        return 0.0
    area = np.trapezoid(values, samples)
    return float(area / max_budget)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= left) & (y_prob < right if right < 1.0 else y_prob <= right)
        if not np.any(mask):
            continue
        total += abs(y_true[mask].mean() - y_prob[mask].mean()) * (mask.mean())
    return float(total)


def calibration_metrics(y_true: list[int], y_prob: list[float]) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "brier": float(brier_score_loss(yt, yp)),
        "ece10": ece_score(yt, yp, bins=10),
        "mace": float(np.mean(np.abs(yt - yp))),
        "auprc": float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else math.nan,
        "spearman": float(spearmanr(yt, yp).correlation) if len(yt) > 1 else math.nan,
    }
    out["auroc"] = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else math.nan
    return out


def paired_summary(a: list[float], b: list[float]) -> dict[str, float | list[float] | None]:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    diff = arr_a - arr_b
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(diff), size=len(diff))
        boots.append(float(diff[idx].mean()))
    boots.sort()
    try:
        pvalue = float(wilcoxon(diff).pvalue)
    except ValueError:
        pvalue = None
    return {
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "ci95": [float(boots[49]), float(boots[1949])],
        "wilcoxon_p": pvalue,
    }


def bootstrap_metric_summary(values: list[float], seed: int = 0, n_boot: int = 2000) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(float(arr[idx].mean()))
    boots.sort()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95": [float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot) - 1])],
    }
