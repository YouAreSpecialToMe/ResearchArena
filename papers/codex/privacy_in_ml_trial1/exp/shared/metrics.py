from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -(probs * np.log(probs)).sum(axis=1)


def modified_entropy(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    true_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    term_true = -(1.0 - true_probs) * np.log(true_probs)
    probs_wo = probs.copy()
    probs_wo[np.arange(len(labels)), labels] = 1e-12
    term_rest = -(probs_wo * np.log(np.clip(1.0 - probs_wo, 1e-12, 1.0))).sum(axis=1)
    return term_true + term_rest


def nll_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def classification_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "nll": nll_from_probs(probs, labels),
    }
    if len(np.unique(labels)) > 2:
        out["macro_f1"] = float(f1_score(labels, preds, average="macro"))
    return out


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr: float = 0.01) -> Dict[str, float]:
    threshold = float(np.quantile(nonmember_scores, 1.0 - fpr))
    tpr = float((member_scores >= threshold).mean())
    return {"threshold": threshold, "tpr": tpr, "fpr": fpr}


def bootstrap_delta_ci(
    a: np.ndarray, b: np.ndarray, iterations: int = 2000, seed: int = 0
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(a[idx].mean() - b[idx].mean()))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {"delta_mean": float(np.mean(deltas)), "ci95_low": float(lo), "ci95_high": float(hi)}


def summarize(values: list[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = math.sqrt(float((vx * vx).sum() * (vy * vy).sum()))
    if denom == 0.0:
        return 0.0
    return float((vx * vy).sum() / denom)
