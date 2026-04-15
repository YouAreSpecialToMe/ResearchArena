from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def gaussian_member_score(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    mu = float(ref_values.mean())
    var = float(ref_values.var() + 1e-6)
    return -((values - mu) ** 2) / (2.0 * var) - 0.5 * np.log(var)


def class_conditional_member_score(values: np.ndarray, labels: np.ndarray, ref_values: np.ndarray, ref_labels: np.ndarray) -> np.ndarray:
    scores = np.zeros_like(values, dtype=np.float64)
    for cls in np.unique(labels):
        mask = labels == cls
        ref_mask = ref_labels == cls
        if not np.any(ref_mask):
            scores[mask] = gaussian_member_score(values[mask], ref_values)
            continue
        scores[mask] = gaussian_member_score(values[mask], ref_values[ref_mask])
    return scores


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr: float = 0.01) -> tuple[float, float]:
    threshold = float(np.quantile(nonmember_scores, 1.0 - fpr))
    tpr = float((member_scores >= threshold).mean())
    return tpr, threshold


def precision_at_top_k(q: np.ndarray, s: np.ndarray, frac: float = 0.10) -> float:
    k = max(1, int(len(q) * frac))
    q_top = np.argsort(q)[-k:]
    s_top = np.argsort(s)[-k:]
    return float(len(set(q_top.tolist()) & set(s_top.tolist())) / k)


def jaccard_overlaps(sets: list[set[int]]) -> list[float]:
    overlaps = []
    for a, b in zip(sets, sets[1:]):
        if not a and not b:
            overlaps.append(1.0)
        else:
            overlaps.append(float(len(a & b) / max(1, len(a | b))))
    return overlaps


def compute_forecast_metrics(q: np.ndarray, attack_scores: np.ndarray) -> dict:
    corr = float(spearmanr(q, attack_scores).statistic)
    if np.isnan(corr):
        corr = 0.0
    return {
        "spearman_q_attack": corr,
        "precision_at_10": precision_at_top_k(q, attack_scores, 0.10),
    }
