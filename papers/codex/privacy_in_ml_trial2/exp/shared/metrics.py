from __future__ import annotations

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, roc_curve


def _beta_lower(successes: int, total: int, alpha: float) -> float:
    if total <= 0:
        return 0.0
    if successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes + 0.5, total - successes + 0.5))


def _beta_upper(successes: int, total: int, alpha: float) -> float:
    if total <= 0:
        return 1.0
    if successes >= total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 0.5, total - successes + 0.5))


def empirical_epsilon_lower_bound(
    y_true: np.ndarray,
    scores: np.ndarray,
    delta: float = 1e-5,
    ci_alpha: float = 0.05,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(y_true).size < 2:
        return 0.0

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    pos_total = int(y_true.sum())
    neg_total = int((1 - y_true).sum())
    eps_values = []
    for fp_rate, tp_rate, threshold in zip(fpr, tpr, thresholds):
        preds = scores >= threshold
        tp = int(np.logical_and(preds, y_true == 1).sum())
        fp = int(np.logical_and(preds, y_true == 0).sum())
        lower_tpr = _beta_lower(tp, pos_total, ci_alpha)
        upper_tpr = _beta_upper(tp, pos_total, ci_alpha)
        upper_fpr = _beta_upper(fp, neg_total, ci_alpha)
        lower_fpr = _beta_lower(fp, neg_total, ci_alpha)
        eps1 = np.log(max(lower_tpr - delta, 1e-12) / max(upper_fpr, 1e-12))
        eps2 = np.log(max(1.0 - upper_fpr - delta, 1e-12) / max(1.0 - lower_tpr, 1e-12))
        eps3 = np.log(max(lower_fpr - delta, 1e-12) / max(upper_tpr, 1e-12))
        eps4 = np.log(max(1.0 - upper_tpr - delta, 1e-12) / max(1.0 - lower_fpr, 1e-12))
        eps_values.append(max(eps1, eps2, eps3, eps4))
        if fp_rate == 0.0 and tp_rate == 0.0:
            continue
    return float(max(0.0, np.max(eps_values) if eps_values else 0.0))


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def audit_metrics(y_true: np.ndarray, scores: np.ndarray, delta: float = 1e-5) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "eps_lb": empirical_epsilon_lower_bound(y_true, scores, delta=delta),
        "tpr@0.1%": tpr_at_fpr(y_true, scores, 0.001),
        "tpr@1%": tpr_at_fpr(y_true, scores, 0.01),
        "tpr@5%": tpr_at_fpr(y_true, scores, 0.05),
    }
