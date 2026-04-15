from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from scipy.stats import spearmanr


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def binary_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if not np.any(mask):
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-8, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def early_commitment_rate(
    probs: np.ndarray,
    final_preds: np.ndarray | None = None,
    threshold: float = 0.8,
) -> float:
    if probs.size == 0:
        return float("nan")
    committed = probs.max(axis=1) > threshold
    if final_preds is not None and len(final_preds) == len(probs):
        committed = committed & (probs.argmax(axis=1) == final_preds)
    return float(committed.mean())


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "ece": binary_ece(probs, labels, n_bins=15),
        "brier": brier_score(probs, labels),
        "log_loss": float(log_loss(labels, probs, labels=[0, 1])),
    }


def civil_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    identity_flags: np.ndarray,
) -> dict[str, float]:
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    overall_f1 = float(f1_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    worst = None
    for group_name, flags in identity_flags.items():
        mask = flags == 1
        if mask.sum() == 0:
            continue
        group_f1 = float(f1_score(labels[mask], preds[mask], zero_division=0))
        worst = group_f1 if worst is None else min(worst, group_f1)
    return {
        "overall_f1": overall_f1,
        "macro_f1": macro_f1,
        "worst_group_f1": float(worst) if worst is not None else float("nan"),
        "ece": binary_ece(probs, labels),
    }


def proxy_validation(
    intermediate_entropy: np.ndarray,
    mask_conf_drop: np.ndarray,
    label_flip: np.ndarray,
    actor_presence: np.ndarray,
    final_confidence: np.ndarray,
    correctness: np.ndarray,
) -> dict[str, float]:
    rho, _ = spearmanr(intermediate_entropy, mask_conf_drop)
    try:
        auroc = float(roc_auc_score(label_flip, -intermediate_entropy))
    except Exception:
        auroc = float("nan")
    X = np.stack([actor_presence, final_confidence, intermediate_entropy], axis=1)
    if len(np.unique(correctness)) < 2:
        return {
            "spearman_entropy_vs_conf_drop": float(rho),
            "auroc_predict_flip": auroc,
            "residual_coef_actor_presence": float("nan"),
            "residual_coef_final_confidence": float("nan"),
            "residual_coef_intermediate_entropy": float("nan"),
            "residual_intercept": float("nan"),
        }
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, correctness)
    return {
        "spearman_entropy_vs_conf_drop": float(rho),
        "auroc_predict_flip": auroc,
        "residual_coef_actor_presence": float(clf.coef_[0][0]),
        "residual_coef_final_confidence": float(clf.coef_[0][1]),
        "residual_coef_intermediate_entropy": float(clf.coef_[0][2]),
        "residual_intercept": float(clf.intercept_[0]),
    }


def summarize_seed_metrics(metrics_list: list[dict[str, float]]) -> dict[str, dict[str, float] | list[float]]:
    keys = sorted({k for m in metrics_list for k in m})
    summary: dict[str, dict[str, float] | list[float]] = {}
    for key in keys:
        vals = [float(m[key]) for m in metrics_list if key in m]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "values": [float(v) for v in vals],
        }
    return summary


def effective_risk_threshold(risk: np.ndarray) -> float:
    if risk.size == 0:
        return 0.0
    raw_median = float(np.median(risk))
    if raw_median > 0.0:
        return raw_median
    positive = risk[risk > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.median(positive))
