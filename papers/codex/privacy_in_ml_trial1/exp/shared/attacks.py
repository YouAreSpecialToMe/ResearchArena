from __future__ import annotations

from typing import Dict

import numpy as np

from .metrics import entropy, modified_entropy, tpr_at_fpr


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.clip(std, 1e-6, None)
    return -0.5 * np.log(2.0 * np.pi * std**2) - ((x - mean) ** 2) / (2.0 * std**2)


def compute_attack_scores(
    probs: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    lira_stats: Dict[str, np.ndarray] | None = None,
    difficulty_stats: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    idx = np.arange(len(labels))
    true_probs = np.clip(probs[idx, labels], 1e-12, 1.0)
    loss = -np.log(true_probs)
    ent = entropy(probs)
    mod_ent = modified_entropy(probs, labels)
    if difficulty_stats is None:
        class_mean = np.bincount(labels, weights=loss, minlength=probs.shape[1]) / np.maximum(
            np.bincount(labels, minlength=probs.shape[1]), 1
        )
        class_std = np.ones(probs.shape[1], dtype=np.float64)
    else:
        class_mean = difficulty_stats["class_mean"]
        class_std = difficulty_stats["class_std"]
    diff_cal_loss = -((loss - class_mean[labels]) / np.clip(class_std[labels], 1e-6, None))
    scores = {
        "confidence": true_probs,
        "entropy": -ent,
        "modified_entropy": -mod_ent,
        "loss": -loss,
        "difficulty_calibrated_loss": diff_cal_loss,
    }
    if lira_stats is not None:
        true_logits = logits[idx, labels]
        in_ll = gaussian_logpdf(true_logits, lira_stats["in_mean"][labels], lira_stats["in_std"][labels])
        out_ll = gaussian_logpdf(true_logits, lira_stats["out_mean"][labels], lira_stats["out_std"][labels])
        scores["lira_lite"] = in_ll - out_ll
    else:
        scores["lira_lite"] = scores["confidence"]
    return scores


def evaluate_attacks(member_scores: Dict[str, np.ndarray], nonmember_scores: Dict[str, np.ndarray], fpr: float = 0.01):
    metrics = {}
    strongest_name = None
    strongest_value = -1.0
    for name, mem in member_scores.items():
        nm = nonmember_scores[name]
        res = tpr_at_fpr(mem, nm, fpr=fpr)
        metrics[name] = res
        if res["tpr"] > strongest_value:
            strongest_name = name
            strongest_value = res["tpr"]
    metrics["strongest_attack"] = {"name": strongest_name, "tpr": strongest_value, "fpr": fpr}
    return metrics
