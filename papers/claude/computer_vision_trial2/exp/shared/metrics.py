"""Evaluation metrics for OOD detection and calibration."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """Compute OOD detection metrics.

    Convention: higher score = more likely OOD.

    Args:
        id_scores: (N_id,) scores for in-distribution data
        ood_scores: (N_ood,) scores for OOD data
    Returns:
        dict with AUROC, AUPR_in, AUPR_out, FPR95
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])

    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return {'AUROC': 0.5, 'AUPR_in': 0.5, 'AUPR_out': 0.5, 'FPR95': 1.0}

    auroc = roc_auc_score(labels, scores)
    aupr_out = average_precision_score(labels, scores)
    aupr_in = average_precision_score(1 - labels, -scores)

    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return {
        'AUROC': float(auroc),
        'AUPR_in': float(aupr_in),
        'AUPR_out': float(aupr_out),
        'FPR95': float(fpr95),
    }


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.

    Args:
        confidences: (N,) predicted confidence (max softmax probability)
        accuracies: (N,) binary correctness indicators
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += mask.sum() / len(confidences) * abs(avg_acc - avg_conf)
    return float(ece)


def compute_mce(confidences: np.ndarray, accuracies: np.ndarray,
                n_bins: int = 15) -> float:
    """Compute Maximum Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_ce = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            max_ce = max(max_ce, abs(avg_acc - avg_conf))
    return float(max_ce)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute Brier Score for top-1 predictions.

    Args:
        probs: (N,) predicted probability of top-1 class
        labels: (N,) binary correctness (1 if top-1 is correct)
    Returns:
        Brier score
    """
    return float(np.mean((probs - labels) ** 2))


def compute_calibration_metrics(logits: np.ndarray, labels: np.ndarray,
                                 temperature: float = 1.0) -> dict:
    """Compute all calibration metrics given logits and true labels.

    Args:
        logits: (N, C) raw logits
        labels: (N,) true class indices
        temperature: temperature for scaling
    Returns:
        dict with ECE, MCE, Brier, accuracy
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Softmax
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (predictions == labels).astype(float)
    accuracy = correct.mean()

    ece = compute_ece(confidences, correct)
    mce = compute_mce(confidences, correct)
    brier = compute_brier_score(confidences, correct)

    return {
        'ECE': ece,
        'MCE': mce,
        'Brier': brier,
        'Accuracy': float(accuracy),
    }
