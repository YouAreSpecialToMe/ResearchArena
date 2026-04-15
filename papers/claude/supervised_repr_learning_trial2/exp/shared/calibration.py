"""Calibration metrics: ECE, MCE, AdaECE, NLL, Brier score."""

import numpy as np
import torch
import torch.nn.functional as F


def compute_calibration_metrics(logits, labels, n_bins=15):
    """Compute all calibration metrics from logits and labels.

    Args:
        logits: (N, C) tensor of logits
        labels: (N,) tensor of true labels
        n_bins: number of bins for ECE/MCE

    Returns:
        dict with ECE, MCE, AdaECE, NLL, Brier, and per-bin reliability data
    """
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    confidences_np = confidences.cpu().numpy()
    accuracies_np = accuracies.cpu().numpy().astype(float)
    probs_np = probs.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # ECE (equal-width bins)
    ece, bin_data = _ece_equal_width(confidences_np, accuracies_np, n_bins)

    # MCE
    mce = max([b['accuracy'] - b['confidence'] for b in bin_data if b['count'] > 0], default=0, key=abs)
    mce = abs(mce)

    # AdaECE (equal-mass bins)
    ada_ece = _ada_ece(confidences_np, accuracies_np, n_bins)

    # NLL
    nll = F.cross_entropy(logits, labels).item()

    # Brier score
    one_hot = np.zeros_like(probs_np)
    one_hot[np.arange(len(labels_np)), labels_np] = 1
    brier = np.mean(np.sum((probs_np - one_hot) ** 2, axis=1))

    return {
        'ece': float(ece),
        'mce': float(mce),
        'ada_ece': float(ada_ece),
        'nll': float(nll),
        'brier': float(brier),
        'reliability_bins': bin_data,
    }


def _ece_equal_width(confidences, accuracies, n_bins):
    """ECE with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum()
        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (count / n) * abs(avg_acc - avg_conf)
            bin_data.append({
                'bin_lo': float(lo), 'bin_hi': float(hi),
                'count': int(count), 'accuracy': float(avg_acc),
                'confidence': float(avg_conf)
            })
        else:
            bin_data.append({
                'bin_lo': float(lo), 'bin_hi': float(hi),
                'count': 0, 'accuracy': 0.0, 'confidence': 0.0
            })

    return ece, bin_data


def _ada_ece(confidences, accuracies, n_bins):
    """Adaptive ECE with equal-mass bins."""
    n = len(confidences)
    sorted_idx = np.argsort(confidences)
    bin_size = n // n_bins

    ada_ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        idx = sorted_idx[start:end]
        if len(idx) == 0:
            continue
        avg_conf = confidences[idx].mean()
        avg_acc = accuracies[idx].mean()
        ada_ece += (len(idx) / n) * abs(avg_acc - avg_conf)

    return ada_ece
