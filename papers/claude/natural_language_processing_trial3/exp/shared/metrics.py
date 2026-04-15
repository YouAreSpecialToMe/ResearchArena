import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def auroc(scores, labels):
    """AUROC for predicting correctness from confidence."""
    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def auprc(scores, labels):
    """Average precision score."""
    if len(set(labels)) < 2:
        return float(np.mean(labels))
    return float(average_precision_score(labels, scores))


def ece(scores, labels, n_bins=10):
    """Expected Calibration Error with equal-width bins."""
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    for i in range(n_bins):
        mask = (scores >= bin_boundaries[i]) & (scores < bin_boundaries[i + 1])
        if i == n_bins - 1:  # include right boundary for last bin
            mask = (scores >= bin_boundaries[i]) & (scores <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = scores[mask].mean()
        bin_acc = labels[mask].mean()
        total_ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(total_ece / len(scores)) if len(scores) > 0 else 0.0


def selective_accuracy(scores, labels, coverages=[0.5, 0.7, 0.9]):
    """Accuracy when answering only the top-X% most confident predictions."""
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=float)
    sorted_idx = np.argsort(-scores)  # descending
    result = {}
    for cov in coverages:
        n = max(1, int(len(scores) * cov))
        top_idx = sorted_idx[:n]
        acc = float(labels[top_idx].mean())
        result[f"acc@{int(cov*100)}"] = acc
    return result


def selective_accuracy_curve(scores, labels, n_points=100):
    """Full selective accuracy curve for plotting."""
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=float)
    sorted_idx = np.argsort(-scores)
    coverages = []
    accuracies = []
    for i in range(1, n_points + 1):
        cov = i / n_points
        n = max(1, int(len(scores) * cov))
        top_idx = sorted_idx[:n]
        coverages.append(cov)
        accuracies.append(float(labels[top_idx].mean()))
    return coverages, accuracies
