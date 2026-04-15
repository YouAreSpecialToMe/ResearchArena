"""Evaluation metrics for hallucination detection."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score
)
from scipy import stats


def compute_auc_roc(labels, scores):
    """Compute AUC-ROC. labels: 1=hallucinated, scores: higher=more hallucinated."""
    labels = np.array(labels, dtype=float)
    scores = np.nan_to_num(np.array(scores, dtype=float), nan=0.5)
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def compute_auc_pr(labels, scores):
    """Compute AUC-PR (average precision). labels: 1=hallucinated."""
    labels = np.array(labels, dtype=float)
    scores = np.nan_to_num(np.array(scores, dtype=float), nan=0.5)
    if len(set(labels)) < 2:
        return float("nan")
    return average_precision_score(labels, scores)


def compute_ece(labels, scores, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        mask = (scores >= bins[i]) & (scores < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = np.mean(np.array(labels)[mask])
        bin_conf = np.mean(np.array(scores)[mask])
        ece += mask.sum() / total * abs(bin_acc - bin_conf)
    return ece


def bootstrap_ci(labels, scores, metric_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for a metric."""
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)
    vals = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            v = metric_fn(labels[idx], scores[idx])
            if not np.isnan(v):
                vals.append(v)
        except:
            pass
    if len(vals) < 10:
        return float("nan"), float("nan"), float("nan")
    vals = sorted(vals)
    lo = vals[int(len(vals) * (1 - ci) / 2)]
    hi = vals[int(len(vals) * (1 + ci) / 2)]
    return np.mean(vals), lo, hi


def paired_bootstrap_test(labels, scores_a, scores_b, metric_fn, n_bootstrap=2000):
    """Paired bootstrap test: is scores_a significantly better than scores_b?"""
    labels = np.array(labels)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(labels)
    count_better = 0
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            va = metric_fn(labels[idx], scores_a[idx])
            vb = metric_fn(labels[idx], scores_b[idx])
            if va > vb:
                count_better += 1
        except:
            pass
    p_value = 1.0 - count_better / n_bootstrap
    return p_value


def compute_all_metrics(labels, scores):
    """Compute all detection metrics."""
    labels = np.array(labels, dtype=float)
    scores = np.array(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.5)
    auc_roc = compute_auc_roc(labels, scores)
    auc_pr = compute_auc_pr(labels, scores)
    ece = compute_ece(labels, scores)

    # Correlation
    if len(set(labels)) >= 2:
        pearson_r, pearson_p = stats.pearsonr(labels, scores)
        spearman_r, spearman_p = stats.spearmanr(labels, scores)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    # Best F1 threshold
    if len(set(labels)) >= 2:
        prec, rec, thresholds = precision_recall_curve(labels, scores)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s)
        best_f1 = f1s[best_idx]
        best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
    else:
        best_f1 = best_threshold = float("nan")

    return {
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "ece": round(float(ece), 4),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_r": round(float(spearman_r), 4),
        "best_f1": round(float(best_f1), 4),
        "best_threshold": round(float(best_threshold), 4),
    }
