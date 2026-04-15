"""
Evaluation metrics for protein fitness prediction.
"""
import numpy as np
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu


def spearman_rho(y_true, y_pred):
    """Compute Spearman correlation."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    rho, pval = spearmanr(y_true[mask], y_pred[mask])
    return rho


def ndcg_at_k(y_true, y_pred, k=100):
    """Compute NDCG@k for ranking evaluation."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < k:
        k = len(y_true)
    if k == 0:
        return np.nan

    # Sort by predicted scores (descending)
    pred_order = np.argsort(-y_pred)
    # DCG: use true fitness as relevance
    # Shift so minimum is 0
    y_shifted = y_true - y_true.min()

    dcg = 0.0
    for i in range(k):
        dcg += y_shifted[pred_order[i]] / np.log2(i + 2)

    # Ideal DCG
    ideal_order = np.argsort(-y_true)
    idcg = 0.0
    for i in range(k):
        idcg += y_shifted[ideal_order[i]] / np.log2(i + 2)

    if idcg == 0:
        return 1.0
    return dcg / idcg


def epistasis_spearman(y_true, y_pred_additive, y_pred_full):
    """
    Compute epistasis-specific Spearman.
    Measures how well the model captures the non-additive component.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred_additive) & np.isfinite(y_pred_full)
    if mask.sum() < 3:
        return np.nan
    residual_true = y_true[mask] - y_pred_additive[mask]
    residual_pred = y_pred_full[mask] - y_pred_additive[mask]
    if np.std(residual_true) < 1e-10 or np.std(residual_pred) < 1e-10:
        return np.nan
    rho, _ = spearmanr(residual_true, residual_pred)
    return rho


def evaluate_predictions(y_true, y_pred, y_pred_additive=None):
    """Compute all metrics for a set of predictions."""
    results = {
        'spearman_rho': spearman_rho(y_true, y_pred),
        'ndcg100': ndcg_at_k(y_true, y_pred, k=100),
        'ndcg10': ndcg_at_k(y_true, y_pred, k=10),
    }
    if y_pred_additive is not None:
        results['epistasis_spearman'] = epistasis_spearman(y_true, y_pred_additive, y_pred)
    return results


def paired_wilcoxon_test(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test."""
    mask = np.isfinite(scores_a) & np.isfinite(scores_b)
    a = scores_a[mask]
    b = scores_b[mask]
    if len(a) < 5:
        return np.nan, np.nan
    stat, pval = wilcoxon(a, b, alternative='two-sided')
    # Effect size: rank-biserial correlation
    n = len(a)
    r = 1 - (2 * stat) / (n * (n + 1))
    return pval, r


def mann_whitney_test(group1, group2, alternative='greater'):
    """Mann-Whitney U test for comparing two groups."""
    mask1 = np.isfinite(group1)
    mask2 = np.isfinite(group2)
    g1 = group1[mask1]
    g2 = group2[mask2]
    if len(g1) < 3 or len(g2) < 3:
        return np.nan, np.nan
    stat, pval = mannwhitneyu(g1, g2, alternative=alternative)
    return pval, stat
