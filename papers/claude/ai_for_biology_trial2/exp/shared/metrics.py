"""Evaluation metrics for fitness prediction."""
import numpy as np
from scipy.stats import spearmanr, pearsonr


def compute_metrics(y_true, y_pred, additive_scores=None, epistasis_true=None,
                    num_mutations=None):
    """Compute comprehensive evaluation metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {'spearman': 0.0, 'rmse': float('inf'), 'n': 0}

    spearman = spearmanr(y_true, y_pred).statistic
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    results = {
        'spearman': float(spearman) if not np.isnan(spearman) else 0.0,
        'rmse': float(rmse),
        'n': int(len(y_true)),
    }

    # Epistasis correlation
    if additive_scores is not None and epistasis_true is not None:
        additive_scores = np.array(additive_scores)[mask]
        epistasis_true = np.array(epistasis_true)[mask]
        epistasis_pred = y_pred - additive_scores
        valid = ~(np.isnan(epistasis_true) | np.isnan(epistasis_pred))
        if valid.sum() >= 3:
            epi_corr = spearmanr(epistasis_true[valid], epistasis_pred[valid]).statistic
            results['epistasis_corr'] = float(epi_corr) if not np.isnan(epi_corr) else 0.0

    # Subset analysis by mutation count
    if num_mutations is not None:
        num_mutations = np.array(num_mutations)[mask]
        doubles = num_mutations == 2
        triples = num_mutations >= 3

        if doubles.sum() >= 3:
            rho = spearmanr(y_true[doubles], y_pred[doubles]).statistic
            results['spearman_doubles'] = float(rho) if not np.isnan(rho) else 0.0

        if triples.sum() >= 3:
            rho = spearmanr(y_true[triples], y_pred[triples]).statistic
            results['spearman_triples'] = float(rho) if not np.isnan(rho) else 0.0

        # High vs low epistasis
        if epistasis_true is not None:
            abs_epi = np.abs(epistasis_true)
            med = np.median(abs_epi[~np.isnan(abs_epi)])
            high_epi = abs_epi > med
            low_epi = abs_epi <= med

            if high_epi.sum() >= 3:
                rho = spearmanr(y_true[high_epi], y_pred[high_epi]).statistic
                results['spearman_high_epi'] = float(rho) if not np.isnan(rho) else 0.0
            if low_epi.sum() >= 3:
                rho = spearmanr(y_true[low_epi], y_pred[low_epi]).statistic
                results['spearman_low_epi'] = float(rho) if not np.isnan(rho) else 0.0

    return results
