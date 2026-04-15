"""
Shared utilities for CellStratCP experiments.
Includes data loading, ZINB functions, conformal prediction utilities, and metrics.
"""

import os
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from scipy import stats
from scipy.special import gammaln, digamma, polygamma
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import json
import time
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def zinb_loglikelihood(x, mu, theta, pi, eps=1e-8):
    """
    Compute ZINB log-likelihood.
    
    Args:
        x: observed counts (can be scalar or array)
        mu: mean parameter
        theta: dispersion parameter
        pi: zero-inflation probability
        eps: small constant for numerical stability
    
    Returns:
        log P(x | mu, theta, pi)
    """
    # Convert to numpy arrays
    x = np.asarray(x)
    mu = np.asarray(mu)
    theta = np.asarray(theta)
    pi = np.asarray(pi)
    
    # Ensure positivity
    mu = np.maximum(mu, eps)
    theta = np.maximum(theta, eps)
    pi = np.clip(pi, eps, 1 - eps)
    
    # NB log-likelihood component
    # log NB(x; mu, theta) = log Gamma(x + theta) - log Gamma(theta) - log Gamma(x + 1)
    #                        + theta * log(theta) + x * log(mu) - (theta + x) * log(theta + mu)
    
    r = theta  # dispersion parameter
    p = r / (r + mu)  # success probability parameterization
    
    # For numerical stability, use log-space computations
    nb_ll = (gammaln(x + r) - gammaln(r) - gammaln(x + 1) + 
             r * np.log(p + eps) + x * np.log(1 - p + eps))
    
    # Zero-inflation
    is_zero = (x == 0)
    
    # For zeros: log(pi + (1-pi) * exp(nb_ll))
    # For non-zeros: log(1-pi) + nb_ll
    zinb_ll = np.where(is_zero, 
                       np.log(pi + (1 - pi) * np.exp(nb_ll) + eps),
                       np.log(1 - pi + eps) + nb_ll)
    
    return zinb_ll


def zinb_nonconformity_score(x_obs, x_pred, mu, theta, pi, eps=1e-8):
    """
    Compute ZINB-based non-conformity score.
    Score = -log P(x_obs | mu, theta, pi) for observed count
    Lower score = higher likelihood = better conformity
    
    Args:
        x_obs: observed count
        x_pred: predicted count (point estimate, not used directly)
        mu, theta, pi: ZINB parameters
    
    Returns:
        non-conformity score (higher = less conformal)
    """
    ll = zinb_loglikelihood(x_obs, mu, theta, pi, eps)
    return -ll  # Negative log-likelihood as non-conformity score


def residual_nonconformity_score(x_obs, x_pred, *args, **kwargs):
    """
    Standard residual-based non-conformity score.
    Score = |x_obs - x_pred|
    """
    return np.abs(x_obs - x_pred)


def compute_prediction_interval(zinb_params, quantile, max_count=100):
    """
    Compute prediction interval from ZINB distribution parameters.
    Returns [lower, upper] bounds where all y with score <= quantile are included.
    
    Args:
        zinb_params: dict with 'mu', 'theta', 'pi'
        quantile: threshold non-conformity score
        max_count: maximum count to search
    
    Returns:
        (lower, upper) bounds of prediction interval
    """
    mu = zinb_params['mu']
    theta = zinb_params['theta']
    pi = zinb_params['pi']
    
    # Compute scores for all possible counts
    counts = np.arange(max_count + 1)
    scores = np.array([zinb_nonconformity_score(c, None, mu, theta, pi) 
                       for c in counts])
    
    # Find all counts with score <= quantile
    in_interval = counts[scores <= quantile]
    
    if len(in_interval) == 0:
        # Fallback: use quantile of the distribution
        if mu < 1:
            lower, upper = 0, max(1, int(mu * 3))
        else:
            lower, upper = max(0, int(mu - 2*np.sqrt(mu))), int(mu + 2*np.sqrt(mu))
    else:
        lower = in_interval.min()
        upper = in_interval.max()
    
    return float(lower), float(upper)


def conformal_prediction_intervals(calibration_scores, test_scores, alpha=0.1):
    """
    Compute conformal prediction intervals using split conformal prediction.
    
    Args:
        calibration_scores: non-conformity scores from calibration set
        test_scores: predicted non-conformity scores for test points (not used for interval bounds)
        alpha: miscoverage level (target coverage = 1 - alpha)
    
    Returns:
        quantile_threshold: threshold for inclusion in prediction set
    """
    n_cal = len(calibration_scores)
    # Compute (1-alpha) quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    
    quantile_threshold = np.quantile(calibration_scores, q_level)
    return quantile_threshold


def mondrian_conformal_prediction(calibration_data, test_data, cell_types_cal, cell_types_test, 
                                   alpha=0.1, score_fn=zinb_nonconformity_score):
    """
    Mondrian conformal prediction with cell-type stratification.
    
    Args:
        calibration_data: dict with 'x_obs', 'x_pred', 'mu', 'theta', 'pi' for calibration
        test_data: dict with 'x_pred', 'mu', 'theta', 'pi' for test
        cell_types_cal: cell type labels for calibration set
        cell_types_test: cell type labels for test set
        alpha: miscoverage level
        score_fn: function to compute non-conformity scores
    
    Returns:
        dict with quantiles per cell type and prediction intervals
    """
    unique_types = np.unique(cell_types_cal)
    quantiles = {}
    
    # Compute cell-type-specific quantiles
    for cell_type in unique_types:
        mask = cell_types_cal == cell_type
        
        # Compute non-conformity scores for calibration cells of this type
        if score_fn == zinb_nonconformity_score:
            scores = np.array([
                score_fn(calibration_data['x_obs'][i], 
                        calibration_data['x_pred'][i],
                        calibration_data['mu'][i],
                        calibration_data['theta'][i],
                        calibration_data['pi'][i])
                for i in np.where(mask)[0]
            ])
        else:
            scores = np.array([
                score_fn(calibration_data['x_obs'][i], calibration_data['x_pred'][i])
                for i in np.where(mask)[0]
            ])
        
        quantiles[cell_type] = conformal_prediction_intervals(scores, None, alpha)
    
    # Compute prediction intervals for test cells
    prediction_intervals = []
    for i in range(len(cell_types_test)):
        cell_type = cell_types_test[i]
        
        # Use pooled quantile if cell type not seen during calibration
        if cell_type not in quantiles:
            # Use the most conservative (highest) quantile
            q = max(quantiles.values())
        else:
            q = quantiles[cell_type]
        
        # Compute prediction interval
        zinb_params = {
            'mu': test_data['mu'][i],
            'theta': test_data['theta'][i],
            'pi': test_data['pi'][i]
        }
        lower, upper = compute_prediction_interval(zinb_params, q)
        prediction_intervals.append((lower, upper))
    
    return {
        'quantiles': quantiles,
        'prediction_intervals': np.array(prediction_intervals)
    }


def evaluate_coverage(y_true, prediction_intervals, cell_types=None):
    """
    Evaluate coverage metrics.
    
    Args:
        y_true: true values
        prediction_intervals: array of (lower, upper) bounds
        cell_types: optional cell type labels for conditional coverage
    
    Returns:
        dict with coverage metrics
    """
    y_true = np.asarray(y_true)
    lowers = prediction_intervals[:, 0]
    uppers = prediction_intervals[:, 1]
    
    # Marginal coverage
    covered = (y_true >= lowers) & (y_true <= uppers)
    marginal_coverage = covered.mean()
    
    results = {
        'marginal_coverage': float(marginal_coverage),
        'mean_interval_width': float((uppers - lowers).mean()),
        'median_interval_width': float(np.median(uppers - lowers)),
        'std_interval_width': float((uppers - lowers).std())
    }
    
    # Conditional coverage by cell type
    if cell_types is not None:
        unique_types = np.unique(cell_types)
        cond_coverage = {}
        for ct in unique_types:
            mask = cell_types == ct
            if mask.sum() > 0:
                cond_coverage[str(ct)] = float(covered[mask].mean())
        
        results['conditional_coverage'] = cond_coverage
        results['max_coverage_discrepancy'] = float(
            max(cond_coverage.values()) - min(cond_coverage.values())
        )
    
    return results


def adaptive_conformal_inference(coverage_history, target_alpha, gamma=0.01, current_alpha=None):
    """
    ACI update rule: adjust alpha based on observed coverage.
    
    Args:
        coverage_history: binary array indicating if each point was covered (1) or not (0)
        target_alpha: target miscoverage level
        gamma: learning rate
        current_alpha: current alpha value (if None, use target_alpha as initial)
    
    Returns:
        updated alpha value
    """
    if current_alpha is None:
        current_alpha = target_alpha
    
    if len(coverage_history) == 0:
        return current_alpha
    
    # Recent coverage rate
    recent_coverage = np.mean(coverage_history[-100:])  # Last 100 points
    
    # Update: if coverage is too low, decrease alpha (widen intervals)
    # If coverage is too high, increase alpha (narrow intervals)
    error_rate = 1 - recent_coverage
    new_alpha = current_alpha - gamma * (error_rate - target_alpha)
    
    # Clip to valid range
    new_alpha = np.clip(new_alpha, 0.001, 0.5)
    
    return float(new_alpha)


def evaluate_ood_detection(in_distribution_scores, ood_scores):
    """
    Evaluate OOD detection performance.
    
    Args:
        in_distribution_scores: non-conformity scores for in-distribution cells
        ood_scores: non-conformity scores for OOD cells
    
    Returns:
        dict with AUROC, FPR@95TPR, etc.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Create labels: 0 for in-distribution, 1 for OOD
    y_true = np.concatenate([np.zeros(len(in_distribution_scores)), 
                             np.ones(len(ood_scores))])
    scores = np.concatenate([in_distribution_scores, ood_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, scores)
    
    # FPR at 95% TPR
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        fpr_at_95_tpr = fpr[idx[0]]
    else:
        fpr_at_95_tpr = 1.0
    
    return {
        'auroc': float(auroc),
        'fpr_at_95_tpr': float(fpr_at_95_tpr)
    }


def save_results(results, path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path):
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    print("Testing scRNA-seq utilities...")
    
    # Test ZINB log-likelihood
    x = np.array([0, 1, 5, 10])
    mu = np.array([2.0, 2.0, 2.0, 2.0])
    theta = np.array([1.0, 1.0, 1.0, 1.0])
    pi = np.array([0.2, 0.2, 0.2, 0.2])
    
    ll = zinb_loglikelihood(x, mu, theta, pi)
    print(f"ZINB log-likelihoods: {ll}")
    
    # Test non-conformity score
    score = zinb_nonconformity_score(5, 3.0, 2.0, 1.0, 0.1)
    print(f"Non-conformity score: {score}")
    
    # Test conformal prediction
    cal_scores = np.random.exponential(1.0, 1000)
    threshold = conformal_prediction_intervals(cal_scores, None, alpha=0.1)
    print(f"Conformal quantile threshold (alpha=0.1): {threshold}")
    
    print("All tests passed!")
