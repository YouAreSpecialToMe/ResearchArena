"""
Evaluation metrics for CleanBP experiments.
"""

import numpy as np
from typing import List, Dict, Tuple


def compute_expected_calibration_error(confidences: np.ndarray, 
                                       accuracies: np.ndarray, 
                                       n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Array of predicted confidence scores
        accuracies: Binary array indicating if prediction was correct (1) or not (0)
        n_bins: Number of bins for calibration
    
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Compute Brier score (proper scoring rule).
    
    Args:
        probabilities: Predicted probabilities
        outcomes: Binary outcomes (0 or 1)
    
    Returns:
        Brier score (lower is better)
    """
    return np.mean((probabilities - outcomes) ** 2)


def compute_repair_metrics(predicted_repairs: Dict, ground_truth: Dict) -> Dict:
    """
    Compute precision, recall, F1 for repairs.
    
    Args:
        predicted_repairs: Dict mapping (row, col) -> repaired_value
        ground_truth: Dict mapping (row, col) -> correct_value
    
    Returns:
        Dictionary with precision, recall, F1
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for key, pred_val in predicted_repairs.items():
        if key in ground_truth:
            if pred_val == ground_truth[key]:
                true_positives += 1
            else:
                false_positives += 1
    
    for key in ground_truth:
        if key not in predicted_repairs:
            false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        x: First sample
        y: Second sample
    
    Returns:
        Cohen's d
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0


def paired_t_test(before: np.ndarray, after: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test.
    
    Returns:
        t-statistic, p-value
    """
    from scipy import stats
    return stats.ttest_rel(before, after)
