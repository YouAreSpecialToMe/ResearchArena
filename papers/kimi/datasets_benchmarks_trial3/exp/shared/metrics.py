"""
Evaluation metrics for PopBench experiments.
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def compute_spearman_correlation(predicted: np.ndarray, true: np.ndarray) -> float:
    """Compute Spearman rank correlation."""
    if len(predicted) != len(true):
        raise ValueError("Arrays must have same length")
    if len(predicted) < 2:
        return 0.0
    rho, _ = stats.spearmanr(predicted, true)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_kendall_tau(predicted: np.ndarray, true: np.ndarray) -> float:
    """Compute Kendall's tau rank correlation."""
    if len(predicted) != len(true):
        raise ValueError("Arrays must have same length")
    if len(predicted) < 2:
        return 0.0
    tau, _ = stats.kendalltau(predicted, true)
    return float(tau) if not np.isnan(tau) else 0.0


def compute_mae(predicted: np.ndarray, true: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return float(np.mean(np.abs(predicted - true)))


def compute_rmse(predicted: np.ndarray, true: np.ndarray) -> float:
    """Compute Root Mean Square Error."""
    return float(np.sqrt(np.mean((predicted - true) ** 2)))


def compute_calibration_score(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    true_values: np.ndarray,
    confidence_levels: List[float] = None
) -> Dict[str, float]:
    """
    Compute calibration score - what % of true values fall within predicted credible intervals.
    
    Args:
        predicted_mean: Predicted means
        predicted_std: Predicted standard deviations
        true_values: True values
        confidence_levels: List of confidence levels to check (default: [0.5, 0.68, 0.9, 0.95])
    
    Returns:
        Dict mapping confidence level to actual coverage percentage
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.68, 0.9, 0.95]
    
    calibration = {}
    for conf in confidence_levels:
        # For normal distribution, conf interval is mean ± z * std
        z = stats.norm.ppf((1 + conf) / 2)
        lower = predicted_mean - z * predicted_std
        upper = predicted_mean + z * predicted_std
        
        coverage = np.mean((true_values >= lower) & (true_values <= upper))
        calibration[f"{int(conf*100)}%"] = float(coverage)
    
    return calibration


def compute_items_to_target(
    mae_history: List[float],
    target_mae: float = 0.05
) -> int:
    """
    Find the number of items needed to reach target MAE.
    
    Returns:
        Number of items, or -1 if target never reached
    """
    for i, mae in enumerate(mae_history):
        if mae <= target_mae:
            return i + 1
    return -1


def aggregate_metrics_across_seeds(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across random seeds.
    
    Returns:
        Dict mapping metric name to {mean, std, min, max}
    """
    if not metrics_list:
        return {}
    
    result = {}
    all_keys = metrics_list[0].keys()
    
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if len(values) > 0:
            result[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) if len(values) > 1 else 0.0,
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return result


def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                          (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_r2_score(predicted: np.ndarray, true: np.ndarray) -> float:
    """Compute R^2 coefficient of determination."""
    ss_res = np.sum((true - predicted) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def compute_nll(probs: np.ndarray, responses: np.ndarray) -> float:
    """Compute negative log-likelihood for binary responses."""
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)
    nll = -np.mean(responses * np.log(probs) + (1 - responses) * np.log(1 - probs))
    return float(nll)


def compute_irt_likelihood(
    theta: np.ndarray,  # (D,) ability vector
    a: np.ndarray,      # (N, D) discrimination parameters
    b: np.ndarray,      # (N,) difficulty parameters
    responses: np.ndarray  # (N,) binary responses
) -> float:
    """Compute IRT likelihood given ability and item parameters."""
    # 2PL MIRT: P(correct) = sigmoid(sum_d a_d * theta_d - b)
    logits = np.sum(a * theta[None, :], axis=1) - b
    probs = 1 / (1 + np.exp(-logits))
    return compute_nll(probs, responses)


class MetricsTracker:
    """Track metrics during adaptive evaluation."""
    
    def __init__(self):
        self.metrics_history = []
        self.items_used = []
    
    def record(self, n_items: int, metrics: Dict[str, float]):
        """Record metrics at current number of items."""
        self.items_used.append(n_items)
        self.metrics_history.append(metrics)
    
    def get_summary(self) -> Dict[str, List]:
        """Get summary of tracked metrics."""
        summary = {'items': self.items_used}
        
        if self.metrics_history:
            all_keys = self.metrics_history[0].keys()
            for key in all_keys:
                summary[key] = [m[key] for m in self.metrics_history if key in m]
        
        return summary
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final recorded metrics."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]
