"""
Evaluation metrics for LayoutLearner experiments.
"""
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from typing import Dict, Tuple


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5  # Default for single class
    
    return metrics


def compute_metrics_with_ci(metrics_list: list, confidence: float = 0.95) -> Dict[str, Dict]:
    """Compute mean and confidence interval across multiple runs."""
    import scipy.stats as stats
    
    result = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        mean = np.mean(values)
        std = np.std(values)
        
        # 95% CI using t-distribution
        n = len(values)
        if n > 1:
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=stats.sem(values))
        else:
            ci = (mean, mean)
        
        result[key] = {
            'mean': mean,
            'std': std,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    return result


def statistical_test(baseline_scores: np.ndarray, method_scores: np.ndarray) -> Dict:
    """Perform statistical significance testing."""
    from scipy import stats
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(method_scores, baseline_scores)
    
    # Cohen's d effect size
    diff_mean = np.mean(method_scores - baseline_scores)
    pooled_std = np.sqrt((np.std(method_scores)**2 + np.std(baseline_scores)**2) / 2)
    cohens_d = diff_mean / pooled_std if pooled_std > 0 else 0
    
    # 95% CI for difference
    n = len(method_scores)
    se = np.std(method_scores - baseline_scores) / np.sqrt(n)
    ci = stats.t.interval(0.95, n-1, loc=diff_mean, scale=se)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': diff_mean,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'significant': p_value < 0.05
    }


def compute_performance_drop(full_metrics: Dict, ablation_metrics: Dict) -> float:
    """Compute percentage performance drop for ablation studies."""
    full_f1 = full_metrics.get('f1_score', {}).get('mean', full_metrics.get('f1_score', 0))
    ablation_f1 = ablation_metrics.get('f1_score', {}).get('mean', ablation_metrics.get('f1_score', 0))
    
    if full_f1 > 0:
        return (full_f1 - ablation_f1) / full_f1 * 100
    return 0


def format_metrics_table(metrics: Dict) -> str:
    """Format metrics as a readable table."""
    lines = ["Metric          Mean    Std     95% CI"]
    lines.append("-" * 50)
    
    for metric, values in metrics.items():
        if isinstance(values, dict):
            mean = values.get('mean', 0)
            std = values.get('std', 0)
            ci_low = values.get('ci_lower', 0)
            ci_high = values.get('ci_upper', 0)
            lines.append(f"{metric:15} {mean:.3f} ± {std:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
        else:
            lines.append(f"{metric:15} {values:.3f}")
    
    return "\n".join(lines)
