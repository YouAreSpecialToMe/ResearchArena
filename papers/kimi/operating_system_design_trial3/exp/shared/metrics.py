"""
Evaluation metrics for KAPHE experiments.
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats


def compute_performance_metrics(predicted_scores: np.ndarray, 
                                oracle_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute performance-based metrics.
    
    Args:
        predicted_scores: Performance scores with predicted configuration
        oracle_scores: Performance scores with optimal configuration
    
    Returns:
        Dictionary of metrics
    """
    # Normalized performance (fraction of oracle)
    normalized = predicted_scores / oracle_scores
    
    # Within 5% of optimal
    within_5pct = np.mean(normalized >= 0.95) * 100
    
    # Within 10% of optimal
    within_10pct = np.mean(normalized >= 0.90) * 100
    
    # Within 20% of optimal
    within_20pct = np.mean(normalized >= 0.80) * 100
    
    return {
        'mean_normalized_score': float(np.mean(normalized)),
        'std_normalized_score': float(np.std(normalized)),
        'within_5pct': float(within_5pct),
        'within_10pct': float(within_10pct),
        'within_20pct': float(within_20pct),
        'min_normalized_score': float(np.min(normalized)),
        'median_normalized_score': float(np.median(normalized)),
    }


def compute_rule_metrics(rules: List[Dict], test_predictions: List[int], 
                        test_oracle: List[int]) -> Dict[str, float]:
    """
    Compute rule-based interpretability metrics.
    
    Args:
        rules: List of extracted rules
        test_predictions: Predicted configuration IDs for test set
        test_oracle: Oracle configuration IDs for test set
    
    Returns:
        Dictionary of metrics
    """
    if not rules:
        return {
            'num_rules': 0,
            'avg_rule_length': 0,
            'max_rule_length': 0,
            'total_conditions': 0,
            'rule_complexity': 0,
            'coverage': 0,
            'fidelity': 0,
        }
    
    # Rule count
    num_rules = len(rules)
    
    # Rule length metrics
    rule_lengths = [len(r.get('conditions', [])) for r in rules]
    avg_rule_length = np.mean(rule_lengths)
    max_rule_length = np.max(rule_lengths)
    total_conditions = sum(rule_lengths)
    
    # Rule complexity (rules + avg conditions per rule)
    rule_complexity = num_rules + avg_rule_length
    
    # Coverage: % of test cases with a matching rule
    # (assuming prediction of -1 means no rule matched)
    coverage = np.mean(np.array(test_predictions) >= 0) * 100
    
    # Fidelity: % agreement with oracle (among covered cases)
    covered_mask = np.array(test_predictions) >= 0
    if np.any(covered_mask):
        fidelity = np.mean(np.array(test_predictions)[covered_mask] == 
                          np.array(test_oracle)[covered_mask]) * 100
    else:
        fidelity = 0
    
    # Overall accuracy (including default for uncovered)
    overall_accuracy = np.mean(np.array(test_predictions) == np.array(test_oracle)) * 100
    
    return {
        'num_rules': num_rules,
        'avg_rule_length': float(avg_rule_length),
        'max_rule_length': int(max_rule_length),
        'total_conditions': int(total_conditions),
        'rule_complexity': float(rule_complexity),
        'coverage': float(coverage),
        'fidelity': float(fidelity),
        'overall_accuracy': float(overall_accuracy),
    }


def compute_statistical_significance(method_scores: np.ndarray,
                                     baseline_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical significance tests.
    
    Args:
        method_scores: Performance scores for our method
        baseline_scores: Performance scores for baseline
    
    Returns:
        Dictionary with p-values and effect sizes
    """
    # Paired t-test
    t_stat, p_value_ttest = stats.ttest_rel(method_scores, baseline_scores)
    
    # Wilcoxon signed-rank test (non-parametric)
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(method_scores, baseline_scores)
    except ValueError:
        p_value_wilcoxon = 1.0
    
    # Effect size (Cohen's d)
    diff = method_scores - baseline_scores
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
    
    # Mean improvement
    mean_improvement = np.mean(diff)
    
    return {
        'p_value_ttest': float(p_value_ttest),
        'p_value_wilcoxon': float(p_value_wilcoxon),
        'cohens_d': float(cohens_d),
        'mean_improvement': float(mean_improvement),
        'is_significant': bool(p_value_ttest < 0.05),
    }


def compute_decision_tree_metrics(tree_model) -> Dict[str, Any]:
    """
    Compute interpretability metrics for a decision tree.
    
    Args:
        tree_model: Fitted sklearn DecisionTreeClassifier
    
    Returns:
        Dictionary of metrics
    """
    tree = tree_model.tree_
    
    # Tree structure metrics
    num_nodes = tree.node_count
    max_depth = tree_model.get_depth()
    num_leaves = tree_model.get_n_leaves()
    
    # Tree complexity (nodes + depth)
    tree_complexity = num_nodes + max_depth
    
    return {
        'num_nodes': num_nodes,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'tree_complexity': tree_complexity,
    }


def compute_overhead_metrics(profiling_times: List[float],
                             recommendation_times: List[float]) -> Dict[str, float]:
    """
    Compute overhead metrics.
    
    Args:
        profiling_times: List of profiling durations in seconds
        recommendation_times: List of recommendation lookup times in ms
    
    Returns:
        Dictionary of metrics
    """
    profiling_times = np.array(profiling_times)
    recommendation_times = np.array(recommendation_times)
    
    return {
        'profiling_time_mean_sec': float(np.mean(profiling_times)),
        'profiling_time_p99_sec': float(np.percentile(profiling_times, 99)),
        'rec_time_mean_ms': float(np.mean(recommendation_times)),
        'rec_time_p50_ms': float(np.median(recommendation_times)),
        'rec_time_p99_ms': float(np.percentile(recommendation_times, 99)),
    }
