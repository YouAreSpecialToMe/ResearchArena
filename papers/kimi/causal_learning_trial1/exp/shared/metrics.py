"""
Evaluation metrics for local causal discovery.
"""
import numpy as np


def compute_precision_recall_f1(predicted, true):
    """Compute precision, recall, and F1 score."""
    predicted_set = set(predicted)
    true_set = set(true)
    
    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_shd(predicted_edges, true_edges, nodes):
    """
    Compute Structural Hamming Distance for local structure.
    For local structure, we compare the skeleton (undirected edges).
    """
    # Convert to undirected edge sets
    def to_undirected(edges):
        undirected = set()
        for u, v in edges:
            if (v, u) not in undirected:
                undirected.add((u, v))
        return undirected
    
    pred_undirected = to_undirected(predicted_edges)
    true_undirected = to_undirected(true_edges)
    
    # SHD = number of edge additions + deletions needed
    missing = len(true_undirected - pred_undirected)
    extra = len(pred_undirected - true_undirected)
    
    return missing + extra


def evaluate_mb_discovery(learned_mb, true_mb):
    """Evaluate Markov blanket discovery."""
    return compute_precision_recall_f1(learned_mb, true_mb)


def evaluate_pc_discovery(learned_pc, true_pc):
    """Evaluate PC set discovery."""
    return compute_precision_recall_f1(learned_pc, true_pc)


def evaluate_parent_child(learned_parents, learned_children, true_parents, true_children):
    """Evaluate parent/child classification."""
    parent_metrics = compute_precision_recall_f1(learned_parents, true_parents)
    child_metrics = compute_precision_recall_f1(learned_children, true_children)
    
    # Combined F1 for orientation
    total_tp = parent_metrics['tp'] + child_metrics['tp']
    total_fp = parent_metrics['fp'] + child_metrics['fp']
    total_fn = parent_metrics['fn'] + child_metrics['fn']
    
    orientation_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    orientation_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    orientation_f1 = 2 * orientation_precision * orientation_recall / (orientation_precision + orientation_recall) if (orientation_precision + orientation_recall) > 0 else 0.0
    
    return {
        'parent_precision': parent_metrics['precision'],
        'parent_recall': parent_metrics['recall'],
        'parent_f1': parent_metrics['f1'],
        'child_precision': child_metrics['precision'],
        'child_recall': child_metrics['recall'],
        'child_f1': child_metrics['f1'],
        'orientation_f1': orientation_f1
    }


def aggregate_results(results_list):
    """Aggregate results across multiple runs (different seeds/targets)."""
    if not results_list:
        return {}
    
    # Get all metric keys
    keys = results_list[0].keys()
    
    aggregated = {}
    for key in keys:
        values = [r[key] for r in results_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return aggregated
