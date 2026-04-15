"""
Evaluation metrics for causal discovery.
"""
import numpy as np
import networkx as nx
from typing import Dict, Tuple


def adjacency_to_skeleton(adj: np.ndarray) -> np.ndarray:
    """Convert directed adjacency to undirected skeleton."""
    skeleton = np.zeros_like(adj)
    skeleton[np.where((adj + adj.T) > 0)] = 1
    return skeleton


def compute_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute Structural Hamming Distance between two adjacency matrices.
    
    SHD counts the number of edge additions, deletions, or reversals needed
    to transform one graph into another.
    
    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        
    Returns:
        Structural Hamming Distance
    """
    # For directed graphs: count differences
    diff = np.abs(true_adj - pred_adj)
    # Reversals count as 1 (they show up as 1 in both directions after OR)
    # but we counted them as 2 in diff, so we need to adjust
    revs = np.logical_and(true_adj > 0, pred_adj.T > 0).sum()
    return int(diff.sum() - revs)


def compute_skeleton_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute SHD on the skeleton (undirected edges only).
    
    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        
    Returns:
        Skeleton SHD
    """
    true_skeleton = adjacency_to_skeleton(true_adj)
    pred_skeleton = adjacency_to_skeleton(pred_adj)
    return int(np.abs(true_skeleton - pred_skeleton).sum())


def compute_f1(true_adj: np.ndarray, pred_adj: np.ndarray, skeleton_only: bool = False) -> Dict[str, float]:
    """
    Compute F1 score for edge prediction.
    
    Args:
        true_adj: Ground truth adjacency matrix
        pred_adj: Predicted adjacency matrix
        skeleton_only: If True, evaluate on skeleton (ignore edge directions)
        
    Returns:
        Dictionary with precision, recall, F1
    """
    if skeleton_only:
        true_edges = adjacency_to_skeleton(true_adj)
        pred_edges = adjacency_to_skeleton(pred_adj)
    else:
        true_edges = true_adj
        pred_edges = pred_adj
    
    # Flatten
    true_flat = true_edges.flatten()
    pred_flat = pred_edges.flatten()
    
    # True positives, false positives, false negatives
    tp = np.sum((true_flat == 1) & (pred_flat == 1))
    fp = np.sum((true_flat == 0) & (pred_flat == 1))
    fn = np.sum((true_flat == 1) & (pred_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def compute_metrics(pred_adj: np.ndarray, true_adj: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred_adj: Predicted adjacency matrix
        true_adj: Ground truth adjacency matrix
        
    Returns:
        Dictionary with all metrics
    """
    # Ensure binary matrices
    pred_adj = (pred_adj > 0).astype(int)
    true_adj = (true_adj > 0).astype(int)
    
    # Directed metrics
    directed_f1 = compute_f1(true_adj, pred_adj, skeleton_only=False)
    
    # Skeleton metrics
    skeleton_f1 = compute_f1(true_adj, pred_adj, skeleton_only=True)
    
    # SHD
    shd = compute_shd(true_adj, pred_adj)
    skeleton_shd = compute_skeleton_shd(true_adj, pred_adj)
    
    return {
        'shd': shd,
        'skeleton_shd': skeleton_shd,
        'f1': directed_f1['f1'],
        'precision': directed_f1['precision'],
        'recall': directed_f1['recall'],
        'skeleton_f1': skeleton_f1['f1'],
        'skeleton_precision': skeleton_f1['precision'],
        'skeleton_recall': skeleton_f1['recall'],
        'tp': directed_f1['tp'],
        'fp': directed_f1['fp'],
        'fn': directed_f1['fn']
    }


def dag_to_adjacency(G: nx.DiGraph) -> np.ndarray:
    """Convert networkx DiGraph to adjacency matrix."""
    n = G.number_of_nodes()
    adj = np.zeros((n, n))
    if n == 0:
        return adj
    
    # Ensure nodes are 0-indexed
    node_list = sorted(G.nodes())
    node_idx = {node: i for i, node in enumerate(node_list)}
    
    for u, v in G.edges():
        adj[node_idx[u], node_idx[v]] = 1
    
    return adj


def cpdag_to_adjacency(cpdag: nx.DiGraph) -> np.ndarray:
    """Convert CPDAG to adjacency matrix (keeps both directions for undirected edges)."""
    return dag_to_adjacency(cpdag)


def summarize_metrics(metrics_list: list) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std of metrics across multiple runs.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not metrics_list:
        return {}
    
    keys = metrics_list[0].keys()
    summary = {}
    
    for key in keys:
        values = [m[key] for m in metrics_list]
        summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return summary


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # Create simple test graphs
    true_adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ])
    
    pred_adj = np.array([
        [0, 1, 0],
        [0, 0, 0],  # Missing edge
        [0, 0, 0]
    ])
    
    metrics = compute_metrics(true_adj, pred_adj)
    print(f"Metrics: {metrics}")
