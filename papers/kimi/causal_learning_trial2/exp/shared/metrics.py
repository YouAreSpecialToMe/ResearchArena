"""
Metrics for evaluating causal discovery algorithms.
"""
import numpy as np
from typing import Dict


def compute_shd(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """Compute Structural Hamming Distance between two adjacency matrices."""
    # Ensure binary matrices
    true_adj = (true_adj != 0).astype(int)
    pred_adj = (pred_adj != 0).astype(int)
    
    # SHD = number of edge additions, deletions, or reversals needed
    diff = true_adj - pred_adj
    
    # Count edge differences
    # For SHD, reversed edges count as 1 (not 2)
    shd = 0
    n = true_adj.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if true_adj[i, j] != pred_adj[i, j]:
                    # Check if it's a reversal (i->j in true, j->i in pred)
                    if true_adj[i, j] == 1 and pred_adj[j, i] == 1:
                        # This will be counted when we process (j, i)
                        pass
                    elif true_adj[j, i] == 1 and pred_adj[i, j] == 1:
                        # Reversal - count as 1
                        shd += 1
                    else:
                        # Addition or deletion
                        shd += 0.5  # Will be counted twice
    
    return int(shd)


def compute_tpr_fdr(true_adj: np.ndarray, pred_adj: np.ndarray) -> Dict[str, float]:
    """Compute True Positive Rate and False Discovery Rate."""
    true_adj = (true_adj != 0).astype(int)
    pred_adj = (pred_adj != 0).astype(int)
    
    n = true_adj.shape[0]
    
    # Count true positives, false positives, etc.
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if true_adj[i, j] == 1 and pred_adj[i, j] == 1:
                    tp += 1
                elif true_adj[i, j] == 0 and pred_adj[i, j] == 1:
                    fp += 1
                elif true_adj[i, j] == 0 and pred_adj[i, j] == 0:
                    tn += 1
                elif true_adj[i, j] == 1 and pred_adj[i, j] == 0:
                    fn += 1
    
    # TPR = TP / (TP + FN)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # FDR = FP / (FP + TP)
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tpr': tpr,
        'fdr': fdr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_sid(true_adj: np.ndarray, pred_adj: np.ndarray) -> int:
    """
    Compute Structural Intervention Distance (approximation).
    SID counts the number of intervention distributions that differ.
    We use a simplified version based on parent set differences.
    """
    true_adj = (true_adj != 0).astype(int)
    pred_adj = (pred_adj != 0).astype(int)
    
    n = true_adj.shape[0]
    sid = 0
    
    for j in range(n):
        true_parents = set(np.where(true_adj[:, j] == 1)[0])
        pred_parents = set(np.where(pred_adj[:, j] == 1)[0])
        
        # Count differences in parent sets
        sid += len(true_parents.symmetric_difference(pred_parents))
    
    return sid


def compute_all_metrics(true_adj: np.ndarray, pred_adj: np.ndarray) -> Dict:
    """Compute all metrics at once."""
    shd = compute_shd(true_adj, pred_adj)
    tpr_fdr = compute_tpr_fdr(true_adj, pred_adj)
    sid = compute_sid(true_adj, pred_adj)
    
    return {
        'shd': shd,
        'sid': sid,
        **tpr_fdr
    }
