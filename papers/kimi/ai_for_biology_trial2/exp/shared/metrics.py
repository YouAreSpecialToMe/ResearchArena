"""Evaluation metrics for GRN inference."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import pearsonr, spearmanr
import torch


def compute_auroc(y_true, y_score):
    """Compute AUROC."""
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def compute_auprc(y_true, y_score):
    """Compute AUPRC."""
    try:
        return average_precision_score(y_true, y_score)
    except:
        return 0.0


def compute_signed_auroc(y_true, y_score, y_sign_true, y_sign_pred):
    """Compute signed AUROC where correct prediction requires correct edge AND sign."""
    # Weight positive examples by sign correctness
    sign_correct = (np.sign(y_sign_true) == np.sign(y_sign_pred)).astype(float)
    weighted_y_true = y_true * sign_correct
    
    try:
        return roc_auc_score(weighted_y_true, y_score)
    except:
        return 0.5


def compute_sign_accuracy(y_sign_true, y_sign_pred, threshold=0.5):
    """Compute accuracy of sign prediction."""
    pred_sign = np.sign(y_sign_pred)
    true_sign = np.sign(y_sign_true)
    correct = (pred_sign == true_sign).astype(float)
    
    # Only consider edges above confidence threshold
    confident = np.abs(y_sign_pred) >= threshold
    if confident.sum() > 0:
        return correct[confident].mean()
    return correct.mean()


def compute_epr(y_true, y_score, k=100):
    """Compute Early Precision Rate at top-k predictions."""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    
    # Precision at k
    precision_k = y_true_sorted[:k].mean()
    
    # Random baseline precision
    baseline = y_true.mean()
    
    if baseline > 0:
        return precision_k / baseline
    return 0.0


def compute_pearson_r(y_true, y_pred):
    """Compute Pearson correlation."""
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return r


def compute_r2(y_true, y_pred):
    """Compute R-squared."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot > 0:
        return 1 - ss_res / ss_tot
    return 0.0


def evaluate_grn_predictions(edges, ground_truth, chipseq_labels=None):
    """Evaluate GRN predictions against ground truth."""
    results = {}
    
    # Convert edges to arrays
    y_score = np.array([e['prob'] for e in edges])
    y_sign_pred = np.array([e.get('sign', 0) for e in edges])
    
    # Match with ground truth
    if ground_truth is not None:
        y_true = np.array([ground_truth.get((e['tf_idx'], e['target_idx']), 0) for e in edges])
        
        results['auroc'] = compute_auroc(y_true, y_score)
        results['auprc'] = compute_auprc(y_true, y_score)
        results['epr_100'] = compute_epr(y_true, y_score, k=100)
        
        # Signed metrics
        if 'sign' in edges[0]:
            y_sign_true = np.array([ground_truth.get(f"{(e['tf_idx'], e['target_idx'])}_sign", 1) for e in edges])
            results['signed_auroc'] = compute_signed_auroc(y_true, y_score, y_sign_true, y_sign_pred)
            results['sign_accuracy'] = compute_sign_accuracy(y_sign_true, y_sign_pred)
    
    # ChIP-seq evaluation
    if chipseq_labels is not None:
        chipseq_aurocs = []
        for tf_idx in chipseq_labels:
            if tf_idx < len(y_score):
                tf_edges = [e for e in edges if e['tf_idx'] == tf_idx]
                if len(tf_edges) > 0:
                    tf_scores = np.array([e['prob'] for e in tf_edges])
                    tf_labels = np.array([chipseq_labels[tf_idx].get(e['target_idx'], 0) for e in tf_edges])
                    if tf_labels.sum() > 0:
                        chipseq_aurocs.append(compute_auroc(tf_labels, tf_scores))
        
        if chipseq_aurocs:
            results['chipseq_auroc'] = np.mean(chipseq_aurocs)
    
    return results


def evaluate_expression_prediction(y_true, y_pred):
    """Evaluate expression prediction."""
    return {
        'pearson_r': compute_pearson_r(y_true.flatten(), y_pred.flatten()),
        'r2': compute_r2(y_true.flatten(), y_pred.flatten()),
        'mse': np.mean((y_true - y_pred) ** 2)
    }
