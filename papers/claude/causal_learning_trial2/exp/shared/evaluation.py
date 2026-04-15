"""Evaluation metrics for causal discovery."""

import numpy as np


def compute_shd(pred, true):
    """Structural Hamming Distance between predicted and true DAGs.

    Counts: missing edges, extra edges, reversed edges.
    Handles both binary DAGs and CPDAGs (with 0.5 for undirected).
    """
    p = pred.shape[0]
    # Binarize predictions
    pred_bin = (pred > 0.1).astype(int)
    true_bin = (true > 0.1).astype(int)

    shd = 0
    for i in range(p):
        for j in range(i + 1, p):
            pred_ij = pred_bin[i, j]
            pred_ji = pred_bin[j, i]
            true_ij = true_bin[i, j]
            true_ji = true_bin[j, i]

            # Determine edge type
            pred_edge = (pred_ij, pred_ji)
            true_edge = (true_ij, true_ji)

            if pred_edge == true_edge:
                continue

            if true_edge == (0, 0):
                # Extra edge (any direction)
                if pred_ij or pred_ji:
                    shd += 1
            elif pred_edge == (0, 0):
                # Missing edge
                if true_ij or true_ji:
                    shd += 1
            else:
                # Both have edge but wrong direction
                if pred_edge != true_edge:
                    shd += 1

    return shd


def compute_f1_skeleton(pred, true):
    """F1 score for edge presence (ignoring direction).

    Returns: f1, precision, recall
    """
    p = pred.shape[0]
    pred_bin = (pred > 0.1).astype(int)
    true_bin = (true > 0.1).astype(int)

    # Skeleton: undirected adjacency
    pred_skel = np.maximum(pred_bin, pred_bin.T)
    true_skel = np.maximum(true_bin, true_bin.T)

    # Only upper triangle
    tp = 0
    fp = 0
    fn = 0
    for i in range(p):
        for j in range(i + 1, p):
            if pred_skel[i, j] and true_skel[i, j]:
                tp += 1
            elif pred_skel[i, j] and not true_skel[i, j]:
                fp += 1
            elif not pred_skel[i, j] and true_skel[i, j]:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1, precision, recall


def compute_orientation_accuracy(pred, true):
    """Among correctly identified edges, fraction with correct orientation."""
    p = pred.shape[0]
    pred_bin = (pred > 0.1).astype(int)
    true_bin = (true > 0.1).astype(int)

    correct_orient = 0
    total_correct_edges = 0

    for i in range(p):
        for j in range(i + 1, p):
            pred_has = pred_bin[i, j] or pred_bin[j, i]
            true_has = true_bin[i, j] or true_bin[j, i]

            if pred_has and true_has:
                total_correct_edges += 1
                # Check if direction matches
                if pred_bin[i, j] == true_bin[i, j] and pred_bin[j, i] == true_bin[j, i]:
                    correct_orient += 1

    return correct_orient / total_correct_edges if total_correct_edges > 0 else 0.0


def compute_calibration(confidence_matrix, true_dag, n_bins=5):
    """Compute expected calibration error for edge confidence scores.

    Args:
        confidence_matrix: p x p matrix of confidence scores
        true_dag: p x p binary ground truth

    Returns:
        ece: expected calibration error
        bin_data: list of (bin_center, observed_freq, count) for calibration plot
    """
    p = confidence_matrix.shape[0]
    true_bin = (true_dag > 0.1).astype(int)

    # Collect all off-diagonal confidence scores
    scores = []
    labels = []
    for i in range(p):
        for j in range(p):
            if i != j:
                scores.append(confidence_matrix[i, j])
                labels.append(true_bin[i, j])

    scores = np.array(scores)
    labels = np.array(labels)

    # Bin scores
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for b in range(n_bins):
        mask = (scores >= bin_edges[b]) & (scores < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = (scores >= bin_edges[b]) & (scores <= bin_edges[b + 1])
        count = mask.sum()
        if count > 0:
            avg_conf = scores[mask].mean()
            avg_acc = labels[mask].mean()
            ece += count * abs(avg_conf - avg_acc)
            bin_data.append((avg_conf, avg_acc, count))
        else:
            bin_data.append(((bin_edges[b] + bin_edges[b + 1]) / 2, 0, 0))

    ece /= len(scores) if len(scores) > 0 else 1

    return ece, bin_data


def compute_all_metrics(pred, true, confidence_matrix=None):
    """Compute all metrics."""
    shd = compute_shd(pred, true)
    f1, precision, recall = compute_f1_skeleton(pred, true)
    orient_acc = compute_orientation_accuracy(pred, true)

    result = {
        'SHD': shd,
        'F1': f1,
        'precision': precision,
        'recall': recall,
        'orientation_accuracy': orient_acc,
    }

    if confidence_matrix is not None:
        ece, bin_data = compute_calibration(confidence_matrix, true)
        result['ECE'] = ece
        result['calibration_bins'] = bin_data

    return result
