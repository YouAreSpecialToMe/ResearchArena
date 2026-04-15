"""
Evaluation metrics for DU-VPT experiments.
Includes accuracy, ECE, mCE, forgetting score, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[float]:
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def expected_calibration_error(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        outputs: Model logits [N, C]
        targets: True labels [N]
        n_bins: Number of bins for calibration
    
    Returns:
        ECE score
    """
    with torch.no_grad():
        softmax = F.softmax(outputs, dim=1)
        confidences, predictions = softmax.max(dim=1)
        accuracies = predictions.eq(targets).float()
        
        confidences = confidences.cpu().numpy()
        accuracies = accuracies.cpu().numpy()
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)


def mean_corruption_error(
    accuracies: Dict[str, float],
    baseline_accuracies: Dict[str, float] = None
) -> float:
    """
    Compute mean Corruption Error (mCE).
    
    Args:
        accuracies: Dict mapping corruption type to accuracy
        baseline_accuracies: Dict mapping corruption type to baseline accuracy
    
    Returns:
        mCE score (lower is better)
    """
    if baseline_accuracies is None:
        # Use AlexNet baseline (simplified - in practice would use actual AlexNet errors)
        baseline_accuracies = {k: 0.5 for k in accuracies.keys()}
    
    ce_scores = []
    for corruption, acc in accuracies.items():
        if corruption in baseline_accuracies:
            baseline_acc = baseline_accuracies[corruption]
            # CE = (1 - accuracy) / (1 - baseline_accuracy)
            ce = (100 - acc) / (100 - baseline_acc + 1e-8)
            ce_scores.append(ce)
    
    return np.mean(ce_scores) * 100 if ce_scores else 0.0


def forgetting_score(
    source_accuracy_before: float,
    source_accuracy_after: float
) -> float:
    """
    Compute catastrophic forgetting score.
    
    Args:
        source_accuracy_before: Accuracy on source domain before adaptation
        source_accuracy_after: Accuracy on source domain after adaptation
    
    Returns:
        Forgetting score (positive = forgetting occurred)
    """
    return source_accuracy_before - source_accuracy_after


def compute_all_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    return_per_class: bool = False
) -> Dict[str, float]:
    """
    Compute all standard metrics.
    
    Args:
        outputs: Model logits [N, C]
        targets: True labels [N]
        return_per_class: Whether to return per-class accuracy
    
    Returns:
        Dict of metrics
    """
    top1_acc, top5_acc = accuracy(outputs, targets, topk=(1, 5))
    ece = expected_calibration_error(outputs, targets)
    
    metrics = {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'ece': ece,
    }
    
    if return_per_class:
        # Per-class accuracy
        num_classes = outputs.shape[1]
        predictions = outputs.argmax(dim=1)
        
        per_class_acc = {}
        for c in range(num_classes):
            class_mask = (targets == c)
            if class_mask.sum() > 0:
                class_correct = (predictions[class_mask] == c).float().mean().item()
                per_class_acc[f'class_{c}'] = class_correct * 100
        
        metrics['per_class_acc'] = per_class_acc
    
    return metrics


def aggregate_metrics(results_list: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple runs/seeds.
    
    Args:
        results_list: List of result dicts from multiple runs
    
    Returns:
        Dict with mean and std for each metric
    """
    if not results_list:
        return {}
    
    # Collect all metric names
    metric_names = set()
    for result in results_list:
        metric_names.update(result.keys())
    
    aggregated = {}
    for metric_name in metric_names:
        values = [r[metric_name] for r in results_list if metric_name in r]
        
        if values and isinstance(values[0], (int, float)):
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return aggregated


def shift_diagnosis_accuracy(
    predicted_shifts: List[str],
    actual_shifts: List[str]
) -> float:
    """
    Compute accuracy of shift type diagnosis.
    
    Args:
        predicted_shifts: List of predicted shift types
        actual_shifts: List of actual shift types
    
    Returns:
        Diagnosis accuracy
    """
    correct = sum(p == a for p, a in zip(predicted_shifts, actual_shifts))
    return 100.0 * correct / len(predicted_shifts) if predicted_shifts else 0.0


def compute_confidence_statistics(outputs: torch.Tensor) -> Dict[str, float]:
    """Compute statistics about model confidence."""
    with torch.no_grad():
        probs = F.softmax(outputs, dim=1)
        max_probs, _ = probs.max(dim=1)
        
        return {
            'mean_confidence': max_probs.mean().item(),
            'std_confidence': max_probs.std().item(),
            'min_confidence': max_probs.min().item(),
            'max_confidence': max_probs.max().item(),
        }
