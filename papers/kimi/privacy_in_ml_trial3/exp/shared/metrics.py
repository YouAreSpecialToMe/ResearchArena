"""
Evaluation metrics for LGSA experiments.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score
from scipy.stats import norm


def compute_auc_roc(scores, labels):
    """
    Compute AUC-ROC score.
    
    Args:
        scores: Higher score indicates more likely to be positive (forget)
        labels: 1 for forget samples, 0 for retain samples
    """
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5


def compute_tpr_at_fpr(scores, labels, target_fpr=0.01):
    """
    Compute TPR at a specific FPR.
    
    Args:
        scores: Higher score indicates more likely to be positive
        labels: 1 for forget samples, 0 for retain samples
        target_fpr: Target false positive rate
    """
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = np.where(fpr <= target_fpr)[0]
        if len(idx) > 0:
            return tpr[idx[-1]]
        return 0.0
    except:
        return 0.0


def compute_precision_recall(scores, labels, threshold=None):
    """
    Compute precision and recall.
    
    Args:
        scores: Higher score indicates more likely to be positive
        labels: 1 for forget samples, 0 for retain samples
        threshold: Classification threshold (if None, use 0.5)
    """
    if threshold is None:
        threshold = 0.5
    
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0)
    
    return precision, recall, f1


def compute_accuracy(model, dataloader, device='cuda'):
    """Compute model accuracy on dataloader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return correct / total if total > 0 else 0.0


def lira_attack_score(in_scores, out_scores, target_scores):
    """
    Compute LiRA (Membership Inference) scores.
    
    Args:
        in_scores: Scores from models trained with target sample
        out_scores: Scores from models trained without target sample
        target_scores: Scores from target model
    """
    # Fit Gaussian distributions
    in_mean, in_std = np.mean(in_scores), np.std(in_scores) + 1e-8
    out_mean, out_std = np.mean(out_scores), np.std(out_scores) + 1e-8
    
    # Compute likelihood ratio
    in_likelihood = norm.logpdf(target_scores, in_mean, in_std)
    out_likelihood = norm.logpdf(target_scores, out_mean, out_std)
    
    score = in_likelihood - out_likelihood
    return score


def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for data."""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    from scipy.stats import t
    h = std * t.ppf((1 + confidence) / 2, n - 1) / np.sqrt(n)
    
    return mean, mean - h, mean + h


import torch


def compute_loss(model, data, target, criterion=None, device='cuda'):
    """Compute loss for a single sample or batch."""
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    model.eval()
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
    
    return loss
