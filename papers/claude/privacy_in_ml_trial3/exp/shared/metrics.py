"""Evaluation metrics including fairness metrics."""

import numpy as np
import torch
from collections import defaultdict


def evaluate_model(model, data_loader, device, num_subgroups=None):
    """Evaluate model and return per-subgroup metrics.

    Returns dict with:
        overall_accuracy, per_subgroup_accuracy, worst_group_accuracy,
        best_group_accuracy, accuracy_gap, per_sample_predictions, per_sample_losses
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    all_preds = []
    all_labels = []
    all_subgroups = []
    all_losses = []
    all_correct = []

    with torch.no_grad():
        for batch in data_loader:
            images, labels, subgroups = batch
            images = images.to(device)
            labels_t = torch.tensor(labels, dtype=torch.long).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            losses = criterion(outputs, labels_t)
            correct = (preds == labels_t)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_t.cpu().numpy())
            if isinstance(subgroups, torch.Tensor):
                all_subgroups.extend(subgroups.numpy())
            else:
                all_subgroups.extend(np.array(subgroups))
            all_losses.extend(losses.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_subgroups = np.array(all_subgroups)
    all_losses = np.array(all_losses)
    all_correct = np.array(all_correct)

    overall_acc = all_correct.mean()

    # Per-subgroup accuracy
    unique_subgroups = sorted(np.unique(all_subgroups))
    per_subgroup_acc = {}
    per_subgroup_count = {}
    for sg in unique_subgroups:
        mask = all_subgroups == sg
        per_subgroup_acc[int(sg)] = float(all_correct[mask].mean())
        per_subgroup_count[int(sg)] = int(mask.sum())

    accs = list(per_subgroup_acc.values())
    worst_group_acc = min(accs)
    best_group_acc = max(accs)
    accuracy_gap = best_group_acc - worst_group_acc

    # Equalized odds difference (simplified: difference in TPR across subgroups)
    eo_diff = compute_equalized_odds_diff(all_preds, all_labels, all_subgroups, unique_subgroups)

    # Demographic parity difference
    dp_diff = compute_demographic_parity_diff(all_preds, all_subgroups, unique_subgroups)

    return {
        "overall_accuracy": float(overall_acc),
        "per_subgroup_accuracy": per_subgroup_acc,
        "per_subgroup_count": per_subgroup_count,
        "worst_group_accuracy": float(worst_group_acc),
        "best_group_accuracy": float(best_group_acc),
        "accuracy_gap": float(accuracy_gap),
        "equalized_odds_diff": float(eo_diff),
        "demographic_parity_diff": float(dp_diff),
        "per_sample_losses": all_losses.tolist(),
        "per_sample_labels": all_labels.tolist(),
        "per_sample_subgroups": all_subgroups.tolist(),
        "per_sample_correct": all_correct.tolist(),
    }


def compute_equalized_odds_diff(preds, labels, subgroups, unique_subgroups):
    """Max difference in per-class accuracy across subgroups.

    For binary classification: difference in TPR across subgroups.
    For multi-class: average across classes of the max subgroup disparity
    in per-class recall, which generalizes equalized odds.
    """
    num_classes = len(np.unique(labels))

    if num_classes <= 2:
        # Binary: max difference in TPR (class=1) across subgroups
        tprs = []
        for sg in unique_subgroups:
            mask = subgroups == sg
            pos_mask = mask & (labels == 1)
            if pos_mask.sum() > 0:
                tpr = (preds[pos_mask] == 1).mean()
                tprs.append(float(tpr))
        if len(tprs) < 2:
            return 0.0
        return max(tprs) - min(tprs)
    else:
        # Multi-class: first try per-class recall disparity across subgroups
        per_class_disparities = []
        for c in range(num_classes):
            class_mask = labels == c
            if class_mask.sum() < 5:
                continue
            recalls = []
            for sg in unique_subgroups:
                sg_class_mask = (subgroups == sg) & class_mask
                if sg_class_mask.sum() >= 5:
                    recall = (preds[sg_class_mask] == c).mean()
                    recalls.append(float(recall))
            if len(recalls) >= 2:
                per_class_disparities.append(max(recalls) - min(recalls))

        if per_class_disparities:
            return float(np.mean(per_class_disparities))

        # Fallback: when classes are entangled with subgroups (e.g., CIFAR-10
        # where minority/majority are defined by class membership), compute
        # the max difference in error rate across subgroups
        error_rates = []
        for sg in unique_subgroups:
            mask = subgroups == sg
            if mask.sum() >= 5:
                error_rates.append(1.0 - float((preds[mask] == labels[mask]).mean()))
        if len(error_rates) >= 2:
            return max(error_rates) - min(error_rates)
        return 0.0


def compute_demographic_parity_diff(preds, subgroups, unique_subgroups):
    """Max difference in positive prediction rate across subgroups."""
    rates = []
    for sg in unique_subgroups:
        mask = subgroups == sg
        if mask.sum() > 0:
            rate = (preds[mask] == 1).mean() if len(preds[mask]) > 0 else 0
            rates.append(float(rate))
    if len(rates) < 2:
        return 0.0
    return max(rates) - min(rates)
