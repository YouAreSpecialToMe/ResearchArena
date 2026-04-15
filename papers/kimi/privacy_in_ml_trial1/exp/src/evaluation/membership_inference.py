"""
Membership Inference Attack (MIA) evaluation.
Implements loss-based threshold attack.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


def compute_sample_losses(model, dataloader, device):
    """Compute per-sample cross-entropy losses."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    all_losses = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            losses = criterion(outputs, targets)
            all_losses.append(losses.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    return np.concatenate(all_losses), np.concatenate(all_labels)


def evaluate_membership_inference(
    model,
    member_loader,
    non_member_loader,
    device,
    attack_type='threshold'
):
    """
    Evaluate membership inference attack.
    
    Args:
        model: Target model
        member_loader: Data loader for member (training) samples
        non_member_loader: Data loader for non-member (test) samples
        device: Device
        attack_type: Type of attack ('threshold' or 'nn')
    
    Returns:
        dict with MIA metrics
    """
    # Get losses for members and non-members
    member_losses, _ = compute_sample_losses(model, member_loader, device)
    non_member_losses, _ = compute_sample_losses(model, non_member_loader, device)
    
    # Create labels (1 for member, 0 for non-member)
    y_true = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(non_member_losses))
    ])
    
    # Create predictions based on loss threshold
    # Lower loss -> more likely to be member
    all_losses = np.concatenate([member_losses, non_member_losses])
    
    # Use negative loss as membership score (higher = more likely member)
    y_scores = -all_losses
    
    # Find optimal threshold
    best_acc = 0
    best_threshold = 0
    for threshold in np.percentile(y_scores, range(0, 101, 5)):
        y_pred = (y_scores >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    # Compute metrics
    y_pred = (y_scores >= best_threshold).astype(int)
    mia_accuracy = accuracy_score(y_true, y_pred)
    mia_auc = roc_auc_score(y_true, y_scores)
    
    # Compute TPR and FPR
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'mia_accuracy': mia_accuracy,
        'mia_auc': mia_auc,
        'mia_tpr': tpr,
        'mia_fpr': fpr,
        'threshold': best_threshold,
        'member_loss_mean': member_losses.mean(),
        'member_loss_std': member_losses.std(),
        'non_member_loss_mean': non_member_losses.mean(),
        'non_member_loss_std': non_member_losses.std()
    }


def prepare_mia_data(dataset, train_ratio=0.9, member_for_attack_ratio=0.5):
    """
    Prepare data splits for MIA evaluation.
    
    Returns:
        member_train_loader: For training target model
        member_eval_loader: Members for MIA evaluation
        non_member_loader: Non-members for MIA evaluation
    """
    # Get all data
    all_data = []
    all_labels = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        all_data.append(x)
        all_labels.append(y)
    
    all_data = torch.stack(all_data)
    all_labels = torch.tensor(all_labels)
    
    # Split into train (for model training) and test (non-members for MIA)
    train_indices, test_indices = train_test_split(
        range(len(all_data)), test_size=1-train_ratio, random_state=42)
    
    # Further split training data for MIA evaluation
    train_indices_for_mia, member_eval_indices = train_test_split(
        train_indices, test_size=member_for_attack_ratio, random_state=42)
    
    # Create datasets
    train_data = torch.utils.data.TensorDataset(
        all_data[train_indices_for_mia], all_labels[train_indices_for_mia])
    member_eval_data = torch.utils.data.TensorDataset(
        all_data[member_eval_indices], all_labels[member_eval_indices])
    non_member_data = torch.utils.data.TensorDataset(
        all_data[test_indices], all_labels[test_indices])
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    member_eval_loader = torch.utils.data.DataLoader(member_eval_data, batch_size=256, shuffle=False)
    non_member_loader = torch.utils.data.DataLoader(non_member_data, batch_size=256, shuffle=False)
    
    return train_loader, member_eval_loader, non_member_loader
