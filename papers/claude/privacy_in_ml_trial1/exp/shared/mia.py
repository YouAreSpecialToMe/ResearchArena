"""Membership Inference Attack evaluation methods."""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset


def compute_sample_losses(model, dataset, indices, batch_size=256, device='cuda'):
    """Compute per-sample cross-entropy loss."""
    model.eval()
    model.to(device)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            per_sample_loss = F.cross_entropy(outputs, targets, reduction='none')
            losses.append(per_sample_loss.cpu().numpy())
    return np.concatenate(losses)


def compute_sample_metrics(model, dataset, indices, batch_size=256, device='cuda'):
    """Compute per-sample confidence, entropy, and modified entropy."""
    model.eval()
    model.to(device)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_conf = []
    all_entropy = []
    all_mentropy = []

    # Get labels
    labels = []
    for idx in indices:
        _, label = dataset[idx]
        labels.append(label)
    labels = np.array(labels)

    label_offset = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            # Confidence (max softmax probability)
            conf = probs.max(dim=1)[0].cpu().numpy()
            all_conf.append(conf)

            # Entropy = -sum p_c log p_c
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).cpu().numpy()
            all_entropy.append(entropy)

            # Modified entropy
            batch_size_actual = len(inputs)
            batch_labels = labels[label_offset:label_offset + batch_size_actual]
            p_true = probs[range(batch_size_actual), batch_labels].cpu().numpy()
            # m_entropy = -(1-p_true)*log(p_true) - sum_{c!=true} p_c*log(1-p_c)
            probs_np = probs.cpu().numpy()
            m_entropy = -(1 - p_true) * np.log(p_true + 1e-10)
            for c in range(probs_np.shape[1]):
                mask = (batch_labels != c)
                m_entropy[mask] -= probs_np[mask, c] * np.log(1 - probs_np[mask, c] + 1e-10)
            all_mentropy.append(m_entropy)

            label_offset += batch_size_actual

    return {
        'confidence': np.concatenate(all_conf),
        'entropy': np.concatenate(all_entropy),
        'mentropy': np.concatenate(all_mentropy),
    }


def compute_mia_auc(member_scores, nonmember_scores, higher_is_member=True):
    """Compute MIA AUC-ROC and TPR at low FPR."""
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])

    if not higher_is_member:
        scores = -scores

    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    # TPR at FPR=1%
    idx_1pct = np.searchsorted(fpr, 0.01)
    tpr_at_1pct = tpr[min(idx_1pct, len(tpr)-1)]

    # TPR at FPR=0.1%
    idx_01pct = np.searchsorted(fpr, 0.001)
    tpr_at_01pct = tpr[min(idx_01pct, len(tpr)-1)]

    return {
        'auc': float(auc),
        'tpr_at_fpr1pct': float(tpr_at_1pct),
        'tpr_at_fpr01pct': float(tpr_at_01pct),
    }


def run_mia_evaluation(model, eval_dataset, member_indices, nonmember_indices,
                       num_eval=2000, device='cuda', seed=42):
    """
    Run full MIA evaluation suite.

    Args:
        model: model to evaluate
        eval_dataset: dataset without augmentation
        member_indices: indices used for training
        nonmember_indices: indices NOT used for training
        num_eval: number of samples to evaluate (per group)
        device: cuda or cpu
        seed: for selecting evaluation subset

    Returns:
        dict with MIA results for each attack type
    """
    rng = np.random.RandomState(seed)
    n_eval = min(num_eval, len(member_indices), len(nonmember_indices))

    eval_member = rng.choice(member_indices, n_eval, replace=False)
    eval_nonmember = rng.choice(nonmember_indices, n_eval, replace=False)

    # Loss-based MIA (members have LOWER loss)
    member_losses = compute_sample_losses(model, eval_dataset, eval_member, device=device)
    nonmember_losses = compute_sample_losses(model, eval_dataset, eval_nonmember, device=device)
    loss_mia = compute_mia_auc(member_losses, nonmember_losses, higher_is_member=False)

    # Metric-based MIA
    member_metrics = compute_sample_metrics(model, eval_dataset, eval_member, device=device)
    nonmember_metrics = compute_sample_metrics(model, eval_dataset, eval_nonmember, device=device)

    conf_mia = compute_mia_auc(member_metrics['confidence'], nonmember_metrics['confidence'],
                               higher_is_member=True)
    entropy_mia = compute_mia_auc(member_metrics['entropy'], nonmember_metrics['entropy'],
                                  higher_is_member=False)
    mentropy_mia = compute_mia_auc(member_metrics['mentropy'], nonmember_metrics['mentropy'],
                                   higher_is_member=False)

    return {
        'mia_loss': loss_mia,
        'mia_confidence': conf_mia,
        'mia_entropy': entropy_mia,
        'mia_mentropy': mentropy_mia,
    }


def lira_evaluation(target_model, shadow_models, shadow_splits, eval_dataset,
                    member_indices, nonmember_indices, num_eval=1000, device='cuda', seed=42):
    """
    Simplified LiRA (Likelihood Ratio Attack) evaluation.

    Args:
        target_model: the model being attacked
        shadow_models: list of trained shadow models
        shadow_splits: list of dicts with 'member'/'nonmember' indices
        eval_dataset: dataset without augmentation
        member_indices: target model's member indices
        nonmember_indices: target model's non-member indices
        num_eval: samples per group
        device: cuda or cpu

    Returns:
        dict with LiRA results
    """
    rng = np.random.RandomState(seed)
    n_eval = min(num_eval, len(member_indices), len(nonmember_indices))

    eval_member = rng.choice(member_indices, n_eval, replace=False)
    eval_nonmember = rng.choice(nonmember_indices, n_eval, replace=False)
    eval_indices = np.concatenate([eval_member, eval_nonmember])
    eval_labels = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

    # Compute losses under each shadow model for each eval sample
    shadow_losses = []
    for sm in shadow_models:
        sm.to(device)
        losses = compute_sample_losses(sm, eval_dataset, eval_indices, device=device)
        shadow_losses.append(losses)
        sm.cpu()
    shadow_losses = np.array(shadow_losses)  # (num_shadows, num_eval_samples)

    # For each eval sample, determine which shadow models included it
    # and fit Gaussians for in/out distributions
    log_ratios = np.zeros(len(eval_indices))

    for i, idx in enumerate(eval_indices):
        in_losses = []
        out_losses = []
        for s, split in enumerate(shadow_splits):
            if idx in set(split['member']):
                in_losses.append(shadow_losses[s, i])
            else:
                out_losses.append(shadow_losses[s, i])

        if len(in_losses) < 2 or len(out_losses) < 2:
            log_ratios[i] = 0.0
            continue

        in_losses = np.array(in_losses)
        out_losses = np.array(out_losses)

        # Fit Gaussians
        mu_in, sigma_in = in_losses.mean(), in_losses.std() + 1e-10
        mu_out, sigma_out = out_losses.mean(), out_losses.std() + 1e-10

        # Target model loss for this sample
        target_model.to(device)
        target_loss = compute_sample_losses(target_model, eval_dataset, [idx], device=device)[0]

        # Log likelihood ratio
        log_in = -0.5 * ((target_loss - mu_in) / sigma_in) ** 2 - np.log(sigma_in)
        log_out = -0.5 * ((target_loss - mu_out) / sigma_out) ** 2 - np.log(sigma_out)
        log_ratios[i] = log_in - log_out

    # Compute AUC
    result = compute_mia_auc(log_ratios[:n_eval], log_ratios[n_eval:], higher_is_member=True)
    result['attack_type'] = 'lira'
    return result
