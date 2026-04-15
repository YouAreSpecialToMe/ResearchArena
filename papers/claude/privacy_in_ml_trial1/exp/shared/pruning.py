"""Pruning methods: Random, Magnitude, Gradient Sensitivity, MemPrune (GDS-based)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


def get_prunable_params(model):
    """Get list of (name, param) for prunable parameters (exclude BN and bias)."""
    prunable = []
    for name, param in model.named_parameters():
        if 'bn' not in name and 'bias' not in name and param.dim() >= 2:
            prunable.append((name, param))
    return prunable


def apply_mask(model, mask_dict):
    """Apply binary mask to model parameters (zero out pruned weights)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask_dict:
                param.mul_(mask_dict[name].to(param.device))


def get_sparsity(model, mask_dict):
    """Compute actual sparsity of masked model."""
    total = 0
    pruned = 0
    for name, param in get_prunable_params(model):
        if name in mask_dict:
            total += param.numel()
            pruned += (mask_dict[name] == 0).sum().item()
    return pruned / total if total > 0 else 0.0


def random_pruning(model, sparsity, seed=42):
    """Random unstructured pruning."""
    rng = np.random.RandomState(seed)
    mask_dict = {}
    for name, param in get_prunable_params(model):
        mask = torch.ones_like(param)
        flat = mask.flatten()
        n_prune = int(len(flat) * sparsity)
        indices = rng.choice(len(flat), n_prune, replace=False)
        flat[indices] = 0
        mask_dict[name] = mask
    return mask_dict


def magnitude_pruning(model, sparsity):
    """Global unstructured magnitude pruning (remove smallest magnitude weights)."""
    prunable = get_prunable_params(model)
    # Collect all magnitudes
    all_mags = []
    for name, param in prunable:
        all_mags.append(param.detach().abs().cpu().flatten())
    all_mags = torch.cat(all_mags)

    # Find threshold
    n_prune = int(len(all_mags) * sparsity)
    if n_prune == 0:
        return {name: torch.ones_like(param) for name, param in prunable}
    threshold = torch.sort(all_mags)[0][n_prune]

    # Create masks
    mask_dict = {}
    for name, param in prunable:
        mask_dict[name] = (param.detach().abs() >= threshold).float().cpu()

    return mask_dict


def gradient_sensitivity_pruning(model, data_loader, sparsity, device='cuda', num_batches=10):
    """Gradient sensitivity pruning: importance = |weight * gradient|."""
    model.eval()
    model.to(device)

    # Accumulate gradients
    model.zero_grad()
    count = 0
    for inputs, targets in data_loader:
        if count >= num_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        count += 1

    # Compute importance scores = |w * grad|
    prunable = get_prunable_params(model)
    all_scores = []
    for name, param in prunable:
        score = (param.detach() * param.grad).abs().cpu().flatten()
        all_scores.append(score)
    all_scores = torch.cat(all_scores)

    # Find threshold (remove LOWEST scores)
    n_prune = int(len(all_scores) * sparsity)
    if n_prune == 0:
        return {name: torch.ones_like(param) for name, param in prunable}
    threshold = torch.sort(all_scores)[0][n_prune]

    mask_dict = {}
    for name, param in prunable:
        score = (param.detach() * param.grad).abs().cpu()
        mask_dict[name] = (score >= threshold).float()

    model.zero_grad()
    return mask_dict


def gds_pruning(model, gds_dict, sparsity, mode='high'):
    """
    GDS-based global unstructured pruning (matches magnitude pruning approach).
    mode='high': prune HIGHEST GDS weights (MemPrune - removes memorization)
    mode='low': prune LOWEST GDS weights (Reverse-GDS ablation)
    """
    param_names = []
    for name, param in get_prunable_params(model):
        if name in gds_dict:
            param_names.append(name)

    if not param_names:
        return {}

    # Global unstructured pruning — same approach as magnitude_pruning
    all_gds = torch.cat([gds_dict[n].flatten() for n in param_names])
    n_total = len(all_gds)
    n_prune = int(n_total * sparsity)
    if n_prune == 0:
        return {name: torch.ones_like(gds_dict[name]) for name in param_names}

    sorted_gds = torch.sort(all_gds)[0]

    mask_dict = {}
    if mode == 'high':
        # Prune highest GDS: keep weights below threshold
        threshold = sorted_gds[n_total - n_prune]
        for name in param_names:
            mask_dict[name] = (gds_dict[name] <= threshold).float()
    else:
        # Prune lowest GDS: keep weights above threshold
        threshold = sorted_gds[n_prune]
        for name in param_names:
            mask_dict[name] = (gds_dict[name] >= threshold).float()

    return mask_dict


def hybrid_pruning(model, gds_dict, sparsity, alpha=0.5):
    """Hybrid criterion: alpha * GDS_norm + (1-alpha) * (-|w|_norm)."""
    all_scores = []
    param_names = []
    for name, param in get_prunable_params(model):
        if name in gds_dict:
            gds_norm = gds_dict[name].flatten()
            mag = param.detach().abs().cpu().flatten()
            mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
            score = alpha * gds_norm + (1 - alpha) * (-mag_norm)
            all_scores.append(score)
            param_names.append(name)

    all_scores_cat = torch.cat(all_scores)
    n_prune = int(len(all_scores_cat) * sparsity)
    threshold = torch.sort(all_scores_cat, descending=True)[0][n_prune]  # prune highest score

    mask_dict = {}
    offset = 0
    for name in param_names:
        gds_norm = gds_dict[name].flatten()
        param = dict(model.named_parameters())[name]
        mag = param.detach().abs().cpu().flatten()
        mag_norm = (mag - mag.mean()) / (mag.std() + 1e-8)
        score = alpha * gds_norm + (1 - alpha) * (-mag_norm)
        mask = (score < threshold).float().reshape(param.shape)
        mask_dict[name] = mask
        offset += len(gds_norm)

    return mask_dict
