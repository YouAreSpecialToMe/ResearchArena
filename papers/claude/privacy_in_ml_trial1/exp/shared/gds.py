"""Gradient Dispersion Score (GDS) computation using per-sample gradients."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import grad, vmap
import numpy as np
from tqdm import tqdm
import copy


def compute_gds(model, data_loader, num_samples=5000, micro_batch_size=32, device='cuda'):
    """
    Compute per-weight Gradient Dispersion Score (variance of per-sample gradients).
    Uses Welford's online algorithm for streaming variance computation.

    Args:
        model: trained model (in eval mode)
        data_loader: DataLoader with training data (no augmentation)
        num_samples: number of samples to use for GDS estimation
        micro_batch_size: samples per micro-batch for vmap
        device: cuda or cpu

    Returns:
        gds_dict: dict mapping param_name -> GDS tensor (variance of per-sample gradients)
        stats: dict with per-layer statistics
    """
    model = model.to(device)
    model.eval()

    # Collect prunable parameter names and shapes
    param_names = []
    param_shapes = {}
    for name, param in model.named_parameters():
        if 'bn' not in name and 'bias' not in name and param.dim() >= 2:
            param_names.append(name)
            param_shapes[name] = param.shape

    # Create functional version for per-sample gradients
    # We'll use a simpler approach: backward per sample in micro-batches
    # For efficiency, we use torch.func.vmap with grad

    # Collect samples
    all_inputs = []
    all_targets = []
    count = 0
    for inputs, targets in data_loader:
        all_inputs.append(inputs)
        all_targets.append(targets)
        count += len(inputs)
        if count >= num_samples:
            break
    all_inputs = torch.cat(all_inputs)[:num_samples].to(device)
    all_targets = torch.cat(all_targets)[:num_samples].to(device)

    print(f"Computing GDS with {len(all_inputs)} samples, micro_batch={micro_batch_size}")

    # Welford's online algorithm accumulators
    welford_count = 0
    welford_mean = {}
    welford_m2 = {}
    for name in param_names:
        welford_mean[name] = torch.zeros(param_shapes[name], device=device, dtype=torch.float64)
        welford_m2[name] = torch.zeros(param_shapes[name], device=device, dtype=torch.float64)

    # Process in micro-batches using standard backward
    for start in tqdm(range(0, len(all_inputs), micro_batch_size), desc="GDS computation"):
        end = min(start + micro_batch_size, len(all_inputs))
        batch_x = all_inputs[start:end]
        batch_y = all_targets[start:end]

        # Compute per-sample gradients via individual backward passes
        for i in range(len(batch_x)):
            model.zero_grad()
            output = model(batch_x[i:i+1])
            loss = F.cross_entropy(output, batch_y[i:i+1])
            loss.backward()

            # Update Welford's accumulators
            welford_count += 1
            for name in param_names:
                param = dict(model.named_parameters())[name]
                grad_val = param.grad.to(torch.float64)
                delta = grad_val - welford_mean[name]
                welford_mean[name] += delta / welford_count
                delta2 = grad_val - welford_mean[name]
                welford_m2[name] += delta * delta2

    # Compute variance (GDS)
    gds_dict = {}
    stats = {}
    for name in param_names:
        variance = (welford_m2[name] / (welford_count - 1)).float()
        gds_dict[name] = variance.cpu()
        stats[name] = {
            'mean': float(variance.mean()),
            'std': float(variance.std()),
            'min': float(variance.min()),
            'max': float(variance.max()),
            'median': float(variance.median()),
            'p99': float(variance.quantile(0.99)),
        }

    return gds_dict, stats


def normalize_gds(gds_dict, mode='layerwise'):
    """
    Normalize GDS values.

    Args:
        gds_dict: dict mapping param_name -> GDS tensor
        mode: 'layerwise' (default), 'global', or 'none'

    Returns:
        normalized GDS dict
    """
    if mode == 'none':
        return gds_dict

    if mode == 'global':
        all_vals = torch.cat([v.flatten() for v in gds_dict.values()])
        mu = all_vals.mean()
        sigma = all_vals.std()
        return {k: (v - mu) / (sigma + 1e-8) for k, v in gds_dict.items()}

    # layerwise z-score
    normed = {}
    for name, gds in gds_dict.items():
        mu = gds.mean()
        sigma = gds.std()
        normed[name] = (gds - mu) / (sigma + 1e-8)
    return normed


def compute_gds_magnitude_correlation(model, gds_dict):
    """Compute per-layer Pearson correlation between GDS and weight magnitude."""
    correlations = {}
    for name, gds in gds_dict.items():
        param = dict(model.named_parameters())[name]
        mag = param.detach().cpu().abs().flatten().float()
        gds_flat = gds.flatten().float()
        corr = float(torch.corrcoef(torch.stack([gds_flat, mag]))[0, 1])
        correlations[name] = corr
    return correlations
