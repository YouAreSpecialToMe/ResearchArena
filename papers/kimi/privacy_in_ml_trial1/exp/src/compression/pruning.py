"""
Model compression methods: magnitude pruning and quantization.
"""

import torch
import torch.nn as nn
import copy


def apply_magnitude_pruning(model, sparsity_ratio):
    """
    Apply unstructured magnitude pruning to a model.
    
    Args:
        model: PyTorch model
        sparsity_ratio: Fraction of weights to prune (e.g., 0.7 means 70% pruned)
    
    Returns:
        pruned_model: Model with pruned weights
        actual_sparsity: Actual sparsity achieved
    """
    pruned_model = copy.deepcopy(model)
    
    # Collect all weights
    all_weights = []
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:  # Only conv and linear weights
            all_weights.append(param.data.abs().flatten())
    
    # Compute global threshold
    all_weights = torch.cat(all_weights)
    k = int(sparsity_ratio * len(all_weights))
    if k > 0:
        threshold = torch.kthvalue(all_weights, k)[0]
    else:
        threshold = 0.0
    
    # Apply pruning
    total_params = 0
    pruned_params = 0
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            mask = (param.data.abs() >= threshold).float()
            param.data = param.data * mask
            total_params += param.numel()
            pruned_params += (mask == 0).sum().item()
    
    actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0
    return pruned_model, actual_sparsity


def apply_layerwise_magnitude_pruning(model, sparsity_ratio):
    """
    Apply layer-wise magnitude pruning (each layer pruned independently).
    
    Args:
        model: PyTorch model
        sparsity_ratio: Fraction of weights to prune per layer
    
    Returns:
        pruned_model: Model with pruned weights
        actual_sparsity: Actual sparsity achieved
    """
    pruned_model = copy.deepcopy(model)
    
    total_params = 0
    pruned_params = 0
    
    for name, param in pruned_model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            # Compute layer-wise threshold
            weights_abs = param.data.abs().flatten()
            k = int(sparsity_ratio * len(weights_abs))
            if k > 0:
                threshold = torch.kthvalue(weights_abs, k)[0]
                mask = (param.data.abs() >= threshold).float()
            else:
                mask = torch.ones_like(param.data)
            
            param.data = param.data * mask
            total_params += param.numel()
            pruned_params += (mask == 0).sum().item()
    
    actual_sparsity = pruned_params / total_params if total_params > 0 else 0.0
    return pruned_model, actual_sparsity


def apply_quantization(model, bits=8):
    """
    Apply post-training quantization.
    
    Args:
        model: PyTorch model
        bits: Number of bits for quantization (8 or 4)
    
    Returns:
        quantized_model: Model with quantized weights
    """
    quantized_model = copy.deepcopy(model)
    
    for name, param in quantized_model.named_parameters():
        if 'weight' in name:
            # Simple uniform quantization
            w_min = param.data.min()
            w_max = param.data.max()
            
            # Quantize to [0, 2^bits - 1]
            levels = 2 ** bits - 1
            param.data = torch.round((param.data - w_min) / (w_max - w_min) * levels)
            param.data = param.data / levels * (w_max - w_min) + w_min
    
    return quantized_model


def count_nonzero_params(model):
    """Count non-zero parameters in model."""
    total = 0
    nonzero = 0
    for param in model.parameters():
        total += param.numel()
        nonzero += (param.data != 0).sum().item()
    return nonzero, total


def evaluate_with_pruning(model, testloader, device, sparsity_ratio):
    """Evaluate model with pruning applied."""
    pruned_model, actual_sparsity = apply_magnitude_pruning(model, sparsity_ratio)
    pruned_model = pruned_model.to(device)
    pruned_model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = pruned_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy, actual_sparsity
