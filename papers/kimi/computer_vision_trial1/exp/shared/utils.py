"""
Utility functions for DU-VPT experiments.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import time


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} took {elapsed:.2f} seconds")
    
    def elapsed(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count number of parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy loss for test-time adaptation."""
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return entropy.mean()


def fisher_regularization(
    current_params: torch.Tensor,
    initial_params: torch.Tensor,
    fisher_info: torch.Tensor
) -> torch.Tensor:
    """
    Compute Fisher regularization penalty.
    
    Args:
        current_params: Current parameter values
        initial_params: Initial parameter values
        fisher_info: Fisher information matrix (diagonal approximation)
    
    Returns:
        Regularization loss
    """
    diff = current_params - initial_params
    penalty = (fisher_info * diff ** 2).sum() / 2.0
    return penalty


def compute_calibration_statistics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_layers: int = 12
) -> Dict[str, torch.Tensor]:
    """
    Compute calibration statistics from a dataset.
    
    Returns:
        Dict with 'mean' and 'std' tensors of shape [n_layers, embed_dim]
    """
    model.eval()
    
    layer_features = [[] for _ in range(n_layers)]
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Extract layer features
            features_list = []
            hooks = []
            
            def make_hook(idx):
                def hook(module, input, output):
                    features_list.append((idx, output))
                return hook
            
            blocks = model.blocks if hasattr(model, 'blocks') else model.transformer.blocks
            for i, block in enumerate(blocks):
                hook = block.register_forward_hook(make_hook(i))
                hooks.append(hook)
            
            _ = model(inputs)
            
            for hook in hooks:
                hook.remove()
            
            # Organize features by layer
            for idx, features in sorted(features_list, key=lambda x: x[0]):
                # Use cls token
                cls_feat = features[:, 0, :].cpu()
                layer_features[idx].append(cls_feat)
    
    # Compute statistics per layer
    means = []
    stds = []
    
    for layer_idx in range(n_layers):
        if layer_features[layer_idx]:
            feats = torch.cat(layer_features[layer_idx], dim=0)
            means.append(feats.mean(dim=0))
            stds.append(feats.std(dim=0))
        else:
            # Default values
            embed_dim = 768  # ViT-B/16
            means.append(torch.zeros(embed_dim))
            stds.append(torch.ones(embed_dim))
    
    return {
        'mean': torch.stack(means),
        'std': torch.stack(stds)
    }


def log_message(message: str, log_file: Optional[str] = None):
    """Print and optionally log a message."""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')


def format_results_table(results: Dict[str, Dict], metrics: list = None) -> str:
    """Format results as a readable table."""
    if metrics is None:
        metrics = ['top1_acc', 'ece']
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"{'Method':<20} {'Top-1 Acc':<15} {'ECE':<10}")
    lines.append("-" * 60)
    
    for method, result in results.items():
        acc = result.get('top1_acc', {})
        ece = result.get('ece', {})
        
        if isinstance(acc, dict):
            acc_str = f"{acc.get('mean', 0):.2f} ± {acc.get('std', 0):.2f}"
        else:
            acc_str = f"{acc:.2f}"
        
        if isinstance(ece, dict):
            ece_str = f"{ece.get('mean', 0):.3f}"
        else:
            ece_str = f"{ece:.3f}"
        
        lines.append(f"{method:<20} {acc_str:<15} {ece_str:<10}")
    
    lines.append("=" * 60)
    
    return '\n'.join(lines)
