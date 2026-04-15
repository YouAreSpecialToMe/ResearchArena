"""
Utility functions for experiments.
"""
import json
import os
import torch
import numpy as np
import random
import time


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.2f} seconds")


def aggregate_seeds(results_list):
    """
    Aggregate results across multiple random seeds.
    
    Args:
        results_list: List of result dictionaries from different seeds
    
    Returns:
        Dictionary with mean and std for each metric
    """
    if not results_list:
        return {}
    
    # Get all metric keys
    keys = results_list[0].keys()
    
    aggregated = {}
    for key in keys:
        values = [r[key] for r in results_list if key in r]
        if values and isinstance(values[0], (int, float)):
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        else:
            aggregated[key] = values[0] if values else None
    
    return aggregated


def print_metrics(metrics, prefix=""):
    """Pretty print metrics."""
    print(f"\n{prefix}Metrics:")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, dict) and 'mean' in value:
            print(f"  {key}: {value['mean']:.4f} ± {value['std']:.4f}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("-" * 50)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
