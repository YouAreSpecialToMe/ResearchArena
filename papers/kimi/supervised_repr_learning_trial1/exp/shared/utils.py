"""
Common utilities for experiments.
"""

import torch
import numpy as np
import random
import json
import os
import time
from datetime import datetime


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(results, filepath):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp():
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    """Simple timer for tracking execution time."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed = time.time() - self.start_time
            self.start_time = None
        return self.elapsed
    
    def get_elapsed(self):
        if self.start_time is not None:
            return time.time() - self.start_time
        return self.elapsed


def save_checkpoint(model, optimizer, epoch, filepath, **kwargs):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, filepath, optimizer=None, device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def count_parameters(model):
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
