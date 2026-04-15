"""
Utility functions for training and evaluation.
"""
import torch
import numpy as np
import random
import json
import os
from pathlib import Path
import time


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def save_results(results, path):
    """Save results to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path):
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


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


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
        
        return self.should_stop


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def log_message(message, log_file=None):
    """Print and optionally log a message."""
    print(message)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(message + '\n')


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    print(f"Random number: {torch.rand(1).item()}")
    
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    print(f"Elapsed: {timer.stop():.3f}s")
