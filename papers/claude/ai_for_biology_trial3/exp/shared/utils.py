"""
Common utility functions.
"""
import os
import json
import time
import torch
import numpy as np
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def save_results(results, path):
    """Save results dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)


class Timer:
    """Simple timer context manager."""
    def __init__(self, name=""):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.name:
            print(f"  [{self.name}] {self.elapsed:.1f}s")
