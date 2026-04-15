"""Common utilities for reproducibility and logging."""
import os
import json
import random
import numpy as np
import torch
import time
from pathlib import Path


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_results(results, path):
    """Save results dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def load_results(path):
    """Load results from JSON."""
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self, name=""):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"[{self.name}] {self.elapsed:.1f}s ({self.elapsed/60:.1f}min)")
