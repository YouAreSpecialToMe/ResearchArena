"""
Shared utilities for CompViz experiments.
"""

import json
import time
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(data: Any, path: str):
    """Save data to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.elapsed is None:
            self.elapsed = time.time() - self.start_time
        return self.elapsed * 1000


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute mean, std, min, max, percentiles."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99))
    }


def calculate_accuracy(predictions: List[Any], ground_truth: List[Any]) -> float:
    """Calculate accuracy given predictions and ground truth."""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if str(p).lower() == str(g).lower())
    return correct / len(predictions)


def bootstrap_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (0.0, 0.0)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return (float(lower), float(upper))
