"""
Utility functions for experiments.
"""
import json
import numpy as np
import os
import time
from typing import Dict, Any


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


class Timer:
    """Simple timer for profiling."""
    
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


def kernel_weights(X_query: np.ndarray, X_ref: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Compute Gaussian kernel weights.
    
    Args:
        X_query: Query point (d,)
        X_ref: Reference points (n, d)
        bandwidth: Kernel bandwidth
    
    Returns:
        weights: (n,) array of kernel weights
    """
    if len(X_ref) == 0:
        return np.array([])
    
    distances = np.linalg.norm(X_ref - X_query, axis=1)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from weights."""
    if np.sum(weights) == 0:
        return 0.0
    weights_norm = weights / np.sum(weights)
    ess = 1.0 / np.sum(weights_norm ** 2)
    return ess


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average."""
    if window <= 1:
        return data
    
    result = np.convolve(data, np.ones(window) / window, mode='valid')
    # Pad at the beginning
    padding = np.zeros(window - 1)
    for i in range(window - 1):
        padding[i] = np.mean(data[:i+1])
    
    return np.concatenate([padding, result])
