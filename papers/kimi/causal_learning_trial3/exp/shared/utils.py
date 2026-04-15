"""
Shared utility functions for experiments.
"""
import numpy as np
import json
import pickle
import time
from typing import Dict, Any


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load a dataset from pickle file."""
    with open(dataset_path, 'rb') as f:
        return pickle.load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
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
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert(results), f, indent=2)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        
    def __float__(self):
        return self.elapsed
        
    def __str__(self):
        return f"{self.elapsed:.4f}s"


def count_ci_tests_pc(n_nodes: int, max_cond_size: int = None) -> int:
    """
    Estimate the number of CI tests performed by PC algorithm.
    This is a rough estimate based on the worst case.
    
    Args:
        n_nodes: Number of nodes
        max_cond_size: Maximum conditioning set size
        
    Returns:
        Estimated number of CI tests
    """
    if max_cond_size is None:
        max_cond_size = n_nodes - 2
    
    n_edges = n_nodes * (n_nodes - 1) // 2
    count = 0
    
    for d in range(max_cond_size + 1):
        # For each edge, we might test with conditioning sets of size d
        # The number of possible conditioning sets is C(n-2, d)
        from math import comb
        count += n_edges * comb(n_nodes - 2, d)
    
    return count


def compute_cost_savings(n_tests_our: int, n_tests_baseline: int, 
                         cost_ratio: float = 15.0) -> Dict[str, float]:
    """
    Compute cost savings metrics.
    
    Args:
        n_tests_our: Number of high-fidelity tests in our method
        n_tests_baseline: Number of high-fidelity tests in baseline
        cost_ratio: Cost ratio of high-fidelity to low-fidelity tests
        
    Returns:
        Dictionary with savings metrics
    """
    absolute_reduction = n_tests_baseline - n_tests_our
    percentage_reduction = 100 * absolute_reduction / n_tests_baseline if n_tests_baseline > 0 else 0
    
    return {
        'baseline_tests': n_tests_baseline,
        'our_tests': n_tests_our,
        'absolute_reduction': absolute_reduction,
        'percentage_reduction': percentage_reduction
    }
