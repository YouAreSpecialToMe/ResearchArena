"""
Common utilities for LayoutLearner experiments.
"""
import numpy as np
import json
import time
import os
from typing import Dict, Any, Tuple


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)


def save_json(data: Dict, filepath: str):
    """Save dictionary to JSON file with numpy type handling."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(data), f, indent=2)


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_csv(data: Dict, filepath: str):
    """Save data to CSV."""
    import pandas as pd
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pd.DataFrame(data).to_csv(filepath, index=False)


def time_function(func, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function execution."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory structure."""
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    return exp_dir


def get_project_paths() -> Dict[str, str]:
    """Get standard project paths."""
    base = '/home/nw366/ResearchArena/outputs/kimi_v5_compiler_optimization/idea_01'
    return {
        'base': base,
        'exp': os.path.join(base, 'exp'),
        'data': os.path.join(base, 'data'),
        'models': os.path.join(base, 'models'),
        'figures': os.path.join(base, 'figures'),
        'benchmarks': os.path.join(base, 'benchmarks')
    }


def log_experiment(exp_dir: str, message: str):
    """Log message to experiment log file."""
    log_file = os.path.join(exp_dir, 'logs', 'experiment.log')
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f}s")


def aggregate_results(results_list: list) -> Dict:
    """Aggregate results from multiple seeds."""
    if not results_list:
        return {}
    
    aggregated = {}
    keys = results_list[0].keys()
    
    for key in keys:
        if isinstance(results_list[0][key], (int, float)):
            values = [r[key] for r in results_list]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        else:
            aggregated[key] = results_list[0][key]
    
    return aggregated
