"""Shared utilities for CAGER experiments."""
import random
import numpy as np
import torch
from typing import Dict, Any, List


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


def save_json(data: Dict[str, Any], path: str):
    """Save data to JSON file with numpy type conversion."""
    import json
    serializable_data = convert_to_serializable(data)
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    import json
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: str):
    """Save data to pickle file."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load data from pickle file."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two vectors."""
    from scipy.stats import spearmanr
    corr, _ = spearmanr(x, y)
    return corr


def compute_pairwise_distances(X: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """Compute pairwise distance matrix.
    
    Args:
        X: (n_samples, n_features) array
        metric: 'cosine', 'euclidean', or 'correlation'
    
    Returns:
        Distance matrix (n_samples, n_samples)
    """
    from sklearn.metrics import pairwise_distances as pdist
    return pdist(X, metric=metric)


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(42)
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    mean = np.mean(data)
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return mean, lower, upper
