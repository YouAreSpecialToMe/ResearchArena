"""Shared utilities for SAE experiments."""
import json
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Any


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results(path: str, results: Dict[str, Any]):
    """Save results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation between two arrays."""
    from scipy.stats import spearmanr
    corr, _ = spearmanr(x, y)
    return corr


def compute_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    from scipy.stats import pearsonr
    corr, _ = pearsonr(x, y)
    return corr
