"""
Metrics for conformal prediction evaluation.
"""
import numpy as np
from typing import Dict, List, Tuple


def marginal_coverage(coverage_indicators: np.ndarray) -> float:
    """Compute marginal coverage."""
    return np.mean(coverage_indicators)


def average_set_width(widths: np.ndarray) -> float:
    """Compute average prediction set width."""
    return np.mean(widths)


def msce(coverage_indicators: np.ndarray, features: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Marginal Sorted Coverage Error (MSCE).
    Bin by feature values and compute coverage deviation in each bin.
    """
    n_samples = len(coverage_indicators)
    bin_size = n_samples // n_bins
    
    # Sort by first feature dimension
    sorted_indices = np.argsort(features[:, 0])
    sorted_coverage = coverage_indicators[sorted_indices]
    
    bin_coverages = []
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n_samples
        bin_coverage = np.mean(sorted_coverage[start:end])
        bin_coverages.append(bin_coverage)
    
    # MSCE is the mean absolute deviation from target
    target_coverage = 0.9  # Assuming alpha=0.1
    msce_value = np.mean(np.abs(np.array(bin_coverages) - target_coverage))
    return msce_value


def wsc(coverage_indicators: np.ndarray, features: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Worst-Slab Coverage (WSC).
    Find the minimum coverage across different "slabs" (feature bins).
    """
    n_samples = len(coverage_indicators)
    bin_size = n_samples // n_bins
    
    # Sort by first feature dimension
    sorted_indices = np.argsort(features[:, 0])
    sorted_coverage = coverage_indicators[sorted_indices]
    
    min_coverage = 1.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else n_samples
        bin_coverage = np.mean(sorted_coverage[start:end])
        min_coverage = min(min_coverage, bin_coverage)
    
    return min_coverage


def compute_all_metrics(
    coverage_indicators: np.ndarray,
    set_widths: np.ndarray,
    features: np.ndarray,
    target_coverage: float = 0.9,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute all metrics."""
    return {
        'marginal_coverage': marginal_coverage(coverage_indicators),
        'avg_set_width': average_set_width(set_widths),
        'msce': msce(coverage_indicators, features, n_bins),
        'wsc': wsc(coverage_indicators, features, n_bins),
        'coverage_deviation': abs(marginal_coverage(coverage_indicators) - target_coverage),
    }


def rolling_coverage(coverage_indicators: np.ndarray, window_size: int = 100) -> np.ndarray:
    """Compute rolling coverage over time."""
    n = len(coverage_indicators)
    rolling = np.zeros(n)
    for i in range(n):
        start = max(0, i - window_size + 1)
        rolling[i] = np.mean(coverage_indicators[start:i+1])
    return rolling


def convergence_time(coverage_indicators: np.ndarray, target: float = 0.9, 
                     threshold: float = 0.03, window_size: int = 100) -> int:
    """
    Compute convergence time: first window where deviation < threshold 
    and stays within tolerance.
    """
    n = len(coverage_indicators)
    rolling = rolling_coverage(coverage_indicators, window_size)
    
    for i in range(window_size, n):
        deviation = abs(rolling[i] - target)
        if deviation < threshold:
            # Check if it stays within tolerance for next 5 windows
            if i + 5 * window_size < n:
                next_deviations = [abs(rolling[j] - target) for j in range(i, i + 5 * window_size, window_size)]
                if all(d < 0.05 for d in next_deviations):
                    return i
            else:
                return i
    return n


def compute_coverage_by_density(
    coverage_indicators: np.ndarray,
    set_widths: np.ndarray,
    features: np.ndarray,
    k: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Compute coverage and width statistics stratified by data density.
    """
    from scipy.spatial import cKDTree
    
    # Compute local density using k-NN
    tree = cKDTree(features)
    distances, _ = tree.query(features, k=k+1)  # +1 because point itself is included
    avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
    
    # Lower distance = higher density
    # Bin into low, medium, high density
    low_threshold = np.percentile(avg_distances, 25)
    high_threshold = np.percentile(avg_distances, 75)
    
    low_density_mask = avg_distances >= high_threshold
    medium_density_mask = (avg_distances > low_threshold) & (avg_distances < high_threshold)
    high_density_mask = avg_distances <= low_threshold
    
    results = {}
    for name, mask in [('low', low_density_mask), ('medium', medium_density_mask), ('high', high_density_mask)]:
        if np.sum(mask) > 0:
            results[name] = {
                'coverage': np.mean(coverage_indicators[mask]),
                'avg_width': np.mean(set_widths[mask]),
                'n_samples': int(np.sum(mask))
            }
    
    return results
