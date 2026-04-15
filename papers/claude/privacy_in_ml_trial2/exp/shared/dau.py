import numpy as np
import torch


def compute_dau_weights(difficulty_scores, alpha=1.0):
    """Compute DAU per-sample weights: w(x) = 1 + alpha * (d(x) - d_mean) / d_std.
    Clamped to [0.1, 10.0].
    """
    d_mean = difficulty_scores.mean()
    d_std = difficulty_scores.std() + 1e-8
    weights = 1.0 + alpha * (difficulty_scores - d_mean) / d_std
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)


def compute_rum_groups(difficulty_scores, n_groups=3):
    """Partition into difficulty-based groups for RUM (Zhao et al., 2024)."""
    percentiles = np.percentile(difficulty_scores, [100/n_groups * i for i in range(1, n_groups)])
    groups = np.digitize(difficulty_scores, percentiles)
    return groups  # 0=easy, 1=medium, 2=hard
