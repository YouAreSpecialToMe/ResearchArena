from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def bootstrap_ci(values: Iterable[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def paired_bootstrap_ci(diffs: Iterable[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed)


def compute_auec(costs: list[float], entropies: list[float], initial_entropy: float, budget: float) -> float:
    if not costs:
        return 0.0
    xs = [0.0] + costs + [budget]
    ys = [initial_entropy] + entropies + [entropies[-1]]
    area = 0.0
    for left, right, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        area += 0.5 * (right - left) * (y0 + y1)
    return float(area)

