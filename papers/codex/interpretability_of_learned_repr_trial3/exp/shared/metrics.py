from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .models import BlockSpec


def variance_explained(x: np.ndarray, recon: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64)
    recon64 = np.asarray(recon, dtype=np.float64)
    numerator = float(np.mean((x64 - recon64) ** 2))
    denominator = float(np.var(x64) + 1e-8)
    return float(max(0.0, 1.0 - numerator / denominator))


def realized_l0(z: np.ndarray) -> float:
    return float(np.mean((np.abs(z) > 1e-8).sum(axis=1)))


def block_change(z1: torch.Tensor, z2: torch.Tensor, block: slice) -> torch.Tensor:
    if block.stop <= block.start:
        return torch.zeros(z1.shape[0], device=z1.device)
    return (z1[:, block] - z2[:, block]).abs().mean(dim=1)


def compute_partition_metrics(z1: torch.Tensor, z2: torch.Tensor, spec: BlockSpec, target_factor: int) -> dict[str, torch.Tensor]:
    delta_inv = block_change(z1, z2, spec.inv_slice)
    delta_factors = [block_change(z1, z2, block) for block in spec.factor_slices]
    delta_res = block_change(z1, z2, spec.residual_slice)
    numerator = delta_factors[target_factor]
    denominator = delta_inv + delta_res
    for delta in delta_factors:
        denominator = denominator + delta
    tfcc = numerator / (denominator + 1e-8)
    stacked = torch.stack(delta_factors, dim=1)
    tba = (stacked.argmax(dim=1) == target_factor).float()
    return {
        "tfcc": tfcc,
        "tba": tba,
        "delta_inv": delta_inv,
        "delta_res": delta_res,
        "delta_factors": stacked,
    }


def bootstrap_ci(values: np.ndarray, samples: int = 200) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = []
    n = len(values)
    for _ in range(samples):
        choice = rng.integers(0, n, size=n)
        means.append(values[choice].mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
