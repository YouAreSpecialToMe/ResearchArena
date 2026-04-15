from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def top1_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float((scores.argmax(dim=1) == labels).float().mean().item() * 100.0)


def expected_calibration_error(scores: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = torch.softmax(scores.float(), dim=1)
    conf, preds = probs.max(dim=1)
    acc = preds.eq(labels)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = torch.tensor(0.0)
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (conf >= start) & (conf < end)
        if mask.any():
            bin_acc = acc[mask].float().mean()
            bin_conf = conf[mask].mean()
            ece += mask.float().mean() * torch.abs(bin_acc - bin_conf)
    return float(ece.item())


def summarize(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "std": math.nan}
    if array.size == 1:
        return {"mean": float(array[0]), "std": 0.0}
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def bootstrap_confidence_interval(
    values: Iterable[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": math.nan, "lower": math.nan, "upper": math.nan}
    if array.size == 1:
        scalar = float(array[0])
        return {"mean": scalar, "lower": scalar, "upper": scalar}
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, array.size), replace=True).mean(axis=1)
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1.0 - alpha / 2))
    return {"mean": float(array.mean()), "lower": lower, "upper": upper}


def cosine_margin(delta_by_family: dict[str, torch.Tensor], residual_by_family: dict[str, torch.Tensor]) -> dict[str, float]:
    families = list(delta_by_family.keys())
    matrix = np.zeros((len(families), len(families)), dtype=np.float32)
    for i, family_i in enumerate(families):
        dv = torch.nn.functional.normalize(delta_by_family[family_i].float(), dim=0)
        for j, family_j in enumerate(families):
            rv = torch.nn.functional.normalize(residual_by_family[family_j].float(), dim=0)
            matrix[i, j] = float(torch.dot(dv, rv).item())
    matched = float(np.mean(np.diag(matrix)))
    mismatched = float((matrix.sum() - np.trace(matrix)) / max(1, matrix.size - len(families)))
    return {
        "matched_mean": matched,
        "mismatched_mean": mismatched,
        "margin": matched - mismatched,
        "matrix": matrix.tolist(),
        "families": families,
    }
