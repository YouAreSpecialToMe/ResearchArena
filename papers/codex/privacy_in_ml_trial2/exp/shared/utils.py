from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_dump(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def now() -> float:
    return time.time()


def elapsed_minutes(start_time: float) -> float:
    return (time.time() - start_time) / 60.0


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        boots.append(values[idx].mean())
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_mean_and_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 123) -> dict:
    values = np.asarray(values, dtype=float)
    ci_low, ci_high = bootstrap_ci(values, n_boot=n_boot, seed=seed)
    std = 0.0 if len(values) <= 1 else float(values.std(ddof=1))
    return {
        "mean": float(values.mean()),
        "std": std,
        "ci95": [ci_low, ci_high],
    }


def safe_float(value: float | np.floating) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_num_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)
