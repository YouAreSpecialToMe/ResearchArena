from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def mean_perturbed_reference_pearson(
    pred: np.ndarray, true: np.ndarray, ref: np.ndarray
) -> tuple[float, np.ndarray]:
    pred_resid = pred - ref[None, :]
    true_resid = true - ref[None, :]
    values = np.array(
        [safe_pearson(pred_resid[i], true_resid[i]) for i in range(true.shape[0])],
        dtype=np.float64,
    )
    return float(values.mean()), values


def rmse(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    per_pert = np.sqrt(((pred - true) ** 2).mean(axis=1))
    return float(np.sqrt(((pred - true) ** 2).mean())), per_pert.astype(np.float64)


def nearest_centroid_metrics(
    pred: np.ndarray, true: np.ndarray, labels: list[str]
) -> tuple[float, float]:
    sims = cosine_similarity(pred, true)
    order = np.argsort(-sims, axis=1)
    correct = 0
    ranks: list[int] = []
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for i, label in enumerate(labels):
        target_idx = label_to_idx[label]
        if order[i, 0] == target_idx:
            correct += 1
        rank = int(np.where(order[i] == target_idx)[0][0]) + 1
        ranks.append(rank)
    return correct / len(labels), float(np.median(ranks))

