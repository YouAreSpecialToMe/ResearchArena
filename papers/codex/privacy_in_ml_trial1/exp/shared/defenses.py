from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from .metrics import entropy, nll_from_probs

try:
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    faiss = None
    from sklearn.neighbors import NearestNeighbors


@dataclass
class DefenseOutput:
    probs: np.ndarray
    lambda_values: np.ndarray
    risk_scores: np.ndarray
    local_risk: np.ndarray
    label_flip_rate: float
    mean_kl: float


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    temps = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0], dtype=np.float64)
    losses = []
    for t in temps:
        scaled = softmax(logits / t)
        losses.append(nll_from_probs(scaled, labels))
    return float(temps[int(np.argmin(losses))])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def add_entropy_noise(logits: np.ndarray, sigma_max: float, threshold: float, seed: int) -> np.ndarray:
    probs = softmax(logits)
    ent = entropy(probs) / np.log(probs.shape[1])
    rng = np.random.default_rng(seed)
    active = ent <= threshold
    noise_scale = np.where(active, sigma_max * (1.0 - ent / max(threshold, 1e-6)), 0.0)
    noisy_logits = logits + rng.normal(0.0, noise_scale[:, None], size=logits.shape)
    return softmax(noisy_logits)


def build_index(embeddings: np.ndarray):
    xb = embeddings.astype(np.float32)
    dim = xb.shape[1]
    if faiss is None:
        idx = NearestNeighbors(metric="euclidean")
        idx.fit(xb)
        return idx
    if len(xb) < 5000:
        idx = faiss.IndexFlatL2(dim)
        idx.add(xb)
        return idx
    nlist = min(1024, max(64, int(np.sqrt(len(xb)))))
    quantizer = faiss.IndexFlatL2(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    idx.train(xb)
    idx.add(xb)
    idx.nprobe = min(32, max(8, nlist // 16))
    return idx


def neighbor_weights(distances: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.exp(-distances / max(temperature, 1e-6))
    return scaled / np.clip(scaled.sum(axis=1, keepdims=True), 1e-12, None)


def compute_local_risk(
    query_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_probs: np.ndarray,
    train_risk: np.ndarray,
    k: int,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    index = build_index(train_embeddings)
    if faiss is None:
        distances, neighbors = index.kneighbors(query_embeddings.astype(np.float32), n_neighbors=k, return_distance=True)
        distances = distances**2
    else:
        distances, neighbors = index.search(query_embeddings.astype(np.float32), k)
    weights = neighbor_weights(distances, temperature)
    neighbor_post = (train_probs[neighbors] * weights[:, :, None]).sum(axis=1)
    local_risk = (train_risk[neighbors] * weights).sum(axis=1)
    return local_risk, neighbor_post


def defend_with_rain(
    raw_probs: np.ndarray,
    raw_logits: np.ndarray,
    labels: np.ndarray,
    query_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_probs: np.ndarray,
    train_risk: np.ndarray,
    k: int,
    neighbor_temp: float,
    tau: float,
    gamma: float,
    lambda_max: float,
    risk_mode: str = "r_loc",
    risk_model: dict | None = None,
    backtrack: bool = True,
) -> DefenseOutput:
    local_risk, neighbor_post = compute_local_risk(
        query_embeddings, train_embeddings, train_probs, train_risk, k, neighbor_temp
    )
    conf = raw_probs.max(axis=1)
    disagreement = 1.0 - neighbor_post[np.arange(len(labels)), raw_probs.argmax(axis=1)]
    if risk_mode == "full":
        if risk_model is None:
            raise ValueError("risk_model is required when risk_mode='full'")
        stacked = np.stack([local_risk, conf, disagreement], axis=1).astype(np.float64)
        mean = np.asarray(risk_model["feature_mean"], dtype=np.float64)
        std = np.clip(np.asarray(risk_model["feature_std"], dtype=np.float64), 1e-6, None)
        orient = np.asarray(risk_model["orientation"], dtype=np.float64)
        coeffs = np.asarray(risk_model["weights"], dtype=np.float64)
        logits_risk = ((stacked - mean) / std * orient) @ coeffs + float(risk_model["intercept"])
        risk = 1.0 / (1.0 + np.exp(-logits_risk))
    else:
        risk = np.clip(local_risk, 0.0, 1.0)
    lam = lambda_max * np.power(np.clip((risk - tau) / max(1e-6, 1.0 - tau), 0.0, 1.0), gamma)
    base_pred = raw_probs.argmax(axis=1)
    if backtrack:
        row = np.arange(len(raw_probs))
        y = base_pred
        base_gap = raw_probs[row, y][:, None] - raw_probs
        neigh_gap = neighbor_post[row, y][:, None] - neighbor_post
        denom = base_gap - neigh_gap
        valid = denom > 1e-12
        limits = np.full_like(base_gap, np.inf, dtype=np.float64)
        limits[valid] = base_gap[valid] / denom[valid]
        limits[row, y] = np.inf
        lam_cap = np.minimum(1.0, limits.min(axis=1))
        lam = np.minimum(lam, np.clip(lam_cap, 0.0, 1.0))
    defended = (1.0 - lam[:, None]) * raw_probs + lam[:, None] * neighbor_post
    flips = float((defended.argmax(axis=1) != base_pred).mean())
    kl = float(
        np.maximum(
            (raw_probs * (np.log(np.clip(raw_probs, 1e-12, 1.0)) - np.log(np.clip(defended, 1e-12, 1.0)))).sum(axis=1),
            0.0,
        ).mean()
    )
    return DefenseOutput(
        probs=defended,
        lambda_values=lam,
        risk_scores=risk,
        local_risk=local_risk,
        label_flip_rate=flips,
        mean_kl=kl,
    )


def defend_with_risk_temperature(
    raw_logits: np.ndarray,
    risk_scores: np.ndarray,
    tau: float,
    gamma: float,
    temp_max: float,
) -> np.ndarray:
    intensity = np.power(np.clip((risk_scores - tau) / max(1e-6, 1.0 - tau), 0.0, 1.0), gamma)
    temps = 1.0 + intensity * (temp_max - 1.0)
    return softmax(raw_logits / temps[:, None])
