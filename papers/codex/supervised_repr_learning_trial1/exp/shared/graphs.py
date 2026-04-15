from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from .utils import ARTIFACTS_ROOT, ensure_dir, load_json, save_json


def feature_cache_dir(dataset: str, shot: int, seed: int) -> Path:
    return ARTIFACTS_ROOT / "features" / dataset / str(shot) / f"seed_{seed}"


def graph_path(dataset: str, shot: int, seed: int, affinity_name: str = "prototype") -> Path:
    return ARTIFACTS_ROOT / "graphs" / dataset / str(shot) / f"seed_{seed}_{affinity_name}.json"


def compute_prototypes(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    prototypes = []
    for cls in range(num_classes):
        prototypes.append(features[labels == cls].mean(axis=0))
    return np.stack(prototypes)


def compute_affinity(features: np.ndarray, labels: np.ndarray, num_classes: int, variant: str = "prototype") -> np.ndarray:
    prototypes = compute_prototypes(features, labels, num_classes)
    proto_sim = cosine_similarity(prototypes)
    np.fill_diagonal(proto_sim, -1.0)
    if variant == "prototype":
        return proto_sim
    if variant == "prototype_plus_neighborhood":
        neighbors = NearestNeighbors(n_neighbors=min(8, len(features) - 1), metric="cosine")
        neighbors.fit(features)
        _, nn_idx = neighbors.kneighbors(features)
        class_neighbor_hist = np.zeros((num_classes, num_classes), dtype=np.float32)
        for idx, cls in enumerate(labels):
            neighbor_classes = labels[nn_idx[idx][1:]]
            for c in neighbor_classes:
                class_neighbor_hist[cls, c] += 1.0
        row_sum = class_neighbor_hist.sum(axis=1, keepdims=True) + 1e-8
        class_neighbor_hist /= row_sum
        return 0.7 * proto_sim + 0.3 * class_neighbor_hist
    raise ValueError(variant)


def bootstrap_graph(features: np.ndarray, labels: np.ndarray, num_classes: int, b: int = 20, k: int = 3, variant: str = "prototype") -> dict:
    affinity = compute_affinity(features, labels, num_classes, variant=variant)
    class_indices = [np.where(labels == cls)[0] for cls in range(num_classes)]
    boot_affinity = []
    topk_counts = np.zeros((num_classes, num_classes), dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(b):
        boot_feats, boot_labels = [], []
        for cls, idxs in enumerate(class_indices):
            sample = rng.choice(idxs, size=len(idxs), replace=True)
            boot_feats.append(features[sample])
            boot_labels.append(np.full(len(sample), cls))
        boot_feats = np.concatenate(boot_feats, axis=0)
        boot_labels = np.concatenate(boot_labels, axis=0)
        aff = compute_affinity(boot_feats, boot_labels, num_classes, variant=variant)
        boot_affinity.append(aff)
        topk = np.argsort(-aff, axis=1)[:, :k]
        for a in range(num_classes):
            topk_counts[a, topk[a]] += 1.0
    boot_affinity = np.stack(boot_affinity)
    retention = topk_counts / float(b)
    mean_aff = boot_affinity.mean(axis=0)
    std_aff = boot_affinity.std(axis=0)
    np.fill_diagonal(retention, 0.0)
    graph = {
        "affinity_variant": variant,
        "affinity": affinity.tolist(),
        "mean_affinity": mean_aff.tolist(),
        "std_affinity": std_aff.tolist(),
        "retention": retention.tolist(),
    }
    return graph


def graph_statistics(graph: dict, rho: float) -> dict:
    mean_aff = np.asarray(graph["mean_affinity"])
    retention = np.asarray(graph["retention"])
    mask = (retention >= rho) & (mean_aff > 0)
    np.fill_diagonal(mask, False)
    row_counts = mask.sum(axis=1)
    kept = retention[mask]
    return {
        "rho": rho,
        "graph_density": float(mask.mean()),
        "mean_retained_edge_stability": float(kept.mean()) if kept.size else 0.0,
        "isolated_class_fraction": float((row_counts == 0).mean()),
    }
