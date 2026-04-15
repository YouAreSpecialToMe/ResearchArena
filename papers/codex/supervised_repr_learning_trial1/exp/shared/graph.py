from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader

from .data import build_datasets, get_dataset_bundle
from .models import create_model
from .utils import analysis_root, device, elapsed_minutes, ensure_dir, now, set_seed, write_json


@torch.no_grad()
def extract_features(dataset_name: str, output_path: Path, batch_size: int = 256, checkpoint_path: Path | None = None) -> Dict:
    bundle = get_dataset_bundle(dataset_name)
    datasets = build_datasets(bundle)
    loader = DataLoader(datasets["train_eval"], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    model = create_model(bundle.num_coarse_classes).to(device()).eval()
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device())
        model.load_state_dict(state["model"], strict=False)
    feats, sample_ids, coarse_labels, fine_labels = [], [], [], []
    start = now()
    for batch in loader:
        images = batch["image"].to(device(), non_blocking=True)
        features = model.forward_backbone(images).cpu()
        feats.append(features)
        sample_ids.append(batch["sample_id"])
        coarse_labels.append(batch["coarse_label"])
        fine_labels.append(batch["fine_label"])
    payload = {
        "features": torch.cat(feats).numpy().astype(np.float32),
        "sample_ids": torch.cat(sample_ids).numpy(),
        "coarse_labels": torch.cat(coarse_labels).numpy(),
        "fine_labels": torch.cat(fine_labels).numpy(),
        "runtime_minutes": elapsed_minutes(start),
    }
    ensure_dir(output_path.parent)
    np.savez(output_path, **payload)
    return payload


def _knn_within_group(features: np.ndarray, indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sub = features[indices]
    sims = 1.0 - pairwise_distances(sub, metric="cosine")
    np.fill_diagonal(sims, -np.inf)
    topk = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    topk_sims = np.take_along_axis(sims, topk, axis=1)
    order = np.argsort(-topk_sims, axis=1)
    topk = np.take_along_axis(topk, order, axis=1)
    topk_sims = np.take_along_axis(topk_sims, order, axis=1)
    return indices[topk], topk_sims


def build_graph_from_features(
    dataset_name: str,
    feature_npz: Path,
    output_path: Path,
    graph_type: str = "pretrained",
    k: int = 10,
    tau_n: float = 0.07,
) -> Dict:
    data = np.load(feature_npz)
    feats = data["features"]
    sample_ids = data["sample_ids"]
    coarse_labels = data["coarse_labels"]
    fine_labels = data["fine_labels"]
    start = now()

    if graph_type == "random":
        rng = np.random.default_rng(0)
        neigh_ids = np.zeros((len(sample_ids), k), dtype=np.int64)
        neigh_sims = np.zeros((len(sample_ids), k), dtype=np.float32)
        for coarse in np.unique(coarse_labels):
            idx = np.where(coarse_labels == coarse)[0]
            for row in idx:
                candidates = idx[idx != row]
                picked = rng.choice(candidates, size=k, replace=False)
                neigh_ids[row] = sample_ids[picked]
                neigh_sims[row] = feats[row] @ feats[picked].T
    else:
        neigh_ids = np.zeros((len(sample_ids), k), dtype=np.int64)
        neigh_sims = np.zeros((len(sample_ids), k), dtype=np.float32)
        for coarse in np.unique(coarse_labels):
            idx = np.where(coarse_labels == coarse)[0]
            ids, sims = _knn_within_group(feats, idx, k)
            neigh_ids[idx] = sample_ids[ids]
            neigh_sims[idx] = sims

    teacher_probs = torch.softmax(torch.tensor(neigh_sims) / tau_n, dim=-1).numpy().astype(np.float32)
    payload = {
        "graph_type": graph_type,
        "k": k,
        "tau_n": tau_n,
        "features": feats,
        "sample_ids": sample_ids,
        "coarse_labels": coarse_labels,
        "fine_labels": fine_labels,
        "neighbor_ids": neigh_ids,
        "neighbor_sims": neigh_sims,
        "teacher_probs": teacher_probs,
        "graph_build_minutes": elapsed_minutes(start),
    }
    ensure_dir(output_path.parent)
    np.savez(output_path, **payload)
    return payload


def graph_metrics(graph_npz: Path, output_json: Path) -> Dict:
    data = np.load(graph_npz)
    fine_by_id = {int(i): int(f) for i, f in zip(data["sample_ids"], data["fine_labels"])}
    coarse_by_id = {int(i): int(c) for i, c in zip(data["sample_ids"], data["coarse_labels"])}
    feat_by_id = {int(i): f for i, f in zip(data["sample_ids"], data["features"])}
    sample_ids = data["sample_ids"]
    neigh = data["neighbor_ids"]
    sims = data["neighbor_sims"]
    purities = []
    per_coarse = defaultdict(list)
    density_gap = []
    rng = np.random.default_rng(0)
    same_coarse_ids = defaultdict(list)
    for sid in sample_ids:
        same_coarse_ids[coarse_by_id[int(sid)]].append(int(sid))
    for i, sid in enumerate(sample_ids):
        sid = int(sid)
        fine = fine_by_id[sid]
        neigh_ids = [int(x) for x in neigh[i]]
        purity = float(np.mean([fine_by_id[n] == fine for n in neigh_ids]))
        purities.append(purity)
        per_coarse[coarse_by_id[sid]].append(purity)
        random_ids = rng.choice([x for x in same_coarse_ids[coarse_by_id[sid]] if x != sid], size=len(neigh_ids), replace=False)
        random_sims = [float(np.dot(feat_by_id[sid], feat_by_id[int(r)])) for r in random_ids]
        density_gap.append(float(np.mean(sims[i]) - np.mean(random_sims)))
    result = {
        "overall_purity_at_10": float(np.mean(purities)),
        "per_coarse_purity_at_10": {str(k): float(np.mean(v)) for k, v in per_coarse.items()},
        "mean_neighbor_cosine_similarity": float(np.mean(sims)),
        "density_gap_top10_vs_random": float(np.mean(density_gap)),
        "graph_type": str(data["graph_type"]),
        "k": int(data["k"]),
    }
    write_json(output_json, result)
    return result


def teacher_graph_paths(dataset_name: str, graph_type: str = "pretrained", k: int = 10) -> Dict[str, Path]:
    root = analysis_root() / dataset_name
    ensure_dir(root)
    feat_name = "teacher_features.npz" if graph_type == "pretrained" else f"teacher_features_{graph_type}.npz"
    feat = root / feat_name
    graph = root / f"teacher_graph_{graph_type}_k{k}.npz"
    metrics = root / f"teacher_graph_{graph_type}_k{k}.json"
    return {"features": feat, "graph": graph, "metrics": metrics}
