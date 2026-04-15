from __future__ import annotations

import itertools
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import (
    ACCEPTANCE_RATES,
    RUN_VERSION,
    SELECTIVE_GRID,
    ensure_dir,
    get_process_peak_rss_bytes,
    get_system_info,
    hash_vector,
    infer_device,
    set_deterministic,
    write_json,
)


DATASET_FILES = {
    "Adamson": "Adamson.h5ad",
    "Norman": "Norman.h5ad",
    "Replogle": "Replogle_exp7.h5ad",
}


@dataclass
class SeedBundle:
    dataset: str
    seed: int
    genes: np.ndarray
    retained_genes: np.ndarray
    responsive_panel: np.ndarray
    top_de_genes: np.ndarray
    perturbations: np.ndarray
    targets: list[list[str]]
    split_labels: np.ndarray
    hardness_labels: np.ndarray
    novelty_quartiles: np.ndarray
    pseudobulk: np.ndarray
    control_pseudobulk: np.ndarray
    descriptors: dict[str, np.ndarray]
    pathways: dict[str, np.ndarray]
    k_neighbors: int
    target_svd_rank: int | None
    baseline_simple_cfg: dict[str, Any]
    baseline_ridge_cfg: dict[str, Any]
    split_config: dict[str, Any]
    split_memberships: dict[str, list[str]]
    preprocessing_stats: dict[str, Any]


class ResidualMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _read_categorical(group: h5py.Group) -> np.ndarray:
    cats = np.asarray(group["categories"][:]).astype(str)
    codes = np.asarray(group["codes"][:]).astype(int)
    return cats[codes]


def _read_obs_columns(handle: h5py.File) -> dict[str, np.ndarray]:
    out = {}
    for key in handle["obs"].keys():
        node = handle["obs"][key]
        if isinstance(node, h5py.Group) and "codes" in node and "categories" in node:
            out[key] = _read_categorical(node)
        elif isinstance(node, h5py.Dataset):
            out[key] = np.asarray(node[:]).astype(str)
    return out


def _read_var_names(handle: h5py.File, dataset: str) -> np.ndarray:
    var = handle["var"]
    candidates = []
    if dataset == "Adamson":
        candidates = ["Gene_symbol", "_index", "gene_name"]
    elif dataset == "Norman":
        candidates = ["gene_name", "_index", "Gene_symbol"]
    else:
        candidates = ["_index", "gene_name", "Gene_symbol"]
    for key in candidates:
        if key in var:
            node = var[key]
            if isinstance(node, h5py.Group):
                return _read_categorical(node)
            return np.asarray(node[:]).astype(str)
    raise KeyError(f"Could not infer gene names for {dataset}")


def load_raw_dataset(dataset: str, root: str | Path = ".") -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    path = Path(root) / "data" / "raw" / DATASET_FILES[dataset]
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["X"][:], dtype=np.float32)
        obs = _read_obs_columns(handle)
        genes = _read_var_names(handle, dataset)
    return x, obs, genes


def parse_targets(dataset: str, perturbation: str) -> list[str]:
    if perturbation == "control":
        return []
    if dataset == "Norman":
        return [token for token in perturbation.split("+") if token]
    return [perturbation]


def _split_norman(perturbations: list[str], seed: int, route_ratio: float = 0.2, cal_ratio: float = 0.1) -> dict[str, list[str]]:
    singles = [p for p in perturbations if p != "control" and "+" not in p]
    doubles = [p for p in perturbations if "+" in p]
    rng = np.random.default_rng(seed)
    component_genes = np.array(sorted({token for perturbation in doubles for token in perturbation.split("+")}))
    rng.shuffle(component_genes)
    n_seen_genes = max(1, int(round(0.6 * len(component_genes))))
    seen_genes = set(component_genes[:n_seen_genes].tolist())
    train_singles = [p for p in singles if p in seen_genes]
    eval_singles = [p for p in singles if p not in seen_genes]

    hardness_groups: dict[str, list[str]] = {"0-seen": [], "1-seen": [], "2-seen": []}
    for perturbation in doubles:
        seen = sum(token in seen_genes for token in perturbation.split("+"))
        hardness_groups[f"{seen}-seen"].append(perturbation)

    train_doubles = []
    eval_doubles = []
    for label, group in hardness_groups.items():
        group = np.array(sorted(group))
        rng.shuffle(group)
        if label == "2-seen":
            n_train_group = int(round(0.6 * len(group)))
            train_doubles.extend(group[:n_train_group].tolist())
            eval_doubles.extend(group[n_train_group:].tolist())
        else:
            eval_doubles.extend(group.tolist())

    route_items: list[str] = []
    cal_items: list[str] = []
    test_items: list[str] = []
    eval_groups = {"singles": eval_singles, **{label: items for label, items in hardness_groups.items() if label != "2-seen"}}
    eval_groups["2-seen"] = [p for p in eval_doubles if sum(token in seen_genes for token in p.split("+")) == 2]
    denom = route_ratio + cal_ratio + 0.1
    for group in eval_groups.values():
        group = np.array(sorted(group))
        if len(group) == 0:
            continue
        rng.shuffle(group)
        n_route = int(round(route_ratio * len(group) / denom))
        n_cal = int(round(cal_ratio * len(group) / denom))
        n_route = min(n_route, max(0, len(group) - 2))
        n_cal = min(n_cal, max(0, len(group) - n_route - 1))
        route_items.extend(group[:n_route].tolist())
        cal_items.extend(group[n_route:n_route + n_cal].tolist())
        test_items.extend(group[n_route + n_cal:].tolist())
    splits = {
        "train": sorted(train_singles + train_doubles),
        "route_dev": sorted(route_items),
        "calibration": sorted(cal_items),
        "test": sorted(test_items),
    }
    return splits


def _split_standard(perturbations: list[str], seed: int, route_ratio: float = 0.2, cal_ratio: float = 0.1) -> dict[str, list[str]]:
    perts = np.array(sorted([p for p in perturbations if p != "control"]))
    rng = np.random.default_rng(seed)
    rng.shuffle(perts)
    n = len(perts)
    n_train = int(round(0.6 * n))
    n_route = int(round(route_ratio * n))
    n_cal = int(round(cal_ratio * n))
    return {
        "train": perts[:n_train].tolist(),
        "route_dev": perts[n_train:n_train + n_route].tolist(),
        "calibration": perts[n_train + n_route:n_train + n_route + n_cal].tolist(),
        "test": perts[n_train + n_route + n_cal:].tolist(),
    }


def _make_pathways(train_delta: np.ndarray, genes: np.ndarray) -> dict[str, np.ndarray]:
    n_clusters = min(8, max(2, train_delta.shape[0] // 5))
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(train_delta.T)
    pathways = {}
    for cluster in range(n_clusters):
        mask = labels == cluster
        if mask.sum() >= 5:
            pathways[f"module_{cluster}"] = genes[mask]
    if not pathways:
        pathways["module_0"] = genes[: min(50, len(genes))]
    return pathways


def build_seed_bundle(dataset: str, seed: int, root: str | Path = ".", route_ratio: float = 0.2, cal_ratio: float = 0.1) -> SeedBundle:
    ratio_tag = f"r{int(round(route_ratio * 100)):02d}_c{int(round(cal_ratio * 100)):02d}"
    cache_path = Path(root) / "data" / "processed" / dataset / f"{RUN_VERSION}_{ratio_tag}_seed_{seed}.joblib"
    if cache_path.exists():
        bundle = joblib.load(cache_path)
        write_feature_artifacts(bundle, root=root)
        return bundle

    ensure_dir(cache_path.parent)
    x, obs, genes = load_raw_dataset(dataset, root=root)
    perturbations = obs["perturbation"]
    gene_labels = obs.get("gene", perturbations)
    control_mask = perturbations == "control"
    non_control = sorted(set(perturbations[~control_mask]))

    min_cells = 30 if dataset == "Norman" else 40
    counts = pd.Series(perturbations).value_counts()
    stable_non_control = [p for p in non_control if counts[p] >= min_cells]
    if dataset == "Norman":
        stable_singles = [p for p in stable_non_control if "+" not in p]
        stable_doubles = [p for p in stable_non_control if "+" in p]
        stable_non_control = stable_singles + stable_doubles
        split_map = _split_norman(stable_non_control, seed, route_ratio=route_ratio, cal_ratio=cal_ratio)
    else:
        split_map = _split_standard(stable_non_control, seed, route_ratio=route_ratio, cal_ratio=cal_ratio)

    rng = np.random.default_rng(seed)
    control_indices = np.where(control_mask)[0]
    rng.shuffle(control_indices)
    n_control_train = max(1, int(0.6 * len(control_indices)))
    train_control_idx = control_indices[:n_control_train]
    control_pseudobulk = x[train_control_idx].mean(axis=0)

    train_perts = split_map["train"]
    train_mask = np.isin(perturbations, train_perts) | np.isin(np.arange(len(perturbations)), train_control_idx)
    train_x = x[train_mask]
    train_var = train_x.var(axis=0)
    hvg_idx = np.argsort(-train_var)[:1500]
    target_genes = sorted({g for p in stable_non_control for g in parse_targets(dataset, p) if g in set(genes)})
    target_idx = np.array([np.where(genes == g)[0][0] for g in target_genes], dtype=int) if target_genes else np.array([], dtype=int)
    retained_idx = np.unique(np.concatenate([hvg_idx, target_idx])).astype(int)
    retained_genes = genes[retained_idx]

    all_perturbations = np.array(stable_non_control)
    pseudo = []
    targets = []
    for p in all_perturbations:
        pseudo.append(x[perturbations == p][:, retained_idx].mean(axis=0))
        targets.append(parse_targets(dataset, p))
    pseudobulk = np.asarray(pseudo, dtype=np.float32)
    control_pseudobulk = control_pseudobulk[retained_idx].astype(np.float32)

    train_indices = np.where(np.isin(all_perturbations, split_map["train"]))[0]
    train_delta = pseudobulk[train_indices] - control_pseudobulk

    gene_counter: Counter[str] = Counter()
    for row in train_delta:
        top = np.argsort(-np.abs(row))[:80]
        gene_counter.update(retained_genes[top].tolist())
    panel = [gene for gene, count in gene_counter.items() if count >= 2]
    panel = sorted(panel, key=lambda gene: (-gene_counter[gene], gene))[:256]
    if not panel:
        panel = retained_genes[np.argsort(-np.abs(train_delta).mean(axis=0))[:256]].tolist()
    responsive_panel = np.asarray(panel)

    top_de_idx = np.argsort(-np.abs(train_delta).mean(axis=0))[: min(200, len(retained_genes))]
    top_de_genes = retained_genes[top_de_idx]

    singles_seen = set()
    for perturbation, toks in zip(all_perturbations, targets):
        if len(toks) == 1 and perturbation in split_map["train"]:
            singles_seen.add(toks[0])
    hardness = []
    for perturbation, toks in zip(all_perturbations, targets):
        if dataset != "Norman" or "+" not in perturbation:
            hardness.append("novelty")
            continue
        seen = sum(token in singles_seen for token in toks)
        hardness.append(f"{seen}-seen")
    hardness_labels = np.asarray(hardness)

    desc_rows = []
    mean_rows = []
    diff_rows = []
    graph = defaultdict(set)
    for toks in targets:
        if len(toks) == 2:
            graph[toks[0]].add(toks[1])
            graph[toks[1]].add(toks[0])
    for toks in targets:
        if not toks:
            a = np.zeros(64, dtype=np.float32)
            b = np.zeros(64, dtype=np.float32)
            base = np.concatenate([a, b]).astype(np.float32)
            mean = np.zeros(64, dtype=np.float32)
            diff = np.zeros(64, dtype=np.float32)
            flags = np.zeros(2, dtype=np.float32)
            net = np.zeros(4, dtype=np.float32)
        elif len(toks) == 1:
            a = hash_vector([toks[0]])
            b = np.zeros(64, dtype=np.float32)
            base = np.concatenate([a, b]).astype(np.float32)
            mean = a.copy()
            diff = np.abs(a - b)
            flags = np.array([float(toks[0] in singles_seen), 0.0], dtype=np.float32)
            net = np.array([len(graph[toks[0]]), 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            a = hash_vector([toks[0]])
            b = hash_vector([toks[1]])
            base = np.concatenate([a, b]).astype(np.float32)
            mean = (a + b) / 2.0
            diff = np.abs(a - b)
            flags = np.array([float(toks[0] in singles_seen), float(toks[1] in singles_seen)], dtype=np.float32)
            one_hop = len(graph[toks[0]]) + len(graph[toks[1]])
            two_hop = len(set().union(*(graph[n] for n in graph[toks[0]] | graph[toks[1]]))) if graph[toks[0]] or graph[toks[1]] else 0
            mean_deg = float(np.mean([len(graph[toks[0]]), len(graph[toks[1]])]))
            net = np.array([one_hop, mean_deg, float(flags.sum()), float(two_hop)], dtype=np.float32)
        if len(toks) <= 1:
            concat = np.concatenate([base, mean, diff, flags, net]).astype(np.float32)
        else:
            concat = np.concatenate([base, mean, diff, flags, net]).astype(np.float32)
        desc_rows.append(concat)
        mean_rows.append(mean.astype(np.float32))
        diff_rows.append(diff.astype(np.float32))
    desc = np.asarray(desc_rows, dtype=np.float32)
    view_mean = np.asarray(mean_rows, dtype=np.float32)
    view_diff = np.asarray(diff_rows, dtype=np.float32)

    scaler = StandardScaler()
    continuous_idx = np.arange(desc.shape[1] - 2, dtype=int)
    desc_scaled = desc.copy()
    desc_scaled[:, continuous_idx] = scaler.fit_transform(desc[:, continuous_idx])
    descriptors = {
        "concat": desc_scaled.astype(np.float32),
        "mean": view_mean.astype(np.float32),
        "diff": view_diff.astype(np.float32),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }

    train_desc = desc_scaled[train_indices]
    sim = cosine_similarity(desc_scaled, train_desc)
    novelty = 1.0 - sim.max(axis=1)
    if dataset == "Norman":
        novelty_quartiles = np.full(len(all_perturbations), "NA", dtype=object)
    else:
        train_novelty = novelty[train_indices]
        bins = np.quantile(train_novelty, [0.25, 0.5, 0.75])
        labels = []
        for value in novelty:
            if value <= bins[0]:
                labels.append("q1")
            elif value <= bins[1]:
                labels.append("q2")
            elif value <= bins[2]:
                labels.append("q3")
            else:
                labels.append("q4")
        novelty_quartiles = np.asarray(labels)

    panel_idx = np.array([np.where(retained_genes == gene)[0][0] for gene in responsive_panel], dtype=int)
    simple_cfg = tune_simple_baseline(dataset, pseudobulk, control_pseudobulk, desc_scaled, all_perturbations, split_map, panel_idx)
    ridge_cfg = tune_ridge_baseline(pseudobulk, desc_scaled, all_perturbations, split_map)
    paths = _make_pathways(train_delta, retained_genes)

    split_labels = np.array(
        [
            "train" if p in split_map["train"] else
            "route_dev" if p in split_map["route_dev"] else
            "calibration" if p in split_map["calibration"] else
            "test"
            for p in all_perturbations
        ]
    )
    bundle = SeedBundle(
        dataset=dataset,
        seed=seed,
        genes=genes,
        retained_genes=retained_genes,
        responsive_panel=responsive_panel,
        top_de_genes=top_de_genes,
        perturbations=all_perturbations,
        targets=targets,
        split_labels=split_labels,
        hardness_labels=hardness_labels,
        novelty_quartiles=novelty_quartiles,
        pseudobulk=pseudobulk.astype(np.float32),
        control_pseudobulk=control_pseudobulk.astype(np.float32),
        descriptors=descriptors,
        pathways=paths,
        k_neighbors=int(simple_cfg["k"] if simple_cfg.get("k") is not None else 3),
        target_svd_rank=ridge_cfg["rank"],
        baseline_simple_cfg=simple_cfg,
        baseline_ridge_cfg=ridge_cfg,
        split_config={"route_ratio": route_ratio, "cal_ratio": cal_ratio, "version": RUN_VERSION},
        split_memberships={key: sorted(value) for key, value in split_map.items()},
        preprocessing_stats={
            "min_cells_per_perturbation": int(min_cells),
            "n_control_train_cells": int(len(train_control_idx)),
            "train_control_cell_indices": train_control_idx.astype(int).tolist(),
            "hvg_indices": hvg_idx.astype(int).tolist(),
            "target_gene_list": list(target_genes),
            "retained_genes": retained_genes.astype(str).tolist(),
            "responsive_panel": responsive_panel.astype(str).tolist(),
            "top_de_genes": top_de_genes.astype(str).tolist(),
            "descriptor_scaler_mean": scaler.mean_.astype(float).tolist(),
            "descriptor_scaler_scale": scaler.scale_.astype(float).tolist(),
        },
    )
    joblib.dump(bundle, cache_path)
    write_feature_artifacts(bundle, root=root)
    return bundle


def write_feature_artifacts(bundle: SeedBundle, root: str | Path = ".") -> None:
    base_dir = ensure_dir(Path(root) / "features" / bundle.dataset / f"seed_{bundle.seed}")
    ratio_tag = f"r{int(round(bundle.split_config['route_ratio'] * 100)):02d}_c{int(round(bundle.split_config['cal_ratio'] * 100)):02d}"
    is_canonical = math.isclose(bundle.split_config["route_ratio"], 0.2) and math.isclose(bundle.split_config["cal_ratio"], 0.1)
    out_dir = base_dir if is_canonical else ensure_dir(base_dir / f"variant_{ratio_tag}")
    for split in ["train", "route_dev", "calibration", "test"]:
        mask = bundle.split_labels == split
        np.save(out_dir / f"{split}_concat.npy", bundle.descriptors["concat"][mask])
        np.save(out_dir / f"{split}_mean.npy", bundle.descriptors["mean"][mask])
        np.save(out_dir / f"{split}_diff.npy", bundle.descriptors["diff"][mask])
        pd.DataFrame(
            {
                "perturbation": bundle.perturbations[mask],
                "split": bundle.split_labels[mask],
                "novelty": bundle.novelty_quartiles[mask],
                "hardness": bundle.hardness_labels[mask],
            }
        ).to_csv(out_dir / f"{split}_meta.csv", index=False)
    write_json(
        out_dir / "preprocessing_stats.json",
        {
            "run_version": RUN_VERSION,
            "dataset": bundle.dataset,
            "seed": bundle.seed,
            "artifact_variant": ratio_tag,
            "split_config": bundle.split_config,
            "split_memberships": bundle.split_memberships,
            "preprocessing_stats": bundle.preprocessing_stats,
            "feature_shapes": {
                split: {
                    "concat": list(bundle.descriptors["concat"][bundle.split_labels == split].shape),
                    "mean": list(bundle.descriptors["mean"][bundle.split_labels == split].shape),
                    "diff": list(bundle.descriptors["diff"][bundle.split_labels == split].shape),
                }
                for split in ["train", "route_dev", "calibration", "test"]
            },
        },
    )


def tune_simple_baseline(dataset: str, y: np.ndarray, control: np.ndarray, desc: np.ndarray, perturbations: np.ndarray, split_map: dict[str, list[str]], panel_idx: np.ndarray) -> dict[str, Any]:
    train_mask = np.isin(perturbations, split_map["train"])
    dev_mask = np.isin(perturbations, split_map["route_dev"])
    train_y, dev_y = y[train_mask], y[dev_mask]
    train_desc, dev_desc = desc[train_mask], desc[dev_mask]
    best = {"rmse": math.inf, "k": 3, "tau": 1.0}
    if dataset == "Norman":
        singles = {}
        train_perts = perturbations[train_mask]
        for perturbation, row in zip(train_perts, train_y):
            targets = parse_targets(dataset, perturbation)
            if len(targets) == 1:
                singles[targets[0]] = row - control
        pred = []
        missing_components = []
        for perturbation in perturbations[dev_mask]:
            targets = parse_targets(dataset, perturbation)
            effect = np.zeros_like(control)
            missing = 0
            for token in targets:
                if token in singles:
                    effect += singles[token]
                else:
                    missing += 1
            pred.append(control + effect)
            missing_components.append(missing)
        pred = np.asarray(pred, dtype=np.float32)
        rmse = np.sqrt(np.mean((pred - dev_y) ** 2, axis=1)).mean()
        best = {
            "mode": "additive_single_effect",
            "rmse": float(rmse),
            "k": None,
            "tau": None,
            "route_dev_missing_component_mean": float(np.mean(missing_components)) if missing_components else 0.0,
            "n_train_single_effects": int(len(singles)),
        }
        return best
    train_delta = train_y - control
    sim = cosine_similarity(dev_desc, train_desc)
    for k, tau in itertools.product([3, 5, 8], [0.5, 1.0, 2.0]):
        pred = []
        for row in sim:
            nn = np.argsort(-row)[:k]
            weights = np.exp(row[nn] / tau)
            weights /= weights.sum() + 1e-8
            pred.append(control + (weights[:, None] * train_delta[nn]).sum(axis=0))
        pred = np.asarray(pred)
        rmse = np.sqrt(np.mean((pred - dev_y) ** 2, axis=1)).mean()
        if rmse < best["rmse"]:
            best = {"rmse": float(rmse), "k": int(k), "tau": float(tau)}
    return best


def tune_ridge_baseline(y: np.ndarray, desc: np.ndarray, perturbations: np.ndarray, split_map: dict[str, list[str]]) -> dict[str, Any]:
    train_mask = np.isin(perturbations, split_map["train"])
    dev_mask = np.isin(perturbations, split_map["route_dev"])
    x_train, x_dev = desc[train_mask], desc[dev_mask]
    y_train, y_dev = y[train_mask], y[dev_mask]
    best = {"rmse": math.inf, "alpha": 1.0, "rank": None}
    for alpha, rank in itertools.product([0.1, 1.0, 10.0, 100.0], [None, 64, 128]):
        if rank is None or rank >= y_train.shape[1]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(x_train, y_train)
            pred = ridge.predict(x_dev)
        else:
            svd = TruncatedSVD(n_components=rank, random_state=0)
            z_train = svd.fit_transform(y_train)
            ridge = Ridge(alpha=alpha)
            ridge.fit(x_train, z_train)
            pred = ridge.predict(x_dev) @ svd.components_
        rmse = np.sqrt(np.mean((pred - y_dev) ** 2, axis=1)).mean()
        if rmse < best["rmse"]:
            best = {"rmse": float(rmse), "alpha": float(alpha), "rank": rank}
    return best


def simple_baseline_predictions(bundle: SeedBundle) -> np.ndarray:
    if bundle.dataset == "Norman":
        train_mask = bundle.split_labels == "train"
        singles = {}
        for perturbation, target, row in zip(bundle.perturbations[train_mask], np.array(bundle.targets, dtype=object)[train_mask], bundle.pseudobulk[train_mask]):
            if len(target) == 1:
                singles[target[0]] = row - bundle.control_pseudobulk
        pred = []
        for perturbation in bundle.perturbations:
            toks = parse_targets(bundle.dataset, perturbation)
            effect = np.zeros_like(bundle.control_pseudobulk)
            for tok in toks:
                effect += singles.get(tok, 0.0)
            pred.append(bundle.control_pseudobulk + effect)
        return np.asarray(pred)

    train_mask = bundle.split_labels == "train"
    train_desc = bundle.descriptors["concat"][train_mask]
    train_delta = bundle.pseudobulk[train_mask] - bundle.control_pseudobulk
    sim = cosine_similarity(bundle.descriptors["concat"], train_desc)
    pred = []
    k = bundle.baseline_simple_cfg["k"]
    tau = bundle.baseline_simple_cfg["tau"]
    for row in sim:
        nn = np.argsort(-row)[:k]
        weights = np.exp(row[nn] / tau)
        weights /= weights.sum() + 1e-8
        pred.append(bundle.control_pseudobulk + (weights[:, None] * train_delta[nn]).sum(axis=0))
    return np.asarray(pred)


def ridge_baseline_predictions(bundle: SeedBundle) -> np.ndarray:
    train_mask = bundle.split_labels == "train"
    x_train = bundle.descriptors["concat"][train_mask]
    y_train = bundle.pseudobulk[train_mask]
    x_all = bundle.descriptors["concat"]
    alpha = bundle.baseline_ridge_cfg["alpha"]
    rank = bundle.baseline_ridge_cfg["rank"]
    if rank is None or rank >= y_train.shape[1]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_train, y_train)
        return ridge.predict(x_all)
    svd = TruncatedSVD(n_components=rank, random_state=0)
    z_train = svd.fit_transform(y_train)
    ridge = Ridge(alpha=alpha)
    ridge.fit(x_train, z_train)
    return ridge.predict(x_all) @ svd.components_


def panel_indices(bundle: SeedBundle) -> np.ndarray:
    return np.array([np.where(bundle.retained_genes == gene)[0][0] for gene in bundle.responsive_panel], dtype=int)


def top_de_indices(bundle: SeedBundle) -> np.ndarray:
    return np.array([np.where(bundle.retained_genes == gene)[0][0] for gene in bundle.top_de_genes], dtype=int)


def choose_frozen_baseline(bundle: SeedBundle, log_fn: callable | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.time()
    cpu_start = time.process_time()
    simple_pred = simple_baseline_predictions(bundle)
    ridge_pred = ridge_baseline_predictions(bundle)
    dev_mask = bundle.split_labels == "route_dev"
    dev_y = bundle.pseudobulk[dev_mask]
    simple_rmse = np.sqrt(np.mean((simple_pred[dev_mask] - dev_y) ** 2, axis=1))
    ridge_rmse = np.sqrt(np.mean((ridge_pred[dev_mask] - dev_y) ** 2, axis=1))
    top_idx = top_de_indices(bundle)
    simple_top = np.sqrt(np.mean((simple_pred[dev_mask][:, top_idx] - dev_y[:, top_idx]) ** 2, axis=1)).mean()
    ridge_top = np.sqrt(np.mean((ridge_pred[dev_mask][:, top_idx] - dev_y[:, top_idx]) ** 2, axis=1)).mean()
    simple_params = int(simple_pred.shape[1] * bundle.descriptors["concat"].shape[1] + simple_pred.shape[1])
    ridge_params = int(ridge_pred.shape[1] * bundle.descriptors["concat"].shape[1] + ridge_pred.shape[1])
    info = {
        "simple_rmse": float(simple_rmse.mean()),
        "ridge_rmse": float(ridge_rmse.mean()),
        "simple_top_de_rmse": float(simple_top),
        "ridge_top_de_rmse": float(ridge_top),
        "runtime_seconds": float(time.time() - start),
        "cpu_seconds": float(time.process_time() - cpu_start),
        "peak_rss_bytes": get_process_peak_rss_bytes(),
        "simple_parameter_count": simple_params,
        "ridge_parameter_count": ridge_params,
        "simple_config": bundle.baseline_simple_cfg,
        "ridge_config": bundle.baseline_ridge_cfg,
    }
    if log_fn is not None:
        log_fn(
            "baseline_selection "
            f"simple_rmse={info['simple_rmse']:.6f} ridge_rmse={info['ridge_rmse']:.6f} "
            f"simple_top_de_rmse={info['simple_top_de_rmse']:.6f} ridge_top_de_rmse={info['ridge_top_de_rmse']:.6f}"
        )
    if ridge_rmse.mean() < simple_rmse.mean() or (math.isclose(ridge_rmse.mean(), simple_rmse.mean()) and ridge_top <= simple_top):
        info["selected"] = "ridge"
        info["selected_parameter_count"] = ridge_params
        if log_fn is not None:
            log_fn(f"baseline_selected model=ridge config={bundle.baseline_ridge_cfg}")
        return ridge_pred.astype(np.float32), info
    info["selected"] = "simple"
    info["selected_parameter_count"] = simple_params
    if log_fn is not None:
        log_fn(f"baseline_selected model=simple config={bundle.baseline_simple_cfg}")
    return simple_pred.astype(np.float32), info


def _build_retrieval_features(bundle: SeedBundle, residual_targets: np.ndarray, use_retrieval: bool = True) -> dict[str, np.ndarray]:
    panel_idx = panel_indices(bundle)
    views = {"concat": bundle.descriptors["concat"], "mean": bundle.descriptors["mean"], "diff": bundle.descriptors["diff"]}
    train_mask = bundle.split_labels == "train"
    outputs = {}
    pooled_proto = []
    best_sim = []
    entropy = []
    disagreement = []
    density = []
    for view_name, view in views.items():
        train_view = view[train_mask]
        sim = cosine_similarity(view, train_view)
        proto = []
        max_s = []
        ent = []
        for idx, row in enumerate(sim):
            if train_mask[idx]:
                row = row.copy()
                train_positions = np.where(train_mask)[0]
                self_pos = np.where(train_positions == idx)[0]
                if len(self_pos):
                    row[self_pos[0]] = -1.0
            nn = np.argsort(-row)[: bundle.k_neighbors]
            nn_sim = np.maximum(row[nn], 0)
            if nn_sim.sum() == 0:
                weights = np.full(len(nn), 1.0 / max(1, len(nn)))
            else:
                weights = nn_sim / (nn_sim.sum() + 1e-8)
            proto.append((weights[:, None] * residual_targets[train_mask][nn]).sum(axis=0))
            max_s.append(float(row[nn[0]]))
            ent.append(float(-(weights * np.log(weights + 1e-8)).sum()))
        outputs[view_name] = np.asarray(proto, dtype=np.float32)
        pooled_proto.append(outputs[view_name])
        best_sim.append(np.asarray(max_s, dtype=np.float32))
        entropy.append(np.asarray(ent, dtype=np.float32))
    pooled = np.mean(np.stack(pooled_proto, axis=0), axis=0)
    disagreement_arr = np.std(np.stack(pooled_proto, axis=0), axis=0).mean(axis=1)
    train_desc = bundle.descriptors["concat"][train_mask]
    desc_sim = cosine_similarity(bundle.descriptors["concat"], train_desc)
    for row in desc_sim:
        nn = np.argsort(-row)[:5]
        density.append(float(np.mean(1.0 - row[nn])))
    return {
        "prototype": pooled if use_retrieval else np.zeros_like(pooled),
        "prototype_norm": np.linalg.norm(pooled, axis=1).astype(np.float32) if use_retrieval else np.zeros(len(bundle.perturbations), dtype=np.float32),
        "best_similarity": np.max(np.stack(best_sim, axis=1), axis=1).astype(np.float32) if use_retrieval else np.zeros(len(bundle.perturbations), dtype=np.float32),
        "retrieval_entropy": np.mean(np.stack(entropy, axis=1), axis=1).astype(np.float32) if use_retrieval else np.zeros(len(bundle.perturbations), dtype=np.float32),
        "view_disagreement": disagreement_arr.astype(np.float32) if use_retrieval else np.zeros(len(bundle.perturbations), dtype=np.float32),
        "density": np.asarray(density, dtype=np.float32),
    }


def train_residual_corrector(bundle: SeedBundle, y0: np.ndarray, direct_prediction: bool = False, use_retrieval: bool = True, root: str | Path = ".", log_fn: callable | None = None) -> dict[str, Any]:
    device = infer_device()
    set_deterministic(bundle.seed)
    panel_idx = panel_indices(bundle)
    train_mask = bundle.split_labels == "train"
    route_mask = bundle.split_labels == "route_dev"
    if direct_prediction:
        residual_targets = bundle.pseudobulk[:, panel_idx]
    else:
        residual_targets = bundle.pseudobulk[:, panel_idx] - y0[:, panel_idx]
    retrieval = _build_retrieval_features(bundle, residual_targets, use_retrieval=use_retrieval)
    inputs = np.concatenate(
        [
            y0[:, panel_idx],
            np.repeat(bundle.control_pseudobulk[panel_idx][None, :], len(bundle.perturbations), axis=0),
            bundle.descriptors["concat"],
            retrieval["prototype"],
            retrieval["prototype_norm"][:, None],
            retrieval["best_similarity"][:, None],
            retrieval["view_disagreement"][:, None],
            retrieval["density"][:, None],
        ],
        axis=1,
    ).astype(np.float32)

    x_train = torch.tensor(inputs[train_mask], dtype=torch.float32, device=device)
    x_dev = torch.tensor(inputs[route_mask], dtype=torch.float32, device=device)
    y_train = torch.tensor(residual_targets[train_mask], dtype=torch.float32, device=device)
    y_dev = torch.tensor(residual_targets[route_mask], dtype=torch.float32, device=device)

    model = ResidualMLP(inputs.shape[1], len(panel_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    best_state = None
    best_loss = math.inf
    patience = 20
    patience_left = patience
    batch_size = 64
    start = time.time()
    cpu_start = time.process_time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    max_epochs = 200
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        train_loss_values = []
        for offset in range(0, x_train.shape[0], batch_size):
            idx = order[offset:offset + batch_size]
            pred = model(x_train[idx])
            loss = criterion(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_values.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            dev_loss = criterion(model(x_dev), y_dev).item()
        mean_train_loss = float(np.mean(train_loss_values)) if train_loss_values else math.nan
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
            improved = True
        else:
            patience_left -= 1
            improved = False
        if log_fn is not None:
            log_fn(
                f"epoch={epoch + 1}/{max_epochs} train_loss={mean_train_loss:.6f} "
                f"dev_loss={dev_loss:.6f} best_dev_loss={best_loss:.6f} "
                f"patience_left={patience_left} improved={int(improved)}"
            )
        if patience_left == 0:
            break
    model.load_state_dict(best_state)
    model.eval()
    all_x = torch.tensor(inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        panel_pred = model(all_x).detach().cpu().numpy()
    mc = []
    model.train()
    with torch.no_grad():
        for _ in range(5):
            mc.append(model(all_x).detach().cpu().numpy())
    mc = np.stack(mc, axis=0)
    y1 = y0.copy()
    if direct_prediction:
        y1[:, panel_idx] = panel_pred
    else:
        y1[:, panel_idx] = y0[:, panel_idx] + panel_pred
    runtime = {
        "seconds": time.time() - start,
        "cpu_seconds": time.process_time() - cpu_start,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated() if device == "cuda" else 0),
        "peak_rss_bytes": get_process_peak_rss_bytes(),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return {
        "inputs": inputs,
        "y1": y1.astype(np.float32),
        "mc": mc.astype(np.float32),
        "runtime": runtime,
        "retrieval": retrieval,
    }


def rmse_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - true) ** 2, axis=1))


def pearson_rows(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    out = []
    for a, b in zip(pred, true):
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = np.linalg.norm(a0) * np.linalg.norm(b0)
        out.append(float((a0 @ b0) / denom) if denom > 0 else 0.0)
    return np.asarray(out, dtype=np.float32)


def pathway_scores(matrix: np.ndarray, genes: np.ndarray, pathways: dict[str, np.ndarray]) -> np.ndarray:
    scores = []
    for members in pathways.values():
        idx = np.where(np.isin(genes, members))[0]
        if len(idx) == 0:
            continue
        scores.append(matrix[:, idx].mean(axis=1))
    return np.vstack(scores).T if scores else np.zeros((matrix.shape[0], 1), dtype=np.float32)


def routing_features(bundle: SeedBundle, y0: np.ndarray, residual_artifacts: dict[str, Any], omit_novelty: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    panel_idx = panel_indices(bundle)
    uncertainty = residual_artifacts["mc"].std(axis=0).mean(axis=1)
    novelty = np.zeros(len(bundle.perturbations), dtype=np.float32)
    train_mask = bundle.split_labels == "train"
    sim = cosine_similarity(bundle.descriptors["concat"], bundle.descriptors["concat"][train_mask])
    novelty[:] = 1.0 - sim.max(axis=1)
    features = {
        "novelty": novelty if not omit_novelty else np.zeros_like(novelty),
        "density": residual_artifacts["retrieval"]["density"] if not omit_novelty else np.zeros_like(novelty),
        "best_similarity": residual_artifacts["retrieval"]["best_similarity"],
        "view_disagreement": residual_artifacts["retrieval"]["view_disagreement"],
        "retrieval_entropy": residual_artifacts["retrieval"]["retrieval_entropy"],
        "baseline_norm": np.linalg.norm(y0[:, panel_idx], axis=1).astype(np.float32),
        "predicted_residual_norm": np.linalg.norm(residual_artifacts["y1"][:, panel_idx] - y0[:, panel_idx], axis=1).astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
    }
    if bundle.dataset == "Norman":
        features["is_0_seen"] = (bundle.hardness_labels == "0-seen").astype(np.float32)
        features["is_1_seen"] = (bundle.hardness_labels == "1-seen").astype(np.float32)
    feature_matrix = np.column_stack([features[key] for key in features]).astype(np.float32)
    return feature_matrix, features


def fit_router_models(bundle: SeedBundle, y0: np.ndarray, residual_artifacts: dict[str, Any], gain_key: str = "rmse", alpha: float = 0.2, omit_novelty: bool = False, uncertainty_only: bool = False, disable_conformal: bool = False, cross_fit_conformal: bool = False, calibration_subsample_fraction: float = 1.0, log_fn: callable | None = None) -> dict[str, Any]:
    start = time.time()
    cpu_start = time.process_time()
    x_feat, feat_dict = routing_features(bundle, y0, residual_artifacts, omit_novelty=omit_novelty)
    route_mask = bundle.split_labels == "route_dev"
    cal_mask = bundle.split_labels == "calibration"
    test_mask = bundle.split_labels == "test"
    y_true = bundle.pseudobulk
    top_idx = top_de_indices(bundle)
    path_true = pathway_scores(y_true - bundle.control_pseudobulk, bundle.retained_genes, bundle.pathways)
    path_y0 = pathway_scores(y0 - bundle.control_pseudobulk, bundle.retained_genes, bundle.pathways)
    path_y1 = pathway_scores(residual_artifacts["y1"] - bundle.control_pseudobulk, bundle.retained_genes, bundle.pathways)
    gain_targets = {
        "rmse": rmse_rows(y0, y_true) - rmse_rows(residual_artifacts["y1"], y_true),
        "topde": rmse_rows(y0[:, top_idx], y_true[:, top_idx]) - rmse_rows(residual_artifacts["y1"][:, top_idx], y_true[:, top_idx]),
        "pearson": pearson_rows(residual_artifacts["y1"] - bundle.control_pseudobulk, y_true - bundle.control_pseudobulk) - pearson_rows(y0 - bundle.control_pseudobulk, y_true - bundle.control_pseudobulk),
        "pathway": pearson_rows(path_y1, path_true) - pearson_rows(path_y0, path_true),
    }
    gains = gain_targets[gain_key]
    feature_names = list(feat_dict.keys())
    if uncertainty_only:
        keep = ["uncertainty", "predicted_residual_norm"]
        keep_idx = [feature_names.index(name) for name in keep if name in feature_names]
        x_feat = x_feat[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
    x_route = x_feat[route_mask]
    x_cal = x_feat[cal_mask]
    x_test = x_feat[test_mask]
    y_route = gains[route_mask]
    y_cal = gains[cal_mask]
    y_test = gains[test_mask]
    if 0 < calibration_subsample_fraction < 1.0 and len(x_cal):
        rng = np.random.default_rng(bundle.seed + 7000)
        kept = max(1, int(round(len(x_cal) * calibration_subsample_fraction)))
        sel = np.sort(rng.choice(len(x_cal), size=kept, replace=False))
        x_cal = x_cal[sel]
        y_cal = y_cal[sel]
    if log_fn is not None:
        log_fn(
            f"router_setup route_n={len(x_route)} calibration_n={len(x_cal)} test_n={len(x_test)} "
            f"gain_key={gain_key} alpha={alpha} uncertainty_only={int(uncertainty_only)} "
            f"disable_conformal={int(disable_conformal)} cross_fit_conformal={int(cross_fit_conformal)} "
            f"calibration_subsample_fraction={calibration_subsample_fraction}"
        )

    positive = (y_route > 0).astype(int)
    if np.unique(positive).size < 2:
        clf_scores = np.full(x_test.shape[0], float(positive.mean()), dtype=np.float32)
    else:
        clf = GradientBoostingClassifier(random_state=bundle.seed, max_depth=3, n_estimators=200, learning_rate=0.05, subsample=0.8)
        clf.fit(x_route, positive)
        clf_scores = clf.predict_proba(x_test)[:, 1]

    uncert_scores = -feat_dict["uncertainty"][test_mask]
    reg = GradientBoostingRegressor(random_state=bundle.seed, max_depth=3, n_estimators=300, learning_rate=0.03, subsample=0.8)
    reg.fit(x_route, y_route)
    reg_scores = reg.predict(x_test)

    qreg = GradientBoostingRegressor(random_state=bundle.seed, loss="quantile", alpha=alpha, max_depth=3, n_estimators=300, learning_rate=0.03, subsample=0.8)
    qreg.fit(x_route, y_route)
    q_cal = qreg.predict(x_cal)
    q_test = qreg.predict(x_test)
    conformal_scores = q_cal - y_cal
    level = min(1.0, math.ceil((len(conformal_scores) + 1) * 0.9) / max(1, len(conformal_scores)))
    adjustment = float(np.quantile(conformal_scores, level, method="higher")) if len(conformal_scores) else 0.0
    conf_lb = q_test - adjustment
    cross_fit_adjustment = None
    if cross_fit_conformal and len(x_cal) and len(x_route):
        qreg_swap = GradientBoostingRegressor(random_state=bundle.seed + 1000, loss="quantile", alpha=alpha, max_depth=3, n_estimators=300, learning_rate=0.03, subsample=0.8)
        qreg_swap.fit(x_cal, y_cal)
        q_route_swap = qreg_swap.predict(x_route)
        q_test_swap = qreg_swap.predict(x_test)
        swap_scores = q_route_swap - y_route
        swap_level = min(1.0, math.ceil((len(swap_scores) + 1) * 0.9) / max(1, len(swap_scores)))
        cross_fit_adjustment = float(np.quantile(swap_scores, swap_level, method="higher")) if len(swap_scores) else 0.0
        conf_lb = 0.5 * ((q_test - adjustment) + (q_test_swap - cross_fit_adjustment))
    if disable_conformal:
        adjustment = 0.0
        conf_lb = q_test.copy()
    qraw_scores = q_test
    if log_fn is not None:
        log_fn(
            f"router_fit classifier_positive_rate={positive.mean():.6f} "
            f"conformal_adjustment={adjustment:.6f} "
            f"cross_fit_adjustment={cross_fit_adjustment if cross_fit_adjustment is not None else 'NA'}"
        )
    runtime = {
        "seconds": time.time() - start,
        "cpu_seconds": time.process_time() - cpu_start,
        "peak_rss_bytes": get_process_peak_rss_bytes(),
        "classifier_parameter_count": int(200 * x_route.shape[1]),
        "uncertainty_parameter_count": 0,
        "gain_regressor_parameter_count": int(300 * x_route.shape[1]),
        "conformal_parameter_count": int(300 * x_route.shape[1]),
    }
    return {
        "features": x_feat,
        "feature_names": feature_names,
        "gains": gains,
        "test_gains": y_test,
        "test_mask": test_mask,
        "route_mask": route_mask,
        "cal_mask": cal_mask,
        "scores": {
            "classifier_gate": clf_scores,
            "uncertainty_gate": uncert_scores,
            "gain_regressor": reg_scores,
            "conformal_gate": conf_lb,
            "quantile_raw_gate": qraw_scores,
        },
        "adjustment": adjustment,
        "swap_adjustment": cross_fit_adjustment,
        "runtime": runtime,
        "system": get_system_info(),
    }


def select_mask_from_scores(scores: np.ndarray, acceptance: float) -> np.ndarray:
    n = len(scores)
    k = max(1, int(round(acceptance * n)))
    idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def evaluate_methods(bundle: SeedBundle, y0: np.ndarray, residual_artifacts: dict[str, Any], router_artifacts: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    test_mask = bundle.split_labels == "test"
    y_true = bundle.pseudobulk[test_mask]
    y0_test = y0[test_mask]
    y1_test = residual_artifacts["y1"][test_mask]
    perts = bundle.perturbations[test_mask]
    novelty = bundle.novelty_quartiles[test_mask]
    hardness = bundle.hardness_labels[test_mask]
    top_idx = top_de_indices(bundle)
    path_true = pathway_scores(y_true - bundle.control_pseudobulk, bundle.retained_genes, bundle.pathways)
    methods = {
        "always_baseline": np.ones(len(perts), dtype=bool),
        "always_correct": np.ones(len(perts), dtype=bool),
    }
    pred_map = {
        "always_baseline": y0_test,
        "always_correct": y1_test,
    }
    for router_name, scores in router_artifacts["scores"].items():
        for acceptance in ACCEPTANCE_RATES:
            methods[f"{router_name}@{int(acceptance*100)}"] = select_mask_from_scores(scores, acceptance)
            pred_map[f"{router_name}@{int(acceptance*100)}"] = np.where(methods[f"{router_name}@{int(acceptance*100)}"][:, None], y1_test, y0_test)

    records = []
    summary: dict[str, Any] = {}
    for method, pred in pred_map.items():
        all_rmse = rmse_rows(pred, y_true)
        top_rmse = rmse_rows(pred[:, top_idx], y_true[:, top_idx])
        all_pearson = pearson_rows(pred - bundle.control_pseudobulk, y_true - bundle.control_pseudobulk)
        top_pearson = pearson_rows(pred[:, top_idx] - bundle.control_pseudobulk[top_idx], y_true[:, top_idx] - bundle.control_pseudobulk[top_idx])
        gain = rmse_rows(y0_test, y_true) - all_rmse
        accepted_mask = methods[method] if method in methods else np.ones(len(perts), dtype=bool)
        pred_path = pathway_scores(pred - bundle.control_pseudobulk, bundle.retained_genes, bundle.pathways)
        path_corr = pearson_rows(pred_path, path_true)
        overlap = []
        sign_acc = []
        for row_pred, row_true in zip(pred, y_true):
            top_true = np.argsort(-np.abs(row_true - bundle.control_pseudobulk))[:20]
            top_pred = np.argsort(-np.abs(row_pred - bundle.control_pseudobulk))[:20]
            overlap.append(len(set(top_true) & set(top_pred)) / 20.0)
            sign_true = np.sign((row_true - bundle.control_pseudobulk)[top_true[:50]])
            sign_pred = np.sign((row_pred - bundle.control_pseudobulk)[top_true[:50]])
            sign_acc.append(float((sign_true == sign_pred).mean()))
        selective_rows = []
        base_score = np.ones(len(perts))
        if method.startswith("classifier_gate"):
            base_score = router_artifacts["scores"]["classifier_gate"]
        elif method.startswith("uncertainty_gate"):
            base_score = router_artifacts["scores"]["uncertainty_gate"]
        elif method.startswith("gain_regressor"):
            base_score = router_artifacts["scores"]["gain_regressor"]
        elif method.startswith("conformal_gate"):
            base_score = router_artifacts["scores"]["conformal_gate"]
        elif method.startswith("quantile_raw_gate"):
            base_score = router_artifacts["scores"]["quantile_raw_gate"]
        for rate in SELECTIVE_GRID:
            mask = select_mask_from_scores(base_score, rate) if base_score is not None else np.ones(len(perts), dtype=bool)
            selective_rows.append(gain[mask].mean())
        selective_auc = float(np.mean(selective_rows))
        accepted_all_rmse = all_rmse[accepted_mask]
        accepted_top_rmse = top_rmse[accepted_mask]
        accepted_all_pearson = all_pearson[accepted_mask]
        accepted_top_pearson = top_pearson[accepted_mask]
        accepted_gain = gain[accepted_mask]
        accepted_path_corr = path_corr[accepted_mask]
        accepted_overlap = np.asarray(overlap)[accepted_mask]
        accepted_sign_acc = np.asarray(sign_acc)[accepted_mask]
        summary[method] = {
            "all_gene_rmse": float(accepted_all_rmse.mean()),
            "top_de_rmse": float(accepted_top_rmse.mean()),
            "all_gene_pearson_delta": float(accepted_all_pearson.mean()),
            "top_de_pearson_delta": float(accepted_top_pearson.mean()),
            "mean_gain": float(accepted_gain.mean()),
            "median_gain": float(np.median(accepted_gain)),
            "pathway_corr": float(accepted_path_corr.mean()),
            "top20_overlap": float(np.mean(accepted_overlap)),
            "sign_accuracy": float(np.mean(accepted_sign_acc)),
            "selective_auc": selective_auc,
            "selective_curve": {str(int(rate * 100)): float(value) for rate, value in zip(SELECTIVE_GRID, selective_rows)},
            "accepted_fraction": float(accepted_mask.mean()),
            "full_test_all_gene_rmse": float(all_rmse.mean()),
            "full_test_top_de_rmse": float(top_rmse.mean()),
            "full_test_mean_gain": float(gain.mean()),
        }
        for idx, (perturbation, rmse_value, top_rmse_value, pearson_value, gain_value, nov, hard, accepted_value) in enumerate(
            zip(perts, all_rmse, top_rmse, all_pearson, gain, novelty, hardness, accepted_mask)
        ):
            row_true_delta = y_true[idx] - bundle.control_pseudobulk
            row_pred_delta = pred[idx] - bundle.control_pseudobulk
            top_true = np.argsort(-np.abs(row_true_delta))[:20]
            top_pred = np.argsort(-np.abs(row_pred_delta))[:20]
            row_path_true = pathway_scores(row_true_delta[None, :], bundle.retained_genes, bundle.pathways)
            row_path_pred = pathway_scores(row_pred_delta[None, :], bundle.retained_genes, bundle.pathways)
            records.append(
                {
                    "perturbation": perturbation,
                    "method": method,
                    "all_gene_rmse": float(rmse_value),
                    "top_de_rmse": float(top_rmse_value),
                    "all_gene_pearson_delta": float(pearson_value),
                    "gain_rmse": float(gain_value),
                    "top20_overlap": float(len(set(top_true) & set(top_pred)) / 20.0),
                    "sign_accuracy": float((np.sign(row_true_delta[top_true[:50]]) == np.sign(row_pred_delta[top_true[:50]])).mean()),
                    "pathway_corr": float(pearson_rows(row_path_pred, row_path_true)[0]),
                    "accepted": bool(accepted_value),
                    "novelty_quartile": nov,
                    "hardness": hard,
                }
            )
    return pd.DataFrame(records), summary


def conformal_interval_stats(bundle: SeedBundle, prediction: np.ndarray) -> dict[str, float]:
    cal_mask = bundle.split_labels == "calibration"
    test_mask = bundle.split_labels == "test"
    panel_idx = panel_indices(bundle)
    all_abs = np.abs(prediction[cal_mask] - bundle.pseudobulk[cal_mask])
    panel_abs = np.abs(prediction[cal_mask][:, panel_idx] - bundle.pseudobulk[cal_mask][:, panel_idx])
    off_idx = np.array([i for i in range(prediction.shape[1]) if i not in set(panel_idx.tolist())], dtype=int)
    off_abs = np.abs(prediction[cal_mask][:, off_idx] - bundle.pseudobulk[cal_mask][:, off_idx]) if len(off_idx) else np.zeros((1, 1))
    q_all = float(np.quantile(all_abs, 0.9))
    q_panel = float(np.quantile(panel_abs, 0.9))
    q_off = float(np.quantile(off_abs, 0.9))
    test_true = bundle.pseudobulk[test_mask]
    return {
        "all_gene_coverage": float((np.abs(prediction[test_mask] - test_true) <= q_all).mean()),
        "all_gene_width": 2 * q_all,
        "panel_coverage": float((np.abs(prediction[test_mask][:, panel_idx] - test_true[:, panel_idx]) <= q_panel).mean()),
        "panel_width": 2 * q_panel,
        "off_panel_coverage": float((np.abs(prediction[test_mask][:, off_idx] - test_true[:, off_idx]) <= q_off).mean()) if len(off_idx) else 1.0,
        "off_panel_width": 2 * q_off,
    }


def save_run_artifacts(out_dir: str | Path, config: dict[str, Any], metrics: dict[str, Any], per_perturbation: pd.DataFrame, predictions: dict[str, np.ndarray], runtime: dict[str, Any]) -> None:
    out_dir = ensure_dir(out_dir)
    ensure_dir(out_dir / "logs")
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metrics.json", metrics)
    per_perturbation.to_csv(out_dir / "per_perturbation.csv", index=False)
    np.savez_compressed(out_dir / "predictions.npz", **predictions)
    write_json(out_dir / "runtime.json", runtime)
