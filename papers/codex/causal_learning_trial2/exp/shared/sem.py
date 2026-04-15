from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .common import BATCH_SIZES, DATA_DIR, OBS_SAMPLES, TOTAL_BUDGET, ensure_dir, save_json, set_seed


@dataclass
class SEMInstance:
    instance_id: str
    p: int
    graph_family: str
    weight_regime: str
    seed: int
    order: list[int]
    adjacency: np.ndarray
    weights: np.ndarray
    noise_var: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    observational_data: np.ndarray
    intervention_streams: dict[int, np.ndarray]

    def to_payload(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "p": self.p,
            "graph_family": self.graph_family,
            "weight_regime": self.weight_regime,
            "seed": self.seed,
            "order": self.order,
            "adjacency": self.adjacency,
            "weights": self.weights,
            "noise_var": self.noise_var,
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "observational_data": self.observational_data,
            "intervention_streams": self.intervention_streams,
        }


def _random_order(rng: np.random.Generator, p: int) -> list[int]:
    return rng.permutation(p).tolist()


def _sample_er_dag(rng: np.random.Generator, p: int, order: list[int]) -> np.ndarray:
    prob = min(0.95, 2.0 / max(p - 1, 1))
    adjacency = np.zeros((p, p), dtype=int)
    pos = {node: idx for idx, node in enumerate(order)}
    for i, j in itertools.combinations(range(p), 2):
        if rng.random() < prob:
            src, dst = (i, j) if pos[i] < pos[j] else (j, i)
            adjacency[src, dst] = 1
    return adjacency


def _sample_scale_free_dag(rng: np.random.Generator, p: int, order: list[int]) -> np.ndarray:
    skeleton = nx.barabasi_albert_graph(p, 1, seed=int(rng.integers(1_000_000)))
    pos = {node: idx for idx, node in enumerate(order)}
    adjacency = np.zeros((p, p), dtype=int)
    for u, v in skeleton.edges():
        src, dst = (u, v) if pos[u] < pos[v] else (v, u)
        adjacency[src, dst] = 1
    return adjacency


def _sample_weights(rng: np.random.Generator, adjacency: np.ndarray, weight_regime: str) -> tuple[np.ndarray, np.ndarray]:
    p = adjacency.shape[0]
    weights = np.zeros((p, p), dtype=float)
    mask = adjacency == 1
    if weight_regime == "weak":
        mags = rng.uniform(0.10, 0.25, size=mask.sum())
    else:
        mix = rng.random(mask.sum()) < 0.5
        mags = np.where(mix, rng.uniform(0.10, 0.25, size=mask.sum()), rng.uniform(0.40, 0.80, size=mask.sum()))
    signs = rng.choice([-1.0, 1.0], size=mask.sum())
    weights[mask] = mags * signs
    noise_var = rng.uniform(0.5, 1.5, size=p)
    return weights, noise_var


def sample_sem(weights: np.ndarray, noise_var: np.ndarray, n: int, rng: np.random.Generator, intervene_node: int | None = None) -> np.ndarray:
    p = weights.shape[0]
    x = np.zeros((n, p), dtype=float)
    order = list(nx.topological_sort(nx.DiGraph((i, j) for i, j in zip(*np.where(weights != 0)))))
    if len(order) < p:
        missing = [node for node in range(p) if node not in order]
        order.extend(missing)
    eps = rng.normal(0.0, np.sqrt(noise_var), size=(n, p))
    for node in order:
        if intervene_node is not None and node == intervene_node:
            x[:, node] = 0.0
            continue
        parents = np.where(weights[:, node] != 0)[0]
        if len(parents):
            x[:, node] = x[:, parents] @ weights[parents, node] + eps[:, node]
        else:
            x[:, node] = eps[:, node]
    return x


def standardize_with_obs(obs: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    std[std < 1e-6] = 1.0
    return (data - mean) / std, mean, std


def generate_instance(p: int, graph_family: str, weight_regime: str, seed: int, prefix: str) -> SEMInstance:
    rng = np.random.default_rng(seed)
    order = _random_order(rng, p)
    adjacency = _sample_er_dag(rng, p, order) if graph_family == "erdos_renyi" else _sample_scale_free_dag(rng, p, order)
    weights, noise_var = _sample_weights(rng, adjacency, weight_regime)
    obs_raw = sample_sem(weights, noise_var, OBS_SAMPLES, rng)
    obs, mean, std = standardize_with_obs(obs_raw, obs_raw)
    streams: dict[int, np.ndarray] = {}
    for node in range(p):
        stream_raw = sample_sem(weights, noise_var, TOTAL_BUDGET * 2, rng, intervene_node=node)
        streams[node], _, _ = standardize_with_obs(obs_raw, stream_raw)
    instance_id = f"{prefix}_p{p}_{graph_family}_{weight_regime}_s{seed}"
    return SEMInstance(
        instance_id=instance_id,
        p=p,
        graph_family=graph_family,
        weight_regime=weight_regime,
        seed=seed,
        order=order,
        adjacency=adjacency,
        weights=weights,
        noise_var=noise_var,
        obs_mean=mean,
        obs_std=std,
        observational_data=obs,
        intervention_streams=streams,
    )


def save_instance(instance: SEMInstance, out_dir: Path) -> None:
    ensure_dir(out_dir)
    np.savez_compressed(out_dir / f"{instance.instance_id}.npz", **instance.to_payload())
    save_json(
        out_dir / f"{instance.instance_id}.json",
        {
            "instance_id": instance.instance_id,
            "p": instance.p,
            "graph_family": instance.graph_family,
            "weight_regime": instance.weight_regime,
            "seed": instance.seed,
            "num_edges": int(instance.adjacency.sum()),
        },
    )


def load_instance(path: Path) -> SEMInstance:
    blob = np.load(path, allow_pickle=True)
    streams = blob["intervention_streams"].item()
    return SEMInstance(
        instance_id=str(blob["instance_id"]),
        p=int(blob["p"]),
        graph_family=str(blob["graph_family"]),
        weight_regime=str(blob["weight_regime"]),
        seed=int(blob["seed"]),
        order=list(blob["order"]),
        adjacency=blob["adjacency"],
        weights=blob["weights"],
        noise_var=blob["noise_var"],
        obs_mean=blob["obs_mean"],
        obs_std=blob["obs_std"],
        observational_data=blob["observational_data"],
        intervention_streams={int(k): v for k, v in streams.items()},
    )


def build_benchmark_sets() -> dict[str, list[Path]]:
    set_seed(0)
    core_dir = ensure_dir(DATA_DIR / "core")
    aux_dir = ensure_dir(DATA_DIR / "aux")
    core_paths: list[Path] = []
    aux_paths: list[Path] = []
    for p in [10, 15]:
        for family in ["erdos_renyi", "scale_free"]:
            for regime in ["weak", "mixed"]:
                for seed in [11, 22, 33]:
                    inst = generate_instance(p, family, regime, seed, "core")
                    save_instance(inst, core_dir)
                    core_paths.append(core_dir / f"{inst.instance_id}.npz")
    for family in ["erdos_renyi", "scale_free"]:
        for regime in ["weak", "mixed"]:
            for seed in [101, 202, 303]:
                inst = generate_instance(8, family, regime, seed, "aux")
                save_instance(inst, aux_dir)
                aux_paths.append(aux_dir / f"{inst.instance_id}.npz")
    return {"core": [str(p) for p in sorted(core_paths)], "aux": [str(p) for p in sorted(aux_paths)]}

