from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
from causallearn.search.ScoreBased.GES import ges


@dataclass
class Instance:
    seed: int
    graph_family: str
    node_count: int
    regime: str
    budget: float
    switch_regime: str
    true_adj: np.ndarray
    true_weights: np.ndarray
    noise_vars: np.ndarray
    obs_data: np.ndarray
    learned_skeleton: np.ndarray
    learned_compelled: np.ndarray
    oracle_skeleton: np.ndarray
    meta: dict


def _random_order(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.permutation(d)


def _erdos_dag(rng: np.random.Generator, d: int, expected_degree: float) -> np.ndarray:
    order = _random_order(rng, d)
    pos = {node: i for i, node in enumerate(order)}
    p = expected_degree / max(d - 1, 1)
    adj = np.zeros((d, d), dtype=int)
    for u, v in itertools.combinations(range(d), 2):
        a, b = (u, v) if pos[u] < pos[v] else (v, u)
        if rng.random() < p:
            adj[a, b] = 1
    return adj


def _scale_free_dag(rng: np.random.Generator, d: int) -> np.ndarray:
    order = _random_order(rng, d)
    adj = np.zeros((d, d), dtype=int)
    degrees = np.ones(d)
    for idx in range(1, d):
        node = order[idx]
        prev = order[:idx]
        probs = degrees[prev] / np.sum(degrees[prev])
        k = min(idx, 1 + rng.binomial(2, 0.55))
        parents = rng.choice(prev, size=k, replace=False, p=probs / probs.sum())
        for pnode in np.atleast_1d(parents):
            adj[pnode, node] = 1
            degrees[pnode] += 1
            degrees[node] += 1
    return adj


def sample_graph(rng: np.random.Generator, d: int, family: str, expected_degree: float) -> np.ndarray:
    for _ in range(200):
        adj = _erdos_dag(rng, d, expected_degree) if family == "erdos_renyi" else _scale_free_dag(rng, d)
        g = nx.Graph(adj + adj.T)
        if nx.is_connected(g):
            return adj
    raise RuntimeError("Failed to sample connected DAG skeleton.")


def sample_weights(rng: np.random.Generator, adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = adj.shape[0]
    weights = np.zeros_like(adj, dtype=float)
    mask = adj.astype(bool)
    mags = rng.uniform(0.5, 1.5, size=int(mask.sum()))
    signs = rng.choice([-1.0, 1.0], size=int(mask.sum()))
    weights[mask] = mags * signs
    noise_vars = rng.uniform(0.8, 1.2, size=d)
    return weights, noise_vars


def simulate_scm(rng: np.random.Generator, adj: np.ndarray, weights: np.ndarray, noise_vars: np.ndarray, n: int, intervention: dict | None = None) -> np.ndarray:
    d = adj.shape[0]
    x = np.zeros((n, d), dtype=float)
    order = list(nx.topological_sort(nx.DiGraph(adj)))
    intervention = intervention or {}
    target = intervention.get("target")
    family = intervention.get("family")
    delta = intervention.get("soft_delta", 0.0)
    var_mult = intervention.get("soft_var_mult", 1.0)
    for node in order:
        parents = np.flatnonzero(adj[:, node])
        mean = x[:, parents] @ weights[parents, node] if parents.size else 0.0
        noise = rng.normal(0.0, np.sqrt(noise_vars[node]), size=n)
        if node == target and family == "hard":
            x[:, node] = rng.normal(0.0, 1.0, size=n)
        elif node == target and family == "soft":
            noise = rng.normal(0.0, np.sqrt(noise_vars[node] * var_mult), size=n)
            x[:, node] = mean + delta + noise
        else:
            x[:, node] = mean + noise
    return x


def _corr_skeleton_fallback(obs_data: np.ndarray, true_edge_count: int) -> np.ndarray:
    corr = np.corrcoef(obs_data, rowvar=False)
    d = corr.shape[0]
    scores = []
    for i in range(d):
        for j in range(i + 1, d):
            scores.append((abs(corr[i, j]), i, j))
    scores.sort(reverse=True)
    target_edges = max(d - 1, int(round(true_edge_count * 0.9)))
    skeleton = np.zeros((d, d), dtype=int)
    for _, i, j in scores[:target_edges]:
        skeleton[i, j] = skeleton[j, i] = 1
    g = nx.Graph(skeleton)
    if not nx.is_connected(g):
        for _, i, j in scores[target_edges:]:
            if nx.is_connected(g):
                break
            skeleton[i, j] = skeleton[j, i] = 1
            g = nx.Graph(skeleton)
    np.fill_diagonal(skeleton, 0)
    return skeleton


def _extract_cpdag_from_ges(graph) -> tuple[np.ndarray, np.ndarray]:
    node_map = graph.get_node_map()
    name_to_idx = {str(name): idx for name, idx in node_map.items()}
    d = len(name_to_idx)
    skeleton = np.zeros((d, d), dtype=int)
    compelled = np.zeros((d, d), dtype=int)
    for edge in graph.get_graph_edges():
        src = name_to_idx[str(edge.get_node1())]
        dst = name_to_idx[str(edge.get_node2())]
        ep1 = str(edge.get_endpoint1())
        ep2 = str(edge.get_endpoint2())
        skeleton[src, dst] = skeleton[dst, src] = 1
        if ep1 == "TAIL" and ep2 == "ARROW":
            compelled[src, dst] = 1
        elif ep1 == "ARROW" and ep2 == "TAIL":
            compelled[dst, src] = 1
    np.fill_diagonal(skeleton, 0)
    np.fill_diagonal(compelled, 0)
    return skeleton, compelled


def estimate_learned_cpdag(obs_data: np.ndarray, true_edge_count: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    backend = cfg.get("cpdag_backend", "ges_bic")
    d = obs_data.shape[1]
    try:
        model = ges(obs_data, score_func="local_score_BIC")["G"]
        skeleton, compelled = _extract_cpdag_from_ges(model)
        if np.triu(skeleton, 1).sum() == 0:
            raise ValueError("GES returned an empty CPDAG.")
        g = nx.Graph(skeleton)
        if not nx.is_connected(g):
            raise ValueError("GES returned a disconnected skeleton.")
        return skeleton, compelled, {"structure_backend": backend, "structure_fallback": False}
    except Exception:
        skeleton = _corr_skeleton_fallback(obs_data, true_edge_count)
        compelled = np.zeros((d, d), dtype=int)
        return skeleton, compelled, {"structure_backend": "corr_fallback", "structure_fallback": True}


def generate_instance(seed: int, graph_family: str, node_count: int, regime: str, budget: float, switch_regime: str, cfg: dict) -> Instance:
    family_code = {"erdos_renyi": 11, "scale_free": 17}[graph_family]
    regime_code = {"mild_soft": 23, "strong_soft": 29}[regime]
    switch_code = {"S0": 31, "S1": 37, "S2": 41, "S3": 43}.get(switch_regime, 47)
    rng = np.random.default_rng(seed * 1009 + node_count * 37 + family_code + regime_code + switch_code)
    adj = sample_graph(rng, node_count, graph_family, cfg["expected_degree"])
    weights, noise_vars = sample_weights(rng, adj)
    obs_data = simulate_scm(rng, adj, weights, noise_vars, cfg["n_obs"])
    learned, learned_compelled, structure_meta = estimate_learned_cpdag(obs_data, int(adj.sum()), cfg)
    oracle = ((adj + adj.T) > 0).astype(int)
    mec_size_cap = cfg["exact_mec_cap"]
    meta = {
        "true_edges": int(adj.sum()),
        "learned_undirected_edges": int(np.triu(learned, 1).sum()),
        "skeleton_shd": int(np.abs(learned - oracle).sum() // 2),
        "mec_size_cap": mec_size_cap,
        **structure_meta,
    }
    return Instance(seed, graph_family, node_count, regime, budget, switch_regime, adj, weights, noise_vars, obs_data, learned, learned_compelled, oracle, meta)
