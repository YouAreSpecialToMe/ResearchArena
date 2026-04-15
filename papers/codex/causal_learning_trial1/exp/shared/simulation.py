from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import EDGE_PROB, EXPECTED_DEGREE, P
from .graph_utils import dag_to_cpdag_matrix


@dataclass
class SimulatedDataset:
    dataset_id: str
    family: str
    regime: str
    n: int
    p: int
    seed: int
    dag: nx.DiGraph
    cpdag: np.ndarray
    raw: pd.DataFrame
    standardized: pd.DataFrame
    metadata: dict


def sample_family_dag(family: str, seed: int, p: int = P) -> tuple[nx.DiGraph, list[int]]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(p))
    rank = {node: idx for idx, node in enumerate(order)}
    dag = nx.DiGraph()
    dag.add_nodes_from(range(p))
    if family == "erdos_renyi":
        for i in range(p):
            for j in range(i + 1, p):
                a, b = order[i], order[j]
                if rng.random() < EDGE_PROB:
                    dag.add_edge(a, b)
    elif family == "scale_free":
        skeleton = nx.barabasi_albert_graph(p, max(1, int(round(EXPECTED_DEGREE / 2))), seed=seed)
        for u, v in skeleton.edges():
            if rank[u] < rank[v]:
                dag.add_edge(u, v)
            else:
                dag.add_edge(v, u)
    else:
        raise ValueError(f"Unknown family: {family}")
    return dag, order


def simulate_dataset(family: str, regime: str, n: int, seed: int, p: int = P) -> SimulatedDataset:
    dag, order = sample_family_dag(family, seed, p=p)
    rng = np.random.default_rng(seed + 1000)
    parents = {node: sorted(dag.predecessors(node), key=lambda x: order.index(x)) for node in dag.nodes()}
    weights = {}
    nonlinear_choice = {}
    for u, v in dag.edges():
        if regime == "near_unfaithful_linear" and rng.random() < 0.2:
            mag = rng.uniform(0.05, 0.15)
        else:
            mag = rng.uniform(0.4, 1.0)
        weights[(u, v)] = mag * rng.choice([-1.0, 1.0])
        nonlinear_choice[(u, v)] = rng.choice(["tanh", "sine", "square"])
    transformed_edge = None
    if regime == "mild_misspecification" and dag.number_of_edges() > 0:
        transformed_edge = list(dag.edges())[int(rng.integers(dag.number_of_edges()))]

    values = np.zeros((n, p), dtype=float)
    topo = list(nx.topological_sort(dag))
    for node in topo:
        parent_vals = np.zeros(n)
        for parent in parents[node]:
            signal = values[:, parent]
            if regime == "nonlinear_anm":
                signal = _apply_transform(signal, nonlinear_choice[(parent, node)])
            elif regime == "mild_misspecification" and transformed_edge == (parent, node):
                signal = _apply_transform(signal, nonlinear_choice[(parent, node)])
            parent_vals += weights[(parent, node)] * signal
        noise_scale = 1.0
        if regime == "mild_misspecification":
            noise_scale = 0.6 + 0.4 * np.abs(parent_vals)
        noise = rng.normal(0.0, noise_scale, size=n)
        values[:, node] = parent_vals + noise

    columns = [f"X{i}" for i in range(p)]
    raw = pd.DataFrame(values, columns=columns)
    standardized = pd.DataFrame(StandardScaler().fit_transform(values), columns=columns)
    cpdag = dag_to_cpdag_matrix(dag)
    metadata = {
        "dataset_id": f"{family}__{regime}__n{n}__s{seed}",
        "family": family,
        "regime": regime,
        "n": n,
        "p": p,
        "seed": seed,
        "edge_count": dag.number_of_edges(),
        "density": dag.number_of_edges() / (p * (p - 1) / 2),
        "topological_order": topo,
        "weights": {f"{u}->{v}": float(w) for (u, v), w in weights.items()},
        "transformed_edge": None if transformed_edge is None else list(transformed_edge),
    }
    return SimulatedDataset(
        dataset_id=metadata["dataset_id"],
        family=family,
        regime=regime,
        n=n,
        p=p,
        seed=seed,
        dag=dag,
        cpdag=cpdag,
        raw=raw,
        standardized=standardized,
        metadata=metadata,
    )


def _apply_transform(values: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(values)
    if name == "sine":
        return np.sin(values)
    if name == "square":
        return np.sign(values) * np.square(values)
    raise ValueError(name)

