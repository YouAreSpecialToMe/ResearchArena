from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .metrics import binary_entropy


@dataclass
class DatasetRecord:
    kind: str
    target: int | None
    family: str | None
    data: np.ndarray


def orient_from_order(skeleton: np.ndarray, compelled: np.ndarray, order: tuple[int, ...]) -> np.ndarray | None:
    pos = {node: idx for idx, node in enumerate(order)}
    d = skeleton.shape[0]
    adj = np.zeros((d, d), dtype=int)
    rows, cols = np.where(np.triu(skeleton, 1) > 0)
    for i, j in zip(rows, cols):
        if compelled[i, j]:
            if pos[i] > pos[j]:
                return None
            adj[i, j] = 1
        elif compelled[j, i]:
            if pos[j] > pos[i]:
                return None
            adj[j, i] = 1
        elif pos[i] < pos[j]:
            adj[i, j] = 1
        else:
            adj[j, i] = 1
    return adj


def enumerate_exact_dags(skeleton: np.ndarray, compelled: np.ndarray, cap: int) -> list[np.ndarray]:
    d = skeleton.shape[0]
    dags: dict[bytes, np.ndarray] = {}
    for order in itertools.permutations(range(d)):
        dag = orient_from_order(skeleton, compelled, order)
        if dag is None:
            continue
        dags[dag.tobytes()] = dag
        if len(dags) > cap:
            return []
    return list(dags.values())


def sample_particle_dags(skeleton: np.ndarray, compelled: np.ndarray, n_particles: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    d = skeleton.shape[0]
    dags: dict[bytes, np.ndarray] = {}
    attempts = 0
    max_attempts = max(n_particles * 200, 2000)
    while len(dags) < n_particles and attempts < max_attempts:
        order = tuple(rng.permutation(d).tolist())
        dag = orient_from_order(skeleton, compelled, order)
        if dag is None:
            attempts += 1
            continue
        dags[dag.tobytes()] = dag
        attempts += 1
    return list(dags.values())


def _gaussian_ll(y: np.ndarray, x: np.ndarray | None) -> float:
    n = len(y)
    if x is None or x.size == 0:
        resid = y - np.mean(y)
    else:
        x_centered = x - np.mean(x, axis=0, keepdims=True)
        y_centered = y - np.mean(y)
        coef, *_ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
        resid = y_centered - x_centered @ coef
    sigma2 = max(float(np.mean(resid**2)), 1e-6)
    return float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))


def score_dag_on_records(dag: np.ndarray, records: list[DatasetRecord]) -> float:
    total = 0.0
    for rec in records:
        x = rec.data
        for node in range(dag.shape[0]):
            parents = np.flatnonzero(dag[:, node])
            if rec.kind == "intervention" and rec.family == "hard" and rec.target == node:
                total += _gaussian_ll(x[:, node], None)
            else:
                parent_x = x[:, parents] if parents.size else None
                total += _gaussian_ll(x[:, node], parent_x)
    return total


@dataclass
class PosteriorState:
    dags: list[np.ndarray]
    weights: np.ndarray
    skeleton: np.ndarray
    compelled: np.ndarray
    datasets: list[DatasetRecord]
    exact: bool
    previous_family: str | None
    ess_before: float = np.nan
    ess_after: float = np.nan
    mh_accept_rate: float = np.nan

    def clone(self) -> "PosteriorState":
        return copy.deepcopy(self)

    def orientation_probabilities(self) -> np.ndarray:
        d = self.skeleton.shape[0]
        probs = np.zeros((d, d), dtype=float)
        for dag, w in zip(self.dags, self.weights):
            probs += w * dag
        return probs

    def orientation_entropy(self) -> float:
        p = self.orientation_probabilities()
        mask = np.triu(self.skeleton, 1) > 0
        edge_ps = p[mask]
        return float(binary_entropy(edge_ps).sum())

    def map_dag(self) -> np.ndarray:
        return self.dags[int(np.argmax(self.weights))]

    def expected_switch_fraction(self, switch_count: int, spent_budget: float) -> float:
        if spent_budget <= 0:
            return 0.0
        return float(switch_count / spent_budget)


@dataclass
class FittedDagParams:
    intercepts: np.ndarray
    coefs: list[np.ndarray]
    noise_vars: np.ndarray


def initialize_posterior(
    skeleton: np.ndarray,
    compelled: np.ndarray,
    obs_data: np.ndarray,
    exact: bool,
    n_particles: int,
    cap: int,
    seed: int,
) -> PosteriorState:
    dags = enumerate_exact_dags(skeleton, compelled, cap) if exact else sample_particle_dags(skeleton, compelled, n_particles, seed)
    if not dags:
        dags = enumerate_exact_dags(skeleton, compelled, min(cap, 2048))
        exact = exact and bool(dags)
    if not dags:
        dags = sample_particle_dags(skeleton, compelled, max(16, min(n_particles, 64)), seed + 17)
        exact = False
    if not dags:
        # Last-resort fallback: orient every undirected edge by one random order that satisfies compelled edges.
        for attempt in range(4096):
            order = tuple(np.random.default_rng(seed + 1000 + attempt).permutation(skeleton.shape[0]).tolist())
            dag = orient_from_order(skeleton, compelled, order)
            if dag is not None:
                dags = [dag]
                exact = False
                break
    if not dags:
        raise RuntimeError("Failed to initialize a non-empty posterior DAG set.")
    records = [DatasetRecord(kind="observational", target=None, family=None, data=obs_data)]
    scores = np.array([score_dag_on_records(dag, records) for dag in dags], dtype=float)
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= weights.sum()
    return PosteriorState(
        dags=dags,
        weights=weights,
        skeleton=skeleton,
        compelled=compelled,
        datasets=records,
        exact=exact,
        previous_family=None,
    )


def update_posterior(state: PosteriorState, rec: DatasetRecord, rejuvenate: bool, seed: int) -> PosteriorState:
    next_state = state.clone()
    next_state.datasets.append(rec)
    scores = np.array([score_dag_on_records(dag, next_state.datasets) for dag in next_state.dags], dtype=float)
    scores -= np.max(scores)
    weights = np.exp(scores)
    weights /= weights.sum()
    next_state.ess_before = float(1.0 / np.sum(state.weights**2))
    next_state.ess_after = float(1.0 / np.sum(weights**2))
    next_state.weights = weights
    next_state.previous_family = rec.family
    next_state.mh_accept_rate = 0.0
    if rejuvenate and not next_state.exact and next_state.ess_after < 0.5 * len(next_state.dags):
        rng = np.random.default_rng(seed)
        new_dags = sample_particle_dags(next_state.skeleton, next_state.compelled, max(8, len(next_state.dags) // 3), int(rng.integers(1_000_000)))
        existing = {dag.tobytes() for dag in next_state.dags}
        accepted = 0
        for dag in new_dags:
            key = dag.tobytes()
            if key in existing:
                continue
            idx = int(np.argmin(next_state.weights))
            next_state.dags[idx] = dag
            existing.add(key)
            accepted += 1
        if accepted:
            scores = np.array([score_dag_on_records(dag, next_state.datasets) for dag in next_state.dags], dtype=float)
            scores -= np.max(scores)
            weights = np.exp(scores)
            weights /= weights.sum()
            next_state.weights = weights
        next_state.mh_accept_rate = accepted / max(len(new_dags), 1)
    return next_state


def fit_dag_parameters(dag: np.ndarray, datasets: list[DatasetRecord]) -> FittedDagParams:
    d = dag.shape[0]
    intercepts = np.zeros(d, dtype=float)
    coefs: list[np.ndarray] = []
    noise_vars = np.ones(d, dtype=float)
    for node in range(d):
        ys_centered = []
        xs_centered = []
        obs_ys = []
        obs_xs = []
        parents = np.flatnonzero(dag[:, node])
        for rec in datasets:
            if rec.kind == "intervention" and rec.target == node:
                if rec.family == "hard":
                    continue
            y = rec.data[:, node]
            if parents.size:
                x = rec.data[:, parents]
                ys_centered.append(y - np.mean(y))
                xs_centered.append(x - np.mean(x, axis=0, keepdims=True))
            else:
                ys_centered.append(y - np.mean(y))
            if rec.kind == "observational":
                obs_ys.append(y)
                if parents.size:
                    obs_xs.append(rec.data[:, parents])
        if not ys_centered:
            coefs.append(np.zeros(parents.size, dtype=float))
            continue
        y_all = np.concatenate(ys_centered, axis=0)
        if parents.size:
            x_all = np.concatenate(xs_centered, axis=0)
            beta, *_ = np.linalg.lstsq(x_all, y_all, rcond=None)
            coef = beta
            coefs.append(coef)
            resid = y_all - x_all @ coef
            if obs_ys:
                obs_y = np.concatenate(obs_ys, axis=0)
                obs_x = np.concatenate(obs_xs, axis=0)
                intercepts[node] = float(np.mean(obs_y) - np.mean(obs_x, axis=0) @ coef)
        else:
            intercepts[node] = float(np.mean(np.concatenate(obs_ys, axis=0))) if obs_ys else 0.0
            coefs.append(np.zeros(0, dtype=float))
            resid = y_all
        noise_vars[node] = max(float(np.mean(resid**2)), 1e-6)
    return FittedDagParams(intercepts=intercepts, coefs=coefs, noise_vars=noise_vars)


def simulate_from_posterior_dag(
    rng: np.random.Generator,
    dag: np.ndarray,
    params: FittedDagParams,
    n: int,
    intervention: dict | None = None,
) -> np.ndarray:
    d = dag.shape[0]
    x = np.zeros((n, d), dtype=float)
    order = list(nx.topological_sort(nx.DiGraph(dag)))
    intervention = intervention or {}
    target = intervention.get("target")
    family = intervention.get("family")
    delta = intervention.get("soft_delta", 0.0)
    var_mult = intervention.get("soft_var_mult", 1.0)
    for node in order:
        parents = np.flatnonzero(dag[:, node])
        mean = np.full(n, params.intercepts[node], dtype=float)
        if parents.size:
            mean += x[:, parents] @ params.coefs[node]
        if node == target and family == "hard":
            x[:, node] = rng.normal(0.0, 1.0, size=n)
        elif node == target and family == "soft":
            x[:, node] = mean + delta + rng.normal(0.0, np.sqrt(params.noise_vars[node] * var_mult), size=n)
        else:
            x[:, node] = mean + rng.normal(0.0, np.sqrt(params.noise_vars[node]), size=n)
    return x


def tv_orientation_error(approx_state: PosteriorState, exact_state: PosteriorState) -> float:
    pa = approx_state.orientation_probabilities()
    pe = exact_state.orientation_probabilities()
    mask = np.triu(exact_state.skeleton, 1) > 0
    return float(np.mean(np.abs(pa[mask] - pe[mask])))


def dag_kl(approx_state: PosteriorState, exact_state: PosteriorState) -> float:
    exact_map = {dag.tobytes(): w for dag, w in zip(exact_state.dags, exact_state.weights)}
    kl = 0.0
    eps = 1e-12
    for dag, w in zip(approx_state.dags, approx_state.weights):
        q = exact_map.get(dag.tobytes(), eps)
        kl += float(w * (np.log(max(w, eps)) - np.log(max(q, eps))))
    return kl
