from __future__ import annotations

import copy
import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from scipy.stats import chi2, ncx2

from .common import BOOTSTRAPS, PARTICLE_EXTENSIONS, PARTICLE_LIMIT, set_seed


@dataclass
class Particle:
    adjacency: np.ndarray
    weight: float
    node_coef: list[np.ndarray]
    node_var: np.ndarray
    bic_score: float


def _ges_to_cpdag(data: np.ndarray) -> np.ndarray:
    record = ges(data, score_func="local_score_BIC")
    g = record["G"].graph
    cpdag = np.zeros_like(g, dtype=int)
    p = g.shape[0]
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if g[j, i] == 1 and g[i, j] == -1:
                cpdag[i, j] = 1
            elif g[i, j] == -1 and g[j, i] == -1 and i < j:
                cpdag[i, j] = cpdag[j, i] = -1
    return cpdag


def _sample_dag_from_cpdag(cpdag: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = cpdag.shape[0]
    dag = (cpdag == 1).astype(int)
    directed = nx.DiGraph([(i, j) for i, j in zip(*np.where(dag == 1))])
    if not nx.is_directed_acyclic_graph(directed):
        dag = np.triu((cpdag != 0).astype(int), 1)
        return dag
    partial_order = list(nx.topological_sort(directed))
    remaining = [node for node in range(p) if node not in partial_order]
    random_order = partial_order + rng.permutation(remaining).tolist()
    rank = {node: idx for idx, node in enumerate(random_order)}
    undirected_pairs = [(i, j) for i in range(p) for j in range(i + 1, p) if cpdag[i, j] == -1 and cpdag[j, i] == -1]
    for i, j in undirected_pairs:
        src, dst = (i, j) if rank[i] < rank[j] else (j, i)
        dag[src, dst] = 1
        dag[dst, src] = 0
    if nx.is_directed_acyclic_graph(nx.DiGraph([(i, j) for i, j in zip(*np.where(dag == 1))])):
        return dag
    return np.triu((cpdag != 0).astype(int), 1)


def fit_linear_sem(adjacency: np.ndarray, observational_data: np.ndarray, interventions: list[dict]) -> tuple[list[np.ndarray], np.ndarray, float]:
    p = adjacency.shape[0]
    coeffs: list[np.ndarray] = []
    variances = np.ones(p, dtype=float)
    total_bic = 0.0
    for node in range(p):
        parents = np.where(adjacency[:, node] == 1)[0]
        blocks = [observational_data]
        masks = [np.zeros(len(observational_data), dtype=bool)]
        for batch in interventions:
            blocks.append(batch["data"])
            masks.append(np.full(len(batch["data"]), batch["target"] == node))
        data = np.vstack(blocks)
        intervened_mask = np.concatenate(masks)
        usable = ~intervened_mask
        y = data[usable, node]
        if len(parents):
            X = data[usable][:, parents]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            resid = y - pred
            coef = np.zeros(p, dtype=float)
            coef[parents] = beta
        else:
            resid = y
            coef = np.zeros(p, dtype=float)
        var = float(np.var(resid) + 1e-6)
        coeffs.append(coef)
        variances[node] = var
        n = max(len(resid), 1)
        loglik = -0.5 * n * (np.log(2.0 * np.pi * var) + 1.0)
        k = len(parents) + 1
        total_bic += -2.0 * loglik + k * np.log(n)
    return coeffs, variances, float(total_bic)


def coefficients_to_matrix(coeffs: list[np.ndarray]) -> np.ndarray:
    p = len(coeffs)
    mat = np.zeros((p, p), dtype=float)
    for node, coef in enumerate(coeffs):
        mat[:, node] = coef
    return mat


def build_particles(observational_data: np.ndarray, interventions: list[dict], seed: int) -> list[Particle]:
    rng = np.random.default_rng(seed)
    n = len(observational_data)
    particles: list[Particle] = []
    seen: set[bytes] = set()
    for b in range(BOOTSTRAPS):
        idx = rng.integers(0, n, size=n)
        cpdag = _ges_to_cpdag(observational_data[idx])
        for r in range(PARTICLE_EXTENSIONS):
            dag = _sample_dag_from_cpdag(cpdag, seed + 1000 * b + r)
            key = dag.tobytes()
            if key in seen:
                continue
            seen.add(key)
            coeffs, variances, bic = fit_linear_sem(dag, observational_data, interventions)
            particles.append(Particle(adjacency=dag, weight=1.0, node_coef=coeffs, node_var=variances, bic_score=bic))
    particles.sort(key=lambda p: p.bic_score)
    particles = particles[:PARTICLE_LIMIT]
    return normalize_weights(particles)


def normalize_weights(particles: list[Particle]) -> list[Particle]:
    if not particles:
        return particles
    scores = np.asarray([-p.bic_score for p in particles], dtype=float)
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    for particle, weight in zip(particles, weights):
        particle.weight = float(weight)
    return particles


def effective_sample_size(particles: list[Particle]) -> float:
    w = np.asarray([p.weight for p in particles], dtype=float)
    return float(1.0 / np.sum(w**2))


def rejuvenate_particles(particles: list[Particle], observational_data: np.ndarray, interventions: list[dict], seed: int) -> list[Particle]:
    rng = np.random.default_rng(seed)
    proposals: list[Particle] = []
    for particle in sorted(particles, key=lambda p: -p.weight)[:4]:
        dag = particle.adjacency
        p = dag.shape[0]
        for _ in range(4):
            cand = dag.copy()
            i, j = rng.choice(p, size=2, replace=False)
            if cand[i, j] == 1:
                cand[i, j] = 0
                if rng.random() < 0.5 and cand[j, i] == 0:
                    cand[j, i] = 1
            elif cand[j, i] == 1:
                cand[j, i] = 0
                if rng.random() < 0.5 and cand[i, j] == 0:
                    cand[i, j] = 1
            else:
                cand[i, j] = 1
            graph = nx.DiGraph([(a, b) for a, b in zip(*np.where(cand == 1))])
            if not nx.is_directed_acyclic_graph(graph):
                continue
            coeffs, variances, bic = fit_linear_sem(cand, observational_data, interventions)
            proposals.append(Particle(adjacency=cand, weight=1.0, node_coef=coeffs, node_var=variances, bic_score=bic))
    merged = particles + proposals
    dedup: dict[bytes, Particle] = {}
    for particle in merged:
        key = particle.adjacency.tobytes()
        if key not in dedup or particle.bic_score < dedup[key].bic_score:
            dedup[key] = copy.deepcopy(particle)
    kept = sorted(dedup.values(), key=lambda p: p.bic_score)[:PARTICLE_LIMIT]
    return normalize_weights(kept)


def update_particles(particles: list[Particle], observational_data: np.ndarray, interventions: list[dict], seed: int) -> list[Particle]:
    if not particles:
        return build_particles(observational_data, interventions, seed)
    refreshed = []
    for particle in particles:
        coeffs, variances, bic = fit_linear_sem(particle.adjacency, observational_data, interventions)
        refreshed.append(Particle(adjacency=particle.adjacency.copy(), weight=particle.weight, node_coef=coeffs, node_var=variances, bic_score=bic))
    refreshed = normalize_weights(refreshed)
    if effective_sample_size(refreshed) < 4.0:
        refreshed = rejuvenate_particles(refreshed, observational_data, interventions, seed + 17)
    return refreshed


def best_particle(particles: list[Particle]) -> Particle:
    return max(particles, key=lambda p: p.weight)


def edge_state_probs(particles: list[Particle]) -> dict[tuple[int, int], dict[str, float]]:
    p = particles[0].adjacency.shape[0]
    weights = np.asarray([part.weight for part in particles], dtype=float)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for i, j in itertools.combinations(range(p), 2):
        fwd = sum(weights[k] for k, part in enumerate(particles) if part.adjacency[i, j] == 1)
        rev = sum(weights[k] for k, part in enumerate(particles) if part.adjacency[j, i] == 1)
        nul = max(0.0, 1.0 - fwd - rev)
        out[(i, j)] = {"fwd": float(fwd), "rev": float(rev), "none": float(nul)}
    return out


def ambiguity_stats(particles: list[Particle], threshold: float = 0.15) -> list[dict]:
    stats = []
    for (i, j), probs in edge_state_probs(particles).items():
        vec = np.asarray([probs["fwd"], probs["rev"], probs["none"]], dtype=float) + 1e-12
        ambiguity = float(-(vec * np.log(vec)).sum() / np.log(3.0))
        removable = float(1.0 - np.square(vec).sum())
        if ambiguity >= threshold:
            stats.append({"edge": [i, j], "p_fwd": probs["fwd"], "p_rev": probs["rev"], "p_0": probs["none"], "A": ambiguity, "R": removable})
    return stats


def _local_nodes(adjacency: np.ndarray, i: int, j: int, intervene: int) -> list[int]:
    nbrs = set([i, j])
    for node in [i, j]:
        nbrs.update(np.where(adjacency[node] == 1)[0].tolist())
        nbrs.update(np.where(adjacency[:, node] == 1)[0].tolist())
    nbrs.discard(intervene)
    return sorted(nbrs)


def _interventional_cov(adjacency: np.ndarray, coeffs: list[np.ndarray], variances: np.ndarray, intervene: int, nodes: list[int]) -> np.ndarray:
    mat = coefficients_to_matrix(coeffs).copy()
    mat[:, intervene] = 0.0
    transform = np.linalg.inv(np.eye(mat.shape[0]) - mat.T)
    cov = transform @ np.diag(variances) @ transform.T
    cov[intervene, :] = 0.0
    cov[:, intervene] = 0.0
    sub = cov[np.ix_(nodes, nodes)]
    return sub + np.eye(len(nodes)) * 1e-6


def _alt_graph_for_edge(particle: Particle, i: int, j: int) -> np.ndarray:
    alt = particle.adjacency.copy()
    if alt[i, j] == 1:
        alt[i, j] = 0
        if alt[j, i] == 0:
            alt[j, i] = 1
            if not nx.is_directed_acyclic_graph(nx.DiGraph([(a, b) for a, b in zip(*np.where(alt == 1))])):
                alt[j, i] = 0
    elif alt[j, i] == 1:
        alt[j, i] = 0
        if alt[i, j] == 0:
            alt[i, j] = 1
            if not nx.is_directed_acyclic_graph(nx.DiGraph([(a, b) for a, b in zip(*np.where(alt == 1))])):
                alt[i, j] = 0
    else:
        alt[i, j] = 1
        if not nx.is_directed_acyclic_graph(nx.DiGraph([(a, b) for a, b in zip(*np.where(alt == 1))])):
            alt[i, j] = 0
            alt[j, i] = 1
            if not nx.is_directed_acyclic_graph(nx.DiGraph([(a, b) for a, b in zip(*np.where(alt == 1))])):
                alt[j, i] = 0
    return alt


def _alt_sem_from_particle(particle: Particle, alt_adj: np.ndarray, i: int, j: int) -> tuple[list[np.ndarray], np.ndarray]:
    coeffs = [coef.copy() for coef in particle.node_coef]
    avg_mag = 0.2
    nonzero = np.abs(coefficients_to_matrix(particle.node_coef))
    if np.any(nonzero > 0):
        avg_mag = float(np.mean(nonzero[nonzero > 0]))
    for node in [i, j]:
        mask = alt_adj[:, node] == 0
        coeffs[node][mask] = 0.0
    if alt_adj[i, j] == 1 and coeffs[j][i] == 0.0:
        coeffs[j][i] = avg_mag
    if alt_adj[j, i] == 1 and coeffs[i][j] == 0.0:
        coeffs[i][j] = avg_mag
    return coeffs, particle.node_var.copy()


def gaussian_kl(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    k = cov_a.shape[0]
    sign1, logdet1 = np.linalg.slogdet(cov_a)
    sign2, logdet2 = np.linalg.slogdet(cov_b)
    if sign1 <= 0 or sign2 <= 0:
        return 0.0
    inv_b = np.linalg.inv(cov_b)
    return float(0.5 * (np.trace(inv_b @ cov_a) - k + logdet2 - logdet1))


def particle_topological_order(adjacency: np.ndarray) -> list[int]:
    graph = nx.DiGraph([(i, j) for i, j in zip(*np.where(adjacency == 1))])
    order = list(nx.topological_sort(graph))
    if len(order) < adjacency.shape[0]:
        missing = [node for node in range(adjacency.shape[0]) if node not in order]
        order.extend(missing)
    return order


def sample_batch_from_particle(
    particle: Particle,
    batch_size: int,
    intervene_node: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = particle.adjacency.shape[0]
    out = np.zeros((batch_size, p), dtype=float)
    noise = rng.normal(0.0, np.sqrt(particle.node_var), size=(batch_size, p))
    for node in particle_topological_order(particle.adjacency):
        if node == intervene_node:
            out[:, node] = 0.0
            continue
        parents = np.where(particle.adjacency[:, node] == 1)[0]
        if len(parents):
            out[:, node] = out[:, parents] @ particle.node_coef[node][parents] + noise[:, node]
        else:
            out[:, node] = noise[:, node]
    return out


def interventional_log_likelihood(
    particle: Particle,
    batch: np.ndarray,
    intervene_node: int,
) -> float:
    total = 0.0
    for node in range(batch.shape[1]):
        if node == intervene_node:
            continue
        parents = np.where(particle.adjacency[:, node] == 1)[0]
        if len(parents):
            mean = batch[:, parents] @ particle.node_coef[node][parents]
        else:
            mean = 0.0
        resid = batch[:, node] - mean
        var = max(float(particle.node_var[node]), 1e-6)
        total += float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + (resid**2) / var))
    return total


def posterior_from_batch(
    particles: list[Particle],
    batch: np.ndarray,
    intervene_node: int,
) -> np.ndarray:
    logw = np.log(np.asarray([max(p.weight, 1e-12) for p in particles], dtype=float))
    loglik = np.asarray([interventional_log_likelihood(p, batch, intervene_node) for p in particles], dtype=float)
    scores = logw + loglik
    scores -= scores.max()
    post = np.exp(scores)
    post /= post.sum()
    return post


def expected_entropy_reduction(
    particles: list[Particle],
    target: int,
    batch_size: int,
    base_seed: int,
    max_generators: int = 8,
) -> float:
    if not particles:
        return 0.0
    weight = np.asarray([p.weight for p in particles], dtype=float)
    cur_entropy = float(-(weight * np.log(weight + 1e-12)).sum())
    order = np.argsort(weight)[::-1][: max_generators]
    gen_weight = weight[order]
    gen_weight /= gen_weight.sum()
    expected_post_entropy = 0.0
    for rank, (idx, gen_w) in enumerate(zip(order, gen_weight)):
        batch = sample_batch_from_particle(particles[idx], batch_size, target, base_seed + 97 * rank + 11)
        post = posterior_from_batch(particles, batch, target)
        post_entropy = float(-(post * np.log(post + 1e-12)).sum())
        expected_post_entropy += float(gen_w * post_entropy)
    return max(0.0, cur_entropy - expected_post_entropy)


def git_expected_gradient_score(
    particles: list[Particle],
    target: int,
    batch_size: int,
    base_seed: int,
    max_generators: int = 8,
) -> float:
    if not particles:
        return 0.0
    weight = np.asarray([p.weight for p in particles], dtype=float)
    order = np.argsort(weight)[::-1][: max_generators]
    gen_weight = weight[order]
    gen_weight /= gen_weight.sum()
    total = 0.0
    for rank, (idx, gen_w) in enumerate(zip(order, gen_weight)):
        batch = sample_batch_from_particle(particles[idx], batch_size, target, base_seed + 193 * rank + 23)
        post = posterior_from_batch(particles, batch, target)
        grad = weight - post
        total += float(gen_w * np.linalg.norm(grad, ord=2))
    return total


def detectability(particles: list[Particle], edge: tuple[int, int], target: int, batch_size: int) -> tuple[float, int]:
    i, j = edge
    total = 0.0
    local_sizes = []
    edge_weight_sum = 0.0
    for particle in particles:
        unresolved = True
        if not unresolved:
            continue
        nodes = _local_nodes(particle.adjacency, i, j, target)
        local_sizes.append(len(nodes))
        if not nodes:
            continue
        alt_adj = _alt_graph_for_edge(particle, i, j)
        alt_coeffs, alt_vars = _alt_sem_from_particle(particle, alt_adj, i, j)
        cov = _interventional_cov(particle.adjacency, particle.node_coef, particle.node_var, target, nodes)
        alt_cov = _interventional_cov(alt_adj, alt_coeffs, alt_vars, target, nodes)
        kl = max(0.0, gaussian_kl(cov, alt_cov))
        lam = max(0.0, 2.0 * batch_size * kl)
        power = float(1.0 - ncx2.cdf(chi2.ppf(0.95, 1), 1, lam))
        total += particle.weight * power
        edge_weight_sum += particle.weight
    if edge_weight_sum <= 0:
        return 0.0, 0
    return float(total / edge_weight_sum), max(local_sizes) if local_sizes else 0


def graph_posterior_entropy(particles: list[Particle]) -> float:
    w = np.asarray([p.weight for p in particles], dtype=float) + 1e-12
    return float(-(w * np.log(w)).sum())


def save_particles(path: Path, particles: list[Particle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(particles, handle)


def load_particles(path: Path) -> list[Particle]:
    with path.open("rb") as handle:
        return pickle.load(handle)
