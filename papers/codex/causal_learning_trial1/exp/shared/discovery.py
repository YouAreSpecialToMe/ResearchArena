from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from scipy.optimize import minimize
from scipy.special import expit


@dataclass
class DiscoveryResult:
    cpdag: np.ndarray
    metadata: dict


def run_pc(data: np.ndarray, alpha: float = 0.01) -> DiscoveryResult:
    res = pc(data, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False)
    cpdag = np.asarray(res.G.graph, dtype=int)
    return DiscoveryResult(cpdag=cpdag, metadata={"alpha": alpha, "runtime_seconds": float(res.PC_elapsed)})


def run_ges(data: np.ndarray) -> DiscoveryResult:
    res = ges(data)
    cpdag = np.asarray(res["G"].graph, dtype=int)
    score = np.asarray(res["score"]).reshape(-1)
    return DiscoveryResult(cpdag=cpdag, metadata={"score": float(score[-1]) if len(score) else 0.0})


def run_notears_linear(
    data: np.ndarray,
    lambda1: float,
    max_iter: int,
    edge_threshold: float,
) -> tuple[np.ndarray, dict]:
    X = data
    n, d = X.shape

    def _adj(w: np.ndarray) -> np.ndarray:
        w_pos = w[: d * d].reshape(d, d)
        w_neg = w[d * d :].reshape(d, d)
        W = w_pos - w_neg
        np.fill_diagonal(W, 0.0)
        return W

    def _h(W: np.ndarray) -> float:
        return float(np.trace(np.linalg.matrix_power(np.eye(d) + (W * W) / d, d)) - d)

    def _loss(W: np.ndarray) -> tuple[float, np.ndarray]:
        R = X - X @ W
        loss = 0.5 / n * np.sum(R * R)
        grad = -1.0 / n * X.T @ R
        return float(loss), grad

    def _func(w: np.ndarray, rho: float, alpha: float) -> tuple[float, np.ndarray]:
        W = _adj(w)
        loss, grad = _loss(W)
        h_val = _h(W)
        smooth_grad = grad + (rho * h_val + alpha) * (2.0 * W) @ np.linalg.matrix_power(
            np.eye(d) + (W * W) / d, d - 1
        )
        obj = loss + lambda1 * w.sum() + 0.5 * rho * h_val * h_val + alpha * h_val
        grad_pos = smooth_grad + lambda1
        grad_neg = -smooth_grad + lambda1
        grad_full = np.concatenate([grad_pos.ravel(), grad_neg.ravel()])
        diag_idx = np.arange(d)
        for i in diag_idx:
            grad_full[i * d + i] = 0.0
            grad_full[d * d + i * d + i] = 0.0
        return obj, grad_full

    w_est = np.zeros(2 * d * d)
    bnds = []
    for i in range(d):
        for j in range(d):
            bound = (0.0, 0.0) if i == j else (0.0, None)
            bnds.append(bound)
    bnds = bnds + bnds
    rho, alpha, h_prev = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        while rho < 1e16:
            sol = minimize(
                lambda w: _func(w, rho, alpha),
                w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds,
            )
            w_new = sol.x
            W_new = _adj(w_new)
            h_new = _h(W_new)
            if (not math.isfinite(h_prev)) or h_new <= 0.25 * h_prev:
                break
            rho *= 10.0
        w_est = w_new
        alpha += rho * h_new
        h_prev = h_new
        if h_new <= 1e-8 or rho >= 1e16:
            break

    W = _adj(w_est)
    W[np.abs(W) < edge_threshold] = 0.0
    dag = _project_to_dag(W)
    from .graph_utils import dag_to_cpdag_matrix
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(d))
    for i in range(d):
        for j in range(d):
            if i != j and dag[i, j] != 0:
                g.add_edge(i, j)
    cpdag = dag_to_cpdag_matrix(g)
    return cpdag, {"nonzero_weights": int(np.count_nonzero(dag))}


def _project_to_dag(W: np.ndarray) -> np.ndarray:
    import networkx as nx

    d = W.shape[0]
    edges = sorted(
        [(abs(W[i, j]), i, j) for i in range(d) for j in range(d) if i != j and W[i, j] != 0.0],
        reverse=True,
    )
    dag = np.zeros_like(W)
    g = nx.DiGraph()
    g.add_nodes_from(range(d))
    for _, i, j in edges:
        g.add_edge(i, j)
        if nx.is_directed_acyclic_graph(g):
            dag[i, j] = W[i, j]
        else:
            g.remove_edge(i, j)
    return dag
