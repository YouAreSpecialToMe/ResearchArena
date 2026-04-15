"""AACD Aggregation module (Stage 3).

Implements sigmoid/learned weighting, edge-level aggregation,
acyclicity enforcement, and threshold selection.
"""

import numpy as np
from collections import defaultdict


# Algorithm assumption mappings
# Each algorithm maps to: (diagnostic_key, sign, description)
# sign = +1 means algorithm benefits from HIGH diagnostic score
# sign = -1 means algorithm benefits from LOW diagnostic score
ALGORITHM_ASSUMPTIONS = {
    'PC': [('D4', -1, 'low faithfulness proximity')],
    'GES': [('D4', -1, 'low faithfulness proximity')],
    'BOSS': [('D4', -1, 'low faithfulness proximity')],
    'DirectLiNGAM': [('D1', +1, 'high linearity'), ('D2', +1, 'high non-Gaussianity')],
    'ANM': [('D3', +1, 'high ANM asymmetry'), ('D1', -1, 'nonlinearity helps')],
}


def sigmoid(x, s=5, t=0.5):
    """Sigmoid function mapping diagnostic score to [0,1]."""
    return 1.0 / (1.0 + np.exp(-s * (x - t)))


def compute_algorithm_weights(diagnostics, algorithms, params=None, confidence_weighting=True):
    """Compute per-algorithm weights based on diagnostics.

    Args:
        diagnostics: dict from compute_diagnostics()
        algorithms: list of algorithm names
        params: dict with 't' (threshold) and 's' (steepness) for sigmoid
        confidence_weighting: whether to apply sample-size confidence weights

    Returns:
        weights: dict mapping algorithm name to weight
    """
    if params is None:
        params = {'t': 0.5, 's': 5}

    t = params['t']
    s = params['s']
    conf = diagnostics['confidence_weights']
    summary = diagnostics['global_summary']

    # Map diagnostic keys to global summary values
    diag_values = {
        'D1': summary['avg_linearity'],
        'D2': summary['avg_nongaussianity'],
        'D3': summary['avg_anm_score'],
        'D4': summary['avg_faithfulness_proximity'],
        'D5': summary['avg_homoscedasticity'],
    }

    weights = {}
    for algo in algorithms:
        if algo not in ALGORITHM_ASSUMPTIONS:
            weights[algo] = 1.0
            continue

        w = 1.0
        for diag_key, sign, desc in ALGORITHM_ASSUMPTIONS[algo]:
            d = diag_values[diag_key]
            # Apply sigmoid: if sign=+1, high d -> high weight
            # if sign=-1, high d -> low weight (use 1-d)
            if sign > 0:
                phi = sigmoid(d, s, t)
            else:
                phi = sigmoid(1 - d, s, t)

            # Apply confidence weighting
            if confidence_weighting:
                c = conf[diag_key]
                # When confidence is low, pull toward 0.5 (neutral)
                phi = 0.5 + c * (phi - 0.5)

            w *= phi

        weights[algo] = max(w, 0.01)  # Floor to prevent zero weights

    return weights


def aggregate_edges(algorithm_outputs, weights, algorithms):
    """Compute edge-level confidence scores.

    Args:
        algorithm_outputs: dict mapping algo name to adjacency matrix
        weights: dict mapping algo name to weight
        algorithms: list of algorithm names

    Returns:
        confidence: p x p confidence matrix
    """
    p = list(algorithm_outputs.values())[0].shape[0]
    confidence = np.zeros((p, p))
    total_weight = sum(weights[a] for a in algorithms if a in weights)

    if total_weight < 1e-10:
        total_weight = 1.0

    for algo in algorithms:
        if algo not in algorithm_outputs or algo not in weights:
            continue
        w = weights[algo]
        adj = algorithm_outputs[algo]
        confidence += w * adj

    confidence /= total_weight
    return confidence


def enforce_acyclicity_eades(confidence, threshold=0.0):
    """Enforce acyclicity using weighted Eades heuristic.

    Greedily assigns nodes to a linear ordering based on
    outgoing vs incoming edge weights.

    Returns:
        dag: p x p binary adjacency matrix
        ordering: topological ordering used
    """
    p = confidence.shape[0]
    # Build weighted digraph from confidence
    C = confidence.copy()
    C[C < threshold] = 0

    # Eades heuristic for minimum feedback arc set
    remaining = set(range(p))
    left = []  # Front of ordering
    right = []  # Back of ordering

    while remaining:
        # Find sinks (no outgoing to remaining)
        sinks = [v for v in remaining
                 if all(C[v, u] == 0 for u in remaining if u != v)]
        while sinks:
            s = sinks.pop()
            right.insert(0, s)
            remaining.discard(s)
            sinks = [v for v in remaining
                     if all(C[v, u] == 0 for u in remaining if u != v)]

        if not remaining:
            break

        # Find sources (no incoming from remaining)
        sources = [v for v in remaining
                   if all(C[u, v] == 0 for u in remaining if u != v)]
        while sources:
            s = sources.pop()
            left.append(s)
            remaining.discard(s)
            sources = [v for v in remaining
                       if all(C[u, v] == 0 for u in remaining if u != v)]

        if not remaining:
            break

        # Pick node with max (out_weight - in_weight)
        best_node = None
        best_delta = -np.inf
        for v in remaining:
            out_w = sum(C[v, u] for u in remaining if u != v)
            in_w = sum(C[u, v] for u in remaining if u != v)
            delta = out_w - in_w
            if delta > best_delta:
                best_delta = delta
                best_node = v
        left.append(best_node)
        remaining.discard(best_node)

    ordering = left + right

    # Build DAG respecting ordering
    order_map = {node: idx for idx, node in enumerate(ordering)}
    dag = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j and C[i, j] > 0 and order_map[i] < order_map[j]:
                dag[i, j] = 1.0

    return dag, ordering


def enforce_acyclicity_greedy(confidence, threshold=0.0):
    """Enforce acyclicity by greedily removing lowest-confidence cycle edges."""
    p = confidence.shape[0]
    dag = (confidence > threshold).astype(float)

    while True:
        cycle = _find_cycle(dag)
        if cycle is None:
            break
        # Find lowest-confidence edge in cycle
        min_conf = np.inf
        min_edge = None
        for k in range(len(cycle)):
            i, j = cycle[k], cycle[(k + 1) % len(cycle)]
            if confidence[i, j] < min_conf:
                min_conf = confidence[i, j]
                min_edge = (i, j)
        if min_edge:
            dag[min_edge[0], min_edge[1]] = 0

    return dag


def _find_cycle(adj):
    """Find a cycle in adjacency matrix. Returns cycle as list of nodes or None."""
    p = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * p
    parent = [-1] * p

    def dfs(u, path):
        color[u] = GRAY
        path.append(u)
        for v in range(p):
            if adj[u, v] > 0:
                if color[v] == GRAY:
                    # Found cycle
                    idx = path.index(v)
                    return path[idx:]
                if color[v] == WHITE:
                    result = dfs(v, path)
                    if result is not None:
                        return result
        path.pop()
        color[u] = BLACK
        return None

    for u in range(p):
        if color[u] == WHITE:
            result = dfs(u, [])
            if result is not None:
                return result
    return None


def run_aacd(data, algorithm_outputs, algorithms, params=None,
             confidence_weighting=True, tau=0.5, acyclicity='eades',
             diagnostics=None, n_star=None, excluded_diagnostics=None):
    """Run full AACD pipeline.

    Args:
        data: n x p array
        algorithm_outputs: dict of algo -> adjacency matrix
        algorithms: list of algorithm names
        params: sigmoid parameters {'t': float, 's': float}
        confidence_weighting: use sample-size confidence
        tau: threshold for edge inclusion
        acyclicity: 'eades' or 'greedy'
        diagnostics: precomputed diagnostics (optional)
        n_star: diagnostic confidence thresholds
        excluded_diagnostics: list of diagnostic keys to exclude (for ablation)

    Returns dict with dag, confidence_matrix, diagnostics, weights.
    """
    from .diagnostics import compute_diagnostics as cd

    if diagnostics is None:
        diagnostics = cd(data, n_star=n_star)

    # Handle ablation: set excluded diagnostics to neutral (0.5)
    if excluded_diagnostics:
        for dk in excluded_diagnostics:
            diag_map = {
                'D1': 'avg_linearity', 'D2': 'avg_nongaussianity',
                'D3': 'avg_anm_score', 'D4': 'avg_faithfulness_proximity',
                'D5': 'avg_homoscedasticity'
            }
            if dk in diag_map:
                diagnostics['global_summary'][diag_map[dk]] = 0.5

    weights = compute_algorithm_weights(diagnostics, algorithms, params, confidence_weighting)
    confidence = aggregate_edges(algorithm_outputs, weights, algorithms)

    if acyclicity == 'eades':
        dag, ordering = enforce_acyclicity_eades(confidence, tau)
    else:
        dag = enforce_acyclicity_greedy(confidence, tau)
        ordering = None

    return {
        'dag': dag,
        'confidence_matrix': confidence,
        'diagnostics': diagnostics,
        'weights': weights,
        'ordering': ordering,
    }


def run_naive_ensemble(algorithm_outputs, algorithms, tau=0.5, acyclicity='eades'):
    """Run naive (equal-weight) ensemble."""
    weights = {a: 1.0 for a in algorithms}
    confidence = aggregate_edges(algorithm_outputs, weights, algorithms)

    if acyclicity == 'eades':
        dag, _ = enforce_acyclicity_eades(confidence, tau)
    else:
        dag = enforce_acyclicity_greedy(confidence, tau)

    return dag, confidence
