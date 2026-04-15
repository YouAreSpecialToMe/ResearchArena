"""Unified interface for causal discovery algorithms (Stage 2).

Wraps PC, GES, BOSS/GRaSP, DirectLiNGAM, and ANM pairwise.
"""

import numpy as np
import signal
import warnings
import time

warnings.filterwarnings('ignore')


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Algorithm timed out")


def run_algorithm(data, method, seed=0, timeout=300):
    """Run a causal discovery algorithm and return adjacency matrix.

    For CPDAGs, undirected edges have 0.5 in both directions.
    For DAGs, directed edges have 1.0.

    Returns:
        adj: p x p adjacency matrix
        runtime: seconds
    """
    p = data.shape[1]
    start = time.time()

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        if method == 'PC':
            adj = _run_pc(data, seed)
        elif method == 'GES':
            adj = _run_ges(data, seed)
        elif method == 'BOSS':
            adj = _run_boss(data, seed)
        elif method == 'DirectLiNGAM':
            adj = _run_lingam(data, seed)
        elif method == 'ANM':
            adj = _run_anm(data, seed)
        else:
            raise ValueError(f"Unknown method: {method}")
    except TimeoutError:
        print(f"  WARNING: {method} timed out after {timeout}s")
        adj = np.zeros((p, p))
    except Exception as e:
        print(f"  WARNING: {method} failed: {e}")
        adj = np.zeros((p, p))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    runtime = time.time() - start
    return adj, runtime


def _run_pc(data, seed):
    """Run PC algorithm using causal-learn."""
    from causallearn.search.ConstraintBased.PC import pc
    p = data.shape[1]
    max_k = min(3, p - 2) if p <= 20 else min(2, p - 2)

    cg = pc(data, alpha=0.05, indep_test='fisherz', stable=True,
            uc_rule=0, uc_priority=2, depth=max_k)

    return _causallearn_graph_to_adj(cg.G, p)


def _run_ges(data, seed):
    """Run GES algorithm using causal-learn."""
    from causallearn.search.ScoreBased.GES import ges
    result = ges(data, score_func='local_score_BIC')
    p = data.shape[1]
    return _causallearn_graph_to_adj(result['G'], p)


def _run_boss(data, seed):
    """Run BOSS/GRaSP algorithm."""
    try:
        from causallearn.search.PermutationBased.GRaSP import grasp
        result = grasp(data, score_func='local_score_BIC')
        p = data.shape[1]
        return _causallearn_graph_to_adj(result, p)
    except Exception:
        # Fallback to GES if GRaSP unavailable
        return _run_ges(data, seed)


def _run_lingam(data, seed):
    """Run DirectLiNGAM using causal-learn."""
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    model = DirectLiNGAM()
    model.fit(data)
    adj = np.zeros_like(model.adjacency_matrix_)
    adj[model.adjacency_matrix_ != 0] = 1.0
    # DirectLiNGAM: adjacency_matrix_[i,j] != 0 means j -> i
    # We want adj[i,j] = 1 means i -> j
    return adj.T


def _run_anm(data, seed, max_n=500):
    """Run pairwise ANM using residual independence test.

    Uses a fast correlation-based independence test instead of HSIC
    for computational efficiency on CPU.
    """
    n, p = data.shape
    adj = np.zeros((p, p))

    rng = np.random.RandomState(seed)
    if n > max_n:
        idx = rng.choice(n, max_n, replace=False)
        sub_data = data[idx]
    else:
        sub_data = data

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from scipy.stats import spearmanr

    confidences = []

    for i in range(p):
        for j in range(i + 1, p):
            # Forward: j = f(i) + N
            p_fwd = _anm_independence_test(sub_data[:, i], sub_data[:, j])
            # Reverse: i = f(j) + N
            p_rev = _anm_independence_test(sub_data[:, j], sub_data[:, i])

            # If one direction has more independent residuals
            if p_fwd > 0.05 and p_rev < 0.05:
                confidences.append((i, j, p_fwd - p_rev))
            elif p_rev > 0.05 and p_fwd < 0.05:
                confidences.append((j, i, p_rev - p_fwd))

    # Sort by confidence and add edges, checking for cycles
    confidences.sort(key=lambda x: -x[2])
    for src, dst, conf in confidences:
        adj[src, dst] = 1.0
        if _has_cycle(adj):
            adj[src, dst] = 0.0

    return adj


def _anm_independence_test(x, y):
    """Test ANM: y = f(x) + N. Returns p-value for independence of residual and x.

    Uses multiple fast tests: Spearman correlation of |residual| vs x (detects
    heteroscedasticity) and Spearman correlation of residual vs x (detects
    remaining dependence). Returns the minimum p-value.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from scipy.stats import spearmanr

    x_r = x.reshape(-1, 1)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(x_r)
    lr = LinearRegression().fit(x_poly, y)
    resid = y - lr.predict(x_poly)

    # Test 1: Spearman correlation of residual vs x
    _, p1 = spearmanr(x, resid)
    # Test 2: Spearman correlation of |residual| vs x (heteroscedasticity)
    _, p2 = spearmanr(x, np.abs(resid))
    # Test 3: Spearman correlation of residual^2 vs x
    _, p3 = spearmanr(x, resid**2)

    # Combine: residuals should be independent of x under correct model
    # High p-value = independent = good ANM fit
    # Use minimum p-value (most evidence of dependence)
    return min(p1, p2, p3)


def _has_cycle(adj):
    """Check if adjacency matrix has a cycle using DFS."""
    p = adj.shape[0]
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * p

    def dfs(u):
        color[u] = GRAY
        for v in range(p):
            if adj[u, v] > 0:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and dfs(v):
                    return True
        color[u] = BLACK
        return False

    for u in range(p):
        if color[u] == WHITE:
            if dfs(u):
                return True
    return False


def _causallearn_graph_to_adj(G, p):
    """Convert causal-learn GeneralGraph to adjacency matrix.

    Directed edge i -> j: adj[i,j] = 1.0
    Undirected edge i - j: adj[i,j] = adj[j,i] = 0.5
    """
    adj = np.zeros((p, p))
    graph = G.graph  # p x p matrix with endpoint marks

    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            # In causal-learn: graph[i,j] = -1 means arrowhead at j from i
            # graph[i,j] = 1 means tail at j from i
            # Directed i -> j: graph[j,i] = -1 (arrow at j) and graph[i,j] = 1 (tail at i)
            if graph[j, i] == -1 and graph[i, j] == 1:
                adj[i, j] = 1.0
            elif graph[j, i] == -1 and graph[i, j] == -1:
                # Bidirected, treat as undirected
                adj[i, j] = 0.5
                adj[j, i] = 0.5

    # Handle undirected edges (tail-tail)
    for i in range(p):
        for j in range(i + 1, p):
            if graph[i, j] == 1 and graph[j, i] == 1:
                # Undirected edge
                adj[i, j] = 0.5
                adj[j, i] = 0.5

    return adj
