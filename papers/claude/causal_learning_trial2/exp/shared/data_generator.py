"""Synthetic data generator for AACD experiments.

Supports 6 data types: HLG, HLNG, HNL, HM, SH, NU.
"""

import numpy as np
from scipy import stats


def generate_dag(p, expected_degree, seed, model='random_dag'):
    """Generate a random DAG as a p x p binary adjacency matrix."""
    rng = np.random.RandomState(seed)
    # Random topological ordering
    perm = rng.permutation(p)
    dag = np.zeros((p, p), dtype=int)
    edge_prob = expected_degree / max(p - 1, 1)
    for i in range(p):
        for j in range(i + 1, p):
            if rng.random() < edge_prob:
                # Edge from perm[i] to perm[j] (respects topological order)
                dag[perm[i], perm[j]] = 1
    return dag


def _sample_weight(rng):
    """Sample edge weight from U([-1,-0.5] union [0.5,1])."""
    w = rng.uniform(0.5, 1.0)
    if rng.random() < 0.5:
        w = -w
    return w


def _nonlinear_func(x, func_type):
    """Apply nonlinear function."""
    if func_type == 0:
        return x + 0.5 * x**2
    elif func_type == 1:
        return np.tanh(2 * x)
    else:
        return np.sin(x) + x


def generate_data(dag, n, data_type, seed, dominant_type='LG'):
    """Generate data from a SEM with given DAG and data type.

    Returns:
        data: n x p array
        mechanism_labels: list of (i, j, mechanism_type) for each edge
    """
    rng = np.random.RandomState(seed)
    p = dag.shape[0]

    # Get edges in topological order
    # Find topological order via DFS
    topo_order = _topological_sort(dag)

    # Assign mechanism types to edges
    edges = list(zip(*np.where(dag == 1)))
    mechanism_labels = []

    if data_type == 'HLG':
        mechanisms = {(i, j): 'LG' for i, j in edges}
    elif data_type == 'HLNG':
        mechanisms = {(i, j): 'LNG' for i, j in edges}
    elif data_type == 'HNL':
        mechanisms = {}
        for i, j in edges:
            mechanisms[(i, j)] = rng.choice(['NLG', 'NLNG'])
    elif data_type == 'HM':
        types = ['LG', 'LNG', 'NLG', 'NLNG']
        mechanisms = {(i, j): rng.choice(types) for i, j in edges}
    elif data_type == 'SH':
        types_map = {'LG': ['LNG', 'NLG', 'NLNG'],
                     'LNG': ['LG', 'NLG', 'NLNG'],
                     'NLG': ['LG', 'LNG', 'NLNG'],
                     'NLNG': ['LG', 'LNG', 'NLG']}
        mechanisms = {}
        for i, j in edges:
            if rng.random() < 0.7:
                mechanisms[(i, j)] = dominant_type
            else:
                mechanisms[(i, j)] = rng.choice(types_map[dominant_type])
    elif data_type == 'NU':
        mechanisms = {(i, j): 'LG' for i, j in edges}
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    for (i, j), m in mechanisms.items():
        mechanism_labels.append((i, j, m))

    # Generate data
    data = np.zeros((n, p))
    weights = {}
    func_types = {}

    for (i, j) in edges:
        weights[(i, j)] = _sample_weight(rng)
        func_types[(i, j)] = rng.randint(0, 3)

    # Near-unfaithful: adjust weights to create near-cancellation
    if data_type == 'NU':
        _create_near_unfaithful(dag, weights, rng)

    for node in topo_order:
        parents = np.where(dag[:, node] == 1)[0]
        if len(parents) == 0:
            # Root node
            data[:, node] = rng.randn(n)
        else:
            signal = np.zeros(n)
            for pa in parents:
                mech = mechanisms[(pa, node)]
                w = weights[(pa, node)]
                if mech in ['LG', 'LNG']:
                    signal += w * data[:, pa]
                else:  # nonlinear
                    ft = func_types[(pa, node)]
                    signal += w * _nonlinear_func(data[:, pa], ft)

            # Add noise
            mech = mechanisms[(parents[0], node)]  # Use first parent's mechanism for noise
            if mech == 'LG':
                noise = rng.randn(n)
            elif mech == 'LNG':
                if rng.random() < 0.5:
                    noise = rng.laplace(0, 1, n)
                else:
                    noise = rng.uniform(-np.sqrt(3), np.sqrt(3), n)
            elif mech == 'NLG':
                noise = rng.randn(n) * 0.5
            else:  # NLNG
                noise = rng.laplace(0, 0.5, n)

            data[:, node] = signal + noise

    # Standardize
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-10)

    return data, mechanism_labels


def _topological_sort(dag):
    """Topological sort of DAG."""
    p = dag.shape[0]
    in_degree = dag.sum(axis=0).astype(int)
    queue = [i for i in range(p) if in_degree[i] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for j in range(p):
            if dag[node, j] == 1:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    queue.append(j)
    return order


def _create_near_unfaithful(dag, weights, rng, fraction=0.2):
    """Adjust weights to create near-cancellation on ~20% of 2-hop paths."""
    p = dag.shape[0]
    # Find 2-hop paths: i -> k -> j where i -> j also exists
    cancellation_targets = []
    for i in range(p):
        for j in range(p):
            if dag[i, j] == 1:
                # Check for 2-hop via some k
                for k in range(p):
                    if dag[i, k] == 1 and dag[k, j] == 1:
                        cancellation_targets.append((i, k, j))

    if not cancellation_targets:
        return

    n_cancel = max(1, int(len(cancellation_targets) * fraction))
    selected = rng.choice(len(cancellation_targets), min(n_cancel, len(cancellation_targets)), replace=False)

    for idx in selected:
        i, k, j = cancellation_targets[idx]
        # Set direct weight to nearly cancel indirect path
        indirect = weights[(i, k)] * weights[(k, j)]
        # Near cancellation: ratio 0.9-1.1
        ratio = rng.uniform(0.9, 1.1)
        weights[(i, j)] = -indirect * ratio


def verify_dag(dag):
    """Verify DAG is acyclic."""
    p = dag.shape[0]
    order = _topological_sort(dag)
    return len(order) == p
