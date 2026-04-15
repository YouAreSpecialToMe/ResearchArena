"""
Fast experiment runner for SPICED - optimized for 8-hour time budget.
Addresses all feedback issues with efficient implementation.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
import json
import time
import glob
from collections import defaultdict

from shared.data_generator import (
    generate_er_dag, generate_sf_dag,
    generate_linear_gaussian_data, generate_linear_nongaussian_data
)
from shared.metrics import compute_all_metrics

# Import SPICED components
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp/spiced'))
from phase1_skeleton_fixed import knn_mutual_information
from phase2_constraints import extract_structural_constraints, create_constraint_penalty_matrix
from phase3_optimization_fixed import constrained_optimization


def fast_knn_mi(x, y, k=3):
    """Faster k-NN MI estimation with smaller k."""
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import digamma
    
    n = len(x)
    xy = np.column_stack([x, y])
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), metric='chebyshev', algorithm='kd_tree').fit(xy)
    distances, _ = nbrs.kneighbors(xy)
    
    epsilon = distances[:, k]
    epsilon = np.maximum(epsilon, 1e-10)
    
    # Count neighbors in marginal spaces
    nx = np.zeros(n)
    ny = np.zeros(n)
    
    for i in range(n):
        nx[i] = np.sum(np.abs(x - x[i]) < epsilon[i]) - 1
        ny[i] = np.sum(np.abs(y - y[i]) < epsilon[i]) - 1
    
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    
    mi = digamma(k) - np.mean(digamma(nx) + digamma(ny)) + digamma(n)
    return max(0, mi)


def compute_skeleton_fast(data, k=3):
    """Fast skeleton discovery using k-NN MI."""
    n_samples, n_nodes = data.shape
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    # Compute pairwise MI matrix
    mi_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            mi = fast_knn_mi(data[:, i], data[:, j], k=k)
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    # Threshold selection
    mi_values = mi_matrix[np.triu_indices(n_nodes, k=1)]
    if len(mi_values) == 0 or np.all(mi_values == 0):
        return np.zeros((n_nodes, n_nodes), dtype=int)
    
    median_mi = np.median(mi_values)
    mad = np.median(np.abs(mi_values - median_mi))
    threshold = median_mi + 2 * mad
    
    min_edges = min(n_nodes, max(3, n_nodes // 2))
    if np.sum(mi_values > threshold) < min_edges:
        threshold = np.percentile(mi_values, 50)
    
    skeleton = (mi_matrix > threshold).astype(int)
    np.fill_diagonal(skeleton, 0)
    
    return skeleton


def compute_scores_fast(data, skeleton, k=3):
    """Fast directed information scores."""
    n_samples, n_nodes = data.shape
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    score_matrix = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if skeleton[i, j] == 0 or i == j:
                continue
            
            mi = fast_knn_mi(data[:, i], data[:, j], k=k)
            score_matrix[i, j] = mi
    
    return score_matrix


def spiced_fast(data, use_it_init=True, use_structural_constraints=True, seed=None):
    """Fast SPICED implementation."""
    if seed is not None:
        np.random.seed(seed)
    
    timing = {}
    
    # Phase 1
    start = time.time()
    skeleton = compute_skeleton_fast(data, k=3)
    it_scores = compute_scores_fast(data, skeleton, k=3)
    timing['phase1'] = time.time() - start
    
    # Phase 2
    start = time.time()
    if use_structural_constraints:
        constraints = extract_structural_constraints(skeleton)
        constraint_matrix = create_constraint_penalty_matrix(constraints, data.shape[1])
    else:
        constraint_matrix = np.zeros((data.shape[1], data.shape[1]))
    timing['phase2'] = time.time() - start
    
    # Phase 3
    start = time.time()
    n_nodes = data.shape[1]
    max_iter = 50 if n_nodes <= 30 else 30
    
    pred_adj = constrained_optimization(
        data, skeleton, it_scores, constraint_matrix,
        lambda1=0.1, lambda2=0.0, lambda3=0.01 if use_structural_constraints else 0.0,
        max_iter=max_iter, w_threshold=0.3,
        use_it_init=use_it_init, seed=seed
    )
    timing['phase3'] = time.time() - start
    timing['total'] = timing['phase1'] + timing['phase2'] + timing['phase3']
    
    return pred_adj, timing


def notears_fast(X, lambda1=0.1, max_iter=50, w_threshold=0.3, seed=None):
    """Fast NOTEARS implementation."""
    from scipy.linalg import expm
    
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    W = np.random.randn(d, d) * 0.1
    np.fill_diagonal(W, 0)
    
    rho = 1.0
    alpha = 0.0
    h_tol = 1e-6
    
    for iteration in range(max_iter):
        W_old = W.copy()
        
        for j in range(d):
            residual = X @ W[:, j] - X[:, j]
            grad_ls = (1.0 / n) * (X.T @ residual)
            grad_l1 = lambda1 * np.sign(W[:, j])
            
            W[:, j] -= 0.1 * (grad_ls + grad_l1)
            W[:, j] = np.sign(W[:, j]) * np.maximum(np.abs(W[:, j]) - 0.001, 0)
            W[j, j] = 0
        
        W_sq = W * W
        try:
            exp_Wsq = expm(W_sq)
            h = np.trace(exp_Wsq) - d
        except:
            h = 1.0
        
        if h > h_tol:
            alpha += rho * h
            rho = min(rho * 2, 1e10)
            W *= 0.95
        
        if np.max(np.abs(W - W_old)) < 1e-4 and h < h_tol:
            break
    
    return (np.abs(W) > w_threshold).astype(int)


def pc_fast(data, alpha=0.05):
    """Fast PC implementation using causal-learn."""
    from causallearn.search.ConstraintBased.PC import pc
    
    cg = pc(data, alpha=alpha, indep_test='fisherz', stable=True, verbose=False)
    
    pred_adj = np.zeros((data.shape[1], data.shape[1]), dtype=int)
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if cg.G.graph[i, j] != 0:
                pred_adj[i, j] = 1
    
    return pred_adj


def run_method_on_datasets(datasets, method_name, config=None):
    """Run a method on a list of datasets."""
    results = []
    
    for i, dataset_file in enumerate(datasets):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(datasets)} - {method_name}")
        
        try:
            data_dict = np.load(dataset_file)
            data = data_dict['data']
            true_adj = data_dict['adj']
            
            graph_id = int(data_dict['graph_id'])
            mechanism = str(data_dict['mechanism'])
            n_samples = int(data_dict['n_samples'])
            seed = int(data_dict['seed'])
            n_nodes = true_adj.shape[0]
            
            start = time.time()
            
            if method_name == 'SPICED':
                pred_adj, timing = spiced_fast(data, use_it_init=True, use_structural_constraints=True, seed=seed)
                runtime = timing['total']
            elif method_name == 'SPICED_no_constraints':
                pred_adj, timing = spiced_fast(data, use_it_init=True, use_structural_constraints=False, seed=seed)
                runtime = timing['total']
            elif method_name == 'SPICED_no_init':
                pred_adj, timing = spiced_fast(data, use_it_init=False, use_structural_constraints=True, seed=seed)
                runtime = timing['total']
            elif method_name == 'NOTEARS':
                pred_adj = notears_fast(data, lambda1=0.1, max_iter=50, seed=seed)
                runtime = time.time() - start
            elif method_name == 'PC':
                pred_adj = pc_fast(data, alpha=0.05)
                runtime = time.time() - start
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            results.append({
                'graph_id': graph_id,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'seed': seed,
                'n_nodes': n_nodes,
                'runtime': runtime,
                **metrics
            })
            
        except Exception as e:
            print(f"  Error on {dataset_file}: {e}")
    
    return results


def main():
    print("=" * 70)
    print("SPICED Fast Experiment Runner (Optimized for 8-hour budget)")
    print("=" * 70)
    
    # Get datasets
    all_files = glob.glob(f"{PROJECT_ROOT}/data/processed/datasets/*.npz")
    print(f"\nTotal datasets: {len(all_files)}")
    
    # Organize by size
    by_size = defaultdict(list)
    for f in all_files:
        try:
            d = np.load(f)
            n_nodes = d['adj'].shape[0]
            by_size[n_nodes].append(f)
        except:
            pass
    
    for size in sorted(by_size.keys()):
        print(f"  n={size}: {len(by_size[size])} datasets")
    
    # Select target datasets - focus on key comparisons
    target_files = []
    
    # n=10: All N <= 200, linear mechanisms
    if 10 in by_size:
        target_files.extend([f for f in by_size[10] 
                            if any(f'N{N}' in f for N in [50, 100, 200]) 
                            and 'linear' in f][:60])
    
    # n=20: All N <= 200, linear mechanisms
    if 20 in by_size:
        target_files.extend([f for f in by_size[20] 
                            if any(f'N{N}' in f for N in [50, 100, 200]) 
                            and 'linear' in f][:60])
    
    # n=30: N=50, 100 only, linear gaussian
    if 30 in by_size:
        target_files.extend([f for f in by_size[30] 
                            if any(f'N{N}' in f for N in [50, 100]) 
                            and 'linear_gaussian' in f][:30])
    
    # n=50: N=50, 100 only, limited set
    if 50 in by_size:
        target_files.extend([f for f in by_size[50] 
                            if any(f'N{N}' in f for N in [50, 100]) 
                            and 'linear_gaussian' in f][:20])
    
    print(f"\nSelected {len(target_files)} datasets for experiments")
    
    # Run experiments
    os.makedirs(f"{PROJECT_ROOT}/results/synthetic", exist_ok=True)
    os.makedirs(f"{PROJECT_ROOT}/results/ablations", exist_ok=True)
    
    # Main comparisons
    print("\n" + "=" * 70)
    print("1. Running SPICED...")
    print("=" * 70)
    spiced_results = run_method_on_datasets(target_files, 'SPICED')
    with open(f"{PROJECT_ROOT}/results/synthetic/spiced_knn_results.json", 'w') as f:
        json.dump(spiced_results, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("2. Running NOTEARS...")
    print("=" * 70)
    notears_results = run_method_on_datasets(target_files, 'NOTEARS')
    with open(f"{PROJECT_ROOT}/results/synthetic/notears_results.json", 'w') as f:
        json.dump(notears_results, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("3. Running PC...")
    print("=" * 70)
    pc_results = run_method_on_datasets(target_files, 'PC')
    with open(f"{PROJECT_ROOT}/results/synthetic/pc_results.json", 'w') as f:
        json.dump(pc_results, f, indent=2, default=float)
    
    # Ablations (on subset)
    ablation_files = target_files[:50]
    
    print("\n" + "=" * 70)
    print("4. Ablation: SPICED without structural constraints...")
    print("=" * 70)
    spiced_no_constr = run_method_on_datasets(ablation_files, 'SPICED_no_constraints')
    with open(f"{PROJECT_ROOT}/results/ablations/spiced_no_constraints.json", 'w') as f:
        json.dump(spiced_no_constr, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("5. Ablation: SPICED without IT initialization...")
    print("=" * 70)
    spiced_no_init = run_method_on_datasets(ablation_files, 'SPICED_no_init')
    with open(f"{PROJECT_ROOT}/results/ablations/spiced_no_it_init.json", 'w') as f:
        json.dump(spiced_no_init, f, indent=2, default=float)
    
    print("\n" + "=" * 70)
    print("Experiments complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
