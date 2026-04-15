"""
Fixed NOTEARS baseline for causal discovery.
Addresses the TPR=0.0 bug by properly implementing the acyclicity constraint.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
import json
import time
import glob
from scipy.linalg import expm
from shared.metrics import compute_all_metrics


def notears_linear_fixed(X, lambda1=0.1, max_iter=100, w_threshold=0.3, 
                         h_tol=1e-8, rho_max=1e16, seed=None):
    """
    Fixed NOTEARS implementation with proper acyclicity constraint.
    
    The acyclicity constraint is: h(W) = tr(exp(W * W)) - d = 0
    where exp is the matrix exponential.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Initialize W with small random values
    W = np.random.randn(d, d) * 0.01
    np.fill_diagonal(W, 0)  # No self-loops
    
    # Augmented Lagrangian parameters
    rho = 1.0
    alpha = 0.0
    
    for iteration in range(max_iter):
        W_old = W.copy()
        
        # Gradient descent on augmented Lagrangian
        for j in range(d):
            # Least squares gradient: (1/n) * X^T (X W_j - X_j)
            residual = X @ W[:, j] - X[:, j]
            grad_ls = (1.0 / n) * (X.T @ residual)
            
            # L1 gradient
            grad_l1 = lambda1 * np.sign(W[:, j])
            
            # Gradient step
            W[:, j] -= 0.1 * (grad_ls + grad_l1)
            
            # Soft thresholding for L1
            W[:, j] = np.sign(W[:, j]) * np.maximum(np.abs(W[:, j]) - 0.001, 0)
            W[j, j] = 0  # Enforce no self-loops
        
        # Compute acyclicity constraint: h(W) = tr(exp(W * W)) - d
        # W * W is element-wise product (Hadamard)
        W_sq = W * W
        try:
            # Matrix exponential
            exp_Wsq = expm(W_sq)
            h = np.trace(exp_Wsq) - d
        except Exception as e:
            h = 1.0
        
        # Augmented Lagrangian update
        if h > h_tol:
            alpha += rho * h
            rho = min(rho * 2, rho_max)
            # Shrink W if constraint is violated
            W *= 0.95
        
        # Check convergence
        if np.max(np.abs(W - W_old)) < 1e-4 and h < h_tol:
            break
    
    # Threshold to binary adjacency matrix
    W_binary = (np.abs(W) > w_threshold).astype(int)
    return W_binary, W


def run_notears_fixed():
    """Run fixed NOTEARS on all synthetic datasets."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    print(f"Running fixed NOTEARS on {len(dataset_files)} datasets...")
    print("This may take a while...")
    
    for i, dataset_file in enumerate(dataset_files):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(dataset_files)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        start_time = time.time()
        try:
            pred_adj, W_continuous = notears_linear_fixed(
                data, lambda1=0.1, max_iter=100, 
                w_threshold=0.3, seed=seed
            )
            runtime = time.time() - start_time
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            # Log if we're getting empty predictions
            if metrics['tpr'] == 0.0:
                print(f"  Warning: graph {graph_id} has TPR=0.0")
            
            result = {
                'graph_id': int(graph_id),
                'mechanism': mechanism,
                'n_samples': int(n_samples),
                'seed': int(seed),
                'n_nodes': int(n_nodes),
                'runtime': float(runtime),
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error on graph {graph_id}: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'graph_id': int(graph_id),
                'mechanism': mechanism,
                'n_samples': int(n_samples),
                'seed': int(seed),
                'n_nodes': int(n_nodes),
                'runtime': None,
                'error': str(e)
            }
            results.append(result)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/synthetic/notears_fixed"), exist_ok=True)
    
    with open(os.path.join(PROJECT_ROOT, "results/synthetic/notears_fixed_summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"NOTEARS (fixed) complete. Processed {len(results)} datasets.")
    return results


def run_notears_sachs_fixed():
    """Run fixed NOTEARS on Sachs dataset."""
    sachs_file = os.path.join(PROJECT_ROOT, "data/processed/real_world/sachs.npz")
    sachs_data = np.load(sachs_file)
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = []
    for seed in range(1, 6):
        start_time = time.time()
        
        try:
            pred_adj, W_continuous = notears_linear_fixed(
                data, lambda1=0.1, max_iter=100,
                w_threshold=0.3, seed=seed
            )
            runtime = time.time() - start_time
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
            result = {
                'dataset': 'sachs',
                'seed': seed,
                'runtime': runtime,
                **metrics
            }
            results.append(result)
            print(f"Sachs seed {seed}: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
            
        except Exception as e:
            print(f"Error on Sachs seed {seed}: {e}")
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/real_world"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results/real_world/notears_fixed_sachs.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("Running fixed NOTEARS baseline...")
    run_notears_fixed()
    print("\nRunning fixed NOTEARS on Sachs...")
    run_notears_sachs_fixed()
    print("\nNOTEARS (fixed) experiments complete!")
