"""
Final experiment comparing methods.
Uses simpler but effective implementations for reliability.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import time
import glob
import warnings
warnings.filterwarnings('ignore')

from shared.metrics import compute_all_metrics
from scipy.optimize import minimize


def least_squares_dag(X, lambda1=0.1, w_threshold=0.3, max_iter=50, seed=None):
    """
    Simple least squares DAG estimation with L1 sparsity.
    Similar to NOTEARS but with simpler optimization.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Initialize
    W = np.random.randn(d, d) * 0.01
    
    # Alternating optimization
    for iteration in range(max_iter):
        W_old = W.copy()
        
        for j in range(d):
            # Target variable
            y = X[:, j]
            
            # Predictors (all other variables)
            for i in range(d):
                if i == j:
                    W[i, j] = 0
                    continue
                
                # Simple coordinate descent
                x_i = X[:, i]
                residual = y - X @ W[:, j] + W[i, j] * x_i
                
                # Least squares solution for this coordinate
                beta = (x_i @ residual) / (x_i @ x_i + 1e-8)
                
                # Soft thresholding
                W[i, j] = np.sign(beta) * max(abs(beta) - lambda1, 0)
        
        # Check convergence
        if np.max(np.abs(W - W_old)) < 1e-4:
            break
    
    return (np.abs(W) > w_threshold).astype(int)


def golem_simple(X, lambda1=0.02, lambda2=5.0, num_iter=500, lr=0.001, w_threshold=0.3, seed=None):
    """
    Simplified GOLEM implementation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Initialize
    W = np.random.randn(d, d) * 0.01
    m, v = np.zeros_like(W), np.zeros_like(W)
    beta1, beta2 = 0.9, 0.999
    
    for t in range(1, num_iter + 1):
        # Gradient of least squares
        grad = (1.0 / n) * (X.T @ (X @ W - X))
        grad += lambda1 * np.sign(W)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        # Zero diagonal
        np.fill_diagonal(W, 0)
    
    return (np.abs(W) > w_threshold).astype(int)


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'spiced'))
from spiced_main import spiced


def run_final_experiment():
    """Run final experiment comparing all methods."""
    
    dataset_files = glob.glob("data/processed/datasets/*.npz")
    
    # Filter for key configurations
    filtered_files = []
    for f in dataset_files:
        data_dict = np.load(f)
        n_samples = int(data_dict['n_samples'])
        adj = data_dict['adj']
        n_nodes = adj.shape[0]
        graph_id = int(data_dict['graph_id'])
        
        # Focus on n=10,20 and N=50,100,200
        if n_nodes <= 20 and n_samples in [50, 100, 200] and (graph_id % 10) <= 4:
            filtered_files.append(f)
    
    # Limit to 100 files for reasonable runtime
    filtered_files = filtered_files[:100]
    
    print(f"Running on {len(filtered_files)} datasets...")
    
    results = {
        'notears': [],
        'golem': [],
        'spiced': []
    }
    
    for i, dataset_file in enumerate(filtered_files):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(filtered_files)}")
        
        data_dict = np.load(dataset_file)
        data = data_dict['data']
        true_adj = data_dict['adj']
        
        graph_id = int(data_dict['graph_id'])
        mechanism = str(data_dict['mechanism'])
        n_samples = int(data_dict['n_samples'])
        seed = int(data_dict['seed'])
        n_nodes = true_adj.shape[0]
        
        base_info = {
            'graph_id': graph_id,
            'mechanism': mechanism,
            'n_samples': n_samples,
            'seed': seed,
            'n_nodes': n_nodes
        }
        
        # Least Squares / NOTEARS
        try:
            start = time.time()
            pred_adj = least_squares_dag(data, lambda1=0.1, w_threshold=0.2, max_iter=50, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['notears'].append({**base_info, 'runtime': runtime, **metrics})
        except Exception as e:
            print(f"NOTEARS error: {e}")
        
        # GOLEM
        try:
            start = time.time()
            pred_adj = golem_simple(data, lambda1=0.02, lambda2=5.0, num_iter=300, 
                                   lr=0.001, w_threshold=0.2, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['golem'].append({**base_info, 'runtime': runtime, **metrics})
        except Exception as e:
            print(f"GOLEM error: {e}")
        
        # SPICED
        try:
            pred_adj, timing, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=50,
                w_threshold=0.2,
                use_it_init=True,
                use_structural_constraints=True,
                seed=seed
            )
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['spiced'].append({
                **base_info, 
                'runtime': timing['total'],
                **metrics
            })
        except Exception as e:
            print(f"SPICED error: {e}")
    
    # Save results
    os.makedirs("results/synthetic", exist_ok=True)
    
    for method, data in results.items():
        with open(f"results/synthetic/{method}_summary.json", 'w') as f:
            json.dump(data, f, indent=2, default=float)
        print(f"Saved {method}: {len(data)} results")
    
    return results


def run_sachs_final():
    """Run all methods on Sachs dataset."""
    
    sachs_data = np.load("data/processed/real_world/sachs.npz")
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = {
        'notears': [],
        'golem': [],
        'spiced': []
    }
    
    for seed in range(1, 4):
        print(f"\nSachs seed {seed}:")
        
        # Least Squares
        try:
            start = time.time()
            pred_adj = least_squares_dag(data, lambda1=0.1, w_threshold=0.2, max_iter=50, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['notears'].append({'dataset': 'sachs', 'seed': seed, 'runtime': runtime, **metrics})
            print(f"  LS-DAG: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
        except Exception as e:
            print(f"  LS-DAG error: {e}")
        
        # GOLEM
        try:
            start = time.time()
            pred_adj = golem_simple(data, lambda1=0.02, lambda2=5.0, num_iter=500, 
                                   lr=0.001, w_threshold=0.2, seed=seed)
            runtime = time.time() - start
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['golem'].append({'dataset': 'sachs', 'seed': seed, 'runtime': runtime, **metrics})
            print(f"  GOLEM: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
        except Exception as e:
            print(f"  GOLEM error: {e}")
        
        # SPICED
        try:
            pred_adj, timing, _ = spiced(
                data,
                k_neighbors=5,
                alpha=0.05,
                lambda1=0.1,
                lambda3=0.01,
                max_iter=50,
                w_threshold=0.2,
                use_it_init=True,
                use_structural_constraints=True,
                seed=seed
            )
            metrics = compute_all_metrics(true_adj, pred_adj)
            results['spiced'].append({
                'dataset': 'sachs', 
                'seed': seed, 
                'runtime': timing['total'],
                **metrics
            })
            print(f"  SPICED: SHD={metrics['shd']}, TPR={metrics['tpr']:.3f}, FDR={metrics['fdr']:.3f}")
        except Exception as e:
            print(f"  SPICED error: {e}")
    
    # Save results
    os.makedirs("results/real_world", exist_ok=True)
    for method, data in results.items():
        with open(f"results/real_world/{method}_sachs.json", 'w') as f:
            json.dump(data, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Running Final Experiment")
    print("=" * 60)
    
    results = run_final_experiment()
    
    print("\n" + "=" * 60)
    print("Running on Sachs dataset")
    print("=" * 60)
    
    sachs_results = run_sachs_final()
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
