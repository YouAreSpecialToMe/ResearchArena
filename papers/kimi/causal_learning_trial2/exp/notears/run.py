"""
NOTEARS baseline for causal discovery.
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
from shared.metrics import compute_all_metrics


def notears_linear(X, lambda1=0.1, max_iter=100, w_threshold=0.3, seed=None):
    """Simplified NOTEARS for linear models."""
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    W = np.random.randn(d, d) * 0.01
    
    rho = 1.0
    alpha = 0.0
    
    for iteration in range(max_iter):
        W_old = W.copy()
        
        for j in range(d):
            grad = (1.0 / n) * (X.T @ (X @ W[:, j] - X[:, j]))
            grad += lambda1 * np.sign(W[:, j])
            W[:, j] -= 0.01 * grad
            W[:, j] = np.sign(W[:, j]) * np.maximum(np.abs(W[:, j]) - 0.01 * lambda1, 0)
            W[j, j] = 0
        
        M = W * W
        try:
            eigvals = np.linalg.eigvalsh(M)
            h = np.sum(np.exp(eigvals)) - d
        except:
            h = 1.0
        
        if h > 1e-8:
            alpha += rho * h
            rho = min(rho * 2, 1e16)
        
        if h > 1e-6:
            W *= 0.95
        
        if np.max(np.abs(W - W_old)) < 1e-4 and h < 1e-6:
            break
    
    W_binary = (np.abs(W) > w_threshold).astype(int)
    return W_binary


def run_notears_baseline():
    """Run NOTEARS on all synthetic datasets."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    print(f"Running NOTEARS on {len(dataset_files)} datasets...")
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
            pred_adj = notears_linear(data, lambda1=0.1, max_iter=100, 
                                     w_threshold=0.3, seed=seed)
            runtime = time.time() - start_time
            
            metrics = compute_all_metrics(true_adj, pred_adj)
            
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
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/synthetic/notears"), exist_ok=True)
    np.savez(os.path.join(PROJECT_ROOT, "results/synthetic/notears_results.npz"), results=results)
    
    with open(os.path.join(PROJECT_ROOT, "results/synthetic/notears_summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"NOTEARS complete. Processed {len(results)} datasets.")
    return results


def run_notears_sachs():
    """Run NOTEARS on Sachs dataset."""
    sachs_file = os.path.join(PROJECT_ROOT, "data/processed/real_world/sachs.npz")
    sachs_data = np.load(sachs_file)
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = []
    for seed in range(1, 6):
        start_time = time.time()
        
        try:
            pred_adj = notears_linear(data, lambda1=0.1, max_iter=100,
                                     w_threshold=0.3, seed=seed)
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
    with open(os.path.join(PROJECT_ROOT, "results/real_world/notears_sachs.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("Running NOTEARS baseline...")
    run_notears_baseline()
    print("\nRunning NOTEARS on Sachs...")
    run_notears_sachs()
    print("\nNOTEARS experiments complete!")
