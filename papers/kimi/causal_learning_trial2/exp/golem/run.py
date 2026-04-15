"""
GOLEM baseline for causal discovery.
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


def golem_ev(X, lambda1=0.02, lambda2=5.0, num_iter=1000, 
             learning_rate=0.001, w_threshold=0.3, seed=None):
    """GOLEM-EV implementation."""
    if seed is not None:
        np.random.seed(seed)
    
    n, d = X.shape
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    W = np.random.randn(d, d) * 0.01
    
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    for t in range(1, num_iter + 1):
        residual = X @ W - X
        loss_ls = (0.5 / n) * np.sum(residual ** 2)
        grad_ls = (1.0 / n) * (X.T @ residual)
        
        loss_l1 = lambda1 * np.sum(np.abs(W))
        grad_l1 = lambda1 * np.sign(W)
        
        M = W * W
        try:
            eigvals = np.linalg.eigvalsh(M)
            h = np.sum(np.exp(eigvals)) - d
            eigvecs = np.linalg.eigh(M)[1]
            exp_M = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
            grad_dag = lambda2 * 2 * W * exp_M.T * h
        except:
            h = 1.0
            grad_dag = np.zeros_like(W)
        
        grad = grad_ls + grad_l1 + grad_dag
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        W = np.clip(W, -10, 10)
    
    W_binary = (np.abs(W) > w_threshold).astype(int)
    return W_binary


def run_golem_baseline():
    """Run GOLEM on all synthetic datasets."""
    results = []
    
    dataset_files = glob.glob(os.path.join(PROJECT_ROOT, "data/processed/datasets/*.npz"))
    
    print(f"Running GOLEM on {len(dataset_files)} datasets...")
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
        
        num_iter = 1000 if n_nodes <= 20 else 500
        
        start_time = time.time()
        try:
            pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0, 
                               num_iter=num_iter, learning_rate=0.001,
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
    
    os.makedirs(os.path.join(PROJECT_ROOT, "results/synthetic/golem"), exist_ok=True)
    np.savez(os.path.join(PROJECT_ROOT, "results/synthetic/golem_results.npz"), results=results)
    
    with open(os.path.join(PROJECT_ROOT, "results/synthetic/golem_summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"GOLEM complete. Processed {len(results)} datasets.")
    return results


def run_golem_sachs():
    """Run GOLEM on Sachs dataset."""
    sachs_file = os.path.join(PROJECT_ROOT, "data/processed/real_world/sachs.npz")
    sachs_data = np.load(sachs_file)
    data = sachs_data['data']
    true_adj = sachs_data['true_dag']
    
    results = []
    for seed in range(1, 6):
        start_time = time.time()
        
        try:
            pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0,
                               num_iter=1000, learning_rate=0.001,
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
    with open(os.path.join(PROJECT_ROOT, "results/real_world/golem_sachs.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    return results


if __name__ == "__main__":
    print("Running GOLEM baseline...")
    run_golem_baseline()
    print("\nRunning GOLEM on Sachs...")
    run_golem_sachs()
    print("\nGOLEM experiments complete!")
