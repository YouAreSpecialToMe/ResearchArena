"""
Run all experiments efficiently.
Processes datasets in batches and saves intermediate results.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'exp'))

import numpy as np
import json
import time
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import methods
from shared.metrics import compute_all_metrics
from causallearn.search.ConstraintBased.PC import pc

# Import NOTEARS
def notears_linear(X, lambda1=0.1, max_iter=100, w_threshold=0.3, seed=None):
    """Simplified NOTEARS."""
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
    
    return (np.abs(W) > w_threshold).astype(int)


def golem_ev(X, lambda1=0.02, lambda2=5.0, num_iter=1000, w_threshold=0.3, seed=None):
    """GOLEM-EV."""
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
        grad_ls = (1.0 / n) * (X.T @ residual)
        grad_l1 = lambda1 * np.sign(W)
        
        M = W * W
        try:
            eigvals = np.linalg.eigvalsh(M)
            h = np.sum(np.exp(eigvals)) - d
            eigvecs = np.linalg.eigh(M)[1]
            exp_M = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
            grad_dag = lambda2 * 2 * W * exp_M.T * h
        except:
            grad_dag = np.zeros_like(W)
        
        grad = grad_ls + grad_l1 + grad_dag
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= 0.001 * m_hat / (np.sqrt(v_hat) + eps)
        W = np.clip(W, -10, 10)
    
    return (np.abs(W) > w_threshold).astype(int)


def run_pc(data, true_adj, metadata):
    """Run PC algorithm."""
    start = time.time()
    try:
        cg = pc(data, alpha=0.05, indep_test='fisherz', stable=True, show_progress=False)
        pred_adj = cg.G.graph
        binary_adj = np.zeros_like(pred_adj)
        binary_adj[pred_adj == 1] = 1
        binary_adj[pred_adj == -1] = 1
        runtime = time.time() - start
        metrics = compute_all_metrics(true_adj, binary_adj)
        return {'method': 'pc', 'runtime': runtime, **metadata, **metrics}
    except Exception as e:
        return {'method': 'pc', 'runtime': None, 'error': str(e), **metadata}


def run_notears(data, true_adj, metadata):
    """Run NOTEARS."""
    start = time.time()
    try:
        n_nodes = true_adj.shape[0]
        max_iter = 100 if n_nodes <= 20 else 50
        pred_adj = notears_linear(data, lambda1=0.1, max_iter=max_iter, w_threshold=0.3, seed=metadata['seed'])
        runtime = time.time() - start
        metrics = compute_all_metrics(true_adj, pred_adj)
        return {'method': 'notears', 'runtime': runtime, **metadata, **metrics}
    except Exception as e:
        return {'method': 'notears', 'runtime': None, 'error': str(e), **metadata}


def run_golem(data, true_adj, metadata):
    """Run GOLEM."""
    start = time.time()
    try:
        n_nodes = true_adj.shape[0]
        num_iter = 1000 if n_nodes <= 20 else 500
        pred_adj = golem_ev(data, lambda1=0.02, lambda2=5.0, num_iter=num_iter, w_threshold=0.3, seed=metadata['seed'])
        runtime = time.time() - start
        metrics = compute_all_metrics(true_adj, pred_adj)
        return {'method': 'golem', 'runtime': runtime, **metadata, **metrics}
    except Exception as e:
        return {'method': 'golem', 'runtime': None, 'error': str(e), **metadata}


def run_method_on_dataset(args):
    """Wrapper for parallel execution."""
    method_name, dataset_file = args
    
    data_dict = np.load(dataset_file)
    data = data_dict['data']
    true_adj = data_dict['adj']
    
    metadata = {
        'graph_id': int(data_dict['graph_id']),
        'mechanism': str(data_dict['mechanism']),
        'n_samples': int(data_dict['n_samples']),
        'seed': int(data_dict['seed']),
        'n_nodes': true_adj.shape[0]
    }
    
    if method_name == 'pc':
        return run_pc(data, true_adj, metadata)
    elif method_name == 'notears':
        return run_notears(data, true_adj, metadata)
    elif method_name == 'golem':
        return run_golem(data, true_adj, metadata)
    
    return None


def run_experiments_parallel(methods=['pc', 'notears', 'golem'], max_workers=2):
    """Run experiments in parallel."""
    dataset_files = glob.glob("data/processed/datasets/*.npz")
    print(f"Found {len(dataset_files)} datasets")
    
    # Create tasks
    tasks = []
    for method in methods:
        for f in dataset_files:
            tasks.append((method, f))
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Running with {max_workers} workers...")
    
    results = {m: [] for m in methods}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_method_on_dataset, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            method_name = task[0]
            
            try:
                result = future.result(timeout=60)
                if result:
                    results[method_name].append(result)
            except Exception as e:
                print(f"Error in {method_name}: {e}")
            
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(tasks)} tasks completed")
    
    # Save results
    os.makedirs("results/synthetic", exist_ok=True)
    for method, method_results in results.items():
        with open(f"results/synthetic/{method}_summary.json", 'w') as f:
            json.dump(method_results, f, indent=2, default=float)
        print(f"Saved {method}: {len(method_results)} results")
    
    return results


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("="*60)
    print("Running All Baseline Experiments")
    print("="*60)
    results = run_experiments_parallel(methods=['pc', 'notears', 'golem'], max_workers=2)
    print("\nAll experiments complete!")
