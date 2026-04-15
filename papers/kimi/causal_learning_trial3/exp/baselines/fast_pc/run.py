"""
Baseline: Fast PC with correlation-based tests only.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from typing import Dict
from scipy.stats import pearsonr
from shared.metrics import compute_metrics
from shared.utils import load_dataset, save_results, Timer


def correlation_test(data: np.ndarray, x: int, y: int, cond_set: list) -> float:
    """
    Correlation-based conditional independence test.
    Uses partial correlation for conditional tests.
    """
    n = data.shape[0]
    
    if len(cond_set) == 0:
        # Unconditional correlation test
        corr, _ = pearsonr(data[:, x], data[:, y])
        # Convert to z-score for p-value
        z = np.arctanh(np.abs(corr)) * np.sqrt(n - 3)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(np.abs(z)))
        return p_value
    else:
        # Partial correlation
        # Regress x and y on cond_set, then correlate residuals
        from sklearn.linear_model import LinearRegression
        
        X_cond = data[:, cond_set]
        
        # Regress x on cond_set
        reg_x = LinearRegression().fit(X_cond, data[:, x])
        resid_x = data[:, x] - reg_x.predict(X_cond)
        
        # Regress y on cond_set
        reg_y = LinearRegression().fit(X_cond, data[:, y])
        resid_y = data[:, y] - reg_y.predict(X_cond)
        
        # Correlation of residuals
        corr, _ = pearsonr(resid_x, resid_y)
        z = np.arctanh(np.abs(corr)) * np.sqrt(n - 3 - len(cond_set))
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(np.abs(z)))
        return p_value


def run_fast_pc(data: np.ndarray, alpha: float = 0.05, max_cond_size: int = None) -> Dict:
    """
    Simple PC implementation with correlation-based tests.
    Uses a simplified skeleton discovery phase.
    """
    n_nodes = data.shape[0]
    n_vars = data.shape[1]
    
    if max_cond_size is None:
        max_cond_size = n_vars - 2
    
    with Timer() as timer:
        # Initialize complete graph
        adj = np.ones((n_vars, n_vars))
        np.fill_diagonal(adj, 0)
        
        n_tests = 0
        
        # Phase 1: Remove edges based on correlation tests
        for d in range(max_cond_size + 1):
            edges = [(i, j) for i in range(n_vars) for j in range(i+1, n_vars) if adj[i, j] == 1]
            
            for i, j in edges:
                # Find neighbors of i excluding j
                neighbors = [k for k in range(n_vars) if adj[i, k] == 1 and k != j]
                
                if len(neighbors) >= d:
                    from itertools import combinations
                    for cond_set in combinations(neighbors, d):
                        n_tests += 1
                        p_value = correlation_test(data, i, j, list(cond_set))
                        
                        if p_value > alpha:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            break
    
    return {
        'pred_adj': adj,
        'runtime': float(timer.elapsed),
        'n_tests': n_tests
    }


def run_experiment(dataset_path: str, alpha: float = 0.05) -> Dict:
    """Run experiment on a single dataset."""
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    result = run_fast_pc(data, alpha=alpha)
    metrics = compute_metrics(true_adj, result['pred_adj'])
    
    return {
        'dataset': dataset_path,
        'metrics': metrics,
        'runtime': result['runtime'],
        'n_tests': result['n_tests'],
        'config': {'alpha': alpha}
    }


def run_all_experiments(data_dir: str = "data/synthetic",
                        output_dir: str = "results/baselines/fast_pc"):
    """Run experiments on all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    
    results = []
    print(f"Running Fast PC on {len(manifest['datasets'])} datasets...")
    
    for i, dataset_info in enumerate(manifest['datasets']):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(manifest['datasets'])}")
        
        dataset_path = dataset_info['path']
        if not dataset_path.startswith('data/synthetic/'):
            dataset_path = os.path.join(data_dir, os.path.basename(dataset_info['path']))
        
        try:
            result = run_experiment(dataset_path)
            result['dataset_name'] = dataset_info['name']
            result['dataset_config'] = dataset_info['config']
            results.append(result)
        except Exception as e:
            print(f"Error on {dataset_info['name']}: {e}")
            results.append({'dataset_name': dataset_info['name'], 'error': str(e)})
    
    save_results(results, os.path.join(output_dir, "results.json"))
    
    # Print summary
    print("\nResults Summary:")
    for n_nodes in [20, 50, 100]:
        subset = [r for r in results if 'metrics' in r and 
                  r.get('dataset_config', {}).get('n_nodes') == n_nodes]
        if subset:
            avg_f1 = np.mean([r['metrics']['f1'] for r in subset])
            avg_shd = np.mean([r['metrics']['shd'] for r in subset])
            print(f"  {n_nodes} nodes: F1={avg_f1:.3f}, SHD={avg_shd:.1f}")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
