"""
Baseline: PC Algorithm with Fisher Z-test.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
import pickle
from typing import Dict, List
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from shared.metrics import compute_metrics, dag_to_adjacency
from shared.utils import load_dataset, save_results, Timer
import time


def run_pc_fisherz(data: np.ndarray, alpha: float = 0.05, stable: bool = False) -> Dict:
    """
    Run PC algorithm with Fisher Z-test.
    
    Args:
        data: Data matrix (n_samples, n_nodes)
        alpha: Significance level
        stable: Whether to use PC-Stable
        
    Returns:
        Dictionary with results
    """
    n_nodes = data.shape[1]
    
    with Timer() as timer:
        # Run PC algorithm
        cg = pc(data, alpha=alpha, indep_test=fisherz, stable=stable, 
                uc_rule=0, uc_priority=2, verbose=False)
    
    # Extract adjacency matrix
    # cg.G.graph is the predicted graph (0: not edge, 1: directed edge, -1: undirected)
    pred_adj = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if cg.G.graph[i, j] == 1:
                pred_adj[i, j] = 1
            elif cg.G.graph[i, j] == -1:  # Undirected edge in CPDAG
                pred_adj[i, j] = 1
                pred_adj[j, i] = 1
    
    # Count CI tests (causal-learn doesn't expose this directly, estimate)
    # We'll track this via the test results
    n_tests = cg.test_cit if hasattr(cg, 'test_cit') else -1
    
    return {
        'pred_adj': pred_adj,
        'runtime': float(timer.elapsed),
        'n_tests': n_tests
    }


def run_experiment(dataset_path: str, alpha: float = 0.05) -> Dict:
    """Run experiment on a single dataset."""
    # Load dataset
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    # Run PC
    result = run_pc_fisherz(data, alpha=alpha, stable=False)
    
    # Compute metrics
    metrics = compute_metrics(true_adj, result['pred_adj'])
    
    return {
        'dataset': dataset_path,
        'metrics': metrics,
        'runtime': result['runtime'],
        'n_tests': result['n_tests'],
        'config': {
            'alpha': alpha,
            'stable': False
        }
    }


def run_all_experiments(data_dir: str = "data/synthetic", 
                        output_dir: str = "results/baselines/pc_fisherz"):
    """Run experiments on all datasets."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load manifest
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    
    results = []
    
    print(f"Running PC-FisherZ on {len(manifest['datasets'])} datasets...")
    
    for i, dataset_info in enumerate(manifest['datasets']):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(manifest['datasets'])}")
        
        # Fix path to be relative to workspace root
        dataset_path = dataset_info['path']
        if dataset_path.startswith('data/synthetic/'):
            dataset_path = dataset_path
        else:
            dataset_path = os.path.join(data_dir, os.path.basename(dataset_info['path']))
        
        try:
            result = run_experiment(dataset_path)
            result['dataset_name'] = dataset_info['name']
            result['dataset_config'] = dataset_info['config']
            results.append(result)
        except Exception as e:
            print(f"Error on {dataset_info['name']}: {e}")
            results.append({
                'dataset_name': dataset_info['name'],
                'error': str(e)
            })
    
    # Save results
    save_results(results, os.path.join(output_dir, "results.json"))
    
    # Print summary
    print("\nResults Summary:")
    n_nodes_list = [20, 50, 100]
    for n_nodes in n_nodes_list:
        subset = [r for r in results if 'metrics' in r and 
                  r.get('dataset_config', {}).get('n_nodes') == n_nodes]
        if subset:
            avg_f1 = np.mean([r['metrics']['f1'] for r in subset])
            avg_shd = np.mean([r['metrics']['shd'] for r in subset])
            avg_time = np.mean([r['runtime'] for r in subset])
            print(f"  {n_nodes} nodes: F1={avg_f1:.3f}, SHD={avg_shd:.1f}, Time={avg_time:.2f}s")
    
    return results


if __name__ == "__main__":
    results = run_all_experiments()
