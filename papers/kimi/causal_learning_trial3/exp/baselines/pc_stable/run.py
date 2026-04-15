"""
Baseline: PC-Stable Algorithm (Colombo & Maathuis, 2014).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from typing import Dict
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from shared.metrics import compute_metrics
from shared.utils import load_dataset, save_results, Timer


def run_pc_stable(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """Run PC-Stable algorithm."""
    n_nodes = data.shape[1]
    
    with Timer() as timer:
        cg = pc(data, alpha=alpha, indep_test=fisherz, stable=True,
                uc_rule=0, uc_priority=2, verbose=False)
    
    # Extract adjacency matrix
    pred_adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if cg.G.graph[i, j] == 1:
                pred_adj[i, j] = 1
            elif cg.G.graph[i, j] == -1:  # Undirected edge
                pred_adj[i, j] = 1
                pred_adj[j, i] = 1
    
    return {
        'pred_adj': pred_adj,
        'runtime': float(timer.elapsed)
    }


def run_experiment(dataset_path: str, alpha: float = 0.05) -> Dict:
    """Run experiment on a single dataset."""
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    result = run_pc_stable(data, alpha=alpha)
    metrics = compute_metrics(true_adj, result['pred_adj'])
    
    return {
        'dataset': dataset_path,
        'metrics': metrics,
        'runtime': result['runtime'],
        'config': {'alpha': alpha, 'stable': True}
    }


def run_all_experiments(data_dir: str = "data/synthetic",
                        output_dir: str = "results/baselines/pc_stable"):
    """Run experiments on all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    
    results = []
    print(f"Running PC-Stable on {len(manifest['datasets'])} datasets...")
    
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
