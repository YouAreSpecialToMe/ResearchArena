"""
Run MF-ACD experiments on synthetic data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from typing import Dict
from shared.metrics import compute_metrics
from shared.utils import load_dataset, save_results, Timer
from mf_acd.mf_acd import MFACD


def run_experiment(dataset_path: str, 
                   budget_allocation: tuple = (0.34, 0.20, 0.46),
                   use_adaptive: bool = True) -> Dict:
    """Run MF-ACD on a single dataset."""
    # Load dataset
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    # Run MF-ACD
    with Timer() as timer:
        mf_acd = MFACD(
            budget_allocation=budget_allocation,
            use_adaptive=use_adaptive,
            alpha1=0.10,
            alpha2=0.05,
            alpha3=0.01,
            cost_weights=(1.0, 1.1, 15.0)
        )
        result = mf_acd.fit(data)
    
    # Compute metrics
    metrics = compute_metrics(true_adj, result['adjacency'])
    
    return {
        'dataset': dataset_path,
        'metrics': metrics,
        'runtime': float(timer.elapsed),
        'phase_costs': result['phase_costs'],
        'n_tests': result['n_tests'],
        'total_cost': result['total_cost'],
        'baseline_cost': result['baseline_cost'],
        'savings_pct': result['savings_pct'],
        'ugfs_overhead': result['ugfs_overhead'],
        'config': {
            'budget_allocation': budget_allocation,
            'use_adaptive': use_adaptive
        }
    }


def run_all_experiments(data_dir: str = "data/synthetic",
                        output_dir: str = "results/mf_acd/main",
                        budget_allocation: tuple = (0.34, 0.20, 0.46),
                        use_adaptive: bool = True):
    """Run MF-ACD on all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    
    results = []
    print(f"Running MF-ACD on {len(manifest['datasets'])} datasets...")
    
    for i, dataset_info in enumerate(manifest['datasets']):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(manifest['datasets'])}")
        
        dataset_path = dataset_info['path']
        if not dataset_path.startswith('data/synthetic/'):
            dataset_path = os.path.join(data_dir, os.path.basename(dataset_info['path']))
        
        try:
            result = run_experiment(dataset_path, budget_allocation, use_adaptive)
            result['dataset_name'] = dataset_info['name']
            result['dataset_config'] = dataset_info['config']
            results.append(result)
        except Exception as e:
            print(f"Error on {dataset_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'dataset_name': dataset_info['name'], 'error': str(e)})
    
    save_results(results, os.path.join(output_dir, "results.json"))
    
    # Print summary
    print("\nMF-ACD Results Summary:")
    for n_nodes in [20, 50, 100]:
        subset = [r for r in results if 'metrics' in r and 
                  r.get('dataset_config', {}).get('n_nodes') == n_nodes]
        if subset:
            avg_f1 = np.mean([r['metrics']['f1'] for r in subset])
            avg_shd = np.mean([r['metrics']['shd'] for r in subset])
            avg_savings = np.mean([r['savings_pct'] for r in subset])
            avg_time = np.mean([r['runtime'] for r in subset])
            print(f"  {n_nodes} nodes: F1={avg_f1:.3f}, SHD={avg_shd:.1f}, "
                  f"Savings={avg_savings:.1f}%, Time={avg_time:.2f}s")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
