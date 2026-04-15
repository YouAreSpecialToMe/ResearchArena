"""
Ablation 1: Fixed vs Adaptive Budget Allocation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
from typing import Dict, List
from shared.metrics import compute_metrics
from shared.utils import load_dataset, save_results, Timer
from mf_acd.mf_acd import MFACD


def run_ablation(data: np.ndarray, true_adj: np.ndarray, 
                 use_adaptive: bool, budget_allocation: tuple) -> Dict:
    """Run MF-ACD with specific configuration."""
    
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
    
    metrics = compute_metrics(true_adj, result['adjacency'])
    
    return {
        'metrics': metrics,
        'runtime': float(timer.elapsed),
        'phase_costs': result['phase_costs'],
        'savings_pct': result['savings_pct'],
        'config': {
            'use_adaptive': use_adaptive,
            'budget_allocation': budget_allocation
        }
    }


def run_experiment(dataset_path: str) -> Dict:
    """Run both variants on a single dataset."""
    dataset = load_dataset(dataset_path)
    data = dataset['data']
    true_adj = dataset['adjacency']
    
    # Fixed allocation (pilot-informed split, no adaptation)
    fixed_result = run_ablation(
        data, true_adj, 
        use_adaptive=False,
        budget_allocation=(0.34, 0.20, 0.46)
    )
    
    # Adaptive allocation
    adaptive_result = run_ablation(
        data, true_adj,
        use_adaptive=True,
        budget_allocation=(0.34, 0.20, 0.46)
    )
    
    return {
        'dataset': dataset_path,
        'fixed': fixed_result,
        'adaptive': adaptive_result
    }


def run_all_experiments(data_dir: str = "data/synthetic",
                        output_dir: str = "results/ablations/fixed_vs_adaptive"):
    """Run ablation on subset: 50 nodes × 2 densities × 10 seeds = 40 runs each."""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    
    # Filter: 50 nodes only, edge_prob 0.1 or 0.2
    datasets = [d for d in manifest['datasets'] 
                if d['config']['n_nodes'] == 50 
                and d['config']['edge_param'] in [0.1, 0.2, 1, 2]]
    
    print(f"Running Fixed vs Adaptive ablation on {len(datasets)} datasets...")
    
    results = []
    for i, dataset_info in enumerate(datasets):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(datasets)}")
        
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
    
    # Statistical comparison
    fixed_f1 = [r['fixed']['metrics']['f1'] for r in results if 'fixed' in r]
    adaptive_f1 = [r['adaptive']['metrics']['f1'] for r in results if 'adaptive' in r]
    
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(adaptive_f1, fixed_f1)
    
    print("\nAblation 1 Summary:")
    print(f"  Fixed F1: {np.mean(fixed_f1):.3f} ± {np.std(fixed_f1):.3f}")
    print(f"  Adaptive F1: {np.mean(adaptive_f1):.3f} ± {np.std(adaptive_f1):.3f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    
    return results


if __name__ == "__main__":
    run_all_experiments()
