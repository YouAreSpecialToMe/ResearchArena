#!/usr/bin/env python3
"""
Efficient experiment runner - focuses on smaller graphs for timely completion.
"""
import os
import sys
import json
import pickle
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comprehensive_experiments import (
    PCBaseline, FastPCBaseline, MFACD, get_all_datasets
)
from shared.metrics import compute_metrics


def run_method(method_name, method_class, kwargs, datasets, output_path):
    """Run a method on datasets."""
    results = []
    print(f"\nRunning {method_name} on {len(datasets)} datasets...")
    
    for i, dataset_info in enumerate(datasets):
        if i % 10 == 0:
            print(f"  {i}/{len(datasets)}")
        
        try:
            with open(dataset_info['path'], 'rb') as f:
                dataset = pickle.load(f)
            
            method = method_class(**kwargs)
            start = time.time()
            result = method.fit(dataset['data'])
            runtime = time.time() - start
            
            metrics = compute_metrics(result['adjacency'], dataset['adjacency'])
            
            result_entry = {
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'metrics': metrics,
                'runtime': runtime
            }
            
            for key in ['n_tests', 'phase_costs', 'total_cost', 'baseline_cost', 'savings_pct']:
                if key in result:
                    result_entry[key] = result[key]
            
            results.append(result_entry)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'error': str(e)
            })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    valid = [r for r in results if 'metrics' in r]
    if valid:
        print(f"  {method_name}: F1={np.mean([r['metrics']['f1'] for r in valid]):.3f}, "
              f"Time={np.mean([r['runtime'] for r in valid]):.2f}s")
    
    return results


def main():
    start_time = time.time()
    
    print("="*70)
    print("EFFICIENT EXPERIMENT RUNNER")
    print("="*70)
    
    all_datasets = get_all_datasets("data/synthetic")
    
    # Focus on 20 and 50 node graphs (100-node PC is too slow)
    datasets_20 = [d for d in all_datasets if d['config'].get('n_nodes') == 20]
    datasets_50 = [d for d in all_datasets if d['config'].get('n_nodes') == 50]
    all_data = datasets_20 + datasets_50
    
    print(f"\nRunning on {len(datasets_20)} 20-node and {len(datasets_50)} 50-node datasets")
    
    # Essential baselines
    run_method('pc_fisherz', PCBaseline, {'alpha': 0.05, 'stable': False}, 
               all_data, 'results/baselines/pc_fisherz/results.json')
    
    run_method('fast_pc', FastPCBaseline, {'alpha': 0.05},
               all_data, 'results/baselines/fast_pc/results.json')
    
    # MF-ACD main
    run_method('mf_acd', MFACD, 
               {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': True, 'use_ugfs': True},
               all_data, 'results/mf_acd/main/results.json')
    
    # Ablations on 50-node
    run_method('mf_acd_fixed', MFACD,
               {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': False, 'use_ugfs': True},
               datasets_50, 'results/ablations/fixed_vs_adaptive/fixed.json')
    
    run_method('mf_acd_adaptive', MFACD,
               {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': True, 'use_ugfs': True},
               datasets_50, 'results/ablations/fixed_vs_adaptive/adaptive.json')
    
    run_method('mf_acd_nougfs', MFACD,
               {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': True, 'use_ugfs': False},
               datasets_50, 'results/ablations/ugfs_components/nougfs.json')
    
    print(f"\n{'='*70}")
    print(f"Completed in {(time.time() - start_time)/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
