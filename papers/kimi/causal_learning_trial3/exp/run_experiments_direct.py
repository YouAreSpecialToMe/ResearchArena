#!/usr/bin/env python3
"""
Direct experiment runner for MF-ACD evaluation.
Runs all experiments without complex orchestration.
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
    PCBaseline, GESBaseline, FastPCBaseline, HCCDBaseline, DCILPBaseline,
    MFACD, get_all_datasets
)
from shared.metrics import compute_metrics


def run_method_on_datasets(method_name, method_class, method_kwargs, datasets, output_path):
    """Run a method on a list of datasets and save results."""
    results = []
    
    print(f"\n{'='*60}")
    print(f"Running {method_name} on {len(datasets)} datasets")
    print(f"{'='*60}")
    
    for i, dataset_info in enumerate(datasets):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(datasets)}")
        
        try:
            with open(dataset_info['path'], 'rb') as f:
                dataset = pickle.load(f)
            
            method = method_class(**method_kwargs)
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
            
            # Add optional fields if present
            for key in ['n_tests', 'phase_costs', 'total_cost', 'baseline_cost', 'savings_pct']:
                if key in result:
                    result_entry[key] = result[key]
            
            results.append(result_entry)
            
        except Exception as e:
            print(f"  Error on {dataset_info['name']}: {e}")
            results.append({
                'dataset': dataset_info['name'],
                'config': dataset_info['config'],
                'error': str(e)
            })
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{method_name} Summary:")
    valid_results = [r for r in results if 'metrics' in r]
    if valid_results:
        avg_f1 = np.mean([r['metrics']['f1'] for r in valid_results])
        avg_shd = np.mean([r['metrics']['shd'] for r in valid_results])
        avg_time = np.mean([r['runtime'] for r in valid_results])
        print(f"  F1: {avg_f1:.3f}, SHD: {avg_shd:.1f}, Time: {avg_time:.2f}s")
        
        if 'savings_pct' in valid_results[0]:
            avg_savings = np.mean([r['savings_pct'] for r in valid_results])
            print(f"  Savings: {avg_savings:.1f}%")
    
    return results


def main():
    """Main experiment execution."""
    start_time = time.time()
    
    print("="*70)
    print("DIRECT EXPERIMENT RUNNER")
    print("="*70)
    
    # Get all datasets
    all_datasets = get_all_datasets("data/synthetic")
    print(f"\nFound {len(all_datasets)} datasets")
    
    # Filter by node count
    datasets_20 = [d for d in all_datasets if d['config'].get('n_nodes') == 20]
    datasets_50 = [d for d in all_datasets if d['config'].get('n_nodes') == 50]
    datasets_100 = [d for d in all_datasets if d['config'].get('n_nodes') == 100]
    
    print(f"  20 nodes: {len(datasets_20)}")
    print(f"  50 nodes: {len(datasets_50)}")
    print(f"  100 nodes: {len(datasets_100)}")
    
    # Run all baselines on all datasets
    all_data = datasets_20 + datasets_50 + datasets_100
    
    experiments = [
        ('pc_fisherz', PCBaseline, {'alpha': 0.05, 'stable': False}, 'results/baselines/pc_fisherz/results.json'),
        ('pc_stable', PCBaseline, {'alpha': 0.05, 'stable': True}, 'results/baselines/pc_stable/results.json'),
        ('fast_pc', FastPCBaseline, {'alpha': 0.05}, 'results/baselines/fast_pc/results.json'),
        ('ges', GESBaseline, {'score_type': 'bic'}, 'results/baselines/ges/results.json'),
    ]
    
    for method_name, method_class, kwargs, output_path in experiments:
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            run_method_on_datasets(method_name, method_class, kwargs, all_data, output_path)
        else:
            print(f"\nSkipping {method_name} - results already exist")
    
    # Run HCCD and DCILP on smaller subset
    smaller_data = datasets_20 + datasets_50[:60]
    
    for method_name, method_class, kwargs, output_path in [
        ('hccd', HCCDBaseline, {'alpha': 0.05}, 'results/baselines/hccd/results.json'),
        ('dcilp', DCILPBaseline, {'alpha': 0.05}, 'results/baselines/dcilp/results.json'),
    ]:
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            run_method_on_datasets(method_name, method_class, kwargs, smaller_data, output_path)
        else:
            print(f"\nSkipping {method_name} - results already exist")
    
    # Run MF-ACD on all datasets
    output_path = 'results/mf_acd/main/results.json'
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
        run_method_on_datasets(
            'mf_acd', MFACD,
            {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': True, 'use_ugfs': True},
            all_data, output_path
        )
    else:
        print(f"\nSkipping mf_acd - results already exist")
    
    # Run ablations on 50-node datasets
    print("\n" + "="*70)
    print("ABLATION STUDIES")
    print("="*70)
    
    ablations = [
        ('mf_acd_fixed', MFACD, {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': False, 'use_ugfs': True},
         'results/ablations/fixed_vs_adaptive/fixed.json'),
        ('mf_acd_adaptive', MFACD, {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': True, 'use_ugfs': True},
         'results/ablations/fixed_vs_adaptive/adaptive.json'),
        ('mf_acd_nougfs', MFACD, {'budget_allocation': (0.34, 0.20, 0.46), 'use_adaptive': True, 'use_ugfs': False},
         'results/ablations/ugfs_components/nougfs.json'),
        ('mf_acd_conservative', MFACD, {'budget_allocation': (0.40, 0.30, 0.30), 'use_adaptive': True, 'use_ugfs': True},
         'results/ablations/allocation_sensitivity/conservative.json'),
        ('mf_acd_aggressive', MFACD, {'budget_allocation': (0.25, 0.15, 0.60), 'use_adaptive': True, 'use_ugfs': True},
         'results/ablations/allocation_sensitivity/aggressive.json'),
    ]
    
    for method_name, method_class, kwargs, output_path in ablations:
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 100:
            run_method_on_datasets(method_name, method_class, kwargs, datasets_50, output_path)
        else:
            print(f"\nSkipping {method_name} - results already exist")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
