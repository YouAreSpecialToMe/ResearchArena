#!/usr/bin/env python3
"""
CleanBP full system experiment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
import time
from shared.data_loader import load_hospital_dataset, load_adult_dataset, evaluate_repairs
from shared.cleanbp import CleanBP


def run_experiment(dataset_path, clean_path, error_cells_path, fds_path, output_path, seed):
    """Run CleanBP experiment."""
    
    # Load data
    if 'hospital' in dataset_path:
        dirty_df = load_hospital_dataset(dataset_path)
        clean_df = load_hospital_dataset(clean_path)
    else:
        dirty_df = load_adult_dataset(dataset_path)
        clean_df = load_adult_dataset(clean_path)
    
    with open(error_cells_path) as f:
        error_data = json.load(f)
        error_cells = set((int(x), y) for x, y in error_data['error_cells'])
    
    with open(fds_path) as f:
        fds_data = json.load(f)
        fds = [(fd['lhs'], fd['rhs']) for fd in fds_data]
    
    print(f"Dataset: {len(dirty_df)} tuples, {len(error_cells)} errors, seed={seed}")
    
    # Run CleanBP
    print("Running CleanBP...")
    cleanbp = CleanBP(
        max_iterations=50,
        convergence_threshold=1e-6,
        damping=0.5,
        verbose=False
    )
    
    start_time = time.time()
    repaired_df, info = cleanbp.repair(
        dirty_df, fds,
        violation_only=True,
        separate_attributes=True,
        priority_scheduling=True
    )
    total_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_repairs(dirty_df, repaired_df, clean_df, error_cells)
    
    print(f"Results: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    print(f"Time: {total_time:.2f}s, BP Time: {info['elapsed_time']:.2f}s, Iterations: {info['iterations']}, Converged: {info['converged']}")
    print(f"Graph: {info['graph_stats']['n_cells']} cells, {info['graph_stats']['n_violations']} violations")
    
    # Save results
    result = {
        'method': 'cleanbp_full',
        'dataset': dataset_path,
        'seed': seed,
        'metrics': metrics,
        'runtime_seconds': total_time,
        'bp_time': info['elapsed_time'],
        'iterations': info['iterations'],
        'converged': info['converged'],
        'graph_stats': info.get('graph_stats', {}),
        'convergence_history': info.get('convergence_history', [])
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='hospital', choices=['hospital', 'adult'])
    parser.add_argument('--error-rate', type=int, default=10)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running CleanBP on {args.dataset} dataset, seed={args.seed}")
    print(f"{'='*60}")
    
    if args.dataset == 'hospital':
        run_experiment(
            dataset_path=f'data/hospital/hospital_dirty_{args.error_rate}pct.csv',
            clean_path='data/hospital/hospital_clean.csv',
            error_cells_path=f'data/hospital/hospital_errors_{args.error_rate}pct.json',
            fds_path='data/hospital/fds.json',
            output_path=f'results/cleanbp_full_hospital_{args.error_rate}pct_seed{args.seed}.json',
            seed=args.seed
        )
    else:
        run_experiment(
            dataset_path='data/adult/adult_dirty.csv',
            clean_path='data/adult/adult_clean.csv',
            error_cells_path='data/adult/adult_errors.json',
            fds_path='data/adult/fds.json',
            output_path=f'results/cleanbp_full_adult_seed{args.seed}.json',
            seed=args.seed
        )
