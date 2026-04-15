#!/usr/bin/env python3
"""
Dense BP baseline experiment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
import time
from shared.data_loader import load_hospital_dataset, evaluate_repairs
from shared.baselines import dense_factor_graph_bp


def run_experiment(dataset_path, clean_path, error_cells_path, fds_path, output_path, seed):
    """Run dense BP experiment."""
    
    # Load data
    dirty_df = load_hospital_dataset(dataset_path)
    clean_df = load_hospital_dataset(clean_path)
    
    with open(error_cells_path) as f:
        error_data = json.load(f)
        error_cells = set((int(x), y) for x, y in error_data['error_cells'])
    
    with open(fds_path) as f:
        fds_data = json.load(f)
        fds = [(fd['lhs'], fd['rhs']) for fd in fds_data]
    
    print(f"Dataset: {len(dirty_df)} tuples, {len(error_cells)} errors, seed={seed}")
    
    # Run dense BP
    print("Running Dense BP...")
    repaired_df, info = dense_factor_graph_bp(
        dirty_df, fds, 
        max_iterations=100, 
        convergence_threshold=1e-6,
        seed=seed
    )
    
    # Evaluate
    metrics = evaluate_repairs(dirty_df, repaired_df, clean_df, error_cells)
    
    print(f"Results: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    print(f"Time: {info['elapsed_time']:.2f}s, Iterations: {info['iterations']}, Converged: {info['converged']}")
    print(f"Graph: {info['graph_stats'].get('n_violations', 'N/A')} violations")
    
    # Save results
    result = {
        'method': 'dense_bp',
        'dataset': dataset_path,
        'seed': seed,
        'metrics': metrics,
        'runtime_seconds': info['elapsed_time'],
        'iterations': info['iterations'],
        'converged': info['converged'],
        'graph_stats': info.get('graph_stats', {})
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--error-rate', type=int, default=10, choices=[5, 10, 15])
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running Dense BP on Hospital dataset with {args.error_rate}% error rate, seed={args.seed}")
    print(f"{'='*60}")
    
    run_experiment(
        dataset_path=f'data/hospital/hospital_dirty_{args.error_rate}pct.csv',
        clean_path='data/hospital/hospital_clean.csv',
        error_cells_path=f'data/hospital/hospital_errors_{args.error_rate}pct.json',
        fds_path='data/hospital/fds.json',
        output_path=f'results/baseline_dense_bp_{args.error_rate}pct_seed{args.seed}.json',
        seed=args.seed
    )
