#!/usr/bin/env python3
"""
Ablation study: Impact of violation-driven sparsification.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
import time
from shared.data_loader import load_hospital_dataset, evaluate_repairs
from shared.cleanbp import CleanBP
from shared.baselines import dense_factor_graph_bp


def run_ablation(variant, dataset_path, clean_path, error_cells_path, fds_path, output_path, seed):
    """Run ablation experiment."""
    
    # Load data
    dirty_df = load_hospital_dataset(dataset_path)
    clean_df = load_hospital_dataset(clean_path)
    
    with open(error_cells_path) as f:
        error_data = json.load(f)
        error_cells = set((int(x), y) for x, y in error_data['error_cells'])
    
    with open(fds_path) as f:
        fds_data = json.load(f)
        fds = [(fd['lhs'], fd['rhs']) for fd in fds_data]
    
    print(f"Variant: {variant}, Dataset: {len(dirty_df)} tuples, seed={seed}")
    
    start_time = time.time()
    
    if variant == 'dense':
        # Dense factor graph (no sparsification)
        repaired_df, info = dense_factor_graph_bp(dirty_df, fds, seed=seed)
        graph_stats = info.get('graph_stats', {})
    
    elif variant == 'violation_only':
        # Only violation-driven sparsification, no attribute separation
        cleanbp = CleanBP(max_iterations=50, convergence_threshold=1e-6, damping=0.5)
        repaired_df, info = cleanbp.repair(
            dirty_df, fds,
            violation_only=True,
            separate_attributes=True,
            priority_scheduling=False
        )
        graph_stats = info.get('graph_stats', {})
    
    elif variant == 'full':
        # Full CleanBP
        cleanbp = CleanBP(max_iterations=50, convergence_threshold=1e-6, damping=0.5)
        repaired_df, info = cleanbp.repair(
            dirty_df, fds,
            violation_only=True,
            separate_attributes=True,
            priority_scheduling=True
        )
        graph_stats = info.get('graph_stats', {})
    
    elapsed = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_repairs(dirty_df, repaired_df, clean_df, error_cells)
    
    print(f"Results: F1={metrics['f1']:.4f}, Time={elapsed:.2f}s")
    
    # Save results
    result = {
        'variant': variant,
        'dataset': dataset_path,
        'seed': seed,
        'metrics': metrics,
        'runtime_seconds': elapsed,
        'graph_stats': graph_stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['dense', 'violation_only', 'full'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running ablation: {args.variant}, seed={args.seed}")
    print(f"{'='*60}")
    
    run_ablation(
        variant=args.variant,
        dataset_path='data/hospital/hospital_dirty_10pct.csv',
        clean_path='data/hospital/hospital_clean.csv',
        error_cells_path='data/hospital/hospital_errors_10pct.json',
        fds_path='data/hospital/fds.json',
        output_path=f'results/ablation_sparsification_{args.variant}_seed{args.seed}.json',
        seed=args.seed
    )
