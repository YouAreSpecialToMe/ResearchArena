#!/usr/bin/env python3
"""
Minimum Repair baseline experiment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
import time
from shared.data_loader import load_hospital_dataset, evaluate_repairs
from shared.baselines import minimum_repair


def run_experiment(dataset_path, clean_path, error_cells_path, fds_path, output_path):
    """Run minimum repair experiment."""
    
    # Load data
    dirty_df = load_hospital_dataset(dataset_path)
    clean_df = load_hospital_dataset(clean_path)
    
    with open(error_cells_path) as f:
        error_data = json.load(f)
        error_cells = set((int(x), y) for x, y in error_data['error_cells'])
    
    with open(fds_path) as f:
        fds_data = json.load(f)
        fds = [(fd['lhs'], fd['rhs']) for fd in fds_data]
    
    print(f"Dataset: {len(dirty_df)} tuples, {len(error_cells)} errors")
    
    # Run minimum repair
    print("Running Minimum Repair...")
    start_time = time.time()
    repaired_df = minimum_repair(dirty_df, fds, cost_model='uniform')
    elapsed = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_repairs(dirty_df, repaired_df, clean_df, error_cells)
    
    print(f"Results: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    print(f"Time: {elapsed:.2f}s")
    
    # Save results
    result = {
        'method': 'minimum_repair',
        'dataset': dataset_path,
        'metrics': metrics,
        'runtime_seconds': elapsed
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    # Run on different error rates
    for error_rate in [5, 10, 15]:
        print(f"\n{'='*60}")
        print(f"Running on Hospital dataset with {error_rate}% error rate")
        print(f"{'='*60}")
        
        run_experiment(
            dataset_path=f'data/hospital/hospital_dirty_{error_rate}pct.csv',
            clean_path='data/hospital/hospital_clean.csv',
            error_cells_path=f'data/hospital/hospital_errors_{error_rate}pct.json',
            fds_path='data/hospital/fds.json',
            output_path=f'results/baseline_minimum_repair_{error_rate}pct.json'
        )
