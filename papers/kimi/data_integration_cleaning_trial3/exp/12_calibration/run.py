#!/usr/bin/env python3
"""
Uncertainty calibration analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import json
import numpy as np
from shared.data_loader import load_hospital_dataset, evaluate_repairs
from shared.cleanbp import CleanBP
from shared.metrics import compute_expected_calibration_error, compute_brier_score


def run_calibration(dataset_path, clean_path, error_cells_path, fds_path, output_path, seed):
    """Run calibration experiment."""
    
    # Load data
    dirty_df = load_hospital_dataset(dataset_path)
    clean_df = load_hospital_dataset(clean_path)
    
    with open(error_cells_path) as f:
        error_data = json.load(f)
        error_cells = set((int(x), y) for x, y in error_data['error_cells'])
    
    with open(fds_path) as f:
        fds_data = json.load(f)
        fds = [(fd['lhs'], fd['rhs']) for fd in fds_data]
    
    print(f"Dataset: {len(dirty_df)} tuples, seed={seed}")
    
    # Run CleanBP
    cleanbp = CleanBP(max_iterations=50, convergence_threshold=1e-6, damping=0.5)
    repaired_df, info = cleanbp.repair(
        dirty_df, fds,
        violation_only=True,
        separate_attributes=True,
        priority_scheduling=True
    )
    
    # Collect confidences and accuracies for repaired cells
    confidences = []
    accuracies = []
    
    for cell, marginal in info['marginals'].items():
        row_idx, attr = cell
        if row_idx >= len(dirty_df) or row_idx >= len(clean_df):
            continue
        
        dirty_val = dirty_df.at[row_idx, attr]
        clean_val = clean_df.at[row_idx, attr]
        
        # Only consider cells that were changed
        map_val = marginal['map_value']
        conf = marginal['map_confidence']
        
        if map_val != dirty_val:
            confidences.append(conf)
            accuracies.append(1.0 if map_val == clean_val else 0.0)
    
    if len(confidences) > 0:
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Compute ECE
        ece = compute_expected_calibration_error(confidences, accuracies, n_bins=10)
        
        # Compute Brier score
        brier = compute_brier_score(confidences, accuracies)
        
        print(f"  Repairs made: {len(confidences)}")
        print(f"  ECE: {ece:.4f}")
        print(f"  Brier score: {brier:.4f}")
    else:
        ece = None
        brier = None
        print("  No repairs made")
    
    # Evaluate overall metrics
    metrics = evaluate_repairs(dirty_df, repaired_df, clean_df, error_cells)
    
    print(f"  F1: {metrics['f1']:.4f}")
    
    # Save results
    result = {
        'dataset': dataset_path,
        'seed': seed,
        'metrics': metrics,
        'ece': ece,
        'brier_score': brier,
        'n_repairs_evaluated': len(confidences) if len(confidences) > 0 else 0,
        'confidences': confidences.tolist() if len(confidences) > 0 else [],
        'accuracies': accuracies.tolist() if len(accuracies) > 0 else []
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Running calibration analysis, seed={args.seed}")
    print(f"{'='*60}")
    
    run_calibration(
        dataset_path='data/hospital/hospital_dirty_10pct.csv',
        clean_path='data/hospital/hospital_clean.csv',
        error_cells_path='data/hospital/hospital_errors_10pct.json',
        fds_path='data/hospital/fds.json',
        output_path=f'results/calibration_hospital_seed{args.seed}.json',
        seed=args.seed
    )
