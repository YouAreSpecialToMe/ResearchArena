"""
Ablation: Multi-Scale vs Single-Scale.
Test contribution of multi-scale structure.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import time
from smak_cp_core import SMAKCP
from shared.metrics import compute_all_metrics
from shared.utils import save_results, set_seed


def run_ablation_multiscale(dataset_name: str, seed: int = 0) -> dict:
    """Run ablation: Multi-Scale vs Single-Scale."""
    set_seed(seed)
    
    # Load data
    df = pd.read_csv(f'data/{dataset_name}.csv')
    
    # Determine target column
    if dataset_name in ['synthetic_hetero', 'synthetic_drift', 'synthetic_density']:
        target_col = 'Y'
        feature_cols = ['X']
    elif dataset_name == 'bike_sharing':
        target_col = 'cnt'
        feature_cols = ['temp', 'atemp', 'hum', 'windspeed', 'hr']
    elif dataset_name == 'concrete':
        target_col = 'strength'
        feature_cols = ['cement', 'slag', 'flyash', 'water', 'superplasticizer', 
                       'coarseaggregate', 'fineaggregate', 'age']
    elif dataset_name == 'air_quality':
        target_col = 'CO_GT'
        feature_cols = ['sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split into train and test
    n = len(X)
    n_train = int(0.4 * n)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    results = {}
    
    # Run with multi-scale (K=4) and single-scale (K=1)
    for method_name, K in [('multiscale', 4), ('singlescale', 1)]:
        predictor = SMAKCP(
            alpha=0.1, K=K, rho=2.0, h0=0.28 if K == 1 else 0.1, eta=0.05,
            window_size=500, n_min=20, T0=100, lambda_reg=0.1, variant='S'
        )
        predictor.fit(X_train, y_train)
        
        # Streaming evaluation
        coverage_indicators = []
        set_widths = []
        
        for i in range(len(X_test)):
            covered, width = predictor.predict_and_update(X_test[i:i+1], y_test[i:i+1])
            coverage_indicators.append(covered[0])
            set_widths.append(width[0])
        
        coverage_indicators = np.array(coverage_indicators)
        set_widths = np.array(set_widths)
        
        # Compute metrics
        metrics = compute_all_metrics(coverage_indicators, set_widths, X_test, target_coverage=0.9)
        
        # Compute density-based analysis
        from shared.metrics import compute_coverage_by_density
        density_analysis = compute_coverage_by_density(coverage_indicators, set_widths, X_test, k=50)
        
        results[method_name] = {
            'metrics': metrics,
            'density_analysis': density_analysis,
        }
    
    # Summary
    summary = {
        'experiment': f'ablation_multiscale_{dataset_name}',
        'dataset': dataset_name,
        'seed': seed,
        'multiscale': results['multiscale'],
        'singlescale': results['singlescale'],
        'comparison': {
            'msce_reduction': results['singlescale']['metrics']['msce'] - results['multiscale']['metrics']['msce'],
            'width_reduction_pct': (results['singlescale']['metrics']['avg_set_width'] - results['multiscale']['metrics']['avg_set_width']) / results['singlescale']['metrics']['avg_set_width'] * 100,
        }
    }
    
    # Add density-specific comparison if available
    if 'low' in results['multiscale']['density_analysis'] and 'low' in results['singlescale']['density_analysis']:
        summary['comparison']['msce_reduction_low_density'] = (
            results['singlescale']['density_analysis']['low']['coverage'] - 
            results['multiscale']['density_analysis']['low']['coverage']
        )
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    results = run_ablation_multiscale(args.dataset, args.seed)
    
    # Save results
    output_file = f'results/ablation_multiscale_{args.dataset}.json'
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"  MSCE reduction (multi vs single): {results['comparison']['msce_reduction']:.4f}")
    print(f"  Width reduction: {results['comparison']['width_reduction_pct']:.2f}%")


if __name__ == '__main__':
    main()
