"""
SMAK-W: Weighted Aggregation Variant.
Weighted combination of quantiles across all active scales.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import time
import sys as sys_module
sys_module.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smak_cp_core import SMAKCP
from shared.metrics import compute_all_metrics
from shared.utils import save_results, set_seed


def run_smak_w(dataset_name: str, seed: int = 0) -> dict:
    """Run SMAK-W on a dataset."""
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
    
    # Initialize SMAK-W
    predictor = SMAKCP(
        alpha=0.1,
        K=4,
        rho=2.0,
        h0=0.1,
        eta=0.05,
        window_size=500,
        n_min=20,
        T0=100,
        lambda_reg=0.1,
        variant='W'
    )
    predictor.fit(X_train, y_train)
    
    # Streaming evaluation
    coverage_indicators = []
    set_widths = []
    runtimes = []
    
    for i in range(len(X_test)):
        start_time = time.time()
        covered, width = predictor.predict_and_update(X_test[i:i+1], y_test[i:i+1])
        runtime = time.time() - start_time
        
        coverage_indicators.append(covered[0])
        set_widths.append(width[0])
        runtimes.append(runtime)
    
    coverage_indicators = np.array(coverage_indicators)
    set_widths = np.array(set_widths)
    
    # Compute metrics
    metrics = compute_all_metrics(coverage_indicators, set_widths, X_test, target_coverage=0.9)
    metrics['avg_runtime_ms'] = np.mean(runtimes) * 1000
    metrics['total_runtime_s'] = np.sum(runtimes)
    
    results = {
        'experiment': f'smak_w_{dataset_name}_seed{seed}',
        'method': 'SMAK-W',
        'dataset': dataset_name,
        'seed': seed,
        'metrics': metrics,
        'config': {
            'alpha': 0.1,
            'K': 4,
            'rho': 2.0,
            'h0': 0.1,
            'eta': 0.05,
            'window_size': 500,
            'n_min': 20,
            'T0': 100,
            'lambda_reg': 0.1,
            'variant': 'W'
        },
        'coverage_history': coverage_indicators.tolist(),
        'set_width_history': set_widths.tolist(),
        'bandwidth_history': predictor.bandwidth_history,
    }
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    results = run_smak_w(args.dataset, args.seed)
    
    # Save results
    output_file = f'results/smak_w_{args.dataset}_seed{args.seed}.json'
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"  Marginal coverage: {results['metrics']['marginal_coverage']:.4f}")
    print(f"  Average set width: {results['metrics']['avg_set_width']:.4f}")
    print(f"  MSCE: {results['metrics']['msce']:.4f}")


if __name__ == '__main__':
    main()
