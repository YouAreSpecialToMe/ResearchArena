"""
Ablation: Fixed vs Adaptive Bandwidth.
Test contribution of online bandwidth adaptation.
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


class SMAKFixedBandwidth(SMAKCP):
    """SMAK-S with fixed bandwidth (no adaptation after warmup)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_bandwidth = None
    
    def update_bandwidth(self, X_query, coverage_discrepancies):
        """Disable bandwidth adaptation after warmup."""
        if self.t <= self.T0:
            # During warmup, just store the bandwidth
            super().update_bandwidth(X_query, coverage_discrepancies)
        elif self.fixed_bandwidth is None:
            # At end of warmup, fix the bandwidth
            self.fixed_bandwidth = self.bandwidths[0].copy()
        else:
            # Keep bandwidth fixed
            self.bandwidths[0] = self.fixed_bandwidth


def run_ablation_adaptation(dataset_name: str, seed: int = 0) -> dict:
    """Run ablation: Fixed vs Adaptive bandwidth."""
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
    
    # Run with adaptive bandwidth (default SMAK-S)
    for method_name, adaptive in [('adaptive', True), ('fixed', False)]:
        if adaptive:
            predictor = SMAKCP(
                alpha=0.1, K=4, rho=2.0, h0=0.1, eta=0.05,
                window_size=500, n_min=20, T0=100, lambda_reg=0.1, variant='S'
            )
        else:
            predictor = SMAKFixedBandwidth(
                alpha=0.1, K=4, rho=2.0, h0=0.1, eta=0.05,
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
        
        # Compute coverage drift over time (50-step windows)
        window_size = 50
        coverage_drift = []
        for i in range(0, len(coverage_indicators) - window_size, window_size):
            window_coverage = np.mean(coverage_indicators[i:i+window_size])
            coverage_drift.append(abs(window_coverage - 0.9))
        
        results[method_name] = {
            'metrics': metrics,
            'coverage_drift': coverage_drift,
            'coverage_history': coverage_indicators.tolist(),
        }
    
    # Summary
    summary = {
        'experiment': f'ablation_adaptation_{dataset_name}',
        'dataset': dataset_name,
        'seed': seed,
        'adaptive': results['adaptive'],
        'fixed': results['fixed'],
        'comparison': {
            'msce_reduction': results['fixed']['metrics']['msce'] - results['adaptive']['metrics']['msce'],
            'width_reduction_pct': (results['fixed']['metrics']['avg_set_width'] - results['adaptive']['metrics']['avg_set_width']) / results['fixed']['metrics']['avg_set_width'] * 100,
            'avg_coverage_drift_adaptive': np.mean(results['adaptive']['coverage_drift']),
            'avg_coverage_drift_fixed': np.mean(results['fixed']['coverage_drift']),
        }
    }
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    results = run_ablation_adaptation(args.dataset, args.seed)
    
    # Save results
    output_file = f'results/ablation_adaptation_{args.dataset}.json'
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"  MSCE reduction (adaptive vs fixed): {results['comparison']['msce_reduction']:.4f}")
    print(f"  Width reduction: {results['comparison']['width_reduction_pct']:.2f}%")


if __name__ == '__main__':
    main()
