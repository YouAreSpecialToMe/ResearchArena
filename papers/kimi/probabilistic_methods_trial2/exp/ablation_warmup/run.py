"""
Ablation: Cold-Start Strategy.
Test contribution of hierarchical warm-up protocol.
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


class SMAKNoWarmup(SMAKCP):
    """SMAK-S without warm-up (adaptation from t=1)."""
    
    def update_bandwidth(self, X_query, coverage_discrepancies):
        """Enable adaptation immediately."""
        # Always adapt, ignoring T0
        eta_t = self.eta
        
        weighted_discrepancy = sum(
            self.scale_weights[k] * coverage_discrepancies.get(k, 0)
            for k in range(self.K)
        )
        
        log_h = np.log(self.bandwidths[0])
        log_h = log_h - eta_t * np.sign(weighted_discrepancy)
        self.bandwidths[0] = np.clip(np.exp(log_h), 0.01, 1.0)


def run_ablation_warmup(dataset_name: str, seed: int = 0) -> dict:
    """Run ablation: Cold-Start Strategy."""
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
    
    # Run with and without warm-up
    for method_name, use_warmup in [('with_warmup', True), ('no_warmup', False)]:
        if use_warmup:
            predictor = SMAKCP(
                alpha=0.1, K=4, rho=2.0, h0=0.1, eta=0.05,
                window_size=500, n_min=20, T0=100, lambda_reg=0.1, variant='S'
            )
        else:
            predictor = SMAKNoWarmup(
                alpha=0.1, K=4, rho=2.0, h0=0.1, eta=0.05,
                window_size=500, n_min=20, T0=100, lambda_reg=0.1, variant='S'
            )
        
        predictor.fit(X_train, y_train)
        
        # Streaming evaluation - focus on first 200 steps
        n_steps = min(200, len(X_test))
        coverage_indicators = []
        
        for i in range(n_steps):
            covered, _ = predictor.predict_and_update(X_test[i:i+1], y_test[i:i+1])
            coverage_indicators.append(covered[0])
        
        coverage_indicators = np.array(coverage_indicators)
        
        # Count coverage violations (|coverage - 0.9| > 0.05)
        # Use rolling window for coverage
        violations = []
        for i in range(20, n_steps):
            rolling_coverage = np.mean(coverage_indicators[max(0, i-20):i+1])
            if abs(rolling_coverage - 0.9) > 0.05:
                violations.append(i)
        
        results[method_name] = {
            'n_violations': len(violations),
            'violation_rate': len(violations) / (n_steps - 20),
            'final_coverage': np.mean(coverage_indicators),
        }
    
    # Summary
    summary = {
        'experiment': f'ablation_warmup_{dataset_name}',
        'dataset': dataset_name,
        'seed': seed,
        'with_warmup': results['with_warmup'],
        'no_warmup': results['no_warmup'],
        'comparison': {
            'violation_reduction': results['no_warmup']['n_violations'] - results['with_warmup']['n_violations'],
            'violation_reduction_pct': (results['no_warmup']['n_violations'] - results['with_warmup']['n_violations']) / max(1, results['no_warmup']['n_violations']) * 100,
        }
    }
    
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    results = run_ablation_warmup(args.dataset, args.seed)
    
    # Save results
    output_file = f'results/ablation_warmup_{args.dataset}.json'
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"  Violation reduction: {results['comparison']['violation_reduction']} ({results['comparison']['violation_reduction_pct']:.1f}%)")


if __name__ == '__main__':
    main()
