"""
Fixed-Bandwidth Kernel CP Baseline.
Single-scale kernel-based CP with fixed bandwidth (represents standard kernel CP methods like RLCP).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import json
import time
from shared.models import RidgePredictor, nonconformity_score, compute_quantile
from shared.metrics import compute_all_metrics
from shared.utils import save_results, set_seed, kernel_weights


class FixedKernelCP:
    """Fixed-Bandwidth Kernel Conformal Predictor."""
    
    def __init__(self, alpha: float = 0.1, bandwidth: float = 0.1, window_size: int = 500):
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.window_size = window_size
        self.predictor = RidgePredictor(alpha=1.0)
        
        # History buffers
        self.X_history = []
        self.y_history = []
        self.scores_history = []
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the predictor."""
        self.predictor.fit(X_train, y_train)
    
    def compute_weighted_quantile(self, X_query: np.ndarray) -> float:
        """Compute weighted quantile based on kernel weights."""
        if len(self.scores_history) == 0:
            return 0.0
        
        # Get recent window
        X_recent = np.array(self.X_history[-self.window_size:])
        scores_recent = np.array(self.scores_history[-self.window_size:])
        
        # Compute kernel weights
        weights = kernel_weights(X_query, X_recent, self.bandwidth)
        
        if np.sum(weights) == 0:
            # Fallback to unweighted quantile
            return compute_quantile(scores_recent, self.alpha)
        
        return compute_quantile(scores_recent, self.alpha, weights)
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict interval for each point.
        Returns (lower, upper) bounds.
        """
        y_pred = self.predictor.predict(X)
        
        # Compute weighted quantile for each point
        quantiles = np.array([self.compute_weighted_quantile(x) for x in X])
        
        lower = y_pred - quantiles
        upper = y_pred + quantiles
        return lower, upper
    
    def predict_and_update(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Predict interval and update history."""
        # Predict
        lower, upper = self.predict(X)
        
        # Check coverage
        covered = (y >= lower) & (y <= upper)
        width = upper - lower
        
        # Update history
        y_pred = self.predictor.predict(X)
        scores = nonconformity_score(y, y_pred)
        
        self.X_history.append(X[0])
        self.y_history.append(y[0])
        self.scores_history.append(scores[0])
        
        # Trim history if needed
        if len(self.X_history) > self.window_size * 2:
            self.X_history = self.X_history[-self.window_size:]
            self.y_history = self.y_history[-self.window_size:]
            self.scores_history = self.scores_history[-self.window_size:]
        
        return covered, width


def run_fixed_kernel_cp(dataset_name: str, seed: int = 0) -> dict:
    """Run Fixed Kernel CP on a dataset."""
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
    
    # Initialize and fit predictor
    predictor = FixedKernelCP(alpha=0.1, bandwidth=0.1, window_size=500)
    predictor.fit(X_train, y_train)
    
    # Warm-up: add initial data to history
    n_warmup = min(100, len(X_test) // 4)
    for i in range(n_warmup):
        y_pred = predictor.predictor.predict(X_test[i:i+1])
        score = nonconformity_score(y_test[i:i+1], y_pred)[0]
        predictor.X_history.append(X_test[i])
        predictor.y_history.append(y_test[i])
        predictor.scores_history.append(score)
    
    # Streaming evaluation
    coverage_indicators = []
    set_widths = []
    runtimes = []
    
    for i in range(n_warmup, len(X_test)):
        start_time = time.time()
        covered, width = predictor.predict_and_update(X_test[i:i+1], y_test[i:i+1])
        runtime = time.time() - start_time
        
        coverage_indicators.append(covered[0])
        set_widths.append(width[0])
        runtimes.append(runtime)
    
    coverage_indicators = np.array(coverage_indicators)
    set_widths = np.array(set_widths)
    
    # Compute metrics
    metrics = compute_all_metrics(coverage_indicators, set_widths, X_test[n_warmup:], target_coverage=0.9)
    metrics['avg_runtime_ms'] = np.mean(runtimes) * 1000
    metrics['total_runtime_s'] = np.sum(runtimes)
    
    results = {
        'experiment': f'fixed_kernel_cp_{dataset_name}_seed{seed}',
        'method': 'FixedKernelCP',
        'dataset': dataset_name,
        'seed': seed,
        'metrics': metrics,
        'config': {
            'alpha': 0.1,
            'bandwidth': 0.1,
            'window_size': 500
        },
        'coverage_history': coverage_indicators.tolist(),
        'set_width_history': set_widths.tolist(),
    }
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    results = run_fixed_kernel_cp(args.dataset, args.seed)
    
    # Save results
    output_file = f'results/fixed_kernel_cp_{args.dataset}_seed{args.seed}.json'
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"  Marginal coverage: {results['metrics']['marginal_coverage']:.4f}")
    print(f"  Average set width: {results['metrics']['avg_set_width']:.4f}")
    print(f"  MSCE: {results['metrics']['msce']:.4f}")


if __name__ == '__main__':
    main()
