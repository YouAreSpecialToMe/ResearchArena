"""
Adaptive Conformal Inference (ACI) Baseline.
Gibbs & Candès 2021 ACI method with online miscoverage level adaptation.
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
from shared.utils import save_results, set_seed


class ACIPredictor:
    """Adaptive Conformal Inference Predictor."""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.05):
        self.alpha = alpha
        self.gamma = gamma  # Adaptation rate
        self.predictor = RidgePredictor(alpha=1.0)
        self.quantile = 0.0
        self.history = []
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the predictor."""
        self.predictor.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict interval for each point.
        Returns (lower, upper) bounds.
        """
        y_pred = self.predictor.predict(X)
        lower = y_pred - self.quantile
        upper = y_pred + self.quantile
        return lower, upper
    
    def predict_and_update(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Predict interval and update quantile.
        Update rule: q_{t+1} = q_t + gamma * (alpha - 1{y_t not in C_t})
        """
        # Predict
        lower, upper = self.predict(X)
        
        # Check coverage
        covered = (y >= lower) & (y <= upper)
        miscoverage = ~covered
        
        # Update quantile
        self.quantile = self.quantile + self.gamma * (self.alpha - miscoverage.astype(float))
        self.quantile = max(0, self.quantile)  # Keep non-negative
        
        width = upper - lower
        return covered, width


def run_aci(dataset_name: str, seed: int = 0) -> dict:
    """Run ACI on a dataset."""
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
    predictor = ACIPredictor(alpha=0.1, gamma=0.05)
    predictor.fit(X_train, y_train)
    
    # Initial quantile estimation using first part of test set
    n_init = min(100, len(X_test) // 4)
    y_init_pred = predictor.predictor.predict(X_test[:n_init])
    scores = nonconformity_score(y_test[:n_init], y_init_pred)
    predictor.quantile = compute_quantile(scores, 0.1)
    
    # Streaming evaluation
    coverage_indicators = []
    set_widths = []
    runtimes = []
    
    for i in range(n_init, len(X_test)):
        start_time = time.time()
        covered, width = predictor.predict_and_update(X_test[i:i+1], y_test[i:i+1])
        runtime = time.time() - start_time
        
        coverage_indicators.append(covered[0])
        set_widths.append(width[0])
        runtimes.append(runtime)
    
    coverage_indicators = np.array(coverage_indicators)
    set_widths = np.array(set_widths)
    
    # Compute metrics
    metrics = compute_all_metrics(coverage_indicators, set_widths, X_test[n_init:], target_coverage=0.9)
    metrics['avg_runtime_ms'] = np.mean(runtimes) * 1000
    metrics['total_runtime_s'] = np.sum(runtimes)
    
    results = {
        'experiment': f'aci_{dataset_name}_seed{seed}',
        'method': 'ACI',
        'dataset': dataset_name,
        'seed': seed,
        'metrics': metrics,
        'config': {
            'alpha': 0.1,
            'gamma': 0.05
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
    
    results = run_aci(args.dataset, args.seed)
    
    # Save results
    output_file = f'results/aci_{args.dataset}_seed{args.seed}.json'
    save_results(results, output_file)
    print(f"Results saved to {output_file}")
    print(f"  Marginal coverage: {results['metrics']['marginal_coverage']:.4f}")
    print(f"  Average set width: {results['metrics']['avg_set_width']:.4f}")
    print(f"  MSCE: {results['metrics']['msce']:.4f}")


if __name__ == '__main__':
    main()
