#!/usr/bin/env python3
"""
Baseline: Simple Metadata Regression for Zero-Shot Prediction.
Uses Ridge regression to predict model performance from metadata features.
FIXED: Proper seed propagation.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from exp.shared.data_loader import MMLUDataset
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation,
    compute_kendall_tau, compute_r2_score
)


def run_baseline_metadata_regression(seed: int) -> dict:
    """Run metadata regression baseline with a specific seed."""
    # FIXED: Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    train_models = split['train_models']
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Prepare training data
    X_train = []
    y_train = []
    
    for model_name in train_models:
        metadata = dataset.models[model_name]
        features = metadata.to_features()
        
        # Target: mean ability (scaled)
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        target = np.mean(true_ability)
        
        X_train.append(features)
        y_train.append(target)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Ridge regression
    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(X_train_scaled, y_train)
    
    # Predict on test models
    X_test = []
    y_true = []
    
    for model_name in test_models:
        metadata = dataset.models[model_name]
        features = metadata.to_features()
        
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        target = np.mean(true_ability)
        
        X_test.append(features)
        y_true.append(target)
    
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    metrics = {
        'mae': compute_mae(y_pred, y_true),
        'rmse': compute_rmse(y_pred, y_true),
        'spearman': compute_spearman_correlation(y_pred, y_true),
        'kendall': compute_kendall_tau(y_pred, y_true),
        'r2': compute_r2_score(y_pred, y_true),
        'items_used': 0  # Zero-shot
    }
    
    return metrics, y_pred, y_true


def main():
    print("=" * 60)
    print("Baseline: Metadata Regression (Zero-Shot)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        metrics, predictions, true_values = run_baseline_metadata_regression(seed)
        all_metrics.append(metrics)
        print(f"Results: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}, R²={metrics['r2']:.4f}")
    
    runtime = (time.time() - start_time) / 60
    
    # Aggregate metrics
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'r2', 'items_used']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    results = {
        'experiment': 'baseline_metadata_regression',
        'description': 'Ridge regression predicting performance from metadata (zero-shot)',
        'metrics': aggregated,
        'config': {
            'model': 'Ridge',
            'alpha': 1.0,
            'seeds': seeds,
            'n_train_models': 60,
            'n_test_models': 20
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/baseline_metadata_regression/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Final Results (mean ± std across seeds):")
    print(f"  MAE: {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Kendall: {aggregated['kendall']['mean']:.4f} ± {aggregated['kendall']['std']:.4f}")
    print(f"  R²: {aggregated['r2']['mean']:.4f} ± {aggregated['r2']['std']:.4f}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
