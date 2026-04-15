#!/usr/bin/env python3
"""
Experiment 07: LEOPARD Train Rule Scorer
- Train MLP: 2 layers, 32 hidden units
- Train GBDT: 30 trees, max_depth=4
- Select best model based on validation performance
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

def train_mlp(X_train, y_train, X_val, y_val, seed=42):
    """Train small MLP."""
    print("  Training MLP (2 layers, 32 units)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = mlp.predict(X_val_scaled)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    pearson_r, _ = pearsonr(y_val, y_pred)
    spearman_r, _ = spearmanr(y_val, y_pred)
    
    # Count parameters
    n_params = (
        X_train.shape[1] * 32 + 32 +  # First layer
        32 * 32 + 32 +  # Second layer
        32 * 1 + 1  # Output layer
    )
    
    # Inference time
    start = time.time()
    for _ in range(1000):
        _ = mlp.predict(X_val_scaled[:1])
    inference_ms = (time.time() - start) * 1000 / 1000
    
    return {
        'model': mlp,
        'scaler': scaler,
        'mse': mse,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'n_params': n_params,
        'inference_ms': inference_ms,
        'name': 'MLP'
    }

def train_gbdt(X_train, y_train, X_val, y_val, seed=42):
    """Train Gradient Boosted Trees."""
    print("  Training GBDT (30 trees, max_depth=4)...")
    
    gbdt = GradientBoostingRegressor(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
        subsample=0.8
    )
    
    gbdt.fit(X_train, y_train)
    
    # Evaluate
    y_pred = gbdt.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    pearson_r, _ = pearsonr(y_val, y_pred)
    spearman_r, _ = spearmanr(y_val, y_pred)
    
    # Estimate parameters
    n_params = sum(
        tree.tree_.node_count * 3  # Rough estimate: 3 values per node
        for tree in gbdt.estimators_.flatten()
    )
    
    # Inference time
    start = time.time()
    for _ in range(1000):
        _ = gbdt.predict(X_val[:1])
    inference_ms = (time.time() - start) * 1000 / 1000
    
    return {
        'model': gbdt,
        'scaler': None,
        'mse': mse,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'n_params': n_params,
        'inference_ms': inference_ms,
        'name': 'GBDT'
    }

def main():
    print("=" * 60)
    print("Experiment 07: LEOPARD Train Rule Scorer")
    print("=" * 60)
    
    # Load training data
    df = pd.read_csv("data/training_data.csv")
    
    # Prepare features and target
    feature_cols = [c for c in df.columns if c.startswith('f')]
    X = df[feature_cols].values
    y = df['eventual_improvement'].values
    
    print(f"\nDataset: {len(df)} samples, {len(feature_cols)} features")
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Train models
    print("\n[1/2] Training models...")
    mlp_results = train_mlp(X_train, y_train, X_val, y_val)
    gbdt_results = train_gbdt(X_train, y_train, X_val, y_val)
    
    # Compare and select best
    print("\n[2/2] Model comparison:")
    print(f"\n  MLP:")
    print(f"    MSE: {mlp_results['mse']:.4f}")
    print(f"    R²: {mlp_results['r2']:.4f}")
    print(f"    Pearson r: {mlp_results['pearson_r']:.4f}")
    print(f"    Parameters: {mlp_results['n_params']:,}")
    print(f"    Inference: {mlp_results['inference_ms']:.4f} ms")
    
    print(f"\n  GBDT:")
    print(f"    MSE: {gbdt_results['mse']:.4f}")
    print(f"    R²: {gbdt_results['r2']:.4f}")
    print(f"    Pearson r: {gbdt_results['pearson_r']:.4f}")
    print(f"    Parameters: {gbdt_results['n_params']:,}")
    print(f"    Inference: {gbdt_results['inference_ms']:.4f} ms")
    
    # Select best based on Pearson correlation
    if mlp_results['pearson_r'] > gbdt_results['pearson_r']:
        best_model = mlp_results
        print(f"\n  Selected: MLP")
    else:
        best_model = gbdt_results
        print(f"\n  Selected: GBDT")
    
    # Save model
    model_data = {
        'model': best_model['model'],
        'scaler': best_model['scaler'],
        'model_type': best_model['name'],
        'metrics': {
            'mse': best_model['mse'],
            'r2': best_model['r2'],
            'pearson_r': best_model['pearson_r'],
            'spearman_r': best_model['spearman_r'],
        },
        'model_info': {
            'n_params': best_model['n_params'],
            'inference_ms': best_model['inference_ms']
        }
    }
    
    with open("models/leopard_scorer.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    # Save results
    results = {
        "mlp": {
            "mse": mlp_results['mse'],
            "r2": mlp_results['r2'],
            "pearson_r": mlp_results['pearson_r'],
            "spearman_r": mlp_results['spearman_r'],
            "n_params": mlp_results['n_params'],
            "inference_ms": mlp_results['inference_ms']
        },
        "gbdt": {
            "mse": gbdt_results['mse'],
            "r2": gbdt_results['r2'],
            "pearson_r": gbdt_results['pearson_r'],
            "spearman_r": gbdt_results['spearman_r'],
            "n_params": gbdt_results['n_params'],
            "inference_ms": gbdt_results['inference_ms']
        },
        "selected": best_model['name'],
        "selected_metrics": {
            "mse": best_model['mse'],
            "r2": best_model['r2'],
            "pearson_r": best_model['pearson_r'],
            "n_params": best_model['n_params'],
            "inference_ms": best_model['inference_ms']
        }
    }
    
    with open("results/training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("exp/07_train_scorer/results.json", "w") as f:
        json.dump({
            "experiment": "07_train_scorer",
            "status": "completed",
            "selected_model": best_model['name'],
            "metrics": results["selected_metrics"]
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Selected model: {best_model['name']}")
    print(f"Pearson r: {best_model['pearson_r']:.4f}")
    print(f"Inference time: {best_model['inference_ms']:.4f} ms")
    print(f"Model saved to models/leopard_scorer.pkl")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())
