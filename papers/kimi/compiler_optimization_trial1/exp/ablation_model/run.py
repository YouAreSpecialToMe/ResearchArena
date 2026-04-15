"""
Ablation Study: Model Complexity
Study effect of model complexity on performance and inference time.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import sys
import os
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import set_seed, save_json, get_project_paths
from metrics import compute_metrics


def load_data():
    """Load train and test data."""
    paths = get_project_paths()
    train_df = pd.read_csv(f"{paths['data']}/processed/train.csv")
    test_df = pd.read_csv(f"{paths['data']}/processed/test.csv")
    
    feature_cols = [c for c in train_df.columns if c not in ['benchmark_name', 'label']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> Dict:
    """Train and evaluate a model, measuring inference time."""
    
    # Train
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    
    # Predict (single prediction timing)
    n_runs = 1000
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(X_test[:1])
    inference_time = (time.perf_counter() - start) / n_runs * 1e6  # microseconds
    
    # Full test evaluation
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    return {
        'model': model_name,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'accuracy': metrics['accuracy'],
        'train_time': train_time,
        'inference_time_us': inference_time
    }


def run_complexity_study(seed: int, X_train, X_test, y_train, y_test) -> List[Dict]:
    """Run model complexity experiments."""
    set_seed(seed)
    
    results = []
    
    # Simple: Single Decision Tree
    print("  Training Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=seed)
    result = evaluate_model(dt, X_train, X_test, y_train, y_test, 'DecisionTree')
    results.append(result)
    
    # Medium: XGBoost with n_estimators=50, max_depth=4
    print("  Training XGBoost (medium)...")
    xgb_medium = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        objective='binary:logistic', random_state=seed,
        use_label_encoder=False
    )
    result = evaluate_model(xgb_medium, X_train, X_test, y_train, y_test, 'XGBoost_Medium')
    results.append(result)
    
    # Standard: XGBoost with n_estimators=100, max_depth=6
    print("  Training XGBoost (standard)...")
    xgb_std = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        objective='binary:logistic', random_state=seed,
        use_label_encoder=False
    )
    result = evaluate_model(xgb_std, X_train, X_test, y_train, y_test, 'XGBoost_Standard')
    results.append(result)
    
    # High: XGBoost with n_estimators=200, max_depth=10
    print("  Training XGBoost (high)...")
    xgb_high = xgb.XGBClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.1,
        objective='binary:logistic', random_state=seed,
        use_label_encoder=False
    )
    result = evaluate_model(xgb_high, X_train, X_test, y_train, y_test, 'XGBoost_High')
    results.append(result)
    
    return results


def main():
    print("=" * 60)
    print("Ablation Study: Model Complexity")
    print("=" * 60)
    
    paths = get_project_paths()
    X_train, X_test, y_train, y_test, feature_cols = load_data()
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\nRunning with seed {seed}...")
        results = run_complexity_study(seed, X_train, X_test, y_train, y_test)
        for r in results:
            r['seed'] = seed
        all_results.extend(results)
    
    # Aggregate results by model
    results_df = pd.DataFrame(all_results)
    
    aggregated = {}
    for model_name in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model_name]
        aggregated[model_name] = {
            'mean_f1': float(model_data['f1_score'].mean()),
            'std_f1': float(model_data['f1_score'].std()),
            'mean_precision': float(model_data['precision'].mean()),
            'mean_recall': float(model_data['recall'].mean()),
            'mean_accuracy': float(model_data['accuracy'].mean()),
            'mean_inference_time_us': float(model_data['inference_time_us'].mean()),
            'std_inference_time_us': float(model_data['inference_time_us'].std()),
            'mean_train_time': float(model_data['train_time'].mean())
        }
    
    # Print results
    print("\n" + "=" * 60)
    print("Model Complexity Results (mean ± std across 3 seeds):")
    print("=" * 60)
    print(f"{'Model':<20} {'F1':<10} {'Inference (μs)':<15} {'Train (s)':<10}")
    print("-" * 60)
    for model_name, r in aggregated.items():
        print(f"{model_name:<20} {r['mean_f1']:.3f}±{r['std_f1']:.3f}   "
              f"{r['mean_inference_time_us']:.1f}±{r['std_inference_time_us']:.1f}      "
              f"{r['mean_train_time']:.3f}")
    
    # Save results
    exp_dir = paths['exp']
    save_json(aggregated, f"{exp_dir}/ablation_model/results.json")
    results_df.to_csv(f"{exp_dir}/ablation_model/results.csv", index=False)
    
    print(f"\nResults saved to: {exp_dir}/ablation_model/")
    
    return aggregated


if __name__ == '__main__':
    main()
