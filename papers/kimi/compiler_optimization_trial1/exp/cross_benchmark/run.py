"""
Cross-Benchmark Generalization Analysis
Test model generalization across different benchmark types.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import set_seed, save_json, get_project_paths
from metrics import compute_metrics


def load_data():
    """Load all data."""
    paths = get_project_paths()
    df = pd.read_csv(f"{paths['data']}/processed/features.csv")
    
    feature_cols = [c for c in df.columns if c not in ['benchmark_name', 'label']]
    
    return df, feature_cols


def get_benchmark_type(benchmark_name: str) -> str:
    """Determine benchmark type from name."""
    polybench_linear = ['gemm', 'syrk', 'syr2k', 'gemver', 'gesummv', '2mm', '3mm', 
                        'doitgen', 'cholesky', 'lu']
    polybench_stencil = ['fdtd-2d', 'jacobi-2d', 'seidel-2d', 'heat-3d']
    
    if benchmark_name in polybench_linear:
        return 'linear_algebra'
    elif benchmark_name in polybench_stencil:
        return 'stencil'
    else:
        return 'synthetic'


def run_cross_benchmark_experiment(train_type: str, test_type: str, 
                                    df: pd.DataFrame, feature_cols: list, seed: int) -> Dict:
    """Train on one benchmark type, test on another."""
    set_seed(seed)
    
    # Add benchmark type to dataframe
    df = df.copy()
    df['benchmark_type'] = df['benchmark_name'].apply(get_benchmark_type)
    
    # Split by type
    train_mask = df['benchmark_type'] == train_type
    test_mask = df['benchmark_type'] == test_type
    
    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, 'label'].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, 'label'].values
    
    # Skip if insufficient data
    if len(X_train) < 5 or len(X_test) < 5:
        return None
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        objective='binary:logistic', random_state=seed,
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    return {
        'train_type': train_type,
        'test_type': test_type,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'accuracy': metrics['accuracy']
    }


def main():
    print("=" * 60)
    print("Cross-Benchmark Generalization Analysis")
    print("=" * 60)
    
    paths = get_project_paths()
    df, feature_cols = load_data()
    
    seeds = [42, 123, 456]
    results = []
    
    # Define cross-benchmark scenarios
    scenarios = [
        ('linear_algebra', 'stencil'),
        ('linear_algebra', 'synthetic'),
        ('stencil', 'linear_algebra'),
        ('stencil', 'synthetic'),
        ('synthetic', 'linear_algebra'),
        ('synthetic', 'stencil'),
    ]
    
    for train_type, test_type in scenarios:
        print(f"\nTraining on {train_type}, testing on {test_type}...")
        scenario_results = []
        
        for seed in seeds:
            result = run_cross_benchmark_experiment(train_type, test_type, 
                                                     df, feature_cols, seed)
            if result:
                scenario_results.append(result)
        
        if scenario_results:
            # Aggregate across seeds
            f1_scores = [r['f1_score'] for r in scenario_results]
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            
            results.append({
                'train_type': train_type,
                'test_type': test_type,
                'mean_f1': float(mean_f1),
                'std_f1': float(std_f1),
                'n_train': scenario_results[0]['n_train'],
                'n_test': scenario_results[0]['n_test']
            })
            
            print(f"  F1: {mean_f1:.3f} ± {std_f1:.3f} (n_train={scenario_results[0]['n_train']}, n_test={scenario_results[0]['n_test']})")
    
    # Compare to mixed training (main experiment result)
    print("\nComparison to Mixed Training (from main experiment):")
    print("  F1: 0.690 (training on all types)")
    
    # Save results
    exp_dir = paths['exp']
    save_json({'cross_benchmark_results': results}, f"{exp_dir}/cross_benchmark/results.json")
    pd.DataFrame(results).to_csv(f"{exp_dir}/cross_benchmark/results.csv", index=False)
    
    print(f"\nResults saved to: {exp_dir}/cross_benchmark/")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
