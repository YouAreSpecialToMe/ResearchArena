"""
Static Heuristics Baseline for LayoutLearner.
Implements simple static heuristics adapted from Ball & Larus for data layout decisions.
"""
import numpy as np
import pandas as pd
import sys
import os
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from utils import set_seed, save_json, get_project_paths
from metrics import compute_metrics


def load_data():
    """Load train and test data."""
    paths = get_project_paths()
    train_df = pd.read_csv(f"{paths['data']}/processed/train.csv")
    test_df = pd.read_csv(f"{paths['data']}/processed/test.csv")
    return train_df, test_df


def apply_heuristics(df: pd.DataFrame, weights: Dict[str, float] = None) -> np.ndarray:
    """
    Apply static heuristics to predict profitable layouts.
    
    Heuristics:
    1. Fields accessed in deeply nested loops are hot
    2. Fields with frequent accesses are hot  
    3. Fields accessed together should be grouped (cooccurrence)
    """
    if weights is None:
        weights = {'loop_depth': 0.33, 'access_freq': 0.33, 'cooccurrence': 0.34}
    
    # Heuristic 1: Loop nesting depth score (deeper = hotter)
    loop_depth_score = df['max_loop_nesting_depth'].values / 4.0  # Normalize to [0,1]
    
    # Heuristic 2: Access frequency score (more accesses = hotter)
    access_freq_score = np.log1p(df['total_access_sites'].values) / np.log(100)
    access_freq_score = np.clip(access_freq_score, 0, 1)
    
    # Heuristic 3: Co-occurrence score (fields accessed together should be grouped)
    cooccurrence_score = df['cooccurrence_score'].values
    
    # Combined score - higher means more likely to need layout optimization
    combined_score = (
        weights['loop_depth'] * loop_depth_score +
        weights['access_freq'] * access_freq_score +
        weights['cooccurrence'] * cooccurrence_score
    )
    
    # Threshold for binary prediction (tune based on validation)
    threshold = 0.5
    predictions = (combined_score > threshold).astype(int)
    
    return predictions


def run_experiment(seed: int) -> Dict:
    """Run heuristic baseline experiment with given seed."""
    set_seed(seed)
    
    # Load data
    train_df, test_df = load_data()
    
    # Get feature columns
    feature_cols = [c for c in train_df.columns if c not in ['benchmark_name', 'label']]
    
    # Apply heuristics
    y_pred = apply_heuristics(test_df)
    y_true = test_df['label'].values
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    return {
        'seed': seed,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'accuracy': metrics['accuracy']
    }


def main():
    print("=" * 60)
    print("Baseline Experiment: Static Heuristics")
    print("=" * 60)
    
    paths = get_project_paths()
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        print(f"\nRunning with seed {seed}...")
        result = run_experiment(seed)
        results.append(result)
        print(f"  Precision: {result['precision']:.3f}")
        print(f"  Recall:    {result['recall']:.3f}")
        print(f"  F1-Score:  {result['f1_score']:.3f}")
    
    # Aggregate results
    import pandas as pd
    results_df = pd.DataFrame(results)
    
    aggregated = {
        'experiment': 'baseline_heuristics',
        'seeds': results,
        'mean': {
            'precision': float(results_df['precision'].mean()),
            'recall': float(results_df['recall'].mean()),
            'f1_score': float(results_df['f1_score'].mean()),
            'accuracy': float(results_df['accuracy'].mean())
        },
        'std': {
            'precision': float(results_df['precision'].std()),
            'recall': float(results_df['recall'].std()),
            'f1_score': float(results_df['f1_score'].std()),
            'accuracy': float(results_df['accuracy'].std())
        }
    }
    
    # Save results
    exp_dir = paths['exp']
    save_json(aggregated, f"{exp_dir}/baseline_heuristics/results.json")
    results_df.to_csv(f"{exp_dir}/baseline_heuristics/results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("Aggregated Results (mean ± std):")
    print("=" * 60)
    for metric in ['precision', 'recall', 'f1_score', 'accuracy']:
        print(f"  {metric:12}: {aggregated['mean'][metric]:.3f} ± {aggregated['std'][metric]:.3f}")
    
    print(f"\nResults saved to: {exp_dir}/baseline_heuristics/")
    
    return aggregated


if __name__ == '__main__':
    main()
