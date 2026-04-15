"""
Ablation Study: Feature Categories
Measure contribution of each feature category (structural, access pattern, context).
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
    """Load train and test data."""
    paths = get_project_paths()
    train_df = pd.read_csv(f"{paths['data']}/processed/train.csv")
    test_df = pd.read_csv(f"{paths['data']}/processed/test.csv")
    
    feature_cols = [c for c in train_df.columns if c not in ['benchmark_name', 'label']]
    
    return train_df, test_df, feature_cols


def get_feature_categories() -> Dict[str, List[str]]:
    """Define feature categories by keyword matching."""
    return {
        'structural': [
            'num_fields', 'struct_size', 'field_size', 'size_variance',
            'num_pointer', 'num_primitive', 'is_kernel'
        ],
        'access_pattern': [
            'loop_nesting', 'access_sites', 'accesses_per_field',
            'access_variance', 'hot_cold_ratio', 'cooccurrence',
            'pointer_arith', 'dominance'
        ],
        'context': [
            'hotness', 'loop_header', 'trip_count', 'alloc_in_loop',
            'linear_algebra', 'stencil', 'synthetic'
        ]
    }


def get_feature_indices(feature_cols: List[str], category_keywords: List[str]) -> List[int]:
    """Get indices of features matching category keywords."""
    indices = []
    for i, col in enumerate(feature_cols):
        for keyword in category_keywords:
            if keyword in col.lower():
                indices.append(i)
                break
    return indices


def train_and_evaluate(X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray,
                       feature_cols: List[str], excluded_category: str = None) -> Dict:
    """Train model with optional feature category exclusion."""
    
    # Filter features if excluding a category
    if excluded_category:
        categories = get_feature_categories()
        excluded_indices = get_feature_indices(feature_cols, categories[excluded_category])
        included_indices = [i for i in range(len(feature_cols)) if i not in excluded_indices]
        
        X_train_filtered = X_train[:, included_indices]
        X_test_filtered = X_test[:, included_indices]
        n_features = len(included_indices)
    else:
        X_train_filtered = X_train
        X_test_filtered = X_test
        n_features = X_train.shape[1]
    
    # Train XGBoost
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_filtered, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_filtered)
    metrics = compute_metrics(y_test, y_pred)
    
    return {
        'n_features': n_features,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'accuracy': metrics['accuracy']
    }


def run_ablation(seed: int, train_df, test_df, feature_cols) -> Dict:
    """Run ablation study with given seed."""
    set_seed(seed)
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    results = {}
    
    # Full model (no ablation)
    results['full'] = train_and_evaluate(X_train, X_test, y_train, y_test, 
                                          feature_cols, excluded_category=None)
    
    # Ablation 1: Remove structural features
    results['no_structural'] = train_and_evaluate(X_train, X_test, y_train, y_test,
                                                   feature_cols, excluded_category='structural')
    
    # Ablation 2: Remove access pattern features
    results['no_access_pattern'] = train_and_evaluate(X_train, X_test, y_train, y_test,
                                                       feature_cols, excluded_category='access_pattern')
    
    # Ablation 3: Remove context features
    results['no_context'] = train_and_evaluate(X_train, X_test, y_train, y_test,
                                                feature_cols, excluded_category='context')
    
    return results


def main():
    print("=" * 60)
    print("Ablation Study: Feature Categories")
    print("=" * 60)
    
    paths = get_project_paths()
    train_df, test_df, feature_cols = load_data()
    
    print(f"\nTotal features: {len(feature_cols)}")
    categories = get_feature_categories()
    for cat, keywords in categories.items():
        indices = get_feature_indices(feature_cols, keywords)
        print(f"  {cat}: {len(indices)} features")
    
    # Run with multiple seeds
    seeds = [42, 123, 456]
    all_results = []
    
    for seed in seeds:
        print(f"\nRunning with seed {seed}...")
        results = run_ablation(seed, train_df, test_df, feature_cols)
        all_results.append(results)
    
    # Aggregate results
    ablation_types = ['full', 'no_structural', 'no_access_pattern', 'no_context']
    aggregated = {}
    
    for ablation_type in ablation_types:
        f1_scores = [r[ablation_type]['f1_score'] for r in all_results]
        precisions = [r[ablation_type]['precision'] for r in all_results]
        recalls = [r[ablation_type]['recall'] for r in all_results]
        
        aggregated[ablation_type] = {
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'mean_precision': float(np.mean(precisions)),
            'mean_recall': float(np.mean(recalls)),
            'performance_drop_pct': None  # Will compute below
        }
    
    # Compute performance drop relative to full model
    full_f1 = aggregated['full']['mean_f1']
    for ablation_type in ['no_structural', 'no_access_pattern', 'no_context']:
        ablation_f1 = aggregated[ablation_type]['mean_f1']
        if full_f1 > 0:
            drop = (full_f1 - ablation_f1) / full_f1 * 100
        else:
            drop = 0
        aggregated[ablation_type]['performance_drop_pct'] = float(drop)
    
    # Print results
    print("\n" + "=" * 60)
    print("Ablation Results (mean F1 across 3 seeds):")
    print("=" * 60)
    for ablation_type in ablation_types:
        r = aggregated[ablation_type]
        drop_str = f"(-{r['performance_drop_pct']:.1f}%)" if r['performance_drop_pct'] else ""
        print(f"  {ablation_type:20}: {r['mean_f1']:.3f} ± {r['std_f1']:.3f} {drop_str}")
    
    # Save results
    exp_dir = paths['exp']
    save_json(aggregated, f"{exp_dir}/ablation_features/results.json")
    
    # CSV format
    csv_data = []
    for ablation_type in ablation_types:
        r = aggregated[ablation_type]
        csv_data.append({
            'ablation_type': ablation_type,
            'mean_f1': r['mean_f1'],
            'std_f1': r['std_f1'],
            'mean_precision': r['mean_precision'],
            'mean_recall': r['mean_recall'],
            'performance_drop_pct': r['performance_drop_pct']
        })
    pd.DataFrame(csv_data).to_csv(f"{exp_dir}/ablation_features/results.csv", index=False)
    
    print(f"\nResults saved to: {exp_dir}/ablation_features/")
    
    return aggregated


if __name__ == '__main__':
    main()
