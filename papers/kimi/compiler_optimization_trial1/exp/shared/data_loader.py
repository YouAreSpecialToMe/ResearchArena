"""
Data loading and preprocessing utilities for LayoutLearner experiments.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import json


def load_features_and_labels(features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load features and labels from CSV files."""
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    
    # Get feature column names (excluding metadata)
    feature_cols = [c for c in features_df.columns 
                   if c not in ['benchmark_name', 'struct_id', 'field_id', 'label']]
    
    X = features_df[feature_cols].values
    y = labels_df['label'].values if 'label' in labels_df.columns else labels_df.values.ravel()
    
    return X, y, feature_cols


def split_by_benchmark(features_df: pd.DataFrame, labels: np.ndarray, 
                       train_benchmarks: List[str], test_benchmarks: List[str]) -> Tuple:
    """Split data by benchmark names to ensure generalization."""
    train_mask = features_df['benchmark_name'].isin(train_benchmarks)
    test_mask = features_df['benchmark_name'].isin(test_benchmarks)
    
    feature_cols = [c for c in features_df.columns 
                   if c not in ['benchmark_name', 'struct_id', 'field_id', 'label']]
    
    X_train = features_df.loc[train_mask, feature_cols].values
    y_train = labels[train_mask.values]
    X_test = features_df.loc[test_mask, feature_cols].values
    y_test = labels[test_mask.values]
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score normalization based on training set statistics."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_test_norm


def save_splits(output_dir: str, splits: Dict):
    """Save train/test splits to disk."""
    for key, value in splits.items():
        np.save(f"{output_dir}/{key}.npy", value)
    
    with open(f"{output_dir}/split_info.json", 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
                  for k, v in splits.items()}, f, indent=2)


def get_feature_categories() -> Dict[str, List[str]]:
    """Get feature categories for ablation studies."""
    return {
        'structural': [
            'num_fields', 'struct_size', 'field_size', 'field_offset',
            'is_pointer', 'is_primitive', 'alignment'
        ],
        'access_pattern': [
            'loop_nesting_depth', 'num_access_sites', 'num_loops_accessed',
            'cooccurrence_score', 'has_pointer_arith', 'dominates_other',
            'post_dominates_other'
        ],
        'context': [
            'estimated_hotness', 'in_loop_header', 'trip_count_estimate',
            'alloc_in_loop', 'alloc_size_known'
        ]
    }


def get_feature_indices(all_features: List[str], category_features: List[str]) -> List[int]:
    """Get indices of features belonging to a category."""
    indices = []
    for cf in category_features:
        for i, af in enumerate(all_features):
            if cf in af:
                indices.append(i)
                break
    return indices
