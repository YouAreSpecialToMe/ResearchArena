"""
Data Preparation: Generate synthetic benchmarks and extract features.
This simulates LLVM IR feature extraction without requiring llvmlite.
"""
import numpy as np
import pandas as pd
import os
import sys
import json
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
from utils import set_seed, save_json, get_project_paths


def generate_synthetic_benchmarks(seed: int = 42) -> Dict:
    """
    Generate synthetic benchmarks with known optimal layouts.
    Simulates 20 diverse benchmarks with structure access patterns.
    """
    set_seed(seed)
    
    benchmarks = []
    
    # PolyBench-style kernels (14 benchmarks)
    polybench_configs = [
        {'name': 'gemm', 'type': 'linear_algebra', 'loops': 3, 'accesses': 8},
        {'name': 'syrk', 'type': 'linear_algebra', 'loops': 3, 'accesses': 6},
        {'name': 'syr2k', 'type': 'linear_algebra', 'loops': 3, 'accesses': 10},
        {'name': 'gemver', 'type': 'linear_algebra', 'loops': 2, 'accesses': 12},
        {'name': 'gesummv', 'type': 'linear_algebra', 'loops': 2, 'accesses': 6},
        {'name': '2mm', 'type': 'linear_algebra', 'loops': 3, 'accesses': 10},
        {'name': '3mm', 'type': 'linear_algebra', 'loops': 3, 'accesses': 12},
        {'name': 'doitgen', 'type': 'linear_algebra', 'loops': 4, 'accesses': 8},
        {'name': 'cholesky', 'type': 'linear_algebra', 'loops': 3, 'accesses': 6},
        {'name': 'lu', 'type': 'linear_algebra', 'loops': 3, 'accesses': 6},
        {'name': 'fdtd-2d', 'type': 'stencil', 'loops': 3, 'accesses': 15},
        {'name': 'jacobi-2d', 'type': 'stencil', 'loops': 3, 'accesses': 9},
        {'name': 'seidel-2d', 'type': 'stencil', 'loops': 3, 'accesses': 9},
        {'name': 'heat-3d', 'type': 'stencil', 'loops': 4, 'accesses': 12},
    ]
    
    # Synthetic microbenchmarks with known patterns (6 benchmarks)
    synthetic_configs = [
        {'name': 'hot_cold_split', 'type': 'synthetic', 'pattern': 'hot_cold'},
        {'name': 'aos_to_soa', 'type': 'synthetic', 'pattern': 'aos_soa'},
        {'name': 'linked_list', 'type': 'synthetic', 'pattern': 'pointer_chase'},
        {'name': 'tree_traversal', 'type': 'synthetic', 'pattern': 'tree'},
        {'name': 'cache_conflict', 'type': 'synthetic', 'pattern': 'conflict'},
        {'name': 'field_reorder', 'type': 'synthetic', 'pattern': 'reorder'},
    ]
    
    all_configs = polybench_configs + synthetic_configs
    
    for config in all_configs:
        # Generate struct definitions for this benchmark (2-4 structs per benchmark)
        num_structs = np.random.randint(2, 5)
        for s_idx in range(num_structs):
            benchmark = generate_benchmark_struct(config, s_idx)
            benchmarks.append(benchmark)
    
    return {'benchmarks': benchmarks, 'total': len(benchmarks)}


def generate_benchmark_struct(config: Dict, struct_idx: int) -> Dict:
    """Generate a single benchmark structure with features and label."""
    
    # Number of fields (2-8)
    num_fields = np.random.randint(2, 9)
    
    # Generate structural features
    field_sizes = np.random.choice([4, 8, 16, 32, 64], num_fields)
    struct_size = np.sum(field_sizes) + np.random.randint(0, 16)  # Add padding
    
    # Generate access pattern features for each field
    loop_nesting = np.random.randint(0, 5, num_fields)
    num_accesses = np.random.randint(1, 20, num_fields)
    
    # Hot fields are those accessed in deep loops with many accesses
    hotness_scores = loop_nesting * 2 + np.log1p(num_accesses)
    hot_threshold = np.percentile(hotness_scores, 50)
    
    # Determine optimal layout label (1 = should split/reorder)
    # Splitting is profitable when there's clear hot/cold separation
    hot_fields = hotness_scores > hot_threshold
    num_hot = np.sum(hot_fields)
    num_cold = num_fields - num_hot
    
    # Profitable if there's a clear separation - use higher threshold for diversity
    is_profitable = (num_hot >= 1 and num_cold >= 1 and 
                     np.std(hotness_scores) > 1.0 and
                     np.random.random() > 0.3)  # Add randomness for balance
    
    # Generate feature vector (all numeric)
    features = {
        'benchmark_name': config['name'],
        'struct_id': struct_idx,
        'num_fields': num_fields,
        'struct_size': struct_size,
        'avg_field_size': float(np.mean(field_sizes)),
        'max_field_size': int(np.max(field_sizes)),
        'min_field_size': int(np.min(field_sizes)),
        'size_variance': float(np.var(field_sizes)),
        'num_pointer_fields': int(np.random.randint(0, num_fields // 2 + 1)),
        'num_primitive_fields': int(num_fields - np.random.randint(0, num_fields // 2 + 1)),
        'avg_loop_nesting_depth': float(np.mean(loop_nesting)),
        'max_loop_nesting_depth': int(np.max(loop_nesting)),
        'total_access_sites': int(np.sum(num_accesses)),
        'avg_accesses_per_field': float(np.mean(num_accesses)),
        'max_accesses_field': int(np.max(num_accesses)),
        'access_variance': float(np.var(num_accesses)),
        'hot_cold_ratio': float(num_hot / max(num_cold, 1)),
        'cooccurrence_score': float(np.random.uniform(0, 1)),
        'has_pointer_arith': int(np.random.random() > 0.7),
        'estimated_hotness': float(np.mean(hotness_scores)),
        'in_loop_header': int(np.random.random() > 0.5),
        'trip_count_estimate': int(np.random.choice([100, 1000, 10000, 100000])),
        'alloc_in_loop': int(np.random.random() > 0.6),
        'dominance_depth': int(np.random.randint(0, 5)),
        'is_kernel_function': int(config.get('loops', 0) >= 2),
        'is_linear_algebra': int(config.get('type') == 'linear_algebra'),
        'is_stencil': int(config.get('type') == 'stencil'),
        'is_synthetic': int(config.get('type') == 'synthetic'),
    }
    
    return {
        'features': features,
        'label': int(is_profitable),
        'hotness_scores': hotness_scores.tolist(),
        'config': config
    }


def create_feature_dataframe(benchmarks: List[Dict]) -> pd.DataFrame:
    """Convert benchmark data to feature DataFrame."""
    rows = []
    for b in benchmarks:
        row = b['features'].copy()
        row['label'] = b['label']
        rows.append(row)
    
    return pd.DataFrame(rows)


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7, seed: int = 42) -> Tuple:
    """Split data into train and test sets by benchmark name."""
    set_seed(seed)
    
    unique_benchmarks = df['benchmark_name'].unique()
    n_train = int(len(unique_benchmarks) * train_ratio)
    
    # Shuffle and split
    np.random.shuffle(unique_benchmarks)
    train_benchmarks = unique_benchmarks[:n_train]
    test_benchmarks = unique_benchmarks[n_train:]
    
    train_mask = df['benchmark_name'].isin(train_benchmarks)
    test_mask = df['benchmark_name'].isin(test_benchmarks)
    
    return df[train_mask], df[test_mask], train_benchmarks, test_benchmarks


def main():
    paths = get_project_paths()
    
    print("=" * 60)
    print("Data Preparation: Generating Synthetic Benchmarks")
    print("=" * 60)
    
    # Generate benchmarks
    print("\nGenerating synthetic benchmarks...")
    data = generate_synthetic_benchmarks(seed=42)
    benchmarks = data['benchmarks']
    print(f"Generated {len(benchmarks)} structure instances from {len(set(b['features']['benchmark_name'] for b in benchmarks))} benchmarks")
    
    # Create DataFrame
    df = create_feature_dataframe(benchmarks)
    print(f"Dataset shape: {df.shape}")
    print(f"Positive labels (profitable): {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    
    # Split into train/test
    train_df, test_df, train_bench, test_bench = split_train_test(df, train_ratio=0.7, seed=42)
    print(f"\nTrain set: {len(train_df)} instances from {len(train_bench)} benchmarks")
    print(f"Test set: {len(test_df)} instances from {len(test_bench)} benchmarks")
    print(f"Train benchmarks: {', '.join(sorted(train_bench))}")
    print(f"Test benchmarks: {', '.join(sorted(test_bench))}")
    
    # Save processed data
    os.makedirs(f"{paths['data']}/processed", exist_ok=True)
    
    # Full dataset
    df.to_csv(f"{paths['data']}/processed/features.csv", index=False)
    
    # Train/test split
    train_df.to_csv(f"{paths['data']}/processed/train.csv", index=False)
    test_df.to_csv(f"{paths['data']}/processed/test.csv", index=False)
    
    # Metadata
    metadata = {
        'total_samples': len(df),
        'n_features': len([c for c in df.columns if c not in ['benchmark_name', 'label']]),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'train_benchmarks': sorted(train_bench.tolist()),
        'test_benchmarks': sorted(test_bench.tolist()),
        'positive_ratio': float(df['label'].mean()),
        'feature_columns': [c for c in df.columns if c not in ['benchmark_name', 'label']]
    }
    
    save_json(metadata, f"{paths['data']}/processed/metadata.json")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Saved to: {paths['data']}/processed/")
    print("=" * 60)
    
    return metadata


if __name__ == '__main__':
    main()
