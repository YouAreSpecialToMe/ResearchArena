#!/usr/bin/env python3
"""
Ablation: Remove Metadata Network.
Test if metadata conditioning helps by using learned family mean instead of 
metadata-predicted mean.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch

from exp.shared.data_loader import MMLUDataset
from exp.shared.models import HierarchicalPopulationModel
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation,
    compute_kendall_tau, compute_r2_score
)


def run_ablation_no_metadata(seed: int) -> dict:
    """Run ablation: zero-shot without metadata network."""
    print(f"\n--- Running ablation (no metadata) with seed {seed} ---")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load trained model
    pop_model = HierarchicalPopulationModel(
        n_dimensions=3,
        n_families=8,
        use_metadata_network=True  # Load with metadata network but don't use it
    )
    model_path = f"models/population_model_seed{seed}.npy"
    pop_model.load(model_path)
    
    # Zero-shot prediction WITHOUT using metadata network
    predictions = []
    true_values = []
    
    for model_name in test_models:
        metadata = dataset.models[model_name]
        
        # Get family index and use ONLY family mean (ignore metadata network)
        family_idx = pop_model.family_to_idx.get(metadata.family, 7)
        pred_mean = pop_model.family_means[family_idx]  # Use family mean directly
        
        # Convert to overall performance estimate
        pred_overall = np.mean(pred_mean)
        
        # True value
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        true_overall = np.mean(true_ability)
        
        predictions.append(pred_overall)
        true_values.append(true_overall)
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Compute metrics
    metrics = {
        'mae': compute_mae(predictions, true_values),
        'rmse': compute_rmse(predictions, true_values),
        'spearman': compute_spearman_correlation(predictions, true_values),
        'kendall': compute_kendall_tau(predictions, true_values),
        'r2': compute_r2_score(predictions, true_values),
        'items_used': 0
    }
    
    print(f"  Results: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}")
    return metrics, predictions, true_values


def main():
    print("=" * 60)
    print("Ablation: Remove Metadata Network")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    all_predictions = []
    
    start_time = time.time()
    
    for seed in seeds:
        metrics, predictions, true_values = run_ablation_no_metadata(seed)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
    
    runtime = (time.time() - start_time) / 60
    
    # Aggregate across seeds
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'r2', 'items_used']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    results = {
        'experiment': 'ablation_no_metadata',
        'description': 'PopBench-NoMeta: Using family mean instead of metadata-predicted mean',
        'metrics': aggregated,
        'config': {
            'seeds': seeds,
            'n_test_models': 20,
            'use_metadata_network': False
        },
        'predictions': {
            'true': true_values.tolist(),
            'predicted_by_seed': [p.tolist() for p in all_predictions]
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/ablation_no_metadata/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Ablation Results (mean ± std across seeds):")
    print(f"  MAE: {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Kendall: {aggregated['kendall']['mean']:.4f} ± {aggregated['kendall']['std']:.4f}")
    print(f"  R²: {aggregated['r2']['mean']:.4f} ± {aggregated['r2']['std']:.4f}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
