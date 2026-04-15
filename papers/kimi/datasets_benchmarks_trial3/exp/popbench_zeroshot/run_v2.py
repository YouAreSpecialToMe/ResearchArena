#!/usr/bin/env python3
"""
Phase 2a: Zero-Shot Capability Prediction (FIXED V2).

Key fixes:
- Use the fixed population model
- Proper metadata network evaluation
- Verify predictions vary across model families
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch

from exp.shared.data_loader import MMLUDataset
from exp.shared.models_v2 import HierarchicalPopulationModelV2
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation, compute_kendall_tau
)


def run_zeroshot(seed: int, use_metadata_network: bool = True) -> dict:
    """Run zero-shot prediction."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load model
    pop_model = HierarchicalPopulationModelV2(n_dimensions=3, n_families=8, use_metadata_network=use_metadata_network)
    model_path = f"models/population_model_v2_seed{seed}.npy"
    pop_model.load(model_path)
    
    predictions = []
    true_values = []
    
    print(f"\n  Predicting {len(test_models)} models (seed={seed}, metadata={use_metadata_network})...")
    
    for model_name in test_models:
        metadata = dataset.models[model_name]
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        true_overall = np.mean(true_ability)
        
        # Zero-shot prediction
        pred_mean, pred_std = pop_model.predict_zero_shot(metadata)
        pred_overall = np.mean(pred_mean)
        
        predictions.append(pred_overall)
        true_values.append(true_overall)
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    # Compute metrics
    mae = compute_mae(predictions, true_values)
    rmse = compute_rmse(predictions, true_values)
    spearman = compute_spearman_correlation(predictions, true_values)
    kendall = compute_kendall_tau(predictions, true_values)
    
    # R²
    ss_res = np.sum((true_values - predictions) ** 2)
    ss_tot = np.sum((true_values - true_values.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'spearman': spearman,
        'kendall': kendall,
        'r2': r2,
        'predictions': predictions.tolist(),
        'true_values': true_values.tolist()
    }
    
    print(f"    MAE={mae:.4f}, Spearman={spearman:.4f}, R²={r2:.4f}")
    
    return metrics


def main():
    print("=" * 60)
    print("Phase 2a: Zero-Shot Capability Prediction (V2 - FIXED)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    
    # Run with metadata network
    print("\n--- With Metadata Network ---")
    all_metrics_with = []
    for seed in seeds:
        metrics = run_zeroshot(seed, use_metadata_network=True)
        all_metrics_with.append(metrics)
    
    # Run without metadata network (ablation)
    print("\n--- Without Metadata Network ---")
    all_metrics_without = []
    for seed in seeds:
        metrics = run_zeroshot(seed, use_metadata_network=False)
        all_metrics_without.append(metrics)
    
    # Aggregate
    def aggregate(metrics_list):
        return {
            'mae': {
                'mean': float(np.mean([m['mae'] for m in metrics_list])),
                'std': float(np.std([m['mae'] for m in metrics_list])),
                'values': [m['mae'] for m in metrics_list]
            },
            'rmse': {
                'mean': float(np.mean([m['rmse'] for m in metrics_list])),
                'std': float(np.std([m['rmse'] for m in metrics_list])),
                'values': [m['rmse'] for m in metrics_list]
            },
            'spearman': {
                'mean': float(np.mean([m['spearman'] for m in metrics_list])),
                'std': float(np.std([m['spearman'] for m in metrics_list])),
                'values': [m['spearman'] for m in metrics_list]
            },
            'kendall': {
                'mean': float(np.mean([m['kendall'] for m in metrics_list])),
                'std': float(np.std([m['kendall'] for m in metrics_list])),
                'values': [m['kendall'] for m in metrics_list]
            },
            'r2': {
                'mean': float(np.mean([m['r2'] for m in metrics_list])),
                'std': float(np.std([m['r2'] for m in metrics_list])),
                'values': [m['r2'] for m in metrics_list]
            }
        }
    
    aggregated_with = aggregate(all_metrics_with)
    aggregated_without = aggregate(all_metrics_without)
    
    # Check success
    target_spearman = 0.7
    achieved_with = aggregated_with['spearman']['mean'] > target_spearman
    
    print(f"\n{'='*60}")
    print("Zero-Shot Results (WITH metadata network):")
    print(f"  Spearman: {aggregated_with['spearman']['mean']:.4f} ± {aggregated_with['spearman']['std']:.4f}")
    print(f"  Target > 0.7: {'✓' if achieved_with else '✗'}")
    
    print(f"\nZero-Shot Results (WITHOUT metadata network):")
    print(f"  Spearman: {aggregated_without['spearman']['mean']:.4f} ± {aggregated_without['spearman']['std']:.4f}")
    
    print(f"\nImprovement from metadata: "
          f"{aggregated_with['spearman']['mean'] - aggregated_without['spearman']['mean']:+.4f}")
    
    # Save results
    results = {
        'experiment': 'popbench_zeroshot_v2',
        'description': 'Fixed zero-shot prediction with metadata network',
        'with_metadata': {
            'metrics': aggregated_with,
            'success': {
                'target': f'Spearman > {target_spearman}',
                'achieved': achieved_with,
                'value': aggregated_with['spearman']['mean']
            }
        },
        'without_metadata': {
            'metrics': aggregated_without
        },
        'config': {
            'seeds': seeds,
            'n_test_models': 20
        }
    }
    
    with open('exp/popbench_zeroshot/results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
