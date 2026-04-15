#!/usr/bin/env python3
"""
Ablation: No Population Prior (Flat Prior N(0,I)).

Tests if population prior initialization helps.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch

from exp.shared.data_loader import MMLUDataset
from exp.shared.models_v2 import HierarchicalPopulationModelV2, AdaptiveEvaluator
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation, compute_kendall_tau
)


def run_ablation(seed: int, max_items: int = 100, target_mae: float = 0.05) -> dict:
    """Run with flat prior (no population structure)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    pop_model = HierarchicalPopulationModelV2(n_dimensions=3, n_families=8)
    model_path = f"models/population_model_v2_seed{seed}.npy"
    pop_model.load(model_path)
    
    evaluator = AdaptiveEvaluator(pop_model, n_dimensions=3)
    
    results_per_model = []
    
    for model_name in test_models:
        metadata = dataset.models[model_name]
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        responses = dataset.responses[model_name]
        
        # KEY DIFFERENCE: Use flat prior
        result = evaluator.evaluate_model(
            model_name=model_name,
            metadata=metadata,
            true_ability=true_ability,
            responses=responses,
            max_items=max_items,
            target_mae=target_mae,
            use_population_prior=False,  # ABlation: flat prior
            use_population_eig=True,
            seed=seed
        )
        results_per_model.append(result)
    
    true_vals = np.array([r['true_ability'] for r in results_per_model])
    pred_vals = np.array([r['final_estimate'] for r in results_per_model])
    items_used = np.array([r['items_used'] for r in results_per_model])
    
    return {
        'mae': compute_mae(pred_vals, true_vals),
        'rmse': compute_rmse(pred_vals, true_vals),
        'spearman': compute_spearman_correlation(pred_vals, true_vals),
        'kendall': compute_kendall_tau(pred_vals, true_vals),
        'items_used_mean': float(np.mean(items_used)),
        'items_used_std': float(np.std(items_used))
    }


def main():
    print("=" * 60)
    print("Ablation: No Population Prior (Flat Prior N(0,I))")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n  Running seed {seed}...")
        metrics = run_ablation(seed)
        all_metrics.append(metrics)
        print(f"    MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}, Items={metrics['items_used_mean']:.1f}")
    
    runtime = (time.time() - start_time) / 60
    
    # Aggregate
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'items_used_mean', 'items_used_std']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    results = {
        'experiment': 'ablation_no_population_prior_v2',
        'description': 'Ablation: Using flat prior N(0,I) instead of learned population prior',
        'metrics': aggregated,
        'config': {'max_items': 100, 'target_mae': 0.05, 'seeds': seeds, 'n_test_models': 20},
        'runtime_minutes': runtime
    }
    
    with open('exp/ablation_no_population_prior/results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Ablation Results (Flat Prior):")
    print(f"  MAE:    {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Items:  {aggregated['items_used_mean']['mean']:.1f} ± {aggregated['items_used_mean']['std']:.1f}")
    print(f"  Runtime: {runtime:.2f} min")


if __name__ == "__main__":
    main()
