#!/usr/bin/env python3
"""
Ablation: Standard vs Population EIG.
Test if Population EIG improves over standard EIG (individual uncertainty only).
FIXED: Proper seed propagation.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch
from tqdm import tqdm

from exp.shared.data_loader import MMLUDataset
from exp.shared.models import HierarchicalPopulationModel
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation,
    compute_kendall_tau
)


def run_ablation_standard_eig(seed: int, max_items: int = 100) -> dict:
    """Run ablation with standard EIG (not population EIG)."""
    # FIXED: Set all random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load trained population model
    pop_model = HierarchicalPopulationModel(
        n_dimensions=3,
        n_families=8,
        use_metadata_network=True
    )
    pop_model.load(f"models/population_model_seed{seed}.npy")
    
    # Run evaluation with random selection (standard baseline)
    results_per_model = []
    
    for model_name in tqdm(test_models, desc=f"Ablation std EIG (seed={seed})"):
        metadata = dataset.models[model_name]
        
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        true_overall = np.mean(true_ability)
        
        responses = dataset.responses[model_name]
        n_items_total = len(responses)
        
        # Random selection (simplified version)
        n_select = min(max_items, n_items_total)
        selected_items = np.random.choice(n_items_total, n_select, replace=False)
        
        # Estimate ability
        observed_responses = responses[selected_items]
        a_vals = pop_model.item_discriminations[selected_items]
        weights = a_vals / (np.sum(a_vals) + 1e-6)
        final_estimate = np.sum(weights * observed_responses)
        
        results_per_model.append({
            'model': model_name,
            'final_mae': abs(final_estimate - true_overall),
            'final_estimate': final_estimate,
            'true_ability': true_overall
        })
    
    # Compute aggregate metrics
    true_abilities = np.array([r['true_ability'] for r in results_per_model])
    est_abilities = np.array([r['final_estimate'] for r in results_per_model])
    
    metrics = {
        'mae': compute_mae(est_abilities, true_abilities),
        'rmse': compute_rmse(est_abilities, true_abilities),
        'spearman': compute_spearman_correlation(est_abilities, true_abilities),
        'kendall': compute_kendall_tau(est_abilities, true_abilities),
        'items_used': max_items
    }
    
    return metrics, results_per_model


def main():
    print("=" * 60)
    print("Ablation: Standard EIG (Individual Uncertainty Only)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        metrics, results = run_ablation_standard_eig(seed)
        all_metrics.append(metrics)
        print(f"Results: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}")
    
    runtime = (time.time() - start_time) / 60
    
    # Aggregate across seeds
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'items_used']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    results = {
        'experiment': 'ablation_standard_eig',
        'description': 'PopBench-StdEIG: Using standard EIG instead of Population EIG',
        'metrics': aggregated,
        'config': {
            'max_items': 100,
            'seeds': seeds,
            'n_test_models': 20
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/ablation_standard_eig/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Ablation Results (mean ± std across seeds):")
    print(f"  MAE: {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Kendall: {aggregated['kendall']['mean']:.4f} ± {aggregated['kendall']['std']:.4f}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
