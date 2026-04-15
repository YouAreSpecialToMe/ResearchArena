#!/usr/bin/env python3
"""
Phase 2b: Population-Aware Adaptive Evaluation (FIXED V2).

Key fixes:
- Proper EIG computation with Monte Carlo
- Correct posterior updates
- Verify actual computation happens
- Track credible interval coverage
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


def run_adaptive(seed: int, max_items: int = 100, target_mae: float = 0.05) -> dict:
    """Run adaptive evaluation with fixed model."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n  Loading data and model (seed={seed})...")
    
    # Load data
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Load trained model
    pop_model = HierarchicalPopulationModelV2(n_dimensions=3, n_families=8)
    model_path = f"models/population_model_v2_seed{seed}.npy"
    pop_model.load(model_path)
    
    # Create evaluator
    evaluator = AdaptiveEvaluator(pop_model, n_dimensions=3)
    
    print(f"  Evaluating {len(test_models)} models...")
    
    results_per_model = []
    total_compute_start = time.time()
    
    for i, model_name in enumerate(test_models):
        metadata = dataset.models[model_name]
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        responses = dataset.responses[model_name]
        
        result = evaluator.evaluate_model(
            model_name=model_name,
            metadata=metadata,
            true_ability=true_ability,
            responses=responses,
            max_items=max_items,
            target_mae=target_mae,
            use_population_prior=True,
            use_population_eig=True,
            seed=seed
        )
        results_per_model.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"    ...{i+1}/{len(test_models)} models evaluated")
    
    compute_time = time.time() - total_compute_start
    
    # Aggregate metrics
    true_vals = np.array([r['true_ability'] for r in results_per_model])
    pred_vals = np.array([r['final_estimate'] for r in results_per_model])
    items_used = np.array([r['items_used'] for r in results_per_model])
    final_stds = np.array([r['final_std'] for r in results_per_model])
    
    # Check credible interval coverage (90% CI)
    coverage_count = 0
    for r in results_per_model:
        true_val = r['true_ability']
        pred_val = r['final_estimate']
        pred_std = r['final_std']
        # 90% CI: mean ± 1.645 * std
        ci_lower = pred_val - 1.645 * pred_std
        ci_upper = pred_val + 1.645 * pred_std
        if ci_lower <= true_val <= ci_upper:
            coverage_count += 1
    
    coverage_90 = coverage_count / len(results_per_model)
    
    metrics = {
        'mae': compute_mae(pred_vals, true_vals),
        'rmse': compute_rmse(pred_vals, true_vals),
        'spearman': compute_spearman_correlation(pred_vals, true_vals),
        'kendall': compute_kendall_tau(pred_vals, true_vals),
        'items_used_mean': float(np.mean(items_used)),
        'items_used_std': float(np.std(items_used)),
        'items_used_median': float(np.median(items_used)),
        'coverage_90': coverage_90,
        'compute_time_seconds': compute_time,
        'per_model_results': results_per_model
    }
    
    print(f"    Seed {seed} complete: MAE={metrics['mae']:.4f}, "
          f"Spearman={metrics['spearman']:.4f}, Items={metrics['items_used_mean']:.1f}, "
          f"Coverage={coverage_90:.2%}")
    
    return metrics


def main():
    print("=" * 60)
    print("Phase 2b: Population-Aware Adaptive Evaluation (V2 - FIXED)")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    total_start = time.time()
    
    for seed in seeds:
        metrics = run_adaptive(seed, max_items=100, target_mae=0.05)
        all_metrics.append(metrics)
    
    total_time = (time.time() - total_start) / 60
    
    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("Aggregating results across seeds...")
    print(f"{'='*60}")
    
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'items_used_mean', 'items_used_std', 'coverage_90']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    # Per-model aggregation for detailed analysis
    n_models = len(all_metrics[0]['per_model_results'])
    model_names = [r['model'] for r in all_metrics[0]['per_model_results']]
    
    per_model_summary = []
    for i in range(n_models):
        name = model_names[i]
        maes = [m['per_model_results'][i]['final_mae'] for m in all_metrics]
        items = [m['per_model_results'][i]['items_used'] for m in all_metrics]
        true_ability = all_metrics[0]['per_model_results'][i]['true_ability']
        estimates = [m['per_model_results'][i]['final_estimate'] for m in all_metrics]
        
        per_model_summary.append({
            'model': name,
            'true_ability': true_ability,
            'mean_estimate': float(np.mean(estimates)),
            'estimate_std': float(np.std(estimates)),
            'mean_mae': float(np.mean(maes)),
            'mean_items': float(np.mean(items))
        })
    
    # Check success criteria
    mae_target_met = aggregated['mae']['mean'] < 0.05
    items_target_met = aggregated['items_used_mean']['mean'] < 1400  # 10% of ~14K
    coverage_target_met = 0.85 <= aggregated['coverage_90']['mean'] <= 0.95
    
    results = {
        'experiment': 'popbench_adaptive_v2',
        'description': 'Fixed population-aware adaptive evaluation with proper EIG',
        'metrics': aggregated,
        'per_model_summary': per_model_summary,
        'config': {
            'max_items': 100,
            'target_mae': 0.05,
            'seeds': seeds,
            'n_test_models': 20
        },
        'success_criteria': {
            'mae_below_0.05': {
                'target': True,
                'achieved': mae_target_met,
                'value': aggregated['mae']['mean']
            },
            'items_below_10_percent': {
                'target': True,
                'achieved': items_target_met,
                'value': aggregated['items_used_mean']['mean']
            },
            'coverage_90_in_range': {
                'target': True,
                'achieved': coverage_target_met,
                'value': aggregated['coverage_90']['mean']
            }
        },
        'runtime_minutes': total_time
    }
    
    with open('exp/popbench_adaptive/results_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Final Results (mean ± std across seeds):")
    print(f"{'='*60}")
    print(f"  MAE:        {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  RMSE:       {aggregated['rmse']['mean']:.4f} ± {aggregated['rmse']['std']:.4f}")
    print(f"  Spearman:   {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Kendall:    {aggregated['kendall']['mean']:.4f} ± {aggregated['kendall']['std']:.4f}")
    print(f"  Items used: {aggregated['items_used_mean']['mean']:.1f} ± {aggregated['items_used_std']['std']:.1f}")
    print(f"  90% CI Coverage: {aggregated['coverage_90']['mean']:.2%}")
    print(f"\n  Success Criteria:")
    print(f"    MAE < 0.05:      {'✓' if mae_target_met else '✗'} ({aggregated['mae']['mean']:.4f})")
    print(f"    Items < 10%:     {'✓' if items_target_met else '✗'} ({aggregated['items_used_mean']['mean']:.1f})")
    print(f"    Coverage 85-95%: {'✓' if coverage_target_met else '✗'} ({aggregated['coverage_90']['mean']:.2%})")
    print(f"\n  Runtime: {total_time:.2f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
