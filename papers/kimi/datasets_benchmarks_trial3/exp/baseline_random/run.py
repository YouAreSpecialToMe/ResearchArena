#!/usr/bin/env python3
"""
Baseline: Random Item Selection with 2PL IRT estimation.
FIXED: Proper seed propagation throughout.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch
from tqdm import tqdm

from exp.shared.data_loader import MMLUDataset
from exp.shared.models import TwoPLIRT
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation,
    compute_kendall_tau
)


def run_baseline_random(seed: int, max_items: int = 200) -> dict:
    """Run random baseline with a specific random seed."""
    # FIXED: Set all random seeds properly
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Train 2PL IRT on training models
    train_models = split['train_models']
    train_responses = np.array([dataset.responses[m] for m in train_models])
    
    irt = TwoPLIRT()
    irt.fit(train_responses, seed=seed)
    
    # Run adaptive evaluation on test models
    results_per_model = []
    
    for model_name in tqdm(test_models, desc=f"Random baseline (seed={seed})"):
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        true_overall = np.mean(true_ability)
        
        responses = dataset.responses[model_name]
        n_items_total = len(responses)
        
        # Random item selection - uses the seeded np.random
        item_order = np.random.permutation(n_items_total)
        selected_items = []
        estimated_abilities = []
        
        for i in range(min(max_items, n_items_total)):
            item_idx = item_order[i]
            selected_items.append(item_idx)
            
            # Estimate ability using MAP
            est_ability = irt.estimate_ability_map(
                responses,
                np.array(selected_items),
                prior_mean=0.0,
                prior_std=1.0,
                seed=seed
            )
            estimated_abilities.append(est_ability)
        
        results_per_model.append({
            'model': model_name,
            'final_mae': abs(estimated_abilities[-1] - true_overall) if estimated_abilities else 1.0,
            'final_ability_estimate': estimated_abilities[-1] if estimated_abilities else 0.0,
            'true_ability': true_overall
        })
    
    # Compute aggregate metrics
    true_abilities = np.array([r['true_ability'] for r in results_per_model])
    est_abilities = np.array([r['final_ability_estimate'] for r in results_per_model])
    
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
    print("Baseline: Random Item Selection")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    start_time = time.time()
    
    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        metrics, results = run_baseline_random(seed)
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
        'experiment': 'baseline_random',
        'description': 'Random item selection with 2PL IRT estimation',
        'metrics': aggregated,
        'config': {
            'max_items': 200,
            'seeds': seeds,
            'n_test_models': 20
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/baseline_random/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Final Results (mean ± std across seeds):")
    print(f"  MAE: {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Kendall: {aggregated['kendall']['mean']:.4f} ± {aggregated['kendall']['std']:.4f}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
