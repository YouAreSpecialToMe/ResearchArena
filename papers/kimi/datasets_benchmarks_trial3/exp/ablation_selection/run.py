"""
Ablation: Wasserstein vs Bin-Matching Selection.

Tests the contribution of Wasserstein-optimal selection by comparing
against simple difficulty bin matching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics
from shared.optimal_transport import sinkhorn_selection, bin_matching_selection, compute_selection_quality


def run_ablation_selection(seeds=[42, 123, 999]):
    """Compare Wasserstein selection vs bin-matching."""
    print("=" * 60)
    print("Ablation: Wasserstein vs Bin-Matching Selection")
    print("=" * 60)
    
    # Load data
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    with open('data/pools/item_pool.jsonl', 'r') as f:
        items = [json.loads(line) for line in f]
    domains = np.array([{'math': 0, 'code': 1, 'science': 2}.get(i['domain'], 0) for i in items])
    
    n_models = 28
    n_timepoints = 5
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        
        wasserstein_results = []
        binmatch_results = []
        
        for t in range(n_timepoints):
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # Target distribution
            target_samples = true_abilities
            
            # Method 1: Wasserstein selection
            w_items, _ = sinkhorn_selection(
                difficulties, target_samples, n_select=500,
                epsilon=0.01, domain_labels=domains, domain_min_fraction=0.1
            )
            w_quality = compute_selection_quality(difficulties[w_items], target_samples)
            
            # Method 2: Bin-matching
            b_items = bin_matching_selection(
                difficulties, target_samples, n_select=500,
                n_bins=5, domain_labels=domains, domain_min_fraction=0.1
            )
            b_quality = compute_selection_quality(difficulties[b_items], target_samples)
            
            # Evaluate both methods
            for method_name, selected_items, results_list in [
                ('wasserstein', w_items, wasserstein_results),
                ('bin_matching', b_items, binmatch_results)
            ]:
                responses = all_responses[:, selected_items]
                
                irt_model = TwoPLModel(n_models, len(selected_items))
                irt_model.difficulties = difficulties[selected_items].copy()
                irt_model.discriminations = discriminations[selected_items].copy()
                
                estimated_abilities = irt_model.estimate_abilities_mle(responses)
                
                metrics = compute_all_metrics(
                    true_abilities, estimated_abilities,
                    irt_model=irt_model, selected_items=None,
                    ability_distribution=true_abilities
                )
                
                results_list.append({
                    'time': t * 3,
                    'metrics': metrics
                })
        
        results_per_seed.append({
            'seed': seed,
            'wasserstein': wasserstein_results,
            'bin_matching': binmatch_results
        })
    
    # Aggregate
    w_tau = []
    b_tau = []
    for r in results_per_seed:
        w_tau.append(np.mean([x['metrics']['kendall_tau'] for x in r['wasserstein']]))
        b_tau.append(np.mean([x['metrics']['kendall_tau'] for x in r['bin_matching']]))
    
    print(f"\nWasserstein: τ={np.mean(w_tau):.4f}±{np.std(w_tau):.4f}")
    print(f"Bin-matching: τ={np.mean(b_tau):.4f}±{np.std(b_tau):.4f}")
    
    # Statistical test
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(w_tau, b_tau)
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    final_results = {
        'experiment': 'ablation_selection',
        'per_seed': results_per_seed,
        'summary': {
            'wasserstein_mean_tau': float(np.mean(w_tau)),
            'wasserstein_std_tau': float(np.std(w_tau)),
            'bin_matching_mean_tau': float(np.mean(b_tau)),
            'bin_matching_std_tau': float(np.std(b_tau)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        }
    }
    
    os.makedirs('exp/ablation_selection', exist_ok=True)
    with open('exp/ablation_selection/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to exp/ablation_selection/results.json")
    return final_results


if __name__ == '__main__':
    run_ablation_selection()
