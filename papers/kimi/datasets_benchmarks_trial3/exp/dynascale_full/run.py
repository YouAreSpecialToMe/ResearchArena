"""
Full DynaScale 12-Month Simulation.

Complete closed-loop system with all three components:
1. Population Ability Tracker
2. Difficulty Optimizer
3. Wasserstein-Optimal Item Selection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
import time
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics, ranking_stability
from shared.optimal_transport import sinkhorn_selection, compute_selection_quality


def run_dynascale_simulation(n_items_select=500, seeds=[42, 123, 999], 
                             use_wasserstein=True):
    """Run full DynaScale simulation.
    
    Args:
        n_items_select: Number of items to select per benchmark version
        seeds: Random seeds
        use_wasserstein: If True, use Wasserstein selection; else use simple matching
    """
    print("=" * 70)
    print("DynaScale: Full 12-Month Simulation")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    # Load item domains
    with open('data/pools/item_pool.jsonl', 'r') as f:
        items = [json.loads(line) for line in f]
    domains = np.array([{'math': 0, 'code': 1, 'science': 2}.get(i['domain'], 0) for i in items])
    
    n_items_total = len(difficulties)
    n_models = 28
    n_timepoints = 5
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        np.random.seed(seed)
        
        time_results = []
        selected_items_history = []
        
        for t in range(n_timepoints):
            print(f"\n--- Time Period t={t*3} ---")
            
            # Load ground truth
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # Step 1: Population Ability Tracker
            # Estimate ability distribution from responses on current benchmark
            if t == 0:
                # Initial period: use uniform selection
                current_items = np.random.choice(n_items_total, n_items_select, replace=False)
            else:
                # Use items from previous iteration
                current_items = selected_items_history[-1]
            
            # Fit IRT to estimate current ability distribution
            irt_model = TwoPLModel(n_models, len(current_items))
            irt_model.difficulties = difficulties[current_items].copy()
            irt_model.discriminations = discriminations[current_items].copy()
            
            responses_current = all_responses[:, current_items]
            estimated_abilities_current = irt_model.estimate_abilities_mle(responses_current)
            
            # Step 2: Difficulty Optimizer
            # Compute target difficulty distribution = ability distribution
            target_distribution = estimated_abilities_current
            
            print(f"  Ability distribution: mean={target_distribution.mean():.3f}, "
                  f"std={target_distribution.std():.3f}")
            
            # Step 3: Dynamic Item Pool Manager
            # Select items matching target distribution
            if t < n_timepoints - 1:  # Don't need to select for last period
                if use_wasserstein:
                    selected_items, _ = sinkhorn_selection(
                        difficulties,
                        target_distribution,
                        n_select=n_items_select,
                        epsilon=0.01,
                        domain_labels=domains,
                        domain_min_fraction=0.1
                    )
                else:
                    # Simple matching: sample from pool with probabilities
                    # proportional to target distribution
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(target_distribution)
                    weights = kde(difficulties)
                    weights = weights / weights.sum()
                    selected_items = np.random.choice(
                        n_items_total, n_items_select, replace=False, p=weights
                    )
                
                selected_items_history.append(selected_items)
                
                # Compute selection quality
                quality = compute_selection_quality(
                    difficulties[selected_items],
                    target_distribution
                )
                print(f"  Selection quality: W_dist={quality['wasserstein_distance']:.4f}, "
                      f"mean_diff={quality['mean_difference']:.4f}")
            else:
                selected_items = current_items
                quality = None
            
            # Evaluate on selected items
            responses = all_responses[:, selected_items]
            
            # Re-estimate abilities for final metrics
            irt_model = TwoPLModel(n_models, len(selected_items))
            irt_model.difficulties = difficulties[selected_items].copy()
            irt_model.discriminations = discriminations[selected_items].copy()
            
            estimated_abilities = irt_model.estimate_abilities_mle(responses)
            
            # Compute metrics
            metrics = compute_all_metrics(
                true_abilities,
                estimated_abilities,
                irt_model=irt_model,
                selected_items=None,
                ability_distribution=true_abilities
            )
            
            print(f"  Metrics: τ={metrics['kendall_tau']:.4f}, "
                  f"acc={metrics['pairwise_accuracy']:.4f}, "
                  f"Fisher={metrics.get('expected_fisher_info', 0):.1f}")
            
            time_results.append({
                'time': t * 3,
                'metrics': metrics,
                'true_abilities': true_abilities.tolist(),
                'estimated_abilities': estimated_abilities.tolist(),
                'target_distribution': {
                    'mean': float(target_distribution.mean()),
                    'std': float(target_distribution.std())
                },
                'selection_quality': quality,
                'n_items': len(selected_items)
            })
        
        # Compute stability
        rankings = [np.array(r['estimated_abilities']) for r in time_results]
        stability = ranking_stability(rankings)
        
        # Compute ranking accuracy trend
        ranking_accuracies = [r['metrics']['pairwise_accuracy'] for r in time_results]
        
        results_per_seed.append({
            'seed': seed,
            'time_results': time_results,
            'stability': stability,
            'ranking_accuracy_trend': ranking_accuracies,
            'selected_items_history': [s.tolist() for s in selected_items_history]
        })
    
    # Aggregate across seeds
    print("\n" + "=" * 70)
    print("AGGREGATION ACROSS SEEDS")
    print("=" * 70)
    
    aggregated = []
    for t_idx in range(n_timepoints):
        tau_values = [r['time_results'][t_idx]['metrics']['kendall_tau'] 
                     for r in results_per_seed]
        acc_values = [r['time_results'][t_idx]['metrics']['pairwise_accuracy'] 
                     for r in results_per_seed]
        fisher_values = [r['time_results'][t_idx]['metrics'].get('expected_fisher_info', 0) 
                        for r in results_per_seed]
        
        aggregated.append({
            'time': t_idx * 3,
            'kendall_tau': {'mean': float(np.mean(tau_values)), 'std': float(np.std(tau_values))},
            'pairwise_accuracy': {'mean': float(np.mean(acc_values)), 'std': float(np.std(acc_values))},
            'expected_fisher_info': {'mean': float(np.mean(fisher_values)), 'std': float(np.std(fisher_values))}
        })
        
        print(f"t={t_idx*3}: τ={np.mean(tau_values):.4f}±{np.std(tau_values):.4f}, "
              f"acc={np.mean(acc_values):.4f}±{np.std(acc_values):.4f}, "
              f"Fisher={np.mean(fisher_values):.1f}")
    
    # Stability
    stability_values = [r['stability']['stability'] for r in results_per_seed]
    ranking_variance_values = [r['stability']['ranking_variance'] for r in results_per_seed]
    
    # Success criteria checks
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA VALIDATION")
    print("=" * 70)
    
    # Criterion 1: Ranking accuracy stability
    acc_means = [a['pairwise_accuracy']['mean'] for a in aggregated]
    acc_stability = np.std(acc_means)
    print(f"1. Ranking accuracy std over time: {acc_stability:.4f} "
          f"(target: <0.05)")
    
    # Criterion 2: Fisher information maintenance
    fisher_means = [a['expected_fisher_info']['mean'] for a in aggregated]
    fisher_final = fisher_means[-1]
    fisher_initial = fisher_means[0]
    fisher_retention = fisher_final / fisher_initial if fisher_initial > 0 else 0
    print(f"2. Fisher information retention: {fisher_retention:.2%} "
          f"(target: >90%)")
    
    # Criterion 3: Wasserstein distance
    w_distances = []
    for seed_result in results_per_seed:
        for t_result in seed_result['time_results']:
            if t_result['selection_quality']:
                w_distances.append(t_result['selection_quality']['wasserstein_distance'])
    avg_w_distance = np.mean(w_distances) if w_distances else float('inf')
    print(f"3. Average Wasserstein distance: {avg_w_distance:.4f} "
          f"(target: <0.5)")
    
    # Criterion 4: Top-k correlation
    tau_means = [a['kendall_tau']['mean'] for a in aggregated]
    min_tau = min(tau_means)
    print(f"4. Minimum Kendall's τ: {min_tau:.4f} (target: >0.95)")
    
    final_results = {
        'experiment': 'dynascale_full',
        'config': {
            'n_items_select': n_items_select,
            'n_models': n_models,
            'n_timepoints': n_timepoints,
            'seeds': seeds,
            'use_wasserstein': use_wasserstein
        },
        'per_seed': results_per_seed,
        'aggregated': aggregated,
        'summary': {
            'mean_stability': float(np.mean(stability_values)),
            'std_stability': float(np.std(stability_values)),
            'ranking_variance': float(np.mean(ranking_variance_values)),
            'ranking_accuracy_stability': float(acc_stability),
            'fisher_retention': float(fisher_retention),
            'avg_wasserstein_distance': float(avg_w_distance),
            'min_kendall_tau': float(min_tau),
            'success_criteria': {
                'ranking_stability_met': acc_stability < 0.05,
                'fisher_retention_met': fisher_retention > 0.90,
                'wasserstein_met': avg_w_distance < 0.5,
                'correlation_met': min_tau > 0.95
            }
        }
    }
    
    # Convert numpy types before saving
    def convert_types(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    final_results = convert_types(final_results)
    
    os.makedirs('exp/dynascale_full', exist_ok=True)
    with open('exp/dynascale_full/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Results saved to exp/dynascale_full/results.json")
    print("=" * 70)
    
    return final_results


if __name__ == '__main__':
    start_time = time.time()
    results = run_dynascale_simulation(n_items_select=500, seeds=[42, 123, 999])
    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed:.1f} seconds")
