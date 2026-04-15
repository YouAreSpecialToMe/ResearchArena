"""
Difficulty-Stratified Random Baseline.

Samples uniformly across difficulty bins at each time period.
Tests whether simple difficulty balancing provides benefits.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
import time
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics


def run_stratified_baseline(n_items_select=500, n_bins=5, seeds=[42, 123, 999]):
    """Run difficulty-stratified random baseline.
    
    Args:
        n_items_select: Number of items to select per time period
        n_bins: Number of difficulty bins
        seeds: Random seeds for reproducibility
    """
    print("=" * 60)
    print("Difficulty-Stratified Random Baseline")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    n_items_total = len(difficulties)
    n_models = 28
    n_timepoints = 5
    
    # Define difficulty bins
    bin_edges = np.linspace(difficulties.min(), difficulties.max(), n_bins + 1)
    print(f"Difficulty bins: {bin_edges}")
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        np.random.seed(seed)
        
        # Results for each time period
        time_results = []
        selected_items_history = []
        
        for t in range(n_timepoints):
            print(f"\n  Time period t={t*3}")
            
            # Load ground truth abilities and responses
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # Stratified sampling: sample uniformly from each bin
            items_per_bin = n_items_select // n_bins
            selected_items = []
            
            for bin_idx in range(n_bins):
                bin_start, bin_end = bin_edges[bin_idx], bin_edges[bin_idx + 1]
                bin_mask = (difficulties >= bin_start) & (difficulties < bin_end)
                bin_items = np.where(bin_mask)[0]
                
                if len(bin_items) > 0:
                    n_sample = min(items_per_bin, len(bin_items))
                    sampled = np.random.choice(bin_items, n_sample, replace=False)
                    selected_items.extend(sampled)
            
            # Fill remaining if needed
            while len(selected_items) < n_items_select:
                remaining = [i for i in range(n_items_total) if i not in selected_items]
                if not remaining:
                    break
                selected_items.append(np.random.choice(remaining))
            
            selected_items = np.array(selected_items[:n_items_select])
            selected_items_history.append(selected_items.tolist())
            
            selected_difficulties = difficulties[selected_items]
            selected_discriminations = discriminations[selected_items]
            
            # Get responses for selected items
            responses = all_responses[:, selected_items]
            
            # Fit IRT and estimate abilities
            irt_model = TwoPLModel(n_models, n_items_select)
            irt_model.difficulties = selected_difficulties.copy()
            irt_model.discriminations = selected_discriminations.copy()
            
            estimated_abilities = irt_model.estimate_abilities_mle(responses)
            
            # Compute metrics
            metrics = compute_all_metrics(
                true_abilities, 
                estimated_abilities,
                irt_model=irt_model,
                selected_items=None,
                ability_distribution=true_abilities
            )
            
            print(f"    Kendall's τ: {metrics['kendall_tau']:.4f}")
            print(f"    Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
            print(f"    Expected Fisher info: {metrics.get('expected_fisher_info', 0):.2f}")
            
            time_results.append({
                'time': t * 3,
                'metrics': metrics,
                'true_abilities': true_abilities.tolist(),
                'estimated_abilities': estimated_abilities.tolist()
            })
        
        # Compute stability
        rankings = [np.array(r['estimated_abilities']) for r in time_results]
        from shared.metrics import ranking_stability
        stability = ranking_stability(rankings)
        
        results_per_seed.append({
            'seed': seed,
            'time_results': time_results,
            'stability': stability,
            'selected_items_history': selected_items_history
        })
    
    # Aggregate across seeds
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
    
    stability_values = [r['stability']['stability'] for r in results_per_seed]
    
    final_results = {
        'experiment': 'baseline_stratified',
        'config': {
            'n_items_select': n_items_select,
            'n_bins': n_bins,
            'n_models': n_models,
            'n_timepoints': n_timepoints,
            'seeds': seeds
        },
        'per_seed': results_per_seed,
        'aggregated': aggregated,
        'summary': {
            'mean_stability': float(np.mean(stability_values)),
            'std_stability': float(np.std(stability_values)),
            'ranking_accuracy_trend': float(aggregated[0]['pairwise_accuracy']['mean'] - aggregated[-1]['pairwise_accuracy']['mean']),
            'final_kendall_tau': aggregated[-1]['kendall_tau']
        }
    }
    
    os.makedirs('exp/baseline_stratified', exist_ok=True)
    with open('exp/baseline_stratified/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to exp/baseline_stratified/results.json")
    print("=" * 60)
    
    return final_results


if __name__ == '__main__':
    start_time = time.time()
    results = run_stratified_baseline(n_items_select=500, n_bins=5, seeds=[42, 123, 999])
    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed:.1f} seconds")
