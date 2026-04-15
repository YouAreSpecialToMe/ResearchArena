"""
Static Benchmark Baseline.

Uses a fixed set of 500 items for all time periods.
This measures natural benchmark saturation as model abilities improve.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
import time
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics


def run_static_baseline(n_items_select=500, seeds=[42, 123, 999]):
    """Run static baseline experiment.
    
    Args:
        n_items_select: Number of items to select (fixed for all time periods)
        seeds: Random seeds for reproducibility
    """
    print("=" * 60)
    print("Static Benchmark Baseline")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    n_items_total = len(difficulties)
    n_models = 28
    n_timepoints = 5
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Running with seed {seed} ---")
        np.random.seed(seed)
        
        # Select fixed items (static benchmark)
        selected_items = np.random.choice(n_items_total, n_items_select, replace=False)
        selected_difficulties = difficulties[selected_items]
        selected_discriminations = discriminations[selected_items]
        
        print(f"Selected {n_items_select} items (fixed for all time periods)")
        
        # Results for each time period
        time_results = []
        
        for t in range(n_timepoints):
            print(f"\n  Time period t={t*3} (simulated month {t*3})")
            
            # Load ground truth abilities and responses
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # Get responses for selected items only
            responses = all_responses[:, selected_items]
            
            # Fit IRT model to estimate abilities
            irt_model = TwoPLModel(n_models, n_items_select)
            irt_model.difficulties = selected_difficulties.copy()
            irt_model.discriminations = selected_discriminations.copy()
            
            # Estimate abilities given item parameters
            estimated_abilities = irt_model.estimate_abilities_mle(responses)
            
            # Compute metrics
            metrics = compute_all_metrics(
                true_abilities, 
                estimated_abilities,
                irt_model=irt_model,
                selected_items=None,  # Already using subset
                ability_distribution=true_abilities
            )
            
            print(f"    Kendall's τ: {metrics['kendall_tau']:.4f}")
            print(f"    Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
            print(f"    Expected Fisher info: {metrics.get('expected_fisher_info', 0):.2f}")
            print(f"    Mean true ability: {true_abilities.mean():.2f}")
            
            time_results.append({
                'time': t * 3,
                'metrics': metrics,
                'true_abilities': true_abilities.tolist(),
                'estimated_abilities': estimated_abilities.tolist()
            })
        
        # Compute stability over time
        rankings = [np.array(r['estimated_abilities']) for r in time_results]
        from shared.metrics import ranking_stability
        stability = ranking_stability(rankings)
        
        # Compute ranking accuracy trend
        ranking_accuracies = [r['metrics']['pairwise_accuracy'] for r in time_results]
        
        results_per_seed.append({
            'seed': seed,
            'time_results': time_results,
            'stability': stability,
            'ranking_accuracy_trend': ranking_accuracies,
            'selected_items': selected_items.tolist()
        })
    
    # Aggregate across seeds
    print("\n" + "=" * 60)
    print("Aggregation across seeds")
    print("=" * 60)
    
    # Mean and std for key metrics at each time point
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
        
        print(f"\nt={t_idx*3}: τ={np.mean(tau_values):.4f}±{np.std(tau_values):.4f}, "
              f"acc={np.mean(acc_values):.4f}±{np.std(acc_values):.4f}")
    
    # Stability across seeds
    stability_values = [r['stability']['stability'] for r in results_per_seed]
    
    # Final results
    final_results = {
        'experiment': 'baseline_static',
        'config': {
            'n_items_select': n_items_select,
            'n_models': n_models,
            'n_timepoints': n_timepoints,
            'seeds': seeds
        },
        'per_seed': results_per_seed,
        'aggregated': aggregated,
        'summary': {
            'mean_stability': float(np.mean(stability_values)),
            'std_stability': float(np.std(stability_values)),
            'ranking_accuracy_degradation': float(
                aggregated[0]['pairwise_accuracy']['mean'] - aggregated[-1]['pairwise_accuracy']['mean']
            ),
            'final_kendall_tau': aggregated[-1]['kendall_tau']
        }
    }
    
    # Save results
    os.makedirs('exp/baseline_static', exist_ok=True)
    with open('exp/baseline_static/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to exp/baseline_static/results.json")
    print("=" * 60)
    
    return final_results


if __name__ == '__main__':
    start_time = time.time()
    results = run_static_baseline(n_items_select=500, seeds=[42, 123, 999])
    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed:.1f} seconds")
