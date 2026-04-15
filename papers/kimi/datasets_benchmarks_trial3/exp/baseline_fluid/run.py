"""
Fluid Benchmarking-Style Baseline.

Individual adaptive testing using Fisher information maximization for each model.
Each model gets a personalized item sequence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
import time
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics, kendalls_tau


def select_next_item_fisher(ability_estimate, available_items, difficulties, discriminations):
    """Select next item maximizing Fisher information at current ability estimate.
    
    Args:
        ability_estimate: Current ability estimate
        available_items: Array of available item indices
        difficulties: All item difficulties
        discriminations: All item discriminations
        
    Returns:
        selected_idx: Index of selected item
    """
    available_difficulties = difficulties[available_items]
    available_discriminations = discriminations[available_items]
    
    # Compute Fisher information at ability_estimate
    theta = ability_estimate
    a = available_discriminations
    b = available_difficulties
    
    # I(θ; a, b) = a² * σ(a(θ-b)) * (1 - σ(a(θ-b)))
    from scipy.special import expit
    probs = expit(a * (theta - b))
    fisher_info = (a ** 2) * probs * (1 - probs)
    
    # Select item with maximum Fisher information
    best_idx = np.argmax(fisher_info)
    return available_items[best_idx]


def run_fluid_baseline(n_items_per_model=100, seeds=[42, 123, 999]):
    """Run Fluid Benchmarking-style adaptive testing baseline.
    
    Args:
        n_items_per_model: Number of items per model (adaptive selection)
        seeds: Random seeds
    """
    print("=" * 60)
    print("Fluid Benchmarking-Style Baseline")
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
        
        time_results = []
        
        for t in range(n_timepoints):
            print(f"\n  Time period t={t*3}")
            
            # Load ground truth
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # For each model, run CAT (Computerized Adaptive Testing)
            estimated_abilities = np.zeros(n_models)
            all_selected_items = []  # Track which items each model sees
            
            for m in range(n_models):
                # Start with neutral ability estimate
                theta_hat = 0.0
                available_items = list(range(n_items_total))
                selected_items = []
                responses = []
                
                # Initial random item
                init_item = np.random.choice(available_items)
                selected_items.append(init_item)
                available_items.remove(init_item)
                responses.append(all_responses[m, init_item])
                
                # Adaptive selection
                for _ in range(n_items_per_model - 1):
                    # Update ability estimate
                    if len(selected_items) >= 1:
                        # Simple MLE update
                        from scipy.optimize import minimize
                        from scipy.special import expit
                        
                        def neg_ll(theta):
                            b = difficulties[selected_items]
                            a = discriminations[selected_items]
                            probs = expit(a * (theta - b))
                            probs = np.clip(probs, 1e-10, 1 - 1e-10)
                            y = np.array(responses)
                            return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                        
                        result = minimize(neg_ll, theta_hat, method='BFGS')
                        theta_hat = result.x[0]
                    
                    # Select next item maximizing Fisher information
                    next_item = select_next_item_fisher(
                        theta_hat, 
                        np.array(available_items),
                        difficulties,
                        discriminations
                    )
                    
                    selected_items.append(next_item)
                    available_items.remove(next_item)
                    responses.append(all_responses[m, next_item])
                
                # Final ability estimate
                from scipy.optimize import minimize
                from scipy.special import expit
                
                def neg_ll_final(theta):
                    b = difficulties[selected_items]
                    a = discriminations[selected_items]
                    probs = expit(a * (theta - b))
                    probs = np.clip(probs, 1e-10, 1 - 1e-10)
                    y = np.array(responses)
                    return -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                
                result = minimize(neg_ll_final, theta_hat, method='BFGS')
                estimated_abilities[m] = result.x[0]
                all_selected_items.append(selected_items)
            
            # Compute metrics (note: items differ per model, so comparability is reduced)
            metrics = compute_all_metrics(
                true_abilities,
                estimated_abilities,
                irt_model=None,  # Can't compute Fisher easily since items differ per model
                selected_items=None,
                ability_distribution=true_abilities
            )
            
            print(f"    Kendall's τ: {metrics['kendall_tau']:.4f}")
            print(f"    Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
            
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
            'stability': stability
        })
    
    # Aggregate
    aggregated = []
    for t_idx in range(n_timepoints):
        tau_values = [r['time_results'][t_idx]['metrics']['kendall_tau'] 
                     for r in results_per_seed]
        acc_values = [r['time_results'][t_idx]['metrics']['pairwise_accuracy'] 
                     for r in results_per_seed]
        
        aggregated.append({
            'time': t_idx * 3,
            'kendall_tau': {'mean': float(np.mean(tau_values)), 'std': float(np.std(tau_values))},
            'pairwise_accuracy': {'mean': float(np.mean(acc_values)), 'std': float(np.std(acc_values))}
        })
    
    stability_values = [r['stability']['stability'] for r in results_per_seed]
    
    final_results = {
        'experiment': 'baseline_fluid',
        'config': {
            'n_items_per_model': n_items_per_model,
            'n_models': n_models,
            'n_timepoints': n_timepoints,
            'seeds': seeds
        },
        'per_seed': results_per_seed,
        'aggregated': aggregated,
        'summary': {
            'mean_stability': float(np.mean(stability_values)),
            'std_stability': float(np.std(stability_values)),
            'final_kendall_tau': aggregated[-1]['kendall_tau']
        }
    }
    
    os.makedirs('exp/baseline_fluid', exist_ok=True)
    with open('exp/baseline_fluid/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to exp/baseline_fluid/results.json")
    print("=" * 60)
    
    return final_results


if __name__ == '__main__':
    start_time = time.time()
    results = run_fluid_baseline(n_items_per_model=100, seeds=[42, 123, 999])
    elapsed = time.time() - start_time
    print(f"\nRuntime: {elapsed:.1f} seconds")
