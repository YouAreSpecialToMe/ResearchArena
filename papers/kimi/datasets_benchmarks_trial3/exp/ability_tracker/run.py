"""
Population Ability Tracker.

Uses Bayesian IRT inference to estimate ability distribution from response patterns.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from shared.irt_models import TwoPLModel


def estimate_ability_distribution(responses, difficulties, discriminations, 
                                  method='mle', n_bootstrap=100):
    """Estimate ability distribution from response matrix.
    
    Args:
        responses: (n_models, n_items) binary response matrix
        difficulties: (n_items,) item difficulties
        discriminations: (n_items,) item discriminations
        method: 'mle' or 'bayesian'
        n_bootstrap: Number of bootstrap samples for uncertainty estimation
        
    Returns:
        distribution: Dict with mean, std, samples, histogram
    """
    n_models, n_items = responses.shape
    
    # Fit IRT model
    irt_model = TwoPLModel(n_models, n_items)
    irt_model.difficulties = difficulties.copy()
    irt_model.discriminations = discriminations.copy()
    
    # Estimate abilities
    estimated_abilities = irt_model.estimate_abilities_mle(responses)
    
    # Compute distribution statistics
    mean_ability = np.mean(estimated_abilities)
    std_ability = np.std(estimated_abilities)
    
    # Create histogram representation
    hist, bin_edges = np.histogram(estimated_abilities, bins=20, density=True)
    
    # Bootstrap for uncertainty (optional)
    bootstrap_means = []
    if n_bootstrap > 0:
        for _ in range(n_bootstrap):
            idx = np.random.choice(n_models, n_models, replace=True)
            bootstrap_means.append(np.mean(estimated_abilities[idx]))
        bootstrap_std = np.std(bootstrap_means)
    else:
        bootstrap_std = 0.0
    
    return {
        'mean': float(mean_ability),
        'std': float(std_ability),
        'samples': estimated_abilities.tolist(),
        'histogram': {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        },
        'bootstrap_std': float(bootstrap_std),
        'n_models': n_models
    }


def track_distribution_over_time(item_selections, difficulties, discriminations, seeds=[42, 123, 999]):
    """Track ability distribution evolution over time.
    
    Args:
        item_selections: Dict mapping time -> selected item indices
        difficulties: All item difficulties
        discriminations: All item discriminations
        seeds: Random seeds
        
    Returns:
        tracking_results: Dict with distribution estimates per time period
    """
    print("=" * 60)
    print("Population Ability Tracker")
    print("=" * 60)
    
    n_timepoints = 5
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        
        time_distributions = []
        
        for t in range(n_timepoints):
            print(f"  Time t={t*3}")
            
            # Load responses
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # Get items for this time period
            if isinstance(item_selections, dict) and t in item_selections:
                selected_items = item_selections[t]
            elif isinstance(item_selections, dict) and str(t) in item_selections:
                selected_items = item_selections[str(t)]
            else:
                # Default: use all items
                selected_items = list(range(len(difficulties)))
            
            selected_items = np.array(selected_items)
            responses = all_responses[:, selected_items]
            
            # Estimate distribution
            dist = estimate_ability_distribution(
                responses,
                difficulties[selected_items],
                discriminations[selected_items],
                method='mle',
                n_bootstrap=50
            )
            
            print(f"    Mean ability: {dist['mean']:.3f} ± {dist['std']:.3f}")
            
            time_distributions.append({
                'time': t * 3,
                'distribution': dist
            })
        
        results_per_seed.append({
            'seed': seed,
            'time_distributions': time_distributions
        })
    
    # Aggregate
    aggregated = []
    for t_idx in range(n_timepoints):
        means = [r['time_distributions'][t_idx]['distribution']['mean'] 
                for r in results_per_seed]
        stds = [r['time_distributions'][t_idx]['distribution']['std'] 
               for r in results_per_seed]
        
        aggregated.append({
            'time': t_idx * 3,
            'mean_ability': {'mean': float(np.mean(means)), 'std': float(np.std(means))},
            'std_ability': {'mean': float(np.mean(stds)), 'std': float(np.std(stds))}
        })
    
    final_results = {
        'experiment': 'ability_tracker',
        'per_seed': results_per_seed,
        'aggregated': aggregated
    }
    
    os.makedirs('exp/ability_tracker', exist_ok=True)
    with open('exp/ability_tracker/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to exp/ability_tracker/results.json")
    print("=" * 60)
    
    return final_results


if __name__ == '__main__':
    # Load item parameters
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    # Test with fixed item selection (static benchmark)
    np.random.seed(42)
    fixed_items = np.random.choice(len(difficulties), 500, replace=False)
    item_selections = {t: fixed_items for t in range(5)}
    
    results = track_distribution_over_time(item_selections, difficulties, discriminations)
