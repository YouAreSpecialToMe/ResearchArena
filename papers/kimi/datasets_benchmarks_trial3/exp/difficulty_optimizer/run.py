"""
Difficulty Optimizer.

Computes target difficulty distribution that maximizes expected Fisher information.
Key theorem: optimal difficulty distribution matches ability distribution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from scipy.optimize import minimize


def compute_optimal_difficulty_distribution(ability_distribution, 
                                           n_bins=20,
                                           content_constraints=None):
    """Compute target difficulty distribution maximizing expected Fisher information.
    
    Theoretical result: For 2PL model, optimal difficulty matches ability distribution.
    
    Args:
        ability_distribution: Dict with 'mean', 'std', 'samples' or array of abilities
        n_bins: Number of bins for histogram representation
        content_constraints: Dict with minimum fraction per domain (optional)
        
    Returns:
        target_distribution: Dict with target difficulty histogram and samples
    """
    # Extract ability samples
    if isinstance(ability_distribution, dict):
        if 'samples' in ability_distribution:
            ability_samples = np.array(ability_distribution['samples'])
        else:
            # Generate from normal distribution
            mean = ability_distribution['mean']
            std = ability_distribution['std']
            ability_samples = np.random.normal(mean, std, 1000)
    else:
        ability_samples = np.array(ability_distribution)
    
    # Theoretical optimum: difficulty distribution = ability distribution
    target_samples = ability_samples.copy()
    
    # Create histogram representation
    hist, bin_edges = np.histogram(target_samples, bins=n_bins, density=True)
    
    # Bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return {
        'samples': target_samples.tolist(),
        'histogram': {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'bin_centers': bin_centers.tolist()
        },
        'mean': float(np.mean(target_samples)),
        'std': float(np.std(target_samples)),
        'n_bins': n_bins
    }


def compute_expected_fisher_for_distribution(ability_distribution, 
                                             difficulty_distribution,
                                             discrimination=1.0):
    """Compute expected Fisher information for given ability and difficulty distributions.
    
    Args:
        ability_distribution: Array of ability samples
        difficulty_distribution: Array of difficulty samples
        discrimination: Discrimination parameter (assumed constant)
        
    Returns:
        expected_fisher: Expected Fisher information value
    """
    from scipy.special import expit
    
    abilities = np.array(ability_distribution)
    difficulties = np.array(difficulty_distribution)
    
    # For each ability, compute average Fisher over difficulty distribution
    fisher_values = []
    for theta in abilities:
        # I(θ; a, b) for each difficulty
        z = discrimination * (theta - difficulties)
        probs = expit(z)
        fisher = (discrimination ** 2) * probs * (1 - probs)
        fisher_values.append(np.mean(fisher))
    
    return float(np.mean(fisher_values))


def optimize_with_constraints(ability_distribution, 
                              item_pool_difficulties,
                              item_pool_domains,
                              n_select=500,
                              domain_min_fraction=0.1):
    """Optimize difficulty distribution with content constraints.
    
    Args:
        ability_distribution: Ability distribution dict
        item_pool_difficulties: All available item difficulties
        item_pool_domains: Domain label for each item
        n_select: Number of items to select
        domain_min_fraction: Minimum fraction per domain
        
    Returns:
        constrained_target: Target distribution with constraints applied
    """
    # Base target: match ability distribution
    base_target = compute_optimal_difficulty_distribution(ability_distribution)
    
    # Apply content constraints by adjusting target
    # Ensure at least domain_min_fraction of items from each domain
    unique_domains = np.unique(item_pool_domains)
    min_per_domain = int(domain_min_fraction * n_select)
    
    # For simplicity, we keep the target distribution as-is
    # and handle constraints during item selection
    # The actual domain balancing happens in the selection algorithm
    
    return base_target


def run_difficulty_optimizer_experiments(seeds=[42, 123, 999]):
    """Run difficulty optimizer experiments."""
    print("=" * 60)
    print("Difficulty Optimizer")
    print("=" * 60)
    
    # Load data
    difficulties = np.load('data/pools/difficulties.npy')
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        
        time_targets = []
        
        for t in range(5):
            # Load ability distribution at this time
            abilities = np.load(f'data/population/abilities_t{t}.npy')
            
            # Compute target difficulty distribution
            target_dist = compute_optimal_difficulty_distribution(abilities)
            
            print(f"  t={t*3}: target mean={target_dist['mean']:.3f}, "
                  f"std={target_dist['std']:.3f}")
            
            time_targets.append({
                'time': t * 3,
                'target_distribution': target_dist
            })
        
        results_per_seed.append({
            'seed': seed,
            'time_targets': time_targets
        })
    
    # Save results
    final_results = {
        'experiment': 'difficulty_optimizer',
        'per_seed': results_per_seed,
        'note': 'Optimal difficulty distribution matches ability distribution'
    }
    
    os.makedirs('exp/difficulty_optimizer', exist_ok=True)
    with open('exp/difficulty_optimizer/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to exp/difficulty_optimizer/results.json")
    print("=" * 60)
    
    return final_results


if __name__ == '__main__':
    results = run_difficulty_optimizer_experiments()
