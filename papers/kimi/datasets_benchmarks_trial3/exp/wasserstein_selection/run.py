"""
Wasserstein-Optimal Item Selection.

Selects items matching target difficulty distribution using optimal transport.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from shared.optimal_transport import sinkhorn_selection, bin_matching_selection, compute_selection_quality


def run_wasserstein_selection_experiments(seeds=[42, 123, 999]):
    """Run Wasserstein selection experiments."""
    print("=" * 60)
    print("Wasserstein-Optimal Item Selection")
    print("=" * 60)
    
    # Load data
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    # Load item domains
    with open('data/pools/item_pool.jsonl', 'r') as f:
        items = [json.loads(line) for line in f]
    domains = np.array([{'math': 0, 'code': 1, 'science': 2}.get(i['domain'], 0) for i in items])
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        
        time_selections = []
        
        for t in range(5):
            # Load target ability distribution
            abilities = np.load(f'data/population/abilities_t{t}.npy')
            
            # Target: match ability distribution
            target_samples = abilities
            
            # Wasserstein-optimal selection
            selected_indices, transport_plan = sinkhorn_selection(
                difficulties,
                target_samples,
                n_select=500,
                epsilon=0.01,
                domain_labels=domains,
                domain_min_fraction=0.1
            )
            
            # Compute quality metrics
            quality = compute_selection_quality(
                difficulties[selected_indices],
                target_samples
            )
            
            print(f"  t={t*3}: Wasserstein distance={quality['wasserstein_distance']:.4f}, "
                  f"KS stat={quality['ks_statistic']:.4f}")
            
            time_selections.append({
                'time': t * 3,
                'selected_indices': selected_indices.tolist(),
                'quality': quality
            })
        
        results_per_seed.append({
            'seed': seed,
            'time_selections': time_selections
        })
    
    # Aggregate quality metrics
    aggregated_quality = []
    for t_idx in range(5):
        w_distances = [r['time_selections'][t_idx]['quality']['wasserstein_distance'] 
                      for r in results_per_seed]
        ks_stats = [r['time_selections'][t_idx]['quality']['ks_statistic'] 
                   for r in results_per_seed]
        
        aggregated_quality.append({
            'time': t_idx * 3,
            'wasserstein_distance': {'mean': float(np.mean(w_distances)), 
                                    'std': float(np.std(w_distances))},
            'ks_statistic': {'mean': float(np.mean(ks_stats)), 
                           'std': float(np.std(ks_stats))}
        })
    
    final_results = {
        'experiment': 'wasserstein_selection',
        'per_seed': results_per_seed,
        'aggregated_quality': aggregated_quality,
        'config': {
            'n_select': 500,
            'epsilon': 0.01,
            'domain_min_fraction': 0.1
        }
    }
    
    os.makedirs('exp/wasserstein_selection', exist_ok=True)
    with open('exp/wasserstein_selection/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to exp/wasserstein_selection/results.json")
    print("=" * 60)
    
    return final_results


if __name__ == '__main__':
    results = run_wasserstein_selection_experiments()
