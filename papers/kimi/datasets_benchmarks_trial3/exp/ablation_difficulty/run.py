"""
Ablation: Fisher-Optimal vs Uniform Difficulty.

Tests theoretical claim that matching difficulty to ability distribution
maximizes discriminative power.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics
from shared.optimal_transport import sinkhorn_selection


def run_ablation_difficulty(seeds=[42, 123, 999]):
    """Compare Fisher-optimal vs uniform difficulty distribution."""
    print("=" * 60)
    print("Ablation: Fisher-Optimal vs Uniform Difficulty")
    print("=" * 60)
    
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
        
        fisher_optimal_results = []
        uniform_results = []
        
        for t in range(n_timepoints):
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            
            # Method 1: Fisher-optimal (match ability distribution)
            target_fisher = true_abilities
            
            # Method 2: Uniform over ability range
            ability_range = (true_abilities.min(), true_abilities.max())
            target_uniform = np.random.uniform(ability_range[0], ability_range[1], 1000)
            
            for method_name, target, results_list in [
                ('fisher_optimal', target_fisher, fisher_optimal_results),
                ('uniform', target_uniform, uniform_results)
            ]:
                selected_items, _ = sinkhorn_selection(
                    difficulties, target, n_select=500,
                    epsilon=0.01, domain_labels=domains, domain_min_fraction=0.1
                )
                
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
            'fisher_optimal': fisher_optimal_results,
            'uniform': uniform_results
        })
    
    # Aggregate
    f_fisher = []
    u_fisher = []
    f_acc = []
    u_acc = []
    
    for r in results_per_seed:
        f_fisher.append(np.mean([x['metrics'].get('expected_fisher_info', 0) 
                                 for x in r['fisher_optimal']]))
        u_fisher.append(np.mean([x['metrics'].get('expected_fisher_info', 0) 
                                 for x in r['uniform']]))
        f_acc.append(np.mean([x['metrics']['pairwise_accuracy'] for x in r['fisher_optimal']]))
        u_acc.append(np.mean([x['metrics']['pairwise_accuracy'] for x in r['uniform']]))
    
    print(f"\nFisher-optimal: Fisher={np.mean(f_fisher):.1f}±{np.std(f_fisher):.1f}, "
          f"acc={np.mean(f_acc):.4f}±{np.std(f_acc):.4f}")
    print(f"Uniform: Fisher={np.mean(u_fisher):.1f}±{np.std(u_fisher):.1f}, "
          f"acc={np.mean(u_acc):.4f}±{np.std(u_acc):.4f}")
    
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(f_fisher, u_fisher)
    print(f"Fisher info t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    final_results = {
        'experiment': 'ablation_difficulty',
        'per_seed': results_per_seed,
        'summary': {
            'fisher_optimal_mean_fisher': float(np.mean(f_fisher)),
            'uniform_mean_fisher': float(np.mean(u_fisher)),
            'fisher_optimal_mean_acc': float(np.mean(f_acc)),
            'uniform_mean_acc': float(np.mean(u_acc)),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        }
    }
    
    os.makedirs('exp/ablation_difficulty', exist_ok=True)
    with open('exp/ablation_difficulty/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to exp/ablation_difficulty/results.json")
    return final_results


if __name__ == '__main__':
    run_ablation_difficulty()
