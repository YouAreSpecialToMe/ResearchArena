"""
Ablation: Update Frequency Analysis.

Tests sensitivity to benchmark update frequency (monthly vs quarterly vs bi-annual).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from shared.irt_models import TwoPLModel
from shared.metrics import compute_all_metrics, ranking_stability
from shared.optimal_transport import sinkhorn_selection


def run_ablation_frequency(seeds=[42, 123, 999]):
    """Compare different update frequencies."""
    print("=" * 60)
    print("Ablation: Update Frequency Analysis")
    print("=" * 60)
    
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    with open('data/pools/item_pool.jsonl', 'r') as f:
        items = [json.loads(line) for line in f]
    domains = np.array([{'math': 0, 'code': 1, 'science': 2}.get(i['domain'], 0) for i in items])
    
    n_models = 28
    n_timepoints = 5
    
    # Define update schedules
    schedules = {
        'monthly': [0, 1, 2, 3, 4],  # Update every month
        'quarterly': [0, 3, 6, 9, 12],  # Update every 3 months (same as time points)
        'bi_annual': [0, 6, 12]  # Update every 6 months
    }
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        
        schedule_results = {name: [] for name in schedules}
        
        for schedule_name, update_times in schedules.items():
            print(f"  Schedule: {schedule_name}")
            
            current_items = None
            
            for t in range(n_timepoints):
                true_abilities = np.load(f'data/population/abilities_t{t}.npy')
                all_responses = np.load(f'data/population/responses_t{t}.npy')
                
                # Check if we should update at this time
                actual_time = t * 3
                if actual_time in update_times or current_items is None:
                    # Update item selection
                    target_samples = true_abilities
                    current_items, _ = sinkhorn_selection(
                        difficulties, target_samples, n_select=500,
                        epsilon=0.01, domain_labels=domains, domain_min_fraction=0.1
                    )
                
                # Evaluate
                responses = all_responses[:, current_items]
                
                irt_model = TwoPLModel(n_models, len(current_items))
                irt_model.difficulties = difficulties[current_items].copy()
                irt_model.discriminations = discriminations[current_items].copy()
                
                estimated_abilities = irt_model.estimate_abilities_mle(responses)
                
                metrics = compute_all_metrics(
                    true_abilities, estimated_abilities,
                    irt_model=irt_model, selected_items=None,
                    ability_distribution=true_abilities
                )
                
                schedule_results[schedule_name].append({
                    'time': actual_time,
                    'metrics': metrics
                })
        
        results_per_seed.append({
            'seed': seed,
            'schedule_results': schedule_results
        })
    
    # Aggregate stability for each schedule
    summary = {}
    for schedule_name in schedules:
        stabilities = []
        for r in results_per_seed:
            rankings = [x['metrics']['kendall_tau'] for x in r['schedule_results'][schedule_name]]
            stabilities.append(np.std(rankings))
        
        summary[schedule_name] = {
            'mean_stability': float(np.mean(stabilities)),
            'std_stability': float(np.std(stabilities))
        }
        print(f"\n{schedule_name}: stability={np.mean(stabilities):.4f}±{np.std(stabilities):.4f}")
    
    final_results = {
        'experiment': 'ablation_frequency',
        'per_seed': results_per_seed,
        'summary': summary
    }
    
    os.makedirs('exp/ablation_frequency', exist_ok=True)
    with open('exp/ablation_frequency/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to exp/ablation_frequency/results.json")
    return final_results


if __name__ == '__main__':
    run_ablation_frequency()
