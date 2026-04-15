"""
Ablation: IRT Model Choice Sensitivity.

Tests whether results hold across different IRT model choices (1PL, 2PL, 3PL).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import numpy as np
from shared.irt_models import OnePLModel, TwoPLModel, ThreePLModel
from shared.metrics import compute_all_metrics


def run_ablation_irt(seeds=[42, 123, 999]):
    """Compare different IRT models."""
    print("=" * 60)
    print("Ablation: IRT Model Choice Sensitivity")
    print("=" * 60)
    
    difficulties = np.load('data/pools/difficulties.npy')
    discriminations = np.load('data/pools/discriminations.npy')
    
    n_models = 28
    n_timepoints = 5
    
    # Fixed item selection for fair comparison
    np.random.seed(42)
    selected_items = np.random.choice(len(difficulties), 500, replace=False)
    
    results_per_seed = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        
        model_results = {
            '1PL': [],
            '2PL': [],
            '3PL': []
        }
        
        for t in range(n_timepoints):
            true_abilities = np.load(f'data/population/abilities_t{t}.npy')
            all_responses = np.load(f'data/population/responses_t{t}.npy')
            responses = all_responses[:, selected_items]
            
            for model_name, model_class in [
                ('1PL', OnePLModel),
                ('2PL', TwoPLModel),
                ('3PL', ThreePLModel)
            ]:
                # Fit IRT model
                irt_model = model_class(n_models, len(selected_items))
                irt_model.difficulties = difficulties[selected_items].copy()
                if hasattr(irt_model, 'discriminations'):
                    irt_model.discriminations = discriminations[selected_items].copy()
                
                # Estimate abilities
                estimated_abilities = irt_model.estimate_abilities_mle(responses)
                
                # Compute metrics
                metrics = compute_all_metrics(
                    true_abilities, estimated_abilities,
                    irt_model=irt_model if model_name == '2PL' else None,
                    selected_items=None,
                    ability_distribution=true_abilities
                )
                
                model_results[model_name].append({
                    'time': t * 3,
                    'metrics': metrics
                })
        
        results_per_seed.append({
            'seed': seed,
            'model_results': model_results
        })
    
    # Aggregate
    summary = {}
    for model_name in ['1PL', '2PL', '3PL']:
        taus = []
        for r in results_per_seed:
            taus.append(np.mean([x['metrics']['kendall_tau'] 
                                for x in r['model_results'][model_name]]))
        
        summary[model_name] = {
            'mean_tau': float(np.mean(taus)),
            'std_tau': float(np.std(taus))
        }
        print(f"{model_name}: τ={np.mean(taus):.4f}±{np.std(taus):.4f}")
    
    # Correlation between models
    print("\nModel correlations:")
    for i, m1 in enumerate(['1PL', '2PL', '3PL']):
        for m2 in ['2PL', '3PL'][i:]:
            if m1 != m2:
                corrs = []
                for r in results_per_seed:
                    for t in range(n_timepoints):
                        est1 = r['model_results'][m1][t]['metrics']['estimated_abilities']
                        est2 = r['model_results'][m2][t]['metrics']['estimated_abilities']
                        corrs.append(np.corrcoef(est1, est2)[0, 1])
                print(f"  {m1} vs {m2}: {np.mean(corrs):.4f}")
    
    final_results = {
        'experiment': 'ablation_irt',
        'per_seed': results_per_seed,
        'summary': summary
    }
    
    os.makedirs('exp/ablation_irt', exist_ok=True)
    with open('exp/ablation_irt/results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nResults saved to exp/ablation_irt/results.json")
    return final_results


if __name__ == '__main__':
    run_ablation_irt()
