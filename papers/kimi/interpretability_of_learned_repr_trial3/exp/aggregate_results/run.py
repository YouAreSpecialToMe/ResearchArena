"""
Aggregate results from all experiments and generate final results.json.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import json
import os
import numpy as np

def load_json(path):
    """Load JSON file if it exists."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def aggregate_seeds(base_path, seeds=[42, 43, 44]):
    """Aggregate results across multiple seeds."""
    all_results = []
    for seed in seeds:
        path = base_path.format(seed=seed)
        result = load_json(path)
        if result:
            all_results.append(result)
    
    if not all_results:
        return None
    
    # Aggregate numeric metrics
    numeric_keys = ['fvu', 'l0_sparsity', 'dead_features_pct', 
                   'feature_diversity_entropy', 'feature_correlation']
    
    aggregated = {}
    for key in numeric_keys:
        values = [r[key] for r in all_results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
    
    return aggregated

def main():
    results = {
        'experiment_info': {
            'name': 'RobustSAE Experiments',
            'description': 'Adversarially robust sparse autoencoders',
            'seeds': [42, 43, 44]
        },
        'baselines': {},
        'ablations': {},
        'evaluations': {}
    }
    
    # Aggregate baseline results
    print("Aggregating baseline results...")
    
    for baseline in ['topk_baseline', 'jumprelu_baseline', 'denoising_baseline']:
        path = f'exp/{baseline}/results_seed{{seed}}.json'
        agg = aggregate_seeds(path)
        if agg:
            results['baselines'][baseline] = agg
    
    # Aggregate RobustSAE results
    print("Aggregating RobustSAE results...")
    
    for method in ['robustsae_full', 'robustsae_no_proxy']:
        path = f'exp/{method}/results_seed{{seed}}.json'
        agg = aggregate_seeds(path)
        if agg:
            results['ablations'][method] = agg
    
    # Load lambda ablation
    print("Loading lambda ablation results...")
    lambda_results = {}
    for lambda_val in [0.0, 0.01, 0.1, 1.0]:
        path = f'exp/ablation_lambda/lambda_{lambda_val}.json'
        result = load_json(path)
        if result:
            lambda_results[str(lambda_val)] = result
    if lambda_results:
        results['ablations']['lambda_sweep'] = lambda_results
    
    # Load robustness evaluation
    print("Loading robustness evaluation...")
    robustness = load_json('results/robustness_evaluation.json')
    if robustness:
        results['evaluations']['robustness'] = robustness
    
    # Load proxy validation
    print("Loading proxy validation...")
    proxy_val = load_json('results/proxy_validation.json')
    if proxy_val:
        results['evaluations']['proxy_validation'] = proxy_val
    
    # Compute summary statistics
    print("Computing summary statistics...")
    
    if 'robustness' in results['evaluations']:
        robustness = results['evaluations']['robustness']
        
        # Check if RobustSAE improves over baselines
        if 'robust' in robustness and 'topk' in robustness:
            baseline_stability = robustness['topk'].get('feature_stability', 0)
            robust_stability = robustness['robust'].get('feature_stability', 0)
            
            if baseline_stability > 0:
                improvement = (robust_stability - baseline_stability) / baseline_stability * 100
                results['summary'] = {
                    'stability_improvement_pct': improvement,
                    'robustsae_stability': robust_stability,
                    'baseline_stability': baseline_stability
                }
    
    # Save final results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nFinal results saved to results.json")
    print("\nSummary:")
    if 'summary' in results:
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
    
    return results

if __name__ == '__main__':
    main()
