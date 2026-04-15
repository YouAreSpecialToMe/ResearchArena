"""
Aggregate all experimental results into a single summary file.
"""

import json
import numpy as np
import os


def load_json(filepath):
    """Load JSON file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def aggregate_all_results():
    """Aggregate results from all experiments."""
    print("=" * 60)
    print("Aggregating Experimental Results")
    print("=" * 60)
    
    # Load all results
    experiments = {
        'baseline_static': load_json('exp/baseline_static/results.json'),
        'baseline_stratified': load_json('exp/baseline_stratified/results.json'),
        'baseline_fluid': load_json('exp/baseline_fluid/results.json'),
        'dynascale_full': load_json('exp/dynascale_full/results.json'),
        'ablation_selection': load_json('exp/ablation_selection/results.json'),
        'ablation_difficulty': load_json('exp/ablation_difficulty/results.json'),
        'ablation_frequency': load_json('exp/ablation_frequency/results.json'),
    }
    
    # Create summary table
    summary = {
        'baselines': {},
        'main_result': {},
        'ablations': {}
    }
    
    # Extract baseline results
    for baseline in ['baseline_static', 'baseline_stratified', 'baseline_fluid']:
        if experiments[baseline]:
            agg = experiments[baseline].get('aggregated', [])
            if agg:
                summary['baselines'][baseline] = {
                    'final_kendall_tau': agg[-1]['kendall_tau'],
                    'final_pairwise_acc': agg[-1]['pairwise_accuracy'],
                    'mean_kendall_tau': {
                        'mean': float(np.mean([x['kendall_tau']['mean'] for x in agg])),
                        'std': float(np.mean([x['kendall_tau']['std'] for x in agg]))
                    }
                }
    
    # Extract main DynaScale result
    if experiments['dynascale_full']:
        dyn = experiments['dynascale_full']
        summary['main_result'] = {
            'ranking_stability': dyn['summary'].get('ranking_accuracy_stability', 0),
            'fisher_retention': dyn['summary'].get('fisher_retention', 0),
            'min_kendall_tau': dyn['summary'].get('min_kendall_tau', 0),
            'mean_stability': dyn['summary'].get('mean_stability', 0),
            'success_criteria': dyn['summary'].get('success_criteria', {})
        }
    
    # Extract ablation results
    if experiments['ablation_selection']:
        summary['ablations']['selection_method'] = experiments['ablation_selection']['summary']
    
    if experiments['ablation_difficulty']:
        summary['ablations']['difficulty_target'] = experiments['ablation_difficulty']['summary']
    
    if experiments['ablation_frequency']:
        summary['ablations']['update_frequency'] = experiments['ablation_frequency']['summary']
    
    # Create full aggregated results
    full_results = {
        'experiments': experiments,
        'summary': summary,
        'meta': {
            'n_experiments': len([e for e in experiments.values() if e is not None]),
            'success_criteria_summary': summary['main_result'].get('success_criteria', {})
        }
    }
    
    # Save
    with open('results.json', 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print("\nSummary:")
    print("-" * 60)
    
    # Print baseline comparison
    print("\nBaseline Comparison (Final Kendall's τ):")
    for name, data in summary['baselines'].items():
        tau = data['final_kendall_tau']
        print(f"  {name}: {tau['mean']:.4f} ± {tau['std']:.4f}")
    
    # Print DynaScale results
    print("\nDynaScale Results:")
    mr = summary['main_result']
    print(f"  Ranking stability: {mr.get('ranking_stability', 0):.4f} (target <0.05)")
    print(f"  Fisher retention: {mr.get('fisher_retention', 0):.2%} (target >90%)")
    print(f"  Min Kendall's τ: {mr.get('min_kendall_tau', 0):.4f} (target >0.95)")
    
    # Print success criteria
    print("\nSuccess Criteria Met:")
    sc = mr.get('success_criteria', {})
    for criterion, met in sc.items():
        status = "✓" if met else "✗"
        print(f"  {status} {criterion}")
    
    print("\n" + "=" * 60)
    print("Full results saved to results.json")
    print("=" * 60)
    
    return full_results


if __name__ == '__main__':
    aggregate_all_results()
