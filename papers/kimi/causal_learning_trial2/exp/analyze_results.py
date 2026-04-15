"""
Analyze experimental results and generate results.json for Stage 2.
Performs statistical tests to verify success criteria.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import json
from collections import defaultdict
from scipy import stats


def load_results(filename):
    """Load results from JSON file."""
    filepath = os.path.join(PROJECT_ROOT, "results/synthetic", filename)
    if not os.path.exists(filepath):
        filepath = os.path.join(PROJECT_ROOT, "results/ablations", filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_statistics(results, filter_fn=None):
    """Compute mean and std of metrics."""
    if filter_fn:
        results = [r for r in results if filter_fn(r)]
    
    if not results:
        return {}
    
    metrics = ['shd', 'tpr', 'fdr', 'precision', 'recall', 'f1', 'runtime']
    stats_dict = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r and r[metric] is not None]
        if values:
            stats_dict[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'n': len(values)
            }
    
    return stats_dict


def paired_wilcoxon_test(results1, results2, metric='shd'):
    """Perform paired Wilcoxon signed-rank test."""
    # Match by graph_id, mechanism, n_samples, seed
    dict1 = {(r['graph_id'], r['mechanism'], r['n_samples'], r['seed']): r 
             for r in results1 if metric in r and r[metric] is not None}
    dict2 = {(r['graph_id'], r['mechanism'], r['n_samples'], r['seed']): r 
             for r in results2 if metric in r and r[metric] is not None}
    
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    if len(common_keys) < 5:
        return {'p_value': 1.0, 'n': len(common_keys), 'significant': False}
    
    values1 = [dict1[k][metric] for k in common_keys]
    values2 = [dict2[k][metric] for k in common_keys]
    
    try:
        statistic, p_value = stats.wilcoxon(values1, values2)
        return {
            'p_value': float(p_value),
            'n': len(common_keys),
            'significant': p_value < 0.05,
            'mean_diff': float(np.mean(np.array(values1) - np.array(values2))),
            'median_diff': float(np.median(np.array(values1) - np.array(values2)))
        }
    except:
        return {'p_value': 1.0, 'n': len(common_keys), 'significant': False}


def cohens_d(values1, values2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(values1), len(values2)
    var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(values1) - np.mean(values2)) / pooled_std


def analyze_main_results():
    """Analyze main method comparisons."""
    print("=" * 60)
    print("Analyzing Main Results")
    print("=" * 60)
    
    # Load results
    spiced_results = load_results("spiced_knn_results.json")
    notears_results = load_results("notears_results.json")
    pc_results = load_results("pc_results.json")
    
    if spiced_results is None:
        print("ERROR: SPICED results not found")
        return {}
    
    analysis = {
        'methods': {},
        'comparisons': {},
        'success_criteria': {}
    }
    
    # Overall statistics by method
    for name, results in [('SPICED', spiced_results), 
                          ('NOTEARS', notears_results),
                          ('PC', pc_results)]:
        if results:
            analysis['methods'][name] = compute_statistics(results)
            print(f"\n{name} Overall:")
            if 'shd' in analysis['methods'][name]:
                s = analysis['methods'][name]['shd']
                print(f"  SHD: {s['mean']:.2f} ± {s['std']:.2f} (n={s['n']})")
    
    # Primary Success Criterion 1: SPICED vs NOTEARS for N <= 200
    print("\n" + "-" * 40)
    print("Primary Criterion 1: SPICED vs NOTEARS (N <= 200)")
    print("-" * 40)
    
    if spiced_results and notears_results:
        mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
        mechanism_pass = 0
        mechanism_results = {}
        
        for mechanism in mechanisms:
            spiced_filtered = [r for r in spiced_results 
                              if r['mechanism'] == mechanism and r['n_samples'] <= 200]
            notears_filtered = [r for r in notears_results 
                               if r['mechanism'] == mechanism and r['n_samples'] <= 200]
            
            if spiced_filtered and notears_filtered:
                stat_test = paired_wilcoxon_test(spiced_filtered, notears_filtered, 'shd')
                
                # Check if SPICED has lower SHD
                spiced_shd = [r['shd'] for r in spiced_filtered if 'shd' in r]
                notears_shd = [r['shd'] for r in notears_filtered if 'shd' in r]
                
                if spiced_shd and notears_shd:
                    lower_shd = np.mean(spiced_shd) < np.mean(notears_shd)
                    passed = stat_test['significant'] and lower_shd
                    
                    mechanism_results[mechanism] = {
                        'spiced_shd_mean': float(np.mean(spiced_shd)),
                        'notears_shd_mean': float(np.mean(notears_shd)),
                        'p_value': stat_test['p_value'],
                        'significant': stat_test['significant'],
                        'lower_shd': lower_shd,
                        'passed': passed,
                        'n': stat_test['n']
                    }
                    
                    if passed:
                        mechanism_pass += 1
                    
                    print(f"  {mechanism}:")
                    print(f"    SPICED SHD: {np.mean(spiced_shd):.2f} ± {np.std(spiced_shd):.2f}")
                    print(f"    NOTEARS SHD: {np.mean(notears_shd):.2f} ± {np.std(notears_shd):.2f}")
                    print(f"    p-value: {stat_test['p_value']:.4f}, Significant: {stat_test['significant']}, Lower: {lower_shd}")
                    print(f"    PASSED: {passed}")
        
        analysis['success_criteria']['criterion_1'] = {
            'description': 'SPICED achieves significantly lower SHD than NOTEARS for N <= 200 on >=3 mechanisms',
            'mechanisms_passed': mechanism_pass,
            'required': 3,
            'passed': mechanism_pass >= 3,
            'mechanism_details': mechanism_results
        }
        
        print(f"\n  Mechanisms passed: {mechanism_pass}/4 (need >=3)")
        print(f"  Criterion 1 PASSED: {mechanism_pass >= 3}")
    
    # Primary Success Criterion 2: Runtime for n=50
    print("\n" + "-" * 40)
    print("Primary Criterion 2: Runtime for n=50 graphs")
    print("-" * 40)
    
    if spiced_results:
        n50_results = [r for r in spiced_results if r.get('n_nodes') == 50]
        if n50_results:
            runtimes = [r['runtime'] for r in n50_results if r.get('runtime')]
            median_runtime = np.median(runtimes) if runtimes else None
            
            analysis['success_criteria']['criterion_2'] = {
                'description': 'SPICED runs in < 5 minutes for n=50 graphs',
                'median_runtime_seconds': float(median_runtime) if median_runtime else None,
                'median_runtime_minutes': float(median_runtime / 60) if median_runtime else None,
                'threshold_minutes': 5,
                'passed': median_runtime < 300 if median_runtime else False,
                'n': len(runtimes)
            }
            
            print(f"  Median runtime: {median_runtime:.2f}s ({median_runtime/60:.2f} min)")
            print(f"  Threshold: 300s (5 min)")
            print(f"  Criterion 2 PASSED: {median_runtime < 300 if median_runtime else False}")
    
    # Comparison by sample size
    print("\n" + "-" * 40)
    print("Analysis by Sample Size")
    print("-" * 40)
    
    for N in [50, 100, 200, 500, 1000]:
        if spiced_results and notears_results:
            spiced_N = [r for r in spiced_results if r['n_samples'] == N]
            notears_N = [r for r in notears_results if r['n_samples'] == N]
            
            if spiced_N and notears_N:
                stat_test = paired_wilcoxon_test(spiced_N, notears_N, 'shd')
                
                spiced_shd = [r['shd'] for r in spiced_N if 'shd' in r]
                notears_shd = [r['shd'] for r in notears_N if 'shd' in r]
                
                print(f"  N={N}:")
                print(f"    SPICED: {np.mean(spiced_shd):.2f} ± {np.std(spiced_shd):.2f}")
                print(f"    NOTEARS: {np.mean(notears_shd):.2f} ± {np.std(notears_shd):.2f}")
                print(f"    p-value: {stat_test['p_value']:.4f}")
    
    return analysis


def analyze_ablations():
    """Analyze ablation study results."""
    print("\n" + "=" * 60)
    print("Analyzing Ablation Studies")
    print("=" * 60)
    
    analysis = {}
    
    # Ablation 1: k-NN vs Kernel MI
    print("\nAblation 1: k-NN vs Kernel/Correlation-based MI")
    spiced_knn = load_results("spiced_knn_results.json")
    spiced_kernel = load_results("spiced_kernel_mi.json")
    
    if spiced_knn and spiced_kernel:
        stat_test = paired_wilcoxon_test(spiced_knn, spiced_kernel, 'shd')
        
        knn_shd = [r['shd'] for r in spiced_knn if 'shd' in r]
        kernel_shd = [r['shd'] for r in spiced_kernel if 'shd' in r]
        
        analysis['phase1_mi_method'] = {
            'knn_shd_mean': float(np.mean(knn_shd)),
            'knn_shd_std': float(np.std(knn_shd)),
            'kernel_shd_mean': float(np.mean(kernel_shd)),
            'kernel_shd_std': float(np.std(kernel_shd)),
            'p_value': stat_test['p_value'],
            'significant': stat_test['significant'],
            'conclusion': 'k-NN superior' if np.mean(knn_shd) < np.mean(kernel_shd) else 'kernel superior'
        }
        
        print(f"  k-NN SHD: {np.mean(knn_shd):.2f} ± {np.std(knn_shd):.2f}")
        print(f"  Kernel SHD: {np.mean(kernel_shd):.2f} ± {np.std(kernel_shd):.2f}")
        print(f"  p-value: {stat_test['p_value']:.4f}")
    
    # Ablation 2: Structural constraints
    print("\nAblation 2: With vs Without Structural Constraints")
    spiced_full = load_results("spiced_knn_results.json")
    spiced_no_constraints = load_results("spiced_no_constraints.json")
    
    if spiced_full and spiced_no_constraints:
        # Filter to same configs
        stat_test = paired_wilcoxon_test(spiced_full, spiced_no_constraints, 'shd')
        
        full_shd = [r['shd'] for r in spiced_full if 'shd' in r]
        no_constr_shd = [r['shd'] for r in spiced_no_constraints if 'shd' in r]
        
        analysis['structural_constraints'] = {
            'with_constraints_shd_mean': float(np.mean(full_shd)) if full_shd else None,
            'without_constraints_shd_mean': float(np.mean(no_constr_shd)) if no_constr_shd else None,
            'p_value': stat_test['p_value'],
            'significant': stat_test['significant'],
            'improvement': float(np.mean(no_constr_shd) - np.mean(full_shd)) if full_shd and no_constr_shd else None
        }
        
        if full_shd and no_constr_shd:
            print(f"  With constraints: {np.mean(full_shd):.2f} ± {np.std(full_shd):.2f}")
            print(f"  Without constraints: {np.mean(no_constr_shd):.2f} ± {np.std(no_constr_shd):.2f}")
            print(f"  Improvement: {np.mean(no_constr_shd) - np.mean(full_shd):.2f}")
            print(f"  p-value: {stat_test['p_value']:.4f}")
    
    # Ablation 3: IT initialization
    print("\nAblation 3: IT Initialization vs Random")
    spiced_it_init = load_results("spiced_knn_results.json")
    spiced_random_init = load_results("spiced_no_it_init.json")
    
    if spiced_it_init and spiced_random_init:
        stat_test = paired_wilcoxon_test(spiced_it_init, spiced_random_init, 'shd')
        
        it_shd = [r['shd'] for r in spiced_it_init if 'shd' in r]
        random_shd = [r['shd'] for r in spiced_random_init if 'shd' in r]
        
        analysis['it_initialization'] = {
            'it_init_shd_mean': float(np.mean(it_shd)) if it_shd else None,
            'random_init_shd_mean': float(np.mean(random_shd)) if random_shd else None,
            'p_value': stat_test['p_value'],
            'significant': stat_test['significant']
        }
        
        if it_shd and random_shd:
            print(f"  IT initialization: {np.mean(it_shd):.2f} ± {np.std(it_shd):.2f}")
            print(f"  Random initialization: {np.mean(random_shd):.2f} ± {np.std(random_shd):.2f}")
            print(f"  p-value: {stat_test['p_value']:.4f}")
    
    return analysis


def analyze_sachs():
    """Analyze Sachs dataset results."""
    print("\n" + "=" * 60)
    print("Analyzing Sachs Dataset Results")
    print("=" * 60)
    
    sachs_files = {
        'SPICED': 'results/real_world/spiced_sachs.json',
        'NOTEARS': 'results/real_world/notears_sachs.json'
    }
    
    analysis = {}
    
    for method, filepath in sachs_files.items():
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                results = json.load(f)
            
            shd_values = [r['shd'] for r in results if 'shd' in r]
            if shd_values:
                analysis[method] = {
                    'shd_mean': float(np.mean(shd_values)),
                    'shd_std': float(np.std(shd_values)),
                    'shd_median': float(np.median(shd_values)),
                    'passed_threshold': np.median(shd_values) < 10
                }
                
                print(f"  {method}: SHD = {np.mean(shd_values):.2f} ± {np.std(shd_values):.2f}")
    
    # Primary Criterion 3
    if 'SPICED' in analysis:
        analysis['criterion_3'] = {
            'description': 'SPICED achieves SHD < 10 on Sachs dataset',
            'median_shd': analysis['SPICED']['shd_median'],
            'threshold': 10,
            'passed': analysis['SPICED']['shd_median'] < 10
        }
        print(f"\n  Criterion 3 (SHD < 10): {analysis['SPICED']['shd_median']:.2f} < 10 = {analysis['SPICED']['shd_median'] < 10}")
    
    return analysis


def generate_results_json():
    """Generate final results.json file."""
    print("\n" + "=" * 60)
    print("Generating Final Results JSON")
    print("=" * 60)
    
    main_analysis = analyze_main_results()
    ablation_analysis = analyze_ablations()
    sachs_analysis = analyze_sachs()
    
    # Compile final results
    results = {
        'experiment_summary': {
            'description': 'SPICED: Sample-Efficient Prior-Informed Causal Estimation via Directed Information',
            'experiments_run': [
                'SPICED with k-NN entropy estimation',
                'NOTEARS baseline',
                'PC baseline',
                'Ablation: k-NN vs Kernel MI',
                'Ablation: With/without structural constraints',
                'Ablation: IT vs random initialization',
                'Sachs dataset evaluation'
            ],
            'graph_sizes': [10, 20, 30, 50],
            'sample_sizes': [50, 100, 200, 500, 1000],
            'mechanisms': ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
        },
        'main_results': main_analysis,
        'ablation_studies': ablation_analysis,
        'real_world_results': sachs_analysis,
        'success_criteria_summary': {
            'criterion_1_sample_efficiency': main_analysis.get('success_criteria', {}).get('criterion_1', {}).get('passed', False),
            'criterion_2_scalability': main_analysis.get('success_criteria', {}).get('criterion_2', {}).get('passed', False),
            'criterion_3_sachs': sachs_analysis.get('criterion_3', {}).get('passed', False),
            'all_passed': (
                main_analysis.get('success_criteria', {}).get('criterion_1', {}).get('passed', False) and
                main_analysis.get('success_criteria', {}).get('criterion_2', {}).get('passed', False) and
                sachs_analysis.get('criterion_3', {}).get('passed', False)
            )
        }
    }
    
    # Save to results.json
    output_path = os.path.join(PROJECT_ROOT, "results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA SUMMARY")
    print("=" * 60)
    print(f"Criterion 1 (Sample Efficiency): {results['success_criteria_summary']['criterion_1_sample_efficiency']}")
    print(f"Criterion 2 (Scalability): {results['success_criteria_summary']['criterion_2_scalability']}")
    print(f"Criterion 3 (Sachs Accuracy): {results['success_criteria_summary']['criterion_3_sachs']}")
    print(f"\nALL CRITERIA PASSED: {results['success_criteria_summary']['all_passed']}")
    
    return results


if __name__ == "__main__":
    results = generate_results_json()
