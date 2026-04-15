#!/usr/bin/env python3
"""
Analyze experimental results and generate aggregated results.json with statistical tests.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats
import sys

sys.path.insert(0, 'exp')
from shared.metrics import statistical_test


def load_all_results():
    """Load all experiment results from outputs/results/ directory."""
    results_dir = Path("outputs/results")
    all_results = {}
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return all_results
    
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)
                exp_name = data.get('experiment', result_file.stem)
                all_results[exp_name] = data
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return all_results


def aggregate_by_method(all_results):
    """Aggregate results by method across seeds."""
    methods = {}
    
    for exp_name, data in all_results.items():
        # Extract method name (remove seed suffix)
        if '_seed' in exp_name:
            method_name = exp_name.rsplit('_seed', 1)[0]
        else:
            method_name = exp_name
        
        if method_name not in methods:
            methods[method_name] = {
                'seeds': [],
                'metrics': {}
            }
        
        methods[method_name]['seeds'].append(data.get('seed', 0))
        
        # Collect metrics
        metrics = data.get('metrics', {})
        for metric_name, value in metrics.items():
            if metric_name not in methods[method_name]['metrics']:
                methods[method_name]['metrics'][metric_name] = []
            methods[method_name]['metrics'][metric_name].append(value)
    
    # Compute mean and std for each method
    aggregated = {}
    for method_name, data in methods.items():
        aggregated[method_name] = {
            'seeds': data['seeds'],
            'num_runs': len(data['seeds'])
        }
        
        for metric_name, values in data['metrics'].items():
            values_array = np.array(values)
            aggregated[method_name][metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'values': [float(v) for v in values],
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array))
            }
    
    return aggregated


def perform_statistical_tests(aggregated):
    """Perform statistical tests between methods."""
    tests = {}
    
    # Get baseline uniform and distflow_idw results
    baseline = aggregated.get('baseline_uniform', {})
    distflow = aggregated.get('distflow_idw', {})
    
    if baseline and distflow:
        # Test CD-far (primary metric)
        if 'cd_far' in baseline and 'cd_far' in distflow:
            baseline_values = baseline['cd_far']['values']
            distflow_values = distflow['cd_far']['values']
            
            # Paired t-test (using independent t-test as approximation)
            t_stat, p_value = stats.ttest_ind(baseline_values, distflow_values)
            
            tests['cd_far_improvement'] = {
                'baseline_mean': baseline['cd_far']['mean'],
                'distflow_mean': distflow['cd_far']['mean'],
                'improvement_percent': (
                    (baseline['cd_far']['mean'] - distflow['cd_far']['mean']) 
                    / baseline['cd_far']['mean'] * 100
                ),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'baseline_values': baseline_values,
                'distflow_values': distflow_values
            }
        
        # Test CD-near (check no degradation)
        if 'cd_near' in baseline and 'cd_near' in distflow:
            baseline_values = baseline['cd_near']['values']
            distflow_values = distflow['cd_near']['values']
            
            t_stat, p_value = stats.ttest_ind(baseline_values, distflow_values)
            
            tests['cd_near_comparison'] = {
                'baseline_mean': baseline['cd_near']['mean'],
                'distflow_mean': distflow['cd_near']['mean'],
                'change_percent': (
                    (distflow['cd_near']['mean'] - baseline['cd_near']['mean'])
                    / baseline['cd_near']['mean'] * 100
                ),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_degradation': bool(p_value < 0.05 and distflow['cd_near']['mean'] > baseline['cd_near']['mean'])
            }
        
        # Test CD-overall
        if 'cd_overall' in baseline and 'cd_overall' in distflow:
            baseline_values = baseline['cd_overall']['values']
            distflow_values = distflow['cd_overall']['values']
            
            t_stat, p_value = stats.ttest_ind(baseline_values, distflow_values)
            
            tests['cd_overall_improvement'] = {
                'baseline_mean': baseline['cd_overall']['mean'],
                'distflow_mean': distflow['cd_overall']['mean'],
                'improvement_percent': (
                    (baseline['cd_overall']['mean'] - distflow['cd_overall']['mean'])
                    / baseline['cd_overall']['mean'] * 100
                ),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
    
    return tests


def check_success_criteria(aggregated, tests):
    """Check if success criteria are met."""
    criteria = {
        'cd_far_improvement_threshold': 25,  # 25% improvement
        'p_value_threshold': 0.05,
        'cd_near_degradation_tolerance': 5  # 5% tolerance
    }
    
    results = {
        'primary_hypothesis_confirmed': False,
        'criteria': {}
    }
    
    # Check CD-far improvement
    if 'cd_far_improvement' in tests:
        improvement = tests['cd_far_improvement']['improvement_percent']
        p_value = tests['cd_far_improvement']['p_value']
        
        results['criteria']['cd_far_improvement'] = {
            'achieved': improvement >= criteria['cd_far_improvement_threshold'],
            'value': improvement,
            'threshold': criteria['cd_far_improvement_threshold']
        }
        
        results['criteria']['statistical_significance'] = {
            'achieved': p_value < criteria['p_value_threshold'],
            'p_value': p_value,
            'threshold': criteria['p_value_threshold']
        }
        
        # Primary hypothesis: significant improvement in CD-far
        results['primary_hypothesis_confirmed'] = bool(
            improvement >= criteria['cd_far_improvement_threshold'] and
            p_value < criteria['p_value_threshold']
        )
    
    # Check no significant degradation in CD-near
    if 'cd_near_comparison' in tests:
        change = tests['cd_near_comparison']['change_percent']
        p_value = tests['cd_near_comparison']['p_value']
        
        # No significant degradation means either improvement or non-significant change
        no_degradation = not (p_value < criteria['p_value_threshold'] and change > criteria['cd_near_degradation_tolerance'])
        
        results['criteria']['cd_near_no_degradation'] = {
            'achieved': no_degradation,
            'change_percent': change,
            'p_value': p_value
        }
    
    return results


def create_comparison_table(aggregated):
    """Create a comparison table of all methods."""
    table = []
    
    method_order = [
        'baseline_uniform',
        'baseline_density',
        'distflow_idw',
        'distflow_law',
        'ablation_no_film',
        'ablation_no_stratify'
    ]
    
    for method in method_order:
        if method in aggregated:
            data = aggregated[method]
            row = {
                'method': method,
                'n_runs': data['num_runs'],
                'cd_overall': f"{data.get('cd_overall', {}).get('mean', 0):.4f} ± {data.get('cd_overall', {}).get('std', 0):.4f}",
                'cd_near': f"{data.get('cd_near', {}).get('mean', 0):.4f} ± {data.get('cd_near', {}).get('std', 0):.4f}",
                'cd_mid': f"{data.get('cd_mid', {}).get('mean', 0):.4f} ± {data.get('cd_mid', {}).get('std', 0):.4f}",
                'cd_far': f"{data.get('cd_far', {}).get('mean', 0):.4f} ± {data.get('cd_far', {}).get('std', 0):.4f}",
                'emd': f"{data.get('emd', {}).get('mean', 0):.4f} ± {data.get('emd', {}).get('std', 0):.4f}"
            }
            table.append(row)
    
    return table


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Analyzing Experimental Results")
    print("=" * 60)
    
    # Load all results
    print("\n1. Loading experiment results...")
    all_results = load_all_results()
    print(f"   Found {len(all_results)} result files")
    
    if not all_results:
        print("   No results found. Exiting.")
        return
    
    # Aggregate by method
    print("\n2. Aggregating results by method...")
    aggregated = aggregate_by_method(all_results)
    print(f"   Found {len(aggregated)} methods:")
    for method in aggregated:
        print(f"     - {method}: {aggregated[method]['num_runs']} runs")
    
    # Perform statistical tests
    print("\n3. Performing statistical tests...")
    tests = perform_statistical_tests(aggregated)
    
    if 'cd_far_improvement' in tests:
        imp = tests['cd_far_improvement']
        print(f"   CD-far improvement: {imp['improvement_percent']:.1f}%")
        print(f"   p-value: {imp['p_value']:.4f}")
        print(f"   Statistically significant: {imp['significant']}")
    
    # Check success criteria
    print("\n4. Checking success criteria...")
    success = check_success_criteria(aggregated, tests)
    print(f"   Primary hypothesis confirmed: {success['primary_hypothesis_confirmed']}")
    
    # Create comparison table
    print("\n5. Creating comparison table...")
    table = create_comparison_table(aggregated)
    
    # Save aggregated results
    print("\n6. Saving aggregated results...")
    final_results = {
        'metadata': {
            'description': 'Aggregated experimental results for DistFlow',
            'num_experiments': len(all_results),
            'num_methods': len(aggregated)
        },
        'aggregated': aggregated,
        'statistical_tests': tests,
        'success_criteria': success,
        'comparison_table': table
    }
    
    with open('results.json', 'w') as f:
        json.dump(convert_to_native(final_results), f, indent=2)
    print("   Saved to results.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nComparison Table:")
    for row in table:
        print(f"\n{row['method']} (n={row['n_runs']}):")
        print(f"  CD-overall: {row['cd_overall']}")
        print(f"  CD-near:    {row['cd_near']}")
        print(f"  CD-mid:     {row['cd_mid']}")
        print(f"  CD-far:     {row['cd_far']}")
        print(f"  EMD:        {row['emd']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
