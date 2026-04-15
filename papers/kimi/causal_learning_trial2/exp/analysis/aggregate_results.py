"""
Aggregate results from all experiments and perform statistical analysis.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import json
import glob
from scipy import stats


def load_results(method):
    """Load results for a method."""
    results_file = os.path.join(PROJECT_ROOT, f"results/synthetic/{method}_summary.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


def load_sachs_results(method):
    """Load Sachs results for a method."""
    results_file = os.path.join(PROJECT_ROOT, f"results/real_world/{method}_sachs.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return []


def aggregate_by_configuration(results):
    """Aggregate results by (n_nodes, mechanism, n_samples)."""
    configs = {}
    
    for r in results:
        if 'error' in r:
            continue
        
        key = (r.get('n_nodes'), r.get('mechanism'), r.get('n_samples'))
        if key not in configs:
            configs[key] = []
        configs[key].append(r)
    
    aggregated = {}
    for key, config_results in configs.items():
        shds = [r['shd'] for r in config_results]
        tprs = [r['tpr'] for r in config_results]
        fdrs = [r['fdr'] for r in config_results]
        runtimes = [r['runtime'] for r in config_results if r.get('runtime')]
        
        aggregated[key] = {
            'n_graphs': len(config_results),
            'shd_mean': np.mean(shds),
            'shd_std': np.std(shds),
            'shd_sem': stats.sem(shds) if len(shds) > 1 else 0,
            'tpr_mean': np.mean(tprs),
            'tpr_std': np.std(tprs),
            'fdr_mean': np.mean(fdrs),
            'fdr_std': np.std(fdrs),
            'runtime_mean': np.mean(runtimes) if runtimes else None,
            'runtime_std': np.std(runtimes) if runtimes else None,
        }
    
    return aggregated


def perform_statistical_tests(results_dict):
    """Perform statistical tests between methods."""
    tests = {}
    
    methods = list(results_dict.keys())
    if len(methods) < 2:
        return tests
    
    # Group by configuration
    configs = set()
    for method in methods:
        configs.update(results_dict[method].keys())
    
    for config in configs:
        config_tests = {}
        
        # Compare SPICED vs NOTEARS for N <= 200
        if 'spiced' in results_dict and 'notears' in results_dict:
            if config in results_dict['spiced'] and config in results_dict['notears']:
                spiced_data = results_dict['spiced'][config]
                notears_data = results_dict['notears'][config]
                
                # Wilcoxon signed-rank test
                if spiced_data['n_graphs'] > 5:
                    # We need raw data for proper test, but we have aggregates
                    # Just record the comparison for now
                    config_tests['spiced_vs_notears'] = {
                        'spiced_shd_mean': spiced_data['shd_mean'],
                        'notears_shd_mean': notears_data['shd_mean'],
                        'spiced_better': spiced_data['shd_mean'] < notears_data['shd_mean']
                    }
        
        tests[str(config)] = config_tests
    
    return tests


def create_summary_table(results_dict):
    """Create summary table for paper."""
    table = []
    
    for method in results_dict:
        for config, agg in results_dict[method].items():
            n_nodes, mechanism, n_samples = config
            
            row = {
                'method': method,
                'n_nodes': n_nodes,
                'mechanism': mechanism,
                'n_samples': n_samples,
                'shd': f"{agg['shd_mean']:.2f} ± {agg['shd_std']:.2f}",
                'shd_mean': agg['shd_mean'],
                'shd_std': agg['shd_std'],
                'tpr': f"{agg['tpr_mean']:.3f} ± {agg['tpr_std']:.3f}",
                'fdr': f"{agg['fdr_mean']:.3f} ± {agg['fdr_std']:.3f}",
                'runtime': f"{agg['runtime_mean']:.2f}s" if agg['runtime_mean'] else 'N/A'
            }
            table.append(row)
    
    return table


def check_success_criteria(results_dict, sachs_results):
    """Check if success criteria are met."""
    criteria = {}
    
    # Criterion 1: SPICED SHD < NOTEARS SHD for N <= 200 on >= 3 mechanisms
    if 'spiced' in results_dict and 'notears' in results_dict:
        better_count = 0
        mechanisms = ['linear_gaussian', 'linear_nongaussian', 'nonlinear', 'anm']
        
        for mechanism in mechanisms:
            spiced_shds = []
            notears_shds = []
            
            for config, agg in results_dict['spiced'].items():
                if config[1] == mechanism and config[2] <= 200:
                    spiced_shds.append(agg['shd_mean'])
            
            for config, agg in results_dict['notears'].items():
                if config[1] == mechanism and config[2] <= 200:
                    notears_shds.append(agg['shd_mean'])
            
            if spiced_shds and notears_shds:
                if np.mean(spiced_shds) < np.mean(notears_shds):
                    better_count += 1
        
        criteria['sample_efficiency'] = {
            'passed': better_count >= 3,
            'better_mechanisms': better_count,
            'required': 3
        }
    
    # Criterion 2: Runtime < 5 min for n=50
    if 'spiced' in results_dict:
        runtimes = []
        for config, agg in results_dict['spiced'].items():
            if config[0] == 50 and agg['runtime_mean']:
                runtimes.append(agg['runtime_mean'])
        
        if runtimes:
            median_runtime = np.median(runtimes)
            criteria['scalability'] = {
                'passed': median_runtime < 300,  # 5 minutes
                'median_runtime': median_runtime,
                'threshold': 300
            }
    
    # Criterion 3: Sachs SHD < 10
    if 'spiced' in sachs_results:
        sachs_shds = [r['shd'] for r in sachs_results['spiced'] if 'shd' in r]
        if sachs_shds:
            median_sachs_shd = np.median(sachs_shds)
            criteria['sachs_accuracy'] = {
                'passed': median_sachs_shd < 10,
                'median_shd': median_sachs_shd,
                'threshold': 10
            }
    
    return criteria


def main():
    """Main aggregation function."""
    print("Aggregating results...")
    
    # Load results for all methods
    methods = ['pc', 'notears', 'golem', 'spiced']
    results_dict = {}
    
    for method in methods:
        results = load_results(method)
        if results:
            results_dict[method] = aggregate_by_configuration(results)
            print(f"Loaded {method}: {len(results)} results, {len(results_dict[method])} configs")
        else:
            print(f"No results for {method}")
    
    # Load Sachs results
    sachs_results = {}
    for method in methods:
        sachs = load_sachs_results(method)
        if sachs:
            sachs_results[method] = sachs
            print(f"Loaded {method} Sachs: {len(sachs)} results")
    
    # Create summary table
    summary_table = create_summary_table(results_dict)
    
    # Perform statistical tests
    statistical_tests = perform_statistical_tests(results_dict)
    
    # Check success criteria
    success_criteria = check_success_criteria(results_dict, sachs_results)
    
    # Save aggregated results
    output = {
        'summary_table': summary_table,
        'by_configuration': {method: {str(k): v for k, v in agg.items()} 
                            for method, agg in results_dict.items()},
        'statistical_tests': statistical_tests,
        'success_criteria': success_criteria,
        'sachs_results': {m: [{'shd': r.get('shd'), 'tpr': r.get('tpr'), 'fdr': r.get('fdr')} 
                             for r in res] for m, res in sachs_results.items()}
    }
    
    with open(os.path.join(PROJECT_ROOT, "results.json"), 'w') as f:
        json.dump(output, f, indent=2, default=float)
    
    print("\nResults saved to results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("SUCCESS CRITERIA CHECK")
    print("="*60)
    for criterion, result in success_criteria.items():
        status = "PASS" if result['passed'] else "FAIL"
        print(f"{criterion}: {status}")
        for key, val in result.items():
            if key != 'passed':
                print(f"  {key}: {val}")
    
    return output


if __name__ == "__main__":
    main()
