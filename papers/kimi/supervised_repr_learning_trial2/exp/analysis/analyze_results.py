"""
Analyze and aggregate results from all experiments.
"""
import os
import sys
import json
import numpy as np
import glob
from collections import defaultdict

def load_all_results():
    """Load all result JSON files from experiment directories."""
    results = []
    
    # Search for results in all experiment directories
    for results_file in glob.glob('exp/**/results*.json', recursive=True):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                data['_source_file'] = results_file
                results.append(data)
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
    
    return results

def aggregate_by_method_dataset(results):
    """Aggregate results by method and dataset configuration."""
    
    # Group by (method, dataset, noise_type, noise_ratio)
    grouped = defaultdict(list)
    
    for r in results:
        key = (
            r.get('method', 'Unknown'),
            r.get('dataset', 'Unknown'),
            r.get('noise_type', 'clean'),
            r.get('noise_ratio', 0.0)
        )
        grouped[key].append(r)
    
    # Compute statistics
    aggregated = []
    for key, group in sorted(grouped.items()):
        method, dataset, noise_type, noise_ratio = key
        
        # Get metric (linear_accuracy for contrastive, test_accuracy for CE)
        accs = []
        for r in group:
            if 'linear_accuracy' in r:
                accs.append(r['linear_accuracy'])
            elif 'test_accuracy' in r:
                accs.append(r['test_accuracy'])
        
        if len(accs) > 0:
            aggregated.append({
                'method': method,
                'dataset': dataset,
                'noise_type': noise_type,
                'noise_ratio': noise_ratio,
                'n_seeds': len(accs),
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)) if len(accs) > 1 else 0.0,
                'accuracy_min': float(np.min(accs)),
                'accuracy_max': float(np.max(accs)),
                'all_accuracies': accs
            })
    
    return aggregated

def generate_latex_table(aggregated):
    """Generate LaTeX table for paper."""
    
    # Filter for CIFAR-100 results
    cifar100_results = [r for r in aggregated if r['dataset'] == 'cifar100']
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Linear evaluation accuracy on CIFAR-100 (mean \\pm std across 3 seeds)}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Method & Clean & 20\\% Symmetric & 40\\% Symmetric \\\\")
    print("\\midrule")
    
    methods = ['CrossEntropy', 'SupCon', 'GC-SCL']
    
    for method in methods:
        row = [method]
        for noise_ratio in [0.0, 0.2, 0.4]:
            # Find matching result
            matches = [r for r in cifar100_results 
                      if r['method'] == method and r['noise_ratio'] == noise_ratio]
            if matches:
                r = matches[0]
                if r['n_seeds'] > 1:
                    row.append(f"{r['accuracy_mean']:.2f} $\\pm$ {r['accuracy_std']:.2f}")
                else:
                    row.append(f"{r['accuracy_mean']:.2f}")
            else:
                row.append("--")
        print(" & ".join(row) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\label{tab:main_results}")
    print("\\end{table}")

def generate_summary_json(aggregated):
    """Generate summary JSON for the paper."""
    
    summary = {
        'main_results': aggregated,
        'success_criteria': {
            'criterion_1': {
                'description': 'GC-SCL achieves >=1% higher accuracy than SupCon on CIFAR-100 clean',
                'verified': False,
                'details': {}
            },
            'criterion_2': {
                'description': 'GC-SCL shows >=2% improvement over SupCon on CIFAR-100 with 20% label noise',
                'verified': False,
                'details': {}
            }
        }
    }
    
    # Check criteria
    supcon_clean = [r for r in aggregated if r['method'] == 'SupCon' and r['dataset'] == 'cifar100' and r['noise_ratio'] == 0.0]
    gcscl_clean = [r for r in aggregated if r['method'] == 'GC-SCL' and r['dataset'] == 'cifar100' and r['noise_ratio'] == 0.0]
    
    if supcon_clean and gcscl_clean:
        improvement_clean = gcscl_clean[0]['accuracy_mean'] - supcon_clean[0]['accuracy_mean']
        summary['success_criteria']['criterion_1']['verified'] = improvement_clean >= 1.0
        summary['success_criteria']['criterion_1']['details'] = {
            'supcon_acc': supcon_clean[0]['accuracy_mean'],
            'gcscl_acc': gcscl_clean[0]['accuracy_mean'],
            'improvement': improvement_clean
        }
    
    supcon_noisy = [r for r in aggregated if r['method'] == 'SupCon' and r['dataset'] == 'cifar100' and r['noise_ratio'] == 0.2]
    gcscl_noisy = [r for r in aggregated if r['method'] == 'GC-SCL' and r['dataset'] == 'cifar100' and r['noise_ratio'] == 0.2]
    
    if supcon_noisy and gcscl_noisy:
        improvement_noisy = gcscl_noisy[0]['accuracy_mean'] - supcon_noisy[0]['accuracy_mean']
        summary['success_criteria']['criterion_2']['verified'] = improvement_noisy >= 2.0
        summary['success_criteria']['criterion_2']['details'] = {
            'supcon_acc': supcon_noisy[0]['accuracy_mean'],
            'gcscl_acc': gcscl_noisy[0]['accuracy_mean'],
            'improvement': improvement_noisy
        }
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    print("Loading all results...")
    results = load_all_results()
    print(f"Found {len(results)} result files")
    
    if len(results) == 0:
        print("No results found. Run experiments first.")
        return
    
    print("\nAggregating by method and dataset...")
    aggregated = aggregate_by_method_dataset(results)
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for r in aggregated:
        print(f"\n{r['method']} - {r['dataset']} - {r['noise_type']} (noise={r['noise_ratio']})")
        print(f"  Accuracy: {r['accuracy_mean']:.2f} ± {r['accuracy_std']:.2f} (n={r['n_seeds']})")
    
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    generate_latex_table(aggregated)
    
    print("\n" + "="*80)
    print("GENERATING SUMMARY JSON")
    print("="*80)
    summary = generate_summary_json(aggregated)
    print(f"Summary saved to results/summary.json")
    
    # Print success criteria status
    print("\n" + "="*80)
    print("SUCCESS CRITERIA VERIFICATION")
    print("="*80)
    for criterion_name, criterion_data in summary['success_criteria'].items():
        status = "✓ VERIFIED" if criterion_data['verified'] else "✗ NOT VERIFIED"
        print(f"\n{criterion_name}: {status}")
        print(f"  Description: {criterion_data['description']}")
        if criterion_data['details']:
            print(f"  Details: {criterion_data['details']}")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.chdir('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_02')
    main()
