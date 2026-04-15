"""
Create visualizations for FedSecure-CL results.
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_all_results(results_dir='./results'):
    """Load all experiment results."""
    results = defaultdict(lambda: defaultdict(list))
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, '*_results.json'))
    
    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            exp_name = data.get('experiment', '')
            seed = data.get('seed', 0)
            
            # Parse experiment type
            if 'baseline_fcl_' in exp_name and 'at' not in exp_name and 'dp' not in exp_name:
                method = 'Standard FCL'
            elif 'baseline_fcl_at' in exp_name:
                method = 'FCL + AT'
            elif 'baseline_fcl_dp_at' in exp_name:
                method = 'FCL + DP + AT'
            elif 'baseline_fcl_dp' in exp_name:
                method = 'FCL + DP'
            elif 'fedsecure' in exp_name and 'ablation' not in exp_name:
                method = 'FedSecure-CL'
            elif 'ablation_no_privacy' in exp_name:
                method = 'Ablation: No Privacy Reg'
            elif 'ablation_no_grad_noise' in exp_name:
                method = 'Ablation: No Grad Noise'
            elif 'ablation_no_adv' in exp_name:
                method = 'Ablation: No Adv Train'
            else:
                method = exp_name
            
            dataset = 'CIFAR-100' if 'cifar100' in exp_name else 'CIFAR-10'
            key = f"{method}_{dataset}"
            
            results[key]['linear_accuracy'].append(data.get('linear_accuracy', 0))
            results[key]['total_time'].append(data.get('total_time_seconds', 0))
            results[key]['seeds'].append(seed)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    return results


def aggregate_results(results):
    """Aggregate results by method and dataset."""
    aggregated = {}
    
    for key, data in results.items():
        if len(data['linear_accuracy']) > 0:
            aggregated[key] = {
                'mean_acc': np.mean(data['linear_accuracy']),
                'std_acc': np.std(data['linear_accuracy']),
                'n_seeds': len(data['linear_accuracy']),
                'seeds': data['seeds']
            }
    
    return aggregated


def plot_comparison(aggregated, output_dir='./figures'):
    """Plot method comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate by dataset
    cifar10_data = {k: v for k, v in aggregated.items() if 'CIFAR-10' in k}
    cifar100_data = {k: v for k, v in aggregated.items() if 'CIFAR-100' in k}
    
    # CIFAR-10 comparison
    if cifar10_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = []
        means = []
        stds = []
        
        # Order methods
        method_order = [
            'Standard FCL_CIFAR-10',
            'FCL + DP_CIFAR-10', 
            'FCL + AT_CIFAR-10',
            'FCL + DP + AT_CIFAR-10',
            'FedSecure-CL_CIFAR-10',
            'Ablation: No Privacy Reg_CIFAR-10',
            'Ablation: No Grad Noise_CIFAR-10',
            'Ablation: No Adv Train_CIFAR-10'
        ]
        
        for method_key in method_order:
            if method_key in cifar10_data:
                method_name = method_key.replace('_CIFAR-10', '')
                methods.append(method_name)
                means.append(cifar10_data[method_key]['mean_acc'])
                stds.append(cifar10_data[method_key]['std_acc'])
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                     color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#95a5a6', '#95a5a6', '#95a5a6'])
        
        ax.set_ylabel('Linear Evaluation Accuracy (%)')
        ax.set_title('CIFAR-10: Method Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cifar10_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/cifar10_comparison.png")
    
    # CIFAR-100 comparison
    if cifar100_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = []
        means = []
        stds = []
        
        for key, data in cifar100_data.items():
            methods.append(key.replace('_CIFAR-100', ''))
            means.append(data['mean_acc'])
            stds.append(data['std_acc'])
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        
        ax.set_ylabel('Linear Evaluation Accuracy (%)')
        ax.set_title('CIFAR-100: Method Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cifar100_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/cifar100_comparison.png")


def create_results_table(aggregated, output_dir='./results'):
    """Create a results table."""
    table = []
    table.append("=" * 80)
    table.append("FedSecure-CL: Experimental Results")
    table.append("=" * 80)
    table.append(f"{'Method':<35} {'Dataset':<12} {'Accuracy':<15} {'Seeds'}")
    table.append("-" * 80)
    
    # Sort by method and dataset
    sorted_keys = sorted(aggregated.keys())
    
    for key in sorted_keys:
        data = aggregated[key]
        method, dataset = key.rsplit('_', 1)
        acc_str = f"{data['mean_acc']:.2f} ± {data['std_acc']:.2f}"
        table.append(f"{method:<35} {dataset:<12} {acc_str:<15} {data['n_seeds']}")
    
    table.append("=" * 80)
    
    table_str = '\n'.join(table)
    print('\n' + table_str)
    
    # Save to file
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write(table_str)
    
    return table_str


def create_aggregated_results_json(aggregated, output_dir='./results'):
    """Create aggregated results JSON."""
    results = {
        'aggregated': aggregated,
        'metadata': {
            'timestamp': str(np.datetime64('now')),
            'num_experiments': len(aggregated)
        }
    }
    
    output_path = os.path.join(output_dir, 'aggregated_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved: {output_path}")
    return results


def main():
    print("Loading results...")
    results = load_all_results('./results')
    
    print(f"Found {len(results)} experiment configurations")
    
    print("\nAggregating results...")
    aggregated = aggregate_results(results)
    
    print("\nCreating visualizations...")
    plot_comparison(aggregated, './figures')
    
    print("\nCreating results table...")
    create_results_table(aggregated, './results')
    
    print("\nCreating aggregated results JSON...")
    create_aggregated_results_json(aggregated, './results')
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
