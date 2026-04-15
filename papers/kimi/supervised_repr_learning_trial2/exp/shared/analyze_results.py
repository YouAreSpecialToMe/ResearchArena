"""
Analyze all experimental results and generate summary tables and figures.
"""
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd


def load_all_results(results_dir='./results'):
    """Load all result JSON files."""
    results = {}
    pattern = os.path.join(results_dir, '*.json')
    
    for path in glob.glob(pattern):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                key = os.path.basename(path).replace('.json', '')
                results[key] = data
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return results


def aggregate_results(results):
    """Aggregate results by method, dataset, and noise rate."""
    aggregated = {}
    
    for key, data in results.items():
        if 'method' not in data or 'dataset' not in data:
            continue
        
        method = data['method']
        dataset = data['dataset']
        noise_rate = data.get('noise_rate', 0)
        
        group_key = f"{method}_{dataset}_n{int(noise_rate*100)}"
        
        if group_key not in aggregated:
            aggregated[group_key] = {
                'method': method,
                'dataset': dataset,
                'noise_rate': noise_rate,
                'accuracies': [],
                'seeds': [],
                'runtimes': []
            }
        
        aggregated[group_key]['accuracies'].append(data.get('final_accuracy', 0))
        aggregated[group_key]['seeds'].append(data.get('seed', 0))
        aggregated[group_key]['runtimes'].append(data.get('runtime_minutes', 0))
    
    # Compute statistics
    for key in aggregated:
        accs = aggregated[key]['accuracies']
        aggregated[key]['mean_acc'] = np.mean(accs)
        aggregated[key]['std_acc'] = np.std(accs)
        aggregated[key]['min_acc'] = np.min(accs)
        aggregated[key]['max_acc'] = np.max(accs)
        aggregated[key]['mean_runtime'] = np.mean(aggregated[key]['runtimes'])
    
    return aggregated


def create_summary_table(aggregated, output_path='./results/summary_table.csv'):
    """Create summary table of all results."""
    rows = []
    
    for key, data in aggregated.items():
        rows.append({
            'Method': data['method'],
            'Dataset': data['dataset'],
            'Noise Rate': f"{data['noise_rate']*100:.0f}%",
            'Mean Acc (%)': f"{data['mean_acc']:.2f}",
            'Std Acc (%)': f"{data['std_acc']:.2f}",
            'Seeds': ','.join(map(str, data['seeds'])),
            'Runtime (min)': f"{data['mean_runtime']:.1f}"
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['Dataset', 'Noise Rate', 'Method'])
    df.to_csv(output_path, index=False)
    print(f"Summary table saved to {output_path}")
    return df


def create_comparison_plot(aggregated, output_path='./figures/main_results.png'):
    """Create bar plot comparing methods."""
    # Filter for CIFAR-10 and CIFAR-100 at 40% noise
    methods = ['supcon', 'supcon_lr', 'laser_scl']
    method_names = ['Vanilla SupCon', 'SupCon+LR', 'LASER-SCL']
    datasets = ['cifar10', 'cifar100']
    dataset_names = ['CIFAR-10', 'CIFAR-100']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (dataset, ds_name) in enumerate(zip(datasets, dataset_names)):
        ax = axes[idx]
        means = []
        stds = []
        labels = []
        
        for method, name in zip(methods, method_names):
            key = f"{method}_{dataset}_n40"
            if key in aggregated:
                means.append(aggregated[key]['mean_acc'])
                stds.append(aggregated[key]['std_acc'])
                labels.append(name)
        
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'{ds_name} (40% Noise)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, max(means) * 1.2])
        
        # Add value labels on bars
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 1, f'{m:.1f}±{s:.1f}', 
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Main results plot saved to {output_path}")
    plt.close()


def create_ablation_plot(aggregated, output_path='./figures/ablation_results.png'):
    """Create ablation study plot."""
    ablations = ['laser_scl', 'ablation_no_curriculum', 'ablation_no_elp', 'ablation_static']
    ablation_names = ['Full LASER-SCL', 'No Curriculum', 'No ELP', 'Static Weighting']
    
    means = []
    stds = []
    valid_names = []
    
    for method, name in zip(ablations, ablation_names):
        key = f"{method}_cifar100_n40"
        if key in aggregated:
            means.append(aggregated[key]['mean_acc'])
            stds.append(aggregated[key]['std_acc'])
            valid_names.append(name)
    
    if len(means) == 0:
        print("No ablation results found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(valid_names))
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=colors[:len(means)])
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study on CIFAR-100 (40% Noise)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(means) * 1.15])
    
    # Add value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.5, f'{m:.1f}±{s:.1f}', 
               ha='center', va='bottom', fontsize=10)
    
    # Add difference annotations
    if len(means) > 1:
        for i in range(1, len(means)):
            diff = means[0] - means[i]
            ax.annotate(f'-{diff:.2f}%', xy=(i, means[i]/2), 
                       ha='center', fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Ablation plot saved to {output_path}")
    plt.close()


def perform_statistical_tests(aggregated, output_path='./results/statistical_tests.json'):
    """Perform statistical significance tests."""
    # Load raw results for paired t-test
    results_dir = './results'
    
    tests = {}
    
    # Compare LASER-SCL vs SupCon+LR on CIFAR-100 40% noise
    laser_files = glob.glob(os.path.join(results_dir, 'laser_scl_cifar100_n40_s*.json'))
    lr_files = glob.glob(os.path.join(results_dir, 'supcon_lr_cifar100_n40_s*.json'))
    
    if laser_files and lr_files:
        laser_accs = []
        lr_accs = []
        
        for f in laser_files:
            with open(f) as fp:
                laser_accs.append(json.load(fp)['final_accuracy'])
        for f in lr_files:
            with open(f) as fp:
                lr_accs.append(json.load(fp)['final_accuracy'])
        
        if len(laser_accs) == len(lr_accs):
            t_stat, p_value = stats.ttest_rel(laser_accs, lr_accs)
            tests['laser_scl_vs_supcon_lr_cifar100_n40'] = {
                'laser_scl_mean': np.mean(laser_accs),
                'supcon_lr_mean': np.mean(lr_accs),
                'difference': np.mean(laser_accs) - np.mean(lr_accs),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
    
    with open(output_path, 'w') as f:
        json.dump(tests, f, indent=2)
    
    print(f"Statistical tests saved to {output_path}")
    return tests


def create_elp_validation_plot(elp_results_path='./results/elp_validation_cifar10_n40.json',
                               output_path='./figures/elp_validation.png'):
    """Create ELP validation trajectory plot."""
    if not os.path.exists(elp_results_path):
        print(f"ELP validation results not found at {elp_results_path}")
        return
    
    with open(elp_results_path, 'r') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trajectory plot
    ax = axes[0]
    clean_traj = data.get('clean_avg_trajectory', [])
    noisy_traj = data.get('noisy_avg_trajectory', [])
    
    if clean_traj and noisy_traj:
        epochs = range(len(clean_traj))
        ax.plot(epochs, clean_traj, label='Clean Samples', linewidth=2, color='green')
        ax.plot(epochs, noisy_traj, label='Noisy Samples', linewidth=2, color='red', linestyle='--')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Loss', fontsize=12)
        ax.set_title('Loss Trajectories: Clean vs Noisy Samples', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
    
    # AUC comparison bar
    ax = axes[1]
    elp_auc = data.get('elp_auc', 0)
    loss_auc = data.get('loss_auc', 0)
    
    bars = ax.bar(['Loss-based', 'ELP-based'], [loss_auc, elp_auc], 
                  color=['#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Clean/Noisy Discrimination', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add delta text
    delta = elp_auc - loss_auc
    ax.text(0.5, 0.6, f'Δ = +{delta:.3f}', transform=ax.transAxes,
           ha='center', fontsize=12, fontweight='bold', color='green')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"ELP validation plot saved to {output_path}")
    plt.close()


def main():
    print("Loading results...")
    results = load_all_results('./results')
    print(f"Loaded {len(results)} result files")
    
    print("\nAggregating results...")
    aggregated = aggregate_results(results)
    
    print("\nCreating summary table...")
    create_summary_table(aggregated)
    
    print("\nCreating plots...")
    create_comparison_plot(aggregated)
    create_ablation_plot(aggregated)
    
    print("\nPerforming statistical tests...")
    perform_statistical_tests(aggregated)
    
    print("\nCreating ELP validation plot...")
    create_elp_validation_plot()
    
    print("\n=== Analysis Complete ===")
    
    # Print key findings
    print("\nKey Results:")
    for key in ['laser_scl_cifar100_n40', 'supcon_lr_cifar100_n40', 'supcon_cifar100_n40']:
        if key in aggregated:
            data = aggregated[key]
            print(f"  {key}: {data['mean_acc']:.2f} ± {data['std_acc']:.2f}%")


if __name__ == '__main__':
    main()
