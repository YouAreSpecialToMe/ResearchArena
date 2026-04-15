#!/usr/bin/env python3
"""
Generate figures for LGSA paper.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


def load_results():
    """Load all results."""
    results = {}
    
    if os.path.exists('results/metrics/lgsa_results.json'):
        with open('results/metrics/lgsa_results.json', 'r') as f:
            results['lgsa'] = json.load(f)
    
    if os.path.exists('results/metrics/ablation_results.json'):
        with open('results/metrics/ablation_results.json', 'r') as f:
            results['ablation'] = json.load(f)
    
    if os.path.exists('results/metrics/complexity_analysis.json'):
        with open('results/metrics/complexity_analysis.json', 'r') as f:
            results['complexity'] = json.load(f)
    
    return results


def plot_auc_comparison(results, save_dir='results/figures'):
    """Plot AUC comparison across methods."""
    os.makedirs(save_dir, exist_ok=True)
    
    lgsa_results = results.get('lgsa', [])
    
    # Group by dataset and model
    configs = {}
    for r in lgsa_results:
        key = f"{r['dataset']}\n{r['model']}"
        if key not in configs:
            configs[key] = []
        configs[key].append(r['auc'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(configs))
    means = [np.mean(v) for v in configs.values()]
    stds = [np.std(v) for v in configs.values()]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('LGSA Verification Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs.keys(), rotation=45, ha='right')
    ax.axhline(y=0.85, color='r', linestyle='--', label='Success Threshold (0.85)')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure_1_auc_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure_1_auc_comparison.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/figure_1_auc_comparison.pdf")


def plot_ablation_metrics(results, save_dir='results/figures'):
    """Plot ablation study results."""
    os.makedirs(save_dir, exist_ok=True)
    
    ablation = results.get('ablation', {})
    individual = ablation.get('individual_metrics', [])
    
    if not individual:
        print("No ablation data available")
        return
    
    # Group by config
    configs = {}
    for entry in individual:
        name = entry['config']
        if name not in configs:
            configs[name] = []
        configs[name].append(entry['auc'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(configs))
    means = [np.mean(configs[k]) for k in configs.keys()]
    stds = [np.std(configs[k]) for k in configs.keys()]
    
    colors = ['#ff7f0e' if 'only' in k else '#2ca02c' if 'all' in k else '#1f77b4' 
              for k in configs.keys()]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xlabel('Metric Configuration', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Ablation: Individual vs Combined Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([k.replace('_', '\n') for k in configs.keys()], fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure_2_ablation_metrics.pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure_2_ablation_metrics.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/figure_2_ablation_metrics.pdf")


def plot_verification_time(results, save_dir='results/figures'):
    """Plot verification time comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    lgsa_results = results.get('lgsa', [])
    
    if not lgsa_results:
        print("No timing data available")
        return
    
    # Get average verification times
    times_by_config = {}
    for r in lgsa_results:
        key = f"{r['dataset']}\n{r['model']}"
        if key not in times_by_config:
            times_by_config[key] = []
        times_by_config[key].append(r['verify_time'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(times_by_config))
    means = [np.mean(v) for v in times_by_config.values()]
    stds = [np.std(v) for v in times_by_config.values()]
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='coral', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Verification Time (seconds)', fontsize=12)
    ax.set_title('LGSA Verification Time (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(times_by_config.keys(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure_3_verification_time.pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure_3_verification_time.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/figure_3_verification_time.pdf")


def plot_complexity_scaling(results, save_dir='results/figures'):
    """Plot complexity scaling with model size."""
    os.makedirs(save_dir, exist_ok=True)
    
    complexity = results.get('complexity', [])
    
    if not complexity:
        print("No complexity data available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = [c['model'] for c in complexity]
    params = [c['n_params'] / 1e6 for c in complexity]  # In millions
    times = [c['time_100_samples'] for c in complexity]
    
    ax.scatter(params, times, s=200, alpha=0.7, c='darkblue')
    
    for i, model in enumerate(models):
        ax.annotate(model, (params[i], times[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Model Parameters (Millions)', fontsize=12)
    ax.set_ylabel('Time for 100 Samples (seconds)', fontsize=12)
    ax.set_title('LGSA Scalability', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/figure_4_complexity_scaling.pdf', bbox_inches='tight')
    plt.savefig(f'{save_dir}/figure_4_complexity_scaling.png', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_dir}/figure_4_complexity_scaling.pdf")


def generate_table_1(results, save_dir='results/tables'):
    """Generate main results table."""
    os.makedirs(save_dir, exist_ok=True)
    
    lgsa_results = results.get('lgsa', [])
    
    if not lgsa_results:
        print("No results for table")
        return
    
    # Group by configuration
    configs = {}
    for r in lgsa_results:
        key = (r['dataset'], r['model'], r['unlearn_method'])
        if key not in configs:
            configs[key] = {'auc': [], 'tpr': [], 'time': []}
        configs[key]['auc'].append(r['auc'])
        configs[key]['tpr'].append(r['tpr_at_1fpr'])
        configs[key]['time'].append(r['verify_time'])
    
    # Create table
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{LGSA Verification Performance}")
    lines.append("\\begin{tabular}{lllc@{\\pm}cc@{\\pm}cc@{\\pm}c}")
    lines.append("\\toprule")
    lines.append("Dataset & Model & Unlearn Method & \\multicolumn{2}{c}{AUC} & \\multicolumn{2}{c}{TPR@1\\%FPR} & \\multicolumn{2}{c}{Time (s)} \\\\")
    lines.append("\\midrule")
    
    for key, values in configs.items():
        dataset, model, method = key
        auc_mean = np.mean(values['auc'])
        auc_std = np.std(values['auc'])
        tpr_mean = np.mean(values['tpr'])
        tpr_std = np.std(values['tpr'])
        time_mean = np.mean(values['time'])
        time_std = np.std(values['time'])
        
        lines.append(f"{dataset} & {model} & {method} & {auc_mean:.3f} & {auc_std:.3f} & "
                    f"{tpr_mean:.3f} & {tpr_std:.3f} & {time_mean:.2f} & {time_std:.2f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\end{table}")
    
    table_text = "\n".join(lines)
    
    with open(f'{save_dir}/table_1_main_results.tex', 'w') as f:
        f.write(table_text)
    
    print(f"Saved: {save_dir}/table_1_main_results.tex")


def main():
    """Generate all figures and tables."""
    print("Generating figures and tables...")
    
    results = load_results()
    
    if not results:
        print("No results found. Run experiments first.")
        return
    
    plot_auc_comparison(results)
    plot_ablation_metrics(results)
    plot_verification_time(results)
    plot_complexity_scaling(results)
    generate_table_1(results)
    
    print("\nAll figures and tables generated successfully!")


if __name__ == '__main__':
    main()
