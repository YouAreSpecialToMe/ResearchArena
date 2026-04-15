"""
Generate paper figures from experiment results.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_results():
    """Load experiment results."""
    with open('results.json', 'r') as f:
        return json.load(f)


def load_main_experiment():
    """Load main experiment raw results."""
    try:
        with open('results/main_experiment.json', 'r') as f:
            return json.load(f)
    except:
        return []


def load_ablation_results():
    """Load ablation study results."""
    try:
        with open('results/ablation_study.json', 'r') as f:
            return json.load(f)
    except:
        return []


def figure_1_performance_vs_sample_size(main_results, output_dir):
    """
    Figure 1: Performance vs Sample Size
    Line plot showing PC F1 vs n for each algorithm.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    sample_sizes = [100, 200, 500, 1000]
    
    for algo in algorithms:
        means = []
        stds = []
        
        for n in sample_sizes:
            f1_scores = [r['pc_f1'] for r in main_results 
                        if r['algorithm'] == algo and r['n_samples'] == n]
            if f1_scores:
                means.append(np.mean(f1_scores))
                stds.append(np.std(f1_scores))
            else:
                means.append(0)
                stds.append(0)
        
        ax.errorbar(sample_sizes, means, yerr=stds, marker='o', 
                   label=algo, capsize=3, linewidth=2)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('PC F1 Score', fontsize=12)
    ax.set_title('Performance vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_performance_vs_sample_size.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_performance_vs_sample_size.png', bbox_inches='tight')
    plt.close()
    print("Generated Figure 1: Performance vs Sample Size")


def figure_2_network_wise_comparison(main_results, output_dir):
    """
    Figure 2: Network-wise Comparison
    Grouped bar chart showing PC F1 for each algorithm on each network.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    networks = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
    
    x = np.arange(len(networks))
    width = 0.15
    
    for i, algo in enumerate(algorithms):
        means = []
        
        for network in networks:
            f1_scores = [r['pc_f1'] for r in main_results 
                        if r['algorithm'] == algo and r['network'] == network]
            if f1_scores:
                means.append(np.mean(f1_scores))
            else:
                means.append(0)
        
        offset = (i - 2) * width
        ax.bar(x + offset, means, width, label=algo)
    
    ax.set_xlabel('Network', fontsize=12)
    ax.set_ylabel('PC F1 Score', fontsize=12)
    ax.set_title('Network-wise Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in networks])
    ax.legend(loc='best', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_network_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_network_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated Figure 2: Network-wise Comparison")


def figure_3_ablation_study(ablation_results, output_dir):
    """
    Figure 3: Ablation Study Impact
    Bar chart showing F1 scores for Full AIT-LCD and each ablation variant.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = ['Full AIT-LCD', 'No Bias Correction', 'Fixed Threshold', 'No Adaptations']
    
    means = []
    stds = []
    
    for variant in variants:
        f1_scores = [r['pc_f1'] for r in ablation_results if r['variant'] == variant]
        if f1_scores:
            means.append(np.mean(f1_scores))
            stds.append(np.std(f1_scores))
        else:
            means.append(0)
            stds.append(0)
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    bars = ax.bar(range(len(variants)), means, yerr=stds, capsize=5, 
                  color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Variant', fontsize=12)
    ax.set_ylabel('PC F1 Score', fontsize=12)
    ax.set_title('Ablation Study: Impact of Components', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_ablation_study.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_ablation_study.png', bbox_inches='tight')
    plt.close()
    print("Generated Figure 3: Ablation Study")


def figure_4_runtime_comparison(main_results, output_dir):
    """
    Figure 4: Runtime Comparison
    Box plot comparing runtime distribution across algorithms.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algorithms = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']
    
    runtime_data = []
    labels = []
    
    for algo in algorithms:
        runtimes = [r['runtime'] for r in main_results if r['algorithm'] == algo and r['runtime'] > 0]
        if runtimes:
            runtime_data.append(runtimes)
            labels.append(algo)
    
    bp = ax.boxplot(runtime_data, labels=labels, patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_runtime_comparison.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_runtime_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated Figure 4: Runtime Comparison")


def figure_5_threshold_adaptation(output_dir):
    """
    Figure 5: Threshold Adaptation
    Visualize how tau(n,k) varies with n and k.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Selected parameters (from pilot study)
    alpha, beta = 0.2, 10
    
    # Plot 1: tau vs n for different k
    ax = axes[0]
    n_values = np.logspace(1, 4, 100)
    
    for k in [1, 2, 3, 5]:
        tau_values = []
        for n in n_values:
            sqrt_term = np.sqrt(k / n)
            log_term = np.log1p(n / (k * beta))
            tau = alpha * sqrt_term * log_term
            tau_values.append(tau)
        ax.plot(n_values, tau_values, label=f'k={k}', linewidth=2)
    
    ax.set_xlabel('Sample Size (n)', fontsize=11)
    ax.set_ylabel('Adaptive Threshold τ(n,k)', fontsize=11)
    ax.set_title('Threshold vs Sample Size', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: tau vs k for different n
    ax = axes[1]
    k_values = np.arange(1, 11)
    
    for n in [100, 200, 500, 1000]:
        tau_values = []
        for k in k_values:
            sqrt_term = np.sqrt(k / n)
            log_term = np.log1p(n / (k * beta))
            tau = alpha * sqrt_term * log_term
            tau_values.append(tau)
        ax.plot(k_values, tau_values, marker='o', label=f'n={n}', linewidth=2)
    
    ax.set_xlabel('Conditioning Set Size (k)', fontsize=11)
    ax.set_ylabel('Adaptive Threshold τ(n,k)', fontsize=11)
    ax.set_title('Threshold vs Conditioning Set Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_threshold_adaptation.pdf', bbox_inches='tight')
    plt.savefig(output_dir / 'figure5_threshold_adaptation.png', bbox_inches='tight')
    plt.close()
    print("Generated Figure 5: Threshold Adaptation")


def create_table_1_main_results(results, output_dir):
    """Create Table 1: Main Results."""
    main_results = results.get('main_results', {})
    
    rows = []
    rows.append("Algorithm,PC F1 (mean ± std),MB F1 (mean ± std),Runtime (s)")
    
    for algo in ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']:
        if algo in main_results:
            pc = main_results[algo]['pc_f1']
            mb = main_results[algo]['mb_f1']
            rt = main_results[algo]['runtime']
            rows.append(f"{algo},{pc['mean']:.3f} ± {pc['std']:.3f},{mb['mean']:.3f} ± {mb['std']:.3f},{rt['mean']:.2f}")
    
    with open(output_dir / 'table1_main_results.csv', 'w') as f:
        f.write('\n'.join(rows))
    
    print("Generated Table 1: Main Results")


def main():
    """Generate all figures."""
    print("="*60)
    print("Generating Figures")
    print("="*60)
    
    # Create figures directory
    figures_dir = Path('../figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    main_results = load_main_experiment()
    ablation_results = load_ablation_results()
    
    # Generate figures
    if main_results:
        figure_1_performance_vs_sample_size(main_results, figures_dir)
        figure_2_network_wise_comparison(main_results, figures_dir)
        figure_4_runtime_comparison(main_results, figures_dir)
    else:
        print("Warning: No main experiment results found")
    
    if ablation_results:
        figure_3_ablation_study(ablation_results, figures_dir)
    else:
        print("Warning: No ablation study results found")
    
    figure_5_threshold_adaptation(figures_dir)
    
    # Generate tables
    create_table_1_main_results(results, figures_dir)
    
    print()
    print("="*60)
    print("All figures generated successfully!")
    print(f"Figures saved to: {figures_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()
