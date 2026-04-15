"""
Generate publication-quality figures for the paper.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def load_results():
    """Load experimental results."""
    results_dir = Path('results')
    
    with open(results_dir / 'main_experiment.json') as f:
        main_results = json.load(f)
    
    with open(results_dir / 'ablation_study.json') as f:
        ablation_results = json.load(f)
    
    return main_results, ablation_results


def figure_performance_vs_sample_size(main_results, output_dir):
    """Figure 1: Performance vs Sample Size"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algorithms = sorted(set(r['algorithm'] for r in main_results))
    sample_sizes = sorted(set(r['n_samples'] for r in main_results))
    
    for alg in algorithms:
        means = []
        stds = []
        
        for n in sample_sizes:
            pc_f1_values = [
                r['pc_f1'] for r in main_results 
                if r['algorithm'] == alg and r['n_samples'] == n
            ]
            means.append(np.mean(pc_f1_values))
            stds.append(np.std(pc_f1_values))
        
        label = 'AIT-LCD' if alg == 'ait-lcd' else alg.upper()
        marker = 'o' if alg == 'ait-lcd' else 's'
        linewidth = 2.5 if alg == 'ait-lcd' else 1.5
        
        ax.errorbar(sample_sizes, means, yerr=stds, label=label, 
                   marker=marker, linewidth=linewidth, capsize=3, markersize=6)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('PC F1 Score', fontsize=12)
    ax.set_title('Performance vs Sample Size', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_performance_vs_sample_size.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_performance_vs_sample_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 1: Performance vs Sample Size")


def figure_network_comparison(main_results, output_dir):
    """Figure 2: Network-wise Comparison at n=500"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = sorted(set(r['algorithm'] for r in main_results))
    networks = sorted(set(r['network'] for r in main_results))
    
    x = np.arange(len(networks))
    width = 0.15
    
    for i, alg in enumerate(algorithms):
        means = []
        
        for net in networks:
            pc_f1_values = [
                r['pc_f1'] for r in main_results 
                if r['algorithm'] == alg and r['network'] == net and r['n_samples'] == 500
            ]
            means.append(np.mean(pc_f1_values) if pc_f1_values else 0)
        
        label = 'AIT-LCD' if alg == 'ait-lcd' else alg.upper()
        ax.bar(x + i*width, means, width, label=label, 
               alpha=0.8 if alg == 'ait-lcd' else 0.6)
    
    ax.set_xlabel('Network', fontsize=12)
    ax.set_ylabel('PC F1 Score', fontsize=12)
    ax.set_title('Network-wise Performance Comparison (n=500)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels([n.capitalize() for n in networks])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_network_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_network_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 2: Network-wise Comparison")


def figure_ablation_impact(ablation_results, output_dir):
    """Figure 3: Ablation Study Impact"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    variants = {
        'full': 'Full AIT-LCD',
        'no_bias_correction': 'No Bias Correction',
        'fixed_threshold': 'Fixed Threshold',
        'no_adaptive_no_bias': 'No Adaptive + No Bias'
    }
    
    means = []
    labels = []
    
    for key, label in variants.items():
        values = [r['pc_f1'] for r in ablation_results if r['variant'] == key]
        if values:
            means.append(np.mean(values))
            labels.append(label)
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']
    bars = ax.bar(range(len(means)), means, color=colors, alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('PC F1 Score', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(means) * 1.2])
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_ablation_impact.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_ablation_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 3: Ablation Study Impact")


def figure_runtime_comparison(main_results, output_dir):
    """Figure 4: Runtime Comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    algorithms = sorted(set(r['algorithm'] for r in main_results))
    
    data_for_plot = []
    labels = []
    
    for alg in algorithms:
        runtimes = [r['runtime'] for r in main_results if r['algorithm'] == alg]
        if runtimes:
            data_for_plot.append(runtimes)
            labels.append('AIT-LCD' if alg == 'ait-lcd' else alg.upper())
    
    bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison Across Algorithms', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_runtime_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_runtime_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 4: Runtime Comparison")


def figure_threshold_adaptation(output_dir):
    """Figure 5: Threshold Adaptation Visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Default parameters
    alpha, beta = 0.1, 10
    
    # Plot 1: Threshold vs sample size for different k
    ax = axes[0]
    sample_sizes = np.logspace(1, 4, 100)
    k_values = [1, 2, 5, 10]
    
    for k in k_values:
        thresholds = [
            alpha * np.sqrt(k/n) * np.log1p(n/(k*beta)) 
            for n in sample_sizes
        ]
        ax.plot(sample_sizes, thresholds, label=f'k={k}', linewidth=2)
    
    ax.set_xlabel('Sample Size (n)', fontsize=12)
    ax.set_ylabel('Threshold τ(n,k)', fontsize=12)
    ax.set_title('Adaptive Threshold vs Sample Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Plot 2: Threshold vs k for different n
    ax = axes[1]
    k_values = np.arange(1, 21)
    n_values = [100, 500, 1000, 5000]
    
    for n in n_values:
        thresholds = [
            alpha * np.sqrt(k/n) * np.log1p(n/(k*beta)) 
            for k in k_values
        ]
        ax.plot(k_values, thresholds, label=f'n={n}', linewidth=2, marker='o', markersize=3)
    
    ax.set_xlabel('Conditioning Set Size (k)', fontsize=12)
    ax.set_ylabel('Threshold τ(n,k)', fontsize=12)
    ax.set_title('Adaptive Threshold vs Conditioning Set Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_threshold_adaptation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_threshold_adaptation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved Figure 5: Threshold Adaptation")


def main():
    print("="*60)
    print("Generating Figures for AIT-LCD Paper")
    print("="*60)
    print()
    
    # Create output directory
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Load results
    main_results, ablation_results = load_results()
    
    # Generate figures
    figure_performance_vs_sample_size(main_results, figures_dir)
    figure_network_comparison(main_results, figures_dir)
    figure_ablation_impact(ablation_results, figures_dir)
    figure_runtime_comparison(main_results, figures_dir)
    figure_threshold_adaptation(figures_dir)
    
    print("\n" + "="*60)
    print(f"All figures saved to {figures_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
