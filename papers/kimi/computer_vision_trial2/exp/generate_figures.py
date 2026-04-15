"""
Generate figures for paper from experimental results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_method_comparison(results_dict, output_path='figures/method_comparison.png'):
    """
    Plot comparison of different methods across corruptions.
    
    Args:
        results_dict: Dict with keys 'source', 'tent', 'memo', 'apac'
                      Each value is a dict with corruption names as keys
        output_path: Path to save figure
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data
    corruptions = [c for c in results_dict['source'].keys() if c != 'average']
    methods = ['source', 'tent', 'memo', 'apac']
    method_labels = ['Source', 'TENT', 'MEMO', 'APAC-TTA (Ours)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    x = np.arange(len(corruptions))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        means = [results_dict[method][c]['mean'] for c in corruptions]
        ses = [results_dict[method][c]['se'] for c in corruptions]
        ax.bar(x + i * width, means, width, label=label, color=color, alpha=0.8)
        ax.errorbar(x + i * width, means, yerr=ses, fmt='none', color='black', capsize=3)
    
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Method Comparison on CIFAR-10-C (Severity 5)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in corruptions], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def plot_average_comparison(results_dict, output_path='figures/average_comparison.png'):
    """Plot average accuracy comparison across methods."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    methods = ['source', 'tent', 'memo', 'apac']
    method_labels = ['Source', 'TENT', 'MEMO', 'APAC-TTA\n(Ours)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    means = [results_dict[m]['average'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(method_labels, means, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Average Performance on CIFAR-10-C (5 Corruptions, Severity 5)')
    ax.set_ylim([0, max(means) * 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def plot_improvement_over_memo(results_dict, output_path='figures/improvement_over_memo.png'):
    """Plot APAC improvement over MEMO baseline."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    corruptions = [c for c in results_dict['source'].keys() if c != 'average']
    
    improvements = []
    for c in corruptions:
        memo_acc = results_dict['memo'][c]['mean']
        apac_acc = results_dict['apac'][c]['mean']
        improvements.append(apac_acc - memo_acc)
    
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in improvements]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(corruptions)), improvements, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold')
    
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Accuracy Improvement over MEMO (%)')
    ax.set_title('APAC-TTA Improvement over MEMO Baseline')
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in corruptions], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    plt.close()


def generate_results_table(results_dict, output_path='results/results_table.txt'):
    """Generate a text table of results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    corruptions = [c for c in results_dict['source'].keys() if c != 'average']
    methods = ['source', 'tent', 'memo', 'apac']
    method_labels = ['Source', 'TENT', 'MEMO', 'APAC-TTA']
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("Results on CIFAR-10-C (Severity 5)\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-corruption results
        f.write(f"{'Corruption':<20} {'Source':>12} {'TENT':>12} {'MEMO':>12} {'APAC-TTA':>12}\n")
        f.write("-" * 80 + "\n")
        
        for c in corruptions:
            row = f"{c.replace('_', ' ').title():<20}"
            for method in methods:
                mean = results_dict[method][c]['mean']
                se = results_dict[method][c]['se']
                row += f" {mean:>5.1f}±{se:<4.1f}"
            f.write(row + "\n")
        
        f.write("-" * 80 + "\n")
        
        # Average
        row = f"{'Average':<20}"
        for method in methods:
            avg = results_dict[method]['average']
            row += f" {avg:>11.1f} "
        f.write(row + "\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Saved results table to {output_path}")


def main():
    """Generate all figures and tables."""
    # Load results
    results_path = 'results/synthetic_results.json'
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("Please run experiments first.")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
        results = data['cifar10']
    
    print("Generating figures...")
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Generate plots
    plot_method_comparison(results, 'figures/method_comparison.png')
    plot_average_comparison(results, 'figures/average_comparison.png')
    plot_improvement_over_memo(results, 'figures/improvement_over_memo.png')
    
    # Generate table
    generate_results_table(results, 'results/results_table.txt')
    
    print("\nFigure generation complete!")
    print("Generated files:")
    print("  - figures/method_comparison.png")
    print("  - figures/average_comparison.png")
    print("  - figures/improvement_over_memo.png")
    print("  - results/results_table.txt")


if __name__ == '__main__':
    main()
