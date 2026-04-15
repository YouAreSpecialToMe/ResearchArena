"""Generate figures from real experimental results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_results():
    """Load aggregated results."""
    with open("results.json") as f:
        return json.load(f)


def plot_accuracy_comparison(data, output_dir):
    """Figure 1: Accuracy comparison across methods."""
    methods = ["vanilla", "entropy_only", "esr", "egl", "bestofn"]
    method_labels = ["Vanilla CoT", "Entropy-Only", "ESR (Ours)", "EGL (Post-hoc)", "Best-of-N"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    results = data["results"]
    
    # Filter to methods that exist
    available_methods = [m for m in methods if m in results]
    available_labels = [method_labels[methods.index(m)] for m in available_methods]
    available_colors = [colors[methods.index(m)] for m in available_methods]
    
    accuracies = [results[m]["accuracy"]["mean"] * 100 for m in available_methods]
    stds = [results[m]["accuracy"]["std"] * 100 for m in available_methods]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(available_labels, accuracies, color=available_colors, alpha=0.8, 
                   yerr=stds, capsize=5, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison Across Methods (GSM8K)')
    ax.set_ylim([0, max(accuracies) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_accuracy_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure1_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure1_accuracy_comparison")


def plot_efficiency_tradeoff(data, output_dir):
    """Figure 2: Accuracy vs Tokens (efficiency frontier)."""
    methods = ["vanilla", "entropy_only", "esr", "egl", "bestofn"]
    method_labels = ["Vanilla CoT", "Entropy-Only", "ESR", "EGL", "Best-of-N"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    results = data["results"]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, method in enumerate(methods):
        if method not in results:
            continue
        acc = results[method]["accuracy"]["mean"] * 100
        tokens = results[method]["tokens"]["mean"]
        ax.scatter(tokens, acc, s=300, c=colors[i], marker=markers[i], 
                   label=method_labels[i], alpha=0.8, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('Avg. Tokens per Problem')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy-Efficiency Tradeoff (GSM8K)')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_efficiency_tradeoff.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure2_efficiency_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure2_efficiency_tradeoff")


def plot_revision_analysis(data, output_dir):
    """Figure 3: Revision rate and effectiveness."""
    methods = ["entropy_only", "esr"]
    method_labels = ["Entropy-Only", "ESR (Ours)"]
    colors = ['#ff7f0e', '#2ca02c']
    
    results = data["results"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Revision rates
    revision_rates = []
    for method in methods:
        if method in results and "revision_rate" in results[method]:
            revision_rates.append(results[method]["revision_rate"]["mean"] * 100)
        else:
            revision_rates.append(0)
    
    bars = ax1.bar(method_labels, revision_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Revision Rate (%)')
    ax1.set_title('Revision Trigger Rate by Method')
    ax1.set_ylim([0, max(revision_rates + [10]) * 1.2])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, rate in zip(bars, revision_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Efficiency (accuracy per 100 tokens)
    efficiency = []
    eff_labels = []
    eff_colors = []
    for i, method in enumerate(methods):
        if method in results:
            acc = results[method]["accuracy"]["mean"] * 100
            tokens = results[method]["tokens"]["mean"]
            efficiency.append(acc / tokens * 100)
            eff_labels.append(method_labels[i])
            eff_colors.append(colors[i])
    
    # Add vanilla for comparison
    if "vanilla" in results:
        acc = results["vanilla"]["accuracy"]["mean"] * 100
        tokens = results["vanilla"]["tokens"]["mean"]
        efficiency.append(acc / tokens * 100)
        eff_labels.append("Vanilla CoT")
        eff_colors.append('#1f77b4')
    
    bars2 = ax2.bar(eff_labels, efficiency, color=eff_colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Accuracy per 100 Tokens')
    ax2.set_title('Efficiency Metric')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, eff in zip(bars2, efficiency):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_revision_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure3_revision_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure3_revision_analysis")


def plot_method_comparison_table(data, output_dir):
    """Create a table figure with detailed results."""
    results = data["results"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Method', 'Accuracy (%)', 'Tokens', 'Revision Rate (%)', 'Acc/1K Tokens']
    rows = []
    
    for method in ["vanilla", "entropy_only", "esr", "egl", "bestofn"]:
        if method not in results:
            continue
        
        acc = results[method]["accuracy"]["mean"] * 100
        acc_std = results[method]["accuracy"]["std"] * 100
        tokens = results[method]["tokens"]["mean"]
        
        rev_rate = "N/A"
        if "revision_rate" in results[method]:
            rev_rate = f"{results[method]['revision_rate']['mean']:.1%}"
        
        efficiency = acc / tokens * 1000
        
        rows.append([
            method.upper() if method in ["esr", "egl"] else method.replace("_", " ").title(),
            f"{acc:.2f} ± {acc_std:.2f}",
            f"{tokens:.1f}",
            rev_rate,
            f"{efficiency:.3f}"
        ])
    
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.25, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Experimental Results Summary (GSM8K)', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_results_table.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure4_results_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure4_results_table")


def main():
    output_dir = "figures"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading results from results.json...")
    data = load_results()
    
    print("\nGenerating figures...")
    plot_accuracy_comparison(data, output_dir)
    plot_efficiency_tradeoff(data, output_dir)
    plot_revision_analysis(data, output_dir)
    plot_method_comparison_table(data, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
