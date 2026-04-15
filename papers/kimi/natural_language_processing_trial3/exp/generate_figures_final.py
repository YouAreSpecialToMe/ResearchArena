"""
Generate publication-ready figures for the paper.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_aggregated_results(results_file):
    """Load aggregated results."""
    with open(results_file) as f:
        data = json.load(f)
    return data["summary"]


def plot_accuracy_comparison(summary, output_dir):
    """Plot accuracy comparison across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    datasets = []
    accuracies = []
    errors = []
    
    for key, data in summary.items():
        method = data["method"]
        dataset = data["dataset"]
        acc = data["accuracy_mean"]
        std = data.get("accuracy_std", 0)
        
        methods.append(method)
        datasets.append(dataset)
        accuracies.append(acc)
        errors.append(std)
    
    # Create grouped bar chart
    x = np.arange(len(set(methods)))
    width = 0.35
    
    gsm8k_accs = [acc for method, dataset, acc in zip(methods, datasets, accuracies) if dataset == "gsm8k"]
    gsm8k_errs = [err for method, dataset, err in zip(methods, datasets, errors) if dataset == "gsm8k"]
    gsm8k_methods = [method for method, dataset in zip(methods, datasets) if dataset == "gsm8k"]
    
    ax.bar(x - width/2, gsm8k_accs, width, yerr=gsm8k_errs, label='GSM8K', capsize=5)
    
    if any(dataset == "math500" for dataset in datasets):
        math_accs = [acc for method, dataset, acc in zip(methods, datasets, accuracies) if dataset == "math500"]
        math_errs = [err for method, dataset, err in zip(methods, datasets, errors) if dataset == "math500"]
        ax.bar(x + width/2, math_accs, width, yerr=math_errs, label='MATH-500', capsize=5)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(set(gsm8k_methods)), rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved accuracy comparison to {output_dir / 'accuracy_comparison.png'}")


def plot_efficiency_comparison(summary, output_dir):
    """Plot accuracy vs tokens scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {'vanilla': '#1f77b4', 'entropy_only': '#ff7f0e', 'esr': '#2ca02c', 
              'egl_posthoc': '#d62728', 'egb_beam': '#9467bd', 'bestofn': '#8c564b'}
    
    for key, data in summary.items():
        method = data["method"]
        dataset = data["dataset"]
        acc = data["accuracy_mean"]
        tokens = data["tokens_mean"] / 1000  # Convert to thousands
        
        color = colors.get(method, '#333333')
        marker = 'o' if dataset == 'gsm8k' else 's'
        
        ax.scatter(tokens, acc, c=color, marker=marker, s=150, alpha=0.7, 
                  edgecolors='black', linewidth=1.5,
                  label=f"{method} ({dataset})")
    
    ax.set_xlabel('Average Tokens per Problem (thousands)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Efficiency Trade-off')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Create unique legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'efficiency_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved efficiency comparison to {output_dir / 'efficiency_comparison.png'}")


def plot_revision_analysis(summary, output_dir):
    """Plot revision rate analysis for ESR."""
    esr_data = {k: v for k, v in summary.items() if v["method"] == "esr"}
    
    if not esr_data:
        print("No ESR data found, skipping revision analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Revision rate by dataset
    datasets = []
    rev_rates = []
    rev_stds = []
    
    for key, data in esr_data.items():
        if "revision_rate_mean" in data:
            datasets.append(data["dataset"])
            rev_rates.append(data["revision_rate_mean"] * 100)
            rev_stds.append(data.get("revision_rate_std", 0) * 100)
    
    if datasets:
        axes[0].bar(datasets, rev_rates, yerr=rev_stds, capsize=5, color='#2ca02c', alpha=0.7)
        axes[0].set_ylabel('Revision Rate (%)')
        axes[0].set_title('ESR Revision Rate by Dataset')
        axes[0].set_ylim([0, 100])
        axes[0].axhline(y=25, color='r', linestyle='--', label='Target (25%)')
        axes[0].legend()
    
    # Plot 2: Comparison with baselines
    methods = ['vanilla', 'entropy_only', 'esr']
    gsm8k_accs = []
    
    for method in methods:
        key = f"{method}_gsm8k"
        if key in summary:
            gsm8k_accs.append(summary[key]["accuracy_mean"])
        else:
            gsm8k_accs.append(0)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[1].bar(methods, gsm8k_accs, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('GSM8K: ESR vs Baselines')
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'revision_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'revision_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved revision analysis to {output_dir / 'revision_analysis.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--output_dir", default="figures")
    args = parser.parse_args()
    
    # Load results
    print("Loading aggregated results...")
    summary = load_aggregated_results(args.results)
    print(f"Loaded {len(summary)} result entries")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_accuracy_comparison(summary, output_dir)
    plot_efficiency_comparison(summary, output_dir)
    plot_revision_analysis(summary, output_dir)
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
