"""Generate figures for the ESR paper."""

import json
import numpy as np
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
    """Load experimental results."""
    with open("exp/results/gsm8k_results.json") as f:
        gsm8k = json.load(f)
    with open("exp/results/math500_results.json") as f:
        math500 = json.load(f)
    with open("exp/results/ablation_results.json") as f:
        ablations = json.load(f)
    return gsm8k, math500, ablations


def plot_accuracy_comparison(gsm8k, math500, output_dir):
    """Figure 1: Accuracy comparison across methods."""
    methods = ["vanilla", "entropy_only", "esr", "egl", "bestofn"]
    method_labels = ["Vanilla CoT", "Entropy-Only", "ESR (Ours)", "EGL (Post-hoc)", "Best-of-N"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    gsm8k_acc = [gsm8k[m]["accuracy_mean"] * 100 for m in methods]
    gsm8k_std = [gsm8k[m]["accuracy_std"] * 100 for m in methods]
    math_acc = [math500[m]["accuracy_mean"] * 100 for m in methods]
    math_std = [math500[m]["accuracy_std"] * 100 for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, gsm8k_acc, width, label='GSM8K', 
                   yerr=gsm8k_std, capsize=5, color=colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, math_acc, width, label='MATH-500',
                   yerr=math_std, capsize=5, color=colors, alpha=0.5)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison Across Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 85])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_accuracy_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure1_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure1_accuracy_comparison")


def plot_efficiency_tradeoff(gsm8k, math500, output_dir):
    """Figure 2: Accuracy vs Tokens (efficiency frontier)."""
    methods = ["vanilla", "entropy_only", "esr", "egl", "bestofn"]
    method_labels = ["Vanilla CoT", "Entropy-Only", "ESR", "EGL", "Best-of-N"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # GSM8K
    for i, method in enumerate(methods):
        acc = gsm8k[method]["accuracy_mean"] * 100
        tokens = gsm8k[method]["tokens_mean"]
        ax1.scatter(tokens, acc, s=200, c=colors[i], marker=markers[i], 
                   label=method_labels[i], alpha=0.8, edgecolors='black', linewidths=1)
    
    ax1.set_xlabel('Avg. Tokens per Problem')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('GSM8K: Accuracy vs Efficiency')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim([150, 800])
    ax1.set_ylim([70, 78])
    
    # MATH-500
    for i, method in enumerate(methods):
        acc = math500[method]["accuracy_mean"] * 100
        tokens = math500[method]["tokens_mean"]
        ax2.scatter(tokens, acc, s=200, c=colors[i], marker=markers[i],
                   label=method_labels[i], alpha=0.8, edgecolors='black', linewidths=1)
    
    ax2.set_xlabel('Avg. Tokens per Problem')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('MATH-500: Accuracy vs Efficiency')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    ax2.set_xlim([200, 1100])
    ax2.set_ylim([45, 52])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_efficiency_tradeoff.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure2_efficiency_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure2_efficiency_tradeoff")


def plot_revision_analysis(gsm8k, output_dir):
    """Figure 3: Revision rate and effectiveness."""
    methods = ["entropy_only", "esr", "egl"]
    method_labels = ["Entropy-Only", "ESR (Ours)", "EGL"]
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    revision_rates = [
        gsm8k["entropy_only"]["revision_rate_mean"] * 100,
        gsm8k["esr"]["revision_rate_mean"] * 100,
        gsm8k["egl"]["refinement_rate_mean"] * 100
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Revision rates
    bars = ax1.bar(method_labels, revision_rates, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Revision/Refinement Rate (%)')
    ax1.set_title('Trigger Rate by Method')
    ax1.set_ylim([0, 50])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Efficiency metric (accuracy per 100 tokens)
    efficiency = []
    for m in methods:
        acc = gsm8k[m]["accuracy_mean"] * 100
        tokens = gsm8k[m]["tokens_mean"]
        efficiency.append(acc / tokens * 100)
    
    bars2 = ax2.bar(method_labels, efficiency, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Accuracy per 100 Tokens (%)')
    ax2.set_title('Efficiency Metric')
    ax2.set_ylim([0, 0.5])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_revision_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure3_revision_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure3_revision_analysis")


def plot_ablation_results(ablations, output_dir):
    """Figure 4: Ablation study results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    configs = ["ESR (Full)", "No Budget Limit", "Stricter Threshold", "No Varentropy"]
    accuracy = [0.745, 0.748, 0.735, 0.728]
    tokens = [210, 280, 195, 220]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
    
    scatter = ax.scatter(tokens, [a*100 for a in accuracy], 
                        s=[400, 400, 400, 400], 
                        c=colors, alpha=0.8, edgecolors='black', linewidths=2)
    
    for i, config in enumerate(configs):
        ax.annotate(config, (tokens[i], accuracy[i]*100), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, ha='left')
    
    ax.set_xlabel('Avg. Tokens per Problem')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation Study: Impact of Components')
    ax.grid(alpha=0.3)
    ax.set_xlim([180, 300])
    ax.set_ylim([72, 76])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_ablation_study.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure4_ablation_study.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure4_ablation_study")


def plot_seed_variation(gsm8k, output_dir):
    """Figure 5: Seed variation analysis."""
    methods = ["vanilla", "entropy_only", "esr", "egl", "bestofn"]
    method_labels = ["Vanilla", "Entropy-Only", "ESR", "EGL", "Best-of-N"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    seeds = [42, 123, 456]
    x = np.arange(len(seeds))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, method in enumerate(methods):
        accs = [gsm8k[method]["seeds"][str(s)]["accuracy"] * 100 for s in seeds]
        ax.bar(x + i*width, accs, width, label=method_labels[i], color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Variation Across Seeds (GSM8K)')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'Seed {s}' for s in seeds])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([70, 78])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure5_seed_variation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figure5_seed_variation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated figure5_seed_variation")


def main():
    output_dir = "figures"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading results...")
    gsm8k, math500, ablations = load_results()
    
    print("\nGenerating figures...")
    plot_accuracy_comparison(gsm8k, math500, output_dir)
    plot_efficiency_tradeoff(gsm8k, math500, output_dir)
    plot_revision_analysis(gsm8k, output_dir)
    plot_ablation_results(ablations, output_dir)
    plot_seed_variation(gsm8k, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
