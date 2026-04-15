#!/usr/bin/env python3
"""
Generate publication-quality figures for the MemSat paper.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set publication-quality defaults
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.dpi'] = 300

RESULTS_DIR = Path("/home/nw366/ResearchArena/outputs/kimi_t3_compiler_optimization/idea_01/results")
FIGURES_DIR = Path("/home/nw366/ResearchArena/outputs/kimi_t3_compiler_optimization/idea_01/figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def figure_1_treewidth_distribution():
    """Figure 1: Treewidth distribution across kernels (histogram + boxplot)."""
    data = load_json(RESULTS_DIR / "treewidth" / "treewidth_results.json")
    
    # Collect all treewidth values
    all_tw = []
    kernel_tw = {}
    
    for kernel, results in data["kernels"].items():
        tw_values = [r["level1_treewidth"] for r in results]
        all_tw.extend(tw_values)
        kernel_tw[kernel] = np.mean(tw_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Histogram
    ax1.hist(all_tw, bins=range(1, 8), edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Threshold (10)')
    ax1.set_xlabel('Treewidth')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Treewidth Distribution')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Boxplot by kernel
    kernels = list(kernel_tw.keys())
    tw_by_kernel = [[r["level1_treewidth"] for r in data["kernels"][k]] for k in kernels]
    
    bp = ax2.boxplot(tw_by_kernel, labels=[k[:6] for k in kernels], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Kernel')
    ax2.set_ylabel('Treewidth')
    ax2.set_title('(b) Treewidth by Kernel')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_1_treewidth_distribution.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_1_treewidth_distribution.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 1: Treewidth Distribution")


def figure_2_extraction_comparison():
    """Figure 2: Extraction time vs solution quality scatter plot."""
    data = load_json(RESULTS_DIR / "extraction" / "treewidth_aware_results.json")
    
    # Collect data points
    greedy_times = []
    tw_times = []
    ilp_times = []
    quality_ratios = []
    
    for kernel, results in data["kernels"].items():
        for r in results:
            greedy_times.append(r["greedy_time_ms"])
            tw_times.append(r["treewidth_time_ms"])
            ilp_times.append(r["ilp_time_ms"])
            quality_ratios.append(r["quality_ratio"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot
    ax.scatter(greedy_times, [0]*len(greedy_times), marker='o', s=100, 
               label='Greedy', alpha=0.6, color='orange')
    ax.scatter(tw_times, quality_ratios, marker='s', s=100,
               label='Treewidth-Aware', alpha=0.6, color='steelblue')
    ax.scatter(ilp_times, [0]*len(ilp_times), marker='^', s=100,
               label='ILP (Optimal)', alpha=0.6, color='green')
    
    # Reference lines
    ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, 
               label='10% Quality Threshold')
    ax.axvline(x=5*np.median(greedy_times), color='purple', linestyle=':', linewidth=1.5,
               label='5x Greedy Time')
    
    ax.set_xlabel('Extraction Time (ms)')
    ax.set_ylabel('Quality Ratio (vs ILP)')
    ax.set_title('Extraction Time vs Solution Quality')
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_2_extraction_comparison.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_2_extraction_comparison.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 2: Extraction Comparison")


def figure_3_joint_vs_sequential():
    """Figure 3: Joint vs Sequential optimization comparison."""
    data = load_json(RESULTS_DIR / "extraction" / "joint_optimization_results.json")
    
    kernels = list(data["kernels"].keys())
    improvements = []
    
    for kernel in kernels:
        kernel_improvements = [r["improvement_pct"] for r in data["kernels"][kernel]]
        improvements.append(np.mean(kernel_improvements))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ['green' if imp >= 10 else 'orange' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(range(len(kernels)), improvements, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10% Target')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Kernel')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Joint vs Sequential Optimization: Cost Reduction')
    ax.set_xticks(range(len(kernels)))
    ax.set_xticklabels([k[:8] for k in kernels], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_3_joint_vs_sequential.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_3_joint_vs_sequential.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 3: Joint vs Sequential")


def figure_4_ablation_summary():
    """Figure 4: Ablation study summary."""
    # Load ablation data
    hierarchy_data = load_json(RESULTS_DIR / "ablation" / "hierarchy_ablation.json")
    layout_data = load_json(RESULTS_DIR / "ablation" / "layout_rules_ablation.json")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # (a) Hierarchy: Flat vs Hierarchical treewidth
    ax1 = axes[0]
    flat_tw = []
    hier_tw = []
    
    for kernel in hierarchy_data["flat"]:
        flat_tw.extend([r["treewidth"] for r in hierarchy_data["flat"][kernel]])
        hier_tw.extend([r["treewidth"] for r in hierarchy_data["hierarchical"][kernel]])
    
    categories = ['Flat', 'Hierarchical']
    values = [np.mean(flat_tw), np.mean(hier_tw)]
    errors = [np.std(flat_tw), np.std(hier_tw)]
    
    ax1.bar(categories, values, yerr=errors, capsize=5, color=['coral', 'lightblue'], 
            edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Mean Treewidth')
    ax1.set_title('(a) Flat vs Hierarchical Decomposition')
    ax1.grid(axis='y', alpha=0.3)
    
    # (b) Layout rules impact
    ax2 = axes[1]
    
    with_costs = []
    without_costs = []
    
    for kernel in layout_data["with_layout"]:
        with_costs.extend([r["total_cost"] for r in layout_data["with_layout"][kernel]])
        without_costs.extend([r["total_cost"] for r in layout_data["without_layout"][kernel]])
    
    categories = ['Without Layout\nRules', 'With Layout\nRules']
    values = [np.mean(without_costs), np.mean(with_costs)]
    errors = [np.std(without_costs), np.std(with_costs)]
    
    ax2.bar(categories, values, yerr=errors, capsize=5, color=['salmon', 'lightgreen'],
            edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Mean Cost')
    ax2.set_title('(b) Impact of Layout Transformation Rules')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_ablation_summary.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_4_ablation_summary.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 4: Ablation Summary")


def figure_5_hypothesis_summary():
    """Figure 5: Hypothesis testing summary."""
    results = load_json(Path("/home/nw366/ResearchArena/outputs/kimi_t3_compiler_optimization/idea_01/results.json"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hypotheses = ['H1\n(Low Treewidth)', 'H2\n(Joint Opt)', 'H3\n(Extraction)']
    confirmed = [
        results["hypotheses"]["H1"]["confirmed"],
        results["hypotheses"]["H2"]["confirmed"],
        results["hypotheses"]["H3"]["confirmed"]
    ]
    
    # Metrics for display
    metrics = [
        f"{results['hypotheses']['H1']['pct_leq_10']:.0f}% ≤ 10",
        f"{results['hypotheses']['H2']['mean_improvement_pct']:.1f}% improv",
        f"{results['hypotheses']['H3']['mean_quality_ratio']*100:.0f}% quality loss"
    ]
    
    colors = ['#2ecc71' if c else '#e74c3c' for c in confirmed]
    bars = ax.bar(hypotheses, [1, 1, 1], color=colors, edgecolor='black', alpha=0.8)
    
    # Add status labels
    for i, (bar, conf, metric) in enumerate(zip(bars, confirmed, metrics)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                'CONFIRMED' if conf else 'REFUTED',
                ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(bar.get_x() + bar.get_width()/2., -0.15,
                metric,
                ha='center', va='top', fontsize=9)
    
    ax.set_ylim(-0.3, 1.2)
    ax.set_ylabel('Status')
    ax.set_title('Hypothesis Testing Results')
    ax.set_yticks([])
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Add summary text
    total_confirmed = sum(confirmed)
    ax.text(0.5, -0.25, f"Overall: {total_confirmed}/3 hypotheses confirmed",
            transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_5_hypothesis_summary.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "figure_5_hypothesis_summary.png", bbox_inches='tight')
    plt.close()
    print("Generated Figure 5: Hypothesis Summary")


def generate_all_figures():
    """Generate all figures."""
    print("="*60)
    print("Generating Figures for MemSat Paper")
    print("="*60)
    
    figure_1_treewidth_distribution()
    figure_2_extraction_comparison()
    figure_3_joint_vs_sequential()
    figure_4_ablation_summary()
    figure_5_hypothesis_summary()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
