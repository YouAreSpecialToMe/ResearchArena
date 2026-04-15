#!/usr/bin/env python3
"""
Generate visualizations and tables for the paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def load_results(pattern):
    """Load all results matching a pattern."""
    results = []
    results_dir = Path('results')
    for f in results_dir.glob(pattern):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def aggregate_by_seed(results, metric='f1'):
    """Aggregate results across seeds."""
    values = [r['metrics'][metric] for r in results if 'metrics' in r]
    if not values:
        return 0, 0
    return np.mean(values), np.std(values)


def plot_main_results():
    """Create main results comparison figure."""
    
    methods = {
        'Minimum Repair': 'baseline_minimum_repair_10pct.json',
        'Dense BP': 'baseline_dense_bp_10pct_seed*.json',
        'ERACER-style': 'baseline_eracer_10pct_seed*.json',
        'CleanBP': 'cleanbp_full_hospital_10pct_seed*.json'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    f1_means = []
    f1_stds = []
    time_means = []
    labels = []
    
    for label, pattern in methods.items():
        results = load_results(pattern)
        if not results:
            continue
        
        mean_f1, std_f1 = aggregate_by_seed(results, 'f1')
        mean_time = np.mean([r.get('runtime_seconds', 0) for r in results])
        
        f1_means.append(mean_f1)
        f1_stds.append(std_f1)
        time_means.append(mean_time)
        labels.append(label)
    
    # F1 comparison
    x = np.arange(len(labels))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ax1.bar(x, f1_means, yerr=f1_stds, capsize=5, color=colors[:len(labels)])
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_title('Repair Quality Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    # Time comparison (log scale)
    ax2.bar(x, time_means, color=colors[:len(labels)])
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_title('Runtime Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/main_results.png', bbox_inches='tight')
    plt.savefig('plots/main_results.pdf', bbox_inches='tight')
    print("Saved plots/main_results.png and plots/main_results.pdf")


def plot_ablation_sparsification():
    """Plot ablation study for sparsification."""
    
    variants = ['dense', 'violation_only', 'full']
    labels = ['Dense', 'Violation-Only', 'Full CleanBP']
    
    f1_scores = []
    times = []
    
    for variant in variants:
        results = load_results(f'ablation_sparsification_{variant}_seed*.json')
        if results:
            f1_scores.append([r['metrics']['f1'] for r in results])
            times.append([r['runtime_seconds'] for r in results])
        else:
            f1_scores.append([0])
            times.append([0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    x = np.arange(len(labels))
    f1_means = [np.mean(scores) for scores in f1_scores]
    f1_stds = [np.std(scores) for scores in f1_scores]
    
    ax1.bar(x, f1_means, yerr=f1_stds, capsize=5, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Impact of Violation-Driven Sparsification', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim([0, 1.1])
    ax1.grid(axis='y', alpha=0.3)
    
    time_means = [np.mean(t) for t in times]
    ax2.bar(x, time_means, color=['#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Runtime by Variant', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/ablation_sparsification.png', bbox_inches='tight')
    plt.savefig('plots/ablation_sparsification.pdf', bbox_inches='tight')
    print("Saved plots/ablation_sparsification.png and plots/ablation_sparsification.pdf")


def plot_scalability():
    """Plot scalability curves."""
    
    sizes = [1000, 5000, 10000]
    violation_rates = [0.01, 0.05, 0.10]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for rate in violation_rates:
        times = []
        violations = []
        
        for size in sizes:
            pattern = f'scalability_{size}_{int(rate*100)}pct.json'
            results = load_results(pattern)
            if results:
                times.append(results[0]['runtime_seconds'])
                violations.append(results[0]['n_violations'])
        
        if times:
            label = f'{int(rate*100)}% violation rate'
            ax1.plot(sizes[:len(times)], times, marker='o', label=label, linewidth=2)
            ax2.plot(violations, times, marker='o', label=label, linewidth=2)
    
    ax1.set_xlabel('Number of Tuples', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Scalability vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Number of Violations', fontsize=12)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Scalability vs Violations', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/scalability.png', bbox_inches='tight')
    plt.savefig('plots/scalability.pdf', bbox_inches='tight')
    print("Saved plots/scalability.png and plots/scalability.pdf")


def plot_calibration():
    """Plot calibration curve."""
    
    results = load_results('calibration_hospital_seed*.json')
    if not results:
        return
    
    # Use first result for plotting
    result = results[0]
    confidences = result.get('confidences', [])
    accuracies = result.get('accuracies', [])
    ece = result.get('ece', 0)
    
    if not confidences:
        return
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Create calibration curve
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_centers = []
    bin_accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [(c > bin_lower) and (c <= bin_upper) for c in confidences]
        if any(in_bin):
            bin_confidences = [c for c, in_b in zip(confidences, in_bin) if in_b]
            bin_accs = [a for a, in_b in zip(accuracies, in_bin) if in_b]
            bin_centers.append(np.mean(bin_confidences))
            bin_accuracies.append(np.mean(bin_accs))
    
    # Plot
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax.plot(bin_centers, bin_accuracies, 'o-', label=f'CleanBP (ECE={ece:.3f})', linewidth=2, markersize=8)
    ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('plots/calibration.png', bbox_inches='tight')
    plt.savefig('plots/calibration.pdf', bbox_inches='tight')
    print("Saved plots/calibration.png and plots/calibration.pdf")


def generate_tables():
    """Generate LaTeX tables."""
    
    # Main results table
    methods = {
        'Minimum Repair': 'baseline_minimum_repair_10pct.json',
        'Dense BP': 'baseline_dense_bp_10pct_seed*.json',
        'ERACER-style': 'baseline_eracer_10pct_seed*.json',
        'CleanBP': 'cleanbp_full_hospital_10pct_seed*.json'
    }
    
    with open('tables/main_results.tex', 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of repair methods on Hospital dataset with 10\\% error rate.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Method & F1 & Precision & Recall & Time (s) \\\\\n")
        f.write("\\midrule\n")
        
        for label, pattern in methods.items():
            results = load_results(pattern)
            if not results:
                continue
            
            f1_mean, f1_std = aggregate_by_seed(results, 'f1')
            p_mean, p_std = aggregate_by_seed(results, 'precision')
            r_mean, r_std = aggregate_by_seed(results, 'recall')
            time_mean = np.mean([r.get('runtime_seconds', 0) for r in results])
            
            f.write(f"{label} & ${f1_mean:.3f} \\pm {f1_std:.3f}$ & "
                   f"${p_mean:.3f} \\pm {p_std:.3f}$ & "
                   f"${r_mean:.3f} \\pm {r_std:.3f}$ & "
                   f"${time_mean:.2f}$ \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print("Saved tables/main_results.tex")
    
    # Scalability table
    with open('tables/scalability.tex', 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Scalability of CleanBP on synthetic datasets.}\n")
        f.write("\\label{tab:scalability}\n")
        f.write("\\begin{tabular}{rrrr}\n")
        f.write("\\toprule\n")
        f.write("Tuples & Violations & Time (s) & Cells \\\\\n")
        f.write("\\midrule\n")
        
        for size in [1000, 5000, 10000]:
            for rate in [0.01, 0.05, 0.10]:
                pattern = f'scalability_{size}_{int(rate*100)}pct.json'
                results = load_results(pattern)
                if results:
                    r = results[0]
                    f.write(f"{r['n_tuples']:,} & {r['n_violations']:,} & "
                           f"{r['runtime_seconds']:.2f} & "
                           f"{r['graph_stats']['n_cells']:,} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print("Saved tables/scalability.tex")


if __name__ == '__main__':
    print("Generating visualizations and tables...")
    
    # Create directories
    Path('plots').mkdir(exist_ok=True)
    Path('tables').mkdir(exist_ok=True)
    
    # Generate figures
    plot_main_results()
    plot_ablation_sparsification()
    plot_scalability()
    plot_calibration()
    
    # Generate tables
    generate_tables()
    
    print("\nAll visualizations and tables generated successfully!")
