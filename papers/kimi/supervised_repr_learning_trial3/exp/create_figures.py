#!/usr/bin/env python3
"""
Generate figures from experiment results.
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def load_results(results_dir='./results'):
    """Load all results.json files."""
    result_files = glob.glob(os.path.join(results_dir, '**/results.json'), recursive=True)
    all_results = {}
    for rf in result_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)
            exp_name = data.get('experiment', os.path.basename(os.path.dirname(rf)))
            all_results[exp_name] = data
        except Exception as e:
            print(f"Error loading {rf}: {e}")
    return all_results


def group_by_method(all_results):
    """Group results by method."""
    methods = {}
    for exp_name, data in all_results.items():
        parts = exp_name.rsplit('_seed', 1)
        method = parts[0] if len(parts) > 1 else exp_name
        if method not in methods:
            methods[method] = []
        methods[method].append(data)
    return methods


def compute_stats(methods):
    """Compute statistics for each method."""
    stats = {}
    for method, runs in methods.items():
        overall = [r['best_overall_acc'] for r in runs if 'best_overall_acc' in r]
        balanced = [r['best_balanced_acc'] for r in runs if 'best_balanced_acc' in r]
        
        many = []
        medium = []
        few = []
        for r in runs:
            if 'final_metrics' in r:
                fm = r['final_metrics']
                if 'many_shot' in fm:
                    many.append(fm['many_shot'])
                if 'medium_shot' in fm:
                    medium.append(fm['medium_shot'])
                if 'few_shot' in fm:
                    few.append(fm['few_shot'])
        
        etf_dev = []
        for r in runs:
            if 'geometric_metrics' in r and 'etf_deviation' in r['geometric_metrics']:
                etf_dev.append(r['geometric_metrics']['etf_deviation'])
        
        stats[method] = {
            'n': len(runs),
            'overall': (np.mean(overall), np.std(overall)) if overall else (0, 0),
            'balanced': (np.mean(balanced), np.std(balanced)) if balanced else (0, 0),
            'many': (np.mean(many), np.std(many)) if many else (0, 0),
            'medium': (np.mean(medium), np.std(medium)) if medium else (0, 0),
            'few': (np.mean(few), np.std(few)) if few else (0, 0),
            'etf_dev': (np.mean(etf_dev), np.std(etf_dev)) if etf_dev else (0, 0),
        }
    return stats


def create_main_comparison(stats, output_path='figures/figure1_main_comparison.pdf'):
    """Create main comparison figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter to main methods
    main_methods = {k: v for k, v in stats.items() 
                   if any(x in k for x in ['ce', 'supcon', 'bcl', 'etfscl']) and 'ablation' not in k}
    
    if not main_methods:
        print("No main methods found for comparison figure")
        return
    
    # Sort by balanced accuracy
    sorted_methods = sorted(main_methods.items(), key=lambda x: x[1]['balanced'][0], reverse=True)
    
    methods = [m.replace('cifar100_', '').replace('_', ' ').upper() for m, _ in sorted_methods]
    overall_means = [s['overall'][0] for _, s in sorted_methods]
    overall_stds = [s['overall'][1] for _, s in sorted_methods]
    balanced_means = [s['balanced'][0] for _, s in sorted_methods]
    balanced_stds = [s['balanced'][1] for _, s in sorted_methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    x = np.arange(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    # Overall accuracy
    axes[0].bar(x, overall_means, yerr=overall_stds, capsize=4, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('Overall Accuracy', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=30, ha='right')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_ylim([0, max(overall_means) * 1.2])
    
    # Balanced accuracy
    axes[1].bar(x, balanced_means, yerr=balanced_stds, capsize=4, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1].set_title('Class-Balanced Accuracy', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=30, ha='right')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_ylim([0, max(balanced_means) * 1.2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_ablation_figure(stats, output_path='figures/figure2_ablation.pdf'):
    """Create ablation study figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get ablation and full method
    ablation_methods = {k: v for k, v in stats.items() if 'ablation' in k}
    full_method = {k: v for k, v in stats.items() if 'etfscl' in k and 'ablation' not in k}
    
    if not ablation_methods:
        print("No ablation results found")
        return
    
    labels = ['Full ETF-SCL']
    balanced_means = [list(full_method.values())[0]['balanced'][0]] if full_method else []
    balanced_stds = [list(full_method.values())[0]['balanced'][1]] if full_method else []
    
    ablation_map = {
        'ablation_no_etf': 'w/o ETF Reg',
        'ablation_no_adaptive': 'w/o Adaptive Mining',
        'ablation_no_temp': 'w/o Temp Scheduling',
        'ablation_etf_only': 'ETF Only',
    }
    
    colors = ['#2E7D32']  # Green for full method
    
    for key, label in ablation_map.items():
        matching = [k for k in ablation_methods.keys() if key in k]
        if matching:
            labels.append(label)
            balanced_means.append(ablation_methods[matching[0]]['balanced'][0])
            balanced_stds.append(ablation_methods[matching[0]]['balanced'][1])
            colors.append('#C62828')  # Red for ablations
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(labels))
    bars = ax.bar(x, balanced_means, yerr=balanced_stds, capsize=6, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Class-Balanced Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Ablation Study: Component Contribution', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(balanced_means) * 1.25])
    
    # Add value labels
    for bar, mean, std in zip(bars, balanced_means, balanced_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_many_medium_few_figure(stats, output_path='figures/figure3_shot_types.pdf'):
    """Create many/medium/few shot accuracy figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter methods
    methods_with_shots = {k: v for k, v in stats.items() 
                         if v['many'][0] > 0 and 'ablation' not in k}
    
    if not methods_with_shots:
        print("No shot-type results found")
        return
    
    sorted_methods = sorted(methods_with_shots.items(), key=lambda x: x[1]['balanced'][0], reverse=True)
    
    methods = [m.replace('cifar100_', '').replace('_', ' ').upper() for m, _ in sorted_methods]
    many_means = [s['many'][0] for _, s in sorted_methods]
    medium_means = [s['medium'][0] for _, s in sorted_methods]
    few_means = [s['few'][0] for _, s in sorted_methods]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(methods))
    width = 0.25
    
    colors = ['#1565C0', '#2E7D32', '#C62828']
    
    bars1 = ax.bar(x - width, many_means, width, label='Many-shot', color=colors[0], alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x, medium_means, width, label='Medium-shot', color=colors[1], alpha=0.85, edgecolor='black')
    bars3 = ax.bar(x + width, few_means, width, label='Few-shot', color=colors[2], alpha=0.85, edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy by Shot Type on CIFAR-100-LT (IF=100)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(max(many_means), max(medium_means), max(few_means)) * 1.2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_etf_deviation_figure(stats, output_path='figures/figure4_etf_deviation.pdf'):
    """Create ETF deviation comparison figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter methods with ETF deviation
    methods_with_etf = {k: v for k, v in stats.items() 
                       if v['etf_dev'][0] > 0 and 'ablation' not in k}
    
    if not methods_with_etf:
        print("No ETF deviation results found")
        return
    
    sorted_methods = sorted(methods_with_etf.items(), key=lambda x: x[1]['etf_dev'][0])
    
    methods = [m.replace('cifar100_', '').replace('_', ' ').upper() for m, _ in sorted_methods]
    etf_means = [s['etf_dev'][0] for _, s in sorted_methods]
    etf_stds = [s['etf_dev'][1] for _, s in sorted_methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(methods))
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    bars = ax.bar(x, etf_means, yerr=etf_stds, capsize=5, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=1)
    
    ax.set_ylabel('ETF Deviation (Frobenius Norm)', fontweight='bold', fontsize=12)
    ax.set_title('ETF Geometry Deviation (Lower is Better)', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_results_table(stats, output_path='figures/results_table.txt'):
    """Create text table of results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("RESULTS TABLE: CIFAR-100-LT (IF=100)\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Method':<25} {'Overall Acc':<18} {'Balanced Acc':<18} {'Few-shot Acc':<18}\n")
        f.write("-"*100 + "\n")
        
        # Sort by balanced accuracy
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['balanced'][0], reverse=True)
        
        for method, s in sorted_stats:
            name = method.replace('cifar100_', '').replace('_', ' ').upper()
            overall = f"{s['overall'][0]:.2f} ± {s['overall'][1]:.2f}"
            balanced = f"{s['balanced'][0]:.2f} ± {s['balanced'][1]:.2f}"
            few = f"{s['few'][0]:.2f} ± {s['few'][1]:.2f}" if s['few'][0] > 0 else "N/A"
            f.write(f"{name:<25} {overall:<18} {balanced:<18} {few:<18}\n")
        
        f.write("="*100 + "\n")
    
    print(f"Saved: {output_path}")


def main():
    print("Loading results...")
    results = load_results('./results')
    print(f"Loaded {len(results)} results")
    
    if not results:
        print("No results found!")
        return
    
    print("Grouping by method...")
    methods = group_by_method(results)
    
    print("Computing statistics...")
    stats = compute_stats(methods)
    
    print("\nGenerating figures...")
    os.makedirs('figures', exist_ok=True)
    
    create_main_comparison(stats)
    create_ablation_figure(stats)
    create_many_medium_few_figure(stats)
    create_etf_deviation_figure(stats)
    create_results_table(stats)
    
    print("\n" + "="*50)
    print("FIGURES GENERATED")
    print("="*50)
    
    # Print summary
    print("\nSummary:")
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['balanced'][0], reverse=True)
    for method, s in sorted_stats:
        if 'ablation' not in method:
            name = method.replace('cifar100_', '').upper()
            print(f"{name:20s}: Balanced={s['balanced'][0]:.2f}%, Few-shot={s['few'][0]:.2f}%")


if __name__ == '__main__':
    main()
