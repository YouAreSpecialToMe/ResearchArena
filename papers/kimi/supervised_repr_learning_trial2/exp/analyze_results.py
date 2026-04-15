#!/usr/bin/env python3
"""
Analyze and summarize LASER-SCL experimental results.
"""

import json
import glob
import os
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load all result JSON files."""
    results = []
    for f in glob.glob('results/*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return results

def compute_statistics(results):
    """Compute statistics by method."""
    method_accs = defaultdict(list)
    method_data = defaultdict(list)
    
    for r in results:
        method = r.get('method', 'unknown')
        acc = r.get('final_accuracy', 0)
        method_accs[method].append(acc)
        method_data[method].append(r)
    
    stats = {}
    for method, accs in method_accs.items():
        stats[method] = {
            'mean': statistics.mean(accs),
            'std': statistics.stdev(accs) if len(accs) > 1 else 0,
            'min': min(accs),
            'max': max(accs),
            'n': len(accs),
            'data': method_data[method]
        }
    
    return stats

def print_summary(stats):
    """Print results summary."""
    print('\n' + '='*80)
    print('LASER-SCL EXPERIMENT RESULTS SUMMARY')
    print('='*80)
    print(f"{'Method':<30} | {'Mean ± Std':<15} | {'Min':<6} | {'Max':<6} | {'N':<3}")
    print('-'*80)
    
    for method in sorted(stats.keys()):
        s = stats[method]
        print(f"{method:<30} | {s['mean']:>6.2f} ± {s['std']:<4.2f} | {s['min']:>6.2f} | {s['max']:>6.2f} | {s['n']:<3}")

def check_success_criteria(stats):
    """Check if success criteria are met."""
    print('\n' + '='*80)
    print('SUCCESS CRITERIA CHECK')
    print('='*80)
    
    # Criterion 1: LASER-SCL vs SupCon+LR by ≥2%
    if 'laser_scl' in stats and 'supcon_lr' in stats:
        laser_mean = stats['laser_scl']['mean']
        lr_mean = stats['supcon_lr']['mean']
        diff = laser_mean - lr_mean
        
        print(f"\n1. Primary: LASER-SCL outperforms SupCon+LR by ≥2%")
        print(f"   LASER-SCL: {laser_mean:.2f}%")
        print(f"   SupCon+LR: {lr_mean:.2f}%")
        print(f"   Difference: {diff:+.2f}%")
        print(f"   Result: {'PASS ✓' if diff >= 2 else 'FAIL ✗'}")
    else:
        print("\n1. Primary: Cannot check (missing results)")
    
    # Criterion 2: Component contributions ≥0.5%
    if 'laser_scl' in stats:
        print(f"\n2. Component contributions (ablations vs full LASER-SCL):")
        laser_mean = stats['laser_scl']['mean']
        
        for ablation in ['ablation_no_curriculum', 'ablation_no_elp', 'ablation_static']:
            if ablation in stats:
                abl_mean = stats[ablation]['mean']
                drop = laser_mean - abl_mean
                print(f"   {ablation}: {abl_mean:.2f}% (drop: {drop:+.2f}%) {'✓' if drop >= 0.5 else '✗'}")
    
    # Criterion 3: Comparison with vanilla SupCon
    if 'laser_scl' in stats and 'supcon' in stats:
        laser_mean = stats['laser_scl']['mean']
        supcon_mean = stats['supcon']['mean']
        diff = laser_mean - supcon_mean
        
        print(f"\n3. LASER-SCL vs Vanilla SupCon:")
        print(f"   Improvement: {diff:+.2f}%")

def create_figures(stats):
    """Create comparison figures."""
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Method comparison bar chart
    methods = ['supcon', 'supcon_lr', 'laser_scl']
    labels = ['Vanilla SupCon', 'SupCon+LR', 'LASER-SCL']
    
    means = []
    stds = []
    valid_labels = []
    
    for method, label in zip(methods, labels):
        if method in stats:
            means.append(stats[method]['mean'])
            stds.append(stats[method]['std'])
            valid_labels.append(label)
    
    if means:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(valid_labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('CIFAR-100 with 40% Label Noise: Method Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, fontsize=11)
        ax.set_ylim([0, max(means) * 1.2])
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                   f'{mean:.1f}±{std:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/main_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('figures/main_comparison.pdf', bbox_inches='tight')
        print(f"\nFigure saved: figures/main_comparison.png")
    
    # Figure 2: Ablation study
    ablations = ['laser_scl', 'ablation_no_curriculum', 'ablation_no_elp', 'ablation_static']
    abl_labels = ['Full LASER-SCL', 'No Curriculum', 'No ELP', 'Static Weighting']
    
    abl_means = []
    abl_stds = []
    valid_abl_labels = []
    
    for method, label in zip(ablations, abl_labels):
        if method in stats:
            abl_means.append(stats[method]['mean'])
            abl_stds.append(stats[method]['std'])
            valid_abl_labels.append(label)
    
    if len(abl_means) > 1:
        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(len(valid_abl_labels))
        colors = ['#2ca02c'] + ['#d62728'] * (len(valid_abl_labels) - 1)
        bars = ax.bar(x, abl_means, yerr=abl_stds, capsize=5, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Ablation Study: Component Contributions', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_abl_labels, fontsize=11, rotation=15, ha='right')
        ax.set_ylim([0, max(abl_means) * 1.2])
        
        # Add value labels
        for bar, mean, std in zip(bars, abl_means, abl_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                   f'{mean:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.savefig('figures/ablation_study.pdf', bbox_inches='tight')
        print(f"Figure saved: figures/ablation_study.png")

def save_results_json(stats):
    """Save aggregated results to JSON."""
    results_summary = {}
    for method, s in stats.items():
        results_summary[method] = {
            'mean_accuracy': s['mean'],
            'std_accuracy': s['std'],
            'min_accuracy': s['min'],
            'max_accuracy': s['max'],
            'num_seeds': s['n']
        }
    
    with open('results/aggregated_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nAggregated results saved: results/aggregated_results.json")

def main():
    print("Loading experimental results...")
    results = load_results()
    
    if not results:
        print("No results found!")
        return
    
    print(f"Loaded {len(results)} result files")
    
    stats = compute_statistics(results)
    print_summary(stats)
    check_success_criteria(stats)
    
    try:
        create_figures(stats)
    except Exception as e:
        print(f"\nWarning: Could not create figures: {e}")
    
    save_results_json(stats)
    
    print('\n' + '='*80)
    print('Analysis complete!')
    print('='*80)

if __name__ == '__main__':
    main()
