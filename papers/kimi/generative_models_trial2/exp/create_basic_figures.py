#!/usr/bin/env python
"""
Create basic figures for the research paper.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def main():
    """Create basic figures."""
    print("Creating figures...")
    
    # Load results
    with open('results.json', 'r') as f:
        results = json.load(f)
    
    agg = results.get('aggregated', results)
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Figure 1: Metric comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    method_mapping = {
        'baseline_uniform': 'Baseline Uniform',
        'baseline_density': 'Baseline Density', 
        'distflow_idw': 'DistFlow-IDW'
    }
    
    available_methods = [(k, v) for k, v in method_mapping.items() if k in agg]
    method_keys = [k for k, v in available_methods]
    methods = [v for k, v in available_methods]
    
    metrics = ['cd_overall', 'cd_near', 'cd_mid', 'cd_far']
    metric_labels = ['CD-Overall', 'CD-Near (0-20m)', 'CD-Mid (20-50m)', 'CD-Far (50m+)']
    
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    for i, (key, method) in enumerate(zip(method_keys, methods)):
        values = [agg[key][m]['mean'] for m in metrics]
        errors = [agg[key][m]['std'] for m in metrics]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=method, color=colors[i], 
               yerr=errors, capsize=3, alpha=0.8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Chamfer Distance (lower is better)')
    ax.set_title('Distance-Stratified Chamfer Distance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'metric_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'metric_comparison.png'}")
    
    # Figure 2: Far-field comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    
    values = [agg[k]['cd_far']['mean'] for k in method_keys]
    errors = [agg[k]['cd_far']['std'] for k in method_keys]
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bars = ax.bar(methods, values, yerr=errors, capsize=5, color=colors, alpha=0.8)
    
    # Add improvement percentage
    baseline = values[0]
    for i, (bar, val) in enumerate(zip(bars, values)):
        improvement = (baseline - val) / baseline * 100
        color = 'green' if improvement > 20 else 'orange' if improvement > 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, val + errors[i] + 0.01,
               f'{improvement:+.1f}%', ha='center', va='bottom', fontweight='bold',
               color=color)
    
    ax.set_ylabel('Chamfer Distance (CD-Far, 50m+)')
    ax.set_title('Far-Field Generation Quality (Lower is Better)')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'far_field_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'far_field_comparison.png'}")
    
    # Figure 3: Statistical test results
    if 'statistical_tests' in results and 'cd_far_improvement' in results['statistical_tests']:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        test = results['statistical_tests']['cd_far_improvement']
        
        methods = ['Baseline\nUniform', 'DistFlow-IDW']
        values = [test['baseline_mean'], test['distflow_mean']]
        
        colors = ['#1f77b4', '#ff7f0e']
        bars = ax.bar(methods, values, color=colors, alpha=0.8)
        
        # Add p-value
        p_value = test['p_value']
        ax.text(0.5, max(values) * 0.9, f'p-value: {p_value:.4f}\n({"Significant" if test["significant"] else "Not Significant"})',
               ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel('CD-Far (Chamfer Distance at 50m+)')
        ax.set_title('Statistical Comparison: Baseline vs DistFlow-IDW')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'statistical_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {figures_dir / 'statistical_comparison.png'}")
    
    print("\nAll figures created successfully!")


if __name__ == "__main__":
    main()
