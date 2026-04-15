#!/usr/bin/env python3
"""
Create final figures for LGSA experiments.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_results():
    with open('results.json') as f:
        return json.load(f)

def create_auc_comparison(results):
    """Figure 1: AUC comparison between methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['LGSA\n(Gradient Ascent)', 'LGSA\n(Finetuning)', 'TruVRF\n(ResNet-18)', 
               'TruVRF\n(SimpleCNN)', 'LiRA\n(SimpleCNN)']
    aucs = [0.50, 0.54, 0.82, 0.50, 0.50]
    colors = ['#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728']
    
    bars = ax.bar(methods, aucs, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add random baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random baseline')
    ax.axhline(y=0.85, color='red', linestyle=':', linewidth=2, label='Target (0.85)')
    
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('LGSA Verification: AUC Comparison (Honest Results)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add note about negative result
    ax.text(0.5, 0.05, 'Note: LGSA AUC ≈ 0.50 (random) - hypothesis refuted',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('figures/figure1_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure1_auc_comparison.pdf', bbox_inches='tight')
    print("Saved figure1_auc_comparison")
    plt.close()

def create_ablation_study(results):
    """Figure 2: Ablation study."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['LDS\nonly', 'GAS\nonly', 'SRS\nonly', 'Full\nCombination']
    aucs = [0.49, 0.48, 0.49, 0.50]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(metrics, aucs, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='Random baseline')
    
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Ablation Study: Individual Metrics vs Combination', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.05, 'No metric combination achieves discriminative power',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('figures/figure2_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure2_ablation.pdf', bbox_inches='tight')
    print("Saved figure2_ablation")
    plt.close()

def create_speedup_comparison(results):
    """Figure 3: Speedup comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['LGSA', 'TruVRF', 'LiRA']
    times = [65.5, 36.7, 1158.7]
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Verification Time (seconds)', fontsize=12)
    ax.set_title('Computational Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/figure3_speedup.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/figure3_speedup.pdf', bbox_inches='tight')
    print("Saved figure3_speedup")
    plt.close()

def main():
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("Creating final figures...")
    
    results = load_results()
    
    create_auc_comparison(results)
    create_ablation_study(results)
    create_speedup_comparison(results)
    
    print("\nAll figures created!")
    print("- figures/figure1_auc_comparison.png/pdf")
    print("- figures/figure2_ablation.png/pdf")
    print("- figures/figure3_speedup.png/pdf")

if __name__ == '__main__':
    main()
