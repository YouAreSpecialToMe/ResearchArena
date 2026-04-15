"""
Create final figures from architecture-matched experiments.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_aggregated(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def create_architecture_comparison():
    """Create architecture comparison figure."""
    
    # Load results from new checkpoints
    vmamba = load_aggregated('checkpoints/vmamba/aggregated.json')
    localmamba = load_aggregated('checkpoints/localmamba/aggregated.json')
    cassvim_4d = load_aggregated('checkpoints/cassvim_4d/aggregated.json')
    cassvim_8d = load_aggregated('checkpoints/cassvim_8d/aggregated.json')
    random_sel = load_aggregated('checkpoints/random_selection/aggregated.json')
    fixed_perlayer = load_aggregated('checkpoints/fixed_perlayer/aggregated.json')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Main comparison - accuracy
    models = ['VMamba\n(16.2M)', 'LocalMamba\n(14.9M)', 
              'CASS-ViM-8D\n(3.9M)', 'Fixed Per-Layer\n(3.9M)',
              'CASS-ViM-4D\n(3.9M)', 'Random\n(3.9M)']
    accs = [
        vmamba['best_acc_mean'],
        localmamba['best_acc_mean'],
        cassvim_8d['best_acc_mean'],
        fixed_perlayer['best_acc_mean'],
        cassvim_4d['best_acc_mean'],
        random_sel['best_acc_mean']
    ]
    stds = [
        vmamba['best_acc_std'],
        localmamba['best_acc_std'],
        cassvim_8d['best_acc_std'],
        fixed_perlayer['best_acc_std'],
        cassvim_4d['best_acc_std'],
        random_sel['best_acc_std']
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b']
    
    x = np.arange(len(models))
    bars = ax1.bar(x, accs, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_title('CIFAR-100 Test Accuracy (50 epochs, 3 seeds)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim([44, 58])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc, std in zip(bars, accs, stds):
        ax1.text(bar.get_x() + bar.get_width()/2, acc + std + 0.3, 
                f'{acc:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight CASS-ViM-8D
    bars[2].set_edgecolor('green')
    bars[2].set_linewidth(3)
    
    # Parameters vs Accuracy scatter
    params = [16.15, 14.92, 3.92, 3.92, 3.92, 3.92]
    ax2.scatter(params, accs, s=300, c=colors, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add labels
    labels = ['VMamba', 'LocalMamba', 'CASS-ViM-8D', 'Fixed Per-Layer', 'CASS-ViM-4D', 'Random']
    for i, label in enumerate(labels):
        offset = (8, 5) if i < 2 else (-40, 5)
        ax2.annotate(label, (params[i], accs[i]), xytext=offset, 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Parameters (Millions)', fontsize=11)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy vs Model Size', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([44, 58])
    ax2.set_xlim([0, 20])
    
    # Highlight region for CASS-ViM
    ax2.axvspan(3, 5, alpha=0.1, color='green', label='CASS-ViM family')
    ax2.legend(loc='lower right')
    
    # Training curves comparison
    test_accs_vmamba = np.array([r['test_accs'] for r in vmamba['individual_results']])
    test_accs_local = np.array([r['test_accs'] for r in localmamba['individual_results']])
    test_accs_8d = np.array([r['test_accs'] for r in cassvim_8d['individual_results']])
    test_accs_4d = np.array([r['test_accs'] for r in cassvim_4d['individual_results']])
    
    epochs = np.arange(1, 51)
    ax3.plot(epochs, test_accs_vmamba.mean(axis=0), label='VMamba', color='#1f77b4', linewidth=2)
    ax3.plot(epochs, test_accs_local.mean(axis=0), label='LocalMamba', color='#ff7f0e', linewidth=2)
    ax3.plot(epochs, test_accs_8d.mean(axis=0), label='CASS-ViM-8D', color='#2ca02c', linewidth=2)
    ax3.plot(epochs, test_accs_4d.mean(axis=0), label='CASS-ViM-4D', color='#d62728', linewidth=2)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax3.set_title('Test Accuracy Curves (mean over 3 seeds)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # Ablation: Gradient vs Random
    ablation_models = ['Gradient\n(CASS-ViM-4D)', 'Random\nSelection']
    ablation_accs = [cassvim_4d['best_acc_mean'], random_sel['best_acc_mean']]
    ablation_stds = [cassvim_4d['best_acc_std'], random_sel['best_acc_std']]
    ablation_colors = ['#2ca02c', '#8c564b']
    
    x_abl = np.arange(len(ablation_models))
    bars_abl = ax4.bar(x_abl, ablation_accs, yerr=ablation_stds, capsize=5, 
                       color=ablation_colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax4.set_title('Ablation: Gradient vs Random Selection\n(Same architecture: 3.9M params)', 
                 fontsize=12, fontweight='bold')
    ax4.set_xticks(x_abl)
    ax4.set_xticklabels(ablation_models, fontsize=10)
    ax4.set_ylim([44, 56])
    ax4.grid(axis='y', alpha=0.3)
    
    # Add improvement annotation
    diff = ablation_accs[0] - ablation_accs[1]
    ax4.annotate('', xy=(0, ablation_accs[0]-0.5), xytext=(1, ablation_accs[1]+0.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax4.text(0.5, (ablation_accs[0] + ablation_accs[1])/2, f'+{diff:.1f}%\n(p=0.0035)', 
            fontsize=11, color='green', fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('figures/final_results.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/final_results.pdf', bbox_inches='tight')
    print("Saved: figures/final_results.png/pdf")
    plt.close()
    
    # Create summary figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Grouped bar chart comparing 4D and 8D with baselines
    methods = ['VMamba', 'LocalMamba', 'Fixed\nPer-Layer', 'CASS-ViM-8D', 'CASS-ViM-4D', 'Random']
    accuracies = [
        vmamba['best_acc_mean'],
        localmamba['best_acc_mean'],
        fixed_perlayer['best_acc_mean'],
        cassvim_8d['best_acc_mean'],
        cassvim_4d['best_acc_mean'],
        random_sel['best_acc_mean']
    ]
    errors = [
        vmamba['best_acc_std'],
        localmamba['best_acc_std'],
        fixed_perlayer['best_acc_std'],
        cassvim_8d['best_acc_std'],
        cassvim_4d['best_acc_std'],
        random_sel['best_acc_std']
    ]
    bar_colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728', '#8c564b']
    
    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, yerr=errors, capsize=6, color=bar_colors, 
                  alpha=0.85, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('CASS-ViM: Architecture-Matched Comparison on CIFAR-100 (50 epochs)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylim([44, 58])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, acc, err in zip(bars, accuracies, errors):
        ax.text(bar.get_x() + bar.get_width()/2, acc + err + 0.4, 
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add parameter count annotations
    param_texts = ['16.2M', '14.9M', '3.9M', '3.9M', '3.9M', '3.9M']
    for i, (bar, params) in enumerate(zip(bars, param_texts)):
        ax.text(bar.get_x() + bar.get_width()/2, 44.5, params, 
                ha='center', va='bottom', fontsize=9, style='italic', color='gray')
    
    # Highlight CASS-ViM results
    for i in [3, 4]:
        bars[i].set_edgecolor('darkgreen')
        bars[i].set_linewidth(2.5)
    
    # Add horizontal line at CASS-ViM-8D level
    ax.axhline(y=cassvim_8d['best_acc_mean'], color='green', linestyle='--', 
              alpha=0.5, linewidth=1.5)
    ax.text(5.5, cassvim_8d['best_acc_mean']+0.3, 'CASS-ViM-8D', 
           fontsize=9, color='green', ha='right')
    
    # Legend for parameter groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Baseline (>10M params)'),
        Patch(facecolor='white', edgecolor='darkgreen', linewidth=2.5, label='CASS-ViM (~4M params)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig('figures/summary_comparison.pdf', bbox_inches='tight')
    print("Saved: figures/summary_comparison.png/pdf")
    plt.close()

if __name__ == '__main__':
    print("Creating final figures...")
    Path('figures').mkdir(exist_ok=True)
    create_architecture_comparison()
    print("\nAll final figures created!")
