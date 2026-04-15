#!/usr/bin/env python3
"""Generate figures from honest results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')
FIGURES_DIR = WORKSPACE / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Load results
with open(WORKSPACE / 'results.json') as f:
    results = json.load(f)

agg = results['aggregated']

# Define method order and labels
methods = ['random', 'correlation', 'cosine', 'crossgrn', 'crossgrn_no_celltype']
labels = ['Random', 'Correlation', 'Cosine', 'CROSS-GRN', 'CROSS-GRN\n(no cell type)']
colors = ['#cccccc', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']

# Extract metrics
auroc_means = [agg[m]['auroc']['mean'] for m in methods]
auroc_stds = [agg[m]['auroc']['std'] for m in methods]

auprc_means = [agg[m]['auprc']['mean'] for m in methods]
auprc_stds = [agg[m]['auprc']['std'] for m in methods]

sign_means = [agg[m]['sign_accuracy']['mean'] for m in methods]
sign_stds = [agg[m]['sign_accuracy']['std'] for m in methods]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AUROC
ax = axes[0]
bars = ax.bar(labels, auroc_means, yerr=auroc_stds, color=colors, edgecolor='black', capsize=5)
ax.set_ylabel('AUROC', fontsize=12)
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
ax.set_title('(a) AUROC Comparison', fontsize=13, fontweight='bold')
ax.legend()
for bar, mean in zip(bars, auroc_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# AUPRC
ax = axes[1]
bars = ax.bar(labels, auprc_means, yerr=auprc_stds, color=colors, edgecolor='black', capsize=5)
ax.set_ylabel('AUPRC', fontsize=12)
ax.set_ylim(0, max(auprc_means) * 1.3)
ax.set_title('(b) AUPRC Comparison', fontsize=13, fontweight='bold')
for bar, mean in zip(bars, auprc_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# Sign Accuracy
ax = axes[2]
bars = ax.bar(labels, sign_means, yerr=sign_stds, color=colors, edgecolor='black', capsize=5)
ax.set_ylabel('Sign Accuracy', fontsize=12)
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
ax.set_title('(c) Sign Prediction Accuracy', fontsize=13, fontweight='bold')
ax.legend()
for bar, mean in zip(bars, sign_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_1_main_results.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'figure_1_main_results.png', dpi=300, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'figure_1_main_results.pdf'}")
plt.close()

# Create ablation figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

ablation_methods = ['crossgrn', 'crossgrn_no_celltype']
ablation_labels = ['CROSS-GRN\n(Full)', 'CROSS-GRN\n(No Cell Type)']
ablation_colors = ['#8da0cb', '#e78ac3']

x = np.arange(3)
width = 0.35

# Full model
full_auroc = agg['crossgrn']['auroc']['mean']
_full_auprc = agg['crossgrn']['auprc']['mean']
full_sign = agg['crossgrn']['sign_accuracy']['mean']

# Ablated
ab_auroc = agg['crossgrn_no_celltype']['auroc']['mean']
ab_auprc = agg['crossgrn_no_celltype']['auprc']['mean']
ab_sign = agg['crossgrn_no_celltype']['sign_accuracy']['mean']

# Grouped bar chart
metrics = ['AUROC', 'AUPRC', 'Sign Acc']
full_vals = [full_auroc, _full_auprc, full_sign]
ab_vals = [ab_auroc, ab_auprc, ab_sign]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, full_vals, width, label='CROSS-GRN (Full)', 
               color='#8da0cb', edgecolor='black')
bars2 = ax.bar(x + width/2, ab_vals, width, label='CROSS-GRN (No Cell Type)', 
               color='#e78ac3', edgecolor='black')

ax.set_ylabel('Score', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()
ax.set_title('Ablation Study: Effect of Cell Type Conditioning', fontsize=13, fontweight='bold')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'figure_2_ablation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'figure_2_ablation.png', dpi=300, bbox_inches='tight')
print(f"Saved: {FIGURES_DIR / 'figure_2_ablation.pdf'}")
plt.close()

# Create table
print("\n" + "="*70)
print("RESULTS TABLE")
print("="*70)
print(f"{'Method':<25} {'AUROC':>12} {'AUPRC':>12} {'Sign Acc':>12}")
print("-"*70)
for method, label in zip(methods, labels):
    auroc = agg[method]['auroc']['mean']
    auroc_std = agg[method]['auroc']['std']
    auprc = agg[method]['auprc']['mean']
    auprc_std = agg[method]['auprc']['std']
    sign = agg[method]['sign_accuracy']['mean']
    sign_std = agg[method]['sign_accuracy']['std']
    
    auroc_str = f"{auroc:.4f}±{auroc_std:.4f}" if auroc_std > 0 else f"{auroc:.4f}"
    auprc_str = f"{auprc:.4f}±{auprc_std:.4f}" if auprc_std > 0 else f"{auprc:.4f}"
    sign_str = f"{sign:.4f}±{sign_std:.4f}" if sign_std > 0 else f"{sign:.4f}"
    
    print(f"{label:<25} {auroc_str:>12} {auprc_str:>12} {sign_str:>12}")
print("="*70)

print("\nFigures generated successfully!")
