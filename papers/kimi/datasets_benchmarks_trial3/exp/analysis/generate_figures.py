#!/usr/bin/env python3
"""
Generate paper figures.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Figure 1: Spearman comparison
experiments = results['experiments']
methods = []
spearman_means = []
spearman_stds = []

for name in ['baseline_random', 'baseline_independent_mat', 'baseline_metadata_regression', 
             'baseline_deep_cat_style', 'popbench_adaptive', 'popbench_zeroshot']:
    if name in experiments:
        data = experiments[name]
        metrics = data.get('metrics', {})
        spearman = metrics.get('spearman', {})
        if spearman:
            methods.append(name.replace('baseline_', '').replace('popbench_', '').replace('_', ' ').title())
            spearman_means.append(spearman.get('mean', 0))
            spearman_stds.append(spearman.get('std', 0))

ax1 = axes[0]
x = np.arange(len(methods))
bars = ax1.bar(x, spearman_means, yerr=spearman_stds, capsize=5, color='steelblue', alpha=0.7)
ax1.axhline(y=0.7, color='r', linestyle='--', label='Target (0.7)')
ax1.set_xlabel('Method', fontsize=12)
ax1.set_ylabel('Spearman Correlation', fontsize=12)
ax1.set_title('Ranking Accuracy Comparison', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.legend()
ax1.set_ylim(0, 1)

# Add value labels
for bar, mean, std in zip(bars, spearman_means, spearman_stds):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
             f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# Figure 2: Ablation study
ax2 = axes[1]
ablation_methods = []
ablation_means = []
ablation_stds = []

for name in ['popbench_zeroshot', 'ablation_no_metadata']:
    if name in experiments:
        data = experiments[name]
        metrics = data.get('metrics', {})
        spearman = metrics.get('spearman', {})
        if spearman:
            ablation_methods.append('With Metadata' if 'no_metadata' not in name else 'Without Metadata')
            ablation_means.append(spearman.get('mean', 0))
            ablation_stds.append(spearman.get('std', 0))

if ablation_methods:
    x = np.arange(len(ablation_methods))
    bars = ax2.bar(x, ablation_means, yerr=ablation_stds, capsize=5, color=['green', 'orange'], alpha=0.7)
    ax2.set_xlabel('Condition', fontsize=12)
    ax2.set_ylabel('Spearman Correlation', fontsize=12)
    ax2.set_title('Ablation: Metadata Network Impact', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ablation_methods)
    ax2.set_ylim(-0.1, 0.6)
    
    for bar, mean, std in zip(bars, ablation_means, ablation_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/comparison.pdf', bbox_inches='tight')
print("Saved figures/comparison.png and figures/comparison.pdf")

# Create summary table as text
with open('figures/results_table.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PopBench Experimental Results\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Method':<40} {'Spearman':>15} {'MAE':>15} {'Items':>8}\n")
    f.write("-"*80 + "\n")
    
    for row in results['summary_table']:
        f.write(f"{row['Method']:<40} {row['Spearman']:>15} {row['MAE']:>15} {row['Items']:>8}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Key Findings:\n")
    f.write("="*80 + "\n")
    
    for key, value in results['key_findings'].items():
        f.write(f"✓ {key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("Success Criteria Evaluation:\n")
    f.write("="*80 + "\n")
    
    for criterion, eval_data in results['success_criteria_evaluation'].items():
        status = '✓' if eval_data['achieved'] else '✗'
        f.write(f"{status} {criterion}:\n")
        f.write(f"   {eval_data['note']}\n\n")

print("Saved figures/results_table.txt")
