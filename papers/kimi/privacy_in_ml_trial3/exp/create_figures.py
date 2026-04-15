#!/usr/bin/env python3
"""
Create figures for the paper.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

# Load results
with open('results.json') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-paper')

# Figure 1: AUC Comparison - LGSA vs TruVRF
fig, ax = plt.subplots(figsize=(8, 6))

categories = ['LGSA\n(Gold Standard)', 'TruVRF\n(Baseline)']
lgsa_aucs = [r['auc'] for r in results['lgsa'] if 'gold_standard' in r.get('unlearn_method', '')]
truvrf_aucs = [r['auc'] for r in results['truvrf']]

means = [np.mean(lgsa_aucs), np.mean(truvrf_aucs)]
stds = [np.std(lgsa_aucs), np.std(truvrf_aucs)]

x = np.arange(len(categories))
bars = ax.bar(x, means, yerr=stds, capsize=10, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')

ax.axhline(y=0.5, color='red', linestyle='--', label='Random (AUC=0.5)', alpha=0.7)
ax.axhline(y=0.85, color='green', linestyle='--', label='Target (AUC=0.85)', alpha=0.7)

ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('LGSA vs TruVRF Verification Performance\n(Gold Standard: Retrain without Forget Set)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim([0.4, 0.9])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/figure1_auc_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/figure1_auc_comparison.png', dpi=300, bbox_inches='tight')
print("Saved Figure 1: AUC Comparison")
plt.close()

# Figure 2: Ablation Study - Individual Metrics
fig, ax = plt.subplots(figsize=(10, 6))

ablation_configs = ['LDS Only', 'GAS Only', 'SRS Only']
ablation_aucs = []
ablation_stds = []

for config in ['lds_only', 'gas_only', 'srs_only']:
    values = [r['auc'] for r in results['ablation'] if config in r.get('config', '')]
    ablation_aucs.append(np.mean(values) if values else 0)
    ablation_stds.append(np.std(values) if values else 0)

x = np.arange(len(ablation_configs))
bars = ax.bar(x, ablation_aucs, yerr=ablation_stds, capsize=10, 
              color=['#F18F01', '#C73E1D', '#3B1F2B'], alpha=0.8, edgecolor='black')

ax.axhline(y=0.5, color='red', linestyle='--', label='Random (AUC=0.5)', alpha=0.7)
ax.axhline(y=0.5227, color='blue', linestyle='--', label='All Three (AUC=0.523)', alpha=0.7)

ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Ablation Study: Individual Metric Performance\n(CIFAR-10 SimpleCNN, Seed 42)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(ablation_configs, fontsize=11)
ax.set_ylim([0.4, 0.6])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

for bar, mean, std in zip(bars, ablation_aucs, ablation_stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/figure2_ablation_study.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/figure2_ablation_study.png', dpi=300, bbox_inches='tight')
print("Saved Figure 2: Ablation Study")
plt.close()

# Figure 3: Verification Time Comparison
fig, ax = plt.subplots(figsize=(8, 6))

categories = ['LGSA', 'TruVRF']
lgsa_times = [r['verify_time'] for r in results['lgsa'] if 'gold_standard' in r.get('unlearn_method', '')]
truvrf_times = [r['verify_time'] for r in results['truvrf']]

means = [np.mean(lgsa_times), np.mean(truvrf_times)]
stds = [np.std(lgsa_times), np.std(truvrf_times)]

x = np.arange(len(categories))
bars = ax.bar(x, means, yerr=stds, capsize=10, color=['#2E86AB', '#A23B72'], alpha=0.8, edgecolor='black')

ax.set_ylabel('Verification Time (seconds)', fontsize=12)
ax.set_title('Verification Time Comparison\n(SimpleCNN, 1000 samples)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
            f'{mean:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/figure3_verification_time.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/figures/figure3_verification_time.png', dpi=300, bbox_inches='tight')
print("Saved Figure 3: Verification Time")
plt.close()

print("\nAll figures saved to results/figures/")
