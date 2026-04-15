#!/usr/bin/env python3
"""Create figures for the paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load aggregated results
with open('./results/aggregated_results.json') as f:
    results = json.load(f)

os.makedirs('./figures', exist_ok=True)

# Figure 1: Accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = list(results.keys())
means = [results[m]['mean'] for m in models]
stds = [results[m]['std'] for m in models]

x = np.arange(len(models))
bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(models)])

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_xlabel('Method', fontsize=12)
ax.set_title('CIFAR-100 Test Accuracy (30 epochs, fast validation)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=15, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 60)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
            f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('./figures/accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('./figures/accuracy_comparison.pdf', bbox_inches='tight')
print('Saved: figures/accuracy_comparison.png/pdf')
plt.close()

# Figure 2: Training curves (for available results)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model_name, ax) in enumerate(zip(['vmamba', 'localmamba'], axes)):
    # Load individual results to get training curves
    for seed in [42, 123, 456]:
        fname = f'./results/{model_name}_seed{seed}.json'
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            epochs = list(range(len(data['test_accs'])))
            ax.plot(epochs, data['test_accs'], alpha=0.5, label=f'Seed {seed}')
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'{model_name.replace("_", " ").title()} Training Curves', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('./figures/training_curves.png', dpi=150, bbox_inches='tight')
plt.savefig('./figures/training_curves.pdf', bbox_inches='tight')
print('Saved: figures/training_curves.png/pdf')
plt.close()

# Figure 3: Model comparison table
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

table_data = []
table_data.append(['Method', 'Accuracy (%)', 'Params (M)', 'Time (min)'])
for model_name in ['vmamba', 'localmamba']:
    r = results[model_name]
    table_data.append([
        model_name.replace('_', ' ').title(),
        f"{r['mean']:.2f} ± {r['std']:.2f}",
        f"{r['n_params']/1e6:.2f}",
        f"{r['avg_time']:.1f}"
    ])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.25, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('CIFAR-100 Results (30 epochs, 3 seeds)', fontsize=14, pad=20)
plt.savefig('./figures/results_table.png', dpi=150, bbox_inches='tight')
plt.savefig('./figures/results_table.pdf', bbox_inches='tight')
print('Saved: figures/results_table.png/pdf')
plt.close()

print('\nAll figures created successfully!')
