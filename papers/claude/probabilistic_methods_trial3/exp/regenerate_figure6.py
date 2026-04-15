"""Regenerate figure 6 with real Yahoo Finance data."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load real-world results
with open('exp/experiment4_realworld/results_real.json') as f:
    real_data = json.load(f)

# Also load original results for latency
with open('exp/experiment4_realworld/results.json') as f:
    orig_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Financial VaR (real S&P 500 data)
ax = axes[0]
seed_data = real_data['financial'][0]  # seed 0 (chronological order)
checkpoints = seed_data['checkpoints']

ax.fill_between(checkpoints, seed_data['aq_ci_lower'], seed_data['aq_ci_upper'],
                alpha=0.3, color='tab:blue', label=r'AdaQuantCS')
ax.fill_between(checkpoints, seed_data['fm_ci_lower'], seed_data['fm_ci_upper'],
                alpha=0.3, color='tab:orange', label='Full-Memory CS')
ax.fill_between(checkpoints, seed_data['bs_ci_lower'], seed_data['bs_ci_upper'],
                alpha=0.3, color='tab:green', label='Reservoir Bootstrap')

ax.axhline(y=seed_data['empirical_quantile'], color='red', linestyle='--',
           linewidth=1.5, label=f'Empirical $Q_{{0.05}}$={seed_data["empirical_quantile"]:.4f}')

ax.set_xlabel('Observations', fontsize=12)
ax.set_ylabel('5th percentile (VaR)', fontsize=12)
ax.set_title('Financial VaR: S&P 500 Daily Log Returns\n(Yahoo Finance, 2000-2024)', fontsize=12)
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(checkpoints[0], checkpoints[-1])

# Panel 2: Network latency p99
ax = axes[1]
seed_data = orig_data['latency'][0]  # seed 0
checkpoints = seed_data['checkpoints']
true_q = seed_data['true_quantile']

ax.fill_between(checkpoints, seed_data['aq_ci_lower'], seed_data['aq_ci_upper'],
                alpha=0.3, color='tab:blue', label=r'AdaQuantCS')
ax.fill_between(checkpoints, seed_data['fm_ci_lower'], seed_data['fm_ci_upper'],
                alpha=0.3, color='tab:orange', label='Full-Memory CS')
# Don't plot bootstrap CI for latency as it's too wide and ruins the scale

ax.axhline(y=true_q, color='red', linestyle='--',
           linewidth=1.5, label=f'True $Q_{{0.99}}$={true_q:.1f}')

ax.set_xlabel('Observations', fontsize=12)
ax.set_ylabel('p99 Latency (ms)', fontsize=12)
ax.set_title('Network Latency p99 Monitoring\n(Synthetic log-normal mixture)', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(checkpoints[0], checkpoints[-1])

plt.tight_layout()
plt.savefig('figures/figure6_realworld.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure6_realworld.png', dpi=300, bbox_inches='tight')
print("Saved figures/figure6_realworld.pdf and .png")
