#!/usr/bin/env python3
"""Update figures with 3-seed SCD data."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
FIG_DIR = WORKSPACE / 'figures'
EXP_DIR = WORKSPACE / 'exp'

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 150
})

# Load all results
def load_results(method_dir, seeds=[42,43,44]):
    results = {}
    for seed in seeds:
        p = EXP_DIR / method_dir / f'results_seed{seed}.json'
        if p.exists():
            results[seed] = json.load(open(str(p)))
    return results

cd = load_results('cd_baseline_v2')
ph = load_results('cd_pseudohuber_v2')
rf = load_results('rectflow_baseline_v2')
scd = load_results('scd_adaptive_v2')

# Also try to load CD-100step if available
cd100 = load_results('cd_100step_teacher')

methods = {
    'CD (MSE)': cd,
    'CD (Pseudo-Huber)': ph,
    'SCD (Ours)': scd,
}

if cd100:
    methods['CD (100-step teacher)'] = cd100

colors = {'CD (MSE)': '#1f77b4', 'CD (Pseudo-Huber)': '#ff7f0e',
           'SCD (Ours)': '#d62728', 'CD (100-step teacher)': '#2ca02c'}

def get_stats(data, key):
    vals = [data[s][key] for s in data]
    return np.mean(vals), np.std(vals) if len(vals) > 1 else 0

# ===== Figure 1: Spectral Error Analysis =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Spectral Error Analysis', fontsize=16, fontweight='bold')

bands = ['Low', 'Mid-Low', 'Mid-High', 'High']
x = np.arange(len(bands))
width = 0.2

for step_label, step_key, ax in [('1-step', '1_step', axes[0]), ('4-step', '4_step', axes[1])]:
    ax.set_title(f'{step_label} Generation', fontsize=14)
    for i, (name, data) in enumerate(methods.items()):
        if name == 'CD (100-step teacher)' and not data:
            continue
        means = []
        stds = []
        for b in range(4):
            vals = [data[s][step_key]['per_band_mse'][b] for s in data]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=name,
               color=colors[name], capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel('Per-Band MSE')
    ax.set_xlabel('Frequency Band')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(str(FIG_DIR / 'figure1_spectral_error.pdf'), bbox_inches='tight')
fig.savefig(str(FIG_DIR / 'figure1_spectral_error.png'), bbox_inches='tight')
plt.close()

# ===== Figure 4: FID vs Steps =====
fig, ax = plt.subplots(figsize=(8, 6))

all_methods = {
    'Rectified Flow': rf,
    'Standard CD': cd,
    'Pseudo-Huber CD': ph,
    'SCD (Ours)': scd,
}
if cd100:
    all_methods['CD (100-step teacher)'] = cd100

fid_colors = {
    'Rectified Flow': '#7f7f7f', 'Standard CD': '#1f77b4',
    'Pseudo-Huber CD': '#ff7f0e', 'SCD (Ours)': '#2ca02c',
    'CD (100-step teacher)': '#9467bd'
}
markers = {'Rectified Flow': 'o', 'Standard CD': 's', 'Pseudo-Huber CD': '^',
           'SCD (Ours)': 'D', 'CD (100-step teacher)': 'v'}

steps = [1, 2, 4]
for name, data in all_methods.items():
    if not data:
        continue
    means = []
    stds_list = []
    for ns in steps:
        key = f'{ns}_step'
        vals = [data[s][key]['fid'] for s in data]
        means.append(np.mean(vals))
        stds_list.append(np.std(vals) if len(vals) > 1 else 0)
    ax.errorbar(steps, means, yerr=stds_list, label=name,
                color=fid_colors[name], marker=markers[name],
                linewidth=2, markersize=8, capsize=4)

ax.set_xlabel('Number of Inference Steps')
ax.set_ylabel('FID (lower is better)')
ax.set_title('FID vs. Number of Inference Steps', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 4])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(str(FIG_DIR / 'figure4_fid_vs_steps.pdf'), bbox_inches='tight')
fig.savefig(str(FIG_DIR / 'figure4_fid_vs_steps.png'), bbox_inches='tight')
plt.close()

# ===== Update CSV tables =====
# Table 1
with open(str(FIG_DIR / 'table1_main_results.csv'), 'w') as f:
    f.write('Method,1-step FID,2-step FID,4-step FID\n')
    for name, data in [('Rectified Flow', rf), ('CD (MSE)', cd), ('CD (Pseudo-Huber)', ph), ('SCD (Ours)', scd)]:
        row = [name]
        for ns in ['1_step', '2_step', '4_step']:
            vals = [data[s][ns]['fid'] for s in data]
            m, s = np.mean(vals), np.std(vals) if len(vals) > 1 else 0
            row.append(f'{m:.2f} +/- {s:.2f}')
        f.write(','.join(row) + '\n')
    if cd100:
        row = ['CD (100-step teacher)']
        for ns in ['1_step', '2_step', '4_step']:
            vals = [cd100[s][ns]['fid'] for s in cd100]
            m, s = np.mean(vals), np.std(vals) if len(vals) > 1 else 0
            row.append(f'{m:.2f} +/- {s:.2f}')
        f.write(','.join(row) + '\n')

print("Figures and tables updated successfully!")
print(f"SCD seeds: {list(scd.keys())}")
if cd100:
    print(f"CD-100step seeds: {list(cd100.keys())}")
    for s in cd100:
        print(f"  seed {s}: 1-step FID = {cd100[s]['1_step']['fid']:.2f}")
