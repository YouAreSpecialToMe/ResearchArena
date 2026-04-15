#!/usr/bin/env python3
"""Generate publication-quality figures for DCUA paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULT_DIR = 'exp/results_v3'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Colorblind-friendly palette
COLORS = {
    'retrain': '#999999',
    'ft': '#E69F00',
    'ga': '#56B4E9',
    'rl': '#009E73',
    'scrub': '#F0E442',
    'neggrad': '#0072B2',
    'ga_dau': '#D55E00',
    'scrub_dau': '#CC79A7',
}

METHOD_NAMES = {
    'retrain': 'Retrain',
    'ft': 'Fine-Tune',
    'ga': 'Grad. Ascent',
    'rl': 'Random Labels',
    'scrub': 'SCRUB',
    'neggrad': 'NegGrad+KD',
    'ga_dau': 'GA-DAU',
    'scrub_dau': 'SCRUB-DAU',
}

DS_NAMES = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'purchase100': 'Purchase-100'}

def load_json(name):
    with open(os.path.join(RESULT_DIR, name)) as f:
        return json.load(f)

dcua = load_json('dcua_results.json')
dau = load_json('dau_results.json')
ablations = load_json('ablation_results.json')
stats = load_json('statistical_tests.json')

# ============================================================================
# Figure 1: Stratified MIA per quintile (main motivation)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
methods = ['retrain', 'ft', 'ga', 'scrub', 'neggrad']
quintile_labels = ['Q1\n(Easy)', 'Q2', 'Q3', 'Q4', 'Q5\n(Hard)']

for idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
    ax = axes[idx]
    x = np.arange(5)
    width = 0.15
    offsets = np.arange(len(methods)) - len(methods)/2 + 0.5

    for m_idx, method in enumerate(methods):
        entries = [e for e in dcua if e['method'] == method and e['dataset'] == ds]
        if not entries:
            continue
        # Average per-quintile across seeds
        per_q = np.array([e['per_quintile'] for e in entries])
        means = per_q.mean(axis=0)
        stds = per_q.std(axis=0)

        bars = ax.bar(x + offsets[m_idx] * width, means, width,
                      yerr=stds, capsize=2,
                      label=METHOD_NAMES[method] if idx == 0 else '',
                      color=COLORS[method], edgecolor='black', linewidth=0.5,
                      alpha=0.85)

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Perfect' if idx == 0 else '')
    ax.set_xticks(x)
    ax.set_xticklabels(quintile_labels)
    ax.set_title(DS_NAMES[ds])
    if idx == 0:
        ax.set_ylabel('MIA-AUC')
    ax.set_ylim(0.2, 1.05)

fig.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.08), frameon=True)
fig.suptitle('Per-Quintile MIA Performance: Aggregate Metrics Mask Vulnerability of Hard Samples',
             y=1.14, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure1_stratified_mia.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure1_stratified_mia.png'), bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# ============================================================================
# Figure 2: Aggregate vs Worst-Quintile scatter
# ============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
    for ds_idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
        markers = ['o', 's', '^']
        entries = [e for e in dcua if e['method'] == method and e['dataset'] == ds]
        for e in entries:
            ax.scatter(e['aggregate_auc'], e['wq_auc'],
                      color=COLORS[method], marker=markers[ds_idx],
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5)

# Add diagonal
lims = [0.2, 1.05]
ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='y = x')
ax.fill_between(lims, lims, [1.05, 1.05], alpha=0.1, color='red', label='WQ > Agg\n(overestimation)')

# Legend for methods
for method in ['ft', 'ga', 'rl', 'scrub', 'neggrad']:
    ax.scatter([], [], color=COLORS[method], marker='o', s=60, label=METHOD_NAMES[method])
# Legend for datasets
for ds_idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
    ax.scatter([], [], color='gray', marker=['o', 's', '^'][ds_idx], s=60, label=DS_NAMES[ds])

ax.set_xlabel('Aggregate MIA-AUC')
ax.set_ylabel('Worst-Quintile MIA-AUC')
ax.set_title('Aggregate vs. Worst-Quintile AUC:\nStandard Evaluation Overestimates Privacy')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure2_aggregate_vs_wq.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'figure2_aggregate_vs_wq.png'))
plt.close()
print("Figure 2 saved")

# ============================================================================
# Figure 3: DAU defense (honest reporting)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
    ax = axes[idx]
    methods_plot = ['ga', 'ga_dau', 'scrub', 'scrub_dau']
    x = np.arange(len(methods_plot))

    means, stds = [], []
    for method in methods_plot:
        if 'dau' in method:
            entries = [e for e in dau if e['method'] == method and e['dataset'] == ds]
        else:
            entries = [e for e in dcua if e['method'] == method and e['dataset'] == ds]
        wqs = [e['wq_auc'] for e in entries] if entries else [0.5]
        means.append(np.mean(wqs))
        stds.append(np.std(wqs))

    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[COLORS[m] for m in methods_plot],
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

    # Add delta labels
    for i in [0, 2]:
        delta = means[i] - means[i+1]
        sign = '+' if delta > 0 else ''
        color = 'green' if delta > 0.01 else ('red' if delta < -0.01 else 'gray')
        ax.annotate(f'Δ={sign}{delta:.3f}', xy=(i+0.5, max(means[i], means[i+1]) + stds[i] + 0.02),
                   ha='center', fontsize=8, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods_plot], rotation=30, ha='right')
    ax.set_title(DS_NAMES[ds])
    if idx == 0:
        ax.set_ylabel('Worst-Quintile MIA-AUC (↓ better)')
    ax.set_ylim(0.3, 1.1)

fig.suptitle('DAU Defense: Honest Assessment\n(Positive Δ = DAU improves privacy; Negative Δ = DAU worsens privacy)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure3_dau_defense.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'figure3_dau_defense.png'))
plt.close()
print("Figure 3 saved")

# ============================================================================
# Figure 4: Alpha sensitivity
# ============================================================================
alpha_data = ablations['alpha']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, metric, ylabel, title in [(ax1, 'wq_auc', 'WQ-AUC (↓ better)', 'Worst-Quintile AUC vs. α'),
                                    (ax2, 'forget_acc', 'Forget Accuracy (↓ better)', 'Forget Accuracy vs. α')]:
    alphas = sorted(set(e['alpha'] for e in alpha_data))
    means, stds = [], []
    for a in alphas:
        vals = [e[metric] for e in alpha_data if e['alpha'] == a]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    ax.errorbar(alphas, means, yerr=stds, marker='o', capsize=4, linewidth=2, markersize=8,
                color=COLORS['ga'], label='GA-DAU')
    ax.set_xlabel('α (DAU strength)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='α=0 (baseline)')

ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure4_alpha_sensitivity.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'figure4_alpha_sensitivity.png'))
plt.close()
print("Figure 4 saved")

# ============================================================================
# Figure 5: Ablation summary (2x2 panel)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Random-weight control
ax = axes[0, 0]
rw_data = ablations['random_weight']
# Compare GA, GA-DAU, GA-random on CIFAR-10
ga_wqs = [e['wq_auc'] for e in dcua if e['method'] == 'ga' and e['dataset'] == 'cifar10']
dau_wqs = [e['wq_auc'] for e in dau if e['method'] == 'ga_dau' and e['dataset'] == 'cifar10']
rw_wqs = [e['wq_auc'] for e in rw_data]

methods_rw = ['GA (standard)', 'GA-DAU\n(true difficulty)', 'GA-DAU\n(random weights)']
vals = [np.mean(ga_wqs), np.mean(dau_wqs), np.mean(rw_wqs)]
errs = [np.std(ga_wqs), np.std(dau_wqs), np.std(rw_wqs)]
bars = ax.bar(range(3), vals, yerr=errs, capsize=4,
              color=[COLORS['ga'], COLORS['ga_dau'], '#888888'],
              edgecolor='black', linewidth=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(methods_rw, fontsize=9)
ax.set_ylabel('WQ-AUC')
ax.set_title('(a) Random-Weight Control (CIFAR-10)')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# (b) Stratification granularity
ax = axes[0, 1]
strata_data = ablations['strata']
for method in ['ga', 'scrub']:
    entries = [e for e in strata_data if e['method'] == method]
    ns = [e['n_strata'] for e in entries]
    wqs = [e['wq_auc'] for e in entries]
    ax.plot(ns, wqs, marker='o', linewidth=2, label=METHOD_NAMES[method], color=COLORS[method])
ax.set_xlabel('Number of Strata')
ax.set_ylabel('Worst-Stratum AUC')
ax.set_title('(b) Stratification Granularity (CIFAR-10)')
ax.legend()

# (c) Forget set size
ax = axes[1, 0]
fs_data = ablations['forget_size']
for method in ['ga', 'scrub']:
    entries = [e for e in fs_data if e['method'] == method]
    sizes = [e['forget_size'] for e in entries]
    dgs = [e['dg'] for e in entries]
    ax.plot(sizes, dgs, marker='o', linewidth=2, label=METHOD_NAMES[method], color=COLORS[method])
ax.set_xlabel('Forget Set Size')
ax.set_ylabel('Difficulty Gap (DG)')
ax.set_title('(c) Forget Set Size vs. DG (CIFAR-10)')
ax.legend()

# (d) K reference models
ax = axes[1, 1]
k_data = ablations['K']
for method in ['ga', 'scrub']:
    entries = [e for e in k_data if e['method'] == method]
    ks = [e['K'] for e in entries]
    wqs = [e['wq_auc'] for e in entries]
    ax.plot(ks, wqs, marker='o', linewidth=2, label=f'{METHOD_NAMES[method]} (WQ-AUC)', color=COLORS[method])

# Add correlation axis
ax2 = ax.twinx()
corrs = [k_data[0]['spearman_corr']]  # K=2
corrs.append(1.0)  # K=4
ax2.plot([2, 4], corrs, marker='s', linewidth=2, linestyle='--', color='gray', label='Spearman ρ')
ax2.set_ylabel('Spearman ρ (vs K=4)')
ax2.set_ylim(0.8, 1.05)

ax.set_xlabel('Number of Reference Models (K)')
ax.set_ylabel('WQ-AUC')
ax.set_title('(d) K Reference Models (CIFAR-10)')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure5_ablations.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'figure5_ablations.png'))
plt.close()
print("Figure 5 saved")

# ============================================================================
# Figure 6: Overestimation heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 5))
methods_hm = ['ft', 'ga', 'rl', 'scrub', 'neggrad']
datasets_hm = ['cifar10', 'cifar100', 'purchase100']

overest_matrix = np.zeros((len(methods_hm), len(datasets_hm)))
for i, method in enumerate(methods_hm):
    for j, ds in enumerate(datasets_hm):
        entries = [e for e in dcua if e['method'] == method and e['dataset'] == ds]
        if entries:
            agg = np.mean([e['aggregate_auc'] for e in entries])
            wq = np.mean([e['wq_auc'] for e in entries])
            overest_matrix[i, j] = wq - agg

im = ax.imshow(overest_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=0.35)
ax.set_xticks(range(len(datasets_hm)))
ax.set_xticklabels([DS_NAMES[d] for d in datasets_hm])
ax.set_yticks(range(len(methods_hm)))
ax.set_yticklabels([METHOD_NAMES[m] for m in methods_hm])

for i in range(len(methods_hm)):
    for j in range(len(datasets_hm)):
        val = overest_matrix[i, j]
        color = 'white' if val > 0.2 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=11, fontweight='bold')

plt.colorbar(im, label='Overestimation (WQ-AUC − Aggregate AUC)')
ax.set_title('Privacy Overestimation by Standard Aggregate Metrics', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure6_overestimation_heatmap.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'figure6_overestimation_heatmap.png'))
plt.close()
print("Figure 6 saved")

# ============================================================================
# Table 1: Main results (LaTeX)
# ============================================================================
with open(os.path.join(FIG_DIR, 'table1_main_results.tex'), 'w') as f:
    f.write(r'\begin{table*}[t]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{Main Results: Stratified vs. Aggregate MIA Evaluation. ')
    f.write(r'WQ-AUC = Worst-Quintile AUC (hardest 20\%), DG = Difficulty Gap. ')
    f.write(r'Lower AUC = better privacy. Retrain is the gold standard ($\approx 0.5$). ')
    f.write(r'Results: mean$\pm$std over 3 seeds.}' + '\n')
    f.write(r'\label{tab:main}' + '\n')
    f.write(r'\resizebox{\textwidth}{!}{' + '\n')
    f.write(r'\begin{tabular}{l|ccc|ccc|ccc}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'& \multicolumn{3}{c|}{CIFAR-10} & \multicolumn{3}{c|}{CIFAR-100} & \multicolumn{3}{c}{Purchase-100} \\' + '\n')
    f.write(r'Method & Agg AUC & WQ-AUC & DG & Agg AUC & WQ-AUC & DG & Agg AUC & WQ-AUC & DG \\' + '\n')
    f.write(r'\midrule' + '\n')

    for method in ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad', 'ga_dau', 'scrub_dau']:
        row = METHOD_NAMES.get(method, method)
        if method in ['ga_dau', 'scrub_dau']:
            row = r'\textit{' + row + '}'
        for ds in ['cifar10', 'cifar100', 'purchase100']:
            if 'dau' in method:
                entries = [e for e in dau if e['method'] == method and e['dataset'] == ds]
            else:
                entries = [e for e in dcua if e['method'] == method and e['dataset'] == ds]
            if entries:
                agg_m = np.mean([e['aggregate_auc'] for e in entries])
                agg_s = np.std([e['aggregate_auc'] for e in entries])
                wq_m = np.mean([e['wq_auc'] for e in entries])
                wq_s = np.std([e['wq_auc'] for e in entries])
                dg_m = np.mean([e['dg'] for e in entries])
                dg_s = np.std([e['dg'] for e in entries])
                row += f' & {agg_m:.3f}$\\pm${agg_s:.3f} & {wq_m:.3f}$\\pm${wq_s:.3f} & {dg_m:.3f}$\\pm${dg_s:.3f}'
            else:
                row += ' & -- & -- & --'
        row += r' \\'
        f.write(row + '\n')
        if method == 'neggrad':
            f.write(r'\midrule' + '\n')

    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}}' + '\n')
    f.write(r'\end{table*}' + '\n')

print("Table 1 saved")

# ============================================================================
# Table 2: Ablation summary (LaTeX)
# ============================================================================
with open(os.path.join(FIG_DIR, 'table2_ablations.tex'), 'w') as f:
    f.write(r'\begin{table}[t]' + '\n')
    f.write(r'\centering' + '\n')
    f.write(r'\caption{Ablation Studies (CIFAR-10, GA unlearning unless noted)}' + '\n')
    f.write(r'\label{tab:ablations}' + '\n')
    f.write(r'\begin{tabular}{lcc}' + '\n')
    f.write(r'\toprule' + '\n')
    f.write(r'Setting & WQ-AUC & DG \\' + '\n')
    f.write(r'\midrule' + '\n')

    # Alpha sensitivity
    f.write(r'\textit{DAU strength ($\alpha$)} & & \\' + '\n')
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        entries = [e for e in alpha_data if e['alpha'] == alpha]
        wq = np.mean([e['wq_auc'] for e in entries])
        dg = np.mean([e['dg'] for e in entries])
        f.write(f'~~$\\alpha={alpha}$ & {wq:.4f} & {dg:.4f} \\\\\n')

    f.write(r'\midrule' + '\n')

    # Random-weight control
    f.write(r'\textit{Weight control} & & \\' + '\n')
    f.write(f'~~True difficulty & {np.mean(dau_wqs):.4f} & -- \\\\\n')
    f.write(f'~~Random weights & {np.mean(rw_wqs):.4f} & -- \\\\\n')

    f.write(r'\midrule' + '\n')

    # Strata
    f.write(r'\textit{Strata granularity (GA)} & & \\' + '\n')
    for ns in [3, 5, 10]:
        e = [x for x in strata_data if x['n_strata'] == ns and x['method'] == 'ga']
        if e:
            f.write(f'~~{ns} strata & {e[0]["wq_auc"]:.4f} & {e[0]["dg"]:.4f} \\\\\n')

    f.write(r'\midrule' + '\n')

    # K
    f.write(r'\textit{Reference models K (GA)} & & \\' + '\n')
    for K in [2, 4]:
        e = [x for x in k_data if x['K'] == K and x['method'] == 'ga']
        if e:
            f.write(f'~~K={K} ($\\rho$={e[0]["spearman_corr"]:.3f}) & {e[0]["wq_auc"]:.4f} & {e[0]["dg"]:.4f} \\\\\n')

    f.write(r'\bottomrule' + '\n')
    f.write(r'\end{tabular}' + '\n')
    f.write(r'\end{table}' + '\n')

print("Table 2 saved")
print("\nAll figures and tables generated successfully!")
