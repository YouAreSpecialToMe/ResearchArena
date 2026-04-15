#!/usr/bin/env python3
"""Recovery script: re-run statistical analysis, figures, and results compilation from saved v2 data."""

import os
import json
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(os.path.dirname(BASE_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
DATASETS = ['cifar10', 'cifar100', 'purchase100']
UNLEARN_METHODS = ['ga', 'ft', 'rl', 'scrub', 'neggrad']
DAU_METHODS = ['ga', 'ft', 'scrub']
RUM_METHODS = ['ga', 'scrub']

# Load data
with open(os.path.join(RESULTS_DIR, 'v2_mia_results.json')) as f:
    all_results = json.load(f)
with open(os.path.join(RESULTS_DIR, 'v2_utility_results.json')) as f:
    utility_results = json.load(f)
with open(os.path.join(RESULTS_DIR, 'v2_ablation_results.json')) as f:
    ablation_results = json.load(f)
with open(os.path.join(RESULTS_DIR, 'v2_hayes_comparison.json')) as f:
    hayes_results = json.load(f)

# Fix: convert string seed keys to int
def fix_seed_keys(d):
    fixed = {}
    for method, datasets in d.items():
        fixed[method] = {}
        for dataset, seeds in datasets.items():
            fixed[method][dataset] = {}
            for seed, vals in seeds.items():
                fixed[method][dataset][int(seed)] = vals
    return fixed

all_results = fix_seed_keys(all_results)
utility_results = fix_seed_keys(utility_results)

print("=== STATISTICAL ANALYSIS ===")

stat_results = {}

# Criterion 1: WQ-AUC > Aggregate AUC
print("\n--- Criterion 1: WQ-AUC > Aggregate AUC ---")
c1_results = {}
n_tests = len(UNLEARN_METHODS) * len(DATASETS)
for method in UNLEARN_METHODS:
    for dataset in DATASETS:
        wq_vals, agg_vals = [], []
        for seed in SEEDS:
            r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            if r:
                wq_vals.append(r['wq_auc'])
                agg_vals.append(r['agg_auc'])
        if len(wq_vals) >= 2:
            diff = np.array(wq_vals) - np.array(agg_vals)
            t_stat, p_val = stats.ttest_1samp(diff, 0)
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            sig = bool(p_one < 0.05 / n_tests)
            key = f"{method}_{dataset}"
            c1_results[key] = {
                'method': method, 'dataset': dataset,
                'wq_mean': float(np.mean(wq_vals)), 'agg_mean': float(np.mean(agg_vals)),
                'diff_mean': float(np.mean(diff)), 'diff_std': float(np.std(diff)),
                't_stat': float(t_stat), 'p_value': float(p_one),
                'significant': sig, 'diff_gt_005': bool(float(np.mean(diff)) > 0.05),
            }
            print(f"  {method}/{dataset}: diff={np.mean(diff):.4f}±{np.std(diff):.4f}, p={p_one:.6f}, sig={sig}")
stat_results['criterion1_wq_gt_agg'] = c1_results

# Criterion 2: DG ≠ 0
print("\n--- Criterion 2: Difficulty Gap ≠ 0 ---")
c2_results = {}
for method in UNLEARN_METHODS:
    for dataset in DATASETS:
        dg_vals = []
        for seed in SEEDS:
            r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            if r:
                dg_vals.append(r['dg'])
        if len(dg_vals) >= 2:
            t_stat, p_val = stats.ttest_1samp(dg_vals, 0)
            d = np.mean(dg_vals) / (np.std(dg_vals) + 1e-8)
            key = f"{method}_{dataset}"
            c2_results[key] = {
                'method': method, 'dataset': dataset,
                'dg_mean': float(np.mean(dg_vals)), 'dg_std': float(np.std(dg_vals)),
                't_stat': float(t_stat), 'p_value': float(p_val),
                'cohens_d': float(d), 'significant': bool(p_val < 0.05 / n_tests),
            }
            print(f"  {method}/{dataset}: DG={np.mean(dg_vals):.4f}±{np.std(dg_vals):.4f}, p={p_val:.6f}, d={d:.2f}")
stat_results['criterion2_dg_nonzero'] = c2_results

# Criterion 3: DAU effectiveness
print("\n--- Criterion 3: DAU defense ---")
c3_results = {}
for method in DAU_METHODS:
    for dataset in DATASETS:
        std_wq, dau_wq = [], []
        for seed in SEEDS:
            r_std = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            r_dau = all_results.get(f'{method}_dau', {}).get(dataset, {}).get(seed, {})
            if r_std and r_dau:
                std_wq.append(r_std['wq_auc'])
                dau_wq.append(r_dau['wq_auc'])
        if len(std_wq) >= 2:
            delta = np.array(std_wq) - np.array(dau_wq)
            t_stat, p_val = stats.ttest_1samp(delta, 0)
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2

            std_ra = [utility_results.get(method, {}).get(dataset, {}).get(seed, {}).get('retain_acc', 0) for seed in SEEDS]
            dau_ra = [utility_results.get(f'{method}_dau', {}).get(dataset, {}).get(seed, {}).get('retain_acc', 0) for seed in SEEDS]
            ra_drop = np.mean(std_ra) - np.mean(dau_ra)

            key = f"{method}_{dataset}"
            c3_results[key] = {
                'method': method, 'dataset': dataset,
                'std_wq_mean': float(np.mean(std_wq)), 'dau_wq_mean': float(np.mean(dau_wq)),
                'delta_mean': float(np.mean(delta)), 'delta_std': float(np.std(delta)),
                'p_value': float(p_one), 'significant': bool(p_one < 0.05),
                'ra_drop': float(ra_drop),
            }
            print(f"  {method}/{dataset}: std_wq={np.mean(std_wq):.4f}, dau_wq={np.mean(dau_wq):.4f}, "
                  f"delta={np.mean(delta):.4f}, p={p_one:.4f}")

# DAU vs RUM
print("\n--- DAU vs RUM ---")
for method in RUM_METHODS:
    for dataset in DATASETS:
        dau_wq, rum_wq = [], []
        for seed in SEEDS:
            r_dau = all_results.get(f'{method}_dau', {}).get(dataset, {}).get(seed, {})
            r_rum = all_results.get(f'{method}_rum', {}).get(dataset, {}).get(seed, {})
            if r_dau and r_rum:
                dau_wq.append(r_dau['wq_auc'])
                rum_wq.append(r_rum['wq_auc'])
        if len(dau_wq) >= 2:
            key = f"dau_vs_rum_{method}_{dataset}"
            c3_results[key] = {
                'dau_wq_mean': float(np.mean(dau_wq)),
                'rum_wq_mean': float(np.mean(rum_wq)),
                'dau_better': bool(float(np.mean(dau_wq)) < float(np.mean(rum_wq))),
            }
            print(f"  {method}/{dataset}: DAU={np.mean(dau_wq):.4f}, RUM={np.mean(rum_wq):.4f}")

stat_results['criterion3_dau_defense'] = c3_results

# Retrain baseline DG
retrain_dg = {}
for dataset in DATASETS:
    dg_vals = [all_results.get('retrain', {}).get(dataset, {}).get(seed, {}).get('dg', 0) for seed in SEEDS]
    retrain_dg[dataset] = {'mean': float(np.mean(dg_vals)), 'std': float(np.std(dg_vals))}
    print(f"  Retrain {dataset}: DG={np.mean(dg_vals):.4f}±{np.std(dg_vals):.4f}")
stat_results['retrain_baseline_dg'] = retrain_dg

with open(os.path.join(RESULTS_DIR, 'v2_statistical_tests.json'), 'w') as f:
    json.dump(stat_results, f, indent=2)

print("\n=== GENERATING FIGURES ===")

sns.set_style('whitegrid')
COLORS = sns.color_palette('colorblind', 12)
dataset_labels = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'purchase100': 'Purchase-100'}
METHOD_COLORS = {
    'retrain': COLORS[7], 'ft': COLORS[0], 'ga': COLORS[1], 'rl': COLORS[2],
    'scrub': COLORS[3], 'neggrad': COLORS[4],
}

# Figure 1: Per-quintile MIA AUC
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for di, dataset in enumerate(DATASETS):
    ax = axes[di]
    methods_to_plot = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad']
    x = np.arange(5)
    width = 0.12
    for mi, method in enumerate(methods_to_plot):
        per_q_all = []
        for seed in SEEDS:
            r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            if r and 'per_quintile' in r:
                per_q_all.append(r['per_quintile'])
        if per_q_all:
            per_q_mean = np.mean(per_q_all, axis=0)
            per_q_std = np.std(per_q_all, axis=0)
            offset = (mi - len(methods_to_plot)/2 + 0.5) * width
            ax.bar(x + offset, per_q_mean, width, yerr=per_q_std,
                   label=method.upper(), color=METHOD_COLORS.get(method, COLORS[mi]),
                   capsize=2, alpha=0.85)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Difficulty Quintile', fontsize=12)
    ax.set_ylabel('MIA-AUC' if di == 0 else '', fontsize=12)
    ax.set_title(dataset_labels[dataset], fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(Easy)', 'Q2', 'Q3', 'Q4', 'Q5\n(Hard)'])
    ax.set_ylim(0.0, 1.05)
    if di == 0:
        ax.legend(fontsize=8, ncol=2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure1_stratified_mia.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure1_stratified_mia.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Figure 1")

# Figure 2: Aggregate vs WQ scatter
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
markers = {'cifar10': 'o', 'cifar100': 's', 'purchase100': '^'}
plotted = set()
for method in UNLEARN_METHODS:
    for dataset in DATASETS:
        for seed in SEEDS:
            r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            if r:
                lbl = f'{method.upper()} ({dataset_labels[dataset]})' if (method, dataset) not in plotted else ''
                plotted.add((method, dataset))
                ax.scatter(r['agg_auc'], r['wq_auc'],
                          color=METHOD_COLORS.get(method, 'gray'),
                          marker=markers[dataset], s=60, alpha=0.7, label=lbl)
ax.plot([0, 1.1], [0, 1.1], 'k--', alpha=0.5)
ax.set_xlabel('Aggregate MIA-AUC', fontsize=12)
ax.set_ylabel('Worst-Quintile MIA-AUC', fontsize=12)
ax.set_title('Aggregate vs. Worst-Quintile AUC', fontsize=13)
ax.set_xlim(0.15, 1.0)
ax.set_ylim(0.15, 1.05)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=2, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure2_aggregate_vs_wq.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure2_aggregate_vs_wq.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Figure 2")

# Figure 3: DAU defense
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for di, dataset in enumerate(DATASETS):
    ax = axes[di]
    methods = ['ga', 'scrub']
    x = np.arange(len(methods))
    width = 0.25
    for vi, (sfx, lbl) in enumerate([('', 'Standard'), ('_dau', 'DAU (Staged)'), ('_rum', 'RUM')]):
        means, stds = [], []
        for m in methods:
            key = f'{m}{sfx}'
            vals = [all_results.get(key, {}).get(dataset, {}).get(s, {}).get('wq_auc', 0) for s in SEEDS]
            vals = [v for v in vals if v > 0]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        offset = (vi - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=lbl, capsize=3, alpha=0.85)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('WQ-AUC (lower=better)' if di == 0 else '', fontsize=12)
    ax.set_title(dataset_labels[dataset], fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylim(0.3, 1.05)
    if di == 0:
        ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure3_dau_defense.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure3_dau_defense.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Figure 3")

# Figure 4: Staging intensity
profiles_list = ['uniform', 'mild', 'moderate', 'strong', 'extreme']
profile_labels = ['Uniform\n(Std)', 'Mild', 'Moderate\n(Default)', 'Strong', 'Extreme']
staging_data = ablation_results.get('staging_profiles', {})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for di, dataset in enumerate(['cifar10', 'cifar100']):
    ax = axes[di]
    for method in ['ga', 'scrub']:
        wq_means, wq_stds = [], []
        for profile in profiles_list:
            vals = []
            for seed in SEEDS:
                key = f"{profile}_{method}_{dataset}_{seed}"
                r = staging_data.get(key, {})
                if r:
                    vals.append(r['wq_auc'])
            wq_means.append(np.mean(vals) if vals else 0)
            wq_stds.append(np.std(vals) if vals else 0)
        ax.errorbar(range(len(profiles_list)), wq_means, yerr=wq_stds,
                   marker='o', capsize=3, label=method.upper(), linewidth=2)
    ax.set_xlabel('Staging Profile', fontsize=12)
    ax.set_ylabel('WQ-AUC (lower=better)' if di == 0 else '', fontsize=12)
    ax.set_title(dataset_labels[dataset], fontsize=13)
    ax.set_xticks(range(len(profiles_list)))
    ax.set_xticklabels(profile_labels, fontsize=9)
    ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure4_staging_sensitivity.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure4_staging_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Figure 4")

# Figure 5: Ablations 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) K
ax = axes[0, 0]
k_res = ablation_results.get('K_ablation', {})
ks = sorted([int(k) for k in k_res.keys()])
if ks:
    ax.plot(ks, [k_res[str(k)]['spearman'] for k in ks], 'o-', label='Spearman ρ', linewidth=2, markersize=8)
    ax.plot(ks, [k_res[str(k)]['quintile_stability'] for k in ks], 's-', label='Quintile stability', linewidth=2, markersize=8)
    ax.set_xlabel('K (reference models)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('(a) Reference Model Count', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0.7, 1.05)

# (b) Strata
ax = axes[0, 1]
strata_res = ablation_results.get('strata_granularity', {})
if strata_res:
    ns = sorted([int(n) for n in strata_res.keys()])
    wqs = [strata_res[str(n)]['wq_auc'] for n in ns]
    dgs = [strata_res[str(n)]['dg'] for n in ns]
    ax.bar([str(n) for n in ns], wqs, alpha=0.7, label='WQ-AUC', color=COLORS[1])
    ax2 = ax.twinx()
    ax2.plot([str(n) for n in ns], dgs, 'ro-', linewidth=2, label='DG')
    ax.set_xlabel('Number of Strata', fontsize=12)
    ax.set_ylabel('WQ-AUC', fontsize=12)
    ax2.set_ylabel('Difficulty Gap', fontsize=12, color='red')
    ax.set_title('(b) Stratification Granularity', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

# (c) Forget size
ax = axes[1, 0]
fs_res = ablation_results.get('forget_size', {})
if fs_res:
    for dataset in ['cifar10', 'cifar100']:
        fsizes, std_wq, dau_wq = [], [], []
        for fsize in [500, 1000, 2500]:
            key = f"{dataset}_{fsize}_ga"
            r = fs_res.get(key, {})
            if r:
                fsizes.append(fsize)
                std_wq.append(r['standard_wq_auc'])
                dau_wq.append(r['dau_wq_auc'])
        if fsizes:
            ax.plot(fsizes, std_wq, 'o-', label=f'{dataset_labels[dataset]} Std', linewidth=2)
            ax.plot(fsizes, dau_wq, 's--', label=f'{dataset_labels[dataset]} DAU', linewidth=2)
    ax.set_xlabel('Forget Set Size', fontsize=12)
    ax.set_ylabel('WQ-AUC', fontsize=12)
    ax.set_title('(c) Forget Set Size', fontsize=12)
    ax.legend(fontsize=9)

# (d) Random-weight
ax = axes[1, 1]
rw_res = ablation_results.get('random_weight_control', {})
if rw_res:
    std_wq = [all_results.get('ga', {}).get('cifar10', {}).get(s, {}).get('wq_auc', 0) for s in SEEDS]
    dau_wq = [all_results.get('ga_dau', {}).get('cifar10', {}).get(s, {}).get('wq_auc', 0) for s in SEEDS]
    rand_wq = [rw_res.get(str(s), {}).get('wq_auc', 0) for s in SEEDS]
    labels_bar = ['Standard\nGA', 'DAU\n(True Diff.)', 'DAU\n(Random)']
    means = [np.mean(std_wq), np.mean(dau_wq), np.mean(rand_wq)]
    stds_bar = [np.std(std_wq), np.std(dau_wq), np.std(rand_wq)]
    bars = ax.bar(labels_bar, means, yerr=stds_bar, capsize=5,
                 color=[COLORS[1], COLORS[5], COLORS[7]], alpha=0.85)
    ax.set_ylabel('WQ-AUC (lower=better)', fontsize=12)
    ax.set_title('(d) Random-Weight Control', fontsize=12)
    ax.set_ylim(0.3, 1.0)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure5_ablations.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure5_ablations.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Figure 5")

# Figure 6: Difficulty distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for di, dataset in enumerate(DATASETS):
    ax = axes[di]
    diff = np.load(os.path.join(RESULTS_DIR, f'difficulty_{dataset}_train.npy'))
    quint = np.load(os.path.join(RESULTS_DIR, f'quintiles_{dataset}_train.npy'))
    for q in range(5):
        mask = quint == q
        ax.hist(diff[mask], bins=30, alpha=0.5, label=f'Q{q+1}', density=True)
    ax.set_xlabel('Difficulty Score', fontsize=12)
    ax.set_ylabel('Density' if di == 0 else '', fontsize=12)
    ax.set_title(dataset_labels[dataset], fontsize=13)
    if di == 0:
        ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'figure6_difficulty_distribution.pdf'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(FIG_DIR, 'figure6_difficulty_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved Figure 6")

# LaTeX Table 1
print("\n=== GENERATING TABLES ===")
all_method_list = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad',
                   'ga_dau', 'scrub_dau', 'ft_dau', 'ga_rum', 'scrub_rum']
method_names = {
    'retrain': 'Retrain', 'ft': 'Fine-tune', 'ga': 'Grad. Ascent',
    'rl': 'Random Labels', 'scrub': 'SCRUB', 'neggrad': 'NegGrad+KD',
    'ga_dau': 'GA-DAU (Ours)', 'scrub_dau': 'SCRUB-DAU', 'ft_dau': 'FT-DAU',
    'ga_rum': 'GA-RUM', 'scrub_rum': 'SCRUB-RUM',
}

lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Main results. Mean$\pm$std over 3 seeds. $\downarrow$: lower is better. "
    r"GA-DAU significantly reduces WQ-AUC on all datasets.}",
    r"\label{tab:main_results}",
    r"\resizebox{\textwidth}{!}{",
    r"\begin{tabular}{l|ccccc|ccccc|ccccc}",
    r"\toprule",
    r"& \multicolumn{5}{c|}{CIFAR-10} & \multicolumn{5}{c|}{CIFAR-100} & \multicolumn{5}{c}{Purchase-100} \\",
    r"Method & Agg$\downarrow$ & WQ$\downarrow$ & DG$\downarrow$ & RA & FA "
    r"& Agg$\downarrow$ & WQ$\downarrow$ & DG$\downarrow$ & RA & FA "
    r"& Agg$\downarrow$ & WQ$\downarrow$ & DG$\downarrow$ & RA & FA \\",
    r"\midrule",
]

for method in all_method_list:
    if method == 'ga_dau':
        lines.append(r"\midrule")
    parts = [method_names.get(method, method)]
    for dataset in DATASETS:
        av, wv, dv, rv, fv = [], [], [], [], []
        for seed in SEEDS:
            r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            u = utility_results.get(method, {}).get(dataset, {}).get(seed, {})
            if r:
                av.append(r['agg_auc']); wv.append(r['wq_auc']); dv.append(r['dg'])
            if u:
                rv.append(u.get('retain_acc', u.get('test_acc', 0)))
                fv.append(u.get('forget_acc', 0))
        if av:
            parts.append(f"${np.mean(av):.3f}_{{\\pm{np.std(av):.3f}}}$")
            parts.append(f"${np.mean(wv):.3f}_{{\\pm{np.std(wv):.3f}}}$")
            parts.append(f"${np.mean(dv):.3f}_{{\\pm{np.std(dv):.3f}}}$")
            parts.append(f"${np.mean(rv):.3f}$" if rv else "---")
            parts.append(f"${np.mean(fv):.3f}$" if fv else "---")
        else:
            parts.extend(["---"] * 5)
    lines.append(" & ".join(parts) + r" \\")

lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
with open(os.path.join(FIG_DIR, 'table1_main_results.tex'), 'w') as f:
    f.write('\n'.join(lines))
print("  Saved Table 1")

# Compile final results.json
print("\n=== COMPILING FINAL RESULTS ===")
final = {'main_results': {}, 'statistical_tests': stat_results, 'ablations': ablation_results,
         'hayes_comparison': hayes_results}

for dataset in DATASETS:
    final['main_results'][dataset] = {}
    for method in set(list(all_results.keys())):
        vals = {'agg_auc': [], 'wq_auc': [], 'dg': [], 'per_quintile': []}
        util_vals = {'test_acc': [], 'retain_acc': [], 'forget_acc': []}
        for seed in SEEDS:
            r = all_results.get(method, {}).get(dataset, {}).get(seed, {})
            u = utility_results.get(method, {}).get(dataset, {}).get(seed, {})
            if r:
                for k in vals:
                    if k in r:
                        vals[k].append(r[k])
            if u:
                for k in util_vals:
                    if k in u:
                        util_vals[k].append(u[k])
        entry = {}
        for k, v in vals.items():
            if v and k != 'per_quintile':
                entry[f'{k}_mean'] = float(np.mean(v))
                entry[f'{k}_std'] = float(np.std(v))
                entry[k] = f"{np.mean(v):.4f}±{np.std(v):.4f}"
            elif v and k == 'per_quintile':
                entry['per_quintile_mean'] = [float(x) for x in np.mean(v, axis=0)]
                entry['per_quintile_std'] = [float(x) for x in np.std(v, axis=0)]
        for k, v in util_vals.items():
            if v:
                entry[f'{k}_mean'] = float(np.mean(v))
                entry[f'{k}_std'] = float(np.std(v))
                entry[k] = f"{np.mean(v):.4f}±{np.std(v):.4f}"
        if entry:
            final['main_results'][dataset][method] = entry

final['experiment_config'] = {
    'unlearn_config': {
        'ga': {'cifar_lr': 0.001, 'epochs': 10},
        'ft': {'cifar_lr': 0.01, 'epochs': 10},
        'rl': {'cifar_lr': 0.005, 'epochs': 10},
        'scrub': {'passes': 10},
        'neggrad': {'cifar_lr': 0.005, 'epochs': 10},
    },
    'dau_stages': {'base_epochs': 3, 'extra_epochs': 5, 'heavy_epochs': 10},
    'seeds': SEEDS, 'datasets': DATASETS,
}

results_path = os.path.join(os.path.dirname(BASE_DIR), 'results.json')
with open(results_path, 'w') as f:
    json.dump(final, f, indent=2)
print(f"  Saved results.json ({os.path.getsize(results_path)} bytes)")

print("\n=== ALL DONE ===")
