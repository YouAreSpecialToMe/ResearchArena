#!/usr/bin/env python3
"""Statistical analysis, figures, and final results.json generation."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# Style
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 12, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 9, 'figure.dpi': 300,
})
sns.set_style('whitegrid')
PALETTE = sns.color_palette('colorblind', 8)
METHOD_COLORS = {
    'retrain': PALETTE[7], 'ft': PALETTE[0], 'ga': PALETTE[1],
    'rl': PALETTE[2], 'scrub': PALETTE[3], 'neggrad': PALETTE[4],
}
METHOD_LABELS = {
    'retrain': 'Retrain', 'ft': 'Fine-Tune', 'ga': 'Grad. Ascent',
    'rl': 'Random Labels', 'scrub': 'SCRUB', 'neggrad': 'NegGrad+KD',
}
DATASETS = ['cifar10', 'cifar100', 'purchase100']
DS_LABELS = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100', 'purchase100': 'Purchase-100'}

# ============================================================
# Figure 1: Stratified MIA — the difficulty-dependent privacy gap
# ============================================================
def figure1():
    strat = load_json('exp/results/mia_stratified.json')
    methods = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        x = np.arange(5)  # Q1-Q5
        width = 0.12
        offsets = np.arange(len(methods)) - (len(methods)-1)/2

        for m_idx, method in enumerate(methods):
            per_q_seeds = {q: [] for q in range(5)}
            for entry in strat:
                if entry['method'] == method and entry['dataset'] == ds and entry.get('variant') == 'standard':
                    for q in range(5):
                        qkey = f'q{q+1}'
                        if qkey in entry:
                            per_q_seeds[q].append(entry[qkey]['best_auc'])

            means = [np.mean(per_q_seeds[q]) if per_q_seeds[q] else 0.5 for q in range(5)]
            stds = [np.std(per_q_seeds[q]) if len(per_q_seeds[q]) > 1 else 0 for q in range(5)]

            ax.bar(x + offsets[m_idx] * width, means, width, yerr=stds,
                   color=METHOD_COLORS.get(method, 'gray'), label=METHOD_LABELS.get(method, method),
                   capsize=2, alpha=0.85)

        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Difficulty Quintile')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\n(Easy)', 'Q2', 'Q3', 'Q4', 'Q5\n(Hard)'])
        ax.set_title(DS_LABELS[ds])
        if ax_idx == 0:
            ax.set_ylabel('MIA AUC')

    axes[0].legend(loc='upper left', framealpha=0.9)
    fig.suptitle('Per-Quintile MIA AUC: Difficulty-Dependent Privacy Gap', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig('figures/figure1_stratified_mia.pdf', bbox_inches='tight')
    fig.savefig('figures/figure1_stratified_mia.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 1 saved.")


# ============================================================
# Figure 2: Aggregate vs Worst-Quintile scatter
# ============================================================
def figure2():
    strat = load_json('exp/results/mia_stratified.json')
    methods = ['ft', 'ga', 'rl', 'scrub', 'neggrad']
    markers = {'cifar10': 'o', 'cifar100': 's', 'purchase100': 'D'}

    fig, ax = plt.subplots(figsize=(7, 6))

    for entry in strat:
        method = entry.get('method')
        ds = entry.get('dataset')
        if method not in methods or entry.get('variant') != 'standard':
            continue

        agg_auc = entry.get('aggregate', {}).get('best_auc', None)
        wq_auc = entry.get('wq_auc', None)
        if agg_auc is None or wq_auc is None:
            continue

        ax.scatter(agg_auc, wq_auc, c=[METHOD_COLORS.get(method, 'gray')],
                   marker=markers.get(ds, 'o'), s=60, alpha=0.8,
                   edgecolors='black', linewidths=0.5)

    # Diagonal
    lims = [0.4, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.4, linewidth=1)
    ax.fill_between(lims, lims, [1.0, 1.0], alpha=0.05, color='red')

    # Legend for methods
    for method in methods:
        ax.scatter([], [], c=[METHOD_COLORS[method]], label=METHOD_LABELS[method], s=60)
    for ds, marker in markers.items():
        ax.scatter([], [], c='gray', marker=marker, label=DS_LABELS[ds], s=60)

    ax.set_xlabel('Aggregate MIA AUC')
    ax.set_ylabel('Worst-Quintile MIA AUC')
    ax.set_title('Aggregate vs. Worst-Quintile AUC')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    fig.savefig('figures/figure2_aggregate_vs_wq.pdf', bbox_inches='tight')
    fig.savefig('figures/figure2_aggregate_vs_wq.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 2 saved.")


# ============================================================
# Figure 3: DAU defense effectiveness
# ============================================================
def figure3():
    strat_std = load_json('exp/results/mia_stratified.json')
    strat_def = load_json('exp/results/mia_stratified_defense.json')
    all_strat = strat_std + strat_def

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    variants = ['standard', 'dau', 'rum']
    variant_labels = {'standard': 'Standard', 'dau': 'DAU', 'rum': 'RUM'}
    variant_colors = {'standard': PALETTE[0], 'dau': PALETTE[2], 'rum': PALETTE[4]}

    for ax_idx, ds in enumerate(DATASETS):
        ax = axes[ax_idx]
        x_positions = []
        x_labels = []
        pos = 0

        for method in ['ga', 'scrub']:
            for v_idx, variant in enumerate(variants):
                if variant == 'standard':
                    key = method
                    var_check = 'standard'
                elif variant == 'dau':
                    key = f'{method}_dau'
                    var_check = 'defense'
                else:
                    key = f'{method}_rum'
                    var_check = 'defense'

                wq_vals = []
                dg_vals = []
                for entry in all_strat:
                    if entry['method'] == key and entry['dataset'] == ds:
                        if variant == 'standard' and entry.get('variant') == 'standard':
                            wq_vals.append(entry['wq_auc'])
                            dg_vals.append(entry['dg'])
                        elif variant != 'standard' and entry.get('variant') == 'defense':
                            wq_vals.append(entry['wq_auc'])
                            dg_vals.append(entry['dg'])

                mean_wq = np.mean(wq_vals) if wq_vals else 0
                std_wq = np.std(wq_vals) if len(wq_vals) > 1 else 0

                ax.bar(pos, mean_wq, 0.7, yerr=std_wq, color=variant_colors[variant],
                       capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)
                x_positions.append(pos)
                x_labels.append(f'{METHOD_LABELS[method]}\n{variant_labels[variant]}')
                pos += 1
            pos += 0.5

        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha='right')
        ax.set_title(DS_LABELS[ds])
        if ax_idx == 0:
            ax.set_ylabel('Worst-Quintile MIA AUC')

    fig.suptitle('DAU Defense: Worst-Quintile AUC Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig('figures/figure3_dau_defense.pdf', bbox_inches='tight')
    fig.savefig('figures/figure3_dau_defense.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 3 saved.")


# ============================================================
# Figure 4: Alpha sensitivity
# ============================================================
def figure4():
    alpha_path = 'exp/results/ablation_alpha.json'
    if not os.path.exists(alpha_path):
        print("Ablation alpha results not found, skipping Figure 4.")
        return

    alpha_data = load_json(alpha_path)

    # Also need baseline (alpha=0) and default (alpha=1) from main results
    strat_std = load_json('exp/results/mia_stratified.json')
    strat_def = load_json('exp/results/mia_stratified_defense.json')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    line_styles = {'cifar10': '-', 'cifar100': '--'}

    for m_idx, method in enumerate(['ga', 'scrub']):
        ax = axes[m_idx]

        for ds in ['cifar10', 'cifar100']:
            # Collect all alphas
            alpha_wq = {}
            alpha_ra = {}

            # alpha=0 (standard baseline)
            for entry in strat_std:
                if entry['method'] == method and entry['dataset'] == ds and entry.get('variant') == 'standard':
                    alpha_wq.setdefault(0.0, []).append(entry['wq_auc'])

            # alpha=1 (DAU default)
            for entry in strat_def:
                if entry['method'] == f'{method}_dau' and entry['dataset'] == ds:
                    alpha_wq.setdefault(1.0, []).append(entry['wq_auc'])

            # Load retain acc for alpha=0
            baselines = load_json('exp/results/unlearning_baselines.json')
            for entry in baselines:
                if entry['method'] == method and entry['dataset'] == ds:
                    alpha_ra.setdefault(0.0, []).append(entry['retain_acc'])

            dau_baselines = load_json('exp/results/dau_rum_baselines.json')
            for entry in dau_baselines:
                if entry['method'] == f'{method}_dau' and entry['dataset'] == ds:
                    alpha_ra.setdefault(1.0, []).append(entry['retain_acc'])

            # Other alphas from ablation
            for entry in alpha_data:
                if entry['method'] == method and entry['dataset'] == ds:
                    a = entry['alpha']
                    alpha_wq.setdefault(a, []).append(entry['wq_auc'])
                    alpha_ra.setdefault(a, []).append(entry['retain_acc'])

            alphas_sorted = sorted(alpha_wq.keys())
            wq_means = [np.mean(alpha_wq[a]) for a in alphas_sorted]
            wq_stds = [np.std(alpha_wq[a]) if len(alpha_wq[a]) > 1 else 0 for a in alphas_sorted]

            color = PALETTE[0] if ds == 'cifar10' else PALETTE[1]
            ax.errorbar(alphas_sorted, wq_means, yerr=wq_stds,
                       marker='o', color=color, linestyle=line_styles[ds],
                       label=f'{DS_LABELS[ds]} WQ-AUC', capsize=3)

            # Secondary axis for RA
            if alpha_ra:
                ax2 = ax.twinx()
                ra_sorted = sorted(alpha_ra.keys())
                ra_means = [np.mean(alpha_ra[a]) for a in ra_sorted]
                ax2.plot(ra_sorted, ra_means, marker='s', color=color, linestyle=':',
                        alpha=0.5, label=f'{DS_LABELS[ds]} RA')
                if m_idx == 1:
                    ax2.set_ylabel('Retain Accuracy')

        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('α (DAU strength)')
        ax.set_ylabel('Worst-Quintile MIA AUC')
        ax.set_title(f'{METHOD_LABELS[method]}-DAU')
        ax.legend(loc='upper left', fontsize=8)

    fig.suptitle('Alpha Sensitivity: Privacy-Utility Tradeoff', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig('figures/figure4_alpha_sensitivity.pdf', bbox_inches='tight')
    fig.savefig('figures/figure4_alpha_sensitivity.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 4 saved.")


# ============================================================
# Figure 5: Ablation summary (2x2)
# ============================================================
def figure5():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) K vs metrics
    k_path = 'exp/results/ablation_K.json'
    if os.path.exists(k_path):
        k_data = load_json(k_path)
        ax = axes[0, 0]
        ks = [2, 4, 8]
        rhos = [k_data[f'K={k}']['spearman_rho'] for k in ks]
        stabs = [k_data[f'K={k}']['quintile_stability'] for k in ks]
        ax.plot(ks, rhos, 'o-', color=PALETTE[0], label='Spearman ρ')
        ax.plot(ks, stabs, 's--', color=PALETTE[1], label='Quintile Stability')
        ax.set_xlabel('K (reference models)')
        ax.set_ylabel('Score')
        ax.set_title('(a) Reference Model Count')
        ax.legend()
        ax.set_xticks(ks)

    # (b) Strata granularity
    strata_path = 'exp/results/ablation_strata.json'
    if os.path.exists(strata_path):
        strata_data = load_json(strata_path)
        ax = axes[0, 1]
        names = ['terciles', 'quintiles', 'deciles']
        for method in ['ga', 'scrub']:
            wq_vals = []
            for name in names:
                if name in strata_data and method in strata_data[name].get('methods', {}):
                    wq_vals.append(strata_data[name]['methods'][method]['wq_auc'])
                else:
                    wq_vals.append(0)
            ax.plot([3, 5, 10], wq_vals, 'o-', label=METHOD_LABELS[method])
        ax.set_xlabel('Number of Strata')
        ax.set_ylabel('Worst-Stratum AUC')
        ax.set_title('(b) Stratification Granularity')
        ax.legend()

    # (c) Forget set size
    fsize_path = 'exp/results/ablation_forget_size.json'
    if os.path.exists(fsize_path):
        fsize_data = load_json(fsize_path)
        ax = axes[1, 0]
        for method in ['ga', 'scrub']:
            for ds in ['cifar10', 'cifar100']:
                dg_vals = {}
                for entry in fsize_data:
                    if entry['method'] == method and entry['dataset'] == ds and entry['variant'] == 'standard':
                        dg_vals[entry['forget_size']] = entry['dg']
                if dg_vals:
                    sizes = sorted(dg_vals.keys())
                    ax.plot(sizes, [dg_vals[s] for s in sizes], 'o-',
                           label=f'{METHOD_LABELS[method]} ({DS_LABELS[ds]})')
        # Include size=1000 from main results
        ax.set_xlabel('Forget Set Size')
        ax.set_ylabel('Difficulty Gap (DG)')
        ax.set_title('(c) Forget Set Size')
        ax.legend(fontsize=8)

    # (d) Random-weight control
    rw_path = 'exp/results/ablation_random_weights.json'
    if os.path.exists(rw_path):
        rw_data = load_json(rw_path)
        ax = axes[1, 1]
        # Compare: standard GA, DAU-GA, random-weight GA
        strat_std = load_json('exp/results/mia_stratified.json')
        strat_def = load_json('exp/results/mia_stratified_defense.json')

        labels = ['Standard', 'DAU (true)', 'DAU (random)']
        wq_means = []
        wq_stds = []

        # Standard GA on cifar10
        std_wq = [e['wq_auc'] for e in strat_std
                  if e['method'] == 'ga' and e['dataset'] == 'cifar10' and e.get('variant') == 'standard']
        wq_means.append(np.mean(std_wq) if std_wq else 0)
        wq_stds.append(np.std(std_wq) if len(std_wq) > 1 else 0)

        # DAU GA on cifar10
        dau_wq = [e['wq_auc'] for e in strat_def
                  if e['method'] == 'ga_dau' and e['dataset'] == 'cifar10']
        wq_means.append(np.mean(dau_wq) if dau_wq else 0)
        wq_stds.append(np.std(dau_wq) if len(dau_wq) > 1 else 0)

        # Random weight
        rw_wq = [e['wq_auc'] for e in rw_data]
        wq_means.append(np.mean(rw_wq) if rw_wq else 0)
        wq_stds.append(np.std(rw_wq) if len(rw_wq) > 1 else 0)

        colors = [PALETTE[0], PALETTE[2], PALETTE[4]]
        bars = ax.bar(labels, wq_means, yerr=wq_stds, color=colors, capsize=4, alpha=0.85)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_ylabel('Worst-Quintile MIA AUC')
        ax.set_title('(d) DAU Specificity Control')

    plt.tight_layout()
    fig.savefig('figures/figure5_ablations.pdf', bbox_inches='tight')
    fig.savefig('figures/figure5_ablations.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 5 saved.")


# ============================================================
# Figure 6: Difficulty distribution
# ============================================================
def figure6():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_idx, ds in enumerate(DATASETS):
        diff = np.load(f'exp/results/difficulty_{ds}_train.npy')
        quint = np.load(f'exp/results/quintiles_{ds}_train.npy')
        ax = axes[ax_idx]
        for q in range(5):
            mask = quint == q
            ax.hist(diff[mask], bins=50, alpha=0.5, label=f'Q{q+1}', density=True)
        ax.set_xlabel('Difficulty Score (mean CE loss)')
        if ax_idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(DS_LABELS[ds])
        ax.legend(fontsize=8)

    fig.suptitle('Difficulty Score Distributions by Quintile', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig('figures/figure6_difficulty_distribution.pdf', bbox_inches='tight')
    fig.savefig('figures/figure6_difficulty_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 6 saved.")


# ============================================================
# Table 1: Main results (LaTeX)
# ============================================================
def table1():
    strat_std = load_json('exp/results/mia_stratified.json')
    strat_def = load_json('exp/results/mia_stratified_defense.json')
    all_strat = strat_std + strat_def

    baselines = load_json('exp/results/unlearning_baselines.json')
    dau_data = load_json('exp/results/dau_rum_baselines.json')

    # Collect: method -> dataset -> {agg, wq, dg, ra, ta} across seeds
    all_methods = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad',
                   'ga_dau', 'scrub_dau', 'ga_rum', 'scrub_rum']
    method_display = {
        'retrain': 'Retrain', 'ft': 'Fine-Tune', 'ga': 'Grad. Ascent',
        'rl': 'Random Labels', 'scrub': 'SCRUB', 'neggrad': 'NegGrad+KD',
        'ga_dau': 'GA-DAU', 'scrub_dau': 'SCRUB-DAU',
        'ga_rum': 'GA-RUM', 'scrub_rum': 'SCRUB-RUM',
    }

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Main results: aggregate MIA AUC, worst-quintile AUC (WQ), difficulty gap (DG), and utility. Mean $\pm$ std over 3 seeds.}',
        r'\label{tab:main}',
        r'\resizebox{\textwidth}{!}{',
        r'\begin{tabular}{l' + 'ccccc' * 3 + '}',
        r'\toprule',
    ]

    # Header row
    header = r'Method'
    for ds in DATASETS:
        header += f' & \\multicolumn{{5}}{{c}}{{{DS_LABELS[ds]}}}'
    header += r' \\'
    lines.append(header)

    subheader = ''
    for _ in DATASETS:
        subheader += r' & Agg & WQ & DG & RA & TA'
    subheader += r' \\'
    lines.append(subheader)
    lines.append(r'\midrule')

    for method in all_methods:
        row = method_display.get(method, method)
        for ds in DATASETS:
            # Get stratified results
            agg_vals, wq_vals, dg_vals = [], [], []
            for e in all_strat:
                if e['method'] == method and e['dataset'] == ds:
                    agg_vals.append(e['aggregate']['best_auc'])
                    wq_vals.append(e['wq_auc'])
                    dg_vals.append(e['dg'])

            # Get utility
            ra_vals, ta_vals = [], []
            for src in [baselines, dau_data]:
                for e in src:
                    m = e['method']
                    if m == method and e['dataset'] == ds:
                        if 'retain_acc' in e:
                            ra_vals.append(e['retain_acc'])
                        if 'test_acc' in e:
                            ta_vals.append(e['test_acc'])

            # For retrain, get from training log
            if method == 'retrain' and not ta_vals:
                try:
                    tlog = load_json('exp/results/model_training_log.json')
                    for e in tlog:
                        if e.get('type') == 'retrain' and e['dataset'] == ds:
                            ta_vals.append(e['test_acc'])
                except:
                    pass

            def fmt(vals):
                if not vals:
                    return '-'
                m = np.mean(vals)
                s = np.std(vals) if len(vals) > 1 else 0
                return f'{m:.3f}$\\pm${s:.3f}'

            row += f' & {fmt(agg_vals)} & {fmt(wq_vals)} & {fmt(dg_vals)} & {fmt(ra_vals)} & {fmt(ta_vals)}'

        row += r' \\'
        if method in ('neggrad', 'scrub_dau'):
            row += r' \midrule'
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}',
    ])

    tex = '\n'.join(lines)
    with open('figures/table1_main_results.tex', 'w') as f:
        f.write(tex)
    print("Table 1 saved.")


# ============================================================
# Statistical analysis
# ============================================================
def statistical_analysis():
    strat_std = load_json('exp/results/mia_stratified.json')
    strat_def = load_json('exp/results/mia_stratified_defense.json')

    results = {'criteria': []}
    METHODS = ['ft', 'ga', 'rl', 'scrub', 'neggrad']

    # Criterion 1: WQ-AUC > Aggregate AUC
    n_sig = 0
    n_total = 0
    for method in METHODS:
        for ds in DATASETS:
            diffs = []
            for e in strat_std:
                if e['method'] == method and e['dataset'] == ds and e.get('variant') == 'standard':
                    diffs.append(e['wq_auc'] - e['aggregate']['best_auc'])
            if len(diffs) >= 2:
                t_stat, p_val = stats.ttest_1samp(diffs, 0)
                p_onesided = p_val / 2 if t_stat > 0 else 1 - p_val / 2
                mean_diff = np.mean(diffs)
                sig = p_onesided < 0.05 / 15 and mean_diff > 0.05
                if sig:
                    n_sig += 1
                n_total += 1
                results['criteria'].append({
                    'criterion': 1, 'method': method, 'dataset': ds,
                    'mean_overestimation': float(mean_diff),
                    'p_value': float(p_onesided),
                    'significant': sig, 'n_seeds': len(diffs),
                })
    results['criterion1_summary'] = f'{n_sig}/{n_total} significant (target >= 9/15)'

    # Criterion 2: DG significantly != 0
    n_sig2 = 0
    n_total2 = 0
    for method in METHODS:
        for ds in DATASETS:
            dg_vals = []
            for e in strat_std:
                if e['method'] == method and e['dataset'] == ds and e.get('variant') == 'standard':
                    dg_vals.append(e['dg'])
            if len(dg_vals) >= 2:
                t_stat, p_val = stats.ttest_1samp(dg_vals, 0)
                p_corrected = min(p_val * 15, 1.0)
                sig = p_corrected < 0.05
                if sig:
                    n_sig2 += 1
                n_total2 += 1
                results['criteria'].append({
                    'criterion': 2, 'method': method, 'dataset': ds,
                    'mean_dg': float(np.mean(dg_vals)),
                    'p_value_corrected': float(p_corrected),
                    'significant': sig,
                })
    results['criterion2_summary'] = f'{n_sig2}/{n_total2} significant'

    # Criterion 3: DAU reduces WQ-AUC
    for method in ['ga', 'scrub']:
        for ds in DATASETS:
            std_wq = []
            dau_wq = []
            for e in strat_std:
                if e['method'] == method and e['dataset'] == ds and e.get('variant') == 'standard':
                    std_wq.append(e['wq_auc'])
            for e in strat_def:
                if e['method'] == f'{method}_dau' and e['dataset'] == ds:
                    dau_wq.append(e['wq_auc'])

            if len(std_wq) >= 2 and len(dau_wq) >= 2:
                min_len = min(len(std_wq), len(dau_wq))
                delta = [std_wq[i] - dau_wq[i] for i in range(min_len)]
                mean_delta = np.mean(delta)
                t_stat, p_val = stats.ttest_1samp(delta, 0)
                results['criteria'].append({
                    'criterion': 3, 'method': method, 'dataset': ds,
                    'mean_wq_reduction': float(mean_delta),
                    'p_value': float(p_val / 2 if t_stat > 0 else 1.0),
                    'dau_improves': mean_delta > 0,
                })

    save_json(results, 'exp/results/success_criteria.json')
    print("Statistical analysis saved.")
    return results


# ============================================================
# Final results.json
# ============================================================
def generate_results_json():
    """Aggregate everything into workspace-root results.json."""
    results = {
        'title': 'Difficulty-Aware Unlearning: Closing the Per-Sample Privacy Gap',
        'datasets': DATASETS,
        'methods': ['ft', 'ga', 'rl', 'scrub', 'neggrad'],
        'seeds': [42, 123, 456],
    }

    # Main results
    strat_std = load_json('exp/results/mia_stratified.json')
    strat_def = load_json('exp/results/mia_stratified_defense.json')

    main_results = {}
    for entry in strat_std + strat_def:
        method = entry['method']
        ds = entry['dataset']
        key = f'{method}_{ds}'
        if key not in main_results:
            main_results[key] = {'wq_auc': [], 'dg': [], 'agg_auc': []}
        main_results[key]['wq_auc'].append(entry['wq_auc'])
        main_results[key]['dg'].append(entry['dg'])
        main_results[key]['agg_auc'].append(entry['aggregate']['best_auc'])

    # Summarize
    summary = {}
    for key, vals in main_results.items():
        summary[key] = {
            'wq_auc': {'mean': float(np.mean(vals['wq_auc'])), 'std': float(np.std(vals['wq_auc']))},
            'dg': {'mean': float(np.mean(vals['dg'])), 'std': float(np.std(vals['dg']))},
            'agg_auc': {'mean': float(np.mean(vals['agg_auc'])), 'std': float(np.std(vals['agg_auc']))},
        }
    results['main_results'] = summary

    # Ablation results
    for abl_name in ['ablation_K', 'ablation_alpha', 'ablation_strata',
                      'ablation_forget_size', 'ablation_random_weights']:
        path = f'exp/results/{abl_name}.json'
        if os.path.exists(path):
            results[abl_name] = load_json(path)

    # Success criteria
    sc_path = 'exp/results/success_criteria.json'
    if os.path.exists(sc_path):
        results['success_criteria'] = load_json(sc_path)

    save_json(results, 'results.json')
    print("Final results.json saved.")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)

    # Statistical analysis
    statistical_analysis()

    # Figures
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()

    # Tables
    table1()

    # Final aggregated results
    generate_results_json()

    print("\n=== All analysis complete ===")
