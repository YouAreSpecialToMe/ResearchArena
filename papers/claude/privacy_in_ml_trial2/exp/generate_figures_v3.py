#!/usr/bin/env python3
"""Generate publication-quality figures from v3 results."""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WORKSPACE)

RESULTS_DIR = 'exp/results_v3'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})
PALETTE = sns.color_palette('colorblind', 8)
METHOD_COLORS = {
    'retrain': PALETTE[7],
    'ft': PALETTE[0],
    'ga': PALETTE[1],
    'rl': PALETTE[2],
    'scrub': PALETTE[3],
    'neggrad': PALETTE[4],
}
METHOD_NAMES = {
    'retrain': 'Retrain',
    'ft': 'Fine-Tune',
    'ga': 'Grad. Ascent',
    'rl': 'Random Labels',
    'scrub': 'SCRUB',
    'neggrad': 'NegGrad+KD',
}
DATASET_NAMES = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'purchase100': 'Purchase-100',
}


def load_results():
    with open('results.json') as f:
        return json.load(f)


def figure1_stratified_mia(results):
    """Figure 1: Per-quintile MIA-AUC showing the difficulty-dependent privacy gap."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    methods_to_plot = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad']

    for ax_idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
        ax = axes[ax_idx]
        ds_results = results['main_results'].get(ds, {})

        x = np.arange(5) + 1  # Q1-Q5
        width = 0.15
        offsets = np.arange(len(methods_to_plot)) - len(methods_to_plot) / 2 + 0.5

        for i, method in enumerate(methods_to_plot):
            if method not in ds_results:
                continue
            mr = ds_results[method]
            if 'per_quintile' not in mr:
                continue
            pq = mr['per_quintile']
            aucs = []
            for q in range(1, 6):
                key = f'q{q}'
                if key in pq:
                    aucs.append(pq[key]['mean'])
                else:
                    aucs.append(0.5)

            ax.bar(x + offsets[i] * width, aucs, width,
                   color=METHOD_COLORS.get(method, 'gray'),
                   label=METHOD_NAMES.get(method, method),
                   edgecolor='white', linewidth=0.5)

        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Difficulty Quintile')
        ax.set_xticks(x)
        ax.set_xticklabels(['Q1\n(Easy)', 'Q2', 'Q3', 'Q4', 'Q5\n(Hard)'])
        ax.set_title(DATASET_NAMES.get(ds, ds))
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel('MIA-AUC')
    axes[0].legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Per-Quintile MIA-AUC Reveals Difficulty-Dependent Privacy Gaps', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure1_stratified_mia.pdf')
    fig.savefig(f'{FIGURES_DIR}/figure1_stratified_mia.png')
    plt.close()
    print('Saved figure1_stratified_mia')


def figure2_aggregate_vs_wq(results):
    """Figure 2: Scatter plot of Aggregate AUC vs WQ-AUC."""
    fig, ax = plt.subplots(figsize=(6, 5.5))

    methods = ['ft', 'ga', 'rl', 'scrub']
    markers = {'cifar10': 'o', 'cifar100': 's', 'purchase100': 'D'}

    for ds in ['cifar10', 'cifar100', 'purchase100']:
        ds_results = results.get('raw_per_seed', {}).get(ds, {})
        for seed_str, seed_data in ds_results.items():
            for method in methods:
                if method not in seed_data:
                    continue
                mr = seed_data[method]
                agg = mr.get('aggregate_auc', 0.5)
                wq = mr.get('wq_auc', 0.5)
                ax.scatter(agg, wq,
                          color=METHOD_COLORS.get(method, 'gray'),
                          marker=markers[ds], s=50, alpha=0.7,
                          edgecolors='white', linewidth=0.5)

    # Legend for methods
    for method in methods:
        ax.scatter([], [], color=METHOD_COLORS[method], marker='o', s=50,
                  label=METHOD_NAMES[method])
    # Legend for datasets
    for ds, marker in markers.items():
        ax.scatter([], [], color='gray', marker=marker, s=50,
                  label=DATASET_NAMES[ds])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='y=x')
    ax.set_xlabel('Aggregate MIA-AUC')
    ax.set_ylabel('Worst-Quintile MIA-AUC')
    ax.set_title('Aggregate Metrics Underestimate Privacy Risk')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.1, 1.05)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure2_aggregate_vs_wq.pdf')
    fig.savefig(f'{FIGURES_DIR}/figure2_aggregate_vs_wq.png')
    plt.close()
    print('Saved figure2_aggregate_vs_wq')


def figure3_dau_defense(results):
    """Figure 3: DAU defense effectiveness — WQ-AUC and DG for Standard vs DAU vs RUM."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    methods = ['ga', 'scrub', 'neggrad']
    variants = ['standard', 'DAU', 'RUM']
    variant_colors = [PALETTE[0], PALETTE[2], PALETTE[4]]

    for ax_idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
        ax = axes[ax_idx]
        ds_results = results['main_results'].get(ds, {})

        x = np.arange(len(methods))
        width = 0.25

        for vi, (variant, color) in enumerate(zip(variants, variant_colors)):
            wq_vals = []
            wq_errs = []
            for method in methods:
                if variant == 'standard':
                    key = method
                elif variant == 'DAU':
                    key = f'{method}_dau'
                else:
                    key = f'{method}_rum'

                if key in ds_results and 'wq_auc' in ds_results[key]:
                    wq_vals.append(ds_results[key]['wq_auc']['mean'])
                    wq_errs.append(ds_results[key]['wq_auc']['std'])
                else:
                    wq_vals.append(0)
                    wq_errs.append(0)

            ax.bar(x + (vi - 1) * width, wq_vals, width,
                   yerr=wq_errs, color=color, label=variant,
                   edgecolor='white', linewidth=0.5, capsize=3)

        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Unlearning Method')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_NAMES[m] for m in methods])
        ax.set_title(DATASET_NAMES.get(ds, ds))
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel('Worst-Quintile MIA-AUC (lower = better privacy)')
    axes[0].legend(loc='upper right', framealpha=0.9)

    fig.suptitle('DAU and RUM Improve Worst-Quintile Privacy', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure3_dau_defense.pdf')
    fig.savefig(f'{FIGURES_DIR}/figure3_dau_defense.png')
    plt.close()
    print('Saved figure3_dau_defense')


def figure4_alpha_sensitivity(results):
    """Figure 4: Alpha sensitivity showing WQ-AUC and RA tradeoff."""
    ablation = results.get('ablation_results', {}).get('alpha', {})
    if not ablation:
        print('No alpha ablation data, skipping figure4')
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]

    for ax_idx, ds in enumerate(['cifar10', 'cifar100']):
        ax = axes[ax_idx]
        ds_data = ablation.get(ds, {})

        for method, color, marker in [('ga', PALETTE[1], 'o'), ('scrub', PALETTE[3], 's')]:
            wq_means, wq_stds, ra_means = [], [], []
            for alpha in alphas:
                key = f'{method}_alpha{alpha}'
                vals_wq, vals_ra = [], []
                for seed_str, seed_data in ds_data.items():
                    if key in seed_data:
                        vals_wq.append(seed_data[key]['wq_auc'])
                        vals_ra.append(seed_data[key]['retain_acc'])
                if vals_wq:
                    wq_means.append(np.mean(vals_wq))
                    wq_stds.append(np.std(vals_wq))
                    ra_means.append(np.mean(vals_ra))
                else:
                    wq_means.append(None)
                    wq_stds.append(0)
                    ra_means.append(None)

            valid = [i for i, v in enumerate(wq_means) if v is not None]
            if valid:
                ax.errorbar([alphas[i] for i in valid],
                           [wq_means[i] for i in valid],
                           yerr=[wq_stds[i] for i in valid],
                           color=color, marker=marker, linewidth=2, capsize=3,
                           label=f'{METHOD_NAMES[method]} WQ-AUC')

                ax2 = ax.twinx()
                ax2.plot([alphas[i] for i in valid],
                        [ra_means[i] for i in valid],
                        color=color, marker=marker, linewidth=1.5,
                        linestyle='--', alpha=0.6)

        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(r'DAU Strength $\alpha$')
        ax.set_title(DATASET_NAMES.get(ds, ds))
        ax.legend(loc='upper right', fontsize=8)

    axes[0].set_ylabel('WQ-AUC (lower = better)')
    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure4_alpha_sensitivity.pdf')
    fig.savefig(f'{FIGURES_DIR}/figure4_alpha_sensitivity.png')
    plt.close()
    print('Saved figure4_alpha_sensitivity')


def figure5_ablations(results):
    """Figure 5: Ablation summary (K, strata, forget size, random weights)."""
    ablation = results.get('ablation_results', {})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) K vs metrics
    ax = axes[0, 0]
    k_data = ablation.get('K', {})
    if k_data:
        ks, rhos, wqs = [], [], []
        for key, val in sorted(k_data.items()):
            ks.append(val['K'])
            rhos.append(val.get('spearman_rho', 1.0))
            wqs.append(val['wq_auc'])
        ax.plot(ks, rhos, 'o-', color=PALETTE[0], linewidth=2, label='Spearman rho')
        ax2 = ax.twinx()
        ax2.plot(ks, wqs, 's--', color=PALETTE[2], linewidth=2, label='WQ-AUC')
        ax.set_xlabel('Number of Reference Models (K)')
        ax.set_ylabel('Spearman Correlation', color=PALETTE[0])
        ax2.set_ylabel('WQ-AUC (GA-DAU)', color=PALETTE[2])
        ax.set_title('(a) Reference Model Count K')
        ax.legend(loc='lower right', fontsize=8)
        ax2.legend(loc='upper left', fontsize=8)

    # (b) Strata granularity
    ax = axes[0, 1]
    strata_data = ablation.get('strata', {})
    if strata_data:
        names, dgs, worsts = [], [], []
        for name in ['terciles', 'quintiles', 'deciles']:
            if name in strata_data:
                names.append(name.capitalize())
                dgs.append(strata_data[name]['dg'])
                worsts.append(strata_data[name]['worst_stratum_auc'])
        x = np.arange(len(names))
        ax.bar(x - 0.15, worsts, 0.3, color=PALETTE[0], label='Worst Stratum AUC')
        ax.bar(x + 0.15, dgs, 0.3, color=PALETTE[2], label='Difficulty Gap')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('AUC')
        ax.set_title('(b) Stratification Granularity')
        ax.legend(fontsize=8)

    # (c) Forget size
    ax = axes[1, 0]
    fsize_data = ablation.get('forget_size', {})
    if fsize_data:
        sizes, std_wqs, dau_wqs = [], [], []
        for key in ['size_500', 'size_1000', 'size_2500']:
            if key in fsize_data:
                sizes.append(int(key.split('_')[1]))
                std_wqs.append(fsize_data[key]['standard']['wq_auc'])
                dau_wqs.append(fsize_data[key]['dau']['wq_auc'])
        if sizes:
            ax.plot(sizes, std_wqs, 'o-', color=PALETTE[0], linewidth=2, label='Standard GA')
            ax.plot(sizes, dau_wqs, 's--', color=PALETTE[2], linewidth=2, label='GA-DAU')
            ax.set_xlabel('Forget Set Size')
            ax.set_ylabel('WQ-AUC')
            ax.set_title('(c) Forget Set Size')
            ax.legend(fontsize=8)

    # (d) Random weight control
    ax = axes[1, 1]
    rw_data = ablation.get('random_weights', {})
    if rw_data:
        # Compare GA standard, GA-DAU, GA-random across seeds
        labels = ['Standard GA', 'GA-DAU', 'GA (Random\nWeights)']
        wq_means, wq_stds = [], []

        # Get standard GA and GA-DAU from main results
        main = results.get('main_results', {}).get('cifar10', {})
        if 'ga' in main and 'wq_auc' in main['ga']:
            wq_means.append(main['ga']['wq_auc']['mean'])
            wq_stds.append(main['ga']['wq_auc']['std'])
        else:
            wq_means.append(0); wq_stds.append(0)

        if 'ga_dau' in main and 'wq_auc' in main['ga_dau']:
            wq_means.append(main['ga_dau']['wq_auc']['mean'])
            wq_stds.append(main['ga_dau']['wq_auc']['std'])
        else:
            wq_means.append(0); wq_stds.append(0)

        rw_vals = [v['wq_auc'] for v in rw_data.values()]
        wq_means.append(np.mean(rw_vals))
        wq_stds.append(np.std(rw_vals))

        colors = [PALETTE[0], PALETTE[2], PALETTE[4]]
        bars = ax.bar(labels, wq_means, yerr=wq_stds, color=colors,
                      edgecolor='white', linewidth=0.5, capsize=3)
        ax.set_ylabel('WQ-AUC (CIFAR-10)')
        ax.set_title('(d) Random Weight Control')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure5_ablations.pdf')
    fig.savefig(f'{FIGURES_DIR}/figure5_ablations.png')
    plt.close()
    print('Saved figure5_ablations')


def figure6_difficulty_distribution(results):
    """Figure 6: Difficulty score distributions per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax_idx, ds in enumerate(['cifar10', 'cifar100', 'purchase100']):
        ax = axes[ax_idx]
        diff = np.load(f'exp/results/difficulty_{ds}_train.npy')
        quint = np.load(f'exp/results/quintiles_{ds}_train.npy')

        for q in range(5):
            mask = quint == q
            ax.hist(diff[mask], bins=50, alpha=0.6,
                    label=f'Q{q+1}', density=True,
                    color=plt.cm.RdYlGn_r(q / 4))

        ax.set_xlabel('Difficulty Score (avg ref model loss)')
        ax.set_title(DATASET_NAMES.get(ds, ds))
        if ax_idx == 0:
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{FIGURES_DIR}/figure6_difficulty_distribution.pdf')
    fig.savefig(f'{FIGURES_DIR}/figure6_difficulty_distribution.png')
    plt.close()
    print('Saved figure6_difficulty_distribution')


def table1_main_results(results):
    """Generate LaTeX table of main results."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Main Results: Aggregate vs.\ Stratified MIA Evaluation and Defense Effectiveness}')
    lines.append(r'\label{tab:main_results}')
    lines.append(r'\small')

    for ds in ['cifar10', 'cifar100', 'purchase100']:
        ds_r = results['main_results'].get(ds, {})
        lines.append(r'\begin{tabular}{l|ccc|cc|c}')
        lines.append(r'\toprule')
        lines.append(f'\\multicolumn{{7}}{{c}}{{\\textbf{{{DATASET_NAMES[ds]}}}}} \\\\')
        lines.append(r'\midrule')
        lines.append(r'Method & FA$\downarrow$ & RA$\uparrow$ & TA$\uparrow$ & Agg AUC & WQ-AUC & DG \\')
        lines.append(r'\midrule')

        method_order = ['retrain', 'ft', 'ga', 'rl', 'scrub', 'neggrad',
                       'ga_dau', 'scrub_dau', 'neggrad_dau', 'ga_rum', 'scrub_rum']
        method_labels = {
            'retrain': 'Retrain (oracle)',
            'ft': 'Fine-Tune',
            'ga': 'Gradient Ascent',
            'rl': 'Random Labels',
            'scrub': 'SCRUB',
            'neggrad': 'NegGrad+KD',
            'ga_dau': 'GA + DAU (ours)',
            'scrub_dau': 'SCRUB + DAU (ours)',
            'neggrad_dau': 'NegGrad + DAU (ours)',
            'ga_rum': 'GA + RUM',
            'scrub_rum': 'SCRUB + RUM',
        }

        for method in method_order:
            if method not in ds_r:
                continue
            r = ds_r[method]
            label = method_labels.get(method, method)

            def fmt(key, digits=3):
                if key in r:
                    return f"${r[key]['mean']:.{digits}f}" + r"\pm" + f"{r[key]['std']:.{digits}f}$"
                return '--'

            fa = fmt('forget_acc')
            ra = fmt('retain_acc')
            ta = fmt('test_acc')
            agg = fmt('aggregate_auc')
            wq = fmt('wq_auc')
            dg = fmt('dg')

            if method in ('ga_dau', 'scrub_dau', 'neggrad_dau'):
                label = r'\textbf{' + label + '}'

            line = f'{label} & {fa} & {ra} & {ta} & {agg} & {wq} & {dg} \\\\'
            if method in ('neggrad', 'scrub_rum'):
                line += r' \midrule'
            lines.append(line)

        lines.append(r'\bottomrule')
        lines.append(r'\end{tabular}')
        lines.append(r'\vspace{0.3em}')
        lines.append('')

    lines.append(r'\end{table*}')

    with open(f'{FIGURES_DIR}/table1_main_results.tex', 'w') as f:
        f.write('\n'.join(lines))
    print('Saved table1_main_results.tex')


def table2_ablations(results):
    """Generate LaTeX table summarizing ablation results."""
    ablation = results.get('ablation_results', {})
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Ablation Study Summary (CIFAR-10)}')
    lines.append(r'\label{tab:ablations}')
    lines.append(r'\small')

    # K ablation
    lines.append(r'\begin{subtable}{\linewidth}')
    lines.append(r'\centering')
    lines.append(r'\caption{Reference Model Count $K$}')
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'\toprule')
    lines.append(r'$K$ & Spearman $\rho$ & Quintile Stability & GA-DAU WQ-AUC \\')
    lines.append(r'\midrule')
    k_data = ablation.get('K', {})
    for key in sorted(k_data.keys()):
        v = k_data[key]
        rho = v.get('spearman_rho', '--')
        stab = v.get('quintile_stability', '--')
        wq = v.get('wq_auc', '--')
        rho_s = f'{rho:.3f}' if isinstance(rho, float) else str(rho)
        stab_s = f'{stab:.3f}' if isinstance(stab, float) else str(stab)
        wq_s = f'{wq:.3f}' if isinstance(wq, float) else str(wq)
        lines.append(f'{v.get("K", key)} & {rho_s} & {stab_s} & {wq_s} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{subtable}')
    lines.append(r'\vspace{0.5em}')

    # Alpha ablation summary
    lines.append(r'\begin{subtable}{\linewidth}')
    lines.append(r'\centering')
    lines.append(r'\caption{DAU Strength $\alpha$ (GA method, CIFAR-10)}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'$\alpha$ & WQ-AUC & DG & RA & TA \\')
    lines.append(r'\midrule')
    alpha_data = ablation.get('alpha', {}).get('cifar10', {})
    if alpha_data:
        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
            key = f'ga_alpha{alpha}'
            vals = [sd[key] for sd in alpha_data.values() if key in sd]
            if vals:
                wq = np.mean([v['wq_auc'] for v in vals])
                dg = np.mean([v['dg'] for v in vals])
                ra = np.mean([v['retain_acc'] for v in vals])
                ta = np.mean([v['test_acc'] for v in vals])
                lines.append(f'{alpha} & {wq:.3f} & {dg:.3f} & {ra:.3f} & {ta:.3f} \\\\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{subtable}')

    lines.append(r'\end{table}')

    with open(f'{FIGURES_DIR}/table2_ablations.tex', 'w') as f:
        f.write('\n'.join(lines))
    print('Saved table2_ablations.tex')


def main():
    results = load_results()
    print(f"Loaded results with {len(results.get('main_results', {}))} datasets")

    figure1_stratified_mia(results)
    figure2_aggregate_vs_wq(results)
    figure3_dau_defense(results)
    figure4_alpha_sensitivity(results)
    figure5_ablations(results)
    figure6_difficulty_distribution(results)
    table1_main_results(results)
    table2_ablations(results)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
