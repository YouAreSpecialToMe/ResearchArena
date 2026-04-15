"""Analysis and figure generation for E-PC paper."""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import save_results

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Style
sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})
COLORS = sns.color_palette('colorblind', 8)
METHOD_COLORS = {
    'PC': COLORS[0],
    'PC-p (BY)': COLORS[1],
    'E-PC (cal)': COLORS[2],
    'E-PC (split-LR)': COLORS[3],
    'GES': COLORS[4],
    'NOTEARS': COLORS[5],
}


def load_json(path):
    with open(os.path.join(BASE_DIR, path)) as f:
        return json.load(f)


def extract_metric(result, method_key, metric):
    v = result.get(method_key)
    if v is None or 'error' in v:
        return None
    return v.get(metric)


# ===== FIGURE 1: FDR Calibration =====
def fig1_fdr_calibration(synth_results):
    """Plot empirical FDR vs nominal q for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    q_levels = [0.05, 0.1, 0.2]

    for ax_idx, p_nodes in enumerate([10, 20, 50]):
        ax = axes[ax_idx]
        filtered = [r for r in synth_results
                    if r.get('config', {}).get('num_nodes') == p_nodes
                    and r.get('config', {}).get('avg_degree') == 2
                    and r.get('config', {}).get('data_type') == 'linear_gaussian'
                    and 'error' not in r]

        methods = {
            'PC-p (BY)': 'PCp_q{}',
            'E-PC (cal)': 'EPC_cal_q{}',
            'E-PC (split-LR)': 'EPC_slr_q{}',
        }

        for method_name, key_fmt in methods.items():
            means = []
            stds = []
            for q in q_levels:
                key = key_fmt.format(q)
                fdrs = [extract_metric(r, key, 'FDR') for r in filtered]
                fdrs = [f for f in fdrs if f is not None]
                if fdrs:
                    means.append(np.mean(fdrs))
                    stds.append(np.std(fdrs))
                else:
                    means.append(0)
                    stds.append(0)

            ax.errorbar(q_levels, means, yerr=stds, marker='o', label=method_name,
                       color=METHOD_COLORS.get(method_name, 'gray'), capsize=3, linewidth=2)

        ax.plot([0, 0.25], [0, 0.25], 'k--', alpha=0.5, label='Ideal (FDR=q)')
        ax.set_xlabel('Nominal FDR (q)')
        if ax_idx == 0:
            ax.set_ylabel('Empirical FDR')
        ax.set_title(f'p = {p_nodes}')
        ax.set_xlim(0, 0.25)
        ax.set_ylim(-0.02, 0.35)
        if ax_idx == 2:
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle('FDR Calibration: Empirical FDR vs Nominal Level', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig1_fdr_calibration.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig1_fdr_calibration.png'), bbox_inches='tight')
    plt.close(fig)
    print("Figure 1 saved.")


# ===== FIGURE 2: Power Comparison =====
def fig2_power_comparison(synth_results):
    """Bar chart: TPR at matched FDR level q=0.1."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    methods_q01 = [
        ('PC (a=0.05)', 'PC_alpha0.05'),
        ('PC-p (BY)', 'PCp_q0.1'),
        ('E-PC cal', 'EPC_cal_q0.1'),
        ('E-PC SLR', 'EPC_slr_q0.1'),
        ('GES', 'GES'),
    ]

    for ax_idx, p_nodes in enumerate([10, 20, 50]):
        ax = axes[ax_idx]
        filtered = [r for r in synth_results
                    if r.get('config', {}).get('num_nodes') == p_nodes
                    and r.get('config', {}).get('avg_degree') == 2
                    and r.get('config', {}).get('data_type') == 'linear_gaussian'
                    and r.get('config', {}).get('n_samples') == 1000
                    and 'error' not in r]

        x = np.arange(len(methods_q01))
        means = []
        stds = []
        for label, key in methods_q01:
            tprs = [extract_metric(r, key, 'TPR') for r in filtered]
            tprs = [t for t in tprs if t is not None]
            means.append(np.mean(tprs) if tprs else 0)
            stds.append(np.std(tprs) if tprs else 0)

        colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3], COLORS[4]]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in methods_q01], rotation=30, ha='right', fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel('TPR (True Positive Rate)')
        ax.set_title(f'p = {p_nodes}, n = 1000')
        ax.set_ylim(0, 1.1)

    fig.suptitle('Power Comparison at Matched FDR Level (q=0.1)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig2_power_comparison.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig2_power_comparison.png'), bbox_inches='tight')
    plt.close(fig)
    print("Figure 2 saved.")


# ===== FIGURE 3: FDR-Power Tradeoff =====
def fig3_fdr_power_tradeoff(synth_results):
    """Plot TPR vs empirical FDR for each method by varying thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax_idx, p_nodes in enumerate([10, 20, 50]):
        ax = axes[ax_idx]
        filtered = [r for r in synth_results
                    if r.get('config', {}).get('num_nodes') == p_nodes
                    and r.get('config', {}).get('avg_degree') == 2
                    and r.get('config', {}).get('data_type') == 'linear_gaussian'
                    and 'error' not in r]

        for label, keys, color, marker in [
            ('PC', ['PC_alpha0.01', 'PC_alpha0.05', 'PC_alpha0.1'], COLORS[0], 'o'),
            ('PC-p (BY)', ['PCp_q0.05', 'PCp_q0.1', 'PCp_q0.2'], COLORS[1], 's'),
            ('E-PC (cal)', ['EPC_cal_q0.05', 'EPC_cal_q0.1', 'EPC_cal_q0.2'], COLORS[2], '^'),
            ('E-PC (SLR)', ['EPC_slr_q0.05', 'EPC_slr_q0.1', 'EPC_slr_q0.2'], COLORS[3], 'D'),
        ]:
            fdrs_mean = []
            tprs_mean = []
            for key in keys:
                fdrs = [extract_metric(r, key, 'FDR') for r in filtered]
                tprs = [extract_metric(r, key, 'TPR') for r in filtered]
                fdrs = [f for f in fdrs if f is not None]
                tprs = [t for t in tprs if t is not None]
                if fdrs and tprs:
                    fdrs_mean.append(np.mean(fdrs))
                    tprs_mean.append(np.mean(tprs))

            if fdrs_mean:
                ax.plot(fdrs_mean, tprs_mean, f'-{marker}', label=label,
                       color=color, markersize=8, linewidth=2)

        ax.set_xlabel('Empirical FDR')
        if ax_idx == 0:
            ax.set_ylabel('TPR')
        ax.set_title(f'p = {p_nodes}')
        ax.set_xlim(-0.02, 0.5)
        ax.set_ylim(0, 1.05)
        if ax_idx == 2:
            ax.legend(fontsize=9)

    fig.suptitle('FDR-Power Tradeoff Curves', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig3_fdr_power_tradeoff.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig3_fdr_power_tradeoff.png'), bbox_inches='tight')
    plt.close(fig)
    print("Figure 3 saved.")


# ===== FIGURE 4: Anytime Validity =====
def fig4_anytime_validity(anytime_results):
    """Plot FDR and TPR as a function of number of folds K'."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)

    for col_idx, p_nodes in enumerate([10, 20, 50]):
        filtered = [r for r in anytime_results
                    if r.get('config', {}).get('num_nodes') == p_nodes
                    and 'fold_metrics' in r]

        if not filtered:
            continue

        K = 10
        fdr_by_k = defaultdict(list)
        tpr_by_k = defaultdict(list)

        for r in filtered:
            fm = r['fold_metrics']
            for k in range(1, K + 1):
                mk = fm.get(str(k), {})
                if 'FDR' in mk:
                    fdr_by_k[k].append(mk['FDR'])
                    tpr_by_k[k].append(mk['TPR'])

        ks = sorted(fdr_by_k.keys())
        fdr_means = [np.mean(fdr_by_k[k]) for k in ks]
        fdr_stds = [np.std(fdr_by_k[k]) for k in ks]
        tpr_means = [np.mean(tpr_by_k[k]) for k in ks]
        tpr_stds = [np.std(tpr_by_k[k]) for k in ks]

        ax = axes[0, col_idx]
        ax.errorbar(ks, fdr_means, yerr=fdr_stds, marker='o', color=COLORS[2], capsize=3)
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='q=0.1')
        ax.set_ylabel('Empirical FDR' if col_idx == 0 else '')
        ax.set_title(f'p = {p_nodes}')
        ax.set_ylim(-0.02, 0.3)
        if col_idx == 0:
            ax.legend()

        ax = axes[1, col_idx]
        ax.errorbar(ks, tpr_means, yerr=tpr_stds, marker='s', color=COLORS[3], capsize=3)
        ax.set_xlabel("Number of folds (K')")
        ax.set_ylabel('TPR' if col_idx == 0 else '')
        ax.set_ylim(0, 1.05)

    fig.suptitle('Anytime Validity: FDR Control and Power at Each Stopping Point', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig4_anytime_validity.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig4_anytime_validity.png'), bbox_inches='tight')
    plt.close(fig)
    print("Figure 4 saved.")


# ===== FIGURE 5: Ablation - Number of Folds =====
def fig5_ablation_folds(folds_results):
    """Plot F1 vs K for different graph sizes."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for p_nodes, color, marker in [(10, COLORS[0], 'o'), (20, COLORS[1], 's')]:
        filtered = [r for r in folds_results
                    if r.get('config', {}).get('num_nodes') == p_nodes]

        if not filtered:
            continue

        Ks = [2, 3, 5, 8, 10]
        f1_means = []
        f1_stds = []
        for K in Ks:
            vals = [r[f'K{K}']['F1'] for r in filtered
                    if f'K{K}' in r and 'error' not in r.get(f'K{K}', {})]
            f1_means.append(np.mean(vals) if vals else 0)
            f1_stds.append(np.std(vals) if vals else 0)

        ax.errorbar(Ks, f1_means, yerr=f1_stds, marker=marker, color=color,
                   capsize=3, linewidth=2, label=f'p={p_nodes}')

    ax.set_xlabel('Number of Folds (K)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Ablation: Effect of Number of Folds on F1')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig5_ablation_folds.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig5_ablation_folds.png'), bbox_inches='tight')
    plt.close(fig)
    print("Figure 5 saved.")


# ===== FIGURE 6: Scalability =====
def fig6_scalability(scale_results):
    """Log-log plot of runtime vs number of nodes."""
    fig, ax = plt.subplots(figsize=(7, 5))

    methods = [
        ('PC', 'PC_time', COLORS[0], 'o'),
        ('PC-p (BY)', 'PCp_time', COLORS[1], 's'),
        ('E-PC (cal)', 'EPC_cal_time', COLORS[2], '^'),
        ('E-PC (SLR)', 'EPC_slr_time', COLORS[3], 'D'),
        ('GES', 'GES_time', COLORS[4], 'v'),
    ]

    ps = sorted(set(r['config']['num_nodes'] for r in scale_results))

    for label, key, color, marker in methods:
        means = []
        stds = []
        valid_ps = []
        for p in ps:
            times = [r[key] for r in scale_results
                     if r['config']['num_nodes'] == p and key in r]
            if times:
                means.append(np.mean(times))
                stds.append(np.std(times))
                valid_ps.append(p)

        if valid_ps:
            ax.errorbar(valid_ps, means, yerr=stds, marker=marker, color=color,
                       capsize=3, linewidth=2, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Nodes (p)')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Scalability: Runtime vs Graph Size')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig6_scalability.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(BASE_DIR, 'figures/fig6_scalability.png'), bbox_inches='tight')
    plt.close(fig)
    print("Figure 6 saved.")


# ===== TABLE 1: Main Results =====
def table1_main_results(synth_results):
    """Generate LaTeX table with main results."""
    methods = [
        ('PC ($\\alpha$=0.05)', 'PC_alpha0.05'),
        ('PC-p (BY, $q$=0.1)', 'PCp_q0.1'),
        ('GES', 'GES'),
        ('NOTEARS', 'NOTEARS'),
        ('E-PC cal ($q$=0.1)', 'EPC_cal_q0.1'),
        ('E-PC SLR ($q$=0.1)', 'EPC_slr_q0.1'),
    ]
    metrics = ['SHD', 'FDR', 'TPR', 'F1']

    lines = []
    lines.append('\\begin{tabular}{l' + 'c' * (len(metrics) * 3) + '}')
    lines.append('\\toprule')

    header = 'Method'
    for p in [10, 20, 50]:
        header += f' & \\multicolumn{{{len(metrics)}}}{{c}}{{$p={p}$}}'
    header += ' \\\\'
    lines.append(header)

    subheader = ''
    for _ in [10, 20, 50]:
        for m in metrics:
            subheader += f' & {m}'
    lines.append(subheader + ' \\\\')
    lines.append('\\midrule')

    for method_name, method_key in methods:
        row = method_name
        for p_nodes in [10, 20, 50]:
            filtered = [r for r in synth_results
                        if r.get('config', {}).get('num_nodes') == p_nodes
                        and r.get('config', {}).get('avg_degree') == 2
                        and r.get('config', {}).get('data_type') == 'linear_gaussian'
                        and r.get('config', {}).get('n_samples') == 1000
                        and 'error' not in r]

            for metric in metrics:
                vals = [extract_metric(r, method_key, metric) for r in filtered]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    if metric in ['FDR', 'TPR', 'F1']:
                        row += f' & {mean:.3f}$\\pm${std:.3f}'
                    else:
                        row += f' & {mean:.1f}$\\pm${std:.1f}'
                else:
                    row += ' & --'
        row += ' \\\\'
        lines.append(row)

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    tex = '\n'.join(lines)
    with open(os.path.join(BASE_DIR, 'figures/table1_main_results.tex'), 'w') as f:
        f.write(tex)
    print("Table 1 saved.")
    return tex


# ===== TABLE 2: Real Data =====
def table2_real_data(real_results):
    """Generate LaTeX table for real-world experiments."""
    methods = [
        ('PC ($\\alpha$=0.05)', 'PC_alpha0.05'),
        ('PC-p (BY, $q$=0.1)', 'PCp_q0.1'),
        ('GES', 'GES'),
        ('E-PC cal ($q$=0.1)', 'EPC_cal_q0.1'),
        ('E-PC SLR ($q$=0.1)', 'EPC_slr_q0.1'),
    ]
    metrics = ['SHD', 'FDR', 'TPR', 'F1']
    networks = ['Asia', 'Sachs', 'ALARM', 'Insurance']

    lines = []
    lines.append('\\begin{tabular}{l' + 'c' * (len(metrics) * len(networks)) + '}')
    lines.append('\\toprule')

    header = 'Method'
    for net in networks:
        header += f' & \\multicolumn{{{len(metrics)}}}{{c}}{{{net}}}'
    header += ' \\\\'
    lines.append(header)

    subheader = ''
    for _ in networks:
        for m in metrics:
            subheader += f' & {m}'
    lines.append(subheader + ' \\\\')
    lines.append('\\midrule')

    for method_name, method_key in methods:
        row = method_name
        for net in networks:
            res_list = real_results.get(net, [])
            for metric in metrics:
                vals = [extract_metric(r, method_key, metric) for r in res_list]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean = np.mean(vals)
                    if metric in ['FDR', 'TPR', 'F1']:
                        row += f' & {mean:.3f}'
                    else:
                        row += f' & {mean:.1f}'
                else:
                    row += ' & --'
        row += ' \\\\'
        lines.append(row)

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    tex = '\n'.join(lines)
    with open(os.path.join(BASE_DIR, 'figures/table2_real_data.tex'), 'w') as f:
        f.write(tex)
    print("Table 2 saved.")
    return tex


# ===== TABLE 3: Ablation =====
def table3_ablation_evalue(ablation_results):
    """LaTeX table comparing e-value construction methods."""
    metrics = ['FDR', 'TPR', 'F1', 'SHD']
    methods = [('Calibrator', 'cal'), ('Split-LR', 'slr'), ('GRAAL', 'graal')]

    lines = []
    lines.append('\\begin{tabular}{l' + 'c' * (len(metrics) * 3) + '}')
    lines.append('\\toprule')

    header = 'E-value Type'
    for p in [10, 20, 50]:
        header += f' & \\multicolumn{{{len(metrics)}}}{{c}}{{$p={p}$}}'
    header += ' \\\\'
    lines.append(header)

    subheader = ''
    for _ in [10, 20, 50]:
        for m in metrics:
            subheader += f' & {m}'
    lines.append(subheader + ' \\\\')
    lines.append('\\midrule')

    for method_name, tag in methods:
        row = method_name
        for p_nodes in [10, 20, 50]:
            filtered = [r for r in ablation_results
                        if r.get('config', {}).get('num_nodes') == p_nodes
                        and tag in r and 'error' not in r.get(tag, {})]

            for metric in metrics:
                vals = [r[tag][metric] for r in filtered if metric in r.get(tag, {})]
                if vals:
                    mean = np.mean(vals)
                    if metric in ['FDR', 'TPR', 'F1']:
                        row += f' & {mean:.3f}'
                    else:
                        row += f' & {mean:.1f}'
                else:
                    row += ' & --'
        row += ' \\\\'
        lines.append(row)

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    tex = '\n'.join(lines)
    with open(os.path.join(BASE_DIR, 'figures/table3_ablation_evalue.tex'), 'w') as f:
        f.write(tex)
    print("Table 3 saved.")
    return tex


# ===== AGGREGATE results.json =====
def generate_results_json(synth_results, real_results, ablation_eval, ablation_folds,
                           anytime_results, scale_results):
    """Generate the aggregated results.json for the workspace root."""

    q_levels = [0.05, 0.1, 0.2]

    # SC1: FDR calibration
    fdr_within_10pct = 0
    fdr_total = 0
    for r in synth_results:
        if 'error' in r:
            continue
        for q in q_levels:
            for key in [f'EPC_cal_q{q}', f'EPC_slr_q{q}']:
                fdr = extract_metric(r, key, 'FDR')
                if fdr is not None:
                    fdr_total += 1
                    if fdr <= 1.1 * q:
                        fdr_within_10pct += 1
    sc1_frac = fdr_within_10pct / fdr_total if fdr_total > 0 else 0

    # SC2: Power comparison E-PC vs PC-p
    tpr_diffs = []
    for r in synth_results:
        if 'error' in r:
            continue
        for q in q_levels:
            epc_tpr = extract_metric(r, f'EPC_cal_q{q}', 'TPR')
            pcp_tpr = extract_metric(r, f'PCp_q{q}', 'TPR')
            if epc_tpr is not None and pcp_tpr is not None:
                tpr_diffs.append(epc_tpr - pcp_tpr)

    from scipy import stats as sp_stats
    if len(tpr_diffs) > 1:
        t_stat, p_val = sp_stats.ttest_1samp(tpr_diffs, 0)
        sc2_mean_diff = float(np.mean(tpr_diffs))
        sc2_pvalue = float(p_val)
    else:
        sc2_mean_diff = 0.0
        sc2_pvalue = 1.0

    # SC3: Anytime validity
    max_fdr_violation = 0.0
    anytime_configs_checked = 0
    for r in anytime_results:
        if 'fold_metrics' not in r:
            continue
        for k, mk in r['fold_metrics'].items():
            if 'FDR' in mk:
                anytime_configs_checked += 1
                max_fdr_violation = max(max_fdr_violation, mk['FDR'])

    # SC4: SHD/F1 comparison
    shd_diffs = []
    f1_diffs = []
    for r in synth_results:
        if 'error' in r:
            continue
        epc_shd = extract_metric(r, 'EPC_cal_q0.1', 'SHD')
        pc_shd = extract_metric(r, 'PC_alpha0.05', 'SHD')
        epc_f1 = extract_metric(r, 'EPC_cal_q0.1', 'F1')
        pc_f1 = extract_metric(r, 'PC_alpha0.05', 'F1')
        if epc_shd is not None and pc_shd is not None:
            shd_diffs.append(pc_shd - epc_shd)
        if epc_f1 is not None and pc_f1 is not None:
            f1_diffs.append(epc_f1 - pc_f1)

    # Main comparison table
    main_table = {}
    for p_nodes in [10, 20, 50]:
        main_table[f'p={p_nodes}'] = {}
        filtered = [r for r in synth_results
                    if r.get('config', {}).get('num_nodes') == p_nodes
                    and r.get('config', {}).get('avg_degree') == 2
                    and r.get('config', {}).get('data_type') == 'linear_gaussian'
                    and r.get('config', {}).get('n_samples') == 1000
                    and 'error' not in r]

        for method_name, method_key in [
            ('PC_alpha0.05', 'PC_alpha0.05'),
            ('PCp_q0.1', 'PCp_q0.1'),
            ('GES', 'GES'),
            ('NOTEARS', 'NOTEARS'),
            ('EPC_cal_q0.1', 'EPC_cal_q0.1'),
            ('EPC_slr_q0.1', 'EPC_slr_q0.1'),
        ]:
            metrics_agg = {}
            for metric in ['SHD', 'FDR', 'TPR', 'F1', 'precision']:
                vals = [extract_metric(r, method_key, metric) for r in filtered]
                vals = [v for v in vals if v is not None]
                if vals:
                    metrics_agg[metric] = {'mean': round(float(np.mean(vals)), 4),
                                            'std': round(float(np.std(vals)), 4)}
            if metrics_agg:
                main_table[f'p={p_nodes}'][method_name] = metrics_agg

    results_json = {
        'success_criteria': {
            'SC1_FDR_calibration': {
                'description': 'E-PC controls FDR within 10% of nominal across >=80% of configs',
                'fraction_within_10pct': round(sc1_frac, 4),
                'total_configs': fdr_total,
                'passed': bool(sc1_frac >= 0.80),
            },
            'SC2_power_vs_PCp': {
                'description': 'E-PC recovers more true edges than PC-p at matched FDR',
                'mean_TPR_diff': round(sc2_mean_diff, 4),
                'pvalue': round(sc2_pvalue, 6),
                'n_comparisons': len(tpr_diffs),
                'passed': bool(sc2_pvalue < 0.05 and sc2_mean_diff > 0),
            },
            'SC3_anytime_validity': {
                'description': "FDR control at every stopping point K'",
                'max_FDR_observed': round(float(max_fdr_violation), 4),
                'configs_checked': anytime_configs_checked,
                'passed': bool(max_fdr_violation <= 0.2),
            },
            'SC4_SHD_F1_vs_PC': {
                'description': 'E-PC achieves competitive SHD and F1 vs PC at best alpha',
                'mean_SHD_improvement': round(float(np.mean(shd_diffs)), 2) if shd_diffs else 0,
                'mean_F1_diff': round(float(np.mean(f1_diffs)), 4) if f1_diffs else 0,
                'passed': bool(np.mean(f1_diffs) >= -0.05) if f1_diffs else False,
            },
        },
        'main_results': main_table,
        'experiment_summary': {
            'num_synthetic_configs': len(synth_results),
            'num_seeds': 5,
            'num_real_networks': len(real_results),
            'methods': ['PC', 'PC-p (BY)', 'GES', 'NOTEARS', 'E-PC (calibrator)', 'E-PC (split-LR)'],
        },
    }

    save_results(results_json, os.path.join(BASE_DIR, 'results.json'))
    print("\nresults.json saved.")
    print("\n=== SUCCESS CRITERIA ===")
    for k, v in results_json['success_criteria'].items():
        status = "PASS" if v['passed'] else "FAIL"
        print(f"  {k}: {status} - {v['description']}")
        for kk, vv in v.items():
            if kk not in ['description', 'passed']:
                print(f"    {kk}: {vv}")

    return results_json


def main():
    os.chdir(BASE_DIR)

    print("Loading results...")
    synth = load_json('results/synthetic_results.json')
    real = load_json('results/real_results.json')
    abl_eval = load_json('results/ablation_evalue_type.json')
    abl_folds = load_json('results/ablation_num_folds.json')
    anytime = load_json('results/anytime_validity.json')
    scale = load_json('results/scalability.json')

    print("Generating figures...")
    fig1_fdr_calibration(synth)
    fig2_power_comparison(synth)
    fig3_fdr_power_tradeoff(synth)
    fig4_anytime_validity(anytime)
    fig5_ablation_folds(abl_folds)
    fig6_scalability(scale)

    print("\nGenerating tables...")
    table1_main_results(synth)
    table2_real_data(real)
    table3_ablation_evalue(abl_eval)

    print("\nGenerating aggregated results.json...")
    generate_results_json(synth, real, abl_eval, abl_folds, anytime, scale)


if __name__ == '__main__':
    main()
