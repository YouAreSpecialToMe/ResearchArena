#!/usr/bin/env python3
"""
Generate publication-quality figures and tables for EGTM paper.
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp.shared.data_loader import ALL_CORRUPTIONS, CORRUPTIONS

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
# Colorblind-friendly palette
COLORS = sns.color_palette("colorblind", 8)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name):
    os.makedirs('figures', exist_ok=True)
    fig.savefig(f'figures/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'figures/{name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved figures/{name}.pdf and .png')


# ============================================================
# Figure 1: Entropy Distribution (Clean vs Corrupted)
# ============================================================
def figure1_entropy_distribution():
    """Violin plot of CLS-row attention entropy for clean vs corrupted."""
    try:
        nored = load_json('results/baseline_no_reduction_deit_s.json')
    except FileNotFoundError:
        print("  Skipping figure1: no results found")
        return

    clean_entropy = nored.get('clean_entropy', {})
    # Get gaussian_noise corruption entropy
    corr_entropy = {}
    try:
        corr_data = nored['corrupted']['severity_5']['gaussian_noise']
        corr_entropy = corr_data.get('entropy', {})
    except (KeyError, TypeError):
        pass

    if not clean_entropy or not corr_entropy:
        print("  Skipping figure1: no entropy data")
        return

    layers = list(range(12))
    clean_means = [clean_entropy.get(str(l), {}).get('mean', 0) for l in layers]
    corr_means = [corr_entropy.get(str(l), {}).get('mean', 0) for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(layers))
    width = 0.35
    ax.bar(x - width/2, clean_means, width, label='Clean', color=COLORS[0], alpha=0.8)
    ax.bar(x + width/2, corr_means, width, label='Gaussian Noise (sev=5)', color=COLORS[1], alpha=0.8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean CLS-Row Attention Entropy')
    ax.set_title('CLS-Row Attention Entropy: Clean vs Corrupted (DeiT-S)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'figure1_entropy_distribution')


# ============================================================
# Figure 2: Adaptive Merging Ratio
# ============================================================
def figure2_adaptive_ratio():
    """Show how EGTM adapts merging ratio on clean vs corrupted inputs."""
    try:
        egtm = load_json('results/egtm_deit_s_seed42.json')
    except FileNotFoundError:
        print("  Skipping figure2: no EGTM results")
        return

    avg_r_clean = egtm.get('avg_merge_ratio_clean', 8)

    # Get per-corruption merge ratios
    corr_ratios = {}
    try:
        for corr in ALL_CORRUPTIONS:
            data = egtm['corrupted']['severity_5'][corr]
            corr_ratios[corr] = data.get('avg_merge_ratio', 8)
    except (KeyError, TypeError):
        pass

    if not corr_ratios:
        print("  Skipping figure2: no merge ratio data")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    corrs = list(corr_ratios.keys())
    ratios = [corr_ratios[c] for c in corrs]

    # Color by corruption category
    cat_colors = {'noise': COLORS[1], 'blur': COLORS[2], 'weather': COLORS[3], 'digital': COLORS[4]}
    bar_colors = []
    for c in corrs:
        for cat, members in CORRUPTIONS.items():
            if c in members:
                bar_colors.append(cat_colors[cat])
                break

    ax.bar(range(len(corrs)), ratios, color=bar_colors, alpha=0.8)
    ax.axhline(y=avg_r_clean, color='black', linestyle='--', linewidth=1.5, label=f'Clean avg r={avg_r_clean:.1f}')
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Average Merging Ratio')
    ax.set_title('EGTM Adaptive Merging Ratio per Corruption (DeiT-S, severity=5)')
    ax.set_xticks(range(len(corrs)))
    ax.set_xticklabels([c.replace('_', '\n') for c in corrs], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'figure2_adaptive_ratio')


# ============================================================
# Figure 3: Per-Corruption Improvement
# ============================================================
def figure3_per_corruption():
    """Bar chart of EGTM improvement over ToMe per corruption type."""
    try:
        analysis = load_json('results/per_corruption_analysis.json')
    except FileNotFoundError:
        print("  Skipping figure3: no analysis results")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    corrs = [c for c in ALL_CORRUPTIONS if c in analysis and isinstance(analysis[c], dict)]
    improvements = [analysis[c].get('egtm_improvement_over_tome', 0) * 100 for c in corrs]
    entropy_dirs = [analysis[c].get('entropy_direction', 'neutral') for c in corrs]

    cat_colors = {'noise': COLORS[1], 'blur': COLORS[2], 'weather': COLORS[3], 'digital': COLORS[4]}
    bar_colors = []
    for c in corrs:
        for cat, members in CORRUPTIONS.items():
            if c in members:
                bar_colors.append(cat_colors[cat])
                break

    bars = ax.bar(range(len(corrs)), improvements, color=bar_colors, alpha=0.8)

    # Mark entropy-decrease corruptions
    for i, (c, ed) in enumerate(zip(corrs, entropy_dirs)):
        if ed == 'decrease':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('EGTM Improvement over ToMe (% points)')
    ax.set_title('Per-Corruption Accuracy Improvement of EGTM over ToMe (DeiT-S, sev=5)')
    ax.set_xticks(range(len(corrs)))
    ax.set_xticklabels([c.replace('_', '\n') for c in corrs], rotation=45, ha='right', fontsize=8)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cat_colors[cat], label=cat.title()) for cat in cat_colors]
    legend_elements.append(Patch(facecolor='white', edgecolor='red', linewidth=2, label='Entropy decrease'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    save_fig(fig, 'figure3_per_corruption')


# ============================================================
# Figure 4: Accuracy-Throughput Pareto
# ============================================================
def figure4_pareto():
    """Scatter plot: accuracy vs throughput for all methods."""
    methods_data = {}
    for name, path in [
        ('No Reduction', 'results/baseline_no_reduction_deit_s.json'),
        ('ToMe r=8', 'results/baseline_tome_deit_s_r8.json'),
        ('Random Drop', 'results/baseline_random_drop_deit_s.json'),
        ('EViT-style', 'results/baseline_evit_style_deit_s.json'),
        ('EGTM', 'results/egtm_deit_s_seed42.json'),
    ]:
        try:
            data = load_json(path)
            methods_data[name] = data
        except FileNotFoundError:
            pass

    if len(methods_data) < 2:
        print("  Skipping figure4: insufficient data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    markers = ['o', 's', '^', 'D', '*', 'v']
    for idx, (name, data) in enumerate(methods_data.items()):
        clean_acc = data.get('clean_acc', 0)
        if isinstance(clean_acc, dict):
            clean_acc = clean_acc['mean']
        throughput = data.get('throughput', data.get('throughput_clean', 0))
        corrupt_acc = data.get('mean_corrupt_acc_sev5', 0)
        if isinstance(corrupt_acc, dict):
            corrupt_acc = corrupt_acc['mean']

        axes[0].scatter(throughput, clean_acc * 100, s=120, marker=markers[idx],
                        label=name, color=COLORS[idx], zorder=5)
        axes[1].scatter(throughput, corrupt_acc * 100, s=120, marker=markers[idx],
                        label=name, color=COLORS[idx], zorder=5)

    for ax, title in zip(axes, ['Clean Accuracy', 'Corrupted Accuracy (sev=5 mean)']):
        ax.set_xlabel('Throughput (images/s)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'figure4_pareto')


# ============================================================
# Figure 5: Hyperparameter Sensitivity
# ============================================================
def figure5_sensitivity():
    """Accuracy vs alpha/beta hyperparameters."""
    try:
        alpha_data = load_json('results/sensitivity_alpha.json')
        beta_data = load_json('results/sensitivity_beta.json')
    except FileNotFoundError:
        print("  Skipping figure5: no sensitivity data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Alpha sweep
    alphas = sorted(alpha_data.keys(), key=float)
    clean_accs = [alpha_data[a]['clean_acc'] * 100 for a in alphas]
    corrupt_accs = [alpha_data[a]['mean_corrupt_acc'] * 100 for a in alphas]
    axes[0].plot([float(a) for a in alphas], clean_accs, 'o-', label='Clean', color=COLORS[0])
    axes[0].plot([float(a) for a in alphas], corrupt_accs, 's-', label='Corrupted (mean)', color=COLORS[1])
    axes[0].set_xlabel(r'$\alpha$ (minimum ratio floor)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(r'EGTM Sensitivity to $\alpha$')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Beta sweep
    betas = sorted(beta_data.keys(), key=float)
    clean_accs = [beta_data[b]['clean_acc'] * 100 for b in betas]
    corrupt_accs = [beta_data[b]['mean_corrupt_acc'] * 100 for b in betas]
    axes[1].plot([float(b) for b in betas], clean_accs, 'o-', label='Clean', color=COLORS[0])
    axes[1].plot([float(b) for b in betas], corrupt_accs, 's-', label='Corrupted (mean)', color=COLORS[1])
    axes[1].set_xlabel(r'$\beta$ (entropy sensitivity)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(r'EGTM Sensitivity to $\beta$')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'figure5_sensitivity')


# ============================================================
# Tables
# ============================================================
def generate_tables():
    """Generate LaTeX and CSV tables."""
    os.makedirs('tables', exist_ok=True)

    # Table 1: Main results
    try:
        final = load_json('results.json')
    except FileNotFoundError:
        print("  Skipping tables: no final results")
        return

    for model_short in ['deit_s', 'deit_b']:
        if model_short not in final.get('models', {}):
            continue
        m = final['models'][model_short]

        rows = []
        for method_name in ['no_reduction', 'tome', 'random_drop', 'evit_style', 'egtm']:
            if method_name not in m:
                continue
            md = m[method_name]
            if isinstance(md.get('clean_acc'), dict):
                clean = f"{md['clean_acc']['mean']*100:.1f} $\\pm$ {md['clean_acc']['std']*100:.1f}"
                corrupt = f"{md['mean_corrupt_acc_sev5']['mean']*100:.1f} $\\pm$ {md['mean_corrupt_acc_sev5']['std']*100:.1f}"
                rel_drop = f"{md['mean_rel_drop_sev5']['mean']*100:.1f} $\\pm$ {md['mean_rel_drop_sev5']['std']*100:.1f}"
                tp = md.get('throughput_clean', 0)
            else:
                clean = f"{md['clean_acc']*100:.1f}"
                corrupt = f"{md['mean_corrupt_acc_sev5']*100:.1f}"
                rel_drop = f"{md['mean_rel_drop_sev5']*100:.1f}"
                tp = md.get('throughput', 0)

            display_name = {
                'no_reduction': 'No Reduction',
                'tome': f'ToMe',
                'random_drop': 'Random Drop',
                'evit_style': 'EViT-style*',
                'egtm': 'EGTM (Ours)',
            }.get(method_name, method_name)

            rows.append(f"  {display_name} & {clean} & {corrupt} & {rel_drop} & {tp:.0f} \\\\")

        table = "\\begin{table}[t]\n\\centering\n"
        table += f"\\caption{{Main results on {model_short.upper().replace('_', '-')} with ImageNet-C severity 5.}}\n"
        table += "\\begin{tabular}{lcccc}\n\\toprule\n"
        table += "Method & Clean Acc (\\%) & Corrupt Acc (\\%) & Rel. Drop (\\%) & Throughput \\\\\n\\midrule\n"
        table += "\n".join(rows) + "\n"
        table += "\\bottomrule\n\\end{tabular}\n"
        table += "\\label{tab:main_" + model_short + "}\n"
        table += "\\end{table}\n"

        with open(f'tables/table1_main_results_{model_short}.tex', 'w') as f:
            f.write(table)
        print(f'  Saved tables/table1_main_results_{model_short}.tex')

    # Table 2: Ablation
    try:
        abl = load_json('results/ablation_results.json')
    except FileNotFoundError:
        print("  Skipping ablation table")
        return

    rows = []
    for name, data in abl.items():
        if name.startswith('_') or isinstance(data, dict) and 'clean_acc' in data:
            if isinstance(data, dict) and 'clean_acc' in data:
                rows.append(f"  {name} & {data['clean_acc']*100:.1f} & {data['mean_corrupt_acc']*100:.1f} & {data['throughput']:.0f} \\\\")

    if rows:
        table = "\\begin{table}[t]\n\\centering\n"
        table += "\\caption{Ablation study on DeiT-S with 3 corruption types at severity 5.}\n"
        table += "\\begin{tabular}{lccc}\n\\toprule\n"
        table += "Variant & Clean Acc (\\%) & Corrupt Acc (\\%) & Throughput \\\\\n\\midrule\n"
        table += "\n".join(rows) + "\n"
        table += "\\bottomrule\n\\end{tabular}\n"
        table += "\\label{tab:ablation}\n\\end{table}\n"

        with open('tables/table2_ablation.tex', 'w') as f:
            f.write(table)
        print('  Saved tables/table2_ablation.tex')

    # CSV table
    try:
        for model_short in ['deit_s', 'deit_b']:
            if model_short not in final.get('models', {}):
                continue
            m = final['models'][model_short]
            csv_lines = ['method,clean_acc,corrupt_acc_sev5,rel_drop_sev5,throughput']
            for method_name in ['no_reduction', 'tome', 'random_drop', 'evit_style', 'egtm']:
                if method_name not in m:
                    continue
                md = m[method_name]
                if isinstance(md.get('clean_acc'), dict):
                    csv_lines.append(f"{method_name},{md['clean_acc']['mean']:.4f},{md['mean_corrupt_acc_sev5']['mean']:.4f},{md['mean_rel_drop_sev5']['mean']:.4f},{md.get('throughput_clean', 0):.0f}")
                else:
                    csv_lines.append(f"{method_name},{md['clean_acc']:.4f},{md['mean_corrupt_acc_sev5']:.4f},{md['mean_rel_drop_sev5']:.4f},{md.get('throughput', 0):.0f}")
            with open(f'tables/table1_main_results_{model_short}.csv', 'w') as f:
                f.write('\n'.join(csv_lines) + '\n')
            print(f'  Saved tables/table1_main_results_{model_short}.csv')
    except Exception as e:
        print(f"  CSV generation error: {e}")


if __name__ == '__main__':
    print("Generating figures...")
    figure1_entropy_distribution()
    figure2_adaptive_ratio()
    figure3_per_corruption()
    figure4_pareto()
    figure5_sensitivity()

    print("\nGenerating tables...")
    generate_tables()

    print("\nDone!")
