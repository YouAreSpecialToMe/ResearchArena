"""Generate all publication-quality figures for FlipBench paper."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DOMAINS = ['propositional_logic', 'arithmetic_reasoning',
           'relational_reasoning', 'function_computation']
DOMAIN_LABELS = {
    'propositional_logic': 'Prop. Logic',
    'arithmetic_reasoning': 'Arithmetic',
    'relational_reasoning': 'Relational',
    'function_computation': 'Function'
}

MODEL_LABELS = {
    'phi35': 'Phi-3.5-mini (3.8B)',
    'llama31_8b': 'Llama-3.1-8B',
    'qwen25_7b': 'Qwen2.5-7B',
    'deepseek_r1_7b': 'DeepSeek-R1-7B',
    'qwen25_32b': 'Qwen2.5-32B'
}

MODELS_ORDER = ['phi35', 'llama31_8b', 'qwen25_7b', 'deepseek_r1_7b', 'qwen25_32b']

# Color scheme
COLORS = {
    'phi35': '#1f77b4',
    'llama31_8b': '#ff7f0e',
    'qwen25_7b': '#2ca02c',
    'deepseek_r1_7b': '#d62728',
    'qwen25_32b': '#9467bd'
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_parsed(model_short, seed='seed_42', cot=False):
    suffix = '_cot' if cot else ''
    path = os.path.join(RESULTS_DIR, 'parsed', f'{model_short}{suffix}_{seed}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig1_drg_heatmap():
    """Figure 1: DRG heatmap (models × domains)."""
    models = [m for m in MODELS_ORDER if load_parsed(m) is not None]
    data = np.zeros((len(models), len(DOMAINS)))

    for i, model in enumerate(models):
        parsed = load_parsed(model)
        for j, domain in enumerate(DOMAINS):
            data[i, j] = parsed[domain]['drg'] * 100  # Convert to percentage points

    fig, ax = plt.subplots(figsize=(8, 5))
    # Diverging colormap: blue = backward better, red = forward better, white = symmetric
    max_abs = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)

    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], rotation=30, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in models])

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(DOMAINS)):
            color = 'white' if abs(data[i, j]) > 20 else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('DRG (pp): Red = Forward > Backward, Blue = Backward > Forward')
    ax.set_title('Directional Reasoning Gap (DRG) by Model and Domain')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig1_drg_heatmap.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")


def fig2_forward_vs_backward():
    """Figure 2: Forward vs Backward accuracy scatter."""
    fig, ax = plt.subplots(figsize=(8, 7))

    markers = {'propositional_logic': 'o', 'arithmetic_reasoning': 's',
               'relational_reasoning': '^', 'function_computation': 'D'}

    for model in MODELS_ORDER:
        parsed = load_parsed(model)
        if not parsed:
            continue
        for domain in DOMAINS:
            fa = parsed[domain]['forward_accuracy'] * 100
            ba = parsed[domain]['backward_accuracy'] * 100
            ax.scatter(fa, ba, c=COLORS[model], marker=markers[domain],
                       s=100, edgecolors='black', linewidths=0.5, zorder=3)

    # Diagonal line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='DRG = 0')
    ax.set_xlabel('Forward Accuracy (%)')
    ax.set_ylabel('Backward Accuracy (%)')
    ax.set_title('Forward vs. Backward Accuracy')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)

    # Legend for models
    model_handles = [plt.Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=COLORS[m], markersize=8, label=MODEL_LABELS[m])
                     for m in MODELS_ORDER if load_parsed(m)]
    # Legend for domains
    domain_handles = [plt.Line2D([0], [0], marker=markers[d], color='w',
                      markerfacecolor='gray', markersize=8, label=DOMAIN_LABELS[d])
                      for d in DOMAINS]

    first_legend = ax.legend(handles=model_handles, loc='upper left', title='Models')
    ax.add_artist(first_legend)
    ax.legend(handles=domain_handles, loc='lower right', title='Domains')

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig2_forward_vs_backward.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")


def fig3_drg_by_difficulty():
    """Figure 3: DRG vs difficulty level, 2×2 subplot by domain."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axes = axes.flatten()

    for idx, domain in enumerate(DOMAINS):
        ax = axes[idx]
        x = np.arange(3)
        width = 0.15
        models_available = [m for m in MODELS_ORDER if load_parsed(m)]

        for i, model in enumerate(models_available):
            parsed = load_parsed(model)
            if not parsed or domain not in parsed:
                continue
            drgs = []
            for diff in [1, 2, 3]:
                key = f'difficulty_{diff}'
                drgs.append(parsed[domain].get(key, {}).get('drg', 0) * 100)

            offset = (i - len(models_available) / 2) * width + width / 2
            ax.bar(x + offset, drgs, width, label=MODEL_LABELS[model],
                   color=COLORS[model], edgecolor='black', linewidth=0.5)

        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xticks(x)
        ax.set_xticklabels(['Easy', 'Medium', 'Hard'])
        ax.set_ylabel('DRG (pp)' if idx % 2 == 0 else '')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    axes[0].legend(fontsize=7, ncol=1, loc='upper left')
    fig.suptitle('DRG by Difficulty Level Across Domains', fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig3_drg_by_difficulty.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")


def fig4_standard_vs_reasoning():
    """Figure 4: Standard vs reasoning-optimized model comparison."""
    standard = load_parsed('qwen25_7b')
    reasoning = load_parsed('deepseek_r1_7b')

    if not standard or not reasoning:
        print("Figure 4 skipped (missing data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DOMAINS))
    width = 0.35

    std_drgs = [standard[d]['drg'] * 100 for d in DOMAINS]
    reas_drgs = [reasoning[d]['drg'] * 100 for d in DOMAINS]

    bars1 = ax.bar(x - width / 2, std_drgs, width, label='Qwen2.5-7B (Standard)',
                   color='#2ca02c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, reas_drgs, width, label='DeepSeek-R1-7B (Reasoning)',
                   color='#d62728', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('DRG (percentage points)')
    ax.set_title('Standard vs. Reasoning-Optimized: Directional Reasoning Gap')
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS])
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            y_pos = height + 0.5 if height >= 0 else height - 1.5
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig4_standard_vs_reasoning.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")


def fig5_cot_ablation():
    """Figure 5: CoT ablation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models_for_cot = [('llama31_8b', 'Llama-3.1-8B'), ('deepseek_r1_7b', 'DeepSeek-R1-7B')]

    for ax_idx, (model_short, model_label) in enumerate(models_for_cot):
        ax = axes[ax_idx]
        no_cot = load_parsed(model_short)
        with_cot = load_parsed(model_short, cot=True)

        if not no_cot or not with_cot:
            ax.set_title(f'{model_label} (data missing)')
            continue

        x = np.arange(len(DOMAINS))
        width = 0.35

        drgs_no = [no_cot[d]['drg'] * 100 for d in DOMAINS]
        drgs_cot = [with_cot[d]['drg'] * 100 for d in DOMAINS]

        ax.bar(x - width / 2, drgs_no, width, label='Standard', color='#4c72b0',
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, drgs_cot, width, label='+ CoT', color='#55a868',
               edgecolor='black', linewidth=0.5)

        ax.set_title(model_label)
        ax.set_xticks(x)
        ax.set_xticklabels([DOMAIN_LABELS[d] for d in DOMAINS], rotation=20, ha='right')
        ax.set_ylabel('DRG (pp)' if ax_idx == 0 else '')
        ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)

    fig.suptitle('Effect of Chain-of-Thought Prompting on DRG', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig5_cot_ablation.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 5 saved.")


def fig6_consistency():
    """Figure 6: Consistency analysis - stacked bar chart."""
    models = [m for m in MODELS_ORDER if load_parsed(m)]

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(14, 5), sharey=True)

    for d_idx, domain in enumerate(DOMAINS):
        ax = axes[d_idx]
        x = np.arange(len(models))

        both_correct = []
        fwd_only = []
        bwd_only = []
        both_wrong = []

        for model in models:
            parsed = load_parsed(model)
            d = parsed[domain]
            total = d['n_pairs']
            both_correct.append(d['both_correct'] / total * 100)
            fwd_only.append(d['fwd_only_correct'] / total * 100)
            bwd_only.append(d['bwd_only_correct'] / total * 100)
            both_wrong.append(d['both_wrong'] / total * 100)

        ax.bar(x, both_correct, label='Both Correct', color='#2ca02c')
        ax.bar(x, fwd_only, bottom=both_correct, label='Fwd Only', color='#ff7f0e')
        ax.bar(x, bwd_only,
               bottom=[a + b for a, b in zip(both_correct, fwd_only)],
               label='Bwd Only', color='#1f77b4')
        ax.bar(x, both_wrong,
               bottom=[a + b + c for a, b, c in zip(both_correct, fwd_only, bwd_only)],
               label='Both Wrong', color='#d62728')

        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m].split('(')[0].strip() for m in models],
                           rotation=45, ha='right', fontsize=8)
        if d_idx == 0:
            ax.set_ylabel('Percentage of Pairs (%)')

    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    fig.suptitle('Pair-level Consistency Analysis', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig6_consistency.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 6 saved.")


def fig7_accuracy_breakdown():
    """Figure 7: Forward and Backward accuracy per model per domain."""
    models = [m for m in MODELS_ORDER if load_parsed(m)]

    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(14, 5), sharey=True)

    for d_idx, domain in enumerate(DOMAINS):
        ax = axes[d_idx]
        x = np.arange(len(models))
        width = 0.35

        fas = []
        bas = []
        for model in models:
            parsed = load_parsed(model)
            fas.append(parsed[domain]['forward_accuracy'] * 100)
            bas.append(parsed[domain]['backward_accuracy'] * 100)

        ax.bar(x - width / 2, fas, width, label='Forward', color='#4c72b0',
               edgecolor='black', linewidth=0.5)
        ax.bar(x + width / 2, bas, width, label='Backward', color='#c44e52',
               edgecolor='black', linewidth=0.5)

        ax.set_title(DOMAIN_LABELS[domain])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m].split('(')[0].strip() for m in models],
                           rotation=45, ha='right', fontsize=8)
        if d_idx == 0:
            ax.set_ylabel('Accuracy (%)')
        if d_idx == len(DOMAINS) - 1:
            ax.legend()

    fig.suptitle('Forward vs. Backward Accuracy by Domain', fontsize=14)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig7_accuracy_breakdown.{ext}'), bbox_inches='tight')
    plt.close()
    print("Figure 7 saved.")


def generate_latex_table():
    """Generate LaTeX table for main results."""
    models = [m for m in MODELS_ORDER if load_parsed(m)]

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Main FlipBench Results. FA = Forward Accuracy, BA = Backward Accuracy, DRG = FA $-$ BA, CR = Consistency Rate. Bold indicates largest DRG per domain.}',
        r'\label{tab:main_results}',
        r'\resizebox{\textwidth}{!}{',
        r'\begin{tabular}{l' + 'cccc' * 4 + '}',
        r'\toprule',
    ]

    # Header
    header = r'Model'
    for domain in DOMAINS:
        header += r' & \multicolumn{4}{c}{' + DOMAIN_LABELS[domain] + '}'
    header += r' \\'
    lines.append(header)

    subheader = ''
    for _ in DOMAINS:
        subheader += r' & FA & BA & DRG & CR'
    subheader += r' \\'
    lines.append(subheader)
    lines.append(r'\midrule')

    # Find max DRG per domain for bolding
    max_drg = {}
    for domain in DOMAINS:
        drgs = []
        for model in models:
            parsed = load_parsed(model)
            drgs.append((parsed[domain]['drg'], model))
        max_drg[domain] = max(drgs, key=lambda x: x[0])[1]

    for model in models:
        parsed = load_parsed(model)
        row = MODEL_LABELS[model]
        for domain in DOMAINS:
            d = parsed[domain]
            drg_str = f"{d['drg']*100:.1f}"
            if model == max_drg[domain]:
                drg_str = r'\textbf{' + drg_str + '}'
            row += f" & {d['forward_accuracy']*100:.1f} & {d['backward_accuracy']*100:.1f} & {drg_str} & {d['consistency_rate']*100:.1f}"
        row += r' \\'
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table}'
    ])

    outpath = os.path.join(FIGURES_DIR, 'table1_main_results.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Table 1 saved to {outpath}")


def generate_seed_stability_table():
    """Generate LaTeX table for cross-seed stability."""
    stat_path = os.path.join(RESULTS_DIR, 'aggregated', 'statistical_tests.json')
    if not os.path.exists(stat_path):
        print("Seed stability table skipped (no statistical tests)")
        return

    with open(stat_path) as f:
        stat_results = json.load(f)

    stability = stat_results.get('cross_seed_stability', {})

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Cross-seed stability of DRG (mean $\pm$ std across 3 seeds).}',
        r'\label{tab:seed_stability}',
        r'\begin{tabular}{l' + 'c' * 4 + '}',
        r'\toprule',
        r'Model & ' + ' & '.join([DOMAIN_LABELS[d] for d in DOMAINS]) + r' \\',
        r'\midrule',
    ]

    for model in ['llama31_8b', 'deepseek_r1_7b']:
        if model not in stability:
            continue
        row = MODEL_LABELS[model]
        for domain in DOMAINS:
            if domain in stability[model]:
                d = stability[model][domain]
                row += f" & {d['mean']*100:.1f} $\\pm$ {d['std']*100:.1f}"
            else:
                row += " & --"
        row += r' \\'
        lines.append(row)

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}'
    ])

    outpath = os.path.join(FIGURES_DIR, 'table2_seed_stability.tex')
    with open(outpath, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Table 2 saved to {outpath}")


def main():
    print("Generating FlipBench figures...")
    fig1_drg_heatmap()
    fig2_forward_vs_backward()
    fig3_drg_by_difficulty()
    fig4_standard_vs_reasoning()
    fig5_cot_ablation()
    fig6_consistency()
    fig7_accuracy_breakdown()
    generate_latex_table()
    generate_seed_stability_table()
    print("\nAll figures generated!")


if __name__ == '__main__':
    main()
