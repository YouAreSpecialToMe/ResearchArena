#!/usr/bin/env python3
"""Generate all figures and tables for the paper."""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Style setup
sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
COLORS = sns.color_palette('colorblind', 5)
ALLOC_COLORS = {
    'uniform': COLORS[0],
    'independent': COLORS[1],
    'proportional': COLORS[2],
    'sketchbudget': COLORS[3],
}
ALLOC_LABELS = {
    'uniform': 'Uniform',
    'independent': 'Independent',
    'proportional': 'Proportional',
    'sketchbudget': 'SketchBudget (Ours)',
}
PIPELINE_LABELS = {
    'P1': 'P1: BF→CMS',
    'P2': 'P2: CMS→Threshold→HLL',
    'P3': 'P3: BF→CMS→Sum',
}

os.makedirs('figures', exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 2: Error vs Memory Budget
# ============================================================
def figure_error_vs_budget():
    data = load_json('results/main_experiments.json')
    ds_name = 'zipfian_1.0'

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, pipeline in enumerate(['P1', 'P2', 'P3']):
        ax = axes[idx]
        for alloc_name in ['uniform', 'independent', 'proportional', 'sketchbudget']:
            subset = [r for r in data if r['pipeline'] == pipeline
                      and r['allocator'] == alloc_name and r['dataset'] == ds_name]
            if not subset:
                continue
            subset.sort(key=lambda x: x['budget'])
            budgets = [r['budget'] for r in subset]
            means = [r['mean'] for r in subset]
            stds = [r['std'] for r in subset]

            ax.errorbar(budgets, means, yerr=stds,
                        label=ALLOC_LABELS[alloc_name],
                        color=ALLOC_COLORS[alloc_name],
                        marker='o', markersize=5, linewidth=2, capsize=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Memory Budget (bytes)')
        ax.set_ylabel('End-to-End Error')
        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/error_vs_budget.pdf', bbox_inches='tight')
    plt.savefig('figures/error_vs_budget.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/error_vs_budget.pdf")


# ============================================================
# Figure 3: Bound tightness comparison
# ============================================================
def figure_bound_tightness():
    data = load_json('results/bound_tightness.json')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, pipeline in enumerate(['P1', 'P2', 'P3']):
        ax = axes[idx]
        subset = [r for r in data if r['pipeline'] == pipeline]
        if not subset:
            continue
        subset.sort(key=lambda x: x['alpha'])
        alphas = [r['alpha'] for r in subset]
        naive_means = [r['naive_tightness_mean'] for r in subset]
        naive_stds = [r['naive_tightness_std'] for r in subset]
        tight_means = [r['tight_tightness_mean'] for r in subset]
        tight_stds = [r['tight_tightness_std'] for r in subset]

        x = np.arange(len(alphas))
        width = 0.35
        ax.bar(x - width/2, naive_means, width, yerr=naive_stds,
               label='Naive Bound', color=COLORS[0], capsize=3)
        ax.bar(x + width/2, tight_means, width, yerr=tight_stds,
               label='Tight Bound (Ours)', color=COLORS[3], capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f'α={a}' for a in alphas])
        ax.set_ylabel('Bound / Observed Error')
        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect tightness')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/bound_tightness.pdf', bbox_inches='tight')
    plt.savefig('figures/bound_tightness.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/bound_tightness.pdf")


# ============================================================
# Figure 4: Memory allocation breakdown
# ============================================================
def figure_allocation_breakdown():
    data = load_json('results/main_experiments.json')
    ds_name = 'zipfian_1.0'
    budget = 500000

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, pipeline in enumerate(['P1', 'P2', 'P3']):
        ax = axes[idx]
        subset = [r for r in data if r['pipeline'] == pipeline
                  and r['dataset'] == ds_name and r['budget'] == budget]
        if not subset:
            continue

        alloc_names = []
        stage_data = {}
        for r in subset:
            alloc_name = r['allocator']
            alloc_names.append(ALLOC_LABELS[alloc_name])
            for stage, mem in r['allocation'].items():
                if stage not in stage_data:
                    stage_data[stage] = []
                stage_data[stage].append(mem)

        x = np.arange(len(alloc_names))
        bottom = np.zeros(len(alloc_names))
        stage_colors = [COLORS[0], COLORS[1], COLORS[2]]

        for i, (stage, values) in enumerate(stage_data.items()):
            ax.bar(x, values, 0.6, bottom=bottom, label=stage.upper(),
                   color=stage_colors[i % len(stage_colors)])
            bottom += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels(alloc_names, rotation=30, ha='right')
        ax.set_ylabel('Memory (bytes)')
        ax.set_title(f'{PIPELINE_LABELS[pipeline]} (500KB)')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/allocation_breakdown.pdf', bbox_inches='tight')
    plt.savefig('figures/allocation_breakdown.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/allocation_breakdown.pdf")


# ============================================================
# Figure 5: Pipeline depth scaling
# ============================================================
def figure_depth_scaling():
    data = load_json('results/ablation_depth.json')

    fig, ax = plt.subplots(figsize=(7, 5))
    for alloc_name in ['uniform', 'independent', 'proportional', 'sketchbudget']:
        subset = [r for r in data if r['allocator'] == alloc_name]
        if not subset:
            continue
        subset.sort(key=lambda x: x['depth'])
        depths = [r['depth'] for r in subset]
        means = [r['mean'] for r in subset]
        stds = [r['std'] for r in subset]

        ax.errorbar(depths, means, yerr=stds,
                    label=ALLOC_LABELS[alloc_name],
                    color=ALLOC_COLORS[alloc_name],
                    marker='o', markersize=6, linewidth=2, capsize=3)

    ax.set_xlabel('Pipeline Depth (stages)')
    ax.set_ylabel('End-to-End Error')
    ax.set_title('Error vs Pipeline Depth')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('figures/depth_scaling.pdf', bbox_inches='tight')
    plt.savefig('figures/depth_scaling.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/depth_scaling.pdf")


# ============================================================
# Figure 6: Distribution sensitivity
# ============================================================
def figure_distribution_sensitivity():
    data = load_json('results/ablation_distribution.json')

    fig, ax = plt.subplots(figsize=(8, 5))
    for alloc_name in ['uniform', 'independent', 'proportional', 'sketchbudget']:
        subset = [r for r in data if r['allocator'] == alloc_name]
        if not subset:
            continue
        subset.sort(key=lambda x: x['alpha'])
        alphas = [r['alpha'] for r in subset]
        means = [r['mean'] for r in subset]
        stds = [r['std'] for r in subset]

        ax.errorbar(alphas, means, yerr=stds,
                    label=ALLOC_LABELS[alloc_name],
                    color=ALLOC_COLORS[alloc_name],
                    marker='o', markersize=6, linewidth=2, capsize=3)

    ax.set_xlabel('Zipf α (0=Uniform, higher=more skewed)')
    ax.set_ylabel('Cardinality Error (P2)')
    ax.set_title('Distribution Sensitivity (P2: CMS→Threshold→HLL, 500KB)')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('figures/distribution_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('figures/distribution_sensitivity.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/distribution_sensitivity.pdf")


# ============================================================
# Figure: Budget sensitivity (improvement ratio)
# ============================================================
def figure_budget_sensitivity():
    data = load_json('results/ablation_budget.json')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, pipeline in enumerate(['P1', 'P2', 'P3']):
        ax = axes[idx]

        # Get uniform as reference
        uniform_data = {r['budget']: r['mean'] for r in data
                        if r['pipeline'] == pipeline and r['allocator'] == 'uniform'}

        for alloc_name in ['independent', 'proportional', 'sketchbudget']:
            subset = [r for r in data if r['pipeline'] == pipeline and r['allocator'] == alloc_name]
            if not subset:
                continue
            subset.sort(key=lambda x: x['budget'])
            budgets = [r['budget'] for r in subset]
            improvements = []
            for r in subset:
                uf = uniform_data.get(r['budget'], r['mean'])
                if uf > 0:
                    improvements.append((uf - r['mean']) / uf * 100)
                else:
                    improvements.append(0)

            ax.plot(budgets, improvements,
                    label=ALLOC_LABELS[alloc_name],
                    color=ALLOC_COLORS[alloc_name],
                    marker='o', markersize=4, linewidth=2)

        ax.set_xscale('log')
        ax.set_xlabel('Memory Budget (bytes)')
        ax.set_ylabel('Improvement over Uniform (%)')
        ax.set_title(PIPELINE_LABELS[pipeline])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('figures/budget_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('figures/budget_sensitivity.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/budget_sensitivity.pdf")


# ============================================================
# Table 1: Main results
# ============================================================
def table_main_results():
    data = load_json('results/main_experiments.json')
    ds_name = 'zipfian_1.0'
    budget = 500000

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{End-to-end error (mean $\pm$ std) at 500KB budget on Zipfian ($\alpha=1.0$) data. Lower is better. Relative improvement of SketchBudget over best baseline shown in last column.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Pipeline & Metric & Uniform & Independent & Proportional & SketchBudget (Ours) \\")
    lines.append(r"\midrule")

    for pipeline in ['P1', 'P2', 'P3']:
        subset = [r for r in data if r['pipeline'] == pipeline
                  and r['dataset'] == ds_name and r['budget'] == budget]
        if not subset:
            continue

        row = {}
        for r in subset:
            row[r['allocator']] = (r['mean'], r['std'])

        metric_name = subset[0]['primary_metric']
        best_baseline = min(row[a][0] for a in ['uniform', 'independent', 'proportional'] if a in row)
        sb_val = row.get('sketchbudget', (0, 0))[0]
        improvement = (best_baseline - sb_val) / best_baseline * 100 if best_baseline > 0 else 0

        cells = [PIPELINE_LABELS[pipeline], metric_name.replace('_', r'\_')]
        for alloc in ['uniform', 'independent', 'proportional', 'sketchbudget']:
            if alloc in row:
                m, s = row[alloc]
                cells.append(f"${m:.2f} \\pm {s:.2f}$")
            else:
                cells.append("--")

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open('figures/main_results_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved figures/main_results_table.tex")


# ============================================================
# Table 2: Memory savings
# ============================================================
def table_memory_savings():
    data = load_json('results/ablation_budget.json')

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Memory savings: budget needed by each allocator to match the error of Uniform at 500KB.}")
    lines.append(r"\label{tab:memory_savings}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Pipeline & Independent & Proportional & SketchBudget & Savings (\%) \\")
    lines.append(r"\midrule")

    for pipeline in ['P1', 'P2', 'P3']:
        # Get uniform error at 500KB
        uniform_at_500k = None
        for r in data:
            if r['pipeline'] == pipeline and r['allocator'] == 'uniform':
                if abs(r['budget'] - 500000) < 50000:
                    uniform_at_500k = r['mean']
                    break

        if uniform_at_500k is None:
            continue

        # Find budget needed by each allocator to achieve this error
        savings_row = [PIPELINE_LABELS[pipeline]]
        for alloc in ['independent', 'proportional', 'sketchbudget']:
            subset = [r for r in data if r['pipeline'] == pipeline and r['allocator'] == alloc]
            subset.sort(key=lambda x: x['budget'])

            needed_budget = None
            for r in subset:
                if r['mean'] <= uniform_at_500k:
                    needed_budget = r['budget']
                    break

            if needed_budget:
                saving = (1 - needed_budget / 500000) * 100
                savings_row.append(f"{needed_budget//1000}KB ({saving:.0f}\\%)")
            else:
                savings_row.append(">10MB")

        # SketchBudget savings
        sb_subset = [r for r in data if r['pipeline'] == pipeline and r['allocator'] == 'sketchbudget']
        sb_subset.sort(key=lambda x: x['budget'])
        sb_needed = None
        for r in sb_subset:
            if r['mean'] <= uniform_at_500k:
                sb_needed = r['budget']
                break
        if sb_needed:
            savings_row.append(f"{(1-sb_needed/500000)*100:.0f}\\%")
        else:
            savings_row.append("N/A")

        lines.append(" & ".join(savings_row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open('figures/memory_savings_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved figures/memory_savings_table.tex")


# ============================================================
# Data distribution figure
# ============================================================
def figure_data_distribution():
    from src.data_gen import generate_zipfian_stream
    from collections import Counter

    stream = generate_zipfian_stream(100000, 1000000, alpha=1.0, seed=42)
    freq = Counter(stream.tolist())
    ranks = sorted(freq.values(), reverse=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(range(1, len(ranks) + 1), ranks, '-', linewidth=1.5, color=COLORS[0])
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Zipfian Stream (α=1.0) Frequency-Rank Distribution')
    plt.tight_layout()
    plt.savefig('figures/data_distribution.pdf', bbox_inches='tight')
    plt.savefig('figures/data_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved figures/data_distribution.pdf")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    figure_data_distribution()
    figure_error_vs_budget()
    figure_bound_tightness()
    figure_allocation_breakdown()
    figure_depth_scaling()
    figure_distribution_sensitivity()
    figure_budget_sensitivity()
    table_main_results()
    table_memory_savings()

    print("\nAll figures and tables generated!")
