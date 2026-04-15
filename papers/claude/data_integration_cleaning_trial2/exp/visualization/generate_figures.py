"""Generate all publication-quality figures."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASETS, DATASET_DIFFICULTY, RESULTS_DIR, FIGURES_DIR

# Style setup
plt.style.use('seaborn-v0_8-paper')
sns.set_palette('Set2')
FIGSIZE_SINGLE = (3.5, 3.0)
FIGSIZE_DOUBLE = (7, 3.5)
FONTSIZE = 11
plt.rcParams.update({'font.size': FONTSIZE, 'axes.labelsize': FONTSIZE,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9})

DIFFICULTY_COLORS = {'easy': '#66c2a5', 'medium': '#fc8d62', 'hard': '#8da0cb'}
DATASET_SHORT = {
    'dblp_acm': 'DBLP-ACM', 'dblp_scholar': 'DBLP-Sch',
    'amazon_google': 'Amz-Ggl', 'walmart_amazon': 'Wal-Amz',
    'abt_buy': 'Abt-Buy', 'fodors_zagats': 'Fod-Zag'
}

os.makedirs(FIGURES_DIR, exist_ok=True)


def fig1_pipeline_schematic():
    """Pipeline architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    boxes = [
        (1, 1.2, 'Blocking\n(PC, RR)', '#a6cee3'),
        (4, 1.2, 'Matching\n(MP, MR)', '#b2df8a'),
        (7, 1.2, 'Clustering\n(CP, CR)', '#fb9a99'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.8, y-0.5), 1.6, 1.0,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    for x1, x2 in [(1.8, 3.2), (4.8, 6.2)]:
        ax.annotate('', xy=(x2, 1.2), xytext=(x1, 1.2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Annotations
    ax.text(2.5, 2.2, 'Irrecoverable\nerrors', ha='center', fontsize=8, color='red', fontstyle='italic')
    ax.text(5.5, 0.3, 'FP amplification\nFN recovery', ha='center', fontsize=8, color='blue', fontstyle='italic')

    ax.text(0, 1.2, 'All\nPairs', ha='center', va='center', fontsize=9)
    ax.text(9.5, 1.2, 'Entity\nClusters', ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(0.2, 1.2), xytext=(-0.2, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(9.3, 1.2), xytext=(7.8, 1.2),
                arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.text(5, 2.8, 'R$_{e2e}$ $\\leq$ PC  (Hard Recall Bound)', ha='center',
            fontsize=11, fontweight='bold', color='darkred',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_pipeline_schematic.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig1_pipeline_schematic.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 1: Pipeline schematic")


def fig2_amplification_curves():
    """Error amplification curves per stage."""
    with open(os.path.join(RESULTS_DIR, 'exp2', 'all_results.json')) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=True)
    stages = ['blocking', 'matching', 'clustering']
    stage_modes = {'blocking': 'both', 'matching': 'both', 'clustering': 'split'}

    for ax_idx, stage in enumerate(stages):
        ax = axes[ax_idx]
        for dataset in DATASETS:
            diff = DATASET_DIFFICULTY[dataset]
            color = DIFFICULTY_COLORS[diff]

            # Get baseline F1
            baselines = [r for r in results if r['dataset'] == dataset and r['stage'] == 'none']
            if not baselines:
                continue
            base_f1 = np.mean([r['degraded_e2e_f1'] for r in baselines])

            # Get degradation results
            stage_results = [r for r in results
                           if r['dataset'] == dataset and r['stage'] == stage
                           and r.get('mode') == stage_modes[stage]]

            levels = sorted(set(r['degradation_level'] for r in stage_results))
            means = []
            stds = []
            for level in levels:
                f1s = [r['degraded_e2e_f1'] for r in stage_results if r['degradation_level'] == level]
                means.append(np.mean(f1s))
                stds.append(np.std(f1s))

            if levels:
                levels_plot = [0] + levels
                means_plot = [base_f1] + means
                stds_plot = [0] + stds

                ax.plot(np.array(levels_plot) * 100, means_plot, '-o', color=color,
                       label=DATASET_SHORT[dataset], markersize=3, linewidth=1.2)
                ax.fill_between(np.array(levels_plot) * 100,
                              np.array(means_plot) - np.array(stds_plot),
                              np.array(means_plot) + np.array(stds_plot),
                              alpha=0.15, color=color)

        ax.set_xlabel('Degradation (%)')
        ax.set_title(stage.capitalize(), fontweight='bold')
        if ax_idx == 0:
            ax.set_ylabel('End-to-End F1')

    axes[2].legend(fontsize=7, loc='lower left', ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_amplification_curves.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig2_amplification_curves.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 2: Amplification curves")


def fig3_epm_validation():
    """EPM validation scatter plot."""
    with open(os.path.join(RESULTS_DIR, 'exp3', 'predictions_vs_actuals.json')) as f:
        pva = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    for ax, split_name, mask_fn in [
        (axes[0], 'Training', lambda x: x),
        (axes[1], 'Validation', lambda x: not x)
    ]:
        pred = [p for p, t in zip(pva['predicted'], pva['is_train']) if mask_fn(t)]
        actual = [a for a, t in zip(pva['actual'], pva['is_train']) if mask_fn(t)]
        ds_labels = [d for d, t in zip(pva['datasets'], pva['is_train']) if mask_fn(t)]

        for ds in DATASETS:
            color = DIFFICULTY_COLORS[DATASET_DIFFICULTY[ds]]
            ds_pred = [p for p, d in zip(pred, ds_labels) if d == ds]
            ds_actual = [a for a, d in zip(actual, ds_labels) if d == ds]
            if ds_pred:
                ax.scatter(ds_actual, ds_pred, c=color, s=15, alpha=0.6,
                          label=DATASET_SHORT[ds], edgecolors='none')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ss_res = sum((a - p) ** 2 for a, p in zip(actual, pred))
        ss_tot = sum((a - np.mean(actual)) ** 2 for a in actual)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=10)
        ax.set_xlabel('Actual E2E F1')
        ax.set_ylabel('Predicted E2E F1')
        ax.set_title(split_name, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    axes[1].legend(fontsize=7, loc='lower right', ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_epm_validation.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig3_epm_validation.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 3: EPM validation")


def fig4_eaf_heatmap():
    """EAF comparison heatmap."""
    with open(os.path.join(RESULTS_DIR, 'exp4', 'eaf_analysis.json')) as f:
        eaf_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    stages = ['blocking', 'matching', 'clustering']
    ds_order = [ds for ds in DATASETS if ds in eaf_data]

    for ax, eaf_type, title in [
        (axes[0], 'empirical', 'Empirical EAFs'),
        (axes[1], 'analytical_normalized', 'Analytical EAFs')
    ]:
        matrix = []
        for ds in ds_order:
            if eaf_type == 'empirical':
                row = [eaf_data[ds]['empirical']['normalized'].get(s, 0) for s in stages]
            else:
                row = [eaf_data[ds]['analytical_normalized'].get(s, 0) for s in stages]
            matrix.append(row)

        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)

        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([s.capitalize() for s in stages])
        ax.set_yticks(range(len(ds_order)))
        ax.set_yticklabels([DATASET_SHORT[ds] for ds in ds_order])
        ax.set_title(title, fontweight='bold')

        for i in range(len(ds_order)):
            for j in range(len(stages)):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                       fontsize=8, color='black' if matrix[i,j] < 0.5 else 'white')

    fig.colorbar(im, ax=axes, shrink=0.8, label='Normalized EAF')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_eaf_heatmap.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig4_eaf_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 4: EAF heatmap")


def fig5_soa_budget():
    """SOA budget-quality tradeoff."""
    with open(os.path.join(RESULTS_DIR, 'exp5', 'soa_validation.json')) as f:
        results = json.load(f)

    ds_list = sorted(set(r['dataset'] for r in results))
    n_ds = len(ds_list)
    n_cols = min(3, n_ds)
    n_rows = (n_ds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 2.5 * n_rows), squeeze=False)

    strategy_colors = {'uniform': '#66c2a5', 'bottleneck': '#fc8d62', 'soa': '#8da0cb'}
    strategy_markers = {'uniform': 's', 'bottleneck': '^', 'soa': 'o'}

    for idx, dataset in enumerate(ds_list):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        for strategy in ['uniform', 'bottleneck', 'soa']:
            strat_results = [r for r in results
                           if r['dataset'] == dataset and r['strategy'] == strategy]
            budgets = sorted(set(r['budget'] for r in strat_results))
            means = []
            stds = []
            for b in budgets:
                f1s = [r['predicted_f1'] for r in strat_results if r['budget'] == b]
                means.append(np.mean(f1s))
                stds.append(np.std(f1s))

            ax.errorbar(budgets, means, yerr=stds, fmt=strategy_markers[strategy] + '-',
                       color=strategy_colors[strategy], label=strategy.upper(),
                       markersize=4, linewidth=1.2, capsize=2)

        ax.set_title(DATASET_SHORT[dataset], fontweight='bold', fontsize=10)
        ax.set_xlabel('Budget')
        if col == 0:
            ax.set_ylabel('Predicted F1')

    axes[0][n_cols-1].legend(fontsize=7, loc='lower right')

    # Hide empty subplots
    for idx in range(n_ds, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_soa_budget.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_soa_budget.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 5: SOA budget tradeoff")


def fig6_ablation():
    """Ablation results."""
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        results = json.load(f)

    ablation = results['ablation']
    variants = ['full_epm', 'no_transitive', 'no_topology', 'linear', 'multiplicative']
    labels = ['Full EPM', 'No Trans.', 'No Topo.', 'Linear', 'Multiplicative']
    r2_values = [ablation[v]['r2'] for v in variants]
    rmse_values = [ablation[v]['rmse'] for v in variants]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    colors = sns.color_palette('Set2', len(variants))
    x = np.arange(len(variants))

    ax1.bar(x, r2_values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax1.set_ylabel('R²')
    ax1.set_title('Model Accuracy (R²)', fontweight='bold')
    ax1.axhline(y=0.85, color='red', linestyle='--', alpha=0.5, label='Target R²=0.85')
    ax1.legend(fontsize=7)

    ax2.bar(x, rmse_values, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('RMSE')
    ax2.set_title('Prediction Error (RMSE)', fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_ablation.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig6_ablation.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 6: Ablation results")


def fig7_transferability():
    """Transferability analysis."""
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        results = json.load(f)

    transfer = results['transferability']
    ds_list = [ds for ds in DATASETS if ds in transfer]

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_SINGLE)

    x = np.arange(len(ds_list))
    width = 0.35

    lodo_r2 = [transfer[ds]['lodo_r2'] for ds in ds_list]
    within_r2 = [transfer[ds]['within_r2'] for ds in ds_list]

    ax.bar(x - width/2, within_r2, width, label='Within-dataset', color='#66c2a5', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, lodo_r2, width, label='Leave-one-out', color='#fc8d62', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_SHORT[ds] for ds in ds_list], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('EPM Transferability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0.85, color='red', linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'fig7_transferability.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig7_transferability.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Fig 7: Transferability")


def main():
    print("Generating figures...")
    fig1_pipeline_schematic()

    # Check which experiment results are available
    for fig_fn, required in [
        (fig2_amplification_curves, 'exp2'),
        (fig3_epm_validation, 'exp3'),
        (fig4_eaf_heatmap, 'exp4'),
        (fig5_soa_budget, 'exp5'),
        (fig6_ablation, 'ablation'),
        (fig7_transferability, 'ablation'),
    ]:
        req_dir = os.path.join(RESULTS_DIR, required)
        if os.path.exists(req_dir) and os.listdir(req_dir):
            try:
                fig_fn()
            except Exception as e:
                print(f"  Error generating {fig_fn.__name__}: {e}")
        else:
            print(f"  Skipping {fig_fn.__name__} (missing {required} results)")

    print("\nDone! Figures saved to", FIGURES_DIR)


if __name__ == '__main__':
    main()
