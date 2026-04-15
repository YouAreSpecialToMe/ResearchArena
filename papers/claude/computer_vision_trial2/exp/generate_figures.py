#!/usr/bin/env python3
"""Generate all paper figures from experiment results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
WORKSPACE = Path(__file__).parent.parent
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})
sns.set_style('whitegrid')
COLORS = sns.color_palette('colorblind', 8)
METHOD_COLORS = {
    'MSP': COLORS[0], 'Energy': COLORS[1], 'ViM': COLORS[2],
    'KNN': COLORS[3], 'AEP': COLORS[4], 'AEP+Fusion': COLORS[5],
}

def save_fig(fig, name):
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(FIGURES_DIR / f'{name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  Saved {name}')


# ============ Load Results ============

with open(RESULTS_DIR / 'ood_results_aggregated.json') as f:
    ood_agg = json.load(f)
with open(RESULTS_DIR / 'ood_results.json') as f:
    ood_raw = json.load(f)
with open(RESULTS_DIR / 'calibration_results_aggregated.json') as f:
    cal_agg = json.load(f)
with open(RESULTS_DIR / 'ablation_components.json') as f:
    abl_comp = json.load(f)
with open(RESULTS_DIR / 'ablation_layers.json') as f:
    abl_layers = json.load(f)
with open(RESULTS_DIR / 'ablation_calsize.json') as f:
    abl_calsize = json.load(f)
with open(RESULTS_DIR / 'statistical_tests.json') as f:
    stat_tests = json.load(f)
with open(RESULTS_DIR / 'overhead_analysis.json') as f:
    overhead = json.load(f)


# ============ Figure 1: Entropy Profile Visualization ============

def figure1_entropy_profiles():
    """Visualize layerwise AEP statistics for ID vs OOD."""
    # Load cached AEP profiles
    cache_dir = WORKSPACE / 'exp' / 'cache'
    id_data = np.load(cache_dir / 'vit_base_id.npz')
    id_profiles = id_data['aep_profiles']

    ood_datasets = {
        'SVHN': np.load(cache_dir / 'vit_base_ood_SVHN.npz')['aep_profiles'],
        'Textures': np.load(cache_dir / 'vit_base_ood_Textures.npz')['aep_profiles'],
        'Flowers102': np.load(cache_dir / 'vit_base_ood_Flowers102.npz')['aep_profiles'],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    feature_names = ['CLS Entropy (mean)', 'Avg Token Entropy', 'Concentration Ratio']
    feature_indices = [0, 2, 3]  # indices within each layer's 5 features
    layers = np.arange(1, 13)

    for ax_idx, (feat_name, feat_idx) in enumerate(zip(feature_names, feature_indices)):
        ax = axes[ax_idx]

        # Extract feature across layers for ID
        id_vals = np.array([id_profiles[:, l*5 + feat_idx] for l in range(12)])  # (12, N)
        id_mean = id_vals.mean(axis=1)
        id_std = id_vals.std(axis=1)

        ax.plot(layers, id_mean, '-o', color=COLORS[0], label='Food101 (ID)',
                linewidth=2, markersize=4)
        ax.fill_between(layers, id_mean - id_std, id_mean + id_std,
                         alpha=0.15, color=COLORS[0])

        for i, (ood_name, ood_prof) in enumerate(ood_datasets.items()):
            ood_vals = np.array([ood_prof[:, l*5 + feat_idx] for l in range(12)])
            ood_mean = ood_vals.mean(axis=1)
            ood_std = ood_vals.std(axis=1)
            ax.plot(layers, ood_mean, '-s', color=COLORS[i+1], label=ood_name,
                    linewidth=2, markersize=4)
            ax.fill_between(layers, ood_mean - ood_std, ood_mean + ood_std,
                             alpha=0.15, color=COLORS[i+1])

        ax.set_xlabel('Layer')
        ax.set_ylabel(feat_name)
        ax.set_xticks(layers)
        if ax_idx == 0:
            ax.legend(loc='best', framealpha=0.9)

    fig.suptitle('Layerwise Attention Entropy Profiles: ID vs. OOD (ViT-Base/16)',
                  fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure_1_entropy_profiles')


# ============ Figure 2: Main OOD Detection Results ============

def figure2_ood_results():
    """Grouped bar chart of AUROC across methods and datasets."""
    methods = ['MSP', 'Energy', 'ViM', 'KNN', 'AEP', 'AEP+Fusion']
    datasets = ['Textures', 'SVHN', 'CIFAR10', 'CIFAR100', 'Flowers102']
    models = list(ood_agg.keys())

    # Average across models
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(datasets))
    width = 0.12
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    for i, method in enumerate(methods):
        aurocs = []
        stds = []
        for ds in datasets:
            vals = []
            for model in models:
                if ds in ood_agg[model] and method in ood_agg[model][ds]:
                    a = ood_agg[model][ds][method].get('AUROC', {})
                    if isinstance(a, dict):
                        vals.append(a['mean'])
            aurocs.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)

        bars = ax.bar(x + offsets[i] * width, aurocs, width, yerr=stds,
                       label=method, color=METHOD_COLORS[method],
                       capsize=2, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.55, 1.02)
    ax.legend(ncol=3, loc='lower left')
    ax.set_title('OOD Detection Performance (averaged over 3 ViT models, 3 seeds)')
    fig.tight_layout()
    save_fig(fig, 'figure_2_ood_results')


# ============ Figure 3: Per-model OOD results (FPR95) ============

def figure3_fpr95():
    """Show FPR@95 across models."""
    methods = ['MSP', 'Energy', 'ViM', 'KNN', 'AEP', 'AEP+Fusion']
    datasets = ['Textures', 'SVHN', 'CIFAR10', 'CIFAR100', 'Flowers102']
    models = list(ood_agg.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for mi, model in enumerate(models):
        ax = axes[mi]
        x = np.arange(len(datasets))
        width = 0.12
        offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

        for i, method in enumerate(methods):
            fpr95s = []
            for ds in datasets:
                f = ood_agg[model][ds][method].get('FPR95', {})
                fpr95s.append(f['mean'] if isinstance(f, dict) else 0)
            ax.bar(x + offsets[i] * width, fpr95s, width,
                   label=method if mi == 0 else '', color=METHOD_COLORS[method],
                   edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha='right')
        ax.set_title(model)
        if mi == 0:
            ax.set_ylabel('FPR@95%TPR ↓')
    axes[0].legend(ncol=2, loc='upper left', fontsize=8)
    fig.suptitle('FPR@95%TPR by Model (lower is better)', fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure_3_fpr95')


# ============ Figure 4: Calibration Results ============

def figure4_calibration():
    """Calibration comparison across shift datasets."""
    methods = ['Raw', 'TempScaling', 'HistBinning', 'AEP_Adaptive']
    method_labels = ['Raw', 'Temp Scaling', 'Hist Binning', 'AEP-Adaptive']
    shifts = ['Food101_eval', 'Textures', 'SVHN', 'CIFAR10', 'CIFAR100', 'Flowers102']
    models = list(cal_agg.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for mi, model in enumerate(models):
        ax = axes[mi]
        x = np.arange(len(shifts))
        width = 0.18
        offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

        for i, (method, label) in enumerate(zip(methods, method_labels)):
            eces = []
            for ds in shifts:
                if ds in cal_agg[model]:
                    e = cal_agg[model][ds][method].get('ECE', {})
                    eces.append(e['mean'] if isinstance(e, dict) else 0)
                else:
                    eces.append(0)
            ax.bar(x + offsets[i] * width, eces, width,
                   label=label if mi == 0 else '', color=COLORS[i],
                   edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(shifts, rotation=45, ha='right', fontsize=8)
        ax.set_title(model)
        if mi == 0:
            ax.set_ylabel('ECE ↓')
    axes[0].legend(fontsize=8)
    fig.suptitle('Expected Calibration Error across Distribution Shifts', fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure_4_calibration')


# ============ Figure 5: Component Ablation ============

def figure5_ablation_components():
    """Ablation: component importance."""
    variants = ['Full', 'No_CLS_entropy', 'No_avg_token_entropy',
                'No_concentration', 'No_head_agreement', 'CLS_entropy_only']
    labels = ['Full AEP', 'No CLS Ent.', 'No Avg Ent.', 'No Conc.', 'No Head Agr.', 'CLS Ent. Only']
    datasets = list(abl_comp['Full'].keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(datasets))
    width = 0.12
    offsets = np.arange(len(variants)) - (len(variants) - 1) / 2

    for i, (var, label) in enumerate(zip(variants, labels)):
        aurocs = [abl_comp[var][ds]['AUROC'] for ds in datasets]
        ax.bar(x + offsets[i] * width, aurocs, width, label=label, color=COLORS[i],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.7, 1.01)
    ax.legend(ncol=3, fontsize=9)
    ax.set_title('Component Ablation: AEP Feature Importance (ViT-Base/16)')
    fig.tight_layout()
    save_fig(fig, 'figure_5_component_ablation')


# ============ Figure 6: Layer Ablation ============

def figure6_layer_ablation():
    """Ablation: layer importance."""
    groups = ['All_layers', 'Early_1-4', 'Middle_5-8', 'Late_9-12']
    labels = ['All (1-12)', 'Early (1-4)', 'Middle (5-8)', 'Late (9-12)']

    datasets_avail = [ds for ds in abl_layers.get('All_layers', {}).keys()]
    if not datasets_avail:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Grouped bars
    ax = axes[0]
    x = np.arange(len(datasets_avail))
    width = 0.18
    offsets = np.arange(len(groups)) - (len(groups) - 1) / 2

    for i, (grp, label) in enumerate(zip(groups, labels)):
        aurocs = [abl_layers[grp][ds]['AUROC'] for ds in datasets_avail]
        ax.bar(x + offsets[i] * width, aurocs, width, label=label, color=COLORS[i],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets_avail, rotation=20, ha='right')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.5, 1.02)
    ax.legend(fontsize=9)
    ax.set_title('(a) Layer Group Performance')

    # Panel B: Per-layer t-statistic heatmap (for one OOD dataset)
    ax = axes[1]
    t_test = abl_layers.get('t_test', {})
    if t_test:
        ood_name = list(t_test.keys())[0]
        feature_names = ['CLS Mean', 'CLS Std', 'Avg Ent', 'Conc.', 'Head Agr.']
        heatmap_data = np.zeros((12, 5))
        for l in range(12):
            layer_key = f'layer_{l+1}'
            if layer_key in t_test[ood_name]:
                feat_keys = list(t_test[ood_name][layer_key].keys())
                for f_idx, fk in enumerate(feat_keys[:5]):
                    heatmap_data[l, f_idx] = abs(t_test[ood_name][layer_key][fk]['t_statistic'])

        im = ax.imshow(heatmap_data.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xticks(range(12))
        ax.set_xticklabels([f'{i+1}' for i in range(12)])
        ax.set_yticks(range(5))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Layer')
        ax.set_title(f'(b) |t-statistic| ID vs. {ood_name}')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Layer Importance Analysis (ViT-Base/16)', fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure_6_layer_ablation')


# ============ Figure 7: Calibration Set Size ============

def figure7_calsize():
    """AUROC vs calibration set size."""
    sizes = sorted(abl_calsize.keys(), key=lambda x: int(x))
    datasets_avail = list(abl_calsize[sizes[0]].keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets_avail):
        aurocs = [abl_calsize[sz][ds]['AUROC'] for sz in sizes]
        ax.plot([int(s) for s in sizes], aurocs, '-o', color=COLORS[i], label=ds,
                linewidth=2, markersize=5)

    ax.set_xlabel('Calibration Set Size')
    ax.set_ylabel('AUROC')
    ax.set_xscale('log')
    ax.set_ylim(0.7, 1.01)
    ax.legend()
    ax.set_title('AEP Sensitivity to Calibration Set Size (ViT-Base/16)')
    fig.tight_layout()
    save_fig(fig, 'figure_7_calsize_sensitivity')


# ============ Figure 8: Score Distributions ============

def figure8_score_distributions():
    """Histogram of AEP scores for ID vs OOD."""
    cache_dir = WORKSPACE / 'exp' / 'cache'
    id_data = np.load(cache_dir / 'vit_base_id.npz')
    from shared.aep import compute_id_statistics, compute_mahalanobis_scores

    # Use seed 42 calibration
    np.random.seed(42)
    n = len(id_data['labels'])
    cal_idx = np.random.choice(n, 1000, replace=False)
    eval_idx = np.setdiff1d(np.arange(n), cal_idx)

    stats = compute_id_statistics(id_data['aep_profiles'][cal_idx])
    id_scores = compute_mahalanobis_scores(id_data['aep_profiles'][eval_idx], stats)

    ood_datasets = ['Textures', 'SVHN', 'CIFAR10', 'CIFAR100', 'Flowers102']

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for i, ood_name in enumerate(ood_datasets):
        ax = axes[i]
        try:
            ood_data = np.load(cache_dir / f'vit_base_ood_{ood_name}.npz')
            ood_scores = compute_mahalanobis_scores(ood_data['aep_profiles'], stats)

            # Log scale for better visualization
            id_log = np.log10(id_scores + 1)
            ood_log = np.log10(ood_scores + 1)

            ax.hist(id_log, bins=50, alpha=0.5, density=True, color=COLORS[0], label='ID')
            ax.hist(ood_log, bins=50, alpha=0.5, density=True, color=COLORS[1], label=ood_name)
            ax.set_title(ood_name)
            ax.set_xlabel('log₁₀(AEP Score + 1)')
            if i == 0:
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha='center', va='center')

    fig.suptitle('AEP Score Distributions: ID (Food101) vs. OOD (ViT-Base/16)', fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure_8_score_distributions')


# ============ Table 1: Main OOD Results ============

def generate_latex_tables():
    """Generate LaTeX tables for the paper."""
    methods = ['MSP', 'Energy', 'ViM', 'KNN', 'AEP', 'AEP+Fusion']
    datasets = ['Textures', 'SVHN', 'CIFAR10', 'CIFAR100', 'Flowers102']
    models = list(ood_agg.keys())

    # Table 1: AUROC
    lines = []
    lines.append('% Table 1: OOD Detection Results (AUROC ↑)')
    lines.append('\\begin{table}[t]')
    lines.append('\\caption{OOD Detection AUROC (\\%) on Food101 as ID. Mean$\\pm$std over 3 seeds. Best in \\textbf{bold}.}')
    lines.append('\\label{tab:ood_auroc}')
    lines.append('\\centering')
    lines.append('\\scriptsize')

    for model in models:
        lines.append(f'\\begin{{tabular}}{{l{"c" * len(datasets)}}}')
        lines.append('\\toprule')
        header = f'{model.replace("_", " ").title()} & ' + ' & '.join(datasets) + ' \\\\'
        lines.append(header)
        lines.append('\\midrule')

        for method in methods:
            vals = []
            for ds in datasets:
                a = ood_agg[model][ds][method].get('AUROC', {})
                if isinstance(a, dict):
                    vals.append((a['mean'] * 100, a['std'] * 100))
                else:
                    vals.append((0, 0))

            # Find best
            best_val = max(v[0] for v in vals)
            cells = []
            for mean, std in vals:
                s = f'{mean:.1f}$\\pm${std:.1f}'
                if abs(mean - best_val) < 0.05:
                    s = f'\\textbf{{{s}}}'
                cells.append(s)

            line = f'{method} & ' + ' & '.join(cells) + ' \\\\'
            lines.append(line)

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\vspace{2mm}')
        lines.append('')

    lines.append('\\end{table}')

    with open(RESULTS_DIR / 'table1_ood_auroc.tex', 'w') as f:
        f.write('\n'.join(lines))
    print('  Saved table1_ood_auroc.tex')

    # Table 2: FPR@95
    lines = []
    lines.append('% Table 2: FPR@95%TPR (lower is better)')
    lines.append('\\begin{table}[t]')
    lines.append('\\caption{FPR@95\\%TPR (\\%) on Food101 as ID. Mean$\\pm$std over 3 seeds. Best in \\textbf{bold}.}')
    lines.append('\\label{tab:ood_fpr95}')
    lines.append('\\centering')
    lines.append('\\scriptsize')

    for model in models:
        lines.append(f'\\begin{{tabular}}{{l{"c" * len(datasets)}}}')
        lines.append('\\toprule')
        header = f'{model.replace("_", " ").title()} & ' + ' & '.join(datasets) + ' \\\\'
        lines.append(header)
        lines.append('\\midrule')

        for method in methods:
            vals = []
            for ds in datasets:
                f_val = ood_agg[model][ds][method].get('FPR95', {})
                if isinstance(f_val, dict):
                    vals.append((f_val['mean'] * 100, f_val['std'] * 100))
                else:
                    vals.append((100, 0))

            best_val = min(v[0] for v in vals)
            cells = []
            for mean, std in vals:
                s = f'{mean:.1f}$\\pm${std:.1f}'
                if abs(mean - best_val) < 0.05:
                    s = f'\\textbf{{{s}}}'
                cells.append(s)

            line = f'{method} & ' + ' & '.join(cells) + ' \\\\'
            lines.append(line)

        lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\vspace{2mm}')
        lines.append('')

    lines.append('\\end{table}')

    with open(RESULTS_DIR / 'table2_fpr95.tex', 'w') as f:
        f.write('\n'.join(lines))
    print('  Saved table2_fpr95.tex')

    # Table 3: Overhead
    lines = []
    lines.append('% Table 3: Computational Overhead')
    lines.append('\\begin{table}[t]')
    lines.append('\\caption{Computational overhead of AEP extraction.}')
    lines.append('\\label{tab:overhead}')
    lines.append('\\centering')
    lines.append('\\begin{tabular}{lccccc}')
    lines.append('\\toprule')
    lines.append('Model & Std (ms/img) & AEP (ms/img) & Overhead (\\%) & Std Mem (MB) & AEP Mem (MB) \\\\')
    lines.append('\\midrule')

    for model in overhead:
        o = overhead[model]
        lines.append(f'{model} & {o["time_standard_ms"]:.1f} & {o["time_aep_ms"]:.1f} & '
                      f'{o["overhead_pct"]:.0f}\\% & {o["mem_standard_MB"]:.0f} & {o["mem_aep_MB"]:.0f} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    with open(RESULTS_DIR / 'table3_overhead.tex', 'w') as f:
        f.write('\n'.join(lines))
    print('  Saved table3_overhead.tex')


# ============ Main ============

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    print('Generating figures...')
    figure1_entropy_profiles()
    figure2_ood_results()
    figure3_fpr95()
    figure4_calibration()
    figure5_ablation_components()
    figure6_layer_ablation()
    figure7_calsize()
    figure8_score_distributions()

    print('\nGenerating LaTeX tables...')
    generate_latex_tables()

    print(f'\nAll figures saved to {FIGURES_DIR}/')
    print(f'LaTeX tables saved to {RESULTS_DIR}/')
