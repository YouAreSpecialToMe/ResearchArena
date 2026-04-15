#!/usr/bin/env python3
"""Generate figures for the revised AEP paper using v2 results."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / 'exp' / 'results_v2'
FIGURES_DIR = Path(__file__).parent.parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# Consistent styling
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

METHOD_COLORS = {
    'MSP': '#8B4513',
    'Energy': '#FF8C00',
    'ViM': '#4169E1',
    'KNN': '#228B22',
    'AEP': '#DC143C',
    'AEP+Fusion': '#9400D3',
}


def load_json(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def figure_1_main_ood_results():
    """Bar chart of AUROC across models and OOD datasets."""
    data = load_json('ood_results_v2_aggregated.json')

    # Get ViT models (not swin for AEP comparison)
    vit_models = ['deit_small', 'deit_base', 'vit_base']
    methods = ['MSP', 'Energy', 'ViM', 'KNN', 'AEP', 'AEP+Fusion']

    # Average across models
    ood_names = [k for k in data['vit_base'].keys() if not k.startswith('_')]

    fig, axes = plt.subplots(1, len(ood_names), figsize=(3.2 * len(ood_names), 3.5), sharey=True)
    if len(ood_names) == 1:
        axes = [axes]

    x = np.arange(len(methods))
    width = 0.65

    for ax_idx, ood_name in enumerate(ood_names):
        ax = axes[ax_idx]
        # Average AUROC across models
        means = []
        stds = []
        for method in methods:
            model_vals = []
            for model_key in vit_models:
                if ood_name in data[model_key] and method in data[model_key][ood_name]:
                    model_vals.append(data[model_key][ood_name][method]['AUROC']['mean'])
            means.append(np.mean(model_vals) * 100 if model_vals else 0)
            stds.append(np.std(model_vals) * 100 if model_vals else 0)

        colors = [METHOD_COLORS[m] for m in methods]
        bars = ax.bar(x, means, width, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_title(ood_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
        ax.set_ylim([max(0, min(means) - 15), 102])
        ax.axhline(y=100, color='grey', linestyle=':', linewidth=0.5, alpha=0.5)

        if ax_idx == 0:
            ax.set_ylabel('AUROC (%)')

    plt.suptitle('OOD Detection Performance (Average over 3 ViT Models)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_1_ood_results_v2.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_1_ood_results_v2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 1: Main OOD results saved")


def figure_2_fpr95():
    """FPR@95 comparison bar chart."""
    data = load_json('ood_results_v2_aggregated.json')
    vit_models = ['deit_small', 'deit_base', 'vit_base']
    methods = ['ViM', 'KNN', 'AEP', 'AEP+Fusion']
    ood_names = [k for k in data['vit_base'].keys() if not k.startswith('_')]

    fig, axes = plt.subplots(1, len(ood_names), figsize=(3.2 * len(ood_names), 3.5), sharey=True)
    if len(ood_names) == 1:
        axes = [axes]

    x = np.arange(len(methods))
    width = 0.65

    for ax_idx, ood_name in enumerate(ood_names):
        ax = axes[ax_idx]
        means = []
        for method in methods:
            model_vals = []
            for model_key in vit_models:
                if ood_name in data[model_key] and method in data[model_key][ood_name]:
                    model_vals.append(data[model_key][ood_name][method]['FPR95']['mean'])
            means.append(np.mean(model_vals) * 100 if model_vals else 0)

        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(x, means, width, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(ood_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=7)
        if ax_idx == 0:
            ax.set_ylabel('FPR@95%TPR (%)')

    plt.suptitle('FPR@95%TPR (Average over 3 ViT Models, lower is better)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_2_fpr95_v2.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_2_fpr95_v2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 2: FPR95 results saved")


def figure_3_calibration():
    """Calibration results under corruption."""
    try:
        data = load_json('calibration_results_v2.json')
    except FileNotFoundError:
        print("Calibration results not found, skipping figure 3")
        return

    model_key = list(data.keys())[0]
    cal_data = data[model_key]

    # Group by corruption type
    corruptions = ['gaussian_noise', 'gaussian_blur', 'shot_noise', 'brightness', 'contrast']
    severities = [1, 3, 5]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ECE plot
    ax = axes[0]
    for method, color, marker in [('Raw', '#888888', 'o'), ('TempScaling', '#4169E1', 's'), ('AEP_Adaptive', '#DC143C', '^')]:
        ece_by_severity = {s: [] for s in severities}
        for corruption in corruptions:
            for severity in severities:
                key = f'{corruption}_s{severity}'
                if key in cal_data:
                    ece_by_severity[severity].append(cal_data[key][method]['ECE'])

        x = severities
        y = [np.mean(ece_by_severity[s]) for s in severities]
        yerr = [np.std(ece_by_severity[s]) for s in severities]
        label = method.replace('_', ' ')
        ax.errorbar(x, y, yerr=yerr, marker=marker, color=color, label=label, linewidth=2, markersize=8, capsize=3)

    # Add clean baseline
    if 'clean' in cal_data:
        for method, color in [('Raw', '#888888'), ('TempScaling', '#4169E1'), ('AEP_Adaptive', '#DC143C')]:
            ax.axhline(y=cal_data['clean'][method]['ECE'], color=color, linestyle=':', alpha=0.3)

    ax.set_xlabel('Corruption Severity')
    ax.set_ylabel('ECE (lower is better)')
    ax.set_title('Calibration Error under Distribution Shift')
    ax.legend()
    ax.set_xticks(severities)

    # Accuracy plot
    ax = axes[1]
    acc_by_severity = {s: [] for s in severities}
    for corruption in corruptions:
        for severity in severities:
            key = f'{corruption}_s{severity}'
            if key in cal_data:
                acc_by_severity[severity].append(cal_data[key]['accuracy'])

    x = severities
    y = [np.mean(acc_by_severity[s]) * 100 for s in severities]
    yerr = [np.std(acc_by_severity[s]) * 100 for s in severities]
    ax.errorbar(x, y, yerr=yerr, marker='o', color='black', linewidth=2, markersize=8, capsize=3)
    if 'clean' in cal_data:
        ax.axhline(y=cal_data['clean']['accuracy'] * 100, color='green', linestyle='--', label='Clean', alpha=0.7)
    ax.set_xlabel('Corruption Severity')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Model Accuracy under Corruption')
    ax.legend()
    ax.set_xticks(severities)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_3_calibration_v2.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_3_calibration_v2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 3: Calibration results saved")


def figure_4_ablation_components():
    """Component ablation heatmap."""
    try:
        data = load_json('ablation_components_v2.json')
    except FileNotFoundError:
        print("Ablation results not found, skipping figure 4")
        return

    configs = list(data.keys())
    ood_names = list(data[configs[0]].keys())

    # Build matrix
    matrix = np.zeros((len(configs), len(ood_names)))
    for i, config in enumerate(configs):
        for j, ood_name in enumerate(ood_names):
            matrix[i, j] = data[config][ood_name]['AUROC'] * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=matrix.min() - 2, vmax=matrix.max() + 1)

    ax.set_xticks(range(len(ood_names)))
    ax.set_xticklabels(ood_names, rotation=45, ha='right')
    ax.set_yticks(range(len(configs)))
    config_labels = [c.replace('_', ' ') for c in configs]
    ax.set_yticklabels(config_labels)

    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(ood_names)):
            text = f'{matrix[i, j]:.1f}'
            color = 'white' if matrix[i, j] < matrix.mean() else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, label='AUROC (%)')
    ax.set_title('Component Ablation (ViT-Base/16)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_4_ablation_v2.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_4_ablation_v2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 4: Ablation results saved")


def figure_5_swin_comparison():
    """Compare Swin-Tiny with ViT models (baselines only)."""
    data = load_json('ood_results_v2_aggregated.json')

    if 'swin_tiny' not in data:
        print("No Swin results, skipping figure 5")
        return

    models = ['deit_small', 'deit_base', 'vit_base', 'swin_tiny']
    methods = ['MSP', 'Energy', 'ViM', 'KNN']
    ood_names = [k for k in data['vit_base'].keys() if not k.startswith('_')]

    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(models))
    width = 0.18

    for m_idx, method in enumerate(methods):
        means = []
        for model_key in models:
            model_vals = []
            for ood_name in ood_names:
                if ood_name in data[model_key] and method in data[model_key][ood_name]:
                    model_vals.append(data[model_key][ood_name][method]['AUROC']['mean'])
            means.append(np.mean(model_vals) * 100 if model_vals else 0)

        offset = (m_idx - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, means, width, color=METHOD_COLORS[method], label=method, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    model_labels = ['DeiT-S', 'DeiT-B', 'ViT-B/16', 'Swin-T']
    ax.set_xticklabels(model_labels)
    ax.set_ylabel('Avg AUROC (%)')
    ax.set_title('Baseline OOD Detection Across Architectures', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim([50, 102])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_5_swin_comparison_v2.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_5_swin_comparison_v2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 5: Swin comparison saved")


def figure_6_overhead():
    """Computational overhead bar chart."""
    try:
        data = load_json('overhead_v2.json')
    except FileNotFoundError:
        print("Overhead results not found, skipping figure 6")
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))

    models = [k for k in data.keys() if 'swin' not in k]
    x = np.arange(len(models))
    width = 0.35

    standard = [data[m]['standard_ms_per_img'] for m in models]
    aep = [data[m]['aep_ms_per_img'] for m in models]

    ax.bar(x - width/2, standard, width, label='Standard', color='#4169E1', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, aep, width, label='With AEP', color='#DC143C', edgecolor='black', linewidth=0.5)

    for i, m in enumerate(models):
        mult = data[m]['multiplier']
        ax.annotate(f'{mult:.1f}×', (x[i] + width/2, aep[i]), textcoords="offset points",
                    xytext=(0, 5), ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    model_labels = ['DeiT-S', 'DeiT-B', 'ViT-B/16']
    ax.set_xticklabels(model_labels)
    ax.set_ylabel('ms / image')
    ax.set_title('Computational Overhead', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure_6_overhead_v2.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure_6_overhead_v2.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 6: Overhead saved")


if __name__ == '__main__':
    print("Generating figures from v2 results...")
    figure_1_main_ood_results()
    figure_2_fpr95()
    figure_3_calibration()
    figure_4_ablation_components()
    figure_5_swin_comparison()
    figure_6_overhead()
    print("All figures generated!")
