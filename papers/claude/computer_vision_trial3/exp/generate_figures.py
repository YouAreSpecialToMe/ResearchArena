"""Generate all figures for the STG paper."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shared'))
os.environ['TMPDIR'] = '/var/tmp'
os.environ['MPLCONFIGDIR'] = '/var/tmp/mplconfig'
os.makedirs('/var/tmp/mplconfig', exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGURES_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

from data_loader import (CORRUPTION_TYPES, CATEGORY_MAP, NOISE, BLUR, WEATHER, DIGITAL,
                         ALEXNET_ERR, REPRESENTATIVE_CORRUPTIONS, compute_mce)

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
PALETTE = sns.color_palette('colorblind')


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if os.path.exists(path):
        return json.load(open(path))
    return None


def fig1_spectral_signatures():
    """Spectral signature visualization."""
    data = load_json('spectral_analysis.json')
    if not data:
        print("Skipping fig1: no spectral analysis data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: Spectral energy ratios for clean vs representative corruptions
    ax = axes[0]
    layer = '6'  # middle layer
    bands = ['Low', 'Mid', 'High']
    x = np.arange(len(bands))
    width = 0.12
    conditions = ['clean'] + ['gaussian_noise', 'defocus_blur', 'fog', 'jpeg_compression']
    labels = ['Clean', 'Gauss. noise', 'Defocus blur', 'Fog', 'JPEG comp.']
    colors = [PALETTE[0]] + [PALETTE[i+1] for i in range(4)]

    for i, (cond, label, color) in enumerate(zip(conditions, labels, colors)):
        if cond == 'clean':
            means = data['clean'][layer]['mean']
        else:
            means = data['corruptions'][cond][layer]['mean']
        ax.bar(x + (i - 2) * width, means, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Energy Ratio')
    ax.set_title('(a) Spectral Energy by Corruption Type')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 0.55)

    # Panel B: Separability across corruption categories
    ax = axes[1]
    categories = {'noise': NOISE, 'blur': BLUR, 'weather': WEATHER, 'digital': DIGITAL}
    cat_labels = ['Noise', 'Blur', 'Weather', 'Digital']
    layers = ['3', '6', '9', '11']
    layer_labels = ['Layer 3', 'Layer 6', 'Layer 9', 'Layer 11']

    sep_data = np.zeros((4, 4))  # categories × layers
    for ci, (cat, corrs) in enumerate(categories.items()):
        for li, l in enumerate(layers):
            seps = []
            for c in corrs:
                if c in data['corruptions'] and l in data['corruptions'][c]:
                    seps.append(data['corruptions'][c][l]['separability_at_95pct'])
            sep_data[ci, li] = np.mean(seps) if seps else 0

    im = ax.imshow(sep_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(4))
    ax.set_yticklabels(cat_labels)
    ax.set_title('(b) Separability (fraction > 95th %ile)')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{sep_data[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if sep_data[i,j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'spectral_signatures.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'spectral_signatures.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Generated: spectral_signatures.pdf")


def _compute_mce(result):
    """Compute mCE from sev5_accs using AlexNet reference errors."""
    if not result or 'sev5_accs' not in result:
        return None
    model_errors = {c: 1.0 - acc for c, acc in result['sev5_accs'].items()}
    return compute_mce(model_errors)


def fig2_main_results():
    """Main results: mCE comparison bar chart."""
    models = ['deit_small', 'deit_base', 'swin_tiny']
    model_labels = ['DeiT-S', 'DeiT-B', 'Swin-T']

    vanilla_mces = []
    tent_mces = []
    stg_mces = []
    stg_stds = []

    for mk in models:
        v = load_json(f'vanilla_{mk}.json')
        t = load_json(f'tent_{mk}.json')
        vanilla_mces.append(_compute_mce(v) * 100 if _compute_mce(v) is not None else 0)
        tent_mces.append(_compute_mce(t) * 100 if _compute_mce(t) is not None else 0)

        # STG: average across seeds if available
        seed_mces = []
        for seed in [42, 123, 456]:
            s = load_json(f'stg_{mk}_seed{seed}.json')
            mce = _compute_mce(s)
            if mce is not None:
                seed_mces.append(mce * 100)
        if seed_mces:
            stg_mces.append(np.mean(seed_mces))
            stg_stds.append(np.std(seed_mces) if len(seed_mces) > 1 else 0)
        else:
            stg_mces.append(0)
            stg_stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    w = 0.25

    bars1 = ax.bar(x - w, vanilla_mces, w, label='Vanilla', color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x, tent_mces, w, label='Tent', color=PALETTE[1], alpha=0.85)
    bars3 = ax.bar(x + w, stg_mces, w, yerr=stg_stds, label='STG (Ours)',
                   color=PALETTE[2], alpha=0.85, capsize=3)

    ax.set_ylabel('mCE (%)')
    ax.set_title('Mean Corruption Error on ImageNet-C')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.set_ylim(0, max(vanilla_mces + tent_mces + stg_mces) * 1.15)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'main_results.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'main_results.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Generated: main_results.pdf")


def fig3_per_corruption_heatmap():
    """Per-corruption accuracy comparison: bar chart for sev5."""
    vanilla = load_json('vanilla_deit_small.json')
    stg = load_json('stg_deit_small_seed42.json')
    if not vanilla or not stg:
        print("Skipping fig3: missing data")
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    corr_labels = [c.replace('_', '\n') for c in CORRUPTION_TYPES]
    x = np.arange(len(CORRUPTION_TYPES))
    w = 0.35

    v_accs = [vanilla['sev5_accs'].get(c, 0) * 100 for c in CORRUPTION_TYPES]
    s_accs = [stg['sev5_accs'].get(c, 0) * 100 for c in CORRUPTION_TYPES]

    ax.bar(x - w/2, v_accs, w, label='Vanilla', color=PALETTE[0], alpha=0.85)
    ax.bar(x + w/2, s_accs, w, label='STG', color=PALETTE[2], alpha=0.85)

    # Add delta annotations
    for i in range(len(CORRUPTION_TYPES)):
        delta = s_accs[i] - v_accs[i]
        y_pos = max(v_accs[i], s_accs[i]) + 1
        color = 'green' if delta > 0 else 'red'
        ax.annotate(f'{delta:+.1f}', xy=(x[i], y_pos), ha='center', fontsize=7, color=color)

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Corruption Accuracy at Severity 5 (DeiT-S)')
    ax.set_xticks(x)
    ax.set_xticklabels(corr_labels, fontsize=8)
    ax.legend()

    # Color-code x-axis labels by category
    cat_colors = {'noise': 'C0', 'blur': 'C1', 'weather': 'C2', 'digital': 'C3'}
    for i, c in enumerate(CORRUPTION_TYPES):
        cat = CATEGORY_MAP.get(c, 'other')
        ax.get_xticklabels()[i].set_color(cat_colors.get(cat, 'black'))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'per_corruption_heatmap.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'per_corruption_heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Generated: per_corruption_heatmap.pdf")


def fig4_ablations():
    """Ablation studies 2×3 subplot grid."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ablation_files = [
        ('K', 'ablation_K.json', 'Number of Frequency Bands (K)'),
        ('layers', 'ablation_layers.json', 'Layer Selection'),
        ('gating', 'ablation_gating.json', 'Gating Function'),
        ('calibration_size', 'ablation_calibration_size.json', 'Calibration Set Size'),
        ('tau', 'ablation_tau.json', 'Threshold Percentile (τ)'),
    ]

    for idx, (name, fname, title) in enumerate(ablation_files):
        ax = axes[idx // 3, idx % 3]
        data = load_json(fname)
        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        labels = [v['label'] for v in data['variants']]
        clean_accs = [v['clean_top1'] * 100 for v in data['variants']]
        corr_accs = [v['mean_corruption_acc'] * 100 for v in data['variants']]

        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w/2, clean_accs, w, label='Clean', color=PALETTE[0], alpha=0.8)
        ax.bar(x + w/2, corr_accs, w, label='Corrupted', color=PALETTE[2], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.legend(fontsize=8)

    # Hide empty subplot
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'ablations.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'ablations.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Generated: ablations.pdf")


def fig5_category_breakdown():
    """Corruption category breakdown using sev5 data."""
    models = ['deit_small', 'deit_base', 'swin_tiny']
    model_labels = ['DeiT-S', 'DeiT-B', 'Swin-T']
    categories = {'Noise': NOISE, 'Blur': BLUR, 'Weather': WEATHER, 'Digital': DIGITAL}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    w = 0.25

    for mi, (mk, ml) in enumerate(zip(models, model_labels)):
        vanilla = load_json(f'vanilla_{mk}.json')
        stg = load_json(f'stg_{mk}_seed42.json')
        if not vanilla or not stg:
            continue

        improvements = []
        for cat_name, corrs in categories.items():
            v_errs = [1 - vanilla['sev5_accs'].get(c, 0) for c in corrs]
            s_errs = [1 - stg['sev5_accs'].get(c, 0) for c in corrs]
            v_mean = np.mean(v_errs)
            s_mean = np.mean(s_errs)
            improvement = (v_mean - s_mean) / v_mean * 100 if v_mean > 0 else 0
            improvements.append(improvement)

        ax.bar(x + (mi - 1) * w, improvements, w, label=ml, color=PALETTE[mi], alpha=0.85)

    ax.set_ylabel('Relative Error Reduction (%)')
    ax.set_title('STG Impact by Corruption Category (Severity 5)')
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories.keys()))
    ax.legend()
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'category_breakdown.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'category_breakdown.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Generated: category_breakdown.pdf")


def generate_latex_tables():
    """Generate LaTeX tables for the paper."""
    models = ['deit_small', 'deit_base', 'swin_tiny']
    model_labels = ['DeiT-S', 'DeiT-B', 'Swin-T']

    # Table 1: Main results
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Main results on ImageNet-C. mCE (\%) computed relative to AlexNet. '
        r'Clean accuracy (\%) on ImageNet validation. STG results averaged over 3 seeds for DeiT-S.}',
        r'\label{tab:main}',
        r'\begin{tabular}{lccccc}',
        r'\toprule',
        r'Model & Method & Clean Acc $\uparrow$ & mCE $\downarrow$ & Rel. Improv. \\',
        r'\midrule',
    ]

    for mk, ml in zip(models, model_labels):
        vanilla = load_json(f'vanilla_{mk}.json')
        tent = load_json(f'tent_{mk}.json')
        v_mce = _compute_mce(vanilla)

        seeds_data = []
        for seed in [42, 123, 456]:
            s = load_json(f'stg_{mk}_seed{seed}.json')
            if s:
                seeds_data.append(s)

        if vanilla and v_mce is not None:
            lines.append(f'{ml} & Vanilla & {vanilla["clean"]["top1"]*100:.1f} & {v_mce*100:.1f} & -- \\\\')
        if tent:
            t_mce = _compute_mce(tent)
            if t_mce is not None and v_mce is not None:
                lines.append(f'{ml} & Tent & {tent["clean"]["top1"]*100:.1f} & {t_mce*100:.1f} & '
                            f'{(v_mce-t_mce)/v_mce*100:+.1f}\\% \\\\')
        if seeds_data:
            mces = [_compute_mce(s)*100 for s in seeds_data if _compute_mce(s) is not None]
            cleans = [s['clean']['top1']*100 for s in seeds_data]
            if mces and v_mce is not None:
                rel = (v_mce*100 - np.mean(mces)) / (v_mce*100) * 100
                if len(seeds_data) > 1:
                    lines.append(f'{ml} & STG (Ours) & {np.mean(cleans):.1f}$\\pm${np.std(cleans):.1f} & '
                               f'{np.mean(mces):.1f}$\\pm${np.std(mces):.1f} & {rel:+.1f}\\% \\\\')
                else:
                    lines.append(f'{ml} & STG (Ours) & {cleans[0]:.1f} & {mces[0]:.1f} & {rel:+.1f}\\% \\\\')
        lines.append(r'\midrule')

    lines[-1] = r'\bottomrule'
    lines.extend([r'\end{tabular}', r'\end{table}'])

    with open(os.path.join(FIGURES_DIR, 'table_main.tex'), 'w') as f:
        f.write('\n'.join(lines))
    print("  Generated: table_main.tex")

    # Table 2: Overhead
    overhead = load_json('overhead.json')
    if overhead:
        lines = [
            r'\begin{table}[t]',
            r'\centering',
            r'\caption{Computational overhead of STG.}',
            r'\label{tab:overhead}',
            r'\begin{tabular}{lcccc}',
            r'\toprule',
            r'Model & Vanilla (ms) & STG (ms) & Overhead (\%) & Mem. Overhead (MB) \\',
            r'\midrule',
        ]
        for mk, ml in zip(models, model_labels):
            if mk in overhead:
                o = overhead[mk]
                lines.append(f'{ml} & {o["vanilla_latency_ms"]["mean_ms"]:.1f} & '
                           f'{o["stg_latency_ms"]["mean_ms"]:.1f} & '
                           f'{o["overhead_pct"]:.1f} & {o["memory_overhead_MB"]:.0f} \\\\')
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        with open(os.path.join(FIGURES_DIR, 'table_overhead.tex'), 'w') as f:
            f.write('\n'.join(lines))
        print("  Generated: table_overhead.tex")


def fig_severity_scaling():
    """Accuracy vs severity from severity_analysis.json."""
    sev_data = load_json('severity_analysis.json')
    if not sev_data:
        print("Skipping severity_scaling: no severity analysis data")
        return

    corruptions = sev_data.get('corruptions', REPRESENTATIVE_CORRUPTIONS)
    fig, axes = plt.subplots(1, len(corruptions), figsize=(4*len(corruptions), 4), sharey=True)
    if len(corruptions) == 1:
        axes = [axes]

    for ax, c in zip(axes, corruptions):
        sevs = [1, 2, 3, 4, 5]
        v_accs = [sev_data['vanilla'][c][str(s)] * 100 for s in sevs]
        s_accs = [sev_data['stg'][c][str(s)] * 100 for s in sevs]

        ax.plot(sevs, v_accs, 'o-', color=PALETTE[0], label='Vanilla', linewidth=2)
        ax.plot(sevs, s_accs, 's-', color=PALETTE[2], label='STG', linewidth=2)
        ax.set_xlabel('Severity')
        ax.set_title(c.replace('_', ' ').title())
        ax.set_xticks(sevs)
        if ax == axes[0]:
            ax.set_ylabel('Accuracy (%)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'severity_scaling.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, 'severity_scaling.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Generated: severity_scaling.pdf")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_spectral_signatures()
    fig2_main_results()
    fig3_per_corruption_heatmap()
    fig4_ablations()
    fig5_category_breakdown()
    fig_severity_scaling()
    generate_latex_tables()
    print("\nAll figures generated!")
