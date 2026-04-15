"""
Generate publication-quality figures from experimental results.
Updated to handle hybrid experiments with position variants (early_clean, middle, early_noisy).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import os

WORKSPACE = Path(__file__).parent.parent
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
    'figure.dpi': 300,
})

COLORS = {
    'cfg': '#2196F3', 'csg': '#FF9800', 'csg_h': '#4CAF50',
    'csg_pl': '#F44336', 'esg': '#9C27B0', 'no_guidance': '#607D8B',
}


def load_results():
    with open(WORKSPACE / 'results.json') as f:
        return json.load(f)


def load_all_metrics():
    metrics = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        mf = d / 'metrics.json'
        if mf.exists():
            with open(mf) as f:
                metrics[d.name] = json.load(f)
    return metrics


def figure1_main_comparison(results):
    """FID comparison: methods x guidance scales."""
    summary = results['summary']
    methods = ['no_guidance', 'esg', 'csg', 'cfg']
    method_labels = ['No Guidance', 'ESG', 'CSG (Ours)', 'CFG (2-pass)']
    scales = [1.5, 4.0, 7.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Low scale (w=1.5) - zoomed in
    bar_width = 0.2
    x = np.arange(1)
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        key = 'no_guidance' if method == 'no_guidance' else f'{method}_w1.5'
        stats = summary.get(key, {})
        fid = stats.get('fid_mean', float('nan'))
        err = stats.get('fid_std', 0)
        if np.isnan(fid):
            continue
        color_key = method.split('_')[0] if method != 'no_guidance' else 'no_guidance'
        ax1.bar(i * bar_width, fid, bar_width, yerr=err, label=label,
                color=COLORS.get(color_key, '#333'), capsize=3, edgecolor='white')

    ax1.set_ylabel('FID-2K (lower is better)')
    ax1.set_title('w = 1.5 (Low Guidance)')
    ax1.set_xticks([i * bar_width for i in range(4)])
    ax1.set_xticklabels(['No\nGuid.', 'ESG', 'CSG', 'CFG'], fontsize=9)
    ax1.set_ylim(0, 50)

    # Panel 2: All scales - shows CSG failure
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        fids = []
        for w in scales:
            key = 'no_guidance' if method == 'no_guidance' else f'{method}_w{w}'
            stats = summary.get(key, {})
            fids.append(stats.get('fid_mean', float('nan')))
        color_key = method.split('_')[0] if method != 'no_guidance' else 'no_guidance'
        ax2.plot(scales, fids, 'o-', color=COLORS.get(color_key, '#333'),
                 label=label, linewidth=2, markersize=8)

    ax2.set_xlabel('Guidance Scale (w)')
    ax2.set_ylabel('FID-2K (lower is better)')
    ax2.set_title('FID vs Guidance Scale')
    ax2.legend()
    ax2.set_yscale('log')
    ax2.set_ylim(10, 500)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure1_main_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure1_main_comparison.png', bbox_inches='tight')
    plt.close()
    print("  Figure 1: Main comparison saved")


def figure2_linearity_heatmap(results):
    """Linearity analysis: error heatmap + error vs scale curves."""
    lin_file = RESULTS_DIR / 'linearity_analysis' / 'linearity_results.json'
    if not lin_file.exists():
        print("  Figure 2: Skipped (no linearity data)")
        return

    with open(lin_file) as f:
        lin_data = json.load(f)

    scales = sorted(set(v['guidance_scale'] for v in lin_data.values()))
    steps = sorted(set(v['step_idx'] for v in lin_data.values()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Error heatmap
    error_matrix = np.full((len(scales), len(steps)), np.nan)
    for key, val in lin_data.items():
        si = steps.index(val['step_idx'])
        wi = scales.index(val['guidance_scale'])
        error_matrix[wi, si] = val['mean_relative_error'] * 100  # percent

    im = axes[0].imshow(error_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[0].set_xlabel('Step Index (0=clean, 49=noisy)')
    axes[0].set_ylabel('Guidance Scale')
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(steps, fontsize=8)
    axes[0].set_yticks(range(len(scales)))
    axes[0].set_yticklabels([f'{s:.1f}' for s in scales])
    axes[0].set_title('CSG Approximation Error (%)')
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label('Relative Error (%)')

    # Panel 2: Error vs step index for selected scales
    for w in [1.5, 3.0, 4.0, 7.5]:
        errors = []
        step_list = []
        for key, val in sorted(lin_data.items()):
            if val['guidance_scale'] == w:
                errors.append(val['mean_relative_error'] * 100)
                step_list.append(val['step_idx'])
        if errors:
            axes[1].plot(step_list, errors, 'o-', label=f'w={w}', linewidth=2, markersize=5)

    axes[1].set_xlabel('Step Index (0=clean, 49=noisy)')
    axes[1].set_ylabel('Relative Error (%)')
    axes[1].set_title('CSG Error by Denoising Step')
    axes[1].legend()
    axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure2_linearity.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure2_linearity.png', bbox_inches='tight')
    plt.close()
    print("  Figure 2: Linearity heatmap saved")


def figure3_hybrid_tradeoff(results):
    """Hybrid CSG quality-speed tradeoff with position comparison."""
    summary = results['summary']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Compare hybrid positions at fixed ratio
    positions = ['early_clean', 'middle', 'early_noisy']
    pos_labels = ['Clean End\n(high error)', 'Middle', 'Noisy End\n(low error)']
    pos_colors = ['#4CAF50', '#FF9800', '#2196F3']

    for ratio in [10, 20, 30, 50]:
        fids = []
        for pos in positions:
            key = f'csg_h_{ratio}pct_{pos}'
            stats = summary.get(key, {})
            fids.append(stats.get('fid_mean', float('nan')))
        if not all(np.isnan(f) for f in fids):
            ax1.plot(range(len(positions)), fids, 'o-', label=f'{ratio}% CFG steps',
                     linewidth=2, markersize=8)

    cfg_fid = summary.get('cfg_w4.0', {}).get('fid_mean', float('nan'))
    if not np.isnan(cfg_fid):
        ax1.axhline(y=cfg_fid, color='blue', linestyle='--', alpha=0.5, label=f'Full CFG (FID={cfg_fid:.1f})')

    ax1.set_xticks(range(len(positions)))
    ax1.set_xticklabels(pos_labels)
    ax1.set_ylabel('FID-2K')
    ax1.set_title('Effect of CFG Step Position')
    ax1.legend(fontsize=9)

    # Panel 2: Speedup vs FID for best position (early_clean)
    points = []
    # Pure CSG
    csg_stats = summary.get('csg_w4.0', {})
    if not np.isnan(csg_stats.get('fid_mean', float('nan'))):
        points.append(('CSG (0%)', csg_stats.get('speedup_vs_cfg', 2.0),
                       csg_stats['fid_mean'], csg_stats.get('fid_std', 0), 'o', COLORS['csg']))

    # Hybrids
    for ratio in [10, 20, 30, 50]:
        key = f'csg_h_{ratio}pct_early_clean'
        stats = summary.get(key, {})
        if not np.isnan(stats.get('fid_mean', float('nan'))):
            speedup = stats.get('speedup_vs_cfg', float('nan'))
            points.append((f'CSG-H {ratio}%', speedup, stats['fid_mean'],
                          stats.get('fid_std', 0), 'D', COLORS['csg_h']))

    # Full CFG
    if not np.isnan(cfg_fid):
        points.append(('CFG', 1.0, cfg_fid,
                       summary.get('cfg_w4.0', {}).get('fid_std', 0), '*', COLORS['cfg']))

    for label, speedup, fid, err, marker, color in points:
        if np.isnan(speedup) or np.isnan(fid):
            continue
        ax2.errorbar(speedup, fid, yerr=err, fmt=marker, color=color,
                     markersize=12, capsize=4, label=label, markeredgecolor='white')
        ax2.annotate(label, (speedup, fid), textcoords="offset points",
                     xytext=(5, 10), fontsize=8)

    ax2.set_xlabel('Speedup vs CFG')
    ax2.set_ylabel('FID-2K (lower is better)')
    ax2.set_title('Hybrid CSG: Quality-Speed Tradeoff (early_clean)')
    ax2.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure3_hybrid.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure3_hybrid.png', bbox_inches='tight')
    plt.close()
    print("  Figure 3: Hybrid tradeoff saved")


def figure4_perlayer(results):
    """Per-layer guidance schedule comparison."""
    summary = results['summary']
    schedules = ['uniform', 'decreasing', 'increasing', 'bell']
    schedule_labels = ['Uniform', 'Decreasing', 'Increasing', 'Bell']

    fids = []
    for s in schedules:
        key = f'csg_pl_{s}'
        stats = summary.get(key, {})
        fids.append(stats.get('fid_mean', float('nan')))

    # Skip if no data
    if all(np.isnan(f) for f in fids):
        print("  Figure 4: Skipped (no per-layer data)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ['#FF9800', '#4CAF50', '#2196F3', '#F44336']
    valid = [(l, f, c) for l, f, c in zip(schedule_labels, fids, colors) if not np.isnan(f)]
    if valid:
        labels, vals, cols = zip(*valid)
        ax.bar(labels, vals, color=cols, edgecolor='white', linewidth=0.5)

    cfg_stats = summary.get('cfg_w4.0', {})
    if not np.isnan(cfg_stats.get('fid_mean', float('nan'))):
        ax.axhline(y=cfg_stats['fid_mean'], color=COLORS['cfg'], linestyle='--',
                    linewidth=2, label=f"CFG (FID={cfg_stats['fid_mean']:.1f})")
        ax.legend()

    ax.set_ylabel('FID-2K')
    ax.set_title('Per-Layer Guidance Schedule Ablation (w_mean=4.0)')
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure4_perlayer.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure4_perlayer.png', bbox_inches='tight')
    plt.close()
    print("  Figure 4: Per-layer ablation saved")


def figure5_error_vs_scale():
    """Error vs guidance scale at different timesteps."""
    lin_file = RESULTS_DIR / 'linearity_analysis' / 'linearity_results.json'
    if not lin_file.exists():
        print("  Figure 5: Skipped")
        return

    with open(lin_file) as f:
        lin_data = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Average error across all steps for each scale
    from collections import defaultdict
    scale_errors = defaultdict(list)
    for val in lin_data.values():
        scale_errors[val['guidance_scale']].append(val['mean_relative_error'])

    scales = sorted(scale_errors.keys())
    mean_errors = [np.mean(scale_errors[w]) * 100 for w in scales]
    max_errors = [np.max(scale_errors[w]) * 100 for w in scales]

    ax.plot(scales, mean_errors, 'o-', color='#FF9800', linewidth=2, markersize=8, label='Mean error')
    ax.plot(scales, max_errors, 's--', color='#F44336', linewidth=2, markersize=8, label='Max error')
    ax.axhline(y=10, color='green', linestyle=':', alpha=0.7, label='10% threshold')
    ax.axhline(y=25, color='red', linestyle=':', alpha=0.7, label='25% threshold')

    ax.fill_between(scales, 0, 10, alpha=0.1, color='green')
    ax.fill_between(scales, 10, 25, alpha=0.1, color='yellow')
    ax.fill_between(scales, 25, max(max_errors) * 1.1, alpha=0.1, color='red')

    ax.set_xlabel('Guidance Scale (w)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('CSG Approximation Error vs Guidance Scale')
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure5_error_vs_scale.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure5_error_vs_scale.png', bbox_inches='tight')
    plt.close()
    print("  Figure 5: Error vs scale saved")


def generate_tables(results):
    """Generate CSV and LaTeX tables."""
    summary = results['summary']

    rows = []
    for key in sorted(summary.keys()):
        s = summary[key]
        fid = s['fid_mean']
        if np.isnan(fid):
            continue
        rows.append({
            'method': key,
            'fid': fid, 'fid_std': s['fid_std'],
            'is': s['is_mean'], 'is_std': s['is_std'],
            'throughput': s['throughput_mean'],
            'speedup': s.get('speedup_vs_cfg', float('nan')),
            'seeds': s['num_seeds'],
        })

    # CSV
    csv_lines = ['Method,FID,FID_std,IS,IS_std,Throughput,Speedup,Seeds']
    for r in rows:
        sp = f"{r['speedup']:.2f}" if not np.isnan(r['speedup']) else ''
        csv_lines.append(f"{r['method']},{r['fid']:.2f},{r['fid_std']:.2f},"
                         f"{r['is']:.1f},{r['is_std']:.1f},{r['throughput']:.2f},{sp},{r['seeds']}")

    with open(FIGURES_DIR / 'table1_main_results.csv', 'w') as f:
        f.write('\n'.join(csv_lines))

    # LaTeX
    tex_lines = [
        r'\begin{table}[t]',
        r'\caption{Main results on ImageNet 256$\times$256 with DiT-XL/2. FID-2K computed over 2000 generated images.}',
        r'\label{tab:main}',
        r'\centering\small',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Method & FID-2K$\downarrow$ & IS$\uparrow$ & Throughput & Speedup \\',
        r'\midrule',
    ]

    for r in rows:
        fid_tex = f"${r['fid']:.1f}" + (f" \\pm {r['fid_std']:.1f}$" if r['fid_std'] > 0 else "$")
        is_tex = f"{r['is']:.1f}"
        tp_tex = f"{r['throughput']:.2f}"
        sp_tex = f"{r['speedup']:.2f}$\\times$" if not np.isnan(r['speedup']) else "-"
        tex_lines.append(f"{r['method']} & {fid_tex} & {is_tex} & {tp_tex} & {sp_tex} \\\\")

    tex_lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])

    with open(FIGURES_DIR / 'table1_main_results.tex', 'w') as f:
        f.write('\n'.join(tex_lines))

    print("  Tables saved")


def main():
    print("Generating figures and tables...")
    results = load_results()

    figure1_main_comparison(results)
    figure2_linearity_heatmap(results)
    figure3_hybrid_tradeoff(results)
    figure4_perlayer(results)
    figure5_error_vs_scale()
    generate_tables(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
