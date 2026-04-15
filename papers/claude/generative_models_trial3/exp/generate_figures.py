"""
Generate publication-quality figures and tables from experimental results.
Run this after run_all.py completes.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import os

WORKSPACE = Path(__file__).parent.parent
RESULTS_DIR = WORKSPACE / 'exp' / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

COLORS = {
    'cfg': '#2196F3',       # blue
    'csg': '#FF9800',       # orange
    'csg_h': '#4CAF50',     # green
    'csg_pl': '#F44336',    # red
    'esg': '#9C27B0',       # purple
    'no_guidance': '#607D8B', # gray
}


def load_results():
    """Load aggregated results."""
    results_file = WORKSPACE / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def load_all_metrics():
    """Load individual metrics files."""
    metrics = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        mf = d / 'metrics.json'
        if mf.exists():
            with open(mf) as f:
                metrics[d.name] = json.load(f)
    return metrics


def figure1_main_comparison(results):
    """Figure 1: FID comparison across methods at different guidance scales."""
    summary = results['summary']

    methods = ['no_guidance', 'esg', 'csg', 'cfg']
    method_labels = ['No Guidance', 'ESG', 'CSG (Ours)', 'CFG (2-pass)']
    scales = [1.5, 4.0, 7.5]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    bar_width = 0.18
    x = np.arange(len(scales))

    for i, (method, label) in enumerate(zip(methods, method_labels)):
        fids = []
        errs = []
        for w in scales:
            if method == 'no_guidance':
                key = 'no_guidance'
            else:
                key = f'{method}_w{w}'
            stats = summary.get(key, {})
            fids.append(stats.get('fid_mean', float('nan')))
            errs.append(stats.get('fid_std', 0))

        color_key = method.split('_')[0] if method != 'no_guidance' else 'no_guidance'
        bars = ax.bar(x + i * bar_width, fids, bar_width, yerr=errs,
                      label=label, color=COLORS.get(color_key, '#333'),
                      capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Guidance Scale (w)')
    ax.set_ylabel('FID-5K (lower is better)')
    ax.set_title('Image Quality: CSG vs Standard CFG')
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels([f'w={s}' for s in scales])
    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure1_main_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure1_main_comparison.png', bbox_inches='tight')
    plt.close()
    print("  Figure 1: Main comparison saved")


def figure2_pareto(results):
    """Figure 2: Speed vs Quality Pareto plot."""
    summary = results['summary']

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    methods_to_plot = [
        ('no_guidance', 'No Guidance', 'no_guidance', 's'),
        ('esg_w4.0', 'ESG', 'esg', '^'),
        ('csg_w4.0', 'CSG (Ours)', 'csg', 'o'),
        ('csg_h_10pct', 'CSG-H 10%', 'csg_h', 'D'),
        ('csg_h_20pct', 'CSG-H 20%', 'csg_h', 'D'),
        ('csg_h_30pct', 'CSG-H 30%', 'csg_h', 'D'),
        ('cfg_w4.0', 'CFG (2-pass)', 'cfg', '*'),
    ]

    for key, label, color_key, marker in methods_to_plot:
        stats = summary.get(key, {})
        fid = stats.get('fid_mean', float('nan'))
        tp = stats.get('throughput_mean', float('nan'))
        if np.isnan(fid) or np.isnan(tp):
            continue

        fid_err = stats.get('fid_std', 0)
        ax.errorbar(tp, fid, yerr=fid_err, fmt=marker, color=COLORS.get(color_key, '#333'),
                     markersize=10, capsize=4, label=label, markeredgecolor='white',
                     markeredgewidth=0.5)

    ax.set_xlabel('Throughput (images/sec)')
    ax.set_ylabel('FID-5K (lower is better)')
    ax.set_title('Quality-Speed Tradeoff')
    ax.legend(loc='upper right')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure2_pareto.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure2_pareto.png', bbox_inches='tight')
    plt.close()
    print("  Figure 2: Pareto plot saved")


def figure3_linearity(results):
    """Figure 3: Linearity analysis heatmap."""
    lin_file = RESULTS_DIR / 'linearity_analysis' / 'linearity_results.json'
    if not lin_file.exists():
        print("  Figure 3: Skipped (no linearity data)")
        return

    with open(lin_file) as f:
        lin_data = json.load(f)

    # Parse data
    scales = sorted(set(v['guidance_scale'] for v in lin_data.values()))
    steps = sorted(set(v['step_idx'] for v in lin_data.values()))

    # Heatmap: relative error by step and scale
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Error heatmap (step x scale) for w=4.0
    error_matrix = np.full((len(scales), len(steps)), np.nan)
    cosine_matrix = np.full((len(scales), len(steps)), np.nan)

    for key, val in lin_data.items():
        si = steps.index(val['step_idx'])
        wi = scales.index(val['guidance_scale'])
        error_matrix[wi, si] = val['mean_relative_error']
        cosine_matrix[wi, si] = val['mean_cosine_similarity']

    im1 = axes[0].imshow(error_matrix, aspect='auto', cmap='YlOrRd',
                          interpolation='nearest')
    axes[0].set_xlabel('Denoising Step Index')
    axes[0].set_ylabel('Guidance Scale')
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(steps, fontsize=8)
    axes[0].set_yticks(range(len(scales)))
    axes[0].set_yticklabels([f'{s:.1f}' for s in scales])
    axes[0].set_title('Relative Error (CSG vs CFG)')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(cosine_matrix, aspect='auto', cmap='RdYlGn',
                          interpolation='nearest', vmin=0.9, vmax=1.0)
    axes[1].set_xlabel('Denoising Step Index')
    axes[1].set_ylabel('Guidance Scale')
    axes[1].set_xticks(range(len(steps)))
    axes[1].set_xticklabels(steps, fontsize=8)
    axes[1].set_yticks(range(len(scales)))
    axes[1].set_yticklabels([f'{s:.1f}' for s in scales])
    axes[1].set_title('Cosine Similarity (CSG vs CFG)')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure3_linearity.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure3_linearity.png', bbox_inches='tight')
    plt.close()
    print("  Figure 3: Linearity analysis saved")


def figure4_perlayer(results):
    """Figure 4: Per-layer guidance schedule comparison."""
    summary = results['summary']

    schedules = ['uniform', 'decreasing', 'increasing', 'bell']
    schedule_labels = ['Uniform', 'Decreasing', 'Increasing', 'Bell']

    fids = []
    errs = []
    for s in schedules:
        key = f'csg_pl_{s}'
        stats = summary.get(key, {})
        fids.append(stats.get('fid_mean', float('nan')))
        errs.append(stats.get('fid_std', 0))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ['#FF9800', '#4CAF50', '#2196F3', '#F44336']
    bars = ax.bar(schedule_labels, fids, yerr=errs, color=colors,
                  capsize=5, edgecolor='white', linewidth=0.5)

    # Add CFG reference line
    cfg_stats = summary.get('cfg_w4.0', {})
    if not np.isnan(cfg_stats.get('fid_mean', float('nan'))):
        ax.axhline(y=cfg_stats['fid_mean'], color=COLORS['cfg'], linestyle='--',
                    linewidth=2, label=f"CFG (FID={cfg_stats['fid_mean']:.1f})")
        ax.legend()

    ax.set_ylabel('FID-5K (lower is better)')
    ax.set_title('Per-Layer Guidance Schedule Ablation (w_mean=4.0)')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure4_perlayer.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure4_perlayer.png', bbox_inches='tight')
    plt.close()
    print("  Figure 4: Per-layer ablation saved")


def figure5_hybrid_tradeoff(results):
    """Figure 5: Hybrid CSG speed-quality tradeoff."""
    summary = results['summary']

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Plot methods
    points = [
        ('csg_w4.0', 'CSG (0% CFG)', COLORS['csg'], 'o'),
        ('csg_h_10pct', 'CSG-H 10%', COLORS['csg_h'], 'D'),
        ('csg_h_20pct', 'CSG-H 20%', COLORS['csg_h'], 's'),
        ('csg_h_30pct', 'CSG-H 30%', COLORS['csg_h'], '^'),
        ('cfg_w4.0', 'CFG (100%)', COLORS['cfg'], '*'),
    ]

    for key, label, color, marker in points:
        stats = summary.get(key, {})
        fid = stats.get('fid_mean', float('nan'))
        speedup = stats.get('speedup_vs_cfg', float('nan'))
        if np.isnan(fid) or np.isnan(speedup):
            continue
        fid_err = stats.get('fid_std', 0)
        ax.errorbar(speedup, fid, yerr=fid_err, fmt=marker, color=color,
                     markersize=12, capsize=4, label=label, markeredgecolor='white')

    ax.set_xlabel('Speedup vs Standard CFG')
    ax.set_ylabel('FID-5K (lower is better)')
    ax.set_title('Hybrid CSG: Quality-Speed Tradeoff')
    ax.legend(loc='upper left')
    ax.axhline(y=summary.get('cfg_w4.0', {}).get('fid_mean', 0), color=COLORS['cfg'],
               linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure5_hybrid.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure5_hybrid.png', bbox_inches='tight')
    plt.close()
    print("  Figure 5: Hybrid tradeoff saved")


def figure6_steps_ablation(results):
    """Figure 6: Steps ablation."""
    all_metrics = load_all_metrics()

    step_counts = [25, 50, 100]
    cfg_fids = []
    csg_fids = []

    for n in step_counts:
        cfg_key = f"steps_cfg_n{n}_seed0" if n != 50 else "cfg_w4.0_seed0"
        csg_key = f"steps_csg_n{n}_seed0" if n != 50 else "csg_w4.0_seed0"
        cfg_fids.append(all_metrics.get(cfg_key, {}).get('fid', float('nan')))
        csg_fids.append(all_metrics.get(csg_key, {}).get('fid', float('nan')))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    x = np.arange(len(step_counts))
    w = 0.3
    ax.bar(x - w/2, cfg_fids, w, label='CFG (2-pass)', color=COLORS['cfg'])
    ax.bar(x + w/2, csg_fids, w, label='CSG (1-pass)', color=COLORS['csg'])

    ax.set_xlabel('Number of DDIM Steps')
    ax.set_ylabel('FID (lower is better)')
    ax.set_title('Effect of Sampling Steps on Quality')
    ax.set_xticks(x)
    ax.set_xticklabels(step_counts)
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure6_steps_ablation.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure6_steps_ablation.png', bbox_inches='tight')
    plt.close()
    print("  Figure 6: Steps ablation saved")


def generate_tables(results):
    """Generate LaTeX tables."""
    summary = results['summary']

    # Table 1: Main results
    rows = [
        ('No Guidance', 'no_guidance'),
        ('ESG (w=1.5)', 'esg_w1.5'),
        ('ESG (w=4.0)', 'esg_w4.0'),
        ('ESG (w=7.5)', 'esg_w7.5'),
        ('CSG (w=1.5)', 'csg_w1.5'),
        ('CSG (w=4.0)', 'csg_w4.0'),
        ('CSG (w=7.5)', 'csg_w7.5'),
        ('CSG-PL best', None),
        ('CSG-H 10%', 'csg_h_10pct'),
        ('CSG-H 20%', 'csg_h_20pct'),
        ('CSG-H 30%', 'csg_h_30pct'),
        ('CFG (w=1.5)', 'cfg_w1.5'),
        ('CFG (w=4.0)', 'cfg_w4.0'),
        ('CFG (w=7.5)', 'cfg_w7.5'),
    ]

    # Find best CSG-PL
    pl_fids = {}
    for sched in ['uniform', 'decreasing', 'increasing', 'bell']:
        key = f'csg_pl_{sched}'
        if key in summary:
            pl_fids[sched] = summary[key]['fid_mean']
    best_pl = min(pl_fids, key=pl_fids.get) if pl_fids else None

    # CSV
    csv_lines = ['Method,FID,FID_std,IS,Throughput,Speedup']
    # LaTeX
    tex_lines = [
        r'\begin{table}[t]',
        r'\caption{Main results on ImageNet 256$\times$256 with DiT-XL/2.}',
        r'\label{tab:main}',
        r'\centering',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Method & FID$\downarrow$ & IS$\uparrow$ & Throughput & Speedup \\',
        r'\midrule',
    ]

    for label, key in rows:
        if key is None and best_pl:
            key = f'csg_pl_{best_pl}'
            label = f'CSG-PL ({best_pl})'
        if key is None or key not in summary:
            continue

        s = summary[key]
        fid_val = s['fid_mean']
        fid_std = s['fid_std']
        is_val = s['is_mean']
        tp = s['throughput_mean']
        speedup = s.get('speedup_vs_cfg', float('nan'))

        fid_str = f"{fid_val:.2f}" if not np.isnan(fid_val) else "-"
        fid_std_str = f"{fid_std:.2f}" if fid_std > 0 else ""
        is_str = f"{is_val:.1f}" if not np.isnan(is_val) else "-"
        tp_str = f"{tp:.2f}" if not np.isnan(tp) else "-"
        sp_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "-"

        csv_lines.append(f"{label},{fid_val},{fid_std},{is_val},{tp},{speedup}")

        fid_tex = f"${fid_val:.2f}" + (f" \\pm {fid_std:.2f}$" if fid_std > 0 else "$")
        tex_lines.append(f"{label} & {fid_tex} & {is_str} & {tp_str} & {sp_str} \\\\")

    tex_lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])

    # Save
    with open(FIGURES_DIR / 'table1_main_results.csv', 'w') as f:
        f.write('\n'.join(csv_lines))
    with open(FIGURES_DIR / 'table1_main_results.tex', 'w') as f:
        f.write('\n'.join(tex_lines))

    print("  Table 1: Main results saved")


def main():
    print("Generating figures and tables...")
    results = load_results()
    if results is None:
        print("No results.json found! Run run_all.py first.")
        return

    figure1_main_comparison(results)
    figure2_pareto(results)
    figure3_linearity(results)
    figure4_perlayer(results)
    figure5_hybrid_tradeoff(results)
    figure6_steps_ablation(results)
    generate_tables(results)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == '__main__':
    main()
