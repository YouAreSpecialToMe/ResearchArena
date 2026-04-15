"""Generate publication-quality figures from experiment results."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})
COLORS = sns.color_palette("tab10")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved {name}")


def figure1_coverage():
    """Coverage validation: CI bounds over time for all methods."""
    print("Generating Figure 1: Coverage validation...")
    data = load_json('experiment1_coverage/results.json')

    distributions = ['gaussian', 'exponential', 'pareto', 'student_t']
    dist_labels = {'gaussian': 'Gaussian(0,1)', 'exponential': 'Exponential(1)',
                   'pareto': 'Pareto(2)', 'student_t': 'Student-t(3)'}
    quantiles = [0.5, 0.95]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)

    for col, dist in enumerate(distributions):
        for row, p in enumerate(quantiles):
            ax = axes[row, col]
            key = f"{dist}_p{p}"
            if key not in data:
                continue

            entry = data[key]
            checkpoints = entry['checkpoints']
            true_q = entry['true_quantile']

            for method_name, color, label in [
                ('adaquantcs', COLORS[0], 'AdaQuantCS'),
                ('fullmemory', COLORS[1], 'Full Memory'),
                ('bootstrap', COLORS[2], 'Bootstrap'),
            ]:
                mdata = entry['methods'][method_name]
                widths_mean = np.array(mdata['ci_width_mean'])
                widths_std = np.array(mdata['ci_width_std'])

                ax.plot(checkpoints, widths_mean, '-o', color=color, label=label,
                        markersize=3, linewidth=1.5)
                ax.fill_between(checkpoints, widths_mean - widths_std,
                                widths_mean + widths_std, alpha=0.15, color=color)

            ax.set_xscale('log')
            ax.set_yscale('log')
            if row == 1:
                ax.set_xlabel('Stream length $t$')
            if col == 0:
                ax.set_ylabel(f'CI Width ($p={p}$)')
            ax.set_title(f'{dist_labels[dist]}')

            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Figure 1: CI Width Convergence Across Distributions', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure1_coverage')


def figure2_memory_accuracy():
    """Memory-accuracy tradeoff."""
    print("Generating Figure 2: Memory-accuracy tradeoff...")
    data = load_json('experiment2_memory_accuracy/results.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CI width vs k at final checkpoint (exclude k=5 which is degenerate)
    k_values = [k for k in data['k_values'] if k >= 10]
    final_widths = []
    final_widths_std = []
    for k in k_values:
        key = f'k={k}'
        w = data['adaquantcs'][key]['ci_width_mean'][-1]
        s = data['adaquantcs'][key]['ci_width_std'][-1]
        final_widths.append(w)
        final_widths_std.append(s)

    fm_width = data['fullmemory']['ci_width_mean'][-1]

    ax1.errorbar(k_values, final_widths, yerr=final_widths_std, fmt='-o',
                 color=COLORS[0], label='AdaQuantCS', capsize=3)
    ax1.axhline(fm_width, color=COLORS[1], linestyle='--', label='Full Memory')
    ax1.set_xlabel('Grid size $k$')
    ax1.set_ylabel('CI Width at $t=10^6$')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.set_title('(a) CI Width vs Grid Size')

    # Right: CI width vs time for selected k values
    checkpoints = data['checkpoints']
    for k, color_idx in [(10, 3), (50, 0), (200, 4)]:
        key = f'k={k}'
        widths = data['adaquantcs'][key]['ci_width_mean']
        ax2.plot(checkpoints, widths, '-o', color=COLORS[color_idx],
                 label=f'AdaQuantCS $k$={k}', markersize=3)

    ax2.plot(checkpoints, data['fullmemory']['ci_width_mean'], '--',
             color=COLORS[1], label='Full Memory', linewidth=2)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Stream length $t$')
    ax2.set_ylabel('CI Width')
    ax2.legend(fontsize=9)
    ax2.set_title('(b) CI Width Convergence')

    fig.suptitle('Figure 2: Memory-Accuracy Tradeoff', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure2_memory_accuracy')


def figure3_convergence():
    """Convergence rate analysis."""
    print("Generating Figure 3: Convergence rate...")
    data = load_json('convergence_rate/results.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    checkpoints = np.array(data['checkpoints'], dtype=float)
    aq_mean = np.array(data['adaquantcs']['ci_width_mean'])
    aq_std = np.array(data['adaquantcs']['ci_width_std'])
    fm_mean = np.array(data['fullmemory']['ci_width_mean'])
    fm_std = np.array(data['fullmemory']['ci_width_std'])

    # Left: log-log plot
    ax1.plot(checkpoints, aq_mean, '-o', color=COLORS[0], label='AdaQuantCS', markersize=4)
    ax1.fill_between(checkpoints, aq_mean - aq_std, aq_mean + aq_std, alpha=0.15, color=COLORS[0])
    ax1.plot(checkpoints, fm_mean, '-s', color=COLORS[1], label='Full Memory', markersize=4)
    ax1.fill_between(checkpoints, fm_mean - fm_std, fm_mean + fm_std, alpha=0.15, color=COLORS[1])

    # Reference line: sqrt(log(log(t)) / t)
    ref = np.sqrt(np.log(np.log(checkpoints)) / checkpoints)
    ref_scaled = ref * (aq_mean[0] / ref[0])  # scale to match
    ax1.plot(checkpoints, ref_scaled, ':', color='gray',
             label=r'$\sqrt{\log\log t / t}$ (reference)', linewidth=1.5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Stream length $t$')
    ax1.set_ylabel('CI Width')
    ax1.legend(fontsize=9)
    ax1.set_title(f'(a) Log-log convergence\nSlope: AQ={data["convergence_exponent_aq"]:.3f}, FM={data["convergence_exponent_fm"]:.3f}')

    # Right: normalized width
    ax2.plot(checkpoints, data['aq_normalized_width'], '-o', color=COLORS[0],
             label='AdaQuantCS', markersize=4)
    ax2.plot(checkpoints, data['fm_normalized_width'], '-s', color=COLORS[1],
             label='Full Memory', markersize=4)
    ax2.set_xscale('log')
    ax2.set_xlabel('Stream length $t$')
    ax2.set_ylabel(r'Width $\times \sqrt{t / \log\log t}$')
    ax2.legend(fontsize=9)
    ax2.set_title('(b) Normalized width (should plateau)')

    fig.suptitle('Figure 3: Convergence Rate Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure3_convergence')


def figure4_peeking():
    """Peeking penalty comparison."""
    print("Generating Figure 4: Peeking penalty...")
    data = load_json('experiment3_comparison/results.json')

    if 'peeking_analysis' not in data:
        print("  No peeking data found, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (key, title) in enumerate([
        ('gaussian_p0.5', 'Gaussian, $p=0.5$'),
        ('student_t_p0.95', 'Student-t, $p=0.95$'),
    ]):
        ax = axes[idx]
        if key not in data['peeking_analysis']:
            continue

        pdata = data['peeking_analysis'][key]
        ax.plot(pdata['peek_counts'], pdata['adaquantcs_coverage'], '-o',
                color=COLORS[0], label='AdaQuantCS', markersize=5)
        ax.plot(pdata['peek_counts'], pdata['bootstrap_coverage'], '-s',
                color=COLORS[2], label='Bootstrap', markersize=5)
        ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='Nominal 95%')
        ax.set_xscale('log')
        ax.set_xlabel('Number of peeks')
        ax.set_ylabel('Anytime coverage')
        ax.set_ylim(0.0, 1.05)
        ax.set_title(title)
        ax.legend()

    fig.suptitle('Figure 4: Peeking Penalty — Bootstrap vs AdaQuantCS', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure4_peeking')


def figure5_ablations():
    """Ablation study summary."""
    print("Generating Figure 5: Ablation studies...")
    data = load_json('experiment6_ablations/results.json')

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # A: Grid type
    ax = axes[0, 0]
    methods = ['adaptive', 'fixed']
    widths = [data['ablation_A_grid_type'][m]['final_width_mean'] for m in methods]
    stds = [data['ablation_A_grid_type'][m]['final_width_std'] for m in methods]
    bars = ax.bar(methods, widths, yerr=stds, capsize=5, color=[COLORS[0], COLORS[3]])
    ax.set_ylabel('Final CI Width')
    ax.set_title('(a) Grid Type')

    # B: Grid size (exclude k=5 which is degenerate)
    ax = axes[0, 1]
    k_values = [k for k in data['ablation_B_grid_size']['k_values'] if k >= 10]
    gauss_widths = [data['ablation_B_grid_size']['gaussian'][f'k={k}']['final_width_mean'] for k in k_values]
    gauss_stds = [data['ablation_B_grid_size']['gaussian'][f'k={k}']['final_width_std'] for k in k_values]
    t_widths = [data['ablation_B_grid_size']['student_t'][f'k={k}']['final_width_mean'] for k in k_values]

    ax.errorbar(k_values, gauss_widths, yerr=gauss_stds, fmt='-o', color=COLORS[0], label='Gaussian', capsize=3)
    ax.plot(k_values, t_widths, '-s', color=COLORS[1], label='Student-t')
    ax.set_xlabel('Grid size $k$')
    ax.set_ylabel('Final CI Width')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('(b) Grid Size Sensitivity')

    # C: CS type
    ax = axes[0, 2]
    cs_types = ['bernstein', 'hoeffding', 'wald']
    cs_widths = [data['ablation_C_cs_type'][cs]['final_width_mean'] for cs in cs_types]
    cs_stds = [data['ablation_C_cs_type'][cs]['final_width_std'] for cs in cs_types]
    cs_covs = [data['ablation_C_cs_type'][cs]['coverage'] for cs in cs_types]

    x_pos = np.arange(len(cs_types))
    bars = ax.bar(x_pos, cs_widths, yerr=cs_stds, capsize=5,
                  color=[COLORS[0], COLORS[1], COLORS[2]])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cs_types)
    ax.set_ylabel('Final CI Width')
    ax.set_title('(c) CS Type')

    # Add coverage annotations
    for i, (w, c) in enumerate(zip(cs_widths, cs_covs)):
        ax.annotate(f'cov={c:.2f}', (i, w), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # D: Epoch schedule
    ax = axes[1, 0]
    schedules = ['doubling', 'tripling', 'fixed']
    sched_widths = [data['ablation_D_epoch_schedule'][s]['final_width_mean'] for s in schedules]
    sched_stds = [data['ablation_D_epoch_schedule'][s]['final_width_std'] for s in schedules]
    sched_mems = [data['ablation_D_epoch_schedule'][s]['memory_mean'] for s in schedules]

    x_pos = np.arange(len(schedules))
    bars = ax.bar(x_pos, sched_widths, yerr=sched_stds, capsize=5,
                  color=[COLORS[0], COLORS[1], COLORS[2]])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(schedules)
    ax.set_ylabel('Final CI Width')
    ax.set_title('(d) Epoch Schedule')

    for i, (w, m) in enumerate(zip(sched_widths, sched_mems)):
        ax.annotate(f'mem={m:.0f}', (i, w), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    # E: Intersection
    ax = axes[1, 1]
    inter_methods = ['with_intersection', 'without_intersection']
    inter_labels = ['With\nIntersection', 'Without\nIntersection']
    inter_widths = [data['ablation_E_intersection'][m]['final_width_mean'] for m in inter_methods]
    inter_stds = [data['ablation_E_intersection'][m]['final_width_std'] for m in inter_methods]

    bars = ax.bar(inter_labels, inter_widths, yerr=inter_stds, capsize=5,
                  color=[COLORS[0], COLORS[3]])
    ax.set_ylabel('Final CI Width')
    ax.set_title('(e) Multi-Epoch Intersection')

    # F: Coverage summary across ablations
    ax = axes[1, 2]
    labels = ['Adaptive', 'Fixed', 'Bernstein', 'Hoeffding', 'Wald',
              'Doubling', 'Tripling', 'Fixed-Ep']
    covs = [
        data['ablation_A_grid_type']['adaptive']['coverage'],
        data['ablation_A_grid_type']['fixed']['coverage'],
        data['ablation_C_cs_type']['bernstein']['coverage'],
        data['ablation_C_cs_type']['hoeffding']['coverage'],
        data['ablation_C_cs_type']['wald']['coverage'],
        data['ablation_D_epoch_schedule']['doubling']['coverage'],
        data['ablation_D_epoch_schedule']['tripling']['coverage'],
        data['ablation_D_epoch_schedule']['fixed']['coverage'],
    ]
    colors_list = [COLORS[0], COLORS[3], COLORS[0], COLORS[1], COLORS[2],
                   COLORS[0], COLORS[1], COLORS[2]]

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, covs, color=colors_list)
    ax.axhline(0.95, color='red', linestyle='--', alpha=0.7, label='Nominal 95%')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coverage')
    ax.set_ylim(0.0, 1.1)
    ax.legend()
    ax.set_title('(f) Coverage Across Ablations')

    fig.suptitle('Figure 5: Ablation Studies', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure5_ablations')


def figure6_realworld():
    """Real-world data demos."""
    print("Generating Figure 6: Real-world demos...")
    data = load_json('experiment4_realworld/results.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Financial data (first seed)
    fin = data['financial'][0]
    checkpoints = fin['checkpoints']

    for method, lo_key, hi_key, color, label in [
        ('aq', 'aq_ci_lower', 'aq_ci_upper', COLORS[0], 'AdaQuantCS'),
        ('fm', 'fm_ci_lower', 'fm_ci_upper', COLORS[1], 'Full Memory'),
        ('bs', 'bs_ci_lower', 'bs_ci_upper', COLORS[2], 'Bootstrap'),
    ]:
        ax1.fill_between(checkpoints, fin[lo_key], fin[hi_key], alpha=0.2, color=color)
        ax1.plot(checkpoints, fin[lo_key], '-', color=color, linewidth=0.8)
        ax1.plot(checkpoints, fin[hi_key], '-', color=color, linewidth=0.8, label=label)

    ax1.set_xlabel('Days')
    ax1.set_ylabel('5th percentile (VaR)')
    ax1.set_title(f'(a) Financial Returns VaR\n'
                  f'Memory: AQ={fin["aq_memory"]}, FM={fin["fm_memory"]}, BS={fin["bs_memory"]}')
    ax1.legend(fontsize=9)

    # Latency data (first seed)
    lat = data['latency'][0]
    checkpoints = lat['checkpoints']

    for method, lo_key, hi_key, color, label in [
        ('aq', 'aq_ci_lower', 'aq_ci_upper', COLORS[0], 'AdaQuantCS'),
        ('fm', 'fm_ci_lower', 'fm_ci_upper', COLORS[1], 'Full Memory'),
        ('bs', 'bs_ci_lower', 'bs_ci_upper', COLORS[2], 'Bootstrap'),
    ]:
        ax2.fill_between(checkpoints, lat[lo_key], lat[hi_key], alpha=0.2, color=color)
        ax2.plot(checkpoints, lat[lo_key], '-', color=color, linewidth=0.8)
        ax2.plot(checkpoints, lat[hi_key], '-', color=color, linewidth=0.8, label=label)

    if 'true_quantile' in lat and not np.isnan(lat['true_quantile']):
        ax2.axhline(lat['true_quantile'], color='black', linestyle=':', label='True p99')

    ax2.set_xlabel('Observations')
    ax2.set_ylabel('p99 Latency (ms)')
    ax2.set_xscale('log')
    ax2.set_title(f'(b) Network Latency p99\n'
                  f'Memory: AQ={lat["aq_memory"]}, FM={lat["fm_memory"]}, BS={lat["bs_memory"]}')
    ax2.legend(fontsize=9)

    fig.suptitle('Figure 6: Real-World Applications', fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, 'figure6_realworld')


def main():
    os.chdir(os.path.dirname(__file__))

    try:
        figure1_coverage()
    except Exception as e:
        print(f"  Figure 1 failed: {e}")
    try:
        figure2_memory_accuracy()
    except Exception as e:
        print(f"  Figure 2 failed: {e}")
    try:
        figure3_convergence()
    except Exception as e:
        print(f"  Figure 3 failed: {e}")
    try:
        figure4_peeking()
    except Exception as e:
        print(f"  Figure 4 failed: {e}")
    try:
        figure5_ablations()
    except Exception as e:
        print(f"  Figure 5 failed: {e}")
    try:
        figure6_realworld()
    except Exception as e:
        print(f"  Figure 6 failed: {e}")

    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
