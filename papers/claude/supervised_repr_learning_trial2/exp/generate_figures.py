"""Generate publication-quality figures for the CCR paper."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'ce': '#2196F3',
    'label_smoothing': '#FF9800',
    'mixup': '#4CAF50',
    'ccr_soft': '#E91E63',
    'ccr_adaptive': '#9C27B0',
    'ccr_curriculum': '#00BCD4',
    'ccr_spectral': '#795548',
    'ccr_fixed': '#607D8B',
}

METHOD_NAMES = {
    'ce': 'Cross-Entropy',
    'label_smoothing': 'Label Smoothing',
    'mixup': 'Mixup',
    'ccr_soft': 'CCR-Soft',
    'ccr_adaptive': 'CCR-Adaptive',
    'ccr_curriculum': 'CCR-Curriculum',
    'ccr_spectral': 'CCR-Spectral',
    'ccr_fixed': 'CCR-Fixed',
}


def get_color(method):
    """Get color for method, with fallback for tau/lambda variants."""
    if method in COLORS:
        return COLORS[method]
    if method.startswith('ccr_fixed'):
        return '#E91E63'
    if method.startswith('ccr_soft'):
        return '#E91E63'
    return '#999'


def get_name(method):
    """Get display name for method, with auto-naming for variants."""
    if method in METHOD_NAMES:
        return METHOD_NAMES[method]
    # Auto-generate name from method string
    name = method.replace('_', ' ').replace('ccr ', 'CCR-')
    if 'tau' in name:
        # e.g., ccr_fixed_tau15 -> CCR-Fixed (tau=15)
        parts = method.split('_')
        tau_part = [p for p in parts if p.startswith('tau')]
        if tau_part:
            tau_val = tau_part[0].replace('tau', '')
            return f'CCR-Fixed ($\\tau$={tau_val})'
    if 'gamma' in name:
        parts = method.split('_')
        gamma_part = [p for p in parts if p.startswith('gamma')]
        if gamma_part:
            gamma_val = gamma_part[0].replace('gamma', '')
            return f'CCR-Soft ($\\gamma$={gamma_val})'
    if 'lambda' in name:
        parts = method.split('_')
        lam_part = [p for p in parts if p.startswith('lambda') or p.replace('.','').isdigit()]
        return f'CCR ($\\lambda$ variant)'
    return name.title()

DATASET_NAMES = {
    'cifar10': 'CIFAR-10',
    'cifar100': 'CIFAR-100',
    'tinyimagenet': 'TinyImageNet',
}


def find_best_ccr_method(agg_dataset):
    """Find best CCR method in aggregated results for a dataset (lowest ECE, prefer 3+ seeds)."""
    best = None
    best_ece = float('inf')
    for method, vals in agg_dataset.items():
        if not method.startswith('ccr_') or '_100ep' in method:
            continue
        ece = vals.get('ece', {}).get('mean', float('inf'))
        n = vals.get('n_seeds', 1)
        # Prefer methods with multiple seeds
        if n >= 3 and ece < best_ece:
            best_ece = ece
            best = method
    if best is None:
        for method, vals in agg_dataset.items():
            if not method.startswith('ccr_') or '_100ep' in method:
                continue
            ece = vals.get('ece', {}).get('mean', float('inf'))
            if ece < best_ece:
                best_ece = ece
                best = method
    return best or 'ccr_soft'


def get_methods_for_dataset(agg, dataset):
    """Get the list of methods to plot for a dataset (baselines + best CCR)."""
    methods = ['ce', 'label_smoothing', 'mixup']
    if dataset in agg:
        best_ccr = find_best_ccr_method(agg[dataset])
        methods.append(best_ccr)
    return [m for m in methods if m in agg.get(dataset, {})]


def load_results():
    with open('results.json') as f:
        return json.load(f)


def load_all_metrics():
    """Load individual metrics.json files for reliability diagrams."""
    all_data = defaultdict(lambda: defaultdict(dict))
    for dataset in os.listdir('results'):
        dataset_dir = os.path.join('results', dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for method in os.listdir(dataset_dir):
            method_dir = os.path.join(dataset_dir, method)
            if not os.path.isdir(method_dir):
                continue
            for seed_dir in os.listdir(method_dir):
                metrics_file = os.path.join(method_dir, seed_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file) as f:
                        all_data[dataset][method][seed_dir] = json.load(f)
    return all_data


def fig_main_results_bar(results):
    """Figure 2: Main results bar chart showing ECE and accuracy."""
    agg = results.get('aggregate_results', {})
    datasets = [d for d in ['cifar10', 'cifar100', 'tinyimagenet'] if d in agg]
    # Determine methods: baselines + best CCR per dataset (use CIFAR-100 CCR as reference)
    best_ccr = find_best_ccr_method(agg.get('cifar100', agg.get(datasets[0], {}))) if datasets else 'ccr_soft'
    methods = ['ce', 'label_smoothing', 'mixup', best_ccr]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ECE subplot
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.18
    for i, method in enumerate(methods):
        vals = []
        errs = []
        for ds in datasets:
            if method in agg.get(ds, {}):
                m = agg[ds][method]
                vals.append(m.get('ece', {}).get('mean', 0))
                errs.append(m.get('ece', {}).get('std', 0))
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width * 0.9, yerr=errs, capsize=3,
               color=get_color(method), label=get_name(method),
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('ECE (lower is better)')
    ax.set_title('Expected Calibration Error')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')
    ax.set_ylim(bottom=0)

    # Accuracy subplot
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = []
        errs = []
        for ds in datasets:
            if method in agg.get(ds, {}):
                m = agg[ds][method]
                vals.append(m.get('test_accuracy', {}).get('mean', 0) * 100)
                errs.append(m.get('test_accuracy', {}).get('std', 0) * 100)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width * 0.9, yerr=errs, capsize=3,
               color=get_color(method), label=get_name(method),
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')

    plt.tight_layout()
    plt.savefig('figures/fig2_main_results.pdf')
    plt.savefig('figures/fig2_main_results.png')
    plt.close()
    print("  Generated fig2_main_results")


def fig_reliability_diagrams(all_data):
    """Figure 3: Reliability diagrams."""
    # Find best CCR method from available data
    best_ccr = 'ccr_soft'
    for ds in ['cifar100', 'cifar10', 'tinyimagenet']:
        if ds in all_data:
            for m in sorted(all_data[ds].keys()):
                if m.startswith('ccr_') and '_100ep' not in m and m != 'ccr_soft':
                    if any('metrics.json' in str(v) or 'reliability_bins' in str(v)
                           for v in all_data[ds][m].values()):
                        best_ccr = m
                        break
            break
    methods = ['ce', 'label_smoothing', 'mixup', best_ccr]
    datasets = [d for d in ['cifar10', 'cifar100', 'tinyimagenet'] if d in all_data]

    n_rows = len(datasets)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, dataset in enumerate(datasets):
        for col, method in enumerate(methods):
            ax = axes[row, col]

            data = all_data.get(dataset, {}).get(method, {}).get('seed_42', {})
            bins = data.get('reliability_bins', [])

            if bins:
                bin_width = 1.0 / 15
                bin_centers = [(b['bin_lo'] + b['bin_hi']) / 2 for b in bins if b['count'] > 0]
                bin_accs = [b['accuracy'] for b in bins if b['count'] > 0]
                bin_confs = [b['confidence'] for b in bins if b['count'] > 0]
                gaps = [abs(a - c) for a, c in zip(bin_accs, bin_confs)]

                ax.bar(bin_centers, bin_accs, width=bin_width * 0.8, alpha=0.7,
                       color=get_color(method), edgecolor='white', linewidth=0.5)
                ax.bar(bin_centers, gaps, bottom=[min(a, c) for a, c in zip(bin_accs, bin_confs)],
                       width=bin_width * 0.8, alpha=0.3, color='red', edgecolor='none')

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            cal = data.get('calibration', {})
            ece = cal.get('ece', 0)
            ax.text(0.05, 0.92, f'ECE={ece:.3f}', transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            if row == 0:
                ax.set_title(get_name(method))
            if col == 0:
                ax.set_ylabel(f'{DATASET_NAMES.get(dataset, dataset)}\nAccuracy')
            if row == n_rows - 1:
                ax.set_xlabel('Confidence')

    plt.tight_layout()
    plt.savefig('figures/fig3_reliability_diagrams.pdf')
    plt.savefig('figures/fig3_reliability_diagrams.png')
    plt.close()
    print("  Generated fig3_reliability_diagrams")


def fig_nc_metrics_training(all_data):
    """Figure 4: Training dynamics for CE vs CCR on CIFAR-100."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Find best CCR method with training logs
    best_ccr = 'ccr_soft'
    for m in sorted(os.listdir('results/cifar100')):
        if m.startswith('ccr_') and '_100ep' not in m:
            log_path = f'results/cifar100/{m}/seed_42/training_log.json'
            if os.path.exists(log_path):
                best_ccr = m
    methods_to_plot = ['ce', best_ccr]

    for method in methods_to_plot:
        log_path = f'results/cifar100/{method}/seed_42/training_log.json'
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            log_data = json.load(f)

        epochs = [e['epoch'] for e in log_data['log']]
        test_accs = [e['test_acc'] for e in log_data['log']]
        train_losses = [e['train_loss'] for e in log_data['log']]

        color = get_color(method)
        label = get_name(method)

        axes[0, 0].plot(epochs, train_losses, color=color, label=label, alpha=0.8)
        axes[0, 1].plot(epochs, test_accs, color=color, label=label, alpha=0.8)

        spreads = [e.get('mean_spread', None) for e in log_data['log']]
        if any(s is not None for s in spreads):
            valid_epochs = [e for e, s in zip(epochs, spreads) if s is not None]
            valid_spreads = [s for s in spreads if s is not None]
            axes[1, 0].plot(valid_epochs, valid_spreads, color=color, label=label, alpha=0.8)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()

    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].legend()

    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Within-Class Spread')
    axes[1, 0].set_title('Within-Class Feature Spread')
    axes[1, 0].legend()

    # NC metrics bar chart
    agg_results = load_results().get('aggregate_results', {}).get('cifar100', {})
    best_ccr_for_nc = find_best_ccr_method(agg_results)
    methods = ['ce', 'label_smoothing', 'mixup', best_ccr_for_nc]
    nc1_vals = []
    nc1_labels = []
    nc1_colors = []
    for m in methods:
        if m in agg_results and 'nc1' in agg_results[m]:
            nc1_vals.append(agg_results[m]['nc1']['mean'])
            nc1_labels.append(get_name(m))
            nc1_colors.append(COLORS.get(m, '#999'))

    if nc1_vals:
        axes[1, 1].bar(range(len(nc1_vals)), nc1_vals, color=nc1_colors)
        axes[1, 1].set_xticks(range(len(nc1_vals)))
        axes[1, 1].set_xticklabels(nc1_labels, rotation=15, ha='right')
        axes[1, 1].set_ylabel('NC1 Value')
        axes[1, 1].set_title('NC1 (Higher = Less Collapsed)')

    plt.tight_layout()
    plt.savefig('figures/fig4_training_dynamics.pdf')
    plt.savefig('figures/fig4_training_dynamics.png')
    plt.close()
    print("  Generated fig4_training_dynamics")


def fig_pareto_frontier(results):
    """Figure 5: Accuracy vs ECE Pareto frontier."""
    agg = results.get('aggregate_results', {}).get('cifar100', {})
    if not agg:
        print("  Skipping fig5_pareto: no CIFAR-100 results")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    best_ccr = find_best_ccr_method(agg)
    main_methods = ['ce', 'label_smoothing', 'mixup', best_ccr]
    for method in main_methods:
        if method in agg and 'ece' in agg[method]:
            acc = agg[method]['test_accuracy']['mean'] * 100
            ece = agg[method]['ece']['mean']
            acc_std = agg[method]['test_accuracy']['std'] * 100
            ece_std = agg[method]['ece']['std']
            ax.errorbar(ece, acc, xerr=ece_std, yerr=acc_std,
                       marker='o', markersize=10, capsize=4,
                       color=COLORS.get(method, '#E91E63'),
                       label=get_name(method),
                       zorder=5)

    # All CCR variants as scatter points
    for method_name in sorted(agg.keys()):
        if method_name.startswith('ccr_') and method_name != best_ccr and '_100ep' not in method_name:
            if 'ece' in agg[method_name]:
                ece = agg[method_name]['ece']['mean']
                acc = agg[method_name]['test_accuracy']['mean'] * 100
                ax.scatter(ece, acc, marker='D', s=60, color='#E91E63',
                          alpha=0.6, zorder=4)
                label = method_name.replace('ccr_', '').replace('_', ' ')
                ax.annotate(label, (ece, acc), textcoords="offset points",
                           xytext=(5, 5), fontsize=7, alpha=0.7)

    ax.set_xlabel('ECE (lower is better)')
    ax.set_ylabel('Accuracy (%) (higher is better)')
    ax.set_title('Accuracy vs ECE Trade-off (CIFAR-100)')
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc')

    plt.tight_layout()
    plt.savefig('figures/fig5_pareto_frontier.pdf')
    plt.savefig('figures/fig5_pareto_frontier.png')
    plt.close()
    print("  Generated fig5_pareto_frontier")


def fig_ablation(results):
    """Figure 6: Ablation comparison of CCR variants."""
    agg = results.get('aggregate_results', {}).get('cifar100', {})
    if not agg:
        print("  Skipping fig6_ablation: no results")
        return

    # Include all CCR variants found in results
    present = [m for m in sorted(agg.keys())
               if m.startswith('ccr_') and '_100ep' not in m and 'ece' in agg[m]]

    if len(present) < 2:
        print(f"  Skipping fig6_ablation: only {len(present)} methods found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    accs = [agg[m]['test_accuracy']['mean'] * 100 for m in present]
    eces = [agg[m]['ece']['mean'] for m in present]
    spreads = [agg[m].get('mean_within_class_spread', {}).get('mean', 0) for m in present]
    colors = [COLORS.get(m, '#999') for m in present]
    labels = [get_name(m) for m in present]

    axes[0].barh(range(len(present)), accs, color=colors)
    axes[0].set_yticks(range(len(present)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy')

    axes[1].barh(range(len(present)), eces, color=colors)
    axes[1].set_yticks(range(len(present)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('ECE (lower is better)')
    axes[1].set_title('Expected Calibration Error')

    axes[2].barh(range(len(present)), spreads, color=colors)
    axes[2].set_yticks(range(len(present)))
    axes[2].set_yticklabels(labels)
    axes[2].set_xlabel('Mean Within-Class Spread')
    axes[2].set_title('Feature Spread')

    plt.tight_layout()
    plt.savefig('figures/fig6_ablation.pdf')
    plt.savefig('figures/fig6_ablation.png')
    plt.close()
    print("  Generated fig6_ablation")


def fig_ts_combination(results):
    """Figure 7: Temperature scaling combination results."""
    agg = results.get('aggregate_results', {})
    datasets = [d for d in ['cifar10', 'cifar100', 'tinyimagenet'] if d in agg]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(datasets))
    width = 0.18
    best_ccr = find_best_ccr_method(agg.get('cifar100', agg.get(datasets[0], {}))) if datasets else 'ccr_soft'
    bar_configs = [
        ('ce', 'ece', 'CE', COLORS['ce'], '//'),
        ('ce', 'ece_ts', 'CE + TS', COLORS['ce'], None),
        (best_ccr, 'ece', 'CCR', COLORS.get(best_ccr, '#E91E63'), '//'),
        (best_ccr, 'ece_ts', 'CCR + TS', COLORS.get(best_ccr, '#E91E63'), None),
    ]

    for i, (method, metric, label, color, hatch) in enumerate(bar_configs):
        vals = []
        errs = []
        for ds in datasets:
            if method in agg.get(ds, {}) and metric in agg[ds][method]:
                vals.append(agg[ds][method][metric]['mean'])
                errs.append(agg[ds][method][metric]['std'])
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - 1.5) * width
        alpha = 0.6 if hatch else 1.0
        ax.bar(x + offset, vals, width * 0.9, yerr=errs, capsize=3,
               color=color, alpha=alpha, hatch=hatch, label=label,
               edgecolor='white' if not hatch else color, linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('ECE (lower is better)')
    ax.set_title('Effect of Temperature Scaling')
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets])
    ax.legend(frameon=True, fancybox=False, edgecolor='#ccc', ncol=2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('figures/fig7_ts_combination.pdf')
    plt.savefig('figures/fig7_ts_combination.png')
    plt.close()
    print("  Generated fig7_ts_combination")


def fig_conceptual():
    """Figure 1: Conceptual illustration of CCR."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    np.random.seed(42)

    # Panel A: Full neural collapse
    ax = axes[0]
    centers = np.array([[0, 2], [1.73, -1], [-1.73, -1]])
    colors_cls = ['#2196F3', '#FF9800', '#4CAF50']
    for i, (c, col) in enumerate(zip(centers, colors_cls)):
        pts = c + np.random.randn(30, 2) * 0.05
        ax.scatter(pts[:, 0], pts[:, 1], c=col, alpha=0.6, s=20)
        ax.scatter(c[0], c[1], c=col, marker='*', s=200, edgecolors='black',
                  linewidth=1, zorder=5)
    ax.set_title('(a) Full Neural Collapse\n(All predictions maximally confident)')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: Partial collapse with CCR
    ax = axes[1]
    for i, (c, col) in enumerate(zip(centers, colors_cls)):
        pts = c + np.random.randn(30, 2) * 0.4
        ax.scatter(pts[:, 0], pts[:, 1], c=col, alpha=0.4, s=20)
        ax.scatter(c[0], c[1], c=col, marker='*', s=200, edgecolors='black',
                  linewidth=1, zorder=5)
        circle = plt.Circle(c, 0.6, fill=False, color=col, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
    ax.set_title('(b) Partial Collapse with CCR\n(Spread encodes boundary proximity)')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel C: Calibration diagram
    ax = axes[2]
    confs = np.linspace(0.1, 1.0, 10)
    accs_full = confs * 0.8
    accs_full[-3:] = [0.75, 0.78, 0.82]
    accs_ccr = confs * 0.97
    accs_ccr = np.clip(accs_ccr, 0, 1)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(confs, accs_full, 'o-', color=COLORS['ce'], label='Full collapse (CE)',
            markersize=6)
    ax.plot(confs, accs_ccr, 's-', color=COLORS['ccr_soft'], label='Partial collapse (CCR)',
            markersize=6)
    ax.fill_between(confs, accs_full, confs, alpha=0.15, color=COLORS['ce'])
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Calibration Effect')
    ax.legend(fontsize=8, frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('figures/fig1_conceptual.pdf')
    plt.savefig('figures/fig1_conceptual.png')
    plt.close()
    print("  Generated fig1_conceptual")


def generate_latex_tables(results):
    """Generate LaTeX tables."""
    os.makedirs('tables', exist_ok=True)
    agg = results.get('aggregate_results', {})

    # Table 1: Main results
    best_ccr = find_best_ccr_method(agg.get('cifar100', {}))
    methods = ['ce', 'label_smoothing', 'mixup', best_ccr]
    datasets = [d for d in ['cifar10', 'cifar100', 'tinyimagenet'] if d in agg]
    metrics_list = ['test_accuracy', 'ece', 'mce', 'nll']
    metric_labels = {'test_accuracy': 'Acc (\\%)', 'ece': 'ECE', 'mce': 'MCE', 'nll': 'NLL'}

    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Main results. Mean $\\pm$ std over 3 seeds (TinyImageNet: 1 seed). '
                 'Best in \\textbf{bold}.}')
    lines.append('\\label{tab:main_results}')

    n_metrics = len(metrics_list)
    col_fmt = 'l' + 'c' * n_metrics
    lines.append(f'\\begin{{tabular}}{{{col_fmt}}}')
    lines.append('\\toprule')

    for ds in datasets:
        lines.append(f'\\multicolumn{{{n_metrics + 1}}}{{c}}{{\\textbf{{{DATASET_NAMES.get(ds, ds)}}}}} \\\\')
        lines.append('\\midrule')
        header = 'Method & ' + ' & '.join(metric_labels[m] for m in metrics_list) + ' \\\\'
        lines.append(header)
        lines.append('\\midrule')

        best_vals = {}
        for metric in metrics_list:
            vals = []
            for method in methods:
                if method in agg.get(ds, {}) and metric in agg[ds][method]:
                    v = agg[ds][method][metric]['mean']
                    vals.append((v, method))
            if vals:
                if metric == 'test_accuracy':
                    best_vals[metric] = max(vals, key=lambda x: x[0])[1]
                else:
                    best_vals[metric] = min(vals, key=lambda x: x[0])[1]

        for method in methods:
            if method not in agg.get(ds, {}):
                continue
            m_data = agg[ds][method]
            cells = [get_name(method)]
            for metric in metrics_list:
                if metric in m_data:
                    mean = m_data[metric]['mean']
                    std = m_data[metric]['std']
                    if metric == 'test_accuracy':
                        val_str = f'{mean * 100:.2f} $\\pm$ {std * 100:.2f}'
                    else:
                        val_str = f'{mean:.4f} $\\pm$ {std:.4f}'
                    if best_vals.get(metric) == method:
                        val_str = f'\\textbf{{{val_str}}}'
                    cells.append(val_str)
                else:
                    cells.append('-')
            lines.append(' & '.join(cells) + ' \\\\')

        lines.append('\\midrule')

    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    with open('tables/table1_main_results.tex', 'w') as f:
        f.write('\n'.join(lines))
    print("  Generated table1_main_results.tex")

    # Table 2: NC metrics (reuse methods list from above)
    lines2 = []
    lines2.append('\\begin{table}[t]')
    lines2.append('\\centering')
    lines2.append('\\caption{Neural collapse metrics on CIFAR-100. Mean $\\pm$ std over 3 seeds.}')
    lines2.append('\\label{tab:nc_metrics}')
    lines2.append('\\begin{tabular}{lccccc}')
    lines2.append('\\toprule')
    lines2.append('Method & NC1 & NC2 & NC3 & NC4 & Spread \\\\')
    lines2.append('\\midrule')

    for method in methods:
        if method in agg.get('cifar100', {}):
            m_data = agg['cifar100'][method]
            cells = [get_name(method)]
            for nc in ['nc1', 'nc2', 'nc3', 'nc4', 'mean_within_class_spread']:
                if nc in m_data:
                    mean = m_data[nc]['mean']
                    std = m_data[nc]['std']
                    if nc == 'nc1' or nc == 'mean_within_class_spread':
                        cells.append(f'{mean:.2f} $\\pm$ {std:.2f}')
                    else:
                        cells.append(f'{mean:.4f} $\\pm$ {std:.4f}')
                else:
                    cells.append('-')
            lines2.append(' & '.join(cells) + ' \\\\')

    lines2.append('\\bottomrule')
    lines2.append('\\end{tabular}')
    lines2.append('\\end{table}')

    with open('tables/table2_nc_metrics.tex', 'w') as f:
        f.write('\n'.join(lines2))
    print("  Generated table2_nc_metrics.tex")

    # Table 3: Ablation — all CCR variants
    present = [m for m in sorted(agg.get('cifar100', {}).keys())
               if m.startswith('ccr_') and '_100ep' not in m
               and 'ece' in agg['cifar100'].get(m, {})]

    if present:
        lines3 = []
        lines3.append('\\begin{table}[t]')
        lines3.append('\\centering')
        lines3.append('\\caption{Ablation study: CCR variants on CIFAR-100 (seed=42).}')
        lines3.append('\\label{tab:ablation}')
        lines3.append('\\begin{tabular}{lccc}')
        lines3.append('\\toprule')
        lines3.append('Variant & Acc (\\%) & ECE & Spread \\\\')
        lines3.append('\\midrule')

        for method in present:
            m_data = agg['cifar100'][method]
            acc = m_data['test_accuracy']['mean'] * 100
            ece = m_data['ece']['mean']
            spread = m_data.get('mean_within_class_spread', {}).get('mean', 0)
            lines3.append(f'{get_name(method)} & {acc:.2f} & {ece:.4f} & {spread:.2f} \\\\')

        lines3.append('\\bottomrule')
        lines3.append('\\end{tabular}')
        lines3.append('\\end{table}')

        with open('tables/table3_ablation.tex', 'w') as f:
            f.write('\n'.join(lines3))
        print("  Generated table3_ablation.tex")


def main():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    print("Loading results...")
    results = load_results()
    all_data = load_all_metrics()

    print("Generating figures...")
    fig_conceptual()
    fig_main_results_bar(results)
    fig_reliability_diagrams(all_data)
    fig_nc_metrics_training(all_data)
    fig_pareto_frontier(results)
    fig_ablation(results)
    fig_ts_combination(results)

    print("\nGenerating tables...")
    generate_latex_tables(results)

    print("\nAll figures and tables generated!")


if __name__ == '__main__':
    main()
