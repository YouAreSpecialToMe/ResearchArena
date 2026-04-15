"""Generate additional analysis figures: NC-ECE correlations, NC tracking, etc."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

COLORS = {
    'ce': '#2196F3', 'label_smoothing': '#FF9800', 'mixup': '#4CAF50',
    'ccr_soft': '#E91E63', 'ccr_adaptive': '#9C27B0', 'ccr_spectral': '#795548',
    'ccr_fixed_tau15': '#607D8B',
}
NAMES = {
    'ce': 'Cross-Entropy', 'label_smoothing': 'Label Smoothing', 'mixup': 'Mixup',
    'ccr_soft': 'CCR-Soft', 'ccr_adaptive': 'CCR-Adaptive', 'ccr_spectral': 'CCR-Spectral',
    'ccr_fixed_tau15': 'CCR-Fixed',
}


def load_all():
    results = defaultdict(lambda: defaultdict(dict))
    for ds in os.listdir('results'):
        ds_dir = os.path.join('results', ds)
        if not os.path.isdir(ds_dir): continue
        for m in os.listdir(ds_dir):
            m_dir = os.path.join(ds_dir, m)
            if not os.path.isdir(m_dir): continue
            for s in os.listdir(m_dir):
                mf = os.path.join(m_dir, s, 'metrics.json')
                if os.path.exists(mf):
                    with open(mf) as f:
                        results[ds][m][s] = json.load(f)
    return results


def fig_nc1_ece_correlation(all_data):
    """Scatter plot: NC1 vs ECE for all methods/seeds, per dataset."""
    datasets = [d for d in ['cifar10', 'cifar100'] if d in all_data]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for method, seeds in sorted(all_data[ds].items()):
            if '_100ep' in method:
                continue
            color = COLORS.get(method, '#999')
            name = NAMES.get(method, method)
            for seed, d in seeds.items():
                nc1 = d.get('nc_metrics', {}).get('nc1')
                ece = d.get('calibration', {}).get('ece')
                if nc1 and ece:
                    ax.scatter(np.log10(nc1), ece, c=color, label=name, s=60, alpha=0.7,
                             edgecolors='white', linewidth=0.5)

        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=True)

        ds_name = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}.get(ds, ds)
        ax.set_xlabel('log$_{10}$(NC1)')
        ax.set_ylabel('ECE')
        ax.set_title(f'{ds_name}: NC1 vs ECE')

    plt.tight_layout()
    plt.savefig('figures/fig_nc1_ece_correlation.pdf')
    plt.savefig('figures/fig_nc1_ece_correlation.png')
    plt.close()
    print("  Generated fig_nc1_ece_correlation")


def fig_spread_vs_ece(all_data):
    """Scatter plot: within-class spread vs ECE."""
    datasets = [d for d in ['cifar10', 'cifar100'] if d in all_data]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for method, seeds in sorted(all_data[ds].items()):
            if '_100ep' in method:
                continue
            color = COLORS.get(method, '#999')
            name = NAMES.get(method, method)
            for seed, d in seeds.items():
                spread = d.get('nc_metrics', {}).get('mean_within_class_spread')
                ece = d.get('calibration', {}).get('ece')
                if spread is not None and ece is not None:
                    ax.scatter(spread, ece, c=color, label=name, s=60, alpha=0.7,
                             edgecolors='white', linewidth=0.5)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=True)

        ds_name = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}.get(ds, ds)
        ax.set_xlabel('Mean Within-Class Spread')
        ax.set_ylabel('ECE')
        ax.set_title(f'{ds_name}: Feature Spread vs ECE')

    plt.tight_layout()
    plt.savefig('figures/fig_spread_vs_ece.pdf')
    plt.savefig('figures/fig_spread_vs_ece.png')
    plt.close()
    print("  Generated fig_spread_vs_ece")


def fig_training_dynamics_cifar100():
    """Training dynamics: loss, accuracy, spread, CCR loss for all methods on CIFAR-100."""
    methods = ['ce', 'ccr_fixed_tau15', 'ccr_soft', 'ccr_adaptive']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for method in methods:
        log_path = f'results/cifar100/{method}/seed_42/training_log.json'
        if not os.path.exists(log_path):
            continue
        with open(log_path) as f:
            data = json.load(f)

        epochs = [e['epoch'] for e in data['log']]
        color = COLORS.get(method, '#999')
        name = NAMES.get(method, method)

        # Training loss
        losses = [e['train_loss'] for e in data['log']]
        axes[0, 0].plot(epochs, losses, color=color, label=name, alpha=0.8)

        # Test accuracy
        test_accs = [e['test_acc'] for e in data['log']]
        axes[0, 1].plot(epochs, test_accs, color=color, label=name, alpha=0.8)

        # Mean spread
        spreads = [e.get('mean_spread') for e in data['log']]
        if any(s is not None for s in spreads):
            valid = [(e, s) for e, s in zip(epochs, spreads) if s is not None]
            axes[1, 0].plot([v[0] for v in valid], [v[1] for v in valid],
                          color=color, label=name, alpha=0.8)

        # CCR loss
        ccr_losses = [e.get('ccr_loss') for e in data['log']]
        if any(c is not None for c in ccr_losses):
            valid = [(e, c) for e, c in zip(epochs, ccr_losses) if c is not None]
            axes[1, 1].plot([v[0] for v in valid], [v[1] for v in valid],
                          color=color, label=name, alpha=0.8)

    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss'); axes[0, 0].legend()

    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Test Accuracy'); axes[0, 1].legend()

    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Mean Within-Class Spread')
    axes[1, 0].set_title('Feature Spread During Training'); axes[1, 0].legend()

    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('CCR Loss')
    axes[1, 1].set_title('CCR Regularization Loss'); axes[1, 1].legend()

    plt.suptitle('Training Dynamics on CIFAR-100', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig_training_dynamics_detailed.pdf')
    plt.savefig('figures/fig_training_dynamics_detailed.png')
    plt.close()
    print("  Generated fig_training_dynamics_detailed")


def fig_calibration_improvement_summary(all_data):
    """Summary figure: ECE reduction for each method vs CE, per dataset."""
    datasets = [d for d in ['cifar10', 'cifar100'] if d in all_data]
    methods_order = ['ccr_fixed_tau15', 'ccr_soft', 'ccr_adaptive', 'label_smoothing', 'mixup']

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ce_eces = []
        for seed, d in all_data[ds].get('ce', {}).items():
            ce_eces.append(d['calibration']['ece'])
        ce_mean = np.mean(ce_eces) if ce_eces else 0

        names = []
        reductions = []
        errs = []
        colors_list = []

        for method in methods_order:
            if method not in all_data[ds]:
                continue
            m_eces = []
            for seed, d in all_data[ds][method].items():
                m_eces.append(d['calibration']['ece'])
            if m_eces and ce_mean > 0:
                red = (ce_mean - np.mean(m_eces)) / ce_mean * 100
                err = np.std(m_eces) / ce_mean * 100 if len(m_eces) > 1 else 0
                names.append(NAMES.get(method, method))
                reductions.append(red)
                errs.append(err)
                colors_list.append(COLORS.get(method, '#999'))

        bars = ax.barh(range(len(names)), reductions, xerr=errs, capsize=3,
                      color=colors_list, edgecolor='white', linewidth=0.5)

        # Add value labels
        for i, (r, bar) in enumerate(zip(reductions, bars)):
            ax.text(r + (1 if r >= 0 else -1), i, f'{r:+.1f}%', va='center', fontsize=9)

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('ECE Reduction vs CE (%)')
        ds_name = {'cifar10': 'CIFAR-10', 'cifar100': 'CIFAR-100'}.get(ds, ds)
        ax.set_title(f'{ds_name}')
        ax.invert_yaxis()

    plt.suptitle('ECE Reduction Relative to Cross-Entropy Baseline', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig_ece_reduction_summary.pdf')
    plt.savefig('figures/fig_ece_reduction_summary.png')
    plt.close()
    print("  Generated fig_ece_reduction_summary")


def main():
    os.makedirs('figures', exist_ok=True)
    all_data = load_all()

    print("Generating extra analysis figures...")
    fig_nc1_ece_correlation(all_data)
    fig_spread_vs_ece(all_data)
    fig_training_dynamics_cifar100()
    fig_calibration_improvement_summary(all_data)
    print("Done!")


if __name__ == '__main__':
    main()
