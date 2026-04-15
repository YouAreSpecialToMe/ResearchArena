"""Generate all paper figures from experiment results."""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
COLORS = sns.color_palette('colorblind')

RESULTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURES, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
EPSILONS = [1, 4, 8]
SPARSITIES = [50, 70, 90]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def get_metric(ds, variant, metric='worst_group_accuracy', seeds=SEEDS, **kwargs):
    """Extract metric values across seeds for a given config."""
    vals = []
    for s in seeds:
        if variant == 'baseline':
            path = f"{RESULTS}/{ds}/baseline/metrics_seed{s}.json"
        elif variant == 'dp_only':
            path = f"{RESULTS}/{ds}/dp_only/metrics_eps{kwargs['eps']}_seed{s}.json"
        elif variant == 'comp_only':
            path = f"{RESULTS}/{ds}/comp_only/metrics_sp{kwargs['sp']}_ft_seed{s}.json"
        elif variant == 'dp_comp':
            path = f"{RESULTS}/{ds}/dp_comp/metrics_eps{kwargs['eps']}_sp{kwargs['sp']}_ft_seed{s}.json"
        elif variant == 'fairprune_dp':
            path = f"{RESULTS}/{ds}/fairprune_dp/metrics_eps{kwargs['eps']}_sp{kwargs['sp']}_ft_seed{s}.json"
        elif variant == 'ablation':
            path = f"{RESULTS}/{ds}/ablation/ablation_{kwargs['method']}_eps{kwargs['eps']}_sp{kwargs['sp']}_seed{s}.json"
        else:
            continue
        d = load_json(path)
        if d and metric in d:
            vals.append(d[metric] * 100 if metric.endswith('accuracy') or metric.endswith('gap') else d[metric])
    return vals


def fig1_compounding_heatmap():
    """Compounding ratio heatmap for both datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for idx, ds in enumerate(['cifar10', 'utkface']):
        cr_matrix = np.full((3, 3), np.nan)
        for i, eps in enumerate(EPSILONS):
            baseline_wga = get_metric(ds, 'baseline')
            dp_wga = get_metric(ds, 'dp_only', eps=eps)
            if not baseline_wga or not dp_wga:
                continue
            for j, sp in enumerate(SPARSITIES):
                comp_wga = get_metric(ds, 'comp_only', sp=sp)
                dc_wga = get_metric(ds, 'dp_comp', eps=eps, sp=sp)
                if not comp_wga or not dc_wga:
                    continue
                n = min(len(baseline_wga), len(dp_wga), len(comp_wga), len(dc_wga))
                crs = []
                for k in range(n):
                    dd = baseline_wga[k] - dp_wga[k]
                    dc = baseline_wga[k] - comp_wga[k]
                    ddc = baseline_wga[k] - dc_wga[k]
                    denom = dd + dc
                    if denom > 0:
                        crs.append(ddc / denom)
                if crs:
                    cr_matrix[i, j] = np.mean(crs)

        ax = axes[idx]
        mask = np.isnan(cr_matrix)
        im = ax.imshow(cr_matrix, cmap='RdYlGn_r', vmin=0.6, vmax=1.4, aspect='auto')
        for i in range(3):
            for j in range(3):
                if not mask[i, j]:
                    color = 'white' if cr_matrix[i, j] > 1.1 or cr_matrix[i, j] < 0.7 else 'black'
                    ax.text(j, i, f'{cr_matrix[i, j]:.2f}', ha='center', va='center',
                           fontsize=14, fontweight='bold', color=color)
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=11, color='gray')
        ax.set_xticks(range(3))
        ax.set_xticklabels([f'{sp}%' for sp in SPARSITIES])
        ax.set_yticks(range(3))
        ax.set_yticklabels([f'ε={eps}' for eps in EPSILONS])
        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Privacy Budget (ε)')
        title = 'CIFAR-10' if ds == 'cifar10' else 'UTKFace'
        ax.set_title(f'{title} Compounding Ratio')

    fig.colorbar(im, ax=axes, label='CR (>1 = super-additive)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig1_compounding_ratio_heatmap.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig1_compounding_ratio_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Generated fig1_compounding_ratio_heatmap")


def fig2_worst_group_accuracy():
    """Worst-group accuracy across configurations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, ds in enumerate(['cifar10', 'utkface']):
        ax = axes[idx]
        x = np.arange(len(SPARSITIES))
        width = 0.2

        for i, eps in enumerate(EPSILONS):
            means = []
            stds = []
            for sp in SPARSITIES:
                vals = get_metric(ds, 'dp_comp', eps=eps, sp=sp)
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + i*width - width, means, width, yerr=stds, label=f'ε={eps}',
                   color=COLORS[i], capsize=3)

        # DP-only baseline (no pruning)
        for i, eps in enumerate(EPSILONS):
            dp_vals = get_metric(ds, 'dp_only', eps=eps)
            if dp_vals:
                ax.axhline(np.mean(dp_vals), color=COLORS[i], linestyle='--', alpha=0.5)

        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Worst-Group Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{sp}%' for sp in SPARSITIES])
        title = 'CIFAR-10' if ds == 'cifar10' else 'UTKFace'
        ax.set_title(f'{title}')
        ax.legend()

    plt.suptitle('Worst-Group Accuracy: DP + Compression', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig2_worst_group_accuracy.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig2_worst_group_accuracy.png', bbox_inches='tight')
    plt.close()
    print("Generated fig2_worst_group_accuracy")


def fig3_accuracy_gap():
    """Accuracy gap vs sparsity for different methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, ds in enumerate(['cifar10', 'utkface']):
        ax = axes[idx]
        eps = 4  # Representative setting

        for method, label, color in [
            ('dp_comp', 'Magnitude', COLORS[0]),
            ('fairprune_dp', 'FairPrune-DP', COLORS[1]),
        ]:
            means = []
            stds = []
            valid_sp = []
            for sp in SPARSITIES:
                if method == 'dp_comp':
                    vals = get_metric(ds, method, metric='accuracy_gap', eps=eps, sp=sp)
                else:
                    vals = get_metric(ds, method, metric='accuracy_gap', eps=eps, sp=sp)
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                    valid_sp.append(sp)

            if means:
                ax.errorbar(valid_sp, means, yerr=stds, marker='o', label=label,
                           color=color, capsize=4, linewidth=2)

        # DP-only baseline
        dp_gap = get_metric(ds, 'dp_only', metric='accuracy_gap', eps=eps)
        if dp_gap:
            ax.axhline(np.mean(dp_gap), color='gray', linestyle='--', label='DP-only (no pruning)')

        ax.set_xlabel('Sparsity (%)')
        ax.set_ylabel('Accuracy Gap (%)')
        title = 'CIFAR-10' if ds == 'cifar10' else 'UTKFace'
        ax.set_title(f'{title} (ε={eps})')
        ax.legend()

    plt.suptitle('Accuracy Gap: Magnitude vs FairPrune-DP', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig3_accuracy_gap_vs_sparsity.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig3_accuracy_gap_vs_sparsity.png', bbox_inches='tight')
    plt.close()
    print("Generated fig3_accuracy_gap_vs_sparsity")


def fig5_fairprune_comparison():
    """FairPrune-DP vs magnitude pruning comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, ds in enumerate(['cifar10', 'utkface']):
        ax = axes[idx]
        eps = 4
        x = np.arange(len(SPARSITIES))
        width = 0.3

        for i, (method, label) in enumerate([('dp_comp', 'Magnitude'), ('fairprune_dp', 'FairPrune-DP')]):
            means = []
            stds = []
            for sp in SPARSITIES:
                vals = get_metric(ds, method, eps=eps, sp=sp)
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + i*width - width/2, means, width, yerr=stds, label=label,
                   color=COLORS[i], capsize=3)

        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Worst-Group Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{sp}%' for sp in SPARSITIES])
        title = 'CIFAR-10' if ds == 'cifar10' else 'UTKFace'
        ax.set_title(f'{title} (ε={eps})')
        ax.legend()

    plt.suptitle('FairPrune-DP vs Magnitude Pruning on DP Models', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig5_fairprune_comparison.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig5_fairprune_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated fig5_fairprune_comparison")


def fig6_per_subgroup():
    """Per-subgroup accuracy across pipeline stages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, ds in enumerate(['cifar10', 'utkface']):
        ax = axes[idx]
        eps = 4
        sp = 70

        stages = ['Baseline', 'DP-only', 'Comp-only', 'DP+Comp']
        configs = [
            ('baseline', {}),
            ('dp_only', {'eps': eps}),
            ('comp_only', {'sp': sp}),
            ('dp_comp', {'eps': eps, 'sp': sp}),
        ]

        # Check if fairprune data exists
        fp_vals = get_metric(ds, 'fairprune_dp', eps=eps, sp=sp)
        if fp_vals:
            stages.append('FairPrune-DP')
            configs.append(('fairprune_dp', {'eps': eps, 'sp': sp}))

        # Collect per-subgroup means
        all_subgroup_means = {}
        for stage, (variant, kw) in zip(stages, configs):
            sg_accs = {}
            for s in SEEDS:
                if variant == 'baseline':
                    path = f"{RESULTS}/{ds}/baseline/metrics_seed{s}.json"
                elif variant == 'dp_only':
                    path = f"{RESULTS}/{ds}/dp_only/metrics_eps{kw['eps']}_seed{s}.json"
                elif variant == 'comp_only':
                    path = f"{RESULTS}/{ds}/comp_only/metrics_sp{kw['sp']}_ft_seed{s}.json"
                elif variant == 'dp_comp':
                    path = f"{RESULTS}/{ds}/dp_comp/metrics_eps{kw['eps']}_sp{kw['sp']}_ft_seed{s}.json"
                elif variant == 'fairprune_dp':
                    path = f"{RESULTS}/{ds}/fairprune_dp/metrics_eps{kw['eps']}_sp{kw['sp']}_ft_seed{s}.json"
                else:
                    continue
                d = load_json(path)
                if d and 'per_subgroup_accuracy' in d:
                    for sg, acc in d['per_subgroup_accuracy'].items():
                        if sg not in sg_accs:
                            sg_accs[sg] = []
                        sg_accs[sg].append(acc * 100)
            all_subgroup_means[stage] = {sg: np.mean(vals) for sg, vals in sg_accs.items()}

        # Plot
        subgroups = sorted(list(all_subgroup_means.get('Baseline', {}).keys()))
        if not subgroups:
            continue

        x = np.arange(len(stages))
        width = 0.8 / len(subgroups)
        sg_names = {
            'cifar10': {'0': 'Majority', '1': 'Minority'},
            'utkface': {'0': 'White', '1': 'Black', '2': 'Asian', '3': 'Indian', '4': 'Others'}
        }

        for j, sg in enumerate(subgroups):
            vals = [all_subgroup_means.get(stage, {}).get(sg, 0) for stage in stages]
            name = sg_names.get(ds, {}).get(sg, f'Group {sg}')
            ax.bar(x + j*width - len(subgroups)*width/2 + width/2, vals, width,
                   label=name, color=COLORS[j % len(COLORS)])

        ax.set_xlabel('Pipeline Stage')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=15, ha='right')
        title = 'CIFAR-10' if ds == 'cifar10' else 'UTKFace'
        ax.set_title(f'{title} (ε={eps}, {sp}% sparsity)')
        ax.legend(fontsize=9)

    plt.suptitle('Per-Subgroup Accuracy Across Pipeline Stages', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES}/fig6_per_subgroup_accuracy.pdf', bbox_inches='tight')
    plt.savefig(f'{FIGURES}/fig6_per_subgroup_accuracy.png', bbox_inches='tight')
    plt.close()
    print("Generated fig6_per_subgroup_accuracy")


if __name__ == '__main__':
    fig1_compounding_heatmap()
    fig2_worst_group_accuracy()
    fig3_accuracy_gap()
    fig5_fairprune_comparison()
    fig6_per_subgroup()
    print("\nAll figures generated!")
