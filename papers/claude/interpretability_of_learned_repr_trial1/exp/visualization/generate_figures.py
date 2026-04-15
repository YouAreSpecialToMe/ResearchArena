"""Generate all paper-quality figures."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import spearmanr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

MODEL_NAMES = {
    "gpt2_small": "GPT-2 Small",
    "pythia_160m": "Pythia-160M",
    "pythia_410m": "Pythia-410M",
}
MODELS = list(MODEL_NAMES.keys())
COLORS = {'core': '#2196F3', 'peripheral': '#FF5722', 'random': '#9E9E9E'}


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fig1_convergence_distribution():
    """Figure 1: Convergence score distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, model_key in zip(axes, MODELS):
        data = load_json(f"convergence_scores_{model_key}.json")
        if data is None:
            continue

        scores = np.array(data["convergence_score_decoder"])
        # Filter to only non-dead features (convergence > 0)
        active = scores[scores > 0.1]

        ax.hist(active, bins=50, density=True, alpha=0.7, color='#1976D2', edgecolor='white', linewidth=0.5)
        for tau, color, ls in [(0.7, '#FFA000', '--'), (0.8, '#D32F2F', '-'), (0.9, '#388E3C', ':')]:
            ax.axvline(tau, color=color, linestyle=ls, linewidth=2, label=f'$\\tau$={tau}')

        core_frac = data["core_fraction"]
        ax.set_title(f'{MODEL_NAMES[model_key]}\n(Core: {100*core_frac:.1f}%)')
        ax.set_xlabel('Convergence Score (Decoder Cosine Sim)')
        if ax == axes[0]:
            ax.set_ylabel('Density')
            ax.legend(loc='upper left', framealpha=0.9)

    fig.suptitle('Distribution of Feature Convergence Scores', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_convergence_distribution.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_convergence_distribution.png'))
    plt.close()
    print("  Figure 1 saved.")


def fig2_stability_vs_universality():
    """Figure 2: Seed stability vs cross-model universality."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, model_key in zip(axes, MODELS):
        conv_data = load_json(f"convergence_scores_{model_key}.json")
        align_data = load_json(f"cross_model_alignment_{model_key}.json")
        if conv_data is None or align_data is None:
            continue

        conv = np.array(conv_data["convergence_score_combined"])
        univ = np.array(align_data["universality_scores"])
        core_mask = np.array(conv_data["core_mask_08_75"])

        # Subsample for plotting
        n = len(conv)
        idx = np.random.RandomState(42).choice(n, min(3000, n), replace=False)

        ax.scatter(conv[idx][~core_mask[idx]], univ[idx][~core_mask[idx]],
                   alpha=0.15, s=3, c=COLORS['peripheral'], label='Peripheral', rasterized=True)
        ax.scatter(conv[idx][core_mask[idx]], univ[idx][core_mask[idx]],
                   alpha=0.25, s=5, c=COLORS['core'], label='Core', rasterized=True)

        rho = align_data["spearman_rho_combined"]
        p = align_data["spearman_p_combined"]
        ax.set_title(f'{MODEL_NAMES[model_key]}\n$\\rho$={rho:.3f}, p={p:.1e}')
        ax.set_xlabel('Convergence Score')
        if ax == axes[0]:
            ax.set_ylabel('Cross-Model Alignment')
            ax.legend(loc='upper left', markerscale=5, framealpha=0.9)

    fig.suptitle('Seed Stability Predicts Cross-Model Universality', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_stability_vs_universality.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_stability_vs_universality.png'))
    plt.close()
    print("  Figure 2 saved.")


def fig3_causal_importance():
    """Figure 3: Causal importance comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, model_key in zip(axes, MODELS):
        causal = load_json(f"causal_importance_{model_key}.json")
        conv_data = load_json(f"convergence_scores_{model_key}.json")
        if causal is None or conv_data is None:
            continue

        kl = np.array(causal["kl_divergences"])
        core_mask = np.array(conv_data["core_mask_08_75"])

        core_kl = kl[core_mask]
        periph_kl = kl[~core_mask]

        # Log scale for better visualization
        core_kl_log = np.log10(core_kl.clip(min=1e-10))
        periph_kl_log = np.log10(periph_kl.clip(min=1e-10))

        parts = ax.violinplot([core_kl_log[core_kl_log > -8], periph_kl_log[periph_kl_log > -8]],
                               positions=[0, 1], showmeans=True, showmedians=True)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor([COLORS['core'], COLORS['peripheral']][i])
            pc.set_alpha(0.7)

        d = causal["cohens_d"]
        p = causal["mann_whitney_p"]
        ax.set_title(f'{MODEL_NAMES[model_key]}\nd={d:.3f}, p={p:.1e}')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Core', 'Peripheral'])
        if ax == axes[0]:
            ax.set_ylabel('log$_{10}$(KL Divergence)')

    fig.suptitle('Peripheral Features Have Higher Per-Feature Causal Importance', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_causal_importance.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_causal_importance.png'))
    plt.close()
    print("  Figure 3 saved.")


def fig4_roc_curves():
    """Figure 4: ROC curves for predicting high-importance features."""
    from sklearn.metrics import roc_curve, auc

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, model_key in zip(axes, MODELS):
        causal = load_json(f"causal_importance_{model_key}.json")
        conv_data = load_json(f"convergence_scores_{model_key}.json")
        if causal is None or conv_data is None:
            continue

        kl = np.array(causal["kl_divergences"])
        top20 = (kl >= np.percentile(kl, 80)).astype(int)

        predictors = {
            f'Convergence (AUC={causal["auc_convergence"]:.3f})':
                np.array(conv_data["convergence_score_combined"]),
            f'Act. Freq. (AUC={causal["auc_freq"]:.3f})':
                None,  # We'll load this separately
            f'Dec. Norm (AUC={causal["auc_norm"]:.3f})':
                None,
        }

        # Load activation data for baselines
        import torch
        sae_dir = os.path.join(os.path.dirname(__file__), '..', 'sae_training')
        act_path = os.path.join(sae_dir, f"activations_{model_key}_seed42.pt")
        sae_acts = torch.load(act_path, map_location="cpu", weights_only=True)
        act_freq = (sae_acts > 0).float().mean(dim=0).numpy()

        cp = torch.load(os.path.join(sae_dir, f"{model_key}_seed42.pt"),
                        map_location="cpu", weights_only=True)
        dec_norm = cp["state_dict"]["W_dec"].norm(dim=1).numpy()

        # Plot ROC curves
        conv_scores = np.array(conv_data["convergence_score_combined"])
        for name, scores, color in [
            (f'Convergence (AUC={causal["auc_convergence"]:.3f})', conv_scores, '#1976D2'),
            (f'Act. Freq. (AUC={causal["auc_freq"]:.3f})', act_freq, '#388E3C'),
            (f'Dec. Norm (AUC={causal["auc_norm"]:.3f})', dec_norm, '#FFA000'),
        ]:
            fpr, tpr, _ = roc_curve(top20, scores)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=name)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.500)')
        ax.set_title(MODEL_NAMES[model_key])
        ax.set_xlabel('False Positive Rate')
        if ax == axes[0]:
            ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle('ROC Curves for Predicting High-Importance Features', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_roc_curves.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_roc_curves.png'))
    plt.close()
    print("  Figure 4 saved.")


def fig5_subspace_analysis():
    """Figure 5: Subspace analysis results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # (a) Principal angles comparison
    ax = axes[0]
    bar_data = {'Core': [], 'Peripheral': [], 'Null': []}
    model_labels = []
    for model_key in MODELS:
        data = load_json(f"subspace_analysis_{model_key}.json")
        if data is None:
            continue
        model_labels.append(MODEL_NAMES[model_key].replace('Pythia-', 'P-'))
        bar_data['Core'].append(data['mean_core_angle'])
        bar_data['Peripheral'].append(data['mean_peripheral_angle'])
        bar_data['Null'].append(data['null_angles_mean'])

    if model_labels:
        x = np.arange(len(model_labels))
        width = 0.25
        ax.bar(x - width, bar_data['Core'], width, color=COLORS['core'], label='Core')
        ax.bar(x, bar_data['Peripheral'], width, color=COLORS['peripheral'], label='Peripheral')
        ax.bar(x + width, bar_data['Null'], width, color=COLORS['random'], label='Null (random)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel('Mean Principal Angle (rad)')
        ax.set_title('Subspace Consistency')
        ax.legend()

    # (b) Reconstruction quality
    ax = axes[1]
    recon_labels = []
    ev_data = {'All': [], 'Core': [], 'Peripheral': [], 'Random': []}
    for model_key in MODELS:
        data = load_json(f"subspace_analysis_{model_key}.json")
        if data is None:
            continue
        recon_labels.append(MODEL_NAMES[model_key].replace('Pythia-', 'P-'))
        ev_data['All'].append(data['reconstruction']['ev_all'])
        ev_data['Core'].append(data['reconstruction']['ev_core'])
        ev_data['Peripheral'].append(data['reconstruction']['ev_peripheral'])
        ev_data['Random'].append(data['reconstruction']['ev_random_subset'])

    if recon_labels:
        x = np.arange(len(recon_labels))
        width = 0.2
        for i, (name, color) in enumerate([('All', '#4CAF50'), ('Core', COLORS['core']),
                                            ('Peripheral', COLORS['peripheral']),
                                            ('Random', COLORS['random'])]):
            ax.bar(x + (i-1.5)*width, ev_data[name], width, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(recon_labels)
        ax.set_ylabel('Explained Variance (relative)')
        ax.set_title('Reconstruction Quality')
        ax.legend(fontsize=8)

    # (c) Feature properties comparison
    ax = axes[2]
    props = []
    for model_key in MODELS:
        data = load_json(f"subspace_analysis_{model_key}.json")
        if data is None:
            continue
        fp = data['feature_properties']
        props.append({
            'Act Freq': (fp['act_freq_core_mean'], fp['act_freq_periph_mean']),
            'Dec Norm': (fp['dec_norm_core_mean'], fp['dec_norm_periph_mean']),
            'E-D Align': (fp['enc_dec_align_core_mean'], fp['enc_dec_align_periph_mean']),
        })

    if props:
        # Average across models and plot
        prop_names = list(props[0].keys())
        core_vals = [np.mean([p[n][0] for p in props]) for n in prop_names]
        periph_vals = [np.mean([p[n][1] for p in props]) for n in prop_names]
        x = np.arange(len(prop_names))
        width = 0.35
        ax.bar(x - width/2, core_vals, width, color=COLORS['core'], label='Core')
        ax.bar(x + width/2, periph_vals, width, color=COLORS['peripheral'], label='Peripheral')
        ax.set_xticks(x)
        ax.set_xticklabels(prop_names)
        ax.set_ylabel('Mean Value')
        ax.set_title('Feature Properties')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_subspace_analysis.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_subspace_analysis.png'))
    plt.close()
    print("  Figure 5 saved.")


def fig6_ablations():
    """Figure 6: Ablation study results."""
    data = load_json("ablation_results.json")
    if data is None:
        print("  Skipping Figure 6 - no ablation results")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Number of seeds
    ax = axes[0]
    for model_key in MODELS:
        if model_key not in data:
            continue
        ns = data[model_key]["num_seeds"]
        seeds = sorted(ns.keys(), key=int)
        rhos = [ns[s]["spearman_rho"] for s in seeds]
        ax.plot([int(s) for s in seeds], rhos, 'o-', label=MODEL_NAMES[model_key], linewidth=2, markersize=6)
    ax.set_xlabel('Number of Seeds')
    ax.set_ylabel("Spearman $\\rho$ with Importance")
    ax.set_title('Effect of Number of Seeds')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (b) Matching method
    ax = axes[1]
    methods = ["top1", "greedy", "hungarian"]
    x = np.arange(len(methods))
    width = 0.25
    for i, model_key in enumerate(MODELS):
        if model_key not in data:
            continue
        mm = data[model_key]["matching_method"]
        rhos = [mm[m]["spearman_rho"] for m in methods]
        ax.bar(x + (i-1)*width, rhos, width, label=MODEL_NAMES[model_key])
    ax.set_xticks(x)
    ax.set_xticklabels(['Top-1', 'Greedy', 'Hungarian'])
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title('Matching Method')
    ax.legend(fontsize=9)

    # (c) Threshold sensitivity
    ax = axes[2]
    for model_key in MODELS:
        if model_key not in data:
            continue
        ts = data[model_key]["threshold_sensitivity"]
        taus = sorted(ts.keys(), key=float)
        ds = [ts[t]["cohens_d"] for t in taus]
        fracs = [ts[t]["core_fraction"] for t in taus]
        ax.plot([float(t) for t in taus], ds, 'o-', label=f'{MODEL_NAMES[model_key]} (d)',
                linewidth=2, markersize=6)
    ax.set_xlabel('Threshold $\\tau$')
    ax.set_ylabel("Cohen's d")
    ax.set_title('Threshold Sensitivity')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_ablations.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_ablations.png'))
    plt.close()
    print("  Figure 6 saved.")


def generate_tables():
    """Generate LaTeX tables for the paper."""
    # Table 1: Main results
    rows = []
    for model_key in MODELS:
        conv = load_json(f"convergence_scores_{model_key}.json")
        align = load_json(f"cross_model_alignment_{model_key}.json")
        causal = load_json(f"causal_importance_{model_key}.json")
        if conv is None or causal is None:
            continue

        row = {
            "model": MODEL_NAMES[model_key],
            "core_frac": f"{100*conv['core_fraction']:.1f}\\%",
            "rho_univ": f"{align['spearman_rho_combined']:.3f}" if align else "---",
            "cohens_d": f"{causal['cohens_d']:.3f}",
            "auc_conv": f"{causal['auc_convergence']:.3f}",
            "auc_freq": f"{causal['auc_freq']:.3f}",
        }
        rows.append(row)

    latex = "\\begin{table}[t]\n\\centering\n\\caption{Main Results}\n"
    latex += "\\begin{tabular}{lccccc}\n\\toprule\n"
    latex += "Model & Core \\% & $\\rho$(Conv,Univ) & Cohen's $d$ & AUC (Conv) & AUC (Freq) \\\\\n\\midrule\n"
    for r in rows:
        latex += f"{r['model']} & {r['core_frac']} & {r['rho_univ']} & {r['cohens_d']} & {r['auc_conv']} & {r['auc_freq']} \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    with open(os.path.join(FIGURES_DIR, 'table1_main_results.tex'), 'w') as f:
        f.write(latex)
    print("  Table 1 saved.")


def main():
    print("Generating figures...")
    fig1_convergence_distribution()
    fig2_stability_vs_universality()
    fig3_causal_importance()
    fig4_roc_curves()
    fig5_subspace_analysis()
    fig6_ablations()
    generate_tables()
    print("\nAll figures generated!")


if __name__ == "__main__":
    main()
