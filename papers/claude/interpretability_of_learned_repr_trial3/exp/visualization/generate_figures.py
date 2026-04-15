"""Generate publication-quality figures for the paper."""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import *

# Consistent styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})

TIER_COLORS = {
    "consensus": COLORS["consensus"],
    "partial": COLORS["partial"],
    "singleton": COLORS["singleton"],
    "random": COLORS["random"],
    "frequency": COLORS["frequency"],
}


def figure1_consensus_distribution():
    """Figure 1: Consensus score histograms per layer."""
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, layer in enumerate(LAYERS):
        ax = axes[idx]
        path = os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy")
        if not os.path.exists(path):
            ax.set_title(f"Layer {layer} (no data)")
            continue

        scores = np.load(path)
        # Filter out zero-score features (dead features)
        scores = scores[scores > 0]

        ax.hist(scores, bins=20, color="#6495ED", edgecolor="white", alpha=0.8)
        ax.axvline(CONSENSUS_HIGH, color=TIER_COLORS["consensus"], linestyle="--",
                   linewidth=2, label=f"High ({CONSENSUS_HIGH})")
        ax.axvline(CONSENSUS_LOW, color=TIER_COLORS["singleton"], linestyle="--",
                   linewidth=2, label=f"Low ({CONSENSUS_LOW})")
        ax.set_xlabel("Consensus Score")
        ax.set_ylabel("Number of Features")
        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize=9)

    fig.suptitle("Distribution of Feature Consensus Scores Across Layers", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_consensus_distribution.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_consensus_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  Saved Figure 1")


def figure2_causal_importance():
    """Figure 2: Consensus predicts causal importance."""
    eval_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "evaluation")
    matching_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "feature_matching")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, layer in enumerate(LAYERS):
        ci_path = os.path.join(eval_base, f"layer_{layer}", "causal_importance.npy")
        cs_path = os.path.join(matching_base, f"layer_{layer}", "consensus_scores.npy")
        tier_path = os.path.join(matching_base, f"layer_{layer}", "tier_labels.json")

        if not all(os.path.exists(p) for p in [ci_path, cs_path, tier_path]):
            continue

        ci = np.load(ci_path)
        cs = np.load(cs_path)
        with open(tier_path) as f:
            tiers = json.load(f)

        # Filter active features
        active = ci > 0
        ci_active = ci[active]
        cs_active = cs[active]
        tiers_active = [tiers[i] for i in range(len(tiers)) if active[i]]

        # Top row: scatter plot
        ax = axes[0, idx]
        for tier, color in TIER_COLORS.items():
            if tier in ["random", "frequency"]:
                continue
            mask = [t == tier for t in tiers_active]
            if any(mask):
                ax.scatter(cs_active[mask], ci_active[mask], c=color, alpha=0.3, s=5, label=tier)

        # Add regression line
        from scipy import stats as sp_stats
        slope, intercept, r, p, se = sp_stats.linregress(cs_active, ci_active)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, slope * x_line + intercept, "k-", linewidth=2)
        ax.set_xlabel("Consensus Score")
        ax.set_ylabel("Causal Importance")
        ax.set_title(f"Layer {layer} (r={r:.3f}, p={p:.2e})")
        ax.legend(markerscale=3, fontsize=9)

        # Bottom row: box plot
        ax2 = axes[1, idx]
        tier_data = {"consensus": [], "partial": [], "singleton": []}
        for c, t in zip(ci_active, tiers_active):
            if t in tier_data:
                tier_data[t].append(c)

        box_data = [tier_data.get(t, []) for t in ["consensus", "partial", "singleton"]]
        box_colors = [TIER_COLORS[t] for t in ["consensus", "partial", "singleton"]]

        bp = ax2.boxplot(box_data, labels=["Consensus", "Partial", "Singleton"],
                        patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel("Causal Importance")
        ax2.set_title(f"Layer {layer}")

    fig.suptitle("Consensus Score Predicts Causal Importance", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_causal_importance.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_causal_importance.png"), bbox_inches="tight")
    plt.close()
    print("  Saved Figure 2")


def figure3_sparse_probing():
    """Figure 3: Sparse probing results."""
    eval_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "evaluation")
    results_path = os.path.join(eval_base, "sparse_probing_results.json")
    if not os.path.exists(results_path):
        print("  No sparse probing results, skipping Figure 3")
        return

    with open(results_path) as f:
        results = json.load(f)

    tasks = list(results.keys())
    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 5))
    if n_tasks == 1:
        axes = [axes]

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = results[task]
        k_values = sorted(task_data.keys())

        for tier in ["consensus", "partial", "singleton", "random"]:
            accs = []
            ks = []
            for k_str in k_values:
                k_val = int(k_str.split("=")[1])
                if tier in task_data[k_str]:
                    acc = task_data[k_str][tier].get("accuracy", 0)
                    accs.append(acc)
                    ks.append(k_val)

            if accs:
                ax.plot(ks, accs, "o-", color=TIER_COLORS.get(tier, "gray"),
                       label=tier.capitalize(), linewidth=2, markersize=6)

        ax.set_xlabel("k (number of features)")
        ax.set_ylabel("Accuracy")
        ax.set_title(task.replace("_", " ").title())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sparse Probing Accuracy by Feature Tier", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure3_sparse_probing.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure3_sparse_probing.png"), bbox_inches="tight")
    plt.close()
    print("  Saved Figure 3")


def figure4_ablations():
    """Figure 4: Ablation studies (2x2 panel)."""
    ablation_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "ablation_studies")
    results_path = os.path.join(ablation_base, "ablation_results.json")
    if not os.path.exists(results_path):
        print("  No ablation results, skipping Figure 4")
        return

    with open(results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Number of seeds
    if "n_seeds" in results:
        ax = axes[0]
        n_seeds_data = results["n_seeds"]
        ns = sorted([int(k) for k in n_seeds_data.keys()])
        means = [n_seeds_data[str(n)]["mean_spearman_r"] for n in ns]
        stds = [n_seeds_data[str(n)]["std_spearman_r"] for n in ns]
        ax.errorbar(ns, means, yerr=stds, fmt="o-", color=TIER_COLORS["consensus"],
                   linewidth=2, markersize=8, capsize=4)
        ax.set_xlabel("Number of Seeds")
        ax.set_ylabel("Spearman Correlation\n(Consensus vs Causal Importance)")
        ax.set_title("(a) Effect of Number of Seeds")
        ax.grid(True, alpha=0.3)

    # Panel 2: Threshold sensitivity
    if "threshold" in results:
        ax = axes[1]
        thresh_data = results["threshold"]
        thresholds = sorted([float(k) for k in thresh_data.keys()])
        cohens_ds = [thresh_data[str(t)]["cohens_d"] for t in thresholds]
        n_features = [thresh_data[str(t)]["n_consensus_features"] for t in thresholds]

        ax.bar(range(len(thresholds)), cohens_ds, color=TIER_COLORS["consensus"], alpha=0.7)
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{t:.3f}" for t in thresholds], rotation=45)
        ax.set_xlabel("Consensus Threshold")
        ax.set_ylabel("Cohen's d (Consensus vs Singleton)")
        ax.set_title("(b) Threshold Sensitivity")

        # Add feature count as secondary axis
        ax2 = ax.twinx()
        ax2.plot(range(len(thresholds)), n_features, "ro-", linewidth=2, markersize=6)
        ax2.set_ylabel("# Consensus Features", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_ablations.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_ablations.png"), bbox_inches="tight")
    plt.close()
    print("  Saved Figure 4")


def figure5_manifold_analysis():
    """Figure 5: Manifold tiling analysis."""
    eval_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "evaluation")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: UMAP
    ax = axes[0]
    umap_path = os.path.join(eval_base, "layer_6", "umap_coords.npy")
    if os.path.exists(umap_path):
        coords = np.load(umap_path)
        scores = np.load(os.path.join(eval_base, "layer_6", "umap_consensus_scores.npy"))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=scores, cmap="RdYlBu",
                           s=3, alpha=0.5, vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label="Consensus Score")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("(a) UMAP of Decoder Vectors")
    else:
        ax.text(0.5, 0.5, "No UMAP data", ha="center", va="center", transform=ax.transAxes)

    # Panel 2-3: Load manifold analysis results
    manifold_path = os.path.join(eval_base, "manifold_analysis_results.json")
    if os.path.exists(manifold_path):
        with open(manifold_path) as f:
            manifold_results = json.load(f)

        # Panel 2: Neighborhood density comparison
        ax = axes[1]
        layer_data = manifold_results.get("6", manifold_results.get(6, {}))
        if "tier_density" in layer_data:
            tiers = ["consensus", "partial", "singleton"]
            means = [layer_data["tier_density"].get(t, {}).get("mean", 0) for t in tiers]
            stds = [layer_data["tier_density"].get(t, {}).get("std", 0) for t in tiers]
            colors = [TIER_COLORS[t] for t in tiers]

            bars = ax.bar(tiers, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
            ax.set_xlabel("Tier")
            ax.set_ylabel("Mean Neighborhood Size")
            ax.set_title("(b) Local Density by Tier")
            ax.set_xticklabels(["Consensus", "Partial", "Singleton"])

        # Panel 3: Cluster fraction
        ax = axes[2]
        if "tier_cluster_fraction" in layer_data:
            tiers = ["consensus", "partial", "singleton"]
            fracs = [layer_data["tier_cluster_fraction"].get(t, 0) for t in tiers]
            colors = [TIER_COLORS[t] for t in tiers]

            ax.bar(tiers, fracs, color=colors, alpha=0.7)
            ax.set_xlabel("Tier")
            ax.set_ylabel("Fraction in Dense Clusters")
            ax.set_title("(c) DBSCAN Cluster Membership")
            ax.set_xticklabels(["Consensus", "Partial", "Singleton"])

    fig.suptitle("Manifold Tiling Analysis (Layer 6)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_manifold_analysis.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_manifold_analysis.png"), bbox_inches="tight")
    plt.close()
    print("  Saved Figure 5")


def figure6_dictionary_eval():
    """Figure 6: Consensus dictionary evaluation."""
    eval_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "evaluation")
    results_path = os.path.join(eval_base, "consensus_dictionary_results.json")
    if not os.path.exists(results_path):
        print("  No dictionary results, skipping Figure 6")
        return

    with open(results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dict_names = ["consensus", "full_reference", "singleton", "random_subsample"]
    display_names = ["Consensus\nDict", "Full\nReference", "Singleton\nDict", "Random\nSubsample"]
    colors = [TIER_COLORS["consensus"], "#666666", TIER_COLORS["singleton"], TIER_COLORS["random"]]

    # Panel 1: Cosine similarity
    ax = axes[0]
    cosines = [results.get(d, {}).get("mean_cosine_sim", 0) for d in dict_names]
    ax.bar(display_names, cosines, color=colors, alpha=0.7)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("(a) Reconstruction Quality")

    # Panel 2: Features vs quality tradeoff
    ax = axes[1]
    n_feats = [results.get(d, {}).get("n_features", 0) for d in dict_names]
    for i, (n, c, name, color) in enumerate(zip(n_feats, cosines, display_names, colors)):
        ax.scatter(n, c, color=color, s=100, zorder=5, label=name.replace("\n", " "))
        ax.annotate(name.replace("\n", " "), (n, c), textcoords="offset points",
                   xytext=(10, 5), fontsize=9)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("(b) Quality vs Dictionary Size")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure6_dictionary_eval.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure6_dictionary_eval.png"), bbox_inches="tight")
    plt.close()
    print("  Saved Figure 6")


def generate_all_figures():
    """Generate all figures."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating figures...")

    figure1_consensus_distribution()
    figure2_causal_importance()
    figure3_sparse_probing()
    figure4_ablations()
    figure5_manifold_analysis()
    figure6_dictionary_eval()

    print("\nAll figures generated!")


if __name__ == "__main__":
    generate_all_figures()
