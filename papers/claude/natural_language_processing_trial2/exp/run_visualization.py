"""
Generate publication-quality figures and tables for SpecCheck paper.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import (
    save_json, load_json, get_model_short,
    MODELS, DATASETS, DATA_DIR, RESULTS_DIR, FIGURES_DIR, BASE_DIR
)
from shared.metrics import compute_auc_pr, compute_auc_roc

# Style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
COLORS = sns.color_palette("colorblind", 8)
METHOD_COLORS = {
    "speccheck": COLORS[0],
    "selfcheck": COLORS[1],
    "verbalized": COLORS[2],
    "logprob": COLORS[3],
    "random": COLORS[4],
}
METHOD_LABELS = {
    "speccheck": "SpecCheck (Ours)",
    "selfcheck": "SelfCheckGPT",
    "verbalized": "Verbalized Conf.",
    "logprob": "Logprob Conf.",
    "random": "Random",
}
DATASET_LABELS = {
    "factscore": "FActScore",
    "longfact": "LongFact",
    "truthfulqa": "TruthfulQA",
}
MODEL_LABELS = {
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B",
    "qwen": "Qwen2.5-7B",
}


def load_eval_stats():
    """Load evaluation statistics."""
    path = os.path.join(RESULTS_DIR, "evaluation_statistics.json")
    if os.path.exists(path):
        return load_json(path)
    return {}


def figure_1_method_overview():
    """Figure 1: SpecCheck pipeline schematic with example confidence profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: True claim (monotonic)
    levels = ["Original\n(L0)", "Approximate\n(L1)", "Category\n(L2)", "Abstract\n(L3)"]
    true_confs = [0.72, 0.78, 0.85, 0.91]
    halluc_confs = [0.68, 0.55, 0.79, 0.84]

    ax = axes[0]
    ax.plot(range(4), true_confs, "o-", color=COLORS[0], linewidth=2.5, markersize=10, label="True claim")
    ax.fill_between(range(4), true_confs, alpha=0.15, color=COLORS[0])
    for i, c in enumerate(true_confs):
        ax.annotate(f"{c:.2f}", (i, c), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color=COLORS[0])
    ax.set_xticks(range(4))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Model Confidence P(True)")
    ax.set_title("True Claim: Monotonic Confidence", fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Right: Hallucinated claim (non-monotonic)
    ax = axes[1]
    ax.plot(range(4), halluc_confs, "s-", color=COLORS[1], linewidth=2.5, markersize=10, label="Hallucinated claim")
    ax.fill_between(range(4), halluc_confs, alpha=0.15, color=COLORS[1])
    for i, c in enumerate(halluc_confs):
        ax.annotate(f"{c:.2f}", (i, c), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color=COLORS[1])
    # Highlight violation
    ax.annotate("Violation!", xy=(1, halluc_confs[1]), xytext=(1.5, 0.48),
                arrowprops=dict(arrowstyle="->", color="red", lw=2),
                fontsize=11, color="red", fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(levels, fontsize=9)
    ax.set_ylabel("Model Confidence P(True)")
    ax.set_title("Hallucinated Claim: Monotonicity Violation", fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"figure_1_method_overview.{ext}"))
    plt.close()
    print("  Figure 1 saved")


def figure_2_main_results():
    """Figure 2: Main results bar chart (AUC-PR across datasets and models)."""
    eval_stats = load_eval_stats()
    if not eval_stats:
        print("  No evaluation stats found for Figure 2")
        return

    models = [get_model_short(m) for m in MODELS]
    methods = ["speccheck", "selfcheck", "verbalized", "logprob", "random"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    bar_width = 0.15

    for di, dataset in enumerate(DATASETS):
        ax = axes[di]
        for mi, method in enumerate(methods):
            vals = []
            for model in models:
                key = f"{model}_{dataset}"
                if key in eval_stats and method in eval_stats[key].get("methods", {}):
                    vals.append(eval_stats[key]["methods"][method]["auc_pr"])
            if vals:
                x = np.arange(len(vals))
                bars = ax.bar(
                    x + mi * bar_width, vals, bar_width,
                    label=METHOD_LABELS.get(method, method),
                    color=METHOD_COLORS.get(method, COLORS[mi]),
                    alpha=0.85, edgecolor="white", linewidth=0.5
                )
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xlabel("Model")
        ax.set_xticks(np.arange(len(models)) + bar_width * 2)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
        ax.set_title(DATASET_LABELS.get(dataset, dataset), fontweight="bold")
        if di == 0:
            ax.set_ylabel("AUC-PR")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8)
    plt.suptitle("Hallucination Detection Performance (AUC-PR)", fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"figure_2_main_results.{ext}"))
    plt.close()
    print("  Figure 2 saved")


def figure_3_monotonicity_analysis():
    """Figure 3: Monotonicity analysis - profiles and distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect all confidence profiles
    factual_profiles = []
    halluc_profiles = []
    factual_mono_scores = []
    halluc_mono_scores = []

    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for dataset_name in DATASETS:
            scores_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
            label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")
            if not os.path.exists(scores_path) or not os.path.exists(label_path):
                continue

            scores = load_json(scores_path)
            labeled = load_json(label_path)
            label_map = {c["claim_id"]: c["label"] for c in labeled}

            for item in scores:
                cid = item["claim_id"]
                if cid not in label_map:
                    continue
                confs = item["confidences"]
                if len(confs) < 4:
                    continue
                if label_map[cid] == 0:
                    factual_profiles.append(confs[:4])
                    factual_mono_scores.append(item["monotonicity_score"])
                else:
                    halluc_profiles.append(confs[:4])
                    halluc_mono_scores.append(item["monotonicity_score"])

    if not factual_profiles or not halluc_profiles:
        print("  Not enough data for Figure 3")
        return

    # Left: Average confidence profiles
    ax = axes[0]
    fp = np.array(factual_profiles)
    hp = np.array(halluc_profiles)
    levels = ["L0\n(Original)", "L1\n(Approx.)", "L2\n(Category)", "L3\n(Abstract)"]

    ax.errorbar(range(4), fp.mean(axis=0), yerr=fp.std(axis=0), fmt="o-",
                color=COLORS[0], linewidth=2.5, markersize=8, capsize=5,
                label=f"Factual (n={len(fp)})")
    ax.errorbar(range(4), hp.mean(axis=0), yerr=hp.std(axis=0), fmt="s-",
                color=COLORS[1], linewidth=2.5, markersize=8, capsize=5,
                label=f"Hallucinated (n={len(hp)})")
    ax.set_xticks(range(4))
    ax.set_xticklabels(levels)
    ax.set_ylabel("Mean Confidence P(True)")
    ax.set_title("Confidence Profiles by Claim Type", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Monotonicity score distribution
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    ax.hist(factual_mono_scores, bins=bins, alpha=0.6, color=COLORS[0],
            label=f"Factual (n={len(factual_mono_scores)})", density=True, edgecolor="white")
    ax.hist(halluc_mono_scores, bins=bins, alpha=0.6, color=COLORS[1],
            label=f"Hallucinated (n={len(halluc_mono_scores)})", density=True, edgecolor="white")
    ax.set_xlabel("Monotonicity Score")
    ax.set_ylabel("Density")
    ax.set_title("Monotonicity Score Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"figure_3_monotonicity_analysis.{ext}"))
    plt.close()
    print("  Figure 3 saved")


def figure_4_claim_types():
    """Figure 4: Per-claim-type performance."""
    # Collect claim type results
    all_types = {}
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for dataset_name in DATASETS:
            p = os.path.join(RESULTS_DIR, f"analysis_claim_types_{mshort}_{dataset_name}.json")
            if os.path.exists(p):
                data = load_json(p)
                for ctype, metrics in data.items():
                    if ctype not in all_types:
                        all_types[ctype] = {"speccheck": [], "selfcheck": []}
                    all_types[ctype]["speccheck"].append(metrics.get("speccheck_auc_pr", 0))
                    all_types[ctype]["selfcheck"].append(metrics.get("selfcheck_auc_pr", 0))

    if not all_types:
        print("  No claim type data for Figure 4")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    types = sorted(all_types.keys())
    x = np.arange(len(types))
    w = 0.35

    spec_means = [np.mean(all_types[t]["speccheck"]) for t in types]
    self_means = [np.mean(all_types[t]["selfcheck"]) for t in types]
    spec_stds = [np.std(all_types[t]["speccheck"]) for t in types]
    self_stds = [np.std(all_types[t]["selfcheck"]) for t in types]

    ax.bar(x - w/2, spec_means, w, yerr=spec_stds, label="SpecCheck (Ours)",
           color=COLORS[0], alpha=0.85, capsize=3)
    ax.bar(x + w/2, self_means, w, yerr=self_stds, label="SelfCheckGPT",
           color=COLORS[1], alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in types], fontsize=11)
    ax.set_ylabel("AUC-PR")
    ax.set_title("Detection Performance by Claim Type", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"figure_4_claim_types.{ext}"))
    plt.close()
    print("  Figure 4 saved")


def figure_5_ablations():
    """Figure 5: Ablation studies (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Ladder depth
    ax = axes[0, 0]
    all_k_results = {}
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for ds in DATASETS:
            p = os.path.join(RESULTS_DIR, f"ablation_ladder_depth_{mshort}_{ds}.json")
            if os.path.exists(p):
                data = load_json(p)
                for k_label, metrics in data.items():
                    if k_label not in all_k_results:
                        all_k_results[k_label] = []
                    all_k_results[k_label].append(metrics["auc_pr"])

    if all_k_results:
        ks = sorted(all_k_results.keys())
        means = [np.mean(all_k_results[k]) for k in ks]
        stds = [np.std(all_k_results[k]) for k in ks]
        ax.errorbar(range(len(ks)), means, yerr=stds, fmt="o-", color=COLORS[0],
                    linewidth=2, markersize=8, capsize=5)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks)
    ax.set_xlabel("Ladder Depth K")
    ax.set_ylabel("AUC-PR")
    ax.set_title("(a) Effect of Ladder Depth", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (b) Score variants
    ax = axes[0, 1]
    all_variants = {}
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for ds in DATASETS:
            p = os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{ds}.json")
            if os.path.exists(p):
                data = load_json(p)
                for variant, metrics in data.items():
                    if variant == "alpha_sweep":
                        continue
                    if variant not in all_variants:
                        all_variants[variant] = []
                    all_variants[variant].append(metrics["auc_pr"])

    if all_variants:
        variants = sorted(all_variants.keys())
        means = [np.mean(all_variants[v]) for v in variants]
        stds = [np.std(all_variants[v]) for v in variants]
        x = np.arange(len(variants))
        ax.bar(x, means, yerr=stds, color=COLORS[:len(variants)], alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
    ax.set_ylabel("AUC-PR")
    ax.set_title("(b) Score Variant Comparison", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # (c) Alpha sweep
    ax = axes[1, 0]
    all_alpha = {}
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for ds in DATASETS:
            p = os.path.join(RESULTS_DIR, f"ablation_score_variants_{mshort}_{ds}.json")
            if os.path.exists(p):
                data = load_json(p)
                if "alpha_sweep" in data:
                    for a_label, metrics in data["alpha_sweep"].items():
                        if a_label not in all_alpha:
                            all_alpha[a_label] = []
                        all_alpha[a_label].append(metrics["auc_pr"])

    if all_alpha:
        alphas = sorted(all_alpha.keys())
        means = [np.mean(all_alpha[a]) for a in alphas]
        stds = [np.std(all_alpha[a]) for a in alphas]
        ax.errorbar(range(len(alphas)), means, yerr=stds, fmt="o-",
                    color=COLORS[0], linewidth=2, markersize=8, capsize=5)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(alphas, fontsize=9)
    ax.set_xlabel("Alpha Value")
    ax.set_ylabel("AUC-PR")
    ax.set_title("(c) SpecCheck Score Alpha Sweep", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (d) Combination results
    ax = axes[1, 1]
    all_combos = {}
    for model_name in MODELS:
        mshort = get_model_short(model_name)
        for ds in DATASETS:
            p = os.path.join(RESULTS_DIR, f"analysis_combination_{mshort}_{ds}.json")
            if os.path.exists(p):
                data = load_json(p)
                for key, val in data.items():
                    if key == "feature_importance":
                        continue
                    auc = val.get("auc_pr_mean", val.get("auc_pr", None))
                    if auc is not None:
                        if key not in all_combos:
                            all_combos[key] = []
                        all_combos[key].append(auc)

    if all_combos:
        combos = sorted(all_combos.keys())
        means = [np.mean(all_combos[c]) for c in combos]
        stds = [np.std(all_combos[c]) for c in combos]
        x = np.arange(len(combos))
        colors = [METHOD_COLORS.get(c, COLORS[i % len(COLORS)]) for i, c in enumerate(combos)]
        ax.barh(x, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(x)
        ax.set_yticklabels([c.replace("_", " ").replace("+", "\n+") for c in combos], fontsize=8)
    ax.set_xlabel("AUC-PR")
    ax.set_title("(d) Method Combinations", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"figure_5_ablations.{ext}"))
    plt.close()
    print("  Figure 5 saved")


def figure_6_granularity_examples():
    """Figure 6: Example specificity ladders with confidence annotations."""
    # Find real examples from the data
    examples = []
    for model_name in MODELS[:1]:  # Use first model
        mshort = get_model_short(model_name)
        for dataset_name in DATASETS[:1]:
            scores_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset_name}.json")
            ladder_path = os.path.join(DATA_DIR, f"ladders_{mshort}_{dataset_name}.json")
            label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset_name}.json")

            if not all(os.path.exists(p) for p in [scores_path, ladder_path, label_path]):
                continue

            scores = load_json(scores_path)
            ladders = load_json(ladder_path)
            labeled = load_json(label_path)

            label_map = {c["claim_id"]: c["label"] for c in labeled}
            ladder_map = {l["claim_id"]: l for l in ladders}
            score_map = {s["claim_id"]: s for s in scores}

            # Find good examples: 2 true (monotonic) + 2 hallucinated (non-monotonic)
            for item in scores:
                cid = item["claim_id"]
                if cid not in label_map or cid not in ladder_map:
                    continue
                if len(examples) >= 4:
                    break
                label = label_map[cid]
                mono = item["monotonicity_score"]
                if label == 0 and mono >= 0.9 and sum(1 for e in examples if e["label"] == 0) < 2:
                    examples.append({
                        "claim_id": cid,
                        "label": label,
                        "ladder": ladder_map[cid],
                        "confidences": item["confidences"],
                        "monotonicity": mono,
                        "granularity_index": item.get("granularity_index", -1),
                    })
                elif label == 1 and mono < 0.8 and sum(1 for e in examples if e["label"] == 1) < 2:
                    examples.append({
                        "claim_id": cid,
                        "label": label,
                        "ladder": ladder_map[cid],
                        "confidences": item["confidences"],
                        "monotonicity": mono,
                        "granularity_index": item.get("granularity_index", -1),
                    })

    if len(examples) < 2:
        print("  Not enough examples for Figure 6")
        return

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3 * len(examples)))
    if len(examples) == 1:
        axes = [axes]

    for i, ex in enumerate(examples):
        ax = axes[i]
        confs = ex["confidences"][:4]
        levels = []
        for lev in ex["ladder"]["levels"][:4]:
            text = lev["text"][:60]
            if len(lev["text"]) > 60:
                text += "..."
            levels.append(f"L{lev['level']}: {text}")

        color = COLORS[0] if ex["label"] == 0 else COLORS[1]
        label_str = "FACTUAL" if ex["label"] == 0 else "HALLUCINATED"

        ax.barh(range(len(confs)), confs, color=color, alpha=0.7)
        ax.set_yticks(range(len(levels)))
        ax.set_yticklabels(levels, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence P(True)")
        ax.set_title(f"[{label_str}] Mono={ex['monotonicity']:.2f}", fontweight="bold", color=color)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)

        for j, c in enumerate(confs):
            ax.text(c + 0.01, j, f"{c:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"figure_6_granularity_examples.{ext}"))
    plt.close()
    print("  Figure 6 saved")


def supp_heatmap():
    """Supplementary: AUC-PR heatmap (models x datasets)."""
    eval_stats = load_eval_stats()
    if not eval_stats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    models = [get_model_short(m) for m in MODELS]

    for pi, method in enumerate(["speccheck", "selfcheck"]):
        ax = axes[pi]
        data = np.full((len(models), len(DATASETS)), np.nan)
        for mi, model in enumerate(models):
            for di, dataset in enumerate(DATASETS):
                key = f"{model}_{dataset}"
                if key in eval_stats and method in eval_stats[key].get("methods", {}):
                    data[mi, di] = eval_stats[key]["methods"][method]["auc_pr"]

        sns.heatmap(data, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=[DATASET_LABELS.get(d, d) for d in DATASETS],
                    yticklabels=[MODEL_LABELS.get(m, m) for m in models],
                    vmin=0.3, vmax=0.9)
        ax.set_title(METHOD_LABELS.get(method, method), fontweight="bold")

    plt.suptitle("AUC-PR Heatmap: SpecCheck vs SelfCheckGPT", fontweight="bold")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"supp_heatmap.{ext}"))
    plt.close()
    print("  Supplementary heatmap saved")


def supp_roc_curves():
    """Supplementary: ROC curves per dataset."""
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for di, dataset in enumerate(DATASETS):
        ax = axes[di]
        for model_name in MODELS:
            mshort = get_model_short(model_name)
            key = f"{mshort}_{dataset}"

            label_path = os.path.join(DATA_DIR, f"labeled_claims_{mshort}_{dataset}.json")
            spec_path = os.path.join(RESULTS_DIR, f"speccheck_scores_{mshort}_{dataset}.json")

            if not os.path.exists(label_path) or not os.path.exists(spec_path):
                continue

            labeled = load_json(label_path)
            scores = load_json(spec_path)
            label_map = {c["claim_id"]: c["label"] for c in labeled}
            score_map = {s["claim_id"]: s["speccheck_score"] for s in scores}

            common = sorted(set(label_map) & set(score_map))
            if len(common) < 10:
                continue

            y_true = [label_map[c] for c in common]
            y_score = [score_map[c] for c in common]

            if len(set(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = compute_auc_roc(y_true, y_score)
            ax.plot(fpr, tpr, linewidth=2,
                    label=f"{MODEL_LABELS.get(mshort, mshort)} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC: {DATASET_LABELS.get(dataset, dataset)}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIGURES_DIR, f"supp_roc_curves.{ext}"))
    plt.close()
    print("  Supplementary ROC curves saved")


def generate_latex_tables():
    """Generate LaTeX tables for the paper."""
    eval_stats = load_eval_stats()
    if not eval_stats:
        return

    models = [get_model_short(m) for m in MODELS]
    methods = ["speccheck", "selfcheck", "verbalized", "logprob", "random"]

    # Table 1: Main results
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hallucination detection results (AUC-PR / AUC-ROC). Best in \textbf{bold}.}",
        r"\label{tab:main_results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l" + "cc" * len(DATASETS) + "}",
        r"\toprule",
    ]
    header = "Method"
    for ds in DATASETS:
        header += f" & \\multicolumn{{2}}{{c}}{{{DATASET_LABELS.get(ds, ds)}}}"
    header += r" \\"
    lines.append(header)
    subheader = ""
    for _ in DATASETS:
        subheader += " & AUC-PR & AUC-ROC"
    subheader += r" \\"
    lines.append(subheader)
    lines.append(r"\midrule")

    for method in methods:
        row = METHOD_LABELS.get(method, method)
        for ds in DATASETS:
            pr_vals = []
            roc_vals = []
            for model in models:
                key = f"{model}_{ds}"
                if key in eval_stats and method in eval_stats[key].get("methods", {}):
                    pr_vals.append(eval_stats[key]["methods"][method]["auc_pr"])
                    roc_vals.append(eval_stats[key]["methods"][method]["auc_roc"])
            pr_mean = np.mean(pr_vals) if pr_vals else 0
            roc_mean = np.mean(roc_vals) if roc_vals else 0
            pr_std = np.std(pr_vals) if len(pr_vals) > 1 else 0
            roc_std = np.std(roc_vals) if len(roc_vals) > 1 else 0
            row += f" & {pr_mean:.3f}$\\pm${pr_std:.3f} & {roc_mean:.3f}$\\pm${roc_std:.3f}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])

    with open(os.path.join(FIGURES_DIR, "table_1_main_results.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  Table 1 saved")


def run_all_visualizations():
    """Generate all figures and tables."""
    print("\n" + "="*60)
    print("GENERATING FIGURES AND TABLES")
    print("="*60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    figure_1_method_overview()
    figure_2_main_results()
    figure_3_monotonicity_analysis()
    figure_4_claim_types()
    figure_5_ablations()
    figure_6_granularity_examples()
    supp_heatmap()
    supp_roc_curves()
    generate_latex_tables()

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    run_all_visualizations()
