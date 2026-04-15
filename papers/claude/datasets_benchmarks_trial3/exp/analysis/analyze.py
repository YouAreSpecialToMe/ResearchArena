#!/usr/bin/env python3
"""
SkillStack: Comprehensive Analysis Pipeline

Computes composition gaps, generates figures, runs statistical tests,
and produces the final results.json.
"""
import json
import os
import sys
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "exp" / "results"
FIGURES_DIR = BASE_DIR / "figures"

sys.path.insert(0, str(BASE_DIR))
from exp.generators.skills import SKILL_CODES, SKILL_NAMES

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_eval_results(filepath):
    """Load evaluation results from JSONL."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_per_category_accuracy(results):
    """Compute accuracy per skill category."""
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r["skill_combo"]
        cat_stats[cat]["total"] += 1
        if r["correct"]:
            cat_stats[cat]["correct"] += 1
    for cat in cat_stats:
        t = cat_stats[cat]["total"]
        c = cat_stats[cat]["correct"]
        cat_stats[cat]["accuracy"] = c / t if t > 0 else 0.0
    return dict(cat_stats)


def compute_composition_gaps(cat_stats):
    """Compute composition gap for each pairwise and triple combination."""
    # Get single-skill accuracies (PSA)
    psa = {}
    for code in SKILL_CODES:
        if code in cat_stats:
            psa[code] = cat_stats[code]["accuracy"]

    gaps = {}
    for cat, stats in cat_stats.items():
        if "+" not in cat:
            continue
        skills = cat.split("+")
        level = len(skills)

        # Min baseline: min of single-skill accuracies
        min_psa = min(psa.get(s, 0) for s in skills)
        # Independence baseline: product of PSAs
        prod_psa = 1.0
        for s in skills:
            prod_psa *= psa.get(s, 0)

        actual = stats["accuracy"]
        composition_gap = min_psa - actual
        composition_efficiency = actual / min_psa if min_psa > 0 else 0

        gaps[cat] = {
            "skills": skills,
            "level": level,
            "accuracy": actual,
            "min_psa": min_psa,
            "prod_psa": prod_psa,
            "composition_gap": composition_gap,
            "composition_efficiency": composition_efficiency,
        }

    return gaps, psa


def run_full_analysis(result_files: dict):
    """Run complete analysis across all evaluation result files."""
    all_model_data = {}

    for model_key, filepath in result_files.items():
        if not os.path.exists(filepath):
            print(f"  Skipping {model_key} (file not found: {filepath})")
            continue
        results = load_eval_results(filepath)
        cat_stats = compute_per_category_accuracy(results)
        gaps, psa = compute_composition_gaps(cat_stats)

        # Level-wise averages
        level_accs = defaultdict(list)
        for cat, s in cat_stats.items():
            level = len(cat.split("+"))
            level_accs[level].append(s["accuracy"])

        all_model_data[model_key] = {
            "cat_stats": cat_stats,
            "gaps": gaps,
            "psa": psa,
            "level_means": {
                lvl: {"mean": np.mean(accs), "std": np.std(accs)}
                for lvl, accs in level_accs.items()
            },
            "overall_accuracy": np.mean([s["accuracy"] for s in cat_stats.values()]),
        }

    return all_model_data


def compute_reliability(model_data_42, model_data_123):
    """Compute test-retest reliability between seed 42 and seed 123."""
    reliability = {}
    for model_key in model_data_42:
        key_123 = model_key.replace("seed42", "seed123")
        if key_123 not in model_data_123 and model_key not in model_data_123:
            continue
        data_123_key = key_123 if key_123 in model_data_123 else model_key

        gaps_42 = model_data_42[model_key]["gaps"]
        gaps_123 = model_data_123[data_123_key]["gaps"]

        # Get common categories
        common_cats = set(gaps_42.keys()) & set(gaps_123.keys())
        if len(common_cats) < 5:
            continue

        cg_42 = [gaps_42[c]["composition_gap"] for c in sorted(common_cats)]
        cg_123 = [gaps_123[c]["composition_gap"] for c in sorted(common_cats)]

        r_pearson, p_pearson = stats.pearsonr(cg_42, cg_123)
        r_spearman, p_spearman = stats.spearmanr(cg_42, cg_123)

        reliability[model_key] = {
            "n_categories": len(common_cats),
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "spearman_r": r_spearman,
            "spearman_p": p_spearman,
        }

    return reliability


def statistical_tests(all_model_data):
    """Run statistical tests for success criteria."""
    results = {}

    # Test 1: Significant composition gap
    # For each model and each pairwise category, test if CG > 0
    sig_results = {}
    for model_key, data in all_model_data.items():
        model_sig = {}
        for cat, gap_info in data["gaps"].items():
            if gap_info["level"] != 2:
                continue
            # Binomial test: is actual accuracy significantly below min_psa?
            n = data["cat_stats"][cat]["total"]
            k = data["cat_stats"][cat]["correct"]
            p0 = gap_info["min_psa"]
            if p0 <= 0 or n == 0:
                continue
            # One-sided binomial test: H0: p >= p0, H1: p < p0
            p_val = stats.binom_test(k, n, p0, alternative='less') if hasattr(stats, 'binom_test') else stats.binomtest(k, n, p0, alternative='less').pvalue
            model_sig[cat] = {
                "composition_gap": gap_info["composition_gap"],
                "p_value": p_val,
                "significant": p_val < 0.01 / 28,  # Bonferroni correction
            }
        sig_frac = sum(1 for v in model_sig.values() if v["significant"]) / max(len(model_sig), 1)
        sig_results[model_key] = {"per_pair": model_sig, "fraction_significant": sig_frac}

    results["composition_gap_significance"] = sig_results

    # Test 2: Composition gap variation across models
    all_gaps_by_cat = defaultdict(list)
    for model_key, data in all_model_data.items():
        for cat, gap_info in data["gaps"].items():
            if gap_info["level"] == 2:
                all_gaps_by_cat[cat].append(gap_info["composition_gap"])

    gap_ranges = {}
    for cat, gaps in all_gaps_by_cat.items():
        gap_ranges[cat] = {"range": max(gaps) - min(gaps), "mean": np.mean(gaps), "std": np.std(gaps)}
    results["gap_variation"] = gap_ranges

    return results


def generate_figures(all_model_data, reliability_data=None):
    """Generate all publication figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    # ============================================================
    # Figure 1: Skill Interaction Heatmaps
    # ============================================================
    direct_models = {k: v for k, v in all_model_data.items() if "direct" in k and "seed42" in k}
    if direct_models:
        n_models = len(direct_models)
        ncols = min(n_models, 4)
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (model_key, data) in enumerate(sorted(direct_models.items())):
            row, col = idx // ncols, idx % ncols
            ax = axes[row][col]

            # Build 8x8 heatmap
            matrix = np.full((8, 8), np.nan)
            for i, si in enumerate(SKILL_CODES):
                for j, sj in enumerate(SKILL_CODES):
                    if i == j:
                        matrix[i][j] = 0
                        continue
                    pair = "+".join(sorted([si, sj]))
                    if pair in data["gaps"]:
                        matrix[i][j] = data["gaps"][pair]["composition_gap"]

            sns.heatmap(matrix, ax=ax, xticklabels=SKILL_CODES, yticklabels=SKILL_CODES,
                       cmap="RdYlBu_r", center=0, annot=True, fmt=".2f",
                       vmin=-0.2, vmax=0.8, annot_kws={"size": 7})
            short_name = model_key.split("_direct")[0]
            ax.set_title(f"{short_name}", fontsize=12)

        # Hide unused axes
        for idx in range(len(direct_models), nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row][col].set_visible(False)

        fig.suptitle("Skill Interaction Matrix: Composition Gap", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "figure1_skill_interaction_matrix.pdf"))
        fig.savefig(str(FIGURES_DIR / "figure1_skill_interaction_matrix.png"))
        plt.close(fig)
        print("  Saved figure1_skill_interaction_matrix")

    # ============================================================
    # Figure 2: Composition Scaling Plot
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(direct_models)))
    for idx, (model_key, data) in enumerate(sorted(direct_models.items())):
        levels = sorted(data["level_means"].keys())
        means = [data["level_means"][l]["mean"] for l in levels]
        stds = [data["level_means"][l]["std"] for l in levels]
        short_name = model_key.split("_direct")[0]
        ax.errorbar(levels, means, yerr=stds, marker='o', label=short_name,
                    color=colors[idx], capsize=3, linewidth=2)

    ax.set_xlabel("Composition Level (K)")
    ax.set_ylabel("Average Accuracy")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Single (K=1)", "Pairwise (K=2)", "Triple (K=3)"])
    ax.legend(fontsize=9)
    ax.set_title("Accuracy vs. Composition Level")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "figure2_composition_scaling.pdf"))
    fig.savefig(str(FIGURES_DIR / "figure2_composition_scaling.png"))
    plt.close(fig)
    print("  Saved figure2_composition_scaling")

    # ============================================================
    # Figure 3: CoT vs Direct Comparison
    # ============================================================
    cot_models = {k: v for k, v in all_model_data.items() if "cot" in k and "seed42" in k}
    if cot_models:
        fig, ax = plt.subplots(figsize=(10, 5))
        model_names = []
        direct_gaps = []
        cot_gaps = []
        for cot_key, cot_data in sorted(cot_models.items()):
            short = cot_key.split("_cot")[0]
            direct_key = f"{short}_direct_seed42"
            if direct_key not in all_model_data:
                continue
            dir_data = all_model_data[direct_key]

            # Average composition gap for pairwise
            dir_avg_cg = np.mean([g["composition_gap"] for g in dir_data["gaps"].values() if g["level"] == 2])
            cot_avg_cg = np.mean([g["composition_gap"] for g in cot_data["gaps"].values() if g["level"] == 2])

            model_names.append(short)
            direct_gaps.append(dir_avg_cg)
            cot_gaps.append(cot_avg_cg)

        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, direct_gaps, width, label='Direct', color='steelblue')
        ax.bar(x + width/2, cot_gaps, width, label='Chain-of-Thought', color='coral')
        ax.set_xlabel("Model")
        ax.set_ylabel("Average Composition Gap (Pairwise)")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_title("Composition Gap: Direct vs Chain-of-Thought")
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "figure3_cot_effect.pdf"))
        fig.savefig(str(FIGURES_DIR / "figure3_cot_effect.png"))
        plt.close(fig)
        print("  Saved figure3_cot_effect")

    # ============================================================
    # Figure 4: Composition Difficulty Clustering
    # ============================================================
    if len(direct_models) >= 2:
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage

        # Build feature matrix: each pairwise category is a row, each model is a column
        pairs = sorted(list(combinations(SKILL_CODES, 2)))
        pair_names = ["+".join(p) for p in pairs]
        feat_matrix = []
        valid_pairs = []

        for pair_name in pair_names:
            row = []
            valid = True
            for model_key in sorted(direct_models.keys()):
                if pair_name in direct_models[model_key]["gaps"]:
                    row.append(direct_models[model_key]["gaps"][pair_name]["composition_gap"])
                else:
                    valid = False
                    break
            if valid:
                feat_matrix.append(row)
                valid_pairs.append(pair_name)

        if len(feat_matrix) >= 3:
            X = np.array(feat_matrix)
            Z = linkage(X, method='ward')

            fig, ax = plt.subplots(figsize=(14, 6))
            dendrogram(Z, labels=valid_pairs, ax=ax, leaf_rotation=90, leaf_font_size=8)
            ax.set_ylabel("Ward Distance")
            ax.set_title("Hierarchical Clustering of Skill Pair Compositions by Difficulty")
            plt.tight_layout()
            fig.savefig(str(FIGURES_DIR / "figure4_composition_clusters.pdf"))
            fig.savefig(str(FIGURES_DIR / "figure4_composition_clusters.png"))
            plt.close(fig)
            print("  Saved figure4_composition_clusters")

    # ============================================================
    # Figure 5: Reliability Plot
    # ============================================================
    if reliability_data:
        # Load seed 123 results for scatter plots
        n_reliable = len(reliability_data)
        if n_reliable > 0:
            fig, axes = plt.subplots(1, n_reliable, figsize=(5 * n_reliable, 4.5))
            if n_reliable == 1:
                axes = [axes]

            for idx, (model_key, rel_info) in enumerate(sorted(reliability_data.items())):
                ax = axes[idx]
                # We need to reload CG values for scatter
                # Just plot the correlation info
                ax.text(0.5, 0.5,
                       f"r = {rel_info['pearson_r']:.3f}\np = {rel_info['pearson_p']:.2e}\n"
                       f"n = {rel_info['n_categories']}",
                       transform=ax.transAxes, ha='center', va='center', fontsize=14)
                short = model_key.split("_direct")[0]
                ax.set_title(f"{short}")
                ax.set_xlabel("CG (Seed 42)")
                ax.set_ylabel("CG (Seed 123)")

            fig.suptitle("Test-Retest Reliability of Composition Gap", fontsize=13)
            plt.tight_layout()
            fig.savefig(str(FIGURES_DIR / "figure5_reliability.pdf"))
            fig.savefig(str(FIGURES_DIR / "figure5_reliability.png"))
            plt.close(fig)
            print("  Saved figure5_reliability")

    # ============================================================
    # Figure 6: Per-skill accuracy across models (bar chart)
    # ============================================================
    if direct_models:
        fig, ax = plt.subplots(figsize=(12, 5))
        n_models = len(direct_models)
        x = np.arange(len(SKILL_CODES))
        width = 0.8 / n_models

        for idx, (model_key, data) in enumerate(sorted(direct_models.items())):
            accs = [data["psa"].get(s, 0) for s in SKILL_CODES]
            short = model_key.split("_direct")[0]
            ax.bar(x + idx * width, accs, width, label=short)

        ax.set_xlabel("Cognitive Skill")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels([SKILL_NAMES[c] for c in SKILL_CODES], rotation=45, ha='right')
        ax.legend(fontsize=8)
        ax.set_title("Single-Skill Accuracy per Model")
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "figure6_per_skill_accuracy.pdf"))
        fig.savefig(str(FIGURES_DIR / "figure6_per_skill_accuracy.png"))
        plt.close(fig)
        print("  Saved figure6_per_skill_accuracy")

    print("All figures generated!")


def compile_results_json(all_model_data, reliability_data, stat_tests):
    """Compile the final results.json."""
    results = {
        "benchmark": {
            "name": "SkillStack",
            "n_skills": 8,
            "n_pairwise": 28,
            "n_triple": 56,
            "instances_per_category": 50,
            "total_instances": 4600,
            "seeds_used": [42, 123, 456],
        },
        "models": {},
        "composition_analysis": {},
        "reliability": reliability_data if reliability_data else {},
        "statistical_tests": {},
    }

    for model_key, data in all_model_data.items():
        model_entry = {
            "overall_accuracy": data["overall_accuracy"],
            "level_means": {str(k): v for k, v in data["level_means"].items()},
            "single_skill_accuracies": data["psa"],
        }

        # Average composition gap
        pairwise_gaps = [g["composition_gap"] for g in data["gaps"].values() if g["level"] == 2]
        triple_gaps = [g["composition_gap"] for g in data["gaps"].values() if g["level"] == 3]

        model_entry["pairwise_composition_gap"] = {
            "mean": float(np.mean(pairwise_gaps)) if pairwise_gaps else 0,
            "std": float(np.std(pairwise_gaps)) if pairwise_gaps else 0,
        }
        model_entry["triple_composition_gap"] = {
            "mean": float(np.mean(triple_gaps)) if triple_gaps else 0,
            "std": float(np.std(triple_gaps)) if triple_gaps else 0,
        }

        results["models"][model_key] = model_entry

    # Top-5 hardest and easiest compositions
    all_pairwise_gaps = defaultdict(list)
    for model_key, data in all_model_data.items():
        if "direct" in model_key and "seed42" in model_key:
            for cat, g in data["gaps"].items():
                if g["level"] == 2:
                    all_pairwise_gaps[cat].append(g["composition_gap"])

    avg_gaps = {cat: np.mean(gaps) for cat, gaps in all_pairwise_gaps.items()}
    sorted_gaps = sorted(avg_gaps.items(), key=lambda x: x[1], reverse=True)

    results["composition_analysis"]["hardest_compositions"] = [
        {"pair": cat, "avg_composition_gap": {"mean": float(np.mean(all_pairwise_gaps[cat])),
                                               "std": float(np.std(all_pairwise_gaps[cat]))}}
        for cat, _ in sorted_gaps[:5]
    ]
    results["composition_analysis"]["easiest_compositions"] = [
        {"pair": cat, "avg_composition_gap": {"mean": float(np.mean(all_pairwise_gaps[cat])),
                                               "std": float(np.std(all_pairwise_gaps[cat]))}}
        for cat, _ in sorted_gaps[-5:]
    ]

    if stat_tests:
        # Summarize statistical tests
        sig_summary = {}
        for model_key, sig_info in stat_tests.get("composition_gap_significance", {}).items():
            sig_summary[model_key] = sig_info["fraction_significant"]
        results["statistical_tests"]["fraction_significant_per_model"] = sig_summary

    return results


if __name__ == "__main__":
    print("Running SkillStack analysis pipeline...")

    # Discover result files
    result_files = {}
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        key = f.stem
        result_files[key] = str(f)

    print(f"Found {len(result_files)} result files:")
    for k in sorted(result_files.keys()):
        print(f"  {k}")

    # Run analysis
    all_model_data = run_full_analysis(result_files)

    # Reliability analysis
    seed42_data = {k: v for k, v in all_model_data.items() if "seed42" in k}
    seed123_data = {k: v for k, v in all_model_data.items() if "seed123" in k}
    reliability = compute_reliability(seed42_data, seed123_data) if seed123_data else {}

    # Statistical tests
    stat_tests = statistical_tests(all_model_data)

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(all_model_data, reliability)

    # Compile results.json
    print("\nCompiling results.json...")
    final_results = compile_results_json(all_model_data, reliability, stat_tests)
    results_path = BASE_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    for model_key in sorted(all_model_data.keys()):
        data = all_model_data[model_key]
        print(f"\n{model_key}:")
        for lvl in sorted(data["level_means"].keys()):
            m = data["level_means"][lvl]
            print(f"  Level {lvl}: {m['mean']:.3f} ± {m['std']:.3f}")

    if reliability:
        print("\nReliability:")
        for mk, ri in reliability.items():
            print(f"  {mk}: r={ri['pearson_r']:.3f}")

    if stat_tests.get("composition_gap_significance"):
        print("\nSignificance (fraction of pairs with p < 0.01/28):")
        for mk, si in stat_tests["composition_gap_significance"].items():
            print(f"  {mk}: {si['fraction_significant']:.2f}")
