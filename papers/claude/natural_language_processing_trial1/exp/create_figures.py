#!/usr/bin/env python3
"""Generate publication-quality figures for C2UD paper."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
EXP_DIR = WORKSPACE / "exp"
FIG_DIR = WORKSPACE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Color scheme
COLORS = {
    "C2UD_full": "#2196F3",
    "CRUX": "#FF9800",
    "SemEntropy": "#4CAF50",
    "TokenProb": "#9E9E9E",
    "Verbalized": "#E91E63",
    "SelfConsist": "#673AB7",
    "Axiomatic": "#795548",
    "C2UD_RS": "#90CAF9",
    "C2UD_CD": "#64B5F6",
    "C2UD_PA": "#42A5F5",
    "C2UD_RS_CD": "#1E88E5",
    "C2UD_RS_PA": "#1565C0",
    "C2UD_CD_PA": "#0D47A1",
}

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})


def load_results():
    """Load all results files."""
    results = {}
    for model in ["llama", "mistral", "phi3"]:
        path = EXP_DIR / f"{model}_results.json"
        if path.exists():
            with open(path) as f:
                results[model] = json.load(f)
    return results


def load_components(model):
    path = EXP_DIR / f"{model}_c2ud_components.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_failure_analysis(model):
    path = EXP_DIR / f"{model}_failure_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_intervention():
    path = EXP_DIR / "llama_intervention_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def table1_main_results(results):
    """Create main results table (CSV and LaTeX)."""
    methods = ["TokenProb", "Verbalized", "SemEntropy", "SelfConsist", "Axiomatic", "CRUX", "C2UD_full"]
    method_labels = ["Token Prob.", "Verbalized", "Sem. Entropy", "Self-Consist.", "Axiomatic", "CRUX", "C2UD (Ours)"]
    datasets = ["nq", "triviaqa", "popqa"]
    ds_labels = ["NQ", "TriviaQA", "PopQA"]
    models = [m for m in ["llama", "mistral", "phi3"] if m in results]

    # CSV
    with open(FIG_DIR / "table1_main_results.csv", "w") as f:
        header = "Method," + ",".join([f"{dl} AUROC,{dl} AUPRC,{dl} Cov@90" for dl in ds_labels])
        f.write(header + "\n")
        for method, label in zip(methods, method_labels):
            row = [label]
            for ds in datasets:
                aurocs, auprcs, covs = [], [], []
                for model in models:
                    agg = results[model].get("aggregated", {})
                    if ds in agg and method in agg[ds]:
                        m = agg[ds][method]
                        aurocs.append(m.get("auroc", {}).get("mean", 0))
                        auprcs.append(m.get("auprc", {}).get("mean", 0))
                        covs.append(m.get("coverage_90", {}).get("mean", 0))
                if aurocs:
                    row.extend([f"{np.mean(aurocs):.3f}", f"{np.mean(auprcs):.3f}", f"{np.mean(covs):.3f}"])
                else:
                    row.extend(["--", "--", "--"])
            f.write(",".join(row) + "\n")

    # LaTeX
    with open(FIG_DIR / "table1_main_results.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Main Results: Selective prediction metrics averaged across models.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\resizebox{\\textwidth}{!}{\n")
        ncols = 1 + 3 * len(datasets)
        f.write(f"\\begin{{tabular}}{{l{'ccc' * len(datasets)}}}\n\\toprule\n")
        f.write("& " + " & ".join([f"\\multicolumn{{3}}{{c}}{{{dl}}}" for dl in ds_labels]) + " \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n")
        f.write("Method & " + " & ".join(["AUROC & AUPRC & Cov@90"] * len(datasets)) + " \\\\\n\\midrule\n")

        # Find best per column
        best_vals = {}
        for ds_i, ds in enumerate(datasets):
            for metric_i, metric in enumerate(["auroc", "auprc", "coverage_90"]):
                best = -1
                for method in methods:
                    vals = []
                    for model in models:
                        agg = results[model].get("aggregated", {})
                        if ds in agg and method in agg[ds]:
                            v = agg[ds][method].get(metric, {}).get("mean", 0)
                            vals.append(v)
                    if vals:
                        avg = np.mean(vals)
                        if avg > best:
                            best = avg
                best_vals[(ds, metric)] = best

        for method, label in zip(methods, method_labels):
            row_parts = [label]
            for ds in datasets:
                for metric in ["auroc", "auprc", "coverage_90"]:
                    vals = []
                    stds = []
                    for model in models:
                        agg = results[model].get("aggregated", {})
                        if ds in agg and method in agg[ds]:
                            v = agg[ds][method].get(metric, {}).get("mean", 0)
                            s = agg[ds][method].get(metric, {}).get("std", 0)
                            vals.append(v)
                            stds.append(s)
                    if vals:
                        avg = np.mean(vals)
                        std = np.mean(stds)
                        cell = f"{avg:.3f}$\\pm${std:.3f}"
                        if abs(avg - best_vals[(ds, metric)]) < 0.001:
                            cell = f"\\textbf{{{cell}}}"
                        row_parts.append(cell)
                    else:
                        row_parts.append("--")
            f.write(" & ".join(row_parts) + " \\\\\n")
            if label == "Axiomatic":
                f.write("\\midrule\n")

        f.write("\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")

    print("Table 1 saved.")


def table2_ablation(results):
    """Create ablation study table."""
    variants = ["C2UD_RS", "C2UD_CD", "C2UD_PA", "C2UD_RS_CD", "C2UD_RS_PA", "C2UD_CD_PA", "C2UD_full"]
    labels = ["RS only", "CD only", "PA only", "RS+CD", "RS+PA", "CD+PA", "Full (RS+CD+PA)"]
    datasets = ["nq", "triviaqa", "popqa"]
    ds_labels = ["NQ", "TriviaQA", "PopQA"]
    models = [m for m in ["llama", "mistral", "phi3"] if m in results]

    with open(FIG_DIR / "table2_ablation.csv", "w") as f:
        f.write("Variant," + ",".join([f"{dl} AUROC" for dl in ds_labels]) + ",Avg AUROC\n")
        for var, label in zip(variants, labels):
            row = [label]
            all_aurocs = []
            for ds in datasets:
                vals = []
                for model in models:
                    agg = results[model].get("aggregated", {})
                    if ds in agg and var in agg[ds]:
                        v = agg[ds][var].get("auroc", {}).get("mean", 0)
                        vals.append(v)
                if vals:
                    avg = np.mean(vals)
                    row.append(f"{avg:.3f}")
                    all_aurocs.append(avg)
                else:
                    row.append("--")
            row.append(f"{np.mean(all_aurocs):.3f}" if all_aurocs else "--")
            f.write(",".join(row) + "\n")

    # LaTeX version
    with open(FIG_DIR / "table2_ablation.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Ablation: AUROC by C2UD variant (averaged across models).}\n")
        f.write("\\label{tab:ablation}\n")
        f.write(f"\\begin{{tabular}}{{l{'c' * (len(datasets) + 1)}}}\n\\toprule\n")
        f.write("Variant & " + " & ".join(ds_labels) + " & Avg \\\\\n\\midrule\n")
        for var, label in zip(variants, labels):
            row_parts = [label]
            all_aurocs = []
            for ds in datasets:
                vals = []
                for model in models:
                    agg = results[model].get("aggregated", {})
                    if ds in agg and var in agg[ds]:
                        v = agg[ds][var].get("auroc", {}).get("mean", 0)
                        vals.append(v)
                if vals:
                    avg = np.mean(vals)
                    row_parts.append(f"{avg:.3f}")
                    all_aurocs.append(avg)
                else:
                    row_parts.append("--")
            row_parts.append(f"{np.mean(all_aurocs):.3f}" if all_aurocs else "--")
            line = " & ".join(row_parts) + " \\\\\n"
            if label == "PA only":
                line += "\\midrule\n"
            f.write(line)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print("Table 2 saved.")


def figure2_calibration(results):
    """Calibration reliability diagrams."""
    model = "llama"
    if model not in results:
        model = list(results.keys())[0] if results else None
    if not model:
        return

    # Load per-seed data for calibration
    per_seed = results[model].get("per_seed", {})
    datasets = ["nq", "triviaqa", "popqa"]
    ds_labels = ["NQ", "TriviaQA", "PopQA"]
    methods = ["C2UD_full", "CRUX", "SemEntropy", "TokenProb"]
    method_labels = ["C2UD (Ours)", "CRUX", "Sem. Entropy", "Token Prob."]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_i, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
        ax = axes[ax_i]
        for method, mlabel in zip(methods, method_labels):
            agg = results[model].get("aggregated", {})
            if ds in agg and method in agg[ds]:
                ece = agg[ds][method].get("ece", {}).get("mean", 0)
                brier = agg[ds][method].get("brier", {}).get("mean", 0)
                ax.bar(mlabel, ece, color=COLORS.get(method, "#999"),
                       alpha=0.8, label=f"{mlabel} (ECE={ece:.3f})")

        ax.set_title(ds_label)
        ax.set_ylabel("ECE" if ax_i == 0 else "")
        ax.set_ylim(0, 0.3)
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle(f"Expected Calibration Error ({model.title()})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_calibration.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "figure2_calibration.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 2 saved.")


def figure3_failure_modes():
    """Failure mode analysis bar chart."""
    model = "llama"
    fa = load_failure_analysis(model)
    if not fa:
        for m in ["mistral", "phi3"]:
            fa = load_failure_analysis(m)
            if fa:
                model = m
                break
    if not fa:
        print("No failure analysis data found.")
        return

    stats = fa.get("failure_stats", {})
    types = ["retrieval_failure", "grounding_failure", "parametric_override"]
    type_labels = ["Retrieval\nFailure", "Grounding\nFailure", "Parametric\nOverride"]
    components = ["RS", "CD", "PA"]
    comp_colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(types))
    width = 0.25

    for i, (comp, color) in enumerate(zip(components, comp_colors)):
        means = [stats.get(t, {}).get(f"{comp}_mean", 0) for t in types]
        stds = [stats.get(t, {}).get(f"{comp}_std", 0) for t in types]
        counts = [stats.get(t, {}).get("count", 0) for t in types]
        ax.bar(x + i * width, means, width, yerr=stds, label=comp,
               color=color, alpha=0.85, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(type_labels)
    ax.set_ylabel("Component Value")
    ax.set_title(f"C2UD Components by Failure Mode ({model.title()})")
    ax.legend()

    # Add counts
    for j, t in enumerate(types):
        n = stats.get(t, {}).get("count", 0)
        ax.text(j + width, -0.05, f"n={n}", ha='center', fontsize=9, style='italic',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_failure_modes.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "figure3_failure_modes.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 3 saved.")


def figure4_components():
    """2D scatter of RS vs CD colored by correctness."""
    model = "llama"
    comps = load_components(model)
    if not comps:
        for m in ["mistral", "phi3"]:
            comps = load_components(m)
            if comps:
                model = m
                break
    if not comps:
        print("No component data found.")
        return

    nq_comps = {k: v for k, v in comps.items() if v["dataset"] == "nq"}

    fig, ax = plt.subplots(figsize=(7, 6))
    correct_rs = [v["RS"] for v in nq_comps.values() if v["correct"]]
    correct_cd = [v["CD"] for v in nq_comps.values() if v["correct"]]
    incorrect_rs = [v["RS"] for v in nq_comps.values() if not v["correct"]]
    incorrect_cd = [v["CD"] for v in nq_comps.values() if not v["correct"]]

    ax.scatter(incorrect_rs, incorrect_cd, c='#EF5350', alpha=0.4, s=20, label='Incorrect', zorder=1)
    ax.scatter(correct_rs, correct_cd, c='#2196F3', alpha=0.4, s=20, label='Correct', zorder=2)

    ax.set_xlabel("Retrieval Sensitivity (RS)")
    ax.set_ylabel("Context Discrimination (CD)")
    ax.set_title(f"RS vs CD on NQ ({model.title()})")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='CD=0.5 threshold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure4_components.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "figure4_components.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 4 saved.")


def figure5_coverage_accuracy(results):
    """Coverage vs accuracy curves."""
    model = "llama"
    if model not in results:
        model = list(results.keys())[0] if results else None
    if not model:
        return

    # We need raw scores to compute coverage curves
    # For now, use the aggregated coverage@90 values to create a summary bar chart
    datasets = ["nq", "triviaqa", "popqa"]
    ds_labels = ["NQ", "TriviaQA", "PopQA"]
    methods = ["C2UD_full", "CRUX", "SemEntropy", "TokenProb"]
    method_labels = ["C2UD (Ours)", "CRUX", "Sem. Entropy", "Token Prob."]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax_i, (ds, ds_label) in enumerate(zip(datasets, ds_labels)):
        ax = axes[ax_i]
        aurocs = []
        ece_vals = []
        for method, mlabel in zip(methods, method_labels):
            agg = results[model].get("aggregated", {})
            if ds in agg and method in agg[ds]:
                auroc = agg[ds][method].get("auroc", {}).get("mean", 0)
                std = agg[ds][method].get("auroc", {}).get("std", 0)
                aurocs.append((mlabel, auroc, std, COLORS.get(method, "#999")))

        if aurocs:
            labels, vals, stds, colors = zip(*aurocs)
            bars = ax.bar(labels, vals, yerr=stds, color=colors, alpha=0.85, capsize=3)
            ax.set_ylim(0.4, 1.0)

        ax.set_title(ds_label)
        ax.set_ylabel("AUROC" if ax_i == 0 else "")
        ax.tick_params(axis='x', rotation=30)

    plt.suptitle(f"AUROC Comparison ({model.title()})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure5_auroc_comparison.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "figure5_auroc_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 5 saved.")


def figure6_intervention():
    """Intervention results bar chart."""
    interv = load_intervention()
    if not interv:
        print("No intervention data found.")
        return

    strategies = interv.get("strategies", {})
    strat_names = ["no_intervention", "uniform_abstention", "uniform_reretrieval", "c2ud_intervene"]
    strat_labels = ["No Intervention", "Uniform\nAbstention", "Uniform\nRe-retrieval", "C2UD-Intervene"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Accuracy
    accs = [strategies.get(s, {}).get("accuracy", 0) for s in strat_names]
    colors = ["#9E9E9E", "#FF9800", "#4CAF50", "#2196F3"]
    ax1.bar(strat_labels, accs, color=colors, alpha=0.85)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy on Answered Queries")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=15)

    # Coverage
    covs = [strategies.get(s, {}).get("coverage", 0) for s in strat_names]
    ax2.bar(strat_labels, covs, color=colors, alpha=0.85)
    ax2.set_ylabel("Coverage")
    ax2.set_title("Query Coverage")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=15)

    plt.suptitle("Targeted Intervention Results (Llama/NQ)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure6_intervention.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "figure6_intervention.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Figure 6 saved.")


def figure_model_comparison(results):
    """AUROC comparison across models."""
    models = [m for m in ["llama", "mistral", "phi3"] if m in results]
    if len(models) < 2:
        return

    methods = ["TokenProb", "SemEntropy", "CRUX", "C2UD_full"]
    method_labels = ["Token Prob.", "Sem. Entropy", "CRUX", "C2UD (Ours)"]
    model_labels = {"llama": "Llama 3.1 8B", "mistral": "Mistral 7B", "phi3": "Phi-3 Mini"}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    width = 0.2

    for i, (method, mlabel) in enumerate(zip(methods, method_labels)):
        avg_aurocs = []
        for model in models:
            agg = results[model].get("aggregated", {})
            ds_aurocs = []
            for ds in ["nq", "triviaqa", "popqa"]:
                if ds in agg and method in agg[ds]:
                    ds_aurocs.append(agg[ds][method].get("auroc", {}).get("mean", 0))
            avg_aurocs.append(np.mean(ds_aurocs) if ds_aurocs else 0)

        ax.bar(x + i * width, avg_aurocs, width, label=mlabel,
               color=COLORS.get(method, "#999"), alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([model_labels.get(m, m) for m in models])
    ax.set_ylabel("Average AUROC")
    ax.set_title("AUROC Across Models (averaged over datasets)")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_model_comparison.pdf", bbox_inches='tight')
    plt.savefig(FIG_DIR / "figure_model_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Model comparison figure saved.")


def main():
    print("Loading results...")
    results = load_results()
    if not results:
        print("No results found! Run experiments first.")
        return

    print(f"Found results for models: {list(results.keys())}")

    table1_main_results(results)
    table2_ablation(results)
    figure2_calibration(results)
    figure3_failure_modes()
    figure4_components()
    figure5_coverage_accuracy(results)
    figure6_intervention()
    figure_model_comparison(results)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
