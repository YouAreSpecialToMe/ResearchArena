"""Generate all figures and tables for the paper."""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "exp", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load aggregated results
with open(os.path.join(RESULTS_DIR, "aggregated_results.json")) as f:
    agg = json.load(f)
stats = agg["statistics"]

# Style
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
COLORS = sns.color_palette("Set2", 10)


def fig_main_comparison():
    """Figure 1: Main results comparison bar chart."""
    methods = ["blastp", "flat_supcon", "joint_hierarchical", "currec"]
    labels = ["BLASTp", "Flat SupCon", "Joint Hier.", "CurrEC (Ours)"]
    colors = [COLORS[7], COLORS[0], COLORS[1], COLORS[2]]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    for ax, benchmark, title in zip(axes, ["new392", "price149"], ["New-392", "Price-149"]):
        means = []
        stds = []
        for m in methods:
            s = stats.get(m, {}).get(benchmark, {}).get("macro_f1", {})
            means.append(s.get("mean", 0))
            stds.append(s.get("std", 0))

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Macro-F1")
        ax.set_title(title)
        ax.set_ylim(0, max(means) * 1.25)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_main_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure1_main_comparison.png"), bbox_inches="tight")
    plt.close()
    print("Saved figure1_main_comparison")


def fig_curriculum_order():
    """Figure 2: Curriculum order comparison."""
    methods = ["currec", "reverse_curriculum", "random_order", "no_consistency"]
    labels = ["Coarse-to-Fine\n(CurrEC)", "Fine-to-Coarse\n(Reverse)", "Random\nOrder", "No Consistency\n(lambda=0)"]
    colors = [COLORS[2], COLORS[3], COLORS[4], COLORS[5]]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    means = []
    stds = []
    for m in methods:
        s = stats.get(m, {}).get("new392", {}).get("macro_f1", {})
        means.append(s.get("mean", 0))
        stds.append(s.get("std", 0))

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro-F1 (New-392)")
    ax.set_title("Effect of Curriculum Order")
    ax.set_ylim(0, max(means) * 1.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_curriculum_order.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure2_curriculum_order.png"), bbox_inches="tight")
    plt.close()
    print("Saved figure2_curriculum_order")


def fig_phase_progression():
    """Figure 3: Phase progression for CurrEC."""
    # Load CurrEC seed 42 results with phase data
    currec_path = os.path.join(RESULTS_DIR, "currec_seed42_results.json")
    with open(currec_path) as f:
        currec = json.load(f)

    phase_results = currec.get("phase_results", [])
    if not phase_results:
        print("No phase progression data available, skipping figure3")
        return

    phases = [pr["phase"] for pr in phase_results]
    levels = [pr["level"] for pr in phase_results]
    f1s = [pr["metrics"].get("new392", {}).get("macro_f1", 0) for pr in phase_results]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(phases)), f1s, color=[COLORS[i] for i in range(len(phases))],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([f"Phase {p}\n({l})" for p, l in zip(phases, levels)], fontsize=9)
    ax.set_ylabel("Macro-F1 (New-392)")
    ax.set_title("CurrEC: Performance After Each Phase")
    ax.set_ylim(0, max(f1s) * 1.3)

    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure3_phase_progression.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure3_phase_progression.png"), bbox_inches="tight")
    plt.close()
    print("Saved figure3_phase_progression")


def fig_lambda_sweep():
    """Figure 4: Lambda sensitivity analysis."""
    lambdas = [0.0, 0.1, 0.25, 0.5, 1.0]
    labels_map = {0.0: "no_consistency", 0.1: "lambda_0.1", 0.25: "lambda_0.25",
                  0.5: "currec", 1.0: "lambda_1.0"}

    f1_new = []
    f1_price = []
    for lam in lambdas:
        method = labels_map[lam]
        s = stats.get(method, {})
        f1_new.append(s.get("new392", {}).get("macro_f1", {}).get("mean", 0))
        f1_price.append(s.get("price149", {}).get("macro_f1", {}).get("mean", 0))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lambdas, f1_new, "o-", color=COLORS[0], label="New-392", markersize=8)
    ax.plot(lambdas, f1_price, "s-", color=COLORS[1], label="Price-149", markersize=8)
    ax.set_xlabel("Consistency Weight (lambda)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Sensitivity to Consistency Regularization Weight")
    ax.legend()
    ax.set_xticks(lambdas)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_lambda_sweep.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure4_lambda_sweep.png"), bbox_inches="tight")
    plt.close()
    print("Saved figure4_lambda_sweep")


def fig_ablation_summary():
    """Figure 5: Full ablation summary."""
    methods = ["flat_supcon", "joint_hierarchical", "currec",
               "reverse_curriculum", "random_order", "no_consistency",
               "no_temp_schedule", "two_phase"]
    labels = ["Flat SupCon", "Joint Hier.", "CurrEC", "Reverse", "Random",
              "No Consist.", "No Temp Sched.", "Two-Phase"]

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(methods))
    width = 0.35

    means_new = []
    stds_new = []
    means_price = []
    stds_price = []

    for m in methods:
        s = stats.get(m, {})
        n = s.get("new392", {}).get("macro_f1", {})
        p = s.get("price149", {}).get("macro_f1", {})
        means_new.append(n.get("mean", 0))
        stds_new.append(n.get("std", 0))
        means_price.append(p.get("mean", 0))
        stds_price.append(p.get("std", 0))

    bars1 = ax.bar(x - width/2, means_new, width, yerr=stds_new, capsize=3,
                   label="New-392", color=COLORS[0], edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, means_price, width, yerr=stds_price, capsize=3,
                   label="Price-149", color=COLORS[1], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Ablation Study: All Methods Comparison")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(means_new) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_ablation.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_ablation.png"), bbox_inches="tight")
    plt.close()
    print("Saved figure5_ablation")


def generate_table1():
    """Table 1: Main results (LaTeX)."""
    methods = [
        ("BLASTp", "blastp"),
        ("Flat SupCon", "flat_supcon"),
        ("Joint Hierarchical", "joint_hierarchical"),
        ("CurrEC (Ours)", "currec"),
    ]

    lines = []
    lines.append(r"\begin{tabular}{l|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c|}{New-392} & \multicolumn{3}{c}{Price-149} \\")
    lines.append(r"Method & F1$_\text{macro}$ & Prec & Recall & F1$_\text{macro}$ & Prec & Recall \\")
    lines.append(r"\midrule")

    for label, method in methods:
        row = [label]
        for bm in ["new392", "price149"]:
            s = stats.get(method, {}).get(bm, {})
            for metric in ["macro_f1", "precision", "recall"]:
                m = s.get(metric, {})
                mean = m.get("mean", 0)
                std = m.get("std", 0)
                n_seeds = stats.get(method, {}).get("n_seeds", 1)
                if n_seeds > 1:
                    row.append(f"{mean:.3f}$\\pm${std:.3f}")
                else:
                    row.append(f"{mean:.3f}")
        lines.append(" & ".join(row) + r" \\")

    # Published references
    lines.append(r"\midrule")
    lines.append(r"\textit{Published (not re-run):} & & & & & & \\")
    lines.append(r"CLEAN \citep{yu2023enzyme} & 0.502 & 0.575 & 0.491 & 0.438 & 0.538 & 0.408 \\")
    lines.append(r"MAPred \citep{rong2025autoregressive} & 0.610 & 0.651 & 0.632 & 0.493 & 0.554 & 0.487 \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = os.path.join(FIGURES_DIR, "table1_main_results.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    # Also save as CSV
    csv_path = os.path.join(FIGURES_DIR, "table1_main_results.csv")
    with open(csv_path, "w") as f:
        f.write("Method,New392_F1,New392_Prec,New392_Recall,Price149_F1,Price149_Prec,Price149_Recall\n")
        for label, method in methods:
            row = [label]
            for bm in ["new392", "price149"]:
                s = stats.get(method, {}).get(bm, {})
                for metric in ["macro_f1", "precision", "recall"]:
                    m = s.get(metric, {})
                    row.append(f"{m.get('mean', 0):.4f}")
            f.write(",".join(row) + "\n")

    print("Saved table1_main_results")


def generate_table2():
    """Table 2: Ablation results (LaTeX)."""
    methods = [
        ("Flat SupCon (no hierarchy)", "flat_supcon"),
        ("Joint Hierarchical", "joint_hierarchical"),
        ("CurrEC (coarse-to-fine)", "currec"),
        ("Reverse (fine-to-coarse)", "reverse_curriculum"),
        ("Random order", "random_order"),
        ("No consistency ($\\lambda=0$)", "no_consistency"),
        ("No temp schedule", "no_temp_schedule"),
        ("Two-phase (L2$\\rightarrow$L4)", "two_phase"),
    ]

    lines = []
    lines.append(r"\begin{tabular}{l|cc|cc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c|}{New-392} & \multicolumn{2}{c}{Price-149} \\")
    lines.append(r"Method & F1$_\text{macro}$ & F1$_\text{rare}$ & F1$_\text{macro}$ & F1$_\text{rare}$ \\")
    lines.append(r"\midrule")

    for label, method in methods:
        row = [label]
        for bm in ["new392", "price149"]:
            s = stats.get(method, {}).get(bm, {})
            for metric in ["macro_f1", "f1_rare"]:
                m = s.get(metric, {})
                mean = m.get("mean", 0)
                std = m.get("std", 0)
                n_seeds = stats.get(method, {}).get("n_seeds", 1)
                if n_seeds > 1 and std > 0:
                    row.append(f"{mean:.3f}$\\pm${std:.3f}")
                else:
                    row.append(f"{mean:.3f}")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = os.path.join(FIGURES_DIR, "table2_ablation.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(FIGURES_DIR, "table2_ablation.csv")
    with open(csv_path, "w") as f:
        f.write("Method,New392_F1,New392_F1_rare,Price149_F1,Price149_F1_rare\n")
        for label, method in methods:
            row = [label]
            for bm in ["new392", "price149"]:
                s = stats.get(method, {}).get(bm, {})
                for metric in ["macro_f1", "f1_rare"]:
                    m = s.get(metric, {})
                    row.append(f"{m.get('mean', 0):.4f}")
            f.write(",".join(row) + "\n")

    print("Saved table2_ablation")


if __name__ == "__main__":
    fig_main_comparison()
    fig_curriculum_order()
    fig_phase_progression()
    fig_lambda_sweep()
    fig_ablation_summary()
    generate_table1()
    generate_table2()
    print("\nAll figures and tables generated!")
