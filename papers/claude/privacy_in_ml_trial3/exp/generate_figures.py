#!/usr/bin/env python3
"""Generate all figures for the paper."""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(WORKSPACE, "results")
FIGURES_DIR = os.path.join(WORKSPACE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None


def fig1_compounding_heatmap():
    """Heatmap of compounding ratios: epsilon x sparsity."""
    cr_data = load_json_safe(os.path.join(RESULTS_DIR, "compounding_ratios.json"))
    if not cr_data:
        print("No compounding ratio data found")
        return

    datasets = set()
    for key, val in cr_data.items():
        if "_ft" not in key:
            datasets.add(val["dataset"])

    for ds in sorted(datasets):
        epsilons = sorted(set(v["epsilon"] for v in cr_data.values() if v.get("dataset") == ds and "_ft" not in str(v.get("finetuned"))))
        sparsities = sorted(set(v["sparsity"] for v in cr_data.values() if v.get("dataset") == ds and "_ft" not in str(v.get("finetuned"))))

        if not epsilons or not sparsities:
            continue

        # Build mean CR matrix
        cr_matrix = np.full((len(epsilons), len(sparsities)), np.nan)
        for i, eps in enumerate(epsilons):
            for j, sp in enumerate(sparsities):
                crs = [v["CR"] for k, v in cr_data.items()
                       if v.get("dataset") == ds and v.get("epsilon") == eps
                       and v.get("sparsity") == sp and "_ft" not in k
                       and not np.isnan(v.get("CR", float("nan")))]
                if crs:
                    cr_matrix[i, j] = np.mean(crs)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        mask = np.isnan(cr_matrix)
        im = sns.heatmap(cr_matrix, ax=ax, annot=True, fmt=".2f",
                         xticklabels=[f"{s:.0%}" for s in sparsities],
                         yticklabels=[f"ε={e}" for e in epsilons],
                         cmap="RdYlGn_r", center=1.0, vmin=0.5, vmax=2.5,
                         mask=mask, linewidths=0.5)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Privacy Budget")
        ax.set_title(f"Compounding Ratio — {ds}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"compounding_ratio_heatmap_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved compounding_ratio_heatmap_{ds}")


def fig2_subgroup_accuracy_bars():
    """Bar chart of per-subgroup accuracy for each model variant."""
    datasets = ["cifar10", "utkface", "celeba"]
    eps_target = 4
    sp_target = 0.7

    for ds in datasets:
        variants = {}
        # Collect data for seed 42
        seed = 42

        base = load_json_safe(os.path.join(RESULTS_DIR, ds, "baseline", f"metrics_seed{seed}.json"))
        if base:
            variants["Baseline"] = base.get("per_subgroup_accuracy", {})

        dp = load_json_safe(os.path.join(RESULTS_DIR, ds, "dp_only", f"metrics_eps{eps_target}_seed{seed}.json"))
        if dp:
            variants["DP-only"] = dp.get("per_subgroup_accuracy", {})

        comp = load_json_safe(os.path.join(RESULTS_DIR, ds, "comp_only", f"metrics_sp{sp_target}_seed{seed}.json"))
        if comp:
            variants["Comp-only"] = comp.get("per_subgroup_accuracy", {})

        dc = load_json_safe(os.path.join(RESULTS_DIR, ds, "dp_comp", f"metrics_eps{eps_target}_sp{sp_target}_seed{seed}.json"))
        if dc:
            variants["DP+Comp"] = dc.get("per_subgroup_accuracy", {})

        fp = load_json_safe(os.path.join(RESULTS_DIR, ds, "fairprune_dp", f"metrics_eps{eps_target}_sp{sp_target}_seed{seed}.json"))
        if fp:
            variants["FairPrune-DP"] = fp.get("per_subgroup_accuracy", {})

        if len(variants) < 2:
            continue

        # Get all subgroup keys
        all_sgs = sorted(set().union(*[v.keys() for v in variants.values()]))

        x = np.arange(len(all_sgs))
        width = 0.15
        fig, ax = plt.subplots(figsize=(7, 4))

        for i, (name, accs) in enumerate(variants.items()):
            vals = [accs.get(sg, 0) for sg in all_sgs]
            ax.bar(x + i * width, vals, width, label=name, color=PALETTE[i % len(PALETTE)])

        stats = load_json_safe(os.path.join(RESULTS_DIR, ds, "data_stats.json"))
        sg_names = stats.get("subgroup_names", {}) if stats else {}
        labels = [sg_names.get(str(sg), sg_names.get(int(sg), f"Group {sg}")) for sg in all_sgs]

        ax.set_xlabel("Subgroup")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-Subgroup Accuracy — {ds} (ε={eps_target}, sparsity={sp_target:.0%})")
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 0)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"subgroup_accuracy_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved subgroup_accuracy_{ds}")


def fig3_fairness_compression_pareto():
    """Accuracy gap vs sparsity for each method."""
    datasets = ["cifar10", "utkface", "celeba"]
    seeds = [42, 123, 456]
    sparsities = [0.5, 0.7, 0.9]
    eps_target = 4

    for ds in datasets:
        methods = {
            "Magnitude": [],
            "Fisher": [],
            "FairPrune-DP": [],
        }

        for sp in sparsities:
            for method_name, subdir, prefix in [
                ("Magnitude", "dp_comp", f"metrics_eps{eps_target}_sp{sp}"),
                ("Fisher", "fisher_prune", f"metrics_eps{eps_target}_sp{sp}"),
                ("FairPrune-DP", "fairprune_dp", f"metrics_eps{eps_target}_sp{sp}"),
            ]:
                gaps = []
                for seed in seeds:
                    m = load_json_safe(os.path.join(RESULTS_DIR, ds, subdir, f"{prefix}_seed{seed}.json"))
                    if m:
                        gaps.append(m.get("accuracy_gap", 0))
                if gaps:
                    methods[method_name].append((sp, np.mean(gaps), np.std(gaps)))

        if not any(methods.values()):
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        for i, (name, data) in enumerate(methods.items()):
            if not data:
                continue
            sps, means, stds = zip(*data)
            ax.errorbar(sps, means, yerr=stds, marker='o', label=name,
                       color=PALETTE[i], capsize=3, linewidth=2)

        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Accuracy Gap (Best - Worst Subgroup)")
        ax.set_title(f"Fairness vs Compression — {ds} (ε={eps_target})")
        ax.legend()
        ax.set_xlim(0.4, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"pareto_frontier_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved pareto_frontier_{ds}")


def fig4_weight_distributions():
    """Weight magnitude distributions for minority vs majority relevant features."""
    mech = load_json_safe(os.path.join(RESULTS_DIR, "mechanistic_analysis.json"))
    if not mech:
        print("No mechanistic analysis data found")
        return

    datasets_data = defaultdict(list)
    for key, val in mech.items():
        ds = key.split("_eps")[0]
        datasets_data[ds].append(val)

    for ds, vals in datasets_data.items():
        baseline_minority = []
        baseline_majority = []
        dp_minority = []
        dp_majority = []

        for v in vals:
            bw = v.get("baseline_weight_stats")
            dw = v.get("dp_weight_stats")
            if bw:
                baseline_minority.append(bw.get("minority_relevant_magnitude_mean", 0))
                baseline_majority.append(bw.get("majority_relevant_magnitude_mean", 0))
            if dw:
                dp_minority.append(dw.get("minority_relevant_magnitude_mean", 0))
                dp_majority.append(dw.get("majority_relevant_magnitude_mean", 0))

        if not baseline_minority:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(2)
        width = 0.35

        ax.bar(x - width/2, [np.mean(baseline_minority), np.mean(dp_minority)],
               width, label='Minority-relevant', color=PALETTE[0],
               yerr=[np.std(baseline_minority), np.std(dp_minority)], capsize=3)
        ax.bar(x + width/2, [np.mean(baseline_majority), np.mean(dp_majority)],
               width, label='Majority-relevant', color=PALETTE[1],
               yerr=[np.std(baseline_majority), np.std(dp_majority)], capsize=3)

        ax.set_xlabel("Model Type")
        ax.set_ylabel("Mean Weight Magnitude")
        ax.set_title(f"Weight Magnitudes by Subgroup Relevance — {ds}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Standard", "DP-trained"])
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"weight_distributions_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved weight_distributions_{ds}")


def fig5_pruning_overlap():
    """Fraction of pruned weights that are minority-relevant."""
    mech = load_json_safe(os.path.join(RESULTS_DIR, "mechanistic_analysis.json"))
    if not mech:
        return

    datasets_data = defaultdict(lambda: defaultdict(lambda: {"base": [], "dp": []}))
    for key, val in mech.items():
        ds = key.split("_eps")[0]
        overlap = val.get("pruning_overlap", {})
        for sp_str, ov in overlap.items():
            sp = float(sp_str)
            datasets_data[ds][sp]["base"].append(ov.get("base_minority_frac", 0))
            datasets_data[ds][sp]["dp"].append(ov.get("dp_minority_frac", 0))

    for ds, sp_data in datasets_data.items():
        sps = sorted(sp_data.keys())
        if not sps:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(sps))
        width = 0.35

        base_means = [np.mean(sp_data[sp]["base"]) for sp in sps]
        dp_means = [np.mean(sp_data[sp]["dp"]) for sp in sps]
        base_stds = [np.std(sp_data[sp]["base"]) for sp in sps]
        dp_stds = [np.std(sp_data[sp]["dp"]) for sp in sps]

        ax.bar(x - width/2, base_means, width, label='Standard', color=PALETTE[0],
               yerr=base_stds, capsize=3)
        ax.bar(x + width/2, dp_means, width, label='DP-trained', color=PALETTE[3],
               yerr=dp_stds, capsize=3)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Fraction of Pruned Weights\nthat are Minority-Relevant")
        ax.set_title(f"Pruning Overlap Analysis — {ds}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sp:.0%}" for sp in sps])
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"pruning_overlap_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved pruning_overlap_{ds}")


def fig6_mia_disparity():
    """MIA disparity across model variants."""
    mia = load_json_safe(os.path.join(RESULTS_DIR, "mia_results.json"))
    if not mia:
        print("No MIA data found")
        return

    datasets_data = defaultdict(lambda: defaultdict(list))
    for key, val in mia.items():
        parts = key.split("_seed")
        variant_parts = parts[0].split("_", 1)
        ds = variant_parts[0]
        variant = variant_parts[1] if len(variant_parts) > 1 else "baseline"
        datasets_data[ds][variant].append(val.get("mia_disparity", 0))

    for ds, variants in datasets_data.items():
        if not variants:
            continue

        fig, ax = plt.subplots(figsize=(6, 3.5))
        names = sorted(variants.keys())
        means = [np.mean(variants[n]) for n in names]
        stds = [np.std(variants[n]) for n in names]

        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=3,
                     color=[PALETTE[i % len(PALETTE)] for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("MIA Disparity (Max - Min Subgroup)")
        ax.set_title(f"MIA Vulnerability Disparity — {ds}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"mia_disparity_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved mia_disparity_{ds}")


def fig7_ablation_criterion():
    """Ablation: comparison of pruning criteria."""
    ablation = load_json_safe(os.path.join(RESULTS_DIR, "ablation_results.json"))
    if not ablation:
        print("No ablation data found")
        return

    for ds in ["cifar10", "utkface"]:
        criterion_data = defaultdict(lambda: {"acc": [], "worst": [], "gap": []})
        for key, val in ablation.items():
            if f"{ds}_criterion" in key:
                for method, metrics in val.items():
                    criterion_data[method]["acc"].append(metrics.get("overall_accuracy", 0))
                    criterion_data[method]["worst"].append(metrics.get("worst_group_accuracy", 0))
                    criterion_data[method]["gap"].append(metrics.get("accuracy_gap", 0))

        if not criterion_data:
            continue

        methods = list(criterion_data.keys())
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        # Worst-group accuracy
        ax = axes[0]
        means = [np.mean(criterion_data[m]["worst"]) for m in methods]
        stds = [np.std(criterion_data[m]["worst"]) for m in methods]
        ax.bar(range(len(methods)), means, yerr=stds, capsize=3,
               color=[PALETTE[i] for i in range(len(methods))])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Worst-Group Accuracy")
        ax.set_title("Worst-Group Accuracy")

        # Accuracy gap
        ax = axes[1]
        means = [np.mean(criterion_data[m]["gap"]) for m in methods]
        stds = [np.std(criterion_data[m]["gap"]) for m in methods]
        ax.bar(range(len(methods)), means, yerr=stds, capsize=3,
               color=[PALETTE[i] for i in range(len(methods))])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Accuracy Gap")
        ax.set_title("Accuracy Gap")

        plt.suptitle(f"Pruning Criterion Ablation — {ds} (ε=4, sparsity=70%)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds}.pdf"), bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, f"ablation_criterion_{ds}.png"), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved ablation_criterion_{ds}")


def generate_latex_tables():
    """Generate LaTeX tables for the paper."""
    # Main results table
    datasets = ["cifar10", "utkface", "celeba"]
    seeds = [42, 123, 456]

    rows = []
    for ds in datasets:
        for variant_name, subdir, file_pattern in [
            ("Baseline", "baseline", "metrics_seed{seed}.json"),
            ("DP-only (ε=4)", "dp_only", "metrics_eps4_seed{seed}.json"),
            ("Comp-only (70%)", "comp_only", "metrics_sp0.7_seed{seed}.json"),
            ("DP+Comp", "dp_comp", "metrics_eps4_sp0.7_seed{seed}.json"),
            ("FairPrune-DP", "fairprune_dp", "metrics_eps4_sp0.7_seed{seed}.json"),
        ]:
            accs, worsts, gaps = [], [], []
            for seed in seeds:
                fp = os.path.join(RESULTS_DIR, ds, subdir, file_pattern.format(seed=seed))
                m = load_json_safe(fp)
                if m:
                    accs.append(m.get("overall_accuracy", 0))
                    worsts.append(m.get("worst_group_accuracy", 0))
                    gaps.append(m.get("accuracy_gap", 0))
            if accs:
                rows.append(f"  {ds} & {variant_name} & "
                           f"${np.mean(accs):.3f} \\pm {np.std(accs):.3f}$ & "
                           f"${np.mean(worsts):.3f} \\pm {np.std(worsts):.3f}$ & "
                           f"${np.mean(gaps):.3f} \\pm {np.std(gaps):.3f}$ \\\\")

    table = "\\begin{table}[t]\n\\centering\n\\small\n"
    table += "\\caption{Main results: accuracy and fairness across model variants.}\n"
    table += "\\label{tab:main_results}\n"
    table += "\\begin{tabular}{llccc}\n\\toprule\n"
    table += "Dataset & Variant & Overall Acc & Worst-Group Acc & Acc Gap \\\\\n\\midrule\n"
    table += "\n".join(rows)
    table += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"

    with open(os.path.join(FIGURES_DIR, "table_main_results.tex"), "w") as f:
        f.write(table)
    print("  Saved table_main_results.tex")

    # Compounding ratio table
    cr_summary = load_json_safe(os.path.join(RESULTS_DIR, "compounding_ratio_summary.json"))
    if cr_summary:
        cr_rows = []
        for key, val in sorted(cr_summary.items()):
            cr_rows.append(f"  {val.get('dataset', '')} & ε={val.get('epsilon', '')} & "
                          f"{val.get('sparsity', 0):.0%} & "
                          f"${val.get('mean_CR', 0):.2f} \\pm {val.get('std_CR', 0):.2f}$ & "
                          f"{val.get('p_value_cr_gt_1', 1):.4f} \\\\")

        cr_table = "\\begin{table}[t]\n\\centering\n\\small\n"
        cr_table += "\\caption{Compounding ratios across configurations.}\n"
        cr_table += "\\label{tab:compounding}\n"
        cr_table += "\\begin{tabular}{llccc}\n\\toprule\n"
        cr_table += "Dataset & ε & Sparsity & CR (mean±std) & p-value \\\\\n\\midrule\n"
        cr_table += "\n".join(cr_rows)
        cr_table += "\n\\bottomrule\n\\end{tabular}\n\\end{table}"

        with open(os.path.join(FIGURES_DIR, "table_compounding.tex"), "w") as f:
            f.write(cr_table)
        print("  Saved table_compounding.tex")


def main():
    print("Generating figures...")
    fig1_compounding_heatmap()
    fig2_subgroup_accuracy_bars()
    fig3_fairness_compression_pareto()
    fig4_weight_distributions()
    fig5_pruning_overlap()
    fig6_mia_disparity()
    fig7_ablation_criterion()
    generate_latex_tables()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
