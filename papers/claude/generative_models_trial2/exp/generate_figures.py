#!/usr/bin/env python3
"""Generate all publication figures for CoPS paper."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

WORKSPACE = Path(__file__).parent.parent
FIGURES = WORKSPACE / "figures"
FIGURES.mkdir(exist_ok=True)

def load_results():
    with open(WORKSPACE / "results.json") as f:
        return json.load(f)


def figure_main_results(R):
    """Figure 2: Main comparison table as bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gather data for COCO
    methods_display = {
        "ddim_50": "DDIM-50",
        "dpm_20": "DPM-20",
        "random_k": "Random-K",
        "pcs_bestofk": "PCS Best-of-K",
        "clip_bestofk": "CLIP Best-of-K",
        "ir_bestofk": "Aesthetic Best-of-K",
        "cops_resample": "CoPS (Ours)",
    }

    clip_means, clip_stds = [], []
    ir_means, ir_stds = [], []
    labels = []

    for method_key, display in methods_display.items():
        # Get multi-seed results
        if method_key in ["ddim_50", "dpm_20"]:
            base = R.get("baselines", {})
            clips = [base.get(f"{method_key}_coco_seed{s}", {}).get("clip_mean", 0) for s in [42, 123, 456]]
            irs = [base.get(f"{method_key}_coco_seed{s}", {}).get("ir_mean", 0) for s in [42, 123, 456]]
        else:
            base = R.get("particles_coco", {})
            clips = [base.get(f"{method_key}_coco_seed{s}", {}).get("clip_mean", 0) for s in [42, 123, 456]]
            irs = [base.get(f"{method_key}_coco_seed{s}", {}).get("ir_mean", 0) for s in [42, 123, 456]]

        clips = [c for c in clips if c != 0]
        irs = [i for i in irs if i != 0]

        if clips:
            clip_means.append(np.mean(clips))
            clip_stds.append(np.std(clips))
            ir_means.append(np.mean(irs) if irs else 0)
            ir_stds.append(np.std(irs) if irs else 0)
            labels.append(display)

    if not labels:
        print("  No data for main results figure")
        return

    x = np.arange(len(labels))
    colors = ["#808080", "#A0A0A0", "#4ECDC4", "#2196F3", "#FF6B6B", "#FFA726", "#E91E63"]
    colors = colors[:len(labels)]

    # CLIP Score
    bars = axes[0].bar(x, clip_means, yerr=clip_stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("CLIP Score")
    axes[0].set_title("Text-Image Alignment (CLIP Score)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylim(bottom=min(clip_means)*0.9 if clip_means else 0)

    # Aesthetic Score
    bars = axes[1].bar(x, ir_means, yerr=ir_stds, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Aesthetic Score")
    axes[1].set_title("Image Quality (Aesthetic Score)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    plt.suptitle("Main Results on COCO-300 (3 seeds)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "figure2_main_results.pdf")
    plt.savefig(FIGURES / "figure2_main_results.png")
    plt.close()
    print("  Saved figure2_main_results")


def figure_pcs_correlation(R):
    """Figure 3: PCS correlation with external metrics."""
    corr = R.get("pcs_correlation", {})
    if not corr:
        print("  No PCS correlation data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart of Spearman correlations
    datasets = list(corr.keys())
    x = np.arange(len(datasets))
    width = 0.35

    rho_clip = [corr[d].get("spearman_pcs_clip_mean", 0) for d in datasets]
    rho_clip_std = [corr[d].get("spearman_pcs_clip_std", 0) for d in datasets]
    rho_ir = [corr[d].get("spearman_pcs_ir_mean", 0) for d in datasets]
    rho_ir_std = [corr[d].get("spearman_pcs_ir_std", 0) for d in datasets]

    axes[0].bar(x - width/2, rho_clip, width, yerr=rho_clip_std, label="PCS vs CLIP", capsize=3, color="#2196F3")
    axes[0].bar(x + width/2, rho_ir, width, yerr=rho_ir_std, label="PCS vs Aesthetic", capsize=3, color="#FF6B6B")
    axes[0].set_ylabel("Spearman Correlation")
    axes[0].set_title("PCS Rank Correlation with Quality Metrics")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.upper() for d in datasets])
    axes[0].legend()
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Agreement rates
    agree_clip = [corr[d].get("agreement_clip", 0) for d in datasets]
    agree_ir = [corr[d].get("agreement_ir", 0) for d in datasets]

    axes[1].bar(x - width/2, agree_clip, width, label="Agrees with CLIP", color="#2196F3")
    axes[1].bar(x + width/2, agree_ir, width, label="Agrees with Aesthetic", color="#FF6B6B")
    axes[1].axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Random chance (1/4)")
    axes[1].set_ylabel("Agreement Rate")
    axes[1].set_title("Selection Agreement: PCS vs External Metrics")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.upper() for d in datasets])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES / "figure3_pcs_correlation.pdf")
    plt.savefig(FIGURES / "figure3_pcs_correlation.png")
    plt.close()
    print("  Saved figure3_pcs_correlation")


def figure_scaling(R):
    """Figure 4: Quality scaling with K."""
    scaling = R.get("scaling", {})
    if not scaling:
        print("  No scaling data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = {"random": ("Random", "#808080", "o"), "pcs": ("PCS (Ours)", "#E91E63", "s"),
               "clip_sel": ("CLIP Oracle", "#2196F3", "^")}

    for m, (label, color, marker) in methods.items():
        Ks, clips, irs = [], [], []
        for key, val in scaling.items():
            if val.get("selection") == m:
                Ks.append(val["K"])
                clips.append(val["clip_mean"])
                irs.append(val["ir_mean"])
        if Ks:
            order = np.argsort(Ks)
            Ks = np.array(Ks)[order]
            clips = np.array(clips)[order]
            irs = np.array(irs)[order]

            axes[0].plot(Ks, clips, f"-{marker}", label=label, color=color, linewidth=2, markersize=8)
            axes[1].plot(Ks, irs, f"-{marker}", label=label, color=color, linewidth=2, markersize=8)

    for ax, title, ylabel in [(axes[0], "CLIP Score vs K", "CLIP Score"),
                               (axes[1], "Aesthetic Score vs K", "Aesthetic Score")]:
        ax.set_xlabel("Number of Particles (K)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.suptitle("Inference-Time Scaling: Quality vs Number of Particles\n(Fixed Total NFE = 200)", y=1.04)
    plt.tight_layout()
    plt.savefig(FIGURES / "figure4_scaling.pdf")
    plt.savefig(FIGURES / "figure4_scaling.png")
    plt.close()
    print("  Saved figure4_scaling")


def figure_ablations(R):
    """Figure 5: Ablation studies (2x2 grid)."""
    abl = R.get("ablations", {})
    if not abl:
        print("  No ablation data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # A1: Distance metric
    ax = axes[0, 0]
    metrics_data = {k: v for k, v in abl.items() if k.startswith("dist_")}
    if metrics_data:
        names = [v.get("metric", k) for k, v in metrics_data.items()]
        clips = [v["clip_mean"] for v in metrics_data.values()]
        ax.bar(names, clips, color=["#2196F3", "#FF6B6B"][:len(names)], edgecolor="black", linewidth=0.5)
        ax.set_ylabel("CLIP Score")
        ax.set_title("A1: Distance Metric for PCS")

    # A2: Resampling frequency
    ax = axes[0, 1]
    resample_data = {k: v for k, v in abl.items() if k.startswith("R_")}
    if resample_data:
        Rs = sorted([v["R"] for v in resample_data.values()])
        clips = [abl[f"R_{R}"]["clip_mean"] for R in Rs]
        irs = [abl[f"R_{R}"]["ir_mean"] for R in Rs]
        ax.plot(Rs, clips, "-o", color="#2196F3", label="CLIP", linewidth=2, markersize=8)
        ax2 = ax.twinx()
        ax2.plot(Rs, irs, "-s", color="#FF6B6B", label="Aesthetic", linewidth=2, markersize=8)
        ax.set_xlabel("Resampling Interval (R)")
        ax.set_ylabel("CLIP Score", color="#2196F3")
        ax2.set_ylabel("Aesthetic Score", color="#FF6B6B")
        ax.set_title("A2: Resampling Frequency")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    # A3: Timestep weighting
    ax = axes[1, 0]
    weight_data = {k: v for k, v in abl.items() if k.startswith("weight_")}
    if weight_data:
        names = [v.get("weight", k).replace("_", "\n") for k, v in weight_data.items()]
        clips = [v["clip_mean"] for v in weight_data.values()]
        colors = ["#2196F3", "#4CAF50", "#FF6B6B", "#FFA726"][:len(names)]
        ax.bar(names, clips, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("CLIP Score")
        ax.set_title("A3: Timestep Weighting for PCS")

    # A5: CFG combination
    ax = axes[1, 1]
    cfg_data = {k: v for k, v in abl.items() if k.startswith("cfg")}
    if cfg_data:
        cfgs = sorted(set(v.get("cfg", 0) for v in cfg_data.values()))
        std_clips = [abl.get(f"cfg{cfg}_std", {}).get("clip_mean", 0) for cfg in cfgs]
        cops_clips = [abl.get(f"cfg{cfg}_cops", {}).get("clip_mean", 0) for cfg in cfgs]

        x = np.arange(len(cfgs))
        width = 0.35
        ax.bar(x - width/2, std_clips, width, label="Standard", color="#808080")
        ax.bar(x + width/2, cops_clips, width, label="+ CoPS", color="#E91E63")
        ax.set_xlabel("CFG Scale")
        ax.set_ylabel("CLIP Score")
        ax.set_title("A5: CFG Scale Combination")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cfgs])
        ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES / "figure5_ablations.pdf")
    plt.savefig(FIGURES / "figure5_ablations.png")
    plt.close()
    print("  Saved figure5_ablations")


def figure_fid(R):
    """Figure 6: FID comparison bar chart."""
    fid_data = R.get("fid", {})
    if not fid_data:
        print("  No FID data")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    display_names = {
        "ddim_50": "DDIM-50",
        "dpm_20": "DPM-20",
        "random_k": "Random-K",
        "pcs_bestofk": "PCS Best-of-K",
        "clip_bestofk": "CLIP Best-of-K",
        "ir_bestofk": "Aesthetic Best-of-K",
        "cops_resample": "CoPS (Ours)",
    }

    names, vals = [], []
    for k in ["ddim_50", "dpm_20", "random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk", "cops_resample"]:
        if k in fid_data and fid_data[k] is not None:
            names.append(display_names.get(k, k))
            vals.append(fid_data[k])

    if not names:
        return

    colors = ["#808080", "#A0A0A0", "#4ECDC4", "#2196F3", "#FF6B6B", "#FFA726", "#E91E63"][:len(names)]
    ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("FID Scores on COCO-300 (seed=42)")
    ax.set_xticklabels(names, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(FIGURES / "figure6_fid.pdf")
    plt.savefig(FIGURES / "figure6_fid.png")
    plt.close()
    print("  Saved figure6_fid")


def figure_cost(R):
    """Figure 7: Cost analysis."""
    cost = R.get("cost", {})
    if not cost:
        print("  No cost data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    names = list(cost.keys())
    display = {"ddim_50": "DDIM-50", "particles_k4": "K=4\n(no resample)", "cops_k4": "CoPS\n(K=4)"}
    labels = [display.get(n, n) for n in names]

    times = [cost[n]["time_per_img"] for n in names]
    mems = [cost[n]["peak_gpu_gb"] for n in names]
    colors = ["#808080", "#4ECDC4", "#E91E63"]

    axes[0].bar(labels, times, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Time per Image (seconds)")
    axes[0].set_title("Wall-Clock Time")

    axes[1].bar(labels, mems, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Peak GPU Memory (GB)")
    axes[1].set_title("GPU Memory Usage")

    plt.suptitle("Computational Cost Analysis", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "figure7_cost.pdf")
    plt.savefig(FIGURES / "figure7_cost.png")
    plt.close()
    print("  Saved figure7_cost")


def figure_pcs_scatter(R):
    """Figure 8: PCS vs CLIP scatter plot from raw data."""
    data_file = WORKSPACE / "exp" / "analysis" / "pcs_data_coco.json"
    if not data_file.exists():
        print("  No PCS scatter data")
        return

    with open(data_file) as f:
        pcs_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    all_pcs, all_clip, all_ir = [], [], []
    for item in pcs_data:
        all_pcs.extend(item["pcs"])
        all_clip.extend(item["clip"])
        all_ir.extend(item["ir"])

    all_pcs = np.array(all_pcs)
    all_clip = np.array(all_clip)
    all_ir = np.array(all_ir)

    axes[0].scatter(all_pcs, all_clip, alpha=0.15, s=8, c="#2196F3")
    axes[0].set_xlabel("PCS Score")
    axes[0].set_ylabel("CLIP Score")
    axes[0].set_title("PCS vs CLIP Score (all particles)")
    from scipy.stats import spearmanr
    rho, p = spearmanr(all_pcs, all_clip)
    axes[0].text(0.05, 0.95, f"Spearman ρ = {rho:.3f}", transform=axes[0].transAxes, va="top")

    axes[1].scatter(all_pcs, all_ir, alpha=0.15, s=8, c="#FF6B6B")
    axes[1].set_xlabel("PCS Score")
    axes[1].set_ylabel("Aesthetic Score")
    axes[1].set_title("PCS vs Aesthetic Score (all particles)")
    rho2, p2 = spearmanr(all_pcs, all_ir)
    axes[1].text(0.05, 0.95, f"Spearman ρ = {rho2:.3f}", transform=axes[1].transAxes, va="top")

    plt.tight_layout()
    plt.savefig(FIGURES / "figure8_pcs_scatter.pdf")
    plt.savefig(FIGURES / "figure8_pcs_scatter.png")
    plt.close()
    print("  Saved figure8_pcs_scatter")


def create_latex_table(R):
    """Create LaTeX table for main results."""
    methods_display = {
        "ddim_50": "DDIM-50",
        "dpm_20": "DPM-Solver-20",
        "random_k": "Random-K ($K$=4)",
        "pcs_bestofk": "PCS Best-of-K ($K$=4)",
        "clip_bestofk": "CLIP Best-of-K ($K$=4)",
        "ir_bestofk": "Aesthetic Best-of-K ($K$=4)",
        "cops_resample": "\\textbf{CoPS (Ours)} ($K$=4)",
    }

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Main results on COCO-300 with 3 random seeds. $\\uparrow$ higher is better, $\\downarrow$ lower is better.}")
    lines.append("\\label{tab:main}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Method & CLIP Score $\\uparrow$ & Aesthetic $\\uparrow$ & FID $\\downarrow$ \\\\")
    lines.append("\\midrule")

    fid_data = R.get("fid", {})

    for mk, display in methods_display.items():
        if mk in ["ddim_50", "dpm_20"]:
            base = R.get("baselines", {})
            clips = [base.get(f"{mk}_coco_seed{s}", {}).get("clip_mean", 0) for s in [42, 123, 456]]
            irs = [base.get(f"{mk}_coco_seed{s}", {}).get("ir_mean", 0) for s in [42, 123, 456]]
        else:
            base = R.get("particles_coco", {})
            clips = [base.get(f"{mk}_coco_seed{s}", {}).get("clip_mean", 0) for s in [42, 123, 456]]
            irs = [base.get(f"{mk}_coco_seed{s}", {}).get("ir_mean", 0) for s in [42, 123, 456]]

        clips = [c for c in clips if c != 0]
        irs = [i for i in irs if i != 0]

        fid_val = fid_data.get(mk, None)
        fid_str = f"{fid_val:.1f}" if fid_val else "-"
        clip_str = f"{np.mean(clips):.4f} $\\pm$ {np.std(clips):.4f}" if clips else "-"
        ir_str = f"{np.mean(irs):.4f} $\\pm$ {np.std(irs):.4f}" if irs else "-"

        lines.append(f"{display} & {clip_str} & {ir_str} & {fid_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(FIGURES / "table1_main_results.tex", "w") as f:
        f.write("\n".join(lines))
    print("  Saved table1_main_results.tex")


def main():
    print("Generating figures...")
    R = load_results()

    figure_main_results(R)
    figure_pcs_correlation(R)
    figure_scaling(R)
    figure_ablations(R)
    figure_fid(R)
    figure_cost(R)
    figure_pcs_scatter(R)
    create_latex_table(R)

    print(f"\nAll figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
