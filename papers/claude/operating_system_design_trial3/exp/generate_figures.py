#!/usr/bin/env python3
"""Generate all paper figures from experiment results."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Style
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
PALETTE = sns.color_palette("colorblind")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)
RESDIR = Path("results")


def fig1_displacement_ratios():
    """Bar chart of displacement ratios by async mechanism."""
    df = pd.read_csv(RESDIR / "displacement_characterization.csv")
    mechs = ["io_uring_io_wq", "io_uring_sqpoll", "softirq_network", "workqueue_cmwq", "mixed"]
    labels = ["io_uring\n(io-wq)", "io_uring\n(SQPOLL)", "Softirq\n(network)", "Workqueue\n(cmwq)", "Mixed\nworkload"]
    means, stds = [], []
    for m in mechs:
        s = df[df["mechanism"] == m]["relay_cpu_fraction"]
        means.append(s.mean())
        stds.append(s.std())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(mechs)), means, yerr=stds, capsize=5, color=PALETTE[:len(mechs)],
                  edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, label='10% threshold')
    ax.set_xticks(range(len(mechs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Displacement Fraction")
    ax.set_title("CPU Time Displacement by Async Mechanism")
    ax.legend()
    ax.set_ylim(0, 0.45)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_displacement_ratios.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig1_displacement_ratios.png", dpi=150)
    plt.close()
    print("  Figure 1: displacement ratios")


def fig2_fairness_scaling():
    """Line plot of J_effective vs N for baselines and heterogeneous."""
    df_b1 = pd.read_csv(RESDIR / "baseline_no_displacement.csv")
    df_b2 = pd.read_csv(RESDIR / "baseline_uniform_displacement.csv")
    df_het = pd.read_csv(RESDIR / "fairness_violation.csv")
    df_het = df_het[df_het["M"] == 2]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for df, label, color, marker in [
        (df_b1, "No displacement (Baseline 1)", PALETTE[0], 'o'),
        (df_b2, "Uniform displacement (Baseline 2)", PALETTE[1], 's'),
        (df_het, "Heterogeneous displacement", PALETTE[2], '^'),
    ]:
        col = "jain_effective" if "jain_effective" in df.columns else "jain_fairness"
        grp = df.groupby("N")[col].agg(["mean", "std"]).reset_index()
        ax.plot(grp["N"], grp["mean"], marker=marker, label=label, color=color, linewidth=2)
        ax.fill_between(grp["N"], grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                       alpha=0.2, color=color)

    ax.axhline(y=0.9, color='red', linestyle=':', linewidth=1, alpha=0.7, label='J=0.9 threshold')
    ax.set_xscale('log', base=2)
    ax.set_xlabel("Number of Processes (N)")
    ax.set_ylabel("Jain's Fairness Index (Effective)")
    ax.set_title("Fairness Degradation Under Heterogeneous Displacement")
    ax.legend(loc='lower left', fontsize=9)
    ax.set_ylim(0.3, 1.05)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels(['4', '8', '16', '32', '64', '128', '256'])
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_fairness_scaling.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig2_fairness_scaling.png", dpi=150)
    plt.close()
    print("  Figure 2: fairness scaling")


def fig3_scheduler_vs_reality():
    """Bar chart comparing scheduler-reported vs effective CPU shares."""
    df = pd.read_csv(RESDIR / "fairness_violation.csv")
    df32 = df[(df["N"] == 32) & (df["M"] == 2) & (df["seed"] == 42)]

    # Reconstruct per-task shares from the simulation
    N = 32
    rng = np.random.RandomState(42)
    io_alphas = rng.beta(2, 3, size=N//2)
    cpu_alphas = np.zeros(N//2)
    all_alphas = np.concatenate([io_alphas, cpu_alphas])

    # Each task gets 1/N reported share
    reported = np.ones(N) / N
    # Effective share = reported / (1 - alpha)
    effective = np.array([1.0/(N*(1.0-a)) if a < 1 else 1.0/N for a in all_alphas])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(N)
    w = 0.35
    ax.bar(x - w/2, reported, w, label='Scheduler-reported share', color=PALETTE[0], alpha=0.8)
    ax.bar(x + w/2, effective, w, label='Effective share (incl. displaced)', color=PALETTE[2], alpha=0.8)

    # Mark IO vs CPU
    ax.axvline(x=N//2 - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(N//4, max(effective)*1.02, 'I/O-heavy tasks', ha='center', fontsize=9, style='italic')
    ax.text(3*N//4, max(effective)*1.02, 'CPU-bound tasks', ha='center', fontsize=9, style='italic')

    ax.set_xlabel("Task ID")
    ax.set_ylabel("CPU Share")
    ax.set_title("Scheduler View vs. Reality (N=32, Heterogeneous Displacement)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_scheduler_vs_reality.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig3_scheduler_vs_reality.png", dpi=150)
    plt.close()
    print("  Figure 3: scheduler vs reality")


def fig4_cgroup_leakage():
    """Stacked bar chart of cgroup accounting under different policies."""
    df = pd.read_csv(RESDIR / "cgroup_accounting.csv")
    df4 = df[df["K"] == 4]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    policies = ["none", "partial", "full"]
    titles = ["No Attribution", "Partial Attribution\n(io_uring only)", "Full Attribution\n(CCP)"]

    for idx, (policy, title) in enumerate(zip(policies, titles)):
        ax = axes[idx]
        sub = df4[df4["attribution_policy"] == policy]
        grp = sub.groupby("cgroup_type").agg(
            reported=("reported_cpu", "mean"),
            actual=("actual_cpu", "mean"),
        ).reset_index()

        x = np.arange(len(grp))
        w = 0.35
        ax.bar(x - w/2, grp["reported"], w, label='Reported CPU', color=PALETTE[0], alpha=0.8)
        ax.bar(x + w/2, grp["actual"], w, label='Actual CPU', color=PALETTE[2], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', '\n') for t in grp["cgroup_type"]], fontsize=8)
        ax.set_title(title)
        if idx == 0:
            ax.set_ylabel("CPU Time (us)")
            ax.legend(fontsize=8)

    plt.suptitle("Cgroup CPU Accounting Accuracy (K=4 Cgroups)", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_cgroup_leakage.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(FIGDIR / "fig4_cgroup_leakage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Figure 4: cgroup leakage")


def fig5_ccp_evaluation():
    """CCP evaluation: fairness vs overhead for all strategies."""
    df = pd.read_csv(RESDIR / "ccp_evaluation.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # (a) Batched CCP: fairness vs batch interval
    batch = df[df["strategy"] == "batched"]
    batch_grp = batch.groupby("param_value").agg(
        j_mean=("jain_with_ccp", "mean"), j_std=("jain_with_ccp", "std"),
        oh_mean=("overhead_pct", "mean")).reset_index()
    # Sort by batch interval
    batch_grp["ms"] = batch_grp["param_value"].str.extract(r'(\d+)').astype(float)
    batch_grp = batch_grp.sort_values("ms")

    ax1.plot(batch_grp["ms"], batch_grp["j_mean"], 'o-', color=PALETTE[0], linewidth=2, label='Fairness (J)')
    ax1.fill_between(batch_grp["ms"], batch_grp["j_mean"]-batch_grp["j_std"],
                     batch_grp["j_mean"]+batch_grp["j_std"], alpha=0.2, color=PALETTE[0])
    ax1.set_xlabel("Batch Interval (ms)")
    ax1.set_ylabel("Jain's Fairness Index", color=PALETTE[0])
    ax1.set_ylim(0.95, 1.005)
    ax1t = ax1.twinx()
    ax1t.plot(batch_grp["ms"], batch_grp["oh_mean"], 's--', color=PALETTE[2], linewidth=2, label='Overhead')
    ax1t.set_ylabel("Overhead (%)", color=PALETTE[2])
    ax1.set_title("(a) BatchedCCP: Fairness vs. Batch Interval")

    # (b) All strategies scatter
    no_ccp_j = df["jain_no_ccp"].mean()
    grp = df.groupby(["strategy", "param_value"]).agg(
        j=("jain_with_ccp", "mean"), oh=("overhead_pct", "mean")).reset_index()
    markers = {"immediate": "o", "batched": "s", "statistical": "^"}
    for strat in grp["strategy"].unique():
        s = grp[grp["strategy"] == strat]
        ax2.scatter(s["oh"], s["j"], marker=markers.get(strat, 'o'), s=100,
                   label=strat.capitalize(), zorder=5)
    ax2.axhline(y=no_ccp_j, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'No CCP (J={no_ccp_j:.2f})')
    ax2.axhline(y=0.95, color='green', linestyle=':', linewidth=1, alpha=0.7, label='J=0.95 target')
    ax2.set_xlabel("Overhead (%)")
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_title("(b) All CCP Strategies: Fairness vs. Overhead")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.7, 1.02)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig5_ccp_evaluation.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig5_ccp_evaluation.png", dpi=150)
    plt.close()
    print("  Figure 5: CCP evaluation")


def fig6_ccp_convergence():
    """Time series of fairness under different CCP strategies."""
    with open(RESDIR / "ccp_convergence.json") as f:
        conv = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = {"no_ccp": "No CCP", "immediate": "Immediate CCP",
              "batched": "Batched CCP (10ms)", "statistical": "Statistical CCP"}
    colors = {"no_ccp": PALETTE[3], "immediate": PALETTE[0],
              "batched": PALETTE[1], "statistical": PALETTE[2]}

    for key, data in conv.items():
        if not data:
            continue
        times = [d[0] for d in data]
        fairness = [d[1] for d in data]
        # Downsample for plotting
        step = max(1, len(times) // 500)
        ax.plot(times[::step], fairness[::step], label=labels.get(key, key),
               color=colors.get(key), linewidth=1.5, alpha=0.8)

    ax.axhline(y=0.95, color='green', linestyle=':', linewidth=1, alpha=0.5, label='J=0.95 target')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Jain's Fairness Index (Effective)")
    ax.set_title("CCP Convergence Under Heterogeneous Displacement")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(0.3, 1.05)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig6_ccp_convergence.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig6_ccp_convergence.png", dpi=150)
    plt.close()
    print("  Figure 6: CCP convergence")


def fig7_sensitivity():
    """2x2 sensitivity analysis subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Fairness gap vs Var(alpha)
    df_v = pd.read_csv(RESDIR / "ablation_variance.csv")
    grp = df_v.groupby("var_level").agg(
        var=("var_alpha", "mean"), gap_mean=("fairness_gap", "mean"),
        gap_std=("fairness_gap", "std")).reset_index()
    grp = grp.sort_values("var")
    ax = axes[0, 0]
    ax.errorbar(grp["var"], grp["gap_mean"], yerr=grp["gap_std"], fmt='o-', color=PALETTE[0],
               linewidth=2, capsize=5)
    # Regression line
    from scipy import stats
    slope, intercept, rv, pv, se = stats.linregress(df_v["var_alpha"], df_v["fairness_gap"])
    x_fit = np.linspace(0, grp["var"].max()*1.1, 100)
    ax.plot(x_fit, slope*x_fit + intercept, '--', color='gray', alpha=0.7, label=f'R²={rv**2:.3f}')
    ax.set_xlabel("Var(α)")
    ax.set_ylabel("Fairness Gap")
    ax.set_title("(a) Sensitivity to Displacement Variance")
    ax.legend()

    # (b) Fairness vs system load
    df_l = pd.read_csv(RESDIR / "ablation_load.csv")
    grp = df_l.groupby("target_utilization").agg(
        j_mean=("jain_effective", "mean"), j_std=("jain_effective", "std")).reset_index()
    ax = axes[0, 1]
    ax.errorbar(grp["target_utilization"], grp["j_mean"], yerr=grp["j_std"],
               fmt='o-', color=PALETTE[1], linewidth=2, capsize=5)
    ax.set_xlabel("Target CPU Utilization")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("(b) Sensitivity to System Load")

    # (c) Fairness vs num cores
    df_c = pd.read_csv(RESDIR / "ablation_cores.csv")
    grp = df_c.groupby("M").agg(
        j_mean=("jain_effective", "mean"), j_std=("jain_effective", "std")).reset_index()
    ax = axes[1, 0]
    ax.errorbar(grp["M"], grp["j_mean"], yerr=grp["j_std"],
               fmt='o-', color=PALETTE[2], linewidth=2, capsize=5)
    ax.set_xscale('log', base=2)
    ax.set_xlabel("Number of Cores (M)")
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("(c) Sensitivity to Number of Cores")

    # (d) CCP ablation
    df_a = pd.read_csv(RESDIR / "ablation_ccp_components.csv")
    grp = df_a.groupby("ablation").agg(
        j_mean=("jain_effective", "mean"), j_std=("jain_effective", "std")).reset_index()
    order = ["no_ccp", "no_propagation", "no_tagging", "full_ccp"]
    labels = ["No CCP", "No\nPropagation", "No\nTagging", "Full CCP"]
    grp_ordered = grp.set_index("ablation").loc[order].reset_index()
    ax = axes[1, 1]
    bars = ax.bar(range(len(order)), grp_ordered["j_mean"], yerr=grp_ordered["j_std"],
                 capsize=5, color=[PALETTE[3], PALETTE[4], PALETTE[5], PALETTE[0]], alpha=0.85,
                 edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("(d) CCP Component Ablation")
    ax.set_ylim(0.5, 1.05)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig7_sensitivity.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig7_sensitivity.png", dpi=150)
    plt.close()
    print("  Figure 7: sensitivity analysis")


def fig8_analytical_validation():
    """Scatter plot of analytical vs simulation fairness."""
    df = pd.read_csv(RESDIR / "trace_validation.csv")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    scenarios = df["trace_scenario"].unique()
    markers = ["o", "s", "^", "D"]
    for i, sc in enumerate(scenarios):
        s = df[df["trace_scenario"] == sc]
        ax.scatter(s["jain_analytical"], s["jain_effective"], marker=markers[i % len(markers)],
                  s=100, label=sc.replace("_", " ").title(), zorder=5)

    # y=x reference line
    lims = [min(df["jain_analytical"].min(), df["jain_effective"].min()) - 0.05,
            max(df["jain_analytical"].max(), df["jain_effective"].max()) + 0.02]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='y = x (perfect prediction)')

    # R-squared
    from scipy import stats
    slope, intercept, rv, pv, se = stats.linregress(df["jain_analytical"], df["jain_effective"])
    ax.set_xlabel("Analytical Prediction (J)")
    ax.set_ylabel("Simulation Result (J)")
    ax.set_title(f"Analytical Model Validation (R² = {rv**2:.4f})")
    ax.legend(fontsize=9)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig8_analytical_validation.pdf", dpi=300)
    fig.savefig(FIGDIR / "fig8_analytical_validation.png", dpi=150)
    plt.close()
    print("  Figure 8: analytical validation")


def main():
    print("Generating figures...")
    fig1_displacement_ratios()
    fig2_fairness_scaling()
    fig3_scheduler_vs_reality()
    fig4_cgroup_leakage()
    fig5_ccp_evaluation()
    fig6_ccp_convergence()
    fig7_sensitivity()
    fig8_analytical_validation()
    print(f"\nAll figures saved to {FIGDIR}/")


if __name__ == "__main__":
    main()
