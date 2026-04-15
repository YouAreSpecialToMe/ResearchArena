from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from exp.shared.config import ROOT
from exp.shared.runner import aggregate_all, run_ablations, run_audits, run_exact_validation, run_main_benchmark, save_table


def main() -> None:
    base = ROOT / "exp"
    figures = ROOT / "figures"
    tables = ROOT / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    main_df = run_main_benchmark(base / "main_benchmark")
    exact_df = run_exact_validation(base / "exact_validation")
    audits_df = run_audits(base / "audits")
    ablations_df = run_ablations(base / "ablations")

    summary = aggregate_all(
        {
            "main_benchmark": main_df,
            "exact_validation": exact_df,
            "audits": audits_df,
            "ablations": ablations_df,
        }
    )
    with (ROOT / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    main_summary = main_df[main_df["is_summary"] == True].copy()
    plot_df = main_summary[main_summary["method"].isin(["Myopic budgeted gain", "Horizon-2 additive-cost", "Proposed H=2 switching-aware"])]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=plot_df, x="budget", y="AUEC_partial", hue="method", style="switch_regime", estimator="mean", errorbar=("ci", 95), ax=ax)
    ax.set_title("Figure 1: AUEC vs Budget")
    fig.tight_layout()
    fig.savefig(figures / "figure1_auec_vs_budget.png", dpi=200)
    fig.savefig(figures / "figure1_auec_vs_budget.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    regime_df = plot_df.groupby(["budget", "switch_regime", "regime_label"]).size().reset_index(name="count")
    regime_df["fraction"] = regime_df["count"] / regime_df.groupby(["budget", "switch_regime"])["count"].transform("sum")
    sns.barplot(data=regime_df, x="switch_regime", y="fraction", hue="regime_label", ax=ax)
    ax.set_title("Figure 2: Regime frequencies")
    fig.tight_layout()
    fig.savefig(figures / "figure2_regime_map.png", dpi=200)
    fig.savefig(figures / "figure2_regime_map.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    exact_summary = exact_df[exact_df["is_summary"] == True].copy()
    sns.boxplot(data=exact_summary, x="switch_regime", y="v_b", hue="method", ax=ax)
    ax.set_title("Figure 3: Exact d=8 validation")
    fig.tight_layout()
    fig.savefig(figures / "figure3_exact_validation.png", dpi=200)
    fig.savefig(figures / "figure3_exact_validation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    audits_summary = audits_df.copy()
    if "TV_orient_error" in audits_summary and "DAG_KL_error" in audits_summary:
        sns.scatterplot(data=audits_summary, x="TV_orient_error", y="DAG_KL_error", hue="method", ax=ax)
        ax.set_title("Figure 4: Posterior audit")
        fig.tight_layout()
        fig.savefig(figures / "figure4_posterior_audit.png", dpi=200)
        fig.savefig(figures / "figure4_posterior_audit.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=ablations_df, x="ablation_tag", y="AUEC_partial", hue="switch_regime", ax=ax)
    ax.set_title("Figure 5: Ablation summary")
    fig.tight_layout()
    fig.savefig(figures / "figure5_ablation_summary.png", dpi=200)
    fig.savefig(figures / "figure5_ablation_summary.pdf")
    plt.close(fig)

    save_table(main_summary, tables / "table1_main_benchmark")
    save_table(exact_summary, tables / "table2_exact_validation")
    save_table(audits_df, tables / "table3_audits")
    save_table(ablations_df, tables / "table4_ablations")


if __name__ == "__main__":
    main()
