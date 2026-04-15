from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.shared.config import FIGURES_DIR, RESULTS_DIR, THREAD_ENV
from exp.shared.io import read_json


def save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_main_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "all_runs.csv")
    df["run_label"] = df["run_label"].fillna(df["method"])
    return df[(df["alpha"] == 0.10)].copy()


def main() -> None:
    os.environ.update(THREAD_ENV)
    sns.set_theme(style="whitegrid")
    df = _load_main_results()
    summary = pd.read_csv(RESULTS_DIR / "summary_table.csv")
    summary.to_csv(RESULTS_DIR / "plot_data.csv", index=False)

    focus = summary[summary["method"].isin(["chip_rlcp", "gmm_rlcp", "batch_mcp", "split_cp"])].copy()
    datasets = sorted(focus["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = focus[focus["dataset"] == dataset]
        ax.errorbar(
            sub["mean_set_size_mean"],
            sub["worst_external_group_coverage_mean"],
            xerr=(sub["mean_set_size_ci_high"] - sub["mean_set_size_mean"]).to_numpy(),
            yerr=(sub["worst_external_group_coverage_ci_high"] - sub["worst_external_group_coverage_mean"]).to_numpy(),
            fmt="none",
            ecolor="gray",
            alpha=0.6,
        )
        sns.scatterplot(data=sub, x="mean_set_size_mean", y="worst_external_group_coverage_mean", hue="method", style="method", s=90, ax=ax)
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("Mean set size")
        ax.set_ylabel("Worst external-group coverage")
    save_fig(fig, "worst_group_vs_set_size")

    chip_runtime = df[df["run_label"] == "chip_rlcp"][["dataset", "seed", "fit_time_sec", "total_runtime_sec"]].copy()
    for _, row in chip_runtime.iterrows():
        result = read_json(ROOT / "exp" / f"{row['dataset']}_seed{int(row['seed'])}_chip_rlcp_alpha0p1" / "summary.json")
        row["num_groups"] = result.get("num_groups", 0)
        row["median_active_groups"] = result.get("median_active_groups", 0)
    runtime_rows = []
    for _, row in chip_runtime.iterrows():
        summary_payload = read_json(ROOT / "exp" / f"{row['dataset']}_seed{int(row['seed'])}_chip_rlcp_alpha0p1" / "summary.json")
        runtime_rows.append(
            {
                "dataset": row["dataset"],
                "seed": row["seed"],
                "fit_time_sec": row["fit_time_sec"],
                "total_runtime_sec": row["total_runtime_sec"],
                "num_groups": summary_payload.get("num_groups", 0),
                "median_active_groups": summary_payload.get("median_active_groups", 0),
            }
        )
    runtime_df = pd.DataFrame(runtime_rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = [
        ("fit_time_sec", 30 * 60),
        ("num_groups", 64),
        ("median_active_groups", 24),
        ("total_runtime_sec", None),
    ]
    for ax, (metric, threshold) in zip(axes.ravel(), metrics):
        sns.barplot(data=runtime_df, x="dataset", y=metric, hue="seed", ax=ax)
        if threshold is not None:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1)
        ax.set_title(metric)
    save_fig(fig, "runtime_budget")

    synth_rows = []
    for method in ["split_cp", "gmm_rlcp", "chip_rlcp", "oracle_rlcp"]:
        for seed in [11, 22, 33]:
            payload = read_json(ROOT / "exp" / f"synthetic_seed{seed}_{method}_alpha0p1" / "results.json")
            for item in payload["groups"]:
                if item["group_family"] in {"coarse", "fine"}:
                    synth_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "group_family": item["group_family"],
                            "group": item["group"],
                            "coverage": item["coverage"],
                        }
                    )
    synth_df = pd.DataFrame(synth_rows)
    synth_plot = (
        synth_df.groupby(["group_family", "group", "method"], as_index=False)["coverage"]
        .mean()
        .sort_values(["group_family", "group", "method"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, family in zip(axes, ["coarse", "fine"]):
        sub = synth_plot[synth_plot["group_family"] == family]
        sns.barplot(data=sub, x="group", y="coverage", hue="method", ax=ax)
        ax.axhline(0.90, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Synthetic {family} groups")
    save_fig(fig, "synthetic_oracle_groupwise")

    anuran_rows = []
    for method in ["split_cp", "gmm_rlcp", "batch_mcp", "chip_rlcp"]:
        for seed in [11, 22, 33]:
            payload = read_json(ROOT / "exp" / f"anuran_seed{seed}_{method}_alpha0p1" / "results.json")
            for item in payload["groups"]:
                if item["group_family"] in {"Family", "Genus"}:
                    anuran_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "group_family": item["group_family"],
                            "group": item["group"],
                            "coverage_deficit": 0.90 - item["coverage"],
                            "mean_set_size": payload["mean_set_size"],
                        }
                    )
    anuran_df = pd.DataFrame(anuran_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, family in zip(axes, ["Family", "Genus"]):
        sub = anuran_df[anuran_df["group_family"] == family]
        agg = sub.groupby("method", as_index=False).agg({"coverage_deficit": "mean", "mean_set_size": "mean"})
        sns.barplot(data=agg, x="method", y="coverage_deficit", ax=ax, color="#5B8FF9")
        ax2 = ax.twinx()
        sns.pointplot(data=agg, x="method", y="mean_set_size", ax=ax2, color="#D46B08")
        ax.set_title(f"Anuran {family}")
        ax.set_ylabel("Mean coverage deficit")
        ax2.set_ylabel("Mean set size")
        ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "anuran_taxonomy")

    parity = summary[summary["method"].isin(["chip_rlcp", "gmm_rlcp"])].copy()
    pivot_cov = parity.pivot(index="dataset", columns="method", values="worst_external_group_coverage_mean").reset_index()
    pivot_rt = parity.pivot(index="dataset", columns="method", values="total_runtime_sec_mean").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=pivot_cov, x="gmm_rlcp", y="chip_rlcp", hue="dataset", s=100, ax=axes[0])
    axes[0].plot([pivot_cov[["gmm_rlcp", "chip_rlcp"]].min().min(), pivot_cov[["gmm_rlcp", "chip_rlcp"]].max().max()],
                 [pivot_cov[["gmm_rlcp", "chip_rlcp"]].min().min(), pivot_cov[["gmm_rlcp", "chip_rlcp"]].max().max()],
                 linestyle="--", color="black")
    axes[0].set_title("Worst-group coverage parity")
    sns.scatterplot(data=pivot_rt, x="gmm_rlcp", y="chip_rlcp", hue="dataset", s=100, ax=axes[1])
    axes[1].plot([pivot_rt[["gmm_rlcp", "chip_rlcp"]].min().min(), pivot_rt[["gmm_rlcp", "chip_rlcp"]].max().max()],
                 [pivot_rt[["gmm_rlcp", "chip_rlcp"]].min().min(), pivot_rt[["gmm_rlcp", "chip_rlcp"]].max().max()],
                 linestyle="--", color="black")
    axes[1].set_title("Runtime parity")
    save_fig(fig, "fallback_parity")


if __name__ == "__main__":
    main()
