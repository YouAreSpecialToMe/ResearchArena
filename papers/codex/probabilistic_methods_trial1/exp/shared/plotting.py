from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from exp.shared.core import ensure_dir


def plot_null_histograms(df: pd.DataFrame, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ["cov_pvalue", "tail_pvalue", "global_pvalue"]):
        ax.hist(df[metric], bins=12, color="#376996", edgecolor="black")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("p-value")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_power(df: pd.DataFrame, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(12, 5))
    y_col = "reject_at_0_05" if "reject_at_0_05" in df.columns else "reject_at_0_05_mean"
    err_col = "reject_at_0_05_ci95" if "reject_at_0_05_ci95" in df.columns else None
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("condition")
        ax.errorbar(
            sub["condition"],
            sub[y_col],
            yerr=None if err_col is None else sub[err_col],
            marker="o",
            capsize=3,
            label=method,
        )
    ax.set_ylabel("Rejection rate @ 0.05")
    ax.set_xlabel("Condition")
    ax.set_title("Synthetic power")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_localization(df: pd.DataFrame, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    value_col = "top1" if "top1" in df.columns else "top1_mean"
    pivot = df.pivot_table(index="condition", columns="method", values=value_col)
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Top-1 recovery")
    ax.set_title("Pair localization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fixed_data(df: pd.DataFrame, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, sub in df.groupby("method"):
        ax.scatter(sub["cov_error"], sub["global_stat"], label=method)
    ax.set_xlabel("Posterior covariance Frobenius error")
    ax.set_ylabel("Diagnostic statistic")
    ax.set_title("Fixed-data posterior-SBC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_budget_sensitivity(df: pd.DataFrame, out_path: str | Path) -> None:
    ensure_dir(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, sub in df.groupby("method"):
        ax.errorbar(
            sub["runtime_minutes_mean"],
            sub["reject_at_0_05_mean"],
            xerr=sub.get("runtime_minutes_ci95"),
            yerr=sub.get("reject_at_0_05_ci95"),
            marker="o",
            capsize=3,
            label=method,
        )
        for _, row in sub.iterrows():
            ax.annotate(row["condition"], (row["runtime_minutes_mean"], row["reject_at_0_05_mean"]), fontsize=8)
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Rejection rate @ 0.05")
    ax.set_title("Budget sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
