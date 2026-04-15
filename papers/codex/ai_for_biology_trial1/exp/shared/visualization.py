from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def save_main_table(df: pd.DataFrame, out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)


def plot_difference_intervals(df: pd.DataFrame, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    panels = [
        ("Residualized Ridge vs Non-residualized Ridge", "residualization"),
        ("Best ReSRP vs Best Residualized Non-retrieval", "retrieval"),
    ]
    for ax, (title, key) in zip(axes, panels):
        sub = df[df["comparison_type"] == key]
        xs = range(len(sub))
        ax.errorbar(
            list(xs),
            sub["pearson_mean_difference"],
            yerr=[
                sub["pearson_mean_difference"] - sub["pearson_ci_low"],
                sub["pearson_ci_high"] - sub["pearson_mean_difference"],
            ],
            fmt="o",
            label="Pearson diff",
        )
        ax.errorbar(
            list(xs),
            sub["rmse_mean_difference"],
            yerr=[
                sub["rmse_mean_difference"] - sub["rmse_ci_low"],
                sub["rmse_ci_high"] - sub["rmse_mean_difference"],
            ],
            fmt="s",
            label="RMSE diff",
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels(sub["dataset"], rotation=20)
        ax.set_title(title)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_descriptor_contingency(df: pd.DataFrame, out_base: Path) -> None:
    ensure_dir(out_base.parent)
    for dataset, sub in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sub["descriptor_variant"], sub["perturbed_reference_pearson"])
        ax.set_ylabel("Perturbed-reference Pearson")
        ax.set_title(f"{dataset} descriptor contingency")
        fig.tight_layout()
        out = out_base.parent / f"{out_base.stem}_{dataset}"
        fig.savefig(out.with_suffix(".png"), dpi=200)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)

