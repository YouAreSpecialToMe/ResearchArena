from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from .config import ABLATIONS, FIGURE_ROOT, MAIN_METHODS, OUTPUT_ROOT, SEEDS


def paired_bootstrap(values_a: np.ndarray, values_b: np.ndarray, num_samples: int = 1000):
    rng = np.random.default_rng(0)
    diffs = []
    n = len(values_a)
    for _ in range(num_samples):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(np.mean(values_a[idx] - values_b[idx])))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def aggregate():
    metric_files = sorted((OUTPUT_ROOT / "metrics").glob("*.json"))
    run_rows = []
    for file in metric_files:
        payload = json.loads(file.read_text())
        if "metrics" not in payload:
            continue
        row = {
            "experiment": payload["experiment"],
            "dataset": payload["dataset"],
            "method": payload["method"],
            "seed": payload["seed"],
            "attack_artifact_path": payload.get("attack_artifact_path"),
            "attack_summary_path": payload.get("attack_summary_path"),
            "trace_path": payload.get("trace_path"),
            "refresh_trace_path": payload.get("refresh_trace_path"),
        }
        row.update(payload["metrics"])
        run_rows.append(row)
    if not run_rows:
        raise RuntimeError("No run metrics found.")
    df = pd.DataFrame(run_rows).sort_values(["dataset", "method", "seed"])
    df.to_csv(OUTPUT_ROOT / "metrics" / "main_runs.csv", index=False)

    grouped = df.groupby(["dataset", "method"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in {"seed"}]
    summary = grouped[numeric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    summary[summary["method"].isin(MAIN_METHODS)].to_csv(OUTPUT_ROOT / "tables" / "main_results.csv", index=False)
    summary[summary["method"].isin(ABLATIONS)].to_csv(OUTPUT_ROOT / "tables" / "ablation_results.csv", index=False)
    df[
        [
            "dataset",
            "method",
            "seed",
            "runtime_minutes",
            "peak_gpu_memory_mb",
            "attack_artifact_path",
            "attack_summary_path",
            "trace_path",
            "refresh_trace_path",
        ]
    ].to_csv(
        OUTPUT_ROOT / "tables" / "runtime_budget.csv",
        index=False,
    )

    subgroup_rows = []
    class_rows = []
    for trace_file in sorted((OUTPUT_ROOT / "traces").glob("*_*.parquet")):
        if trace_file.name.endswith("_refreshes.parquet"):
            continue
        parts = trace_file.stem.split("_")
        dataset = parts[0]
        seed = int(parts[-1])
        method = "_".join(parts[1:-1])
        frame = pd.read_parquet(trace_file)
        q = frame["q_i"].to_numpy()
        pct = _percentile_rank(q)
        groups = {
            "bottom": pct < 0.1,
            "middle": (pct >= 0.5) & (pct < 0.7),
            "top": pct >= 0.9,
        }
        refresh_path = OUTPUT_ROOT / "traces" / f"{dataset}_{method}_{seed}_refreshes.parquet"
        if refresh_path.exists():
            refresh = pd.read_parquet(refresh_path)
        else:
            refresh = pd.DataFrame(columns=["epoch", "sample_id", "class_id"])
        if "sample_id" not in refresh.columns:
            refresh["sample_id"] = pd.Series(dtype=int)
        if "class_id" not in refresh.columns:
            refresh["class_id"] = pd.Series(dtype=int)
        if "epoch" not in refresh.columns:
            refresh["epoch"] = pd.Series(dtype=int)
        intervention_frequency = refresh.groupby("sample_id").size() if not refresh.empty else pd.Series(dtype=float)
        class_frequency = refresh.groupby("class_id").size() if not refresh.empty else pd.Series(dtype=float)
        for name, mask in groups.items():
            group_frame = frame.loc[mask]
            subgroup_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "group": name,
                    "num_samples": int(mask.sum()),
                    "accuracy": float((group_frame["pred"] == group_frame["label"]).mean()) if mask.any() else float("nan"),
                    "avg_attack_score": float(group_frame["final_attack_score"].mean()) if mask.any() else float("nan"),
                    "avg_class_attack_score": float(group_frame["class_attack_score"].mean()) if mask.any() else float("nan"),
                    "avg_loss": float(group_frame["loss"].mean()) if mask.any() else float("nan"),
                    "primary_tpr_at_1_fpr": float(group_frame["primary_member_pred"].mean()) if mask.any() else float("nan"),
                    "class_tpr_at_1_fpr": float(group_frame["class_member_pred"].mean()) if mask.any() else float("nan"),
                    "loss_tpr_at_1_fpr": float(group_frame["loss_member_pred"].mean()) if mask.any() else float("nan"),
                    "mean_intervention_frequency": float(intervention_frequency.reindex(group_frame["sample_id"]).fillna(0.0).mean())
                    if mask.any()
                    else float("nan"),
                }
            )
        class_hist = frame["label"].value_counts().sort_index()
        for class_id, class_size in class_hist.items():
            intervention_count = int(class_frequency.get(class_id, 0))
            class_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "group": "per_class",
                    "class_id": int(class_id),
                    "class_size": int(class_size),
                    "intervention_frequency": float(intervention_count / max(1, class_size * max(1, refresh["epoch"].nunique())))
                    if "epoch" in refresh
                    else 0.0,
                }
            )
    subgroup_df = pd.concat([pd.DataFrame(subgroup_rows), pd.DataFrame(class_rows)], ignore_index=True, sort=False)
    subgroup_df.to_csv(OUTPUT_ROOT / "tables" / "subgroup_results.csv", index=False)

    study_scope = {
        "positioning": "budget-matched empirical test of whether early artifact forecasts improve selective intervention over random, loss-only, and lightweight global defenses",
        "precursors": ["Long et al.", "MIST", "AdaMixup", "MIAShield", "RelaxLoss"],
        "seeds": SEEDS,
        "negative_result_policy": "If targeted_forecast fails the matched-budget criteria, report the study as a negative result about forecast actionability rather than claim a new defense benefit.",
    }
    (OUTPUT_ROOT / "tables" / "study_scope.json").write_text(json.dumps(study_scope, indent=2))

    bootstrap_df = build_paired_bootstrap_table(df)
    bootstrap_df.to_csv(OUTPUT_ROOT / "tables" / "paired_bootstrap.csv", index=False)

    make_plots(df)
    results = build_root_results(df, bootstrap_df)
    (Path(OUTPUT_ROOT).parent / "results.json").write_text(json.dumps(results, indent=2))
    for method, method_frame in summary.groupby("method"):
        out_path = OUTPUT_ROOT.parent / "exp" / method / "results.json"
        payload = method_frame.to_dict(orient="records")
        out_path.write_text(json.dumps(payload, indent=2))
    return df, summary, results


def make_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    matched_subset = df[df["method"].isin(["targeted_random", "targeted_loss_only", "targeted_forecast"])]
    y_min = float(matched_subset["worst_decile_leakage"].min()) if not matched_subset.empty else 0.0
    y_max = float(matched_subset["worst_decile_leakage"].max()) if not matched_subset.empty else 1.0
    padding = max(0.01, 0.05 * (y_max - y_min + 1e-6))
    method_labels = {
        "erm": "ERM",
        "global_mixup": "Global mixup",
        "relaxloss": "RelaxLoss",
        "targeted_random": "Targeted-random",
        "targeted_loss_only": "Targeted-loss-only",
        "targeted_forecast": "Targeted-forecast",
        "forecast_single_artifact": "Single-artifact",
        "forecast_no_refresh": "No refresh",
        "forecast_targeted_penalty": "Forecast + penalty",
    }
    method_colors = {
        "erm": "#4c72b0",
        "global_mixup": "#dd8452",
        "relaxloss": "#55a868",
        "targeted_random": "#8172b3",
        "targeted_loss_only": "#c44e52",
        "targeted_forecast": "#000000",
        "forecast_single_artifact": "#937860",
        "forecast_no_refresh": "#da8bc3",
        "forecast_targeted_penalty": "#8c8c8c",
    }
    for dataset, data in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(7, 5))
        runtime = data["runtime_minutes"]
        size = 120 + 700 * (runtime - runtime.min()) / max(1e-8, runtime.max() - runtime.min())
        for _, row in data.iterrows():
            ax.scatter(
                row["test_accuracy"],
                row["primary_tpr_at_1_fpr"],
                s=float(size.loc[row.name]),
                color=method_colors.get(row["method"], "#808080"),
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
            )
        label_offsets = {
            "erm": (0.002, -0.00010),
            "relaxloss": (0.002, -0.00035),
            "targeted_random": (0.002, 0.00022),
            "targeted_loss_only": (0.002, 0.00005),
            "targeted_forecast": (0.002, -0.00055),
        }
        for _, row in data.iterrows():
            if row["method"] in label_offsets:
                dx, dy = label_offsets[row["method"]]
                text = f"{method_labels[row['method']]}\n({row['test_accuracy'] * 100:.2f}%, {row['primary_tpr_at_1_fpr'] * 100:.2f}%)"
                ax.annotate(
                    text,
                    (row["test_accuracy"], row["primary_tpr_at_1_fpr"]),
                    xytext=(row["test_accuracy"] + dx, row["primary_tpr_at_1_fpr"] + dy),
                    textcoords="data",
                    fontsize=7.5,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
                )
        legend_values = np.linspace(runtime.min(), runtime.max(), 3)
        legend_sizes = 120 + 700 * (legend_values - runtime.min()) / max(1e-8, runtime.max() - runtime.min())
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"{value:.2f} min",
                markerfacecolor="#808080",
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=np.sqrt(s / np.pi),
            )
            for value, s in zip(legend_values, legend_sizes)
        ]
        ax.legend(handles=legend_handles, title="Runtime", loc="upper right", frameon=True, fontsize=8, title_fontsize=8)
        ax.annotate("better privacy", xy=(0.02, 0.98), xytext=(0.02, 0.82), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.0}, fontsize=8, va="top")
        ax.annotate("better utility", xy=(0.98, 0.05), xytext=(0.76, 0.05), xycoords="axes fraction", textcoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.0}, fontsize=8, va="center")
        ax.set_xlabel("Test accuracy")
        ax.set_ylabel("Primary LiRA-lite TPR@1% FPR")
        ax.set_title(f"{dataset}: privacy-utility trade-off")
        for ext in ["png", "pdf"]:
            fig.savefig(OUTPUT_ROOT / "plots" / f"{dataset}_privacy_utility.{ext}", bbox_inches="tight")
            fig.savefig(FIGURE_ROOT / f"{dataset}_privacy_utility.{ext}", bbox_inches="tight")
        plt.close(fig)

        subset = data[data["method"].isin(["targeted_random", "targeted_loss_only", "targeted_forecast"])]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(subset, x="method", y="worst_decile_leakage", ax=ax)
        ax.set_ylabel("Worst-decile leakage (TPR@1% FPR)")
        ax.set_xlabel("")
        ax.set_title(f"{dataset}: matched-budget comparison")
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.tick_params(axis="x", rotation=15)
        for ext in ["png", "pdf"]:
            fig.savefig(OUTPUT_ROOT / "plots" / f"{dataset}_matched_budget.{ext}", bbox_inches="tight")
            fig.savefig(FIGURE_ROOT / f"{dataset}_matched_budget.{ext}", bbox_inches="tight")
        plt.close(fig)

        tf = data[data["method"] == "targeted_forecast"]
        if not tf.empty:
            decile_rows = []
            overlaps = []
            for seed in tf["seed"].tolist():
                trace = pd.read_parquet(OUTPUT_ROOT / "traces" / f"{dataset}_targeted_forecast_{seed}.parquet")
                trace = trace[["q_i", "final_attack_score"]].copy()
                trace["decile"] = pd.qcut(trace["q_i"], 10, labels=False, duplicates="drop")
                decile_summary = (
                    trace.groupby("decile", as_index=False)["final_attack_score"]
                    .mean()
                    .assign(seed=seed)
                )
                decile_rows.append(decile_summary)
                refresh = pd.read_parquet(OUTPUT_ROOT / "traces" / f"{dataset}_targeted_forecast_{seed}_refreshes.parquet")
                if not refresh.empty:
                    refresh_sets = [set(chunk["sample_id"].tolist()) for _, chunk in refresh.groupby("epoch")]
                    for i, val in enumerate(_jaccards(refresh_sets)):
                        overlaps.append({"seed": seed, "step": i + 1, "jaccard": val})
            if decile_rows:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                decile_df = pd.concat(decile_rows, ignore_index=True)
                decile_plot = (
                    decile_df.groupby("decile")["final_attack_score"]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                axes[0].plot(
                    decile_plot["decile"],
                    decile_plot["mean"],
                    marker="o",
                    linewidth=2.0,
                    color=method_colors["targeted_forecast"],
                )
                axes[0].fill_between(
                    decile_plot["decile"],
                    decile_plot["mean"] - decile_plot["std"].fillna(0.0),
                    decile_plot["mean"] + decile_plot["std"].fillna(0.0),
                    color=method_colors["targeted_forecast"],
                    alpha=0.15,
                )
                axes[0].set_title("Warm-up forecast decile vs final attack score")
                axes[0].set_xlabel("Warm-up forecast decile")
                axes[0].set_ylabel("Mean final primary attack score")
                summary_stats = {
                    "spearman": float(tf["spearman_q_attack"].mean()),
                    "precision_at_10": float(tf["precision_at_10"].mean()),
                    "mean_refresh_jaccard": float(tf["mean_refresh_jaccard"].mean()),
                }
                axes[0].text(
                    0.03,
                    0.97,
                    (
                        f"Spearman={summary_stats['spearman']:.3f}\n"
                        f"P@10={summary_stats['precision_at_10']:.3f}"
                    ),
                    transform=axes[0].transAxes,
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
                )
                if overlaps:
                    overlap_df = pd.DataFrame(overlaps)
                    overlap_plot = overlap_df.groupby("step")["jaccard"].agg(["mean", "std"]).reset_index()
                    axes[1].plot(
                        overlap_plot["step"],
                        overlap_plot["mean"],
                        marker="o",
                        linewidth=2.0,
                        color=method_colors["targeted_forecast"],
                    )
                    axes[1].fill_between(
                        overlap_plot["step"],
                        overlap_plot["mean"] - overlap_plot["std"].fillna(0.0),
                        overlap_plot["mean"] + overlap_plot["std"].fillna(0.0),
                        color=method_colors["targeted_forecast"],
                        alpha=0.15,
                    )
                    axes[1].axhline(summary_stats["mean_refresh_jaccard"], linestyle="--", linewidth=1.0, color="#c44e52")
                    axes[1].text(
                        0.03,
                        0.97,
                        f"Mean Jaccard={summary_stats['mean_refresh_jaccard']:.3f}",
                        transform=axes[1].transAxes,
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
                    )
                axes[1].set_title("Risky-set Jaccard overlap")
                axes[1].set_xlabel("Refresh transition")
                axes[1].set_ylabel("Jaccard")
                axes[1].set_xticks([1, 2])
                for ext in ["png", "pdf"]:
                    fig.savefig(OUTPUT_ROOT / "plots" / f"{dataset}_forecast_quality.{ext}", bbox_inches="tight")
                    fig.savefig(FIGURE_ROOT / f"{dataset}_forecast_quality.{ext}", bbox_inches="tight")
                plt.close(fig)


def _jaccards(sets: list[set[int]]) -> list[float]:
    out = []
    for a, b in zip(sets, sets[1:]):
        out.append(float(len(a & b) / max(1, len(a | b))))
    return out


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=False)
    return ranks


def build_paired_bootstrap_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparators = ["erm", "global_mixup", "relaxloss", "targeted_random", "targeted_loss_only"]
    metrics = ["worst_decile_leakage", "privacy_disparity", "test_accuracy"]
    for dataset, data in df.groupby("dataset"):
        tf = data[data["method"] == "targeted_forecast"].sort_values("seed")
        if tf.empty:
            continue
        for comparator in comparators:
            comp = data[data["method"] == comparator].sort_values("seed")
            if comp.empty:
                continue
            merged = tf[["seed", *metrics]].merge(
                comp[["seed", *metrics]],
                on="seed",
                suffixes=("_forecast", "_comparator"),
            )
            for metric in metrics:
                values_a = merged[f"{metric}_forecast"].to_numpy()
                values_b = merged[f"{metric}_comparator"].to_numpy()
                lo, hi = paired_bootstrap(values_a, values_b)
                rows.append(
                    {
                        "dataset": dataset,
                        "comparator": comparator,
                        "metric": metric,
                        "forecast_minus_comparator_mean": float((values_a - values_b).mean()),
                        "ci95_low": lo,
                        "ci95_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def build_root_results(df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> dict:
    results = {"datasets": {}, "success_checks": {}, "paired_bootstrap": [], "narrative": {}}
    for dataset, data in df.groupby("dataset"):
        by_method = {}
        for method, frame in data.groupby("method"):
            by_method[method] = {
                metric: {
                    "mean": float(frame[metric].mean()),
                    "std": float(frame[metric].std(ddof=1)),
                }
                for metric in [
                    "primary_tpr_at_1_fpr",
                    "class_conditional_tpr_at_1_fpr",
                    "loss_tpr_at_1_fpr",
                    "worst_decile_leakage",
                    "privacy_disparity",
                    "test_accuracy",
                    "runtime_minutes",
                    "peak_gpu_memory_mb",
                    "spearman_q_attack",
                    "precision_at_10",
                    "mean_refresh_jaccard",
                    "best_val_accuracy",
                    "best_val_loss",
                ]
            }
        results["datasets"][dataset] = by_method
        tf = data[data["method"] == "targeted_forecast"]
        erm = data[data["method"] == "erm"]
        gm = data[data["method"] == "global_mixup"]
        tl = data[data["method"] == "targeted_loss_only"]
        tr = data[data["method"] == "targeted_random"]
        results["success_checks"][dataset] = {
            "primary_success_over_erm_or_mixup": bool(
                (
                    (tf["worst_decile_leakage"].mean() < erm["worst_decile_leakage"].mean())
                    or (tf["privacy_disparity"].mean() < erm["privacy_disparity"].mean())
                )
                and (
                    (tf["worst_decile_leakage"].mean() < gm["worst_decile_leakage"].mean())
                    or (tf["privacy_disparity"].mean() < gm["privacy_disparity"].mean())
                )
            )
            if not tf.empty and not erm.empty and not gm.empty
            else False,
            "comparator_success_vs_targeted_loss_only": bool(
                tf["worst_decile_leakage"].mean() < tl["worst_decile_leakage"].mean()
            )
            if not tf.empty and not tl.empty
            else False,
            "comparator_success_vs_targeted_random": bool(
                tf["worst_decile_leakage"].mean() < tr["worst_decile_leakage"].mean()
            )
            if not tf.empty and not tr.empty
            else False,
            "accuracy_tolerance_met": bool(
                tf["test_accuracy"].mean() >= erm["test_accuracy"].mean() - 0.015
            )
            if not tf.empty and not erm.empty
            else False,
        }
    purchase = results["datasets"].get("purchase100", {})
    cifar = results["datasets"].get("cifar10", {})
    purchase_tf = purchase.get("targeted_forecast", {})
    purchase_tl = purchase.get("targeted_loss_only", {})
    purchase_tr = purchase.get("targeted_random", {})
    cifar_tf = cifar.get("targeted_forecast", {})
    cifar_gm = cifar.get("global_mixup", {})
    cifar_tr = cifar.get("targeted_random", {})
    results["narrative"] = {
        "headline": "Negative result: forecast-driven targeting did not beat the pre-registered matched-budget baselines under the primary criteria.",
        "summary": (
            "Across Purchase100 and CIFAR-10, targeted_forecast failed the comparator success criterion against the matched-budget baselines."
        ),
        "details": [
            (
                "On Purchase100, targeted_forecast worst-decile leakage was "
                f"{purchase_tf.get('worst_decile_leakage', {}).get('mean', float('nan')):.5f}, "
                "worse than targeted_loss_only at "
                f"{purchase_tl.get('worst_decile_leakage', {}).get('mean', float('nan')):.5f} "
                "and targeted_random at "
                f"{purchase_tr.get('worst_decile_leakage', {}).get('mean', float('nan')):.5f}."
            ),
            (
                "On CIFAR-10, targeted_forecast worst-decile leakage was "
                f"{cifar_tf.get('worst_decile_leakage', {}).get('mean', float('nan')):.5f}, "
                "worse than targeted_random at "
                f"{cifar_tr.get('worst_decile_leakage', {}).get('mean', float('nan')):.5f} "
                "and global_mixup at "
                f"{cifar_gm.get('worst_decile_leakage', {}).get('mean', float('nan')):.5f}."
            ),
            "Likely confounds include modest forecast quality, instability in the refreshed risky set, and a weak gap between the composite forecast and simpler loss-based ranking under the fixed 10% budget.",
            "The results support a narrow interpretation: early artifact forecasts were measurable, but under this protocol they were not actionable enough to justify stronger claims beyond a negative benchmark finding.",
        ],
    }
    results["paired_bootstrap"] = bootstrap_df.to_dict(orient="records")
    return results


if __name__ == "__main__":
    aggregate()
