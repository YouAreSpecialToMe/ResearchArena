from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp.shared.utils import RUN_VERSION, write_json


DATASETS = ["Adamson", "Norman", "Replogle"]
MAIN_METHODS = [
    "always_baseline",
    "always_correct",
    "classifier_gate@40",
    "uncertainty_gate@40",
    "gain_regressor@40",
    "conformal_gate@40",
]
ROUTER_METHODS = ["uncertainty_gate", "gain_regressor", "conformal_gate"]
ACCEPTANCE_TAGS = [20, 40, 60]


def collect_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int, str], dict[str, np.ndarray]]]:
    metric_rows = []
    per_rows = []
    runtime_rows = []
    score_payloads: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for config_path in root.glob("exp/*/*/seed_*/config.json"):
        config = json.loads(config_path.read_text())
        if config.get("run_version") != RUN_VERSION:
            continue
        run_dir = config_path.parent
        metrics = json.loads((run_dir / "metrics.json").read_text())
        runtime = json.loads((run_dir / "runtime.json").read_text())
        dataset = config["dataset"]
        seed = config["seed"]
        tag = config["tag"]
        predictions = np.load(run_dir / "predictions.npz")
        score_payloads[(dataset, seed, tag)] = {key: predictions[key] for key in predictions.files}
        for method, vals in metrics.items():
            if not isinstance(vals, dict) or "all_gene_rmse" not in vals:
                continue
            metric_rows.append({"dataset": dataset, "seed": seed, "tag": tag, "method": method, **vals})
        runtime_rows.append({"dataset": dataset, "seed": seed, "tag": tag, **runtime})
        per_df = pd.read_csv(run_dir / "per_perturbation.csv")
        per_df["dataset"] = dataset
        per_df["seed"] = seed
        per_df["tag"] = tag
        per_rows.append(per_df)
    return pd.DataFrame(metric_rows), pd.concat(per_rows, ignore_index=True), pd.DataFrame(runtime_rows), score_payloads


def summarize_metrics(metrics_df: pd.DataFrame, runtime_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runtime_group = runtime_df.groupby(["dataset", "tag"])
    for (dataset, tag, method), grp in metrics_df.groupby(["dataset", "tag", "method"]):
        rec = {"dataset": dataset, "tag": tag, "method": method}
        for col in [
            "all_gene_rmse",
            "top_de_rmse",
            "all_gene_pearson_delta",
            "top_de_pearson_delta",
            "mean_gain",
            "median_gain",
            "pathway_corr",
            "top20_overlap",
            "sign_accuracy",
            "selective_auc",
            "accepted_fraction",
            "full_test_all_gene_rmse",
            "full_test_top_de_rmse",
            "full_test_mean_gain",
        ]:
            rec[f"{col}_mean"] = float(grp[col].mean())
            rec[f"{col}_std"] = float(grp[col].std(ddof=0))
        n = max(1, len(grp))
        rec["mean_gain_ci95"] = float(1.96 * grp["mean_gain"].std(ddof=0) / np.sqrt(n))
        rec["seed_count"] = int(n)
        if (dataset, tag) in runtime_group.groups:
            rgrp = runtime_group.get_group((dataset, tag))
            rec["runtime_seconds_mean"] = float(rgrp["total_runtime_seconds"].mean())
            rec["runtime_seconds_std"] = float(rgrp["total_runtime_seconds"].std(ddof=0))
            rec["peak_gpu_bytes_mean"] = float(rgrp["peak_gpu_bytes"].mean())
            rec["cpu_seconds_mean"] = float(rgrp["total_cpu_seconds"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def write_main_table(root: Path, summary_df: pd.DataFrame) -> None:
    fig_dir = root / "figures"
    rows = []
    main = summary_df[summary_df["tag"] == "main"]
    for dataset in DATASETS:
        for method_prefix in ["always_baseline", "always_correct", "classifier_gate", "uncertainty_gate", "gain_regressor", "conformal_gate"]:
            method40 = method_prefix if "always_" in method_prefix else f"{method_prefix}@40"
            row = main[(main["dataset"] == dataset) & (main["method"] == method40)]
            if row.empty:
                continue
            row = row.iloc[0].to_dict()
            for acc in [20, 60]:
                m = method_prefix if "always_" in method_prefix else f"{method_prefix}@{acc}"
                match = summary_df[(summary_df["dataset"] == dataset) & (summary_df["tag"] == "main") & (summary_df["method"] == m)]
                row[f"gain{acc}_mean"] = float(match["mean_gain_mean"].iloc[0]) if not match.empty else np.nan
                row[f"gain{acc}_std"] = float(match["mean_gain_std"].iloc[0]) if not match.empty else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(fig_dir / "main_table.csv", index=False)
    out.to_markdown(fig_dir / "main_table.md", index=False)


def write_calibration_counts(root: Path) -> None:
    rows = []
    for split_path in root.glob("splits/*/seed_*.json"):
        payload = json.loads(split_path.read_text())
        if payload.get("run_version") != RUN_VERSION:
            continue
        rows.append(
            {
                "dataset": payload["dataset"],
                "seed": payload["seed"],
                **payload["split_counts"],
                "calibration_underpowered": payload["calibration_underpowered"],
            }
        )
    pd.DataFrame(rows).to_csv(root / "figures" / "calibration_counts.csv", index=False)


def plot_selective_risk(root: Path, metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = metrics_df[(metrics_df["dataset"] == dataset) & (metrics_df["tag"] == "main") & (metrics_df["method"].isin(MAIN_METHODS + ["classifier_gate@20", "classifier_gate@60", "uncertainty_gate@20", "uncertainty_gate@60", "gain_regressor@20", "gain_regressor@60", "conformal_gate@20", "conformal_gate@60"]))]
        method_curves: dict[str, list[dict[str, float]]] = {}
        for _, row in subset.iterrows():
            base = row["method"].split("@")[0]
            curve = row["selective_curve"]
            if isinstance(curve, str):
                curve = json.loads(curve.replace("'", "\""))
            method_curves.setdefault(base, []).append(curve)
        for method, curves in method_curves.items():
            xs = np.array(sorted(int(k) for k in curves[0].keys()))
            ys = np.array([[curve[str(x)] for x in xs] for curve in curves], dtype=float)
            ax.plot(xs, ys.mean(axis=0), label=method)
        ax.set_title(dataset)
        ax.set_xlabel("Acceptance (%)")
        ax.set_ylabel("Accepted-set mean RMSE gain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "selective_risk.png", dpi=200)
    plt.close(fig)


def plot_matched_acceptance(root: Path, summary_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        x = np.arange(len(ACCEPTANCE_TAGS))
        width = 0.22
        for offset, method_prefix in zip([-width, 0.0, width], ROUTER_METHODS):
            vals = []
            errs = []
            for acc in ACCEPTANCE_TAGS:
                row = summary_df[(summary_df["dataset"] == dataset) & (summary_df["tag"] == "main") & (summary_df["method"] == f"{method_prefix}@{acc}")]
                vals.append(float(row["mean_gain_mean"].iloc[0]))
                errs.append(float(row["mean_gain_ci95"].iloc[0]))
            ax.bar(x + offset, vals, width=width, yerr=errs, label=method_prefix)
        ax.set_xticks(x, [str(acc) for acc in ACCEPTANCE_TAGS])
        ax.set_title(dataset)
        ax.set_xlabel("Acceptance (%)")
        ax.set_ylabel("Accepted-set mean RMSE gain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "matched_acceptance_bars.png", dpi=200)
    plt.close(fig)


def plot_hard_subsets(root: Path, per_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row")
    subset = per_df[(per_df["tag"] == "main") & (per_df["accepted"]) & (per_df["method"].isin(["uncertainty_gate@40", "conformal_gate@40"]))]
    for col, dataset in enumerate(DATASETS):
        ds = subset[subset["dataset"] == dataset]
        group_col = "hardness" if dataset == "Norman" else "novelty_quartile"
        if dataset != "Norman" and ds[group_col].nunique(dropna=True) <= 1:
            axes[0, col].text(0.5, 0.5, "Held-out perturbations occupy one novelty bin.\nQuartile stratification is not informative.", ha="center", va="center")
            axes[0, col].set_axis_off()
            axes[1, col].text(0.5, 0.5, "Biological hard-subset annotations omitted\nfor collapsed novelty stratification.", ha="center", va="center")
            axes[1, col].set_axis_off()
            continue
        gain = ds.groupby([group_col, "method"])["gain_rmse"].mean().unstack()
        gain.plot(kind="bar", ax=axes[0, col])
        axes[0, col].set_title(dataset)
        axes[0, col].set_ylabel("Accepted-set RMSE gain")
        bio = (
            ds.groupby([group_col, "method"])[["top20_overlap", "sign_accuracy", "pathway_corr"]]
            .mean()
            .reset_index()
        )
        bio["bio_score"] = bio[["top20_overlap", "sign_accuracy", "pathway_corr"]].mean(axis=1)
        bio_plot = bio.pivot(index=group_col, columns="method", values="bio_score")
        bio_plot.plot(kind="bar", ax=axes[1, col])
        axes[1, col].set_ylabel("Accepted-set biological score")
    fig.tight_layout()
    fig.savefig(root / "figures" / "hard_subset_results.png", dpi=200)
    plt.close(fig)


def plot_router_diagnostics(root: Path, per_df: pd.DataFrame, scores: dict[tuple[str, int, str], dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    for col, dataset in enumerate(DATASETS):
        reg_rows = []
        conf_rows = []
        novelty_rows = []
        overlap_rows = []
        for seed in sorted(per_df[per_df["dataset"] == dataset]["seed"].unique()):
            payload = scores[(dataset, seed, "main")]
            gain = payload["router_test_gain"]
            reg_rows.append(pd.DataFrame({"score": payload["router_gain_regressor_score"], "gain": gain}))
            conf_rows.append(pd.DataFrame({"score": payload["router_conformal_score"], "gain": gain}))
            seed_df = per_df[(per_df["dataset"] == dataset) & (per_df["seed"] == seed) & (per_df["tag"] == "main")]
            conf_sel = seed_df[(seed_df["method"] == "conformal_gate@40")][["perturbation", "accepted", "novelty_quartile"]]
            unc_sel = seed_df[(seed_df["method"] == "uncertainty_gate@40")][["perturbation", "accepted", "novelty_quartile"]]
            novelty_rows.append(
                conf_sel.groupby("novelty_quartile")["accepted"].mean().reset_index().assign(method="conformal")
            )
            novelty_rows.append(
                unc_sel.groupby("novelty_quartile")["accepted"].mean().reset_index().assign(method="uncertainty")
            )
            conf_set = set(conf_sel.loc[conf_sel["accepted"], "perturbation"])
            unc_set = set(unc_sel.loc[unc_sel["accepted"], "perturbation"])
            union = conf_set | unc_set
            overlap_rows.append(
                {
                    "seed": seed,
                    "jaccard": float(len(conf_set & unc_set) / len(union)) if union else 1.0,
                }
            )
        reg_df = pd.concat(reg_rows, ignore_index=True)
        conf_df = pd.concat(conf_rows, ignore_index=True)
        axes[0, col].scatter(reg_df["score"], reg_df["gain"], alpha=0.6, s=12)
        axes[0, col].set_title(dataset)
        axes[0, col].set_xlabel("Non-conformal gain score")
        axes[0, col].set_ylabel("Realized gain")
        axes[1, col].scatter(conf_df["score"], conf_df["gain"], alpha=0.6, s=12)
        axes[1, col].set_xlabel("Conformal lower-bound score")
        axes[1, col].set_ylabel("Realized gain")
        nov = pd.concat(novelty_rows, ignore_index=True)
        nov["accepted"] = nov["accepted"].astype(float)
        nov_pivot = nov.groupby(["novelty_quartile", "method"])["accepted"].mean().unstack()
        if not nov_pivot.empty and len(nov_pivot.index) > 1:
            nov_pivot.plot(kind="bar", ax=axes[2, col])
            axes[2, col].set_ylabel("Acceptance fraction")
        else:
            axes[2, col].text(0.5, 0.5, "Novelty stratification collapsed", ha="center", va="center")
            axes[2, col].set_axis_off()
        overlap_df = pd.DataFrame(overlap_rows)
        axes[3, col].bar(["conformal vs uncertainty"], [overlap_df["jaccard"].mean()])
        axes[3, col].set_ylim(0, 1)
        axes[3, col].set_ylabel("Accepted-set Jaccard")
    fig.tight_layout()
    fig.savefig(root / "figures" / "router_diagnostics.png", dpi=200)
    plt.close(fig)


def paired_primary_comparisons(metrics_df: pd.DataFrame) -> list[dict[str, float | str]]:
    rows = []
    main40 = metrics_df[(metrics_df["tag"] == "main") & (metrics_df["method"].isin(["conformal_gate@40", "uncertainty_gate@40", "gain_regressor@40"]))]
    pivot = main40.pivot_table(index=["dataset", "seed"], columns="method", values="mean_gain")
    for dataset, grp in pivot.groupby(level=0):
        grp = grp.reset_index(drop=True)
        if {"conformal_gate@40", "uncertainty_gate@40", "gain_regressor@40"} - set(grp.columns):
            continue
        for baseline in ["uncertainty_gate@40", "gain_regressor@40"]:
            diff = grp["conformal_gate@40"] - grp[baseline]
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": f"conformal_minus_{baseline.replace('@40', '')}",
                    "seed_count": int(diff.shape[0]),
                    "delta_mean": float(diff.mean()),
                    "delta_std": float(diff.std(ddof=0)),
                    "delta_ci95": float(1.96 * diff.std(ddof=0) / np.sqrt(max(1, diff.shape[0]))),
                }
            )
    return rows


def plot_sensitivity(root: Path, summary_df: pd.DataFrame) -> None:
    sens = summary_df[summary_df["tag"].str.startswith("sensitivity_")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    split_df = sens[sens["tag"].str.startswith("sensitivity_split_") & (sens["method"] == "conformal_gate@40")]
    for dataset in DATASETS:
        ds = split_df[split_df["dataset"] == dataset]
        axes[0].plot(ds["tag"], ds["mean_gain_mean"], marker="o", label=dataset)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Split Sensitivity")
    axes[0].set_ylabel("Accepted-set mean gain")
    alpha_df = sens[sens["tag"].str.startswith("sensitivity_alpha_") & (sens["method"] == "conformal_gate@40")]
    for dataset in DATASETS:
        ds = alpha_df[alpha_df["dataset"] == dataset]
        axes[1].plot(ds["tag"], ds["mean_gain_mean"], marker="o", label=dataset)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Alpha Sensitivity")
    gain_df = sens[sens["tag"].str.startswith("sensitivity_gain_") & (sens["method"] == "conformal_gate@40")]
    for dataset in DATASETS:
        ds = gain_df[gain_df["dataset"] == dataset]
        axes[2].plot(ds["tag"], ds["mean_gain_mean"], marker="o", label=dataset)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title("Gain-Target Sensitivity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "sensitivity.png", dpi=200)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    (root / "figures").mkdir(exist_ok=True)
    metrics_df, per_df, runtime_df, scores = collect_runs(root)
    metrics_df.to_csv(root / "figures" / "all_metrics.csv", index=False)
    summary_df = summarize_metrics(metrics_df, runtime_df)
    write_main_table(root, summary_df)
    write_calibration_counts(root)
    plot_selective_risk(root, metrics_df)
    plot_matched_acceptance(root, summary_df)
    plot_hard_subsets(root, per_df)
    plot_router_diagnostics(root, per_df, scores)
    plot_sensitivity(root, summary_df)
    main40 = summary_df[(summary_df["tag"] == "main") & (summary_df["method"].isin(["uncertainty_gate@40", "gain_regressor@40", "conformal_gate@40"]))]
    dataset_comparisons = []
    success_count = 0
    decisive_success_count = 0
    calibration_flags = {
        (json.loads(path.read_text())["dataset"], json.loads(path.read_text())["seed"]): json.loads(path.read_text())["calibration_underpowered"]
        for path in root.glob("splits/*/seed_*.json")
        if json.loads(path.read_text()).get("run_version") == RUN_VERSION
    }
    for dataset in DATASETS:
        ds = main40[main40["dataset"] == dataset].set_index("method")
        if ds.empty:
            continue
        conformal = float(ds.loc["conformal_gate@40", "mean_gain_mean"])
        uncertainty = float(ds.loc["uncertainty_gate@40", "mean_gain_mean"])
        gain_reg = float(ds.loc["gain_regressor@40", "mean_gain_mean"])
        better_than_both = conformal > uncertainty and conformal > gain_reg
        success_count += int(better_than_both)
        underpowered = all(calibration_flags.get((dataset, seed), False) for seed in [42, 43, 44] if (dataset, seed) in calibration_flags)
        decisive_success_count += int(better_than_both and not underpowered)
        dataset_comparisons.append(
            {
                "dataset": dataset,
                "conformal_gate_40_mean_gain": conformal,
                "uncertainty_gate_40_mean_gain": uncertainty,
                "gain_regressor_40_mean_gain": gain_reg,
                "conformal_beats_both_at_40": better_than_both,
                "all_seeds_underpowered_calibration": underpowered,
            }
        )
    negative_result = decisive_success_count < 2
    pairwise = paired_primary_comparisons(metrics_df)
    results = {
        "run_version": RUN_VERSION,
        "summary": {
            "framing": "negative_benchmark_result" if negative_result else "method_positive_signal",
            "primary_success_criterion_met": not negative_result,
            "datasets_where_conformal_beats_both_at_40": success_count,
            "datasets_where_conformal_beats_both_at_40_without_underpowered_calibration": decisive_success_count,
            "dataset_comparisons_at_40": dataset_comparisons,
            "conformal_claim_note": (
                "Adamson and Replogle remain underpowered for decisive conformal claims; the benchmark should still be framed as negative overall."
                if negative_result
                else "Rerun changed the main comparison enough to warrant rechecking the claim scope."
            ),
            "feature_scope_note": "Curated GO/pathway descriptors were not restored. Executed runs use hashed target descriptors, simple co-target graph features, and training-derived gene modules for pathway-style analysis.",
            "novelty_stratification_note": "Adamson and Replogle held-out perturbations collapse into one novelty bin under the current training-only descriptor distance definition, so quartile hard-subset analysis is not informative there.",
            "primary_pairwise_comparisons": pairwise,
        },
        "main": summary_df[summary_df["tag"] == "main"].to_dict(orient="records"),
        "ablations": summary_df[summary_df["tag"].str.startswith("ablation_")].to_dict(orient="records"),
        "sensitivity": summary_df[summary_df["tag"].str.startswith("sensitivity_")].to_dict(orient="records"),
    }
    write_json(root / "results.json", results)


if __name__ == "__main__":
    main()
