from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from .utils import ROOT, append_log, exp_log_path, json_dump


def _save(fig, stem: str) -> None:
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures() -> None:
    start = time.perf_counter()
    log_path = exp_log_path("visualization")
    append_log(log_path, "Starting figure generation.")
    sns.set_theme(style="whitegrid")
    pred_path = ROOT / "artifacts" / "predictions" / "strict_test_predictions.parquet"
    made = []
    if not pred_path.exists():
        json_dump(
            {"experiment": "visualization", "status": "missing_predictions", "runtime_minutes": (time.perf_counter() - start) / 60.0},
            ROOT / "exp" / "visualization" / "results.json",
        )
        return

    df = pd.read_parquet(pred_path)
    strict = df[df["strict_label"].notna()].copy()
    if len(strict) == 0:
        json_dump(
            {"experiment": "visualization", "status": "empty_strict_slice", "runtime_minutes": (time.perf_counter() - start) / 60.0},
            ROOT / "exp" / "visualization" / "results.json",
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in ["support_only_mean", "support_compactness_mean", "full_context_removal_mean", "full_detector_mean"]:
        if model not in strict.columns:
            continue
        precision, recall, _ = precision_recall_curve(strict["strict_label"].to_numpy(dtype=int), strict[model].to_numpy(dtype=float))
        ax.plot(recall, precision, label=model.replace("_mean", ""))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Strict-Slice Precision Recall")
    ax.legend()
    _save(fig, "strict_pr_curve")
    made.append("strict_pr_curve")

    tau_table = ROOT / "artifacts" / "tables" / "tau_stability.csv"
    if tau_table.exists():
        stability = pd.read_csv(tau_table)
        if len(stability):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
            metrics = [
                ("test_auprc_full_detector", "Strict AUPRC"),
                ("smin_change_rate_vs_locked", "S_min Change Rate"),
                ("mean_jaccard_vs_locked", "Mean Jaccard"),
            ]
            for ax, (col, title) in zip(axes, metrics):
                sns.lineplot(data=stability, x="tau", y=col, marker="o", ax=ax, color="#5177a5")
                ax.set_title(title)
                ax.set_xlabel("tau")
            _save(fig, "threshold_stability")
            made.append("threshold_stability")

    main_table = ROOT / "artifacts" / "tables" / "main_results.csv"
    if main_table.exists():
        results = pd.read_csv(main_table)
        ablations = results[results["experiment"].str.startswith("ablation_")].copy()
        full_row = results[results["experiment"] == "full_detector"]
        if len(ablations) and len(full_row):
            ablations["delta_macro_f1"] = ablations["strict_macro_f1"] - float(full_row["strict_macro_f1"].iloc[0])
            ablations["delta_auprc"] = ablations["strict_auprc"] - float(full_row["strict_auprc"].iloc[0])
            plot_df = ablations.melt(
                id_vars=["experiment"],
                value_vars=["delta_macro_f1", "delta_auprc"],
                var_name="metric",
                value_name="delta",
            )
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(data=plot_df, x="experiment", y="delta", hue="metric", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Delta vs Full Detector")
            ax.tick_params(axis="x", rotation=25)
            _save(fig, "ablation_deltas")
            made.append("ablation_deltas")

    calibration_path = ROOT / "artifacts" / "tables" / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            for name, payload in calibration.items():
                curve = payload.get("curve", {})
                ax.plot(
                    curve.get("mean_predicted_value", []),
                    curve.get("fraction_of_positives", []),
                    marker="o",
                    label=name,
                )
            ax.set_xlabel("Mean predicted value")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Validation Calibration")
            ax.legend()
            _save(fig, "calibration_curve")
            made.append("calibration_curve")

    runtime_path = ROOT / "results.json"
    if runtime_path.exists():
        payload = json.loads(runtime_path.read_text())
        timings = payload.get("timings", {})
        if timings:
            plot_df = pd.DataFrame(
                [{"stage": key.replace("_minutes", ""), "minutes": value} for key, value in timings.items() if key.endswith("_minutes")]
            )
            if len(plot_df):
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=plot_df, x="stage", y="minutes", ax=ax, color="#8c5a3c")
                ax.tick_params(axis="x", rotation=30)
                ax.set_xlabel("")
                ax.set_ylabel("Minutes")
                ax.set_title("Runtime Breakdown")
                _save(fig, "runtime_breakdown")
                made.append("runtime_breakdown")

    payload = {
        "experiment": "visualization",
        "status": "completed",
        "figures": made,
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
    }
    json_dump(payload, ROOT / "exp" / "visualization" / "results.json")
    json_dump(payload, ROOT / "exp" / "visualization" / "config.json")
    append_log(log_path, f"Completed figure generation with {len(made)} figures.")
