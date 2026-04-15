from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared import config
from exp.shared.pipeline import RunResult, bootstrap_difference
from exp.shared.utils import capture_config_snapshot, capture_environment_metadata, save_json
from exp.shared.visualization import plot_descriptor_contingency, plot_difference_intervals, save_main_table


def load_run_result(stage: str, dataset: str, seed: int, model: str, metrics_row: pd.Series) -> RunResult:
    path = ROOT / "exp" / stage / "predictions" / f"{dataset}_seed{seed}_{model.replace(' ', '_').replace('/', '-')}.npz"
    blob = np.load(path, allow_pickle=True)
    return RunResult(
        model_name=model,
        dataset=dataset,
        seed=seed,
        metrics={k: float(metrics_row[k]) for k in [
            "perturbed_reference_pearson",
            "rmse",
            "top1_accuracy",
            "median_rank",
            "pearson_top100_hvg",
        ]},
        predictions=blob["predictions"],
        true=blob["true"],
        labels=blob["labels"].tolist(),
        hyperparams={},
        runtime_minutes=float(metrics_row["runtime_minutes"]),
        peak_memory_mb=float(metrics_row["peak_memory_mb"]),
        peak_gpu_memory_mb=(
            None
            if "peak_gpu_memory_mb" not in metrics_row.index or pd.isna(metrics_row["peak_gpu_memory_mb"])
            else float(metrics_row["peak_gpu_memory_mb"])
        ),
    )


def select_model_with_rmse_guardrail(
    ds_summary: pd.DataFrame,
    candidate_models: list[str],
    rmse_reference: float,
) -> tuple[str, bool]:
    candidates = ds_summary.loc[candidate_models].copy()
    valid = candidates[candidates["rmse_mean"] <= rmse_reference + 1e-6]
    if valid.empty:
        chosen = candidates.sort_values(
            ["perturbed_reference_pearson_mean", "rmse_mean"],
            ascending=[False, True],
        ).index[0]
        return chosen, False
    chosen = valid.sort_values(
        ["perturbed_reference_pearson_mean", "rmse_mean"],
        ascending=[False, True],
    ).index[0]
    return chosen, True


def main() -> None:
    environment = capture_environment_metadata()
    config_snapshot = capture_config_snapshot(config)
    baseline = pd.read_csv(ROOT / "exp" / "baseline_ladder" / "metrics.csv")
    retrieval = pd.read_csv(ROOT / "exp" / "retrieval_models" / "metrics.csv")
    ablations = pd.read_csv(ROOT / "exp" / "ablations" / "metrics.csv")

    all_main = pd.concat([baseline, retrieval], ignore_index=True)
    all_main["rank_pearson"] = all_main.groupby(["dataset", "seed"])["perturbed_reference_pearson"].rank(ascending=False, method="average")
    all_main["rank_rmse"] = all_main.groupby(["dataset", "seed"])["rmse"].rank(ascending=True, method="average")
    all_main["rank_top1"] = all_main.groupby(["dataset", "seed"])["top1_accuracy"].rank(ascending=False, method="average")
    all_main["rank_median"] = all_main.groupby(["dataset", "seed"])["median_rank"].rank(ascending=True, method="average")
    all_main["composite_rank"] = all_main[["rank_pearson", "rank_rmse", "rank_top1", "rank_median"]].mean(axis=1)
    summary = (
        all_main.groupby(["dataset", "model"], as_index=False)
        .agg(
            perturbed_reference_pearson_mean=("perturbed_reference_pearson", "mean"),
            perturbed_reference_pearson_std=("perturbed_reference_pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            top1_accuracy_mean=("top1_accuracy", "mean"),
            top1_accuracy_std=("top1_accuracy", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
            composite_rank_mean=("composite_rank", "mean"),
            composite_rank_std=("composite_rank", "std"),
            runtime_minutes_mean=("runtime_minutes", "mean"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
        )
    )
    save_main_table(summary, ROOT / "figures" / "main_results_table.csv")

    comp_rows = []
    selection_rows = []
    for dataset in config.DATASETS:
        ds = all_main[all_main["dataset"] == dataset]
        ds_summary = summary[summary["dataset"] == dataset].set_index("model")
        train_mean_rmse = float(ds_summary.loc["Train Perturbed Mean", "rmse_mean"])
        best_resid_raw = ds_summary.loc[
            ["Residualized Ridge", "Residualized PLS", "Residualized Linear Embedding"]
        ].sort_values(["perturbed_reference_pearson_mean", "rmse_mean"], ascending=[False, True]).index[0]
        best_resid, resid_guardrail_applied = select_model_with_rmse_guardrail(
            ds_summary,
            ["Residualized Ridge", "Residualized PLS", "Residualized Linear Embedding"],
            train_mean_rmse,
        )
        best_resrp_raw = ds_summary.loc[["ReSRP-Linear", "ReSRP-MLP"]].sort_values(
            ["perturbed_reference_pearson_mean", "rmse_mean"],
            ascending=[False, True],
        ).index[0]
        best_resrp, resrp_guardrail_applied = select_model_with_rmse_guardrail(
            ds_summary,
            ["ReSRP-Linear", "ReSRP-MLP"],
            float(ds_summary.loc[best_resid, "rmse_mean"]),
        )
        selection_rows.append(
            {
                "dataset": dataset,
                "train_perturbed_mean_rmse_reference": train_mean_rmse,
                "best_residualized_model_by_pearson": best_resid_raw,
                "guardrailed_best_residualized_model": best_resid,
                "residualized_guardrail_applied": resid_guardrail_applied,
                "best_resrp_model_by_pearson": best_resrp_raw,
                "guardrailed_best_resrp_model": best_resrp,
                "resrp_guardrail_applied": resrp_guardrail_applied,
            }
        )

        rr_runs = []
        nr_runs = []
        best_resid_runs = []
        best_resrp_runs = []
        for seed in config.SEEDS:
            rr_row = baseline[(baseline["dataset"] == dataset) & (baseline["seed"] == seed) & (baseline["model"] == "Residualized Ridge")].iloc[0]
            nr_row = baseline[(baseline["dataset"] == dataset) & (baseline["seed"] == seed) & (baseline["model"] == "Non-residualized Ridge")].iloc[0]
            br_row = all_main[(all_main["dataset"] == dataset) & (all_main["seed"] == seed) & (all_main["model"] == best_resid)].iloc[0]
            bm_row = all_main[(all_main["dataset"] == dataset) & (all_main["seed"] == seed) & (all_main["model"] == best_resrp)].iloc[0]
            rr_runs.append(load_run_result("baseline_ladder", dataset, seed, "Residualized Ridge", rr_row))
            nr_runs.append(load_run_result("baseline_ladder", dataset, seed, "Non-residualized Ridge", nr_row))
            best_resid_stage = "baseline_ladder"
            best_resrp_stage = "retrieval_models"
            best_resid_runs.append(load_run_result(best_resid_stage, dataset, seed, best_resid, br_row))
            best_resrp_runs.append(load_run_result(best_resrp_stage, dataset, seed, best_resrp, bm_row))

        pearson_ci = bootstrap_difference(rr_runs, nr_runs, "perturbed_reference_pearson")
        rmse_ci = bootstrap_difference(rr_runs, nr_runs, "rmse")
        comp_rows.append(
            {
                "dataset": dataset,
                "comparison_type": "residualization",
                "model_a": "Residualized Ridge",
                "model_b": "Non-residualized Ridge",
                "pearson_mean_difference": pearson_ci["mean_difference"],
                "pearson_ci_low": pearson_ci["ci_low"],
                "pearson_ci_high": pearson_ci["ci_high"],
                "rmse_mean_difference": rmse_ci["mean_difference"],
                "rmse_ci_low": rmse_ci["ci_low"],
                "rmse_ci_high": rmse_ci["ci_high"],
            }
        )
        pearson_ci = bootstrap_difference(best_resrp_runs, best_resid_runs, "perturbed_reference_pearson")
        rmse_ci = bootstrap_difference(best_resrp_runs, best_resid_runs, "rmse")
        comp_rows.append(
            {
                "dataset": dataset,
                "comparison_type": "retrieval",
                "model_a": best_resrp,
                "model_b": best_resid,
                "pearson_mean_difference": pearson_ci["mean_difference"],
                "pearson_ci_low": pearson_ci["ci_low"],
                "pearson_ci_high": pearson_ci["ci_high"],
                "rmse_mean_difference": rmse_ci["mean_difference"],
                "rmse_ci_low": rmse_ci["ci_low"],
                "rmse_ci_high": rmse_ci["ci_high"],
            }
        )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(ROOT / "figures" / "bootstrap_comparisons.csv", index=False)
    plot_difference_intervals(comp_df, ROOT / "figures" / "difference_intervals")

    contingency = ablations[ablations["descriptor_variant"].notna()].copy()
    if not contingency.empty:
        plot_descriptor_contingency(
            contingency.groupby(["dataset", "descriptor_variant"], as_index=False)["perturbed_reference_pearson"].mean(),
            ROOT / "figures" / "descriptor_contingency",
        )

    ablation_rows = ablations[ablations["ablation"].notna()].copy()
    ablation_summary = (
        ablation_rows.groupby(["dataset", "ablation"], as_index=False)
        .agg(
            perturbed_reference_pearson_mean=("perturbed_reference_pearson", "mean"),
            perturbed_reference_pearson_std=("perturbed_reference_pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            top1_accuracy_mean=("top1_accuracy", "mean"),
            top1_accuracy_std=("top1_accuracy", "std"),
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
        )
    )
    ablation_summary.to_csv(ROOT / "figures" / "ablation_table.csv", index=False)
    contingency_summary = (
        contingency.groupby(["dataset", "descriptor_variant"], as_index=False)
        .agg(
            perturbed_reference_pearson_mean=("perturbed_reference_pearson", "mean"),
            perturbed_reference_pearson_std=("perturbed_reference_pearson", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
    )

    claim_rows = []
    for dataset in config.DATASETS:
        ds = summary[summary["dataset"] == dataset].set_index("model")
        rr = ds.loc["Residualized Ridge"]
        nr = ds.loc["Non-residualized Ridge"]
        mean_baseline = ds.loc["Train Perturbed Mean"]
        best_resid_name, resid_guardrail_applied = select_model_with_rmse_guardrail(
            ds,
            ["Residualized Ridge", "Residualized PLS", "Residualized Linear Embedding"],
            float(mean_baseline["rmse_mean"]),
        )
        best_resid = ds.loc[best_resid_name]
        best_resrp_name, resrp_guardrail_applied = select_model_with_rmse_guardrail(
            ds,
            ["ReSRP-Linear", "ReSRP-MLP"],
            float(best_resid["rmse_mean"]),
        )
        best_resrp = ds.loc[best_resrp_name]
        knn = ds.loc["Retrieval-only Residual kNN"]
        claim_rows.append(
            {
                "dataset": dataset,
                "residualized_ridge_beats_nonresidualized_ridge": bool(
                    rr["perturbed_reference_pearson_mean"] > nr["perturbed_reference_pearson_mean"]
                ),
                "residualized_ridge_rmse_guardrail": bool(rr["rmse_mean"] <= nr["rmse_mean"] + 1e-6),
                "some_residualized_baseline_beats_train_mean": bool(
                    best_resid["perturbed_reference_pearson_mean"] > mean_baseline["perturbed_reference_pearson_mean"]
                ),
                "best_resrp_beats_best_residualized": bool(
                    best_resrp["perturbed_reference_pearson_mean"] > best_resid["perturbed_reference_pearson_mean"]
                ),
                "best_resrp_rmse_guardrail": bool(best_resrp["rmse_mean"] <= best_resid["rmse_mean"] + 1e-6),
                "best_resrp_beats_knn": bool(
                    best_resrp["perturbed_reference_pearson_mean"] > knn["perturbed_reference_pearson_mean"]
                ),
                "best_residualized_model": best_resid_name,
                "best_resrp_model": best_resrp_name,
                "best_residualized_model_selected_with_rmse_guardrail": resid_guardrail_applied,
                "best_resrp_model_selected_with_rmse_guardrail": resrp_guardrail_applied,
            }
        )

    residualization_supported = all(
        row["residualized_ridge_beats_nonresidualized_ridge"]
        and row["residualized_ridge_rmse_guardrail"]
        and row["some_residualized_baseline_beats_train_mean"]
        for row in claim_rows
    )
    retrieval_supported = all(
        row["best_resrp_beats_best_residualized"]
        and row["best_resrp_rmse_guardrail"]
        and row["best_resrp_beats_knn"]
        for row in claim_rows
    )

    narrative = {
        "residualization": (
            "Residualized Ridge versus Non-residualized Ridge is now evaluated against a no-centering full-target "
            "TruncatedSVD comparator rather than the previously vacuous centered full-target PCA baseline. "
            "Interpret the residualization claim through the bootstrap intervals and the RMSE guardrail together."
        ),
        "retrieval": (
            "The retrieval add-on claim is interpreted against the guardrailed best residualized non-retrieval "
            "baseline, not the raw highest-Pearson residualized model if that model fails the RMSE guardrail."
        ),
    }

    root_results = {
        "environment": environment,
        "config": config_snapshot,
        "per_seed_main_rows": all_main.to_dict(orient="records"),
        "main_results": summary.to_dict(orient="records"),
        "bootstrap_comparisons": comp_df.to_dict(orient="records"),
        "model_selection": selection_rows,
        "ablations": {
            "per_seed_rows": ablation_rows.to_dict(orient="records"),
            "summary": ablation_summary.to_dict(orient="records"),
        },
        "descriptor_contingency": {
            "per_seed_rows": contingency.to_dict(orient="records"),
            "summary": contingency_summary.to_dict(orient="records"),
        },
        "narrative": narrative,
        "claims": {
            "per_dataset": claim_rows,
            "residualization_supported": residualization_supported,
            "retrieval_supported": retrieval_supported,
        },
    }
    save_json(ROOT / "results.json", root_results)
    save_json(
        ROOT / "exp" / "analysis" / "results.json",
        {"experiment": "analysis", **root_results},
    )


if __name__ == "__main__":
    main()
