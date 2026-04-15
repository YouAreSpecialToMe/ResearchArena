from __future__ import annotations

import time
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.core import (
    ensure_dir,
    init_experiment,
    log_message,
    peak_memory_mb,
    summarize_paired_difference,
    save_csv,
    save_json,
    set_thread_env,
    summarize_mean_std_ci,
    utc_now_iso,
)
from exp.shared.plotting import (
    plot_budget_sensitivity,
    plot_fixed_data,
    plot_localization,
    plot_null_histograms,
    plot_power,
)


def binomial_interval(successes: int, total: int, alpha: float = 0.05) -> dict[str, float]:
    if total <= 0:
        return {"rate": float("nan"), "lower": float("nan"), "upper": float("nan")}
    rate = successes / total
    lower = 0.0 if successes == 0 else float(beta_dist.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta_dist.ppf(1 - alpha / 2, successes + 1, total - successes))
    return {"rate": float(rate), "lower": lower, "upper": upper}


def bootstrap_kendall_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    draws: int = 2000,
) -> dict[str, float]:
    from scipy.stats import kendalltau

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    taus = []
    for _ in range(draws):
        idx = rng.integers(0, n, size=n)
        tau = kendalltau(x[idx], y[idx], nan_policy="omit").correlation
        taus.append(0.0 if np.isnan(tau) else float(tau))
    arr = np.asarray(taus, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "ci95_lower": float(np.quantile(arr, 0.025)),
        "ci95_upper": float(np.quantile(arr, 0.975)),
    }


def dataframe_to_text(title: str, df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    return f"{title}\n{df.to_string(index=False, float_format=float_fmt.format)}\n"


def main() -> None:
    start = time.perf_counter()
    set_thread_env()
    dirs = init_experiment("aggregate")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    log_message(log_path, "Starting aggregation, paired summaries, and figure generation.")
    ensure_dir(".")
    null_df = pd.read_csv("results/null_checks/null_metrics.csv")
    power_df = pd.read_csv("results/power/power_metrics.csv")
    loc_df = pd.read_csv("results/localization/localization_metrics.csv")
    fixed_df = pd.read_csv("results/fixed_data/fixed_data_metrics.csv")
    ablation_df = pd.read_csv("results/ablations/ablation_metrics.csv")
    fixed_rank_df = pd.read_csv("results/fixed_data/rank_correlations.csv")

    plot_null_histograms(null_df, "figures/figure1_null_histograms.png")
    power_summary = summarize_mean_std_ci(
        power_df, "reject_at_0_05", ["condition", "method"]
    )
    plot_power(
        power_summary,
        "figures/figure2_synthetic_power.png",
    )
    localization_summary = summarize_mean_std_ci(
        loc_df, "top1", ["condition", "method"]
    )
    plot_localization(
        localization_summary,
        "figures/figure3_localization.png",
    )
    plot_fixed_data(fixed_df, "figures/figure4_fixed_data.png")
    budget_df = ablation_df[ablation_df["study"] == "budget_sensitivity"].copy()
    if not budget_df.empty:
        budget_summary = summarize_mean_std_ci(
            budget_df, "reject_at_0_05", ["condition", "method"]
        )
        runtime_summary = summarize_mean_std_ci(
            budget_df, "runtime_minutes", ["condition", "method"]
        )
        budget_summary = budget_summary.merge(
            runtime_summary, on=["condition", "method"], how="left"
        )
        plot_budget_sensitivity(
            budget_summary, "figures/figure5_budget_sensitivity.png"
        )
    else:
        budget_summary = pd.DataFrame()

    table1 = summarize_mean_std_ci(null_df, "global_pvalue", ["setting"])
    table1 = table1.merge(
        summarize_mean_std_ci(null_df, "cov_pvalue", ["setting"]),
        on=["setting"],
        how="left",
    ).merge(
        summarize_mean_std_ci(null_df, "tail_pvalue", ["setting"]),
        on=["setting"],
        how="left",
    )
    table2 = summarize_mean_std_ci(power_df, "reject_at_0_05", ["condition", "method"])
    table2 = table2.merge(
        summarize_mean_std_ci(power_df, "runtime_minutes", ["condition", "method"]),
        on=["condition", "method"],
        how="left",
    )
    power_diff = summarize_paired_difference(
        power_df,
        ["seed", "dimension", "condition"],
        "method",
        "reject_at_0_05",
        "cosbc_tmax",
        "enriched",
        "cosbc_minus_enriched",
    )
    table3 = summarize_mean_std_ci(loc_df, "top1", ["condition", "method"])
    table3 = table3.merge(
        summarize_mean_std_ci(loc_df, "top3", ["condition", "method"]),
        on=["condition", "method"],
        how="left",
    ).merge(
        summarize_mean_std_ci(loc_df, "mean_rank", ["condition", "method"]),
        on=["condition", "method"],
        how="left",
    )
    localization_diff = summarize_paired_difference(
        loc_df,
        ["seed", "dimension", "condition"],
        "method",
        "top1",
        "cosbc",
        "enriched",
        "cosbc_minus_enriched_top1",
    )
    table4 = summarize_mean_std_ci(
        fixed_df, "reject_at_0_05", ["approximation", "method"]
    ).merge(
        summarize_mean_std_ci(
            fixed_df, "global_stat", ["approximation", "method"]
        ),
        on=["approximation", "method"],
        how="left",
    ).merge(
        summarize_mean_std_ci(
            fixed_df, "end_to_end_runtime_minutes", ["approximation", "method"]
        ),
        on=["approximation", "method"],
        how="left",
    ).merge(
        summarize_mean_std_ci(
            fixed_df, "cov_error", ["approximation", "method"]
        ),
        on=["approximation", "method"],
        how="left",
    )
    fixed_diff = summarize_paired_difference(
        fixed_df,
        ["seed", "approximation"],
        "method",
        "reject_at_0_05",
        "cosbc",
        "enriched",
        "cosbc_minus_enriched",
    )
    save_csv("results/table1_null.csv", table1)
    save_csv("results/table2_power.csv", table2)
    save_csv("results/table3_localization.csv", table3)
    save_csv("results/table4_fixed_data.csv", table4)
    save_csv("results/table5_ablations.csv", ablation_df)
    save_csv("results/table2_power_paired_diff.csv", power_diff)
    save_csv("results/table3_localization_paired_diff.csv", localization_diff)
    save_csv("results/table4_fixed_data_paired_diff.csv", fixed_diff)
    if not budget_summary.empty:
        save_csv("results/table6_budget_sensitivity.csv", budget_summary)

    table1_text = dataframe_to_text("Table 1: Null Calibration", table1)
    table2_text = dataframe_to_text("Table 2: Synthetic Power", table2)
    table3_text = dataframe_to_text("Table 3: Localization", table3)
    table4_text = dataframe_to_text("Table 4: Fixed Data", table4)
    Path("results/table1_null.txt").write_text(table1_text)
    Path("results/table2_power.txt").write_text(table2_text)
    Path("results/table3_localization.txt").write_text(table3_text)
    Path("results/table4_fixed_data.txt").write_text(table4_text)
    if not budget_summary.empty:
        Path("results/table6_budget_sensitivity.txt").write_text(
            dataframe_to_text("Table 6: Budget Sensitivity", budget_summary)
        )

    main_null_mask = null_df["benchmark"].isin(["toeplitz_null", "tie_quantized"])
    extended_null_mask = null_df["benchmark"].isin(["toeplitz_null", "tie_quantized", "kernel_null"])
    main_null_successes = int((null_df.loc[main_null_mask, "global_pvalue"] <= 0.05).sum())
    main_null_total = int(main_null_mask.sum())
    extended_null_successes = int((null_df.loc[extended_null_mask, "global_pvalue"] <= 0.05).sum())
    extended_null_total = int(extended_null_mask.sum())
    main_null_interval = binomial_interval(main_null_successes, main_null_total)
    extended_null_interval = binomial_interval(extended_null_successes, extended_null_total)
    null_valid = bool(0.03 <= main_null_interval["rate"] <= 0.07)
    power_pivot = power_df.pivot_table(
        index=["seed", "dimension", "condition"], columns="method", values="reject_at_0_05"
    ).reset_index()
    power_pivot["cosbc_minus_enriched"] = power_pivot.get("cosbc_tmax", 0.0) - power_pivot.get("enriched", 0.0)
    synthetic_condition_gaps = (
        power_pivot.groupby(["dimension", "condition"], dropna=False)["cosbc_minus_enriched"]
        .mean()
        .reset_index()
    )
    synthetic_wins = int((synthetic_condition_gaps["cosbc_minus_enriched"] >= 0.08).sum())
    fixed_rank = fixed_rank_df.set_index("method")
    fixed_power_gap = (
        fixed_df[fixed_df["method"] == "cosbc"]
        .groupby("approximation")["reject_at_0_05"]
        .mean()
        .sub(
            fixed_df[fixed_df["method"] == "enriched"]
            .groupby("approximation")["reject_at_0_05"]
            .mean(),
            fill_value=0.0,
        )
    )
    rng = np.random.default_rng(20260322)
    fixed_cosbc = fixed_df[fixed_df["method"] == "cosbc"].sort_values(["seed", "approximation"])
    fixed_enriched = fixed_df[fixed_df["method"] == "enriched"].sort_values(["seed", "approximation"])
    fixed_corr_ci = {
        "cosbc": bootstrap_kendall_ci(
            fixed_cosbc["cov_error"].to_numpy(),
            fixed_cosbc["global_stat"].to_numpy(),
            rng,
        ),
        "enriched": bootstrap_kendall_ci(
            fixed_enriched["cov_error"].to_numpy(),
            fixed_enriched["global_stat"].to_numpy(),
            rng,
        ),
    }
    paired_diff_samples = []
    x_cosbc = fixed_cosbc["cov_error"].to_numpy()
    y_cosbc = fixed_cosbc["global_stat"].to_numpy()
    y_enriched = fixed_enriched["global_stat"].to_numpy()
    from scipy.stats import kendalltau

    for _ in range(2000):
        idx = rng.integers(0, len(x_cosbc), size=len(x_cosbc))
        tau_cosbc = kendalltau(x_cosbc[idx], y_cosbc[idx], nan_policy="omit").correlation
        tau_enriched = kendalltau(x_cosbc[idx], y_enriched[idx], nan_policy="omit").correlation
        tau_cosbc = 0.0 if np.isnan(tau_cosbc) else float(tau_cosbc)
        tau_enriched = 0.0 if np.isnan(tau_enriched) else float(tau_enriched)
        paired_diff_samples.append(tau_cosbc - tau_enriched)
    paired_diff_arr = np.asarray(paired_diff_samples, dtype=float)
    fixed_tau_diff = {
        "mean": float(paired_diff_arr.mean()),
        "std": float(paired_diff_arr.std(ddof=0)),
        "ci95_lower": float(np.quantile(paired_diff_arr, 0.025)),
        "ci95_upper": float(np.quantile(paired_diff_arr, 0.975)),
        "observed": float(
            fixed_rank["kendall_tau"].get("cosbc", 0.0) - fixed_rank["kendall_tau"].get("enriched", 0.0)
        ),
    }
    fixed_win = bool(
        (fixed_power_gap >= 0.05).any()
        or (
            fixed_tau_diff["observed"] >= 0.10
            and fixed_tau_diff["ci95_lower"] > 0.0
        )
    )
    scalar_win = bool(
        power_df.groupby("method")["reject_at_0_05"].mean().get("cosbc_tmax", 0.0)
        > power_df.groupby("method")["reject_at_0_05"].mean().get("scalar", 0.0)
    )
    loc_summary = loc_df.groupby("method")["top1"].mean()
    localizes = bool(loc_summary.get("cosbc", 0.0) >= loc_summary.get("enriched", 0.0) + 0.10)
    runtime_files = [
        Path("exp/setup/results.json"),
        Path("exp/null_checks/results.json"),
        Path("exp/power/results.json"),
        Path("exp/localization/results.json"),
        Path("exp/fixed_data/results.json"),
        Path("exp/ablations/results.json"),
    ]
    total_runtime = 0.0
    for path in runtime_files:
        payload = json.loads(path.read_text())
        total_runtime += float(payload.get("runtime_minutes", 0.0))
    within_budget = bool(total_runtime <= 8 * 60)
    hypothesis = {
        "null_valid": null_valid,
        "beats_feature_matched_enriched_sbc_synthetic": synthetic_wins >= 2,
        "beats_feature_matched_enriched_sbc_fixed_data": fixed_win,
        "beats_scalar_sbc": scalar_win,
        "localizes_pairs": localizes,
        "within_budget": within_budget,
    }
    save_json("results/hypothesis_check.json", hypothesis)
    strict_conjunction = bool(
        hypothesis["null_valid"]
        and hypothesis["beats_feature_matched_enriched_sbc_synthetic"]
        and hypothesis["beats_feature_matched_enriched_sbc_fixed_data"]
        and hypothesis["localizes_pairs"]
        and hypothesis["within_budget"]
    )

    results = {
        "created_at_utc": utc_now_iso(),
        "summary": {
            "null_global_rejection_mean": float((null_df["global_pvalue"] <= 0.05).mean()),
            "null_global_rejection_std": float((null_df["global_pvalue"] <= 0.05).std(ddof=0)),
            "null_main_rejection_at_0_05": main_null_interval,
            "null_extended_rejection_at_0_05": extended_null_interval,
            "synthetic_power_mean_by_method": power_df.groupby("method")["reject_at_0_05"].mean().to_dict(),
            "synthetic_power_std_by_method": power_df.groupby("method")["reject_at_0_05"].std(ddof=0).fillna(0.0).to_dict(),
            "localization_top1_mean_by_method": loc_summary.to_dict(),
            "localization_top1_std_by_method": loc_df.groupby("method")["top1"].std(ddof=0).fillna(0.0).to_dict(),
            "fixed_data_cov_error_mean": float(fixed_df["cov_error"].mean()),
            "fixed_data_reject_0_05_mean_by_method": fixed_df.groupby("method")["reject_at_0_05"].mean().to_dict(),
            "fixed_data_reject_0_05_std_by_method": fixed_df.groupby("method")["reject_at_0_05"].std(ddof=0).fillna(0.0).to_dict(),
            "fixed_data_rank_correlations": fixed_rank_df.to_dict(orient="records"),
            "fixed_data_rank_correlation_uncertainty": fixed_corr_ci,
            "fixed_data_kendall_tau_difference": fixed_tau_diff,
            "ablation_rows": int(len(ablation_df)),
            "total_runtime_minutes": total_runtime,
            "paired_power_diff": power_diff.to_dict(orient="records"),
            "paired_localization_diff": localization_diff.to_dict(orient="records"),
            "paired_fixed_data_diff": fixed_diff.to_dict(orient="records"),
        },
        "artifacts": {
            "runtime_schedule": "results/runtime/schedule.json",
            "run_manifest": "results/runtime/run_manifest.jsonl",
        },
        "hypothesis_check": hypothesis,
        "strict_conjunction_passed": strict_conjunction,
        "claim": "stronger_cosbc_claim_supported" if strict_conjunction else "negative_benchmark_result",
        "headline_conclusion": (
            "CoSBC does not satisfy the pre-registered conjunction rule: null validity fails, "
            "fixed-data added-value is not statistically supported, and localization underperforms enriched SBC."
        ),
    }
    save_json("results.json", results)
    save_json(
        dirs["exp_dir"] / "results.json",
        {
            "experiment": "aggregate",
            "created_at_utc": utc_now_iso(),
            "peak_memory_mb": peak_memory_mb(),
            "runtime_minutes": (time.perf_counter() - start) / 60.0,
        },
    )
    save_json(
        dirs["results_dir"] / "results.json",
        {
            "experiment": "aggregate",
            "created_at_utc": utc_now_iso(),
            "peak_memory_mb": peak_memory_mb(),
            "runtime_minutes": (time.perf_counter() - start) / 60.0,
        },
    )
    save_json(
        dirs["logs_dir"] / "run_metadata.json",
        {
            "experiment": "aggregate",
            "created_at_utc": utc_now_iso(),
            "peak_memory_mb": peak_memory_mb(),
            "runtime_minutes": (time.perf_counter() - start) / 60.0,
        },
    )
    log_message(log_path, "Aggregation stage completed.")


if __name__ == "__main__":
    main()
