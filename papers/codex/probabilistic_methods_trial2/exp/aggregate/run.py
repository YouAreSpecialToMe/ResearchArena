from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from exp.shared.config import RESULTS_DIR, THREAD_ENV
from exp.shared.io import read_json, safe_mean_std, write_json


def holm_correction(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = [0.0] * len(pvals)
    m = len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def _manifest_rows() -> list[dict]:
    rows = []
    for manifest_name in ["main_manifest.json", "ablation_manifest.json", "secondary_manifest.json"]:
        manifest_path = RESULTS_DIR / manifest_name
        if manifest_path.exists():
            rows.extend(read_json(manifest_path))
    return rows


def _coverage_ok(row: pd.Series) -> bool:
    return abs(float(row["marginal_coverage"]) - (1.0 - float(row["alpha"]))) <= 0.015


def main() -> None:
    os.environ.update(THREAD_ENV)
    rows = _manifest_rows()
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No manifest rows found. Run pilot and benchmarks first.")
    df["run_label"] = df["run_label"].fillna(df["method"])
    df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)

    main_labels = {"split_cp", "class_conditional_cp", "knn_rlcp", "gmm_rlcp", "batch_mcp", "chip_rlcp", "oracle_rlcp"}
    main_df = df[np.isclose(df["alpha"], 0.10) & df["run_label"].isin(main_labels)].copy()
    grouped_rows = []
    for (dataset, label), part in main_df.groupby(["dataset", "run_label"]):
        row = {"dataset": dataset, "method": label}
        for metric in [
            "marginal_coverage",
            "worst_external_group_coverage",
            "mean_external_group_coverage_gap",
            "mean_set_size",
            "total_runtime_sec",
        ]:
            stats = safe_mean_std(part[metric].tolist())
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        grouped_rows.append(row)
    summary_df = pd.DataFrame(grouped_rows).sort_values(["dataset", "method"])
    summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)

    paired_rows = []
    for dataset in sorted(main_df["dataset"].unique()):
        for seed in sorted(main_df["seed"].unique()):
            subset = main_df[(main_df["dataset"] == dataset) & (main_df["seed"] == seed)]
            required = {"chip_rlcp", "split_cp", "gmm_rlcp", "batch_mcp"}
            if required - set(subset["run_label"]):
                continue
            lookup = {row["run_label"]: row for _, row in subset.iterrows()}
            paired_rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "chip_minus_split_worst_group": float(
                        lookup["chip_rlcp"]["worst_external_group_coverage"] - lookup["split_cp"]["worst_external_group_coverage"]
                    ),
                    "chip_minus_gmm_worst_group": float(
                        lookup["chip_rlcp"]["worst_external_group_coverage"] - lookup["gmm_rlcp"]["worst_external_group_coverage"]
                    ),
                    "chip_minus_batch_mean_set_size": float(
                        lookup["chip_rlcp"]["mean_set_size"] - lookup["batch_mcp"]["mean_set_size"]
                    ),
                    "chip_coverage_ok": _coverage_ok(lookup["chip_rlcp"]),
                    "batch_coverage_ok": _coverage_ok(lookup["batch_mcp"]),
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(RESULTS_DIR / "paired_differences.csv", index=False)

    tests: list[tuple[str, float | None, int, str]] = []
    if not paired_df.empty:
        tests.append(
            (
                "chip_vs_split_worst_group",
                float(wilcoxon(paired_df["chip_minus_split_worst_group"]).pvalue),
                int(len(paired_df)),
                "ok",
            )
        )
        tests.append(
            (
                "chip_vs_gmm_worst_group",
                float(wilcoxon(paired_df["chip_minus_gmm_worst_group"]).pvalue),
                int(len(paired_df)),
                "ok",
            )
        )
        valid = paired_df[paired_df["chip_coverage_ok"] & paired_df["batch_coverage_ok"]]
        if not valid.empty:
            tests.append(
                (
                    "chip_vs_batch_mean_set_size",
                    float(wilcoxon(valid["chip_minus_batch_mean_set_size"]).pvalue),
                    int(len(valid)),
                    "ok",
                )
            )
        else:
            tests.append(
                (
                    "chip_vs_batch_mean_set_size",
                    None,
                    0,
                    "skipped_no_pairs_met_coverage_filter",
                )
            )
    valid_pvals = [p for _, p, _, status in tests if status == "ok" and p is not None]
    corrected = holm_correction(valid_pvals) if valid_pvals else []
    corrected_iter = iter(corrected)
    test_payload = [
        {
            "test": name,
            "p_value": p,
            "holm_corrected_p_value": next(corrected_iter) if status == "ok" and p is not None else None,
            "n_pairs": n,
            "status": status,
        }
        for name, p, n, status in tests
    ]
    write_json(RESULTS_DIR / "significance_tests.json", test_payload)

    pilot = read_json(RESULTS_DIR / "pilot_gate.json")
    chip_summary = summary_df[summary_df["method"] == "chip_rlcp"].to_dict(orient="records")
    hypothesis_supported = True
    for row in chip_summary:
        if abs(row["marginal_coverage_mean"] - 0.90) > 0.015:
            hypothesis_supported = False
    gmm_compare = summary_df.pivot(index="dataset", columns="method", values="worst_external_group_coverage_mean")
    if "chip_rlcp" in gmm_compare and "gmm_rlcp" in gmm_compare:
        if any((gmm_compare["chip_rlcp"] - gmm_compare["gmm_rlcp"]) < 0.02):
            hypothesis_supported = False

    results_payload = {
        "study_type": "scoped_negative_result_note",
        "implementation_note": pilot["decision"]["implementation_note"],
        "pilot_decision": pilot["decision"],
        "summary_table": grouped_rows,
        "paired_tests": test_payload,
        "full_hypothesis_supported": hypothesis_supported,
    }
    write_json(ROOT / "results.json", results_payload)


if __name__ == "__main__":
    main()
