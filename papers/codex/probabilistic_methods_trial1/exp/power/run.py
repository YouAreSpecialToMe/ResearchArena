from __future__ import annotations

import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.core import (
    SEEDS,
    append_jsonl,
    evaluate_contextual_baseline,
    evaluate_family_bundle,
    init_experiment,
    load_array,
    load_json,
    log_message,
    peak_memory_mb,
    save_csv,
    save_json,
    set_thread_env,
    utc_now_iso,
)


def method_pvalue(result: dict) -> float:
    return float(result["global"]["pvalue"])


def main() -> None:
    start = time.perf_counter()
    set_thread_env()
    dirs = init_experiment("power")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    selection_log_path = dirs["logs_dir"] / "discriminative_selection.jsonl"
    if selection_log_path.exists():
        selection_log_path.unlink()
    log_message(log_path, "Starting synthetic power study with paired saved replicate pools.")
    rows = []
    conditions = []
    for d in [8, 16]:
        conditions.extend([(d, "Diag")])
        conditions.extend([(d, f"Shrink({lam:.2f})") for lam in [0.25, 0.50, 0.75]])
        conditions.extend([(d, "TailMix")])
    for seed in SEEDS:
        spec = load_json(f"results/setup/seed_{seed}_spec.json")
        for d, approx in conditions:
            setting_name = f"toeplitz_power_{approx.replace('(', '_').replace(')', '').replace('.', 'p')}_d{d}"
            setting = spec["settings"][setting_name]
            bundle = {"x": load_array(setting["x_path"])}
            log_message(log_path, f"Seed {seed}: running power condition {approx} at d={d}.")
            methods = {
                "cosbc_cov": evaluate_family_bundle(
                    x=bundle["x"], seed=seed, method="cosbc", families=["cov"], b=199, randomization_spec=setting
                ),
                "cosbc_tail": evaluate_family_bundle(
                    x=bundle["x"], seed=seed + 10, method="cosbc", families=["tail"], b=199, randomization_spec=setting
                ),
                "cosbc_tmax": evaluate_family_bundle(
                    x=bundle["x"], seed=seed + 20, method="cosbc", families=["cov", "tail"], b=199, randomization_spec=setting
                ),
                "enriched": evaluate_family_bundle(
                    x=bundle["x"], seed=seed + 30, method="enriched", families=["cov", "tail"], b=199, randomization_spec=setting
                ),
                "scalar": evaluate_family_bundle(
                    x=bundle["x"], seed=seed + 40, method="scalar", families=["coord", "radius"], b=199, randomization_spec=setting
                ),
                "discriminative": evaluate_contextual_baseline(
                    x=bundle["x"],
                    seed=seed + 50,
                    method="discriminative",
                    b=199,
                    randomization_spec=setting,
                    selection_log_path=selection_log_path,
                    selection_context={"seed": seed, "dimension": d, "condition": approx, "method": "discriminative"},
                ),
                "energy_distance": evaluate_contextual_baseline(
                    x=bundle["x"], seed=seed + 60, method="energy_distance", b=199, randomization_spec=setting
                ),
            }
            for method, result in methods.items():
                approximation = approx if approx != "Diag" else "Diag"
                global_result = result["global"]
                family_results = result.get("families", {})
                rows.append(
                    {
                        "experiment": "power",
                        "seed": seed,
                        "benchmark": "toeplitz_power",
                        "dimension": d,
                        "condition": approximation,
                        "method": method,
                        "approximation": approximation,
                        "score_kind": "energy",
                        "aggregation": "tmax",
                        "R": 100,
                        "M": 24,
                        "B": 199,
                        "global_pvalue": method_pvalue(result),
                        "global_stat": float(global_result.get("statistic", np.nan)),
                        "cov_pvalue": float(family_results.get("cov", {}).get("pvalue", np.nan)),
                        "tail_pvalue": float(family_results.get("tail", {}).get("pvalue", np.nan)),
                        "reject_at_0_05": float(method_pvalue(result) <= 0.05),
                        "reject_at_0_10": float(method_pvalue(result) <= 0.10),
                        "runtime_minutes": float(result.get("runtime_minutes", np.nan)),
                        "peak_memory_mb": float(result.get("peak_memory_mb", np.nan)),
                    }
                )
    df = pd.DataFrame(rows)
    save_csv(dirs["results_dir"] / "power_metrics.csv", df)
    append_jsonl(dirs["runtime_dir"] / "run_manifest.jsonl", rows)
    paired = (
        df[df["method"].isin(["cosbc_tmax", "enriched"])]
        .pivot_table(
            index=["seed", "dimension", "condition"],
            columns="method",
            values="reject_at_0_05",
        )
        .reset_index()
    )
    if {"cosbc_tmax", "enriched"}.issubset(paired.columns):
        paired["cosbc_minus_enriched"] = paired["cosbc_tmax"] - paired["enriched"]
        condition_gaps = (
            paired.groupby(["dimension", "condition"], dropna=False)["cosbc_minus_enriched"]
            .agg(["mean", "std"])
            .reset_index()
        )
        synthetic_wins = int((condition_gaps["mean"] >= 0.08).sum())
    else:
        synthetic_wins = 0
    payload = {
        "experiment": "power",
        "created_at_utc": utc_now_iso(),
        "metrics": {
            "synthetic_conditions_meeting_gap": synthetic_wins,
            "cosbc_reject_0.05_mean": float(
                df.loc[df["method"] == "cosbc_tmax", "reject_at_0_05"].mean()
            ),
            "enriched_reject_0.05_mean": float(
                df.loc[df["method"] == "enriched", "reject_at_0_05"].mean()
            ),
            "rows": int(len(df)),
        },
        "peak_memory_mb": peak_memory_mb(),
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
    }
    save_json(dirs["exp_dir"] / "results.json", payload)
    save_json(dirs["results_dir"] / "results.json", payload)
    save_json(dirs["logs_dir"] / "run_metadata.json", payload)
    log_message(log_path, "Synthetic power study completed.")


if __name__ == "__main__":
    main()
