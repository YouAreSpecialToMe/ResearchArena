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
    evaluate_family_bundle,
    generate_gaussian_replicates,
    init_experiment,
    load_array,
    load_json,
    log_message,
    peak_memory_mb,
    save_csv,
    save_json,
    set_thread_env,
    toeplitz_cov,
    utc_now_iso,
)


def main() -> None:
    start = time.perf_counter()
    set_thread_env()
    dirs = init_experiment("ablations")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    log_message(log_path, "Starting ablation studies.")
    rows = []
    for seed in SEEDS:
        spec = load_json(f"results/setup/seed_{seed}_spec.json")
        for condition in ["Shrink(0.50)", "TailMix"]:
            setting_name = f"toeplitz_power_{condition.replace('(', '_').replace(')', '').replace('.', 'p')}_d16"
            setting = spec["settings"][setting_name]
            bundle = {"x": load_array(setting["x_path"])}
            for name, families in [
                ("cov_only", ["cov"]),
                ("tail_only", ["tail"]),
                ("combined", ["cov", "tail"]),
            ]:
                res = evaluate_family_bundle(
                    x=bundle["x"], seed=seed, method="cosbc", families=families, b=199, randomization_spec=setting
                )
                rows.append(
                    {
                        "experiment": "ablations",
                        "seed": seed,
                        "study": "feature_family",
                        "condition": condition,
                        "variant": name,
                        "method": "cosbc",
                        "approximation": condition,
                        "aggregation": "tmax",
                        "R": 100,
                        "M": 24,
                        "B": 199,
                        "dimension": 16,
                        "pvalue": float(res["global"]["pvalue"]),
                        "reject_at_0_05": float(res["global"]["pvalue"] <= 0.05),
                        "runtime_minutes": float(res["runtime_minutes"]),
                        "peak_memory_mb": float(res["peak_memory_mb"]),
                    }
                )
            for method in ["cosbc", "enriched"]:
                res = evaluate_family_bundle(
                    x=bundle["x"], seed=seed + 5, method=method, families=["cov", "tail"], b=199, randomization_spec=setting
                )
                rows.append(
                    {
                        "experiment": "ablations",
                        "seed": seed,
                        "study": "pooled_ranking",
                        "condition": condition,
                        "variant": method,
                        "method": method,
                        "approximation": condition,
                        "aggregation": "tmax",
                        "R": 100,
                        "M": 24,
                        "B": 199,
                        "dimension": 16,
                        "pvalue": float(res["global"]["pvalue"]),
                        "reject_at_0_05": float(res["global"]["pvalue"] <= 0.05),
                        "runtime_minutes": float(res["runtime_minutes"]),
                        "peak_memory_mb": float(res["peak_memory_mb"]),
                    }
                )
        for score_kind, condition in [("kernel", "exact"), ("kernel", "Shrink(0.50)")]:
            setting_name = (
                "toeplitz_kernel_d8"
                if condition == "exact"
                else f"toeplitz_power_{condition.replace('(', '_').replace(')', '').replace('.', 'p')}_d8"
            )
            setting = spec["settings"][setting_name]
            bundle = {"x": load_array(setting["x_path"])}
            res = evaluate_family_bundle(
                x=bundle["x"], seed=seed + 10, method="cosbc", families=["cov", "tail"], b=199, score_kind=score_kind, randomization_spec=setting
            )
            rows.append(
                {
                    "experiment": "ablations",
                    "seed": seed,
                    "study": "score_choice",
                    "condition": condition,
                    "variant": score_kind,
                    "method": "cosbc",
                    "approximation": condition,
                    "aggregation": "tmax",
                    "R": 100,
                    "M": 24,
                    "B": 199,
                    "dimension": 8,
                    "pvalue": float(res["global"]["pvalue"]),
                    "reject_at_0_05": float(res["global"]["pvalue"] <= 0.05),
                    "runtime_minutes": float(res["runtime_minutes"]),
                    "peak_memory_mb": float(res["peak_memory_mb"]),
                }
            )
        agg_setting = spec["settings"]["toeplitz_power_Shrink_0p50_d16"]
        agg_bundle = {"x": load_array(agg_setting["x_path"])}
        for method in ["cosbc", "enriched"]:
            res = evaluate_family_bundle(
                x=agg_bundle["x"], seed=seed + 20, method=method, families=["cov", "tail"], b=199, randomization_spec=agg_setting
            )
            for aggregation in ["tmax", "bonferroni", "fisher"]:
                agg = res["aggregations"][aggregation]
                rows.append(
                    {
                        "experiment": "ablations",
                        "seed": seed,
                        "study": "aggregation",
                        "condition": "Shrink(0.50)",
                        "variant": aggregation,
                        "method": method,
                        "approximation": "Shrink(0.50)",
                        "aggregation": aggregation,
                        "R": 100,
                        "M": 24,
                        "B": 199,
                        "dimension": 16,
                        "pvalue": float(agg["pvalue"]),
                        "reject_at_0_05": float(agg["pvalue"] <= 0.05),
                        "runtime_minutes": float(res["runtime_minutes"]),
                        "peak_memory_mb": float(res["peak_memory_mb"]),
                    }
                )
        for r_val, m_val in [(80, 16), (100, 24), (140, 32)]:
            for method in ["cosbc", "enriched"]:
                bundle = generate_gaussian_replicates(
                    seed=seed * 1000 + r_val * 10 + m_val,
                    d=8,
                    r=r_val,
                    m=m_val,
                    prior_cov=toeplitz_cov(8),
                    approx="Shrink(0.50)",
                )
                res = evaluate_family_bundle(
                    x=bundle["x"],
                    seed=seed + r_val,
                    method=method,
                    families=["cov", "tail"] if method != "scalar" else ["coord", "radius"],
                    b=199,
                )
                rows.append(
                    {
                        "experiment": "ablations",
                        "seed": seed,
                        "study": "budget_sensitivity",
                        "condition": f"R{r_val}_M{m_val}",
                        "variant": method,
                        "method": method,
                        "approximation": "Shrink(0.50)",
                        "aggregation": "tmax",
                        "R": r_val,
                        "M": m_val,
                        "B": 199,
                        "dimension": 8,
                        "pvalue": float(res["global"]["pvalue"]),
                        "reject_at_0_05": float(res["global"]["pvalue"] <= 0.05),
                        "runtime_minutes": float(res["runtime_minutes"]),
                        "peak_memory_mb": float(res["peak_memory_mb"]),
                    }
                )
    df = pd.DataFrame(rows)
    save_csv(dirs["results_dir"] / "ablation_metrics.csv", df)
    append_jsonl(dirs["runtime_dir"] / "run_manifest.jsonl", rows)
    payload = {
        "experiment": "ablations",
        "created_at_utc": utc_now_iso(),
        "metrics": {"rows": int(len(df))},
        "peak_memory_mb": peak_memory_mb(),
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
    }
    save_json(dirs["exp_dir"] / "results.json", payload)
    save_json(dirs["results_dir"] / "results.json", payload)
    save_json(dirs["logs_dir"] / "run_metadata.json", payload)
    log_message(log_path, "Ablation stage completed.")


if __name__ == "__main__":
    main()
