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
    exhaustive_global_pvalue,
    family_rank_matrix,
    init_experiment,
    ks_uniform_distance,
    load_array,
    load_json,
    log_message,
    monte_carlo_global_pvalue,
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
    dirs = init_experiment("null_checks")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    log_message(log_path, "Starting null calibration, tie-handling, and tiny exact checks.")
    rows = []
    rank_rows = []
    for seed in SEEDS:
        spec = load_json(f"results/setup/seed_{seed}_spec.json")
        log_message(log_path, f"Running null-check seed {seed} with saved replicate specifications.")
        for d in [8, 16]:
            setting = spec["settings"][f"toeplitz_null_d{d}"]
            bundle = {"x": load_array(setting["x_path"])}
            eval_res = evaluate_family_bundle(
                x=bundle["x"],
                seed=seed,
                method="cosbc",
                families=["cov", "tail"],
                b=199,
                randomization_spec=setting,
            )
            row = {
                "experiment": "null_checks",
                "seed": seed,
                "benchmark": "toeplitz_null",
                "setting": f"toeplitz_null_d{d}",
                "method": "cosbc",
                "approximation": "exact",
                "score_kind": "energy",
                "aggregation": "tmax",
                "R": 100,
                "M": 24,
                "B": 199,
                "dimension": d,
                "cov_pvalue": eval_res["families"]["cov"]["pvalue"],
                "tail_pvalue": eval_res["families"]["tail"]["pvalue"],
                "global_pvalue": eval_res["global"]["pvalue"],
                "cov_ks": ks_uniform_distance(
                    np.array(eval_res["families"]["cov"]["ranks"]), m=24
                ),
                "tail_ks": ks_uniform_distance(
                    np.array(eval_res["families"]["tail"]["ranks"]), m=24
                ),
                "runtime_minutes": float(eval_res["runtime_minutes"]),
                "peak_memory_mb": float(eval_res["peak_memory_mb"]),
            }
            rows.append(row)
            rank_rows.extend(
                {
                    "experiment": "null_checks",
                    "seed": seed,
                    "benchmark": "toeplitz_null",
                    "setting": f"toeplitz_null_d{d}",
                    "family": fam,
                    "method": "cosbc",
                    "approximation": "exact",
                    "R": 100,
                    "M": 24,
                    "B": 199,
                    "dimension": d,
                    "rank": rank,
                    "pit": pit,
                }
                for fam in ["cov", "tail"]
                for rank, pit in zip(
                    eval_res["families"][fam]["ranks"], eval_res["families"][fam]["pits"]
                )
            )
        tie_setting = spec["settings"]["tie_quantized_d8"]
        tie_bundle = {"x": load_array(tie_setting["x_path"])}
        tie_eval = evaluate_family_bundle(
            x=tie_bundle["x"],
            seed=seed + 200,
            method="cosbc",
            families=["cov", "tail"],
            b=199,
            quantize=0.25,
            randomization_spec=tie_setting,
        )
        rows.append(
            {
                "experiment": "null_checks",
                "seed": seed,
                "benchmark": "tie_quantized",
                "setting": "tie_quantized_d8",
                "method": "cosbc",
                "approximation": "exact",
                "score_kind": "energy",
                "aggregation": "tmax",
                "R": 100,
                "M": 24,
                "B": 199,
                "dimension": 8,
                "cov_pvalue": tie_eval["families"]["cov"]["pvalue"],
                "tail_pvalue": tie_eval["families"]["tail"]["pvalue"],
                "global_pvalue": tie_eval["global"]["pvalue"],
                "cov_ks": ks_uniform_distance(
                    np.array(tie_eval["families"]["cov"]["ranks"]), m=24
                ),
                "tail_ks": ks_uniform_distance(
                    np.array(tie_eval["families"]["tail"]["ranks"]), m=24
                ),
                "runtime_minutes": float(tie_eval["runtime_minutes"]),
                "peak_memory_mb": float(tie_eval["peak_memory_mb"]),
            }
        )
        tiny_setting = spec["settings"]["toeplitz_tiny_d8"]
        tiny = {"x": load_array(tiny_setting["x_path"])}
        family_pits = {
            fam: family_rank_matrix(
                tiny["x"],
                family=fam,
                method="cosbc",
                seed=seed + i,
                score_kind="energy",
                transform_seeds=tiny_setting["transform_seeds"],
                score_seeds=tiny_setting["score_seeds"][fam],
            )["pits"]
            for i, fam in enumerate(["cov", "tail"])
        }
        exact_stat, exact_p = exhaustive_global_pvalue(family_pits)
        mc_stat, mc_p = monte_carlo_global_pvalue(
            family_pits,
            rng=np.random.default_rng(seed + 500),
            draws=999,
            relabel_indices=np.asarray(tiny_setting["relabel_indices_999"], dtype=int),
        )
        rows.append(
            {
                "experiment": "null_checks",
                "seed": seed,
                "benchmark": "tiny_exact_check",
                "setting": "tiny_exact_check",
                "method": "cosbc",
                "approximation": "exact",
                "score_kind": "energy",
                "aggregation": "tmax",
                "R": 4,
                "M": 7,
                "B": 999,
                "dimension": 8,
                "cov_pvalue": np.nan,
                "tail_pvalue": np.nan,
                "global_pvalue": mc_p,
                "cov_ks": np.nan,
                "tail_ks": np.nan,
                "exact_global_stat": mc_stat if np.isclose(exact_stat, mc_stat) else exact_stat,
                "exact_global_pvalue": exact_p,
                "mc_global_pvalue": mc_p,
                "abs_p_gap": abs(exact_p - mc_p),
                "runtime_minutes": np.nan,
                "peak_memory_mb": peak_memory_mb(),
            }
        )
        kernel_setting = spec["settings"]["toeplitz_kernel_d8"]
        kernel_bundle = {"x": load_array(kernel_setting["x_path"])}
        kernel_eval = evaluate_family_bundle(
            x=kernel_bundle["x"],
            seed=seed + 800,
            method="cosbc",
            families=["cov", "tail"],
            b=199,
            score_kind="kernel",
            randomization_spec=kernel_setting,
        )
        rows.append(
            {
                "experiment": "null_checks",
                "seed": seed,
                "benchmark": "kernel_null",
                "setting": "kernel_null_d8",
                "method": "cosbc",
                "approximation": "exact",
                "score_kind": "kernel",
                "aggregation": "tmax",
                "R": 100,
                "M": 24,
                "B": 199,
                "dimension": 8,
                "cov_pvalue": kernel_eval["families"]["cov"]["pvalue"],
                "tail_pvalue": kernel_eval["families"]["tail"]["pvalue"],
                "global_pvalue": kernel_eval["global"]["pvalue"],
                "cov_ks": ks_uniform_distance(
                    np.array(kernel_eval["families"]["cov"]["ranks"]), m=24
                ),
                "tail_ks": ks_uniform_distance(
                    np.array(kernel_eval["families"]["tail"]["ranks"]), m=24
                ),
                "runtime_minutes": float(kernel_eval["runtime_minutes"]),
                "peak_memory_mb": float(kernel_eval["peak_memory_mb"]),
            }
        )
    df = pd.DataFrame(rows)
    save_csv(dirs["results_dir"] / "null_metrics.csv", df)
    save_csv(dirs["results_dir"] / "null_ranks.csv", pd.DataFrame(rank_rows))
    append_jsonl(dirs["runtime_dir"] / "run_manifest.jsonl", rows)
    payload = {
        "experiment": "null_checks",
        "created_at_utc": utc_now_iso(),
        "metrics": {
            "global_reject_0.05_mean": float(np.mean(df["global_pvalue"] <= 0.05)),
            "global_reject_0.05_std": float(np.std(df["global_pvalue"] <= 0.05, ddof=0)),
            "global_reject_0.10_mean": float(np.mean(df["global_pvalue"] <= 0.10)),
            "tiny_exact_gap_mean": float(
                df.loc[df["setting"] == "tiny_exact_check", "abs_p_gap"].mean()
            ),
        },
        "n_rows": int(len(df)),
        "peak_memory_mb": peak_memory_mb(),
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
    }
    save_json(dirs["exp_dir"] / "results.json", payload)
    save_json(dirs["results_dir"] / "results.json", payload)
    save_json(dirs["logs_dir"] / "run_metadata.json", payload)
    log_message(log_path, "Null-check stage completed.")


if __name__ == "__main__":
    main()
