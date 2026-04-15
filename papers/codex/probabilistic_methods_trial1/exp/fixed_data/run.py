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
    kendall_safe,
    load_json,
    log_message,
    peak_memory_mb,
    save_csv,
    save_json,
    set_thread_env,
    utc_now_iso,
)
from exp.shared.fixed_data import (
    download_and_prepare_radon,
    fit_advi_samples,
    fit_laplace_samples,
    fit_reference_posterior,
)


def build_replicates(
    ref_samples: np.ndarray, approx_samples: np.ndarray, spec: dict
) -> np.ndarray:
    ref_idx = np.asarray(spec["reference_indices"], dtype=int)
    approx_idx = np.asarray(spec["approx_indices"], dtype=int)
    x = np.empty((ref_idx.shape[0], approx_idx.shape[1] + 1, ref_samples.shape[1]), dtype=float)
    for i in range(ref_idx.shape[0]):
        x[i, 0] = ref_samples[ref_idx[i] % ref_samples.shape[0]]
        x[i, 1:] = approx_samples[np.mod(approx_idx[i], approx_samples.shape[0])]
    return x


def covariance_error(ref_samples: np.ndarray, approx_samples: np.ndarray) -> float:
    ref_cov = np.cov(ref_samples, rowvar=False)
    approx_cov = np.cov(approx_samples, rowvar=False)
    return float(np.linalg.norm(approx_cov - ref_cov, ord="fro"))


def main() -> None:
    start = time.perf_counter()
    set_thread_env()
    dirs = init_experiment("fixed_data")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    log_message(log_path, "Starting fixed-data posterior-SBC study on the radon benchmark.")
    meta = download_and_prepare_radon("data")
    df = pd.read_csv(meta["prepared_path"])
    ref_dir = dirs["results_dir"] / "reference"
    log_message(log_path, "Fitting the reference NUTS posterior from a clean fixed-data model specification.")
    if not (ref_dir / "reference_samples.npy").exists():
        reference = fit_reference_posterior(df, seed=101, output_dir=ref_dir)
    else:
        reference = {
            "samples": np.load(ref_dir / "reference_samples.npy"),
            "meta": load_json(ref_dir / "reference_meta.json"),
        }
    log_message(
        log_path,
        (
            "Reference posterior ready: "
            f"n_samples={reference['meta'].get('n_samples')} "
            f"runtime_seconds={reference['meta'].get('runtime_seconds', 'cached')}"
        ),
    )
    rows = []
    fit_rows = []
    for seed in SEEDS:
        spec = load_json(f"results/setup/seed_{seed}_spec.json")
        fixed_spec = spec["fixed_data"]
        log_message(log_path, f"Fitting or loading fixed-data approximations for seed {seed}.")
        approx_specs = {
            "mean_field_advi": fit_advi_samples(
                df, seed=seed * 100 + 401, method="advi", draws=2000, output_dir=f"results/fixed_data/seed_{seed}"
            ),
            "full_rank_advi": fit_advi_samples(
                df, seed=seed * 100 + 402, method="fullrank_advi", draws=2000, output_dir=f"results/fixed_data/seed_{seed}"
            ),
            "laplace": fit_laplace_samples(
                df, seed=seed * 100 + 403, draws=2000, output_dir=f"results/fixed_data/seed_{seed}"
            ),
        }
        for approx_name, approx in approx_specs.items():
            if approx["samples"] is None:
                skip_path = dirs["exp_dir"] / f"{approx_name}_SKIPPED.md"
                skip_path.write_text(
                    "Approximation failed during execution; see results/fixed_data/seed_* metadata for the exception."
                )
                log_message(log_path, f"{approx_name} failed for seed {seed}; wrote skip note.")
                continue
            fit_meta = approx.get("meta", {})
            fit_rows.append(
                {
                    "seed": seed,
                    "approximation": approx_name,
                    "fit_method": fit_meta.get("method", approx_name),
                    "fit_runtime_seconds": float(fit_meta.get("runtime_seconds", 0.0)),
                    "draws": int(fit_meta.get("draws", 0)),
                    "cache_hit": bool(fit_meta.get("cache_hit", False)),
                    "fit_metadata_path": f"results/fixed_data/seed_{seed}/{fit_meta.get('method', approx_name)}_meta.json",
                }
            )
            log_message(
                log_path,
                (
                    f"Seed {seed} {approx_name} ready: draws={fit_meta.get('draws')} "
                    f"fit_runtime_seconds={fit_meta.get('runtime_seconds', 'cached')}"
                ),
            )
            x = build_replicates(reference["samples"], approx["samples"], spec=fixed_spec)
            log_message(log_path, f"Evaluating diagnostics for seed {seed} approximation {approx_name}.")
            method_specs = [
                ("cosbc", evaluate_family_bundle(
                    x=x, seed=seed, method="cosbc", families=["cov", "tail"], b=199, randomization_spec=fixed_spec
                )),
                ("enriched", evaluate_family_bundle(
                    x=x, seed=seed + 10, method="enriched", families=["cov", "tail"], b=199, randomization_spec=fixed_spec
                )),
                ("scalar", evaluate_family_bundle(
                    x=x, seed=seed + 20, method="scalar", families=["coord", "radius"], b=199, randomization_spec=fixed_spec
                )),
                ("discriminative", evaluate_contextual_baseline(
                    x=x, seed=seed + 30, method="discriminative", b=199, randomization_spec=None
                )),
            ]
            cov_err = covariance_error(reference["samples"], approx["samples"])
            for method_name, result in method_specs:
                eval_runtime = float(result["runtime_minutes"]) * 60.0
                total_runtime = float(fit_meta.get("runtime_seconds", 0.0)) + eval_runtime
                log_message(
                    log_path,
                    (
                        f"Seed {seed} {approx_name} {method_name}: "
                        f"global_pvalue={result['global']['pvalue']:.4f} "
                        f"fit_runtime_seconds={fit_meta.get('runtime_seconds', 0.0):.2f} "
                        f"eval_runtime_seconds={eval_runtime:.2f}"
                    ),
                )
                rows.append(
                    {
                        "experiment": "fixed_data",
                        "seed": seed,
                        "benchmark": "radon_posterior_sbc",
                        "approximation": approx_name,
                        "method": method_name,
                        "R": 60,
                        "M": 24,
                        "B": 199,
                        "dimension": int(x.shape[-1]),
                        "global_pvalue": float(result["global"]["pvalue"]),
                        "global_stat": float(result["global"]["statistic"]),
                        "reject_at_0_05": float(result["global"]["pvalue"] <= 0.05),
                        "reject_at_0_10": float(result["global"]["pvalue"] <= 0.10),
                        "cov_error": cov_err,
                        "cov_pvalue": float(result["families"].get("cov", {}).get("pvalue", np.nan)),
                        "tail_pvalue": float(result["families"].get("tail", {}).get("pvalue", np.nan)),
                        "method_runtime_minutes": float(result["runtime_minutes"]),
                        "fit_runtime_minutes": float(fit_meta.get("runtime_seconds", 0.0)) / 60.0,
                        "end_to_end_runtime_minutes": total_runtime / 60.0,
                        "fit_draws": int(fit_meta.get("draws", 0)),
                        "fit_cache_hit": bool(fit_meta.get("cache_hit", False)),
                        "peak_memory_mb": float(result["peak_memory_mb"]),
                    }
                )
            stale_skip = dirs["exp_dir"] / f"{approx_name}_SKIPPED.md"
            if stale_skip.exists():
                stale_skip.unlink()
    fixed_df = pd.DataFrame(rows)
    save_csv(dirs["results_dir"] / "fixed_data_metrics.csv", fixed_df)
    save_csv(dirs["results_dir"] / "fit_metrics.csv", pd.DataFrame(fit_rows))
    append_jsonl(dirs["runtime_dir"] / "run_manifest.jsonl", rows)
    rank_corrs = []
    for method, sub in fixed_df.groupby("method"):
        rank_corrs.append(
            {
                "method": method,
                "kendall_tau": kendall_safe(sub["cov_error"].tolist(), sub["global_stat"].tolist()),
            }
        )
    save_csv(dirs["results_dir"] / "rank_correlations.csv", pd.DataFrame(rank_corrs))
    payload = {
        "experiment": "fixed_data",
        "created_at_utc": utc_now_iso(),
        "metrics": {
            "rows": int(len(fixed_df)),
            "cosbc_rank_corr": float(
                pd.DataFrame(rank_corrs).set_index("method").get("kendall_tau", pd.Series()).get("cosbc", 0.0)
            ),
            "cosbc_reject_0.05_mean": float(
                fixed_df.loc[fixed_df["method"] == "cosbc", "reject_at_0_05"].mean()
            ),
        },
        "peak_memory_mb": peak_memory_mb(),
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
        "reference_runtime_minutes": float(reference["meta"].get("runtime_seconds", 0.0)) / 60.0,
        "fit_runtime_minutes_total": float(pd.DataFrame(fit_rows)["fit_runtime_seconds"].sum()) / 60.0 if fit_rows else 0.0,
        "discriminative_calibration_skipped": False,
        "discriminative_skip_reason": "",
        "model_spec": {
            "response": "log_radon",
            "predictors": ["floor"],
            "hierarchy": "county varying intercepts",
        },
    }
    save_json(dirs["exp_dir"] / "results.json", payload)
    save_json(dirs["results_dir"] / "results.json", payload)
    save_json(dirs["logs_dir"] / "run_metadata.json", payload)
    skip_note = dirs["exp_dir"] / "discriminative_SKIPPED.md"
    if skip_note.exists():
        skip_note.unlink()
    log_message(log_path, "Fixed-data stage completed.")


if __name__ == "__main__":
    main()
