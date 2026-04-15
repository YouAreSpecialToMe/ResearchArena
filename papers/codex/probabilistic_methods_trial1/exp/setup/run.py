from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.core import (
    SEEDS,
    THREAD_ENV,
    block_pair_cov,
    generate_gaussian_replicates,
    init_experiment,
    peak_memory_mb,
    save_array,
    save_json,
    set_thread_env,
    log_message,
    toeplitz_cov,
    utc_now_iso,
)
from exp.shared.fixed_data import download_and_prepare_radon


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    start = time.perf_counter()
    set_thread_env()
    dirs = init_experiment("setup")
    log_path = dirs["logs_dir"] / "run_stage2.log"
    log_path.write_text("")
    log_message(log_path, "Starting setup stage with deterministic CPU-only configuration.")
    for seed in SEEDS:
        seed_dir = dirs["results_dir"] / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_message(log_path, f"Creating shared replicate specification for seed {seed}.")
        rng = np.random.default_rng(seed)
        settings = {}
        for d in [8, 16]:
            synthetic_settings = [
                (f"toeplitz_null_d{d}", "exact", toeplitz_cov(d), 100, 24, None),
                (f"block_localization_ZeroPair_d{d}", "ZeroPair", block_pair_cov(d), 100, 24, (0, 1)),
                (f"block_localization_FlipPair_d{d}", "FlipPair", block_pair_cov(d), 100, 24, (0, 1)),
            ]
            if d == 8:
                synthetic_settings.extend(
                    [
                        (f"toeplitz_tiny_d{d}", "exact", toeplitz_cov(d), 4, 7, None),
                        (f"toeplitz_kernel_d{d}", "exact", toeplitz_cov(d), 100, 24, None),
                        (f"tie_quantized_d{d}", "exact", toeplitz_cov(d), 100, 24, None),
                    ]
                )
            for name, approx, prior_cov, r, m, pair in synthetic_settings:
                bundle = generate_gaussian_replicates(
                    seed=int(rng.integers(0, 2**32 - 1)),
                    d=d,
                    r=r,
                    m=m,
                    prior_cov=prior_cov,
                    approx=approx,
                    pair=pair,
                )
                prefix = seed_dir / name
                save_array(f"{prefix}_x.npy", bundle["x"])
                save_array(f"{prefix}_theta.npy", bundle["theta"])
                save_array(f"{prefix}_y.npy", bundle["y"])
                settings[name] = {
                    "x_path": str(prefix.with_name(prefix.name + "_x.npy")),
                    "theta_path": str(prefix.with_name(prefix.name + "_theta.npy")),
                    "y_path": str(prefix.with_name(prefix.name + "_y.npy")),
                    "r": r,
                    "m": m,
                    "dimension": d,
                    "transform_seeds": rng.integers(0, 2**32 - 1, size=r, dtype="uint32").astype(int).tolist(),
                    "score_seeds": {
                        family: rng.integers(0, 2**32 - 1, size=r, dtype="uint32").astype(int).tolist()
                        for family in ["cov", "tail", "coord", "radius"]
                    },
                    "relabel_indices_199": rng.integers(0, m + 1, size=(199, r), dtype="int64").tolist(),
                    "relabel_indices_999": rng.integers(0, m + 1, size=(999, r), dtype="int64").tolist(),
                    "context_negative_indices": rng.integers(1, m + 1, size=r, dtype="int64").tolist(),
                    "context_label_permutations_199": [
                        rng.permutation(2 * r).astype(int).tolist() for _ in range(199)
                    ],
                    "pair": pair,
                    "approximation": approx,
                }
            for approx in ["Diag", "Shrink(0.25)", "Shrink(0.50)", "Shrink(0.75)", "TailMix"]:
                name = f"toeplitz_power_{approx.replace('(', '_').replace(')', '').replace('.', 'p')}_d{d}"
                bundle = generate_gaussian_replicates(
                    seed=int(rng.integers(0, 2**32 - 1)),
                    d=d,
                    r=100,
                    m=24,
                    prior_cov=toeplitz_cov(d),
                    approx=approx,
                )
                prefix = seed_dir / name
                save_array(f"{prefix}_x.npy", bundle["x"])
                save_array(f"{prefix}_theta.npy", bundle["theta"])
                save_array(f"{prefix}_y.npy", bundle["y"])
                settings[name] = {
                    "x_path": str(prefix.with_name(prefix.name + "_x.npy")),
                    "theta_path": str(prefix.with_name(prefix.name + "_theta.npy")),
                    "y_path": str(prefix.with_name(prefix.name + "_y.npy")),
                    "r": 100,
                    "m": 24,
                    "dimension": d,
                    "transform_seeds": rng.integers(0, 2**32 - 1, size=100, dtype="uint32").astype(int).tolist(),
                    "score_seeds": {
                        family: rng.integers(0, 2**32 - 1, size=100, dtype="uint32").astype(int).tolist()
                        for family in ["cov", "tail", "coord", "radius"]
                    },
                    "relabel_indices_199": rng.integers(0, 25, size=(199, 100), dtype="int64").tolist(),
                    "relabel_indices_999": rng.integers(0, 25, size=(999, 100), dtype="int64").tolist(),
                    "context_negative_indices": rng.integers(1, 25, size=100, dtype="int64").tolist(),
                    "context_label_permutations_199": [
                        rng.permutation(200).astype(int).tolist() for _ in range(199)
                    ],
                    "pair": None,
                    "approximation": approx,
                }
        fixed_r = 60
        fixed_m = 24
        fixed_ref_size = 3000
        fixed_approx_size = 2000
        spec = {
            "seed": seed,
            "created_at_utc": utc_now_iso(),
            "seed_dir": str(seed_dir),
            "settings": settings,
            "fixed_data": {
                "reference_indices": rng.integers(0, fixed_ref_size, size=fixed_r, dtype="int64").astype(int).tolist(),
                "approx_indices": rng.integers(0, fixed_approx_size, size=(fixed_r, fixed_m), dtype="int64").astype(int).tolist(),
                "transform_seeds": rng.integers(0, 2**32 - 1, size=fixed_r, dtype="uint32").astype(int).tolist(),
                "score_seeds": {
                    family: rng.integers(0, 2**32 - 1, size=fixed_r, dtype="uint32").astype(int).tolist()
                    for family in ["cov", "tail", "coord", "radius"]
                },
                "relabel_indices_199": rng.integers(0, fixed_m + 1, size=(199, fixed_r), dtype="int64").astype(int).tolist(),
            },
        }
        save_json(dirs["results_dir"] / f"seed_{seed}_spec.json", spec)
    radon_meta = download_and_prepare_radon("data")
    log_message(log_path, "Prepared radon fixed-data benchmark and saved runtime schedule.")
    nproc_visible = command_output(["nproc"])
    nproc_all = command_output(["nproc", "--all"])
    nproc_onln = command_output(["getconf", "_NPROCESSORS_ONLN"])
    schedule = {
        "created_at_utc": utc_now_iso(),
        "resources": {
            "nproc": nproc_visible,
            "nproc_all": nproc_all,
            "getconf_nprocessors_onln": nproc_onln,
            "free_h": command_output(["free", "-h"]),
            "nvidia_smi": command_output(["nvidia-smi"]),
            "cpu_budget_cores": 2,
            "ram_budget_gb": 128,
            "gpu_budget": 0,
        },
        "resource_notes": {
            "parallel_execution_note": (
                "Stage-2 runs were scheduled for at most 2 concurrent jobs. "
                f"System probes returned nproc={nproc_visible}, nproc --all={nproc_all}, "
                f"and getconf _NPROCESSORS_ONLN={nproc_onln}."
            )
        },
        "thread_env": THREAD_ENV,
        "planned_runtime_hours": 8.0,
        "batches": [
            {"order": 1, "name": "setup", "experiments": ["setup"], "parallel_jobs": 1, "budget_hours": 0.3},
            {"order": 2, "name": "null_validation", "experiments": ["null_checks"], "parallel_jobs": 1, "budget_hours": 1.2},
            {"order": 3, "name": "synthetic", "experiments": ["power", "localization"], "parallel_jobs": 2, "budget_hours": 4.2},
            {"order": 4, "name": "fixed_data", "experiments": ["fixed_data"], "parallel_jobs": 1, "budget_hours": 1.7},
            {"order": 5, "name": "ablations", "experiments": ["ablations"], "parallel_jobs": 1, "budget_hours": 0.4},
            {"order": 6, "name": "aggregation", "experiments": ["aggregate"], "parallel_jobs": 1, "budget_hours": 0.2},
        ],
        "seed_policy": {
            "experiment_seeds": SEEDS,
            "shared_spec_files": [
                str(dirs["results_dir"] / f"seed_{seed}_spec.json") for seed in SEEDS
            ],
        },
        "cwd": os.getcwd(),
    }
    save_json(dirs["runtime_dir"] / "schedule.json", schedule)
    payload = {
        "experiment": "setup",
        "created_at_utc": utc_now_iso(),
        "python_env": THREAD_ENV,
        "seeds": SEEDS,
        "radon": radon_meta,
        "schedule_path": str(dirs["runtime_dir"] / "schedule.json"),
        "peak_memory_mb": peak_memory_mb(),
        "runtime_minutes": (time.perf_counter() - start) / 60.0,
    }
    save_json(dirs["exp_dir"] / "results.json", payload)
    save_json(dirs["results_dir"] / "results.json", payload)
    save_json(dirs["logs_dir"] / "run_metadata.json", payload)
    log_message(log_path, "Setup stage completed.")


if __name__ == "__main__":
    main()
