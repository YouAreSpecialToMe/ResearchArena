from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from exp.shared.common import ensure_dir, load_json, save_json, set_thread_env
from exp.shared.discovery import build_particles, load_particles, save_particles
from exp.shared.pipeline import RolloutConfig, run_rollout
from exp.shared.sem import load_instance


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "_".join(str(part) for part in col if part).strip("_") if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def _run_one(task: dict) -> dict:
    instance = load_instance(Path(task["path"]))
    config = RolloutConfig(**task["config"])
    result_path = Path(task["out_dir"]) / "results.json"
    if result_path.exists() and not task["force_rerun"]:
        result = load_json(result_path)
    else:
        initial_particles = load_particles(Path(task["initial_particles_path"]))
        result = run_rollout(instance, config, Path(task["out_dir"]), task["seed"], initial_particles=initial_particles)
    result["instance_id"] = instance.instance_id
    result["method"] = task["name"]
    result["graph_family"] = instance.graph_family
    result["weight_regime"] = instance.weight_regime
    result["p"] = instance.p
    result["seed"] = instance.seed
    return result


def _build_initial_cache(task: dict) -> str:
    cache_path = Path(task["cache_path"])
    if cache_path.exists() and not task["force_rerun"]:
        return str(cache_path)
    instance = load_instance(Path(task["path"]))
    save_particles(cache_path, build_particles(instance.observational_data, [], instance.seed + 9000))
    return str(cache_path)


def main() -> None:
    set_thread_env()
    force_rerun = os.environ.get("FORCE_RERUN", "0") == "1"
    ensure_dir(Path(__file__).resolve().parent / "logs")
    prep = load_json(Path(__file__).resolve().parents[2] / "exp" / "data_prep" / "results.json")
    pilot = load_json(Path(__file__).resolve().parents[2] / "exp" / "pilot" / "results.json")
    shortlist_k = pilot["shortlist_k"]
    eps = pilot["best_pacer"]["epsilon_stop"]
    tau = pilot["best_aoed"]["tau_stop"]
    eta = pilot["best_aoed"]["eta_stop"]
    methods = {
        "fges_only": {"method": "fges_only"},
        "random_active": {"method": "random_active"},
        "git": {"method": "git"},
        "aoed_lite": {"method": "aoed_lite", "tau_stop": tau, "eta_stop": eta, "shortlist_k": shortlist_k},
        "pacer_no_d": {"method": "pacer_no_d", "epsilon_stop": eps, "disable_detectability": True, "shortlist_k": shortlist_k},
        "pacer_cert": {"method": "pacer_cert", "epsilon_stop": eps, "shortlist_k": shortlist_k},
        "pacer_fixed_batch": {"method": "pacer_fixed_batch", "epsilon_stop": eps, "fixed_batch_size": 50, "shortlist_k": shortlist_k},
        "pacer_full_budget": {"method": "pacer_cert", "epsilon_stop": eps, "disable_early_stop": True, "shortlist_k": shortlist_k},
    }
    cache_tasks = []
    for path in prep["core"]:
        cache_path = Path(__file__).resolve().parents[2] / "artifacts" / "states" / f"{Path(path).stem}_initial.pkl"
        cache_tasks.append({"path": path, "cache_path": str(cache_path), "force_rerun": force_rerun})
    with ProcessPoolExecutor(max_workers=2) as pool:
        cache_paths = list(pool.map(_build_initial_cache, cache_tasks))
    initial_particle_paths = {task["path"]: cache_path for task, cache_path in zip(cache_tasks, cache_paths)}
    tasks = []
    for path in prep["core"]:
        instance_seed = load_instance(Path(path)).seed
        for name, cfg in methods.items():
            tasks.append(
                {
                    "path": path,
                    "name": name,
                    "config": cfg,
                    "seed": instance_seed,
                    "initial_particles_path": initial_particle_paths[path],
                    "out_dir": str(ensure_dir(Path(__file__).resolve().parent / name / Path(path).stem)),
                    "force_rerun": force_rerun,
                }
            )
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(_run_one, tasks))
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parent / "benchmark_rollouts.csv", index=False)
    checkpoint_rows = []
    for row in rows:
        for checkpoint in row.get("checkpoints", []):
            checkpoint_rows.append(
                {
                    "instance_id": row["instance_id"],
                    "method": row["method"],
                    "graph_family": row["graph_family"],
                    "weight_regime": row["weight_regime"],
                    "p": row["p"],
                    "seed": row["seed"],
                    **checkpoint,
                }
            )
    pd.DataFrame(checkpoint_rows).to_csv(Path(__file__).resolve().parent / "checkpoint_records.csv", index=False)
    summary = flatten_columns(
        df.groupby("method")[["directed_f1", "shd", "unused_budget", "runtime_seconds", "auc_directed_f1", "peak_rss_mb"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(Path(__file__).resolve().parent / "summary.csv", index=False)
    save_json(
        Path(__file__).resolve().parent / "results.json",
        {
            "num_rollouts": int(len(rows)),
            "methods": sorted(df["method"].unique().tolist()),
            "summary_path": str(Path(__file__).resolve().parent / "summary.csv"),
            "rollouts_path": str(Path(__file__).resolve().parent / "benchmark_rollouts.csv"),
            "checkpoints_path": str(Path(__file__).resolve().parent / "checkpoint_records.csv"),
            "force_rerun": force_rerun,
        },
    )


if __name__ == "__main__":
    main()
