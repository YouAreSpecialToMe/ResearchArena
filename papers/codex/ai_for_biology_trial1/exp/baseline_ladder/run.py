from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared import config
from exp.shared.pipeline import run_model
from exp.shared.utils import capture_config_snapshot, capture_environment_metadata, save_json, slugify


MODELS = [
    "Train Perturbed Mean",
    "Train Perturbed Median",
    "Non-residualized Ridge",
    "Residualized Ridge",
    "Residualized PLS",
    "Residualized Linear Embedding",
    "Retrieval-only Residual kNN",
]


def main() -> None:
    rows = []
    environment = capture_environment_metadata()
    config_snapshot = capture_config_snapshot(config)
    out_dir = ROOT / "exp" / "baseline_ladder"
    pred_dir = out_dir / "predictions"
    log_dir = out_dir / "logs"
    hp_dir = out_dir / "hyperparams"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    hp_dir.mkdir(parents=True, exist_ok=True)

    preprocess = save_json  # placate linter in this environment
    del preprocess
    preprocess_results = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "seed": int(seed),
                "preprocess_runtime_minutes": audit["runtime_minutes"],
            }
            for dataset, seeds in (
                __import__("json").load((ROOT / "exp" / "preprocess" / "results.json").open("r", encoding="utf-8"))
            )["datasets"].items()
            for seed, audit in seeds.items()
        ]
    )
    for dataset in config.DATASETS:
        for seed in config.SEEDS:
            prep_runtime = float(
                preprocess_results[
                    (preprocess_results["dataset"] == dataset) & (preprocess_results["seed"] == seed)
                ]["preprocess_runtime_minutes"].iloc[0]
            )
            for model_name in MODELS:
                result = run_model(
                    model_name,
                    dataset,
                    seed,
                    log_path=log_dir / f"{dataset}_seed{seed}_{slugify(model_name)}.jsonl",
                    preprocess_runtime_minutes=prep_runtime,
                )
                out_name = f"{dataset}_seed{seed}_{model_name.replace(' ', '_').replace('/', '-')}.npz"
                np.savez_compressed(
                    pred_dir / out_name,
                    predictions=result.predictions,
                    true=result.true,
                    labels=np.array(result.labels, dtype=object),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model_name,
                        **result.metrics,
                        "runtime_minutes": result.runtime_minutes,
                        "peak_memory_mb": result.peak_memory_mb,
                        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
                        "hyperparams": result.hyperparams,
                    }
                )
                save_json(
                    hp_dir / f"{dataset}_seed{seed}_{slugify(model_name)}.json",
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "model": model_name,
                        "hyperparams": result.hyperparams,
                        "metrics": result.metrics,
                        "runtime_minutes": result.runtime_minutes,
                        "peak_memory_mb": result.peak_memory_mb,
                        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
                        "environment": environment,
                        "config": config_snapshot,
                    },
                )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics.csv", index=False)
    save_json(
        out_dir / "results.json",
        {
            "experiment": "baseline_ladder",
            "environment": environment,
            "config": config_snapshot,
            "rows": df.to_dict(orient="records"),
        },
    )


if __name__ == "__main__":
    main()
