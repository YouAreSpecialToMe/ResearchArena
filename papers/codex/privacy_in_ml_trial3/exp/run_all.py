from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from exp.shared.aggregate import aggregate
from exp.shared.config import ABLATIONS, MAIN_METHODS, OUTPUT_ROOT, PILOT_ROOT, PILOTS, SEEDS, ensure_output_dirs


ROOT = Path(__file__).resolve().parents[1]


def _log_paths(dataset: str, method: str, seed: int, stage: str, suffix: str = "") -> tuple[Path, Path]:
    log_dir = ROOT / "exp" / method / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stage}_{dataset}_{method}_{seed}{suffix}"
    return log_dir / f"{stem}.stdout.log", log_dir / f"{stem}.stderr.log"


def _archive_pilot_artifacts(run_name: str, pilot_stem: str, stdout_path: Path, stderr_path: Path) -> None:
    mapping = {
        OUTPUT_ROOT / "configs" / f"{run_name}.json": PILOT_ROOT / "configs" / f"{pilot_stem}.json",
        OUTPUT_ROOT / "checkpoints" / f"{run_name}.pt": PILOT_ROOT / "checkpoints" / f"{pilot_stem}.pt",
        OUTPUT_ROOT / "traces" / f"{run_name}.parquet": PILOT_ROOT / "traces" / f"{pilot_stem}.parquet",
        OUTPUT_ROOT / "traces" / f"{run_name}_refreshes.parquet": PILOT_ROOT / "traces" / f"{pilot_stem}_refreshes.parquet",
        OUTPUT_ROOT / "metrics" / f"{run_name}.json": PILOT_ROOT / "metrics" / f"{pilot_stem}.json",
        OUTPUT_ROOT / "attacks" / f"{run_name}_scores.parquet": PILOT_ROOT / "attacks" / f"{pilot_stem}_scores.parquet",
        OUTPUT_ROOT / "attacks" / f"{run_name}_summary.json": PILOT_ROOT / "attacks" / f"{pilot_stem}_summary.json",
        stdout_path: PILOT_ROOT / "logs" / stdout_path.name,
        stderr_path: PILOT_ROOT / "logs" / stderr_path.name,
    }
    for src, dst in mapping.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _append_stage_manifest(stage: str, row: dict) -> None:
    out_path = PILOT_ROOT / "tables" / f"{stage}_runs.csv" if stage == "pilot" else OUTPUT_ROOT / "tables" / "full_run_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = out_path.exists()
    with out_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def run_subprocess(args, stage: str):
    dataset = args[args.index("--dataset") + 1]
    method = args[args.index("--method") + 1]
    seed = int(args[args.index("--seed") + 1])
    lambda_suffix = ""
    if "--relaxloss-lambda" in args:
        lambda_suffix = f"_lambda_{args[args.index('--relaxloss-lambda') + 1].replace('.', 'p')}"
    stdout_path, stderr_path = _log_paths(dataset, method, seed, stage, suffix=lambda_suffix)
    cmd = [sys.executable, "-m", "exp.shared.runner", *args]
    start = time.time()
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=stdout_handle, stderr=stderr_handle)
    elapsed = time.time() - start
    run_name = f"{dataset}_{method}_{seed}"
    _append_stage_manifest(
        stage,
        {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "stage": stage,
            "wall_clock_seconds": elapsed,
            "stdout_log": str(stdout_path.resolve()),
            "stderr_log": str(stderr_path.resolve()),
        },
    )
    if stage == "pilot":
        _archive_pilot_artifacts(run_name, f"{run_name}{lambda_suffix}", stdout_path, stderr_path)


def tune_relaxloss():
    candidates = [0.5, 1.0, 2.0]
    tuned = {}
    for dataset in ["purchase100", "cifar10"]:
        best = None
        best_acc = -1.0
        scores = []
        for coeff in candidates:
            run_subprocess(
                ["--dataset", dataset, "--method", "relaxloss", "--seed", "11", "--relaxloss-lambda", str(coeff)],
                stage="pilot",
            )
            row = json.loads((OUTPUT_ROOT / "metrics" / f"{dataset}_relaxloss_11.json").read_text())
            val_acc = row["metrics"]["best_val_accuracy"]
            val_loss = row["metrics"]["best_val_loss"]
            scores.append(
                {
                    "lambda": coeff,
                    "best_val_accuracy": val_acc,
                    "best_val_loss": val_loss,
                    "seed": 11,
                }
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best = coeff
            elif abs(val_acc - best_acc) < 1e-8 and best is not None:
                prev = next(item for item in scores if item["lambda"] == best)
                if val_loss < prev["best_val_loss"]:
                    best = coeff
        tuned[dataset] = {"selected_lambda": best, "candidates": scores}
    (OUTPUT_ROOT / "tables" / "relaxloss_tuning.json").write_text(json.dumps(tuned, indent=2))
    return tuned


def main(mode: str):
    ensure_output_dirs()
    tuned = tune_relaxloss()
    if mode in {"pilot", "full"}:
        for dataset, method, seed in PILOTS:
            extra = []
            if method == "relaxloss":
                extra = ["--relaxloss-lambda", str(tuned[dataset]["selected_lambda"])]
            run_subprocess(["--dataset", dataset, "--method", method, "--seed", str(seed), *extra], stage="pilot")
    if mode == "pilot":
        return
    for dataset in ["purchase100", "cifar10"]:
        for method in MAIN_METHODS:
            for seed in SEEDS:
                if (dataset, method, seed) in PILOTS:
                    continue
                extra = []
                if method == "relaxloss":
                    extra = ["--relaxloss-lambda", str(tuned[dataset]["selected_lambda"])]
                run_subprocess(["--dataset", dataset, "--method", method, "--seed", str(seed), *extra], stage="full")
        for method in ABLATIONS:
            for seed in SEEDS:
                run_subprocess(["--dataset", dataset, "--method", method, "--seed", str(seed)], stage="full")
    aggregate()
    pd.read_csv(OUTPUT_ROOT / "metrics" / "main_runs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    args = parser.parse_args()
    main(args.mode)
