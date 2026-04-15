import argparse
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEEDS = [11, 22, 33]
DATASETS = ["waterbirds", "cub"]
MAIN_METHODS = [
    "linear_probe",
    "ce_adapter",
    "contrastive_adapter",
    "fixed_k_contrastive",
    "fixed_k_noncontrastive",
    "pb_spread",
    "adaptive_vmf",
]
ABLATIONS = ["ablation_no_adapt", "ablation_no_vmf", "ablation_no_occ"]
SENSITIVITY = ["adaptive_margin_0025", "adaptive_margin_0050"]


def task_list(stage):
    tasks = []
    if stage in {"main", "all"}:
        for dataset in DATASETS:
            for method in MAIN_METHODS:
                for seed in SEEDS:
                    tasks.append((dataset, method, seed))
    if stage in {"ablation", "all"}:
        for dataset in DATASETS:
            for method in ABLATIONS:
                for seed in SEEDS:
                    tasks.append((dataset, method, seed))
        for seed in SEEDS:
            for method in SENSITIVITY:
                tasks.append(("waterbirds", method, seed))
    return tasks


def run_one(task):
    dataset, method, seed = task
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, "exp/train/run.py", "--dataset", dataset, "--method", method, "--seed", str(seed)]
    completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return task, completed.returncode, completed.stdout[-4000:], completed.stderr[-4000:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["main", "ablation", "all"], default="all")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    tasks = task_list(args.stage)
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                dataset, method, seed = task
                _, code, stdout, stderr = future.result()
                print(f"[done] {dataset} {method} seed={seed} code={code}")
                if code != 0:
                    failures.append({"task": task, "stdout": stdout, "stderr": stderr})
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
