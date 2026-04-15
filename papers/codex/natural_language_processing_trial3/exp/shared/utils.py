from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def now() -> float:
    return time.time()


def runtime_minutes(start: float) -> float:
    return round((time.time() - start) / 60.0, 4)


def safe_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def capture_environment(out_dir: Path) -> None:
    ensure_dir(out_dir)
    freeze_path = out_dir / "pip_freeze.txt"
    try:
        proc = subprocess.run(
            ["python3", "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        freeze_path.write_text(proc.stdout, encoding="utf-8")
    except Exception as exc:
        freeze_path.write_text(f"FAILED: {exc}\n", encoding="utf-8")

    env = {
        "python": os.popen("python3 -V").read().strip(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "git_commit": safe_git_commit(),
        "nproc": os.popen("nproc").read().strip(),
        "free_h": os.popen("free -h").read().strip(),
        "nvidia_smi": os.popen("nvidia-smi").read().strip() if torch.cuda.is_available() else "",
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            env["gpu_properties"] = {
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "multi_processor_count": torch.cuda.get_device_properties(0).multi_processor_count,
            }
        except Exception:
            pass
    write_json(out_dir / "environment.json", env)


def gpu_memory_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024**3), 4),
    }


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(a))
    diffs = []
    for _ in range(n_resamples):
        sample = rng.choice(indices, size=len(indices), replace=True)
        diffs.append(float(a[sample].mean() - b[sample].mean()))
    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }
