from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "figures"

CORE_SEEDS = [11, 22, 33]
AUX_SEEDS = [101, 202, 303]
P_VALUES = [10, 15]
GRAPH_FAMILIES = ["erdos_renyi", "scale_free"]
WEIGHT_REGIMES = ["weak", "mixed"]
BATCH_SIZES = [25, 50, 100]
TOTAL_BUDGET = 300
OBS_SAMPLES = 200
BOOTSTRAPS = 24
PARTICLE_LIMIT = 24
PARTICLE_EXTENSIONS = 3
CHECKPOINTS = [0, 50, 100, 150, 200, 250, 300]
CALIBRATION_CONTINUATIONS = 40


def set_thread_env() -> None:
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    raise TypeError(f"Unsupported JSON type: {type(obj)!r}")


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=json_default))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def now() -> float:
    return time.perf_counter()


def summarize_mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}

