from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "exp"
TRACE_ROOT = ROOT / "traces"
CALIBRATION_ROOT = ROOT / "calibration"
REPLAY_ROOT = ROOT / "replay_results"
LIVE_ROOT = ROOT / "live_validation"
FIGURE_ROOT = ROOT / "figures"

SEEDS = [11, 23, 37]
PILOT_SEED = 5
ACCESS_PAGE_SIZE = 4096
WINDOW = 4096
EPOCH_REFS_PER_TENANT = 2000
REGRET_WINDOW = 2048
MAX_CPU_WORKERS = 2
ACTION_GRID_QUANTA = 10
POLICY_MENU = ["LRU", "SCAN", "FREQ"]
SRV_ALPHA = 0.875
ETA_DEFAULT = 0.5
SHARED_FRACTION_THRESHOLD = 0.05
SRV_CONFIDENCE_ENTROPY_THRESHOLD = 0.95
SRV_CONFIDENCE_MIN_REUSES = 2.0
CPU_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def pinned_cpu_list(max_cores: int = MAX_CPU_WORKERS) -> list[int]:
    try:
        available = sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(max_cores))
    return available[:max_cores] if available else []


def pin_process_affinity(max_cores: int = MAX_CPU_WORKERS) -> list[int]:
    cpus = pinned_cpu_list(max_cores=max_cores)
    if cpus:
        try:
            os.sched_setaffinity(0, set(cpus))
        except (AttributeError, OSError):
            pass
    return cpus


def set_reproducible(seed: int) -> random.Random:
    for key, value in CPU_ENV.items():
        os.environ[key] = value
    pin_process_affinity()
    random.seed(seed)
    np.random.seed(seed)
    return random.Random(seed)


def ensure_layout() -> None:
    for path in [TRACE_ROOT, CALIBRATION_ROOT, REPLAY_ROOT, LIVE_ROOT, FIGURE_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [
        EXP_ROOT / "shared",
        EXP_ROOT / "smoke",
        EXP_ROOT / "primary",
        EXP_ROOT / "ablations",
        EXP_ROOT / "oracle",
        EXP_ROOT / "live_validation",
        EXP_ROOT / "external_validation",
        EXP_ROOT / "analysis",
    ]:
        (path / "logs").mkdir(parents=True, exist_ok=True)
        (path / "runs").mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def jain_index(values: list[float]) -> float:
    if not values:
        return 0.0
    num = sum(values) ** 2
    den = len(values) * sum(v * v for v in values)
    return float(num / den) if den else 0.0


def bootstrap_ci(deltas: list[float], n_resamples: int = 10_000, seed: int = 17) -> dict[str, float]:
    if not deltas:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    arr = np.array(deltas, dtype=float)
    samples = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    return {
        "low": float(np.quantile(samples, 0.025)),
        "mid": float(np.mean(samples)),
        "high": float(np.quantile(samples, 0.975)),
    }


def wilcoxon_signed_rank(a: list[float], b: list[float]) -> float:
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(a, b, zero_method="pratt").pvalue)
    except Exception:
        return math.nan


def peak_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


@dataclass
class RunSpec:
    experiment: str
    workload_family: str
    cache_budget: str
    method: str
    seed: int
    tenant_count: int
    trace_path: str
    budget_pages: int
    miss_cost_mode: str = "FamilyConst"
    debt_half_life_turnovers: float = 1.0
    reduction_enabled: bool = True
    srv_mode: str = "decayed"
    oracle: bool = False
    input_artifacts: list[str] = field(default_factory=list)

    def run_dir(self) -> Path:
        return EXP_ROOT / self.experiment / "runs" / (
            f"{self.workload_family}__{self.cache_budget}__{self.method}__seed{self.seed}"
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self.started
