from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
FIGURES_DIR = ROOT / "figures"


SEEDS = [11, 17, 23]
EPOCH_LENGTH = 100_000
TRACE_LENGTH = 1_800_000
PAGE_SIZE = 4096
SHADOW_RATE = 0.01
DEFAULT_SENTINEL_FRACTION = 0.02
DEFAULT_SENTINEL_PLACEMENT = "uniform-hash"
DEFAULT_HYSTERESIS = 0.02
DEFAULT_MIN_DWELL = 2
RECENT_WINDOW_EPOCHS = 2
RESIDUAL_HISTORY_EPOCHS = 3
LFU_AGING_DECAY_EPOCHS = 4
LFU_AGING_DECAY_FACTOR = 0.5
LFU_COUNTER_BITS = 8
LHD_SIZE_BINS = 8
LHD_AGE_BUCKETS = 16
ACTIVE_METHODS = ["RecentWindow", "LeaderSetDuel", "NoCalibration", "DuelCache"]
PRIMARY_METHODS = ["LRU", "ARC", "RecentWindow", "LeaderSetDuel", "NoCalibration", "DuelCache"]
ABLATION_METHODS = ["NoCalibration", "NoSentinelScaling", "DirectSentinelOnly"]
EXPERTS = ["LRU", "MRU", "LFU-aging", "LHD"]
WORKLOADS = ["PhaseLoop", "TwoTenantMix", "SkewShift", "StationaryZipf"]
CACHE_RATIOS = [0.5, 0.8]


def controller_hyperparameters(
    *,
    switch_penalty: float,
    sentinel_fraction: float = DEFAULT_SENTINEL_FRACTION,
    sentinel_placement: str = DEFAULT_SENTINEL_PLACEMENT,
    hysteresis: float = DEFAULT_HYSTERESIS,
    min_dwell: int = DEFAULT_MIN_DWELL,
    shadow_rate: float = SHADOW_RATE,
) -> dict:
    return {
        "policy_portfolio": list(EXPERTS),
        "page_size_bytes": PAGE_SIZE,
        "epoch_length_refs": EPOCH_LENGTH,
        "shadow_rate": shadow_rate,
        "sentinel_fraction": sentinel_fraction,
        "sentinel_placement": sentinel_placement,
        "recent_window_epochs": RECENT_WINDOW_EPOCHS,
        "residual_history_epochs": RESIDUAL_HISTORY_EPOCHS,
        "hysteresis_margin": hysteresis,
        "minimum_dwell_epochs": min_dwell,
        "switch_penalty": switch_penalty,
        "lfu_aging": {
            "decay_epochs": LFU_AGING_DECAY_EPOCHS,
            "decay_factor": LFU_AGING_DECAY_FACTOR,
            "counter_bits": LFU_COUNTER_BITS,
        },
        "lhd": {
            "size_bins": LHD_SIZE_BINS,
            "age_buckets": LHD_AGE_BUCKETS,
            "page_size_constant_bytes": PAGE_SIZE,
        },
    }


def ensure_dirs() -> None:
    for path in [ARTIFACTS, RUNS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                reset_dir(child)
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def mean_std(values: Iterable[float]) -> dict[str, float]:
    vals = list(map(float, values))
    if not vals:
        return {"mean": math.nan, "std": math.nan}
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
    }


def bootstrap_ci(deltas: list[float], rng_seed: int = 123, n_resamples: int = 10_000) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    data = np.asarray(deltas, dtype=float)
    if data.size == 0:
        return {"low": math.nan, "high": math.nan}
    idx = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def bootstrap_mean_std_ci(values: Iterable[float], rng_seed: int = 123, n_resamples: int = 10_000) -> dict[str, float]:
    vals = np.asarray(list(map(float, values)), dtype=float)
    if vals.size == 0:
        return {"mean": math.nan, "std": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
    ci = bootstrap_ci(vals.tolist(), rng_seed=rng_seed, n_resamples=n_resamples)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "ci95_low": ci["low"],
        "ci95_high": ci["high"],
    }


def wilcoxon_effect(x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) != len(y) or not x:
        return {"pvalue": None, "effect_r": None}
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.allclose(diffs, 0):
        return {"pvalue": 1.0, "effect_r": 0.0}
    stat = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="approx")
    z = stats.norm.isf(stat.pvalue / 2.0) if stat.pvalue > 0 else np.inf
    effect_r = float(z / math.sqrt(len(x)))
    return {"pvalue": float(stat.pvalue), "effect_r": effect_r}


def kendall_tau(scores_a: list[float], scores_b: list[float]) -> float:
    value = stats.kendalltau(scores_a, scores_b, variant="b").correlation
    return float(0.0 if np.isnan(value) else value)


def spearman_rho(scores_a: list[float], scores_b: list[float]) -> float:
    value = stats.spearmanr(scores_a, scores_b).statistic
    return float(0.0 if np.isnan(value) else value)


def mape(pred: list[float], truth: list[float]) -> float:
    arr_p = np.asarray(pred, dtype=float)
    arr_t = np.asarray(truth, dtype=float)
    denom = np.maximum(np.abs(arr_t), 1e-9)
    return float(np.mean(np.abs(arr_p - arr_t) / denom))


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def env_metadata() -> dict[str, str]:
    import platform
    import subprocess
    import sys

    def cmd(text: str) -> str:
        return subprocess.check_output(text, shell=True, cwd=ROOT, text=True).strip()

    return {
        "cwd": str(ROOT),
        "uname": cmd("uname -a"),
        "kernel_version": cmd("uname -r"),
        "compiler_version": cmd("(cc --version || gcc --version || clang --version) 2>/dev/null | head -n 1"),
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": cmd("git rev-parse HEAD 2>/dev/null || echo NO_GIT"),
        "nproc": cmd("nproc"),
        "free_h": cmd("free -h"),
    }


def stable_hash(text: str) -> int:
    value = 2166136261
    for char in text.encode("utf-8"):
        value ^= char
        value *= 16777619
        value &= 0xFFFFFFFF
    return value


def make_rng(*parts: object) -> random.Random:
    seed = 0
    for part in parts:
        seed ^= stable_hash(str(part))
    return random.Random(seed)


def hashed_fraction(*parts: object) -> float:
    value = stable_hash("::".join(str(part) for part in parts))
    return value / 0xFFFFFFFF


def run_stem(*parts: object) -> str:
    pieces = [str(part).replace("/", "_").replace(" ", "_") for part in parts]
    return "__".join(pieces)
