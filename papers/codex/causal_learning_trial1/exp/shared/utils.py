from __future__ import annotations

import hashlib
import os
import random
import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import THREAD_ENV


def configure_environment() -> None:
    for key, value in THREAD_ENV.items():
        os.environ[key] = value


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_int_seed(*parts: object, modulus: int = 2**32) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, "big") % modulus


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


@dataclass
class RuntimeStats:
    runtime_seconds: float
    peak_memory_mb: float


@contextmanager
def timed_block() -> Iterator[RuntimeStats]:
    start = time.perf_counter()
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = RuntimeStats(0.0, 0.0)
    try:
        yield stats
    finally:
        end = time.perf_counter()
        end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1024.0 if os.uname().sysname != "Darwin" else 1024.0 * 1024.0
        stats.runtime_seconds = end - start
        stats.peak_memory_mb = max(start_mem, end_mem) / scale


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }
