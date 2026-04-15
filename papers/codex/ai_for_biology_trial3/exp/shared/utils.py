import json
import math
import os
import random
import resource
import time
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


SEEDS = [42, 43, 44]
ACCEPTANCE_RATES = [0.2, 0.4, 0.6]
SELECTIVE_GRID = [i / 10 for i in range(1, 11)]
RUN_VERSION = "stage2_retry1_v3"


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timed_block() -> float:
    start = time.time()
    slot = {"start": start}
    yield slot
    slot["end"] = time.time()
    slot["seconds"] = slot["end"] - start


def summarize(values: list[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def infer_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def hash_vector(tokens: list[str], dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        slot = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"slot-v1").digest(),
            "little",
        ) % dim
        sign_hash = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"sign-v1").digest(),
            "little",
        )
        sign = 1.0 if (sign_hash % 2 == 0) else -1.0
        vec[slot] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def flatten_metrics(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_metrics(name, value))
        else:
            out[name] = value
    return out


def get_system_info() -> Dict[str, Any]:
    ram_bytes = None
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    gpu_name = None
    gpu_total_bytes = 0
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total_bytes = int(props.total_memory)
    return {
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "gpu_name": gpu_name,
        "gpu_total_bytes": gpu_total_bytes,
        "device": infer_device(),
    }


def get_process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(peak)
    return int(peak * 1024)
