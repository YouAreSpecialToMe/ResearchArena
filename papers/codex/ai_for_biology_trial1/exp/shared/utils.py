from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return [_json_safe(v) for v in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        val = float(value)
        return None if not math.isfinite(val) else val
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=False, allow_nan=False)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=False, allow_nan=False) + "\n")


def slugify(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def capture_environment_metadata() -> dict[str, Any]:
    package_names = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "anndata",
        "scanpy",
        "pyarrow",
        "matplotlib",
        "seaborn",
        "joblib",
        "torch",
    ]
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cuda_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024**2), 2),
                }
            )

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": cuda_devices,
        },
        "deterministic_settings": {
            "python_random_seeded": True,
            "numpy_seeded": True,
            "torch_manual_seed": True,
            "torch_cuda_manual_seed_all": True,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "residual_nondeterminism_notes": [
            "CUDA kernels may still show minor floating-point variation across devices or driver stacks.",
            "BLAS and multithreaded reductions can change least-significant bits despite fixed seeds.",
            "Peak memory is process RSS on CPU; GPU memory is reported only when explicitly captured by a caller.",
        ],
        "runtime_accounting": {
            "runtime_minutes_fields_include_preprocess": True,
            "peak_memory_field": "peak_memory_mb",
            "peak_memory_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        },
    }


def capture_config_snapshot(config_module: Any) -> dict[str, Any]:
    snapshot = {}
    for key in dir(config_module):
        if not key.isupper():
            continue
        snapshot[key] = _json_safe(getattr(config_module, key))
    return snapshot


@dataclass
class Timer:
    started: float = 0.0

    def __enter__(self) -> "Timer":
        self.started = time.time()
        return self

    def __exit__(self, *_: object) -> None:
        return None

    @property
    def minutes(self) -> float:
        return (time.time() - self.started) / 60.0
