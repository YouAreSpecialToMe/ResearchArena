from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preferred_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Unsupported type: {type(obj)!r}")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=json_default))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def append_registry_row(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_command(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def collect_environment_snapshot() -> dict[str, Any]:
    packages = [
        "torch",
        "torchvision",
        "timm",
        "transformers",
        "open_clip_torch",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "pillow",
        "tqdm",
        "h5py",
    ]
    snapshot = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    dist_names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "transformers": "transformers",
        "open_clip_torch": "open-clip-torch",
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "scikit-learn",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pillow": "pillow",
        "tqdm": "tqdm",
        "h5py": "h5py",
    }
    for package in packages:
        try:
            version = importlib.metadata.version(dist_names[package])
        except Exception:
            version = "missing"
        snapshot[f"pkg_{package}"] = version
    return snapshot


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *_: Any) -> None:
        self.end = time.time()
        self.seconds = self.end - self.start
