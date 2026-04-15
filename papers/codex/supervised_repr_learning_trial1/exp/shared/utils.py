import json
import math
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def now() -> float:
    return time.perf_counter()


def elapsed_minutes(start_time: float) -> float:
    return (time.perf_counter() - start_time) / 60.0


def to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, sort_keys=True)


def append_csv_row(path: str | Path, row: Dict[str, Any]) -> None:
    import csv

    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def max_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    return ensure_dir(repo_root() / "data")


def analysis_root() -> Path:
    return ensure_dir(repo_root() / "analysis")
