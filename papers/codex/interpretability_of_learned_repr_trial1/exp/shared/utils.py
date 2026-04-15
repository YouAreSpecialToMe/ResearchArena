from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
EXP = ROOT / "exp"

SEEDS = [11, 22, 33]
PRIMARY_METHODS = ["sae", "ssae", "asd"]
ABLATION_METHODS = ["asd_no_tie", "asd_no_share"]
ALL_METHODS = PRIMARY_METHODS + ABLATION_METHODS
GAMMAS = [0.5, 1.0, 1.5]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_workspace() -> None:
    for rel in [
        "cache",
        "pairs",
        "models",
        "metrics",
        "plots",
        "tables",
        "logs",
    ]:
        ensure_dir(OUTPUTS / rel)
    ensure_dir(DATA)
    ensure_dir(ROOT / "figures")


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def now() -> float:
    return time.time()


def elapsed_minutes(start: float) -> float:
    return (time.time() - start) / 60.0


class CSVLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        ensure_dir(path.parent)
        self.path = path
        self.fieldnames = fieldnames
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def torch_save(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def append_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def get_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def maybe_int(x: Any) -> Any:
    try:
        xi = int(x)
        if xi == x:
            return xi
    except Exception:
        return x
    return x
