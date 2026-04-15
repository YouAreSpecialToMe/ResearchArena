import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize_for_json(payload), handle, indent=2, sort_keys=True, allow_nan=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, np.generic):
        return _sanitize_for_json(value.item())
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def append_csv_row(path, fieldnames, row):
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self):
        self.start = time.time()

    def minutes(self) -> float:
        return (time.time() - self.start) / 60.0
