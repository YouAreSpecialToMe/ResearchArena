from __future__ import annotations

import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ROOT / "data" / "RAGTruth" / "dataset"

SEEDS = [13, 21, 42]
TAU_GRID = [0.55, 0.60, 0.65, 0.70]
LOGREG_C_GRID = [0.1, 1.0, 10.0]


def ensure_dirs() -> None:
    for rel in [
        "artifacts/cache",
        "artifacts/features",
        "artifacts/predictions",
        "artifacts/tables",
        "artifacts/figures",
        "artifacts/annotations",
        "artifacts/logs",
        "figures",
        "data/processed",
        "exp/bm25/logs",
        "exp/support_only/logs",
        "exp/support_compactness/logs",
        "exp/full_context_removal/logs",
        "exp/localized_only/logs",
        "exp/training_free_additive/logs",
        "exp/full_detector/logs",
        "exp/ablation_no_localized_perturbation/logs",
        "exp/ablation_remove_only/logs",
        "exp/ablation_drop_one_only/logs",
        "exp/ablation_swap_only/logs",
        "exp/ablation_fixed_topk_support/logs",
        "exp/ablation_no_redundancy/logs",
        "exp/data_prep/logs",
        "exp/pilot/logs",
        "exp/analysis_threshold_stability/logs",
        "exp/audit_uncertainty/logs",
        "exp/visualization/logs",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def jsonl_write(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a") as f:
        f.write(f"[{stamp}] {message}\n")


def exp_log_path(name: str) -> Path:
    return ROOT / "exp" / name / "logs" / "run.log"


def now() -> float:
    return time.perf_counter()


def runtime_minutes(start: float) -> float:
    return (time.perf_counter() - start) / 60.0


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return ""


def build_manifest() -> dict[str, Any]:
    import matplotlib
    import pandas
    import pyarrow
    import scipy
    import seaborn
    import sklearn
    import spacy
    import statsmodels
    import transformers
    import tqdm

    manifest = {
        "python": command_output(["python", "--version"]),
        "python_executable": command_output(["which", "python"]),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
        "spacy": spacy.__version__,
        "pandas": pandas.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "statsmodels": statsmodels.__version__,
        "tqdm": tqdm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "cpu_count": os.cpu_count(),
        "platform_uname": command_output(["uname", "-a"]),
        "git_status": command_output(["git", "status", "--short"]) if (ROOT / ".git").exists() else "",
    }
    return manifest
