import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "exp"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"

MODEL_NAME = "ViT-B-32"
PRETRAINED_NAME = "laion2b_s34b_b79k"
IMAGE_SIZE = 224
BATCH_SIZE = 512
NUM_WORKERS = 4
SEEDS = [7, 17, 27]
CORRUPTION_FAMILIES = ["gaussian_noise", "motion_blur", "fog", "jpeg_compression"]
SEVERITIES = [1, 2, 3, 4, 5]


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: os.PathLike[str] | str, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def append_jsonl(path: os.PathLike[str] | str, payload: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def now() -> float:
    return time.perf_counter()


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3)


def tensor_to_cpu_fp32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().float()


@dataclass
class RunArtifacts:
    experiment_dir: Path
    logs_dir: Path
    results_path: Path
    config_path: Path


def experiment_artifacts(name: str) -> RunArtifacts:
    experiment_dir = ensure_dir(EXP_DIR / name)
    logs_dir = ensure_dir(experiment_dir / "logs")
    return RunArtifacts(
        experiment_dir=experiment_dir,
        logs_dir=logs_dir,
        results_path=experiment_dir / "results.json",
        config_path=experiment_dir / "config.json",
    )


def write_skipped(experiment_name: str, reason: str) -> None:
    artifacts = experiment_artifacts(experiment_name)
    with (artifacts.experiment_dir / "SKIPPED.md").open("w") as handle:
        handle.write(reason.rstrip() + "\n")


def save_system_info() -> None:
    ensure_dir(RESULTS_DIR)
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_name": MODEL_NAME,
        "pretrained_name": PRETRAINED_NAME,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "seeds": SEEDS,
        "corruption_families": CORRUPTION_FAMILIES,
        "severities": SEVERITIES,
    }
    save_json(RESULTS_DIR / "system_info.json", payload)


class StageLogger:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.artifacts = experiment_artifacts(experiment_name)
        self.stdout_path = self.artifacts.logs_dir / "run_stdout.txt"
        self.stderr_path = self.artifacts.logs_dir / "run_stderr.txt"
        self.start_time = time.perf_counter()
        ensure_dir(self.stdout_path.parent)
        self._stdout_handle = self.stdout_path.open("w")
        self._stderr_handle = self.stderr_path.open("w")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self._stdout_handle.write(line + "\n")
        self._stdout_handle.flush()

    def log_command(self, command: str, cwd: os.PathLike[str] | str | None = None) -> dict[str, Any]:
        cwd = str(cwd or ROOT)
        self.log(f"COMMAND cwd={cwd} cmd={command}")
        started = time.perf_counter()
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        self.log(f"COMMAND_DONE exit_code={proc.returncode} elapsed_seconds={elapsed:.3f}")
        if proc.stdout.strip():
            self._stdout_handle.write(proc.stdout.rstrip() + "\n")
            self._stdout_handle.flush()
        if proc.stderr.strip():
            self._stderr_handle.write(proc.stderr.rstrip() + "\n")
            self._stderr_handle.flush()
        return {
            "command": command,
            "cwd": cwd,
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    def close(self) -> None:
        elapsed = time.perf_counter() - self.start_time
        self.log(f"STAGE_DONE experiment={self.experiment_name} elapsed_seconds={elapsed:.3f}")
        self._stdout_handle.close()
        self._stderr_handle.close()


def stage_logger(experiment_name: str) -> StageLogger:
    return StageLogger(experiment_name)
