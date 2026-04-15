import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path, payload):
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def run_cmd(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()


def collect_environment() -> dict:
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": str(ROOT),
        "timestamp": time.time(),
    }
    for key, cmd in {
        "nvidia_smi_summary": "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
        "nvidia_smi_full": "nvidia-smi",
        "cpu_cores": "nproc",
        "memory": "free -h",
    }.items():
        try:
            env[key] = run_cmd(cmd)
        except Exception as exc:
            env[key] = f"ERROR: {exc}"
    env["package_summary"] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda": getattr(torch.version, "cuda", None),
    }
    try:
        env["pip_freeze"] = run_cmd(f"{sys.executable} -m pip freeze")
    except Exception as exc:
        env["pip_freeze"] = f"ERROR: {exc}"
    try:
        env["git_commit"] = run_cmd("git rev-parse HEAD")
    except Exception:
        env["git_commit"] = None
    env["cuda_available"] = torch.cuda.is_available()
    env["cuda_version"] = torch.version.cuda
    env["torch_version"] = torch.__version__
    return env


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end = time.time()
        self.seconds = self.end - self.start
