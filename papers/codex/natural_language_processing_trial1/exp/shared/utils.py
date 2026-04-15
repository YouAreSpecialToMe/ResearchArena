import json
import os
import random
import subprocess
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"

SEEDS = [11, 23, 47]
TRAINING_SEED = 11
SPLIT_SEED = 101
PLANNING_SEED_BANK = [301, 302, 303, 304, 305]
AUDIT_SEED = 2027
BOOTSTRAP_SEED = 2028

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LEN = 512
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
PROOFWRITER_PROOF_DEPTH = 1
PLANNING_HORIZON = 8


def ensure_dirs():
    for path in [DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, LOGS_DIR, ROOT / "exp"]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def git_commit():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return None


def gpu_info():
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "total_vram_gb": round(props.total_memory / (1024**3), 2),
    }


def now_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def detect_cuda_version():
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True)
        for line in out.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].split()[0]
    except Exception:
        return None
    return None


def package_versions():
    versions = {}
    for module_name in [
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "datasets",
        "evaluate",
        "scipy",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "sentencepiece",
    ]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"missing: {type(exc).__name__}: {exc}"
    return versions


def system_info():
    info = {
        "timestamp": now_ts(),
        "cwd": str(ROOT),
        "python_version": os.popen("python --version").read().strip() or None,
        "cpu_cores": os.cpu_count(),
        "cuda_version": detect_cuda_version(),
        "gpu": gpu_info(),
        "git_commit": git_commit(),
        "packages": package_versions(),
    }
    try:
        info["ram"] = os.popen("free -h").read().strip()
    except Exception:
        info["ram"] = None
    return info


def run_config(condition=None, seed=None, extra=None):
    config = {
        "model_name": MODEL_NAME,
        "seeds": SEEDS,
        "split_seed": SPLIT_SEED,
        "planning_seed_bank": PLANNING_SEED_BANK,
        "audit_seed": AUDIT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "training": {
            "max_seq_len": MAX_SEQ_LEN,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "quantization": "4bit-nf4",
            "compute_dtype": "bf16",
        },
        "validator": {
            "proof_depth": PROOFWRITER_PROOF_DEPTH,
            "planning_horizon": PLANNING_HORIZON,
            "planning_solver": "bounded-bfs",
        },
    }
    if condition is not None:
        config["condition"] = condition
    if seed is not None:
        config["seed"] = seed
    if extra:
        config.update(extra)
    return config


def save_run_config(path, condition=None, seed=None, extra=None):
    config = run_config(condition=condition, seed=seed, extra=extra)
    dump_json(config, path)
    return config


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return {"center": 0.0, "lower": 0.0, "upper": 0.0}
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = (
        z
        * ((phat * (1 - phat) / total + z**2 / (4 * total**2)) ** 0.5)
        / denom
    )
    return {"center": center, "lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def bootstrap_ci(values, n_boot=2000, seed=BOOTSTRAP_SEED):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    samples = []
    n = arr.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(float(arr[idx].mean()))
    samples = np.asarray(samples)
    return {
        "mean": float(arr.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }


def paired_bootstrap_delta(a_values, b_values, n_boot=2000, seed=BOOTSTRAP_SEED):
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if a.size != b.size:
        raise ValueError("Bootstrap arrays must have equal length")
    if a.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = np.random.default_rng(seed)
    deltas = a - b
    samples = []
    n = deltas.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(float(deltas[idx].mean()))
    samples = np.asarray(samples)
    return {
        "mean": float(deltas.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }
