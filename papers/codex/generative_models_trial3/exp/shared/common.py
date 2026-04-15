import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "figures"
EXP_DIR = ROOT / "exp"

PYTHON_BIN = os.environ.get("PARADG_PYTHON_BIN", "python")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DETECTOR_MODEL_ID = "google/owlvit-base-patch32"
DINO_MODEL_ID = "facebook/dinov2-base"
IMAGE_REWARD_MODEL_ID = "ImageReward-v1.0"

BENCHMARK_REPO = "https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/main"
BENCHMARK_FILES = {
    "attribute_binding": f"{BENCHMARK_REPO}/examples/dataset/color_val.txt",
    "relations": f"{BENCHMARK_REPO}/examples/dataset/spatial_val.txt",
    "numeracy": f"{BENCHMARK_REPO}/examples/dataset/numeracy_val.txt",
}

SEEDS = [11, 22, 33]
PILOT_SEED = 11
IMAGE_SIZE = 512
NUM_STEPS = 30
CFG_SCALE = 7.5
GUIDED_STEPS = [2, 4, 6, 9, 12, 16, 20, 24]
REDUCED_GUIDED_STEPS = [4, 8, 14, 20]
DEFAULT_SCHEDULE = [0.2, 0.3, 0.5, 0.6, 0.6, 0.5, 0.3, 0.2]
FLAT_SCHEDULE = [0.4] * len(GUIDED_STEPS)
EARLY_HEAVY = [0.6, 0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.2]
MIDDLE_HEAVY = DEFAULT_SCHEDULE
GATE_C_VALUES = [0.5, 1.0, 1.5]

MAIN_HELD_OUT_LIMIT = 48
ROBUSTNESS_LIMIT = 18
NON_EQ_LIMIT = 12
PILOT_PER_CATEGORY = 4
HELD_OUT_PER_CATEGORY = 20


@dataclass
class PromptRecord:
    prompt_id: str
    source_dataset: str
    source_file: str
    category: str
    split: str
    overlap_subset_flag: bool
    original_prompt: str
    object_1: str
    count_1: str
    attribute_1: str
    relation: str
    object_2: str
    count_2: str
    attribute_2: str
    approved_paraphrases: list[str]
    rewrite_types: list[str]
    preserved_slots: dict[str, str]
    checklist_result: dict[str, bool]
    audit_status: str
    confounder_prompt: str
    non_equivalent_aux_prompt: str
    candidate_paraphrases: list[str] = field(default_factory=list)
    candidate_rewrite_types: list[str] = field(default_factory=list)
    candidate_audit: list[dict[str, Any]] = field(default_factory=list)


def ensure_dirs() -> None:
    for path in [DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, EXP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def latent_hash(seed: int, prompt_id: str) -> str:
    return stable_hash({"seed": seed, "prompt_id": prompt_id})[:16]


def run_command(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _safe_run_command(cmd: list[str]) -> str | None:
    try:
        return run_command(cmd)
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def package_versions() -> dict[str, str | None]:
    packages = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "xformers",
        "open-clip-torch",
        "lpips",
        "image-reward",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "Pillow",
    ]
    return {name: _pkg_version(name) for name in packages}


def environment_manifest() -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": run_command(["which", PYTHON_BIN]) if "/" not in PYTHON_BIN else PYTHON_BIN,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_driver_version": _safe_run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpu_memory_total_mb": _safe_run_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]),
        "cpu_count": os.cpu_count(),
        "package_versions": package_versions(),
        "seeds": SEEDS,
        "generation_defaults": {
            "model_id": MODEL_ID,
            "clip_model_id": CLIP_MODEL_ID,
            "detector_model_id": DETECTOR_MODEL_ID,
            "dino_model_id": DINO_MODEL_ID,
            "image_reward_model_id": IMAGE_REWARD_MODEL_ID,
            "scheduler": "DDIM",
            "num_steps": NUM_STEPS,
            "cfg_scale": CFG_SCALE,
            "image_size": IMAGE_SIZE,
            "guided_steps": GUIDED_STEPS,
            "negative_prompt": "",
            "attention_backend": "attention_slicing",
        },
    }


def records_from_jsonl(path: Path) -> list[PromptRecord]:
    return [PromptRecord(**row) for row in read_jsonl(path)]


def timer() -> float:
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    return time.perf_counter() - start


def prompt_slot_map(record: PromptRecord) -> dict[str, str]:
    return {
        "object_1": record.object_1,
        "count_1": record.count_1,
        "attribute_1": record.attribute_1,
        "relation": record.relation,
        "object_2": record.object_2,
        "count_2": record.count_2,
        "attribute_2": record.attribute_2,
    }


def record_to_dict(record: PromptRecord) -> dict[str, Any]:
    return asdict(record)
