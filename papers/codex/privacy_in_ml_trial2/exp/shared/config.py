from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
FIGURES_DIR = ROOT / "figures"
CACHE_VERSION = "stage2_protocol_fix_v1"

SEEDS = [7, 17, 27, 37]
ORDERING_SEEDS = list(range(20))
DATASETS = ["fashion_mnist", "cifar10"]
EPSILON_TARGETS = [4.0, 8.0]
DELTA = 1e-5
AUDIT_POOL_SIZE = 240
CANARYS_PER_KIND = 120
QUERY_BUDGETS = list(range(40, 1201, 40))
LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
PRIMARY_K = 4
ABLATION_KS = [2, 4]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    num_classes: int
    private_train_size: int
    calibration_size: int
    proxy_size: int
    screening_size: int
    reserve_size: int
    fgsm_epsilon: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    momentum: float
    max_grad_norm: float


DATASET_CONFIGS = {
    "fashion_mnist": DatasetConfig(
        name="fashion_mnist",
        num_classes=10,
        private_train_size=42000,
        calibration_size=6000,
        proxy_size=4000,
        screening_size=4000,
        reserve_size=4000,
        fgsm_epsilon=0.10,
        batch_size=512,
        epochs=10,
        lr=0.55,
        weight_decay=1e-4,
        momentum=0.9,
        max_grad_norm=1.0,
    ),
    "cifar10": DatasetConfig(
        name="cifar10",
        num_classes=10,
        private_train_size=32000,
        calibration_size=6000,
        proxy_size=4000,
        screening_size=4000,
        reserve_size=4000,
        fgsm_epsilon=2.0 / 255.0,
        batch_size=1024,
        epochs=8,
        lr=0.35,
        weight_decay=5e-4,
        momentum=0.9,
        max_grad_norm=1.0,
    ),
}


def combo_name(dataset: str, epsilon: float, seed: int) -> str:
    return f"{dataset}_eps{int(epsilon)}_seed{seed}"


def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        ARTIFACTS_DIR / "environment",
        ARTIFACTS_DIR / "checkpoints",
        ARTIFACTS_DIR / "results",
        ARTIFACTS_DIR / "results" / "audit_pairs",
        ARTIFACTS_DIR / "results" / "evaluations",
        ARTIFACTS_DIR / "results" / "lambda_selection",
        ARTIFACTS_DIR / "results" / "noise_multipliers",
        ARTIFACTS_DIR / "results" / "runtime_guardrails",
        ARTIFACTS_DIR / "results" / "screening",
        ARTIFACTS_DIR / "results" / "splits",
        ARTIFACTS_DIR / "scores",
        ARTIFACTS_DIR / "figures" / "source_data",
        FIGURES_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def config_dict() -> dict:
    return {
        "seeds": SEEDS,
        "ordering_seeds": ORDERING_SEEDS,
        "cache_version": CACHE_VERSION,
        "datasets": {k: asdict(v) for k, v in DATASET_CONFIGS.items()},
        "epsilon_targets": EPSILON_TARGETS,
        "delta": DELTA,
        "audit_pool_size": AUDIT_POOL_SIZE,
        "query_budgets": QUERY_BUDGETS,
        "lambda_grid": LAMBDA_GRID,
        "primary_k": PRIMARY_K,
        "ablation_ks": ABLATION_KS,
    }
