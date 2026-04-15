from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"
FIGURE_ROOT = ROOT / "figures"
PILOT_ROOT = OUTPUT_ROOT / "pilots"

SEEDS = [11, 22, 33]
DATASETS = ["purchase100", "cifar10"]
METHODS = [
    "erm",
    "global_mixup",
    "relaxloss",
    "targeted_random",
    "targeted_loss_only",
    "targeted_forecast",
    "forecast_single_artifact",
    "forecast_no_refresh",
    "forecast_targeted_penalty",
]
MAIN_METHODS = METHODS[:6]
ABLATIONS = METHODS[6:]
PILOTS = [
    ("purchase100", "erm", 11),
    ("purchase100", "targeted_forecast", 11),
    ("cifar10", "erm", 11),
    ("cifar10", "targeted_forecast", 11),
]


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    train_size: int
    val_size: int
    test_size: int
    ref_size: int
    batch_size: int
    epochs: int
    warmup_epochs: int
    refresh_epochs: tuple[int, ...]
    intervention_budget: float
    num_workers: int
    amp: bool
    lr: float
    weight_decay: float
    optimizer: str
    momentum: float = 0.0
    mixup_alpha: float = 0.0
    targeted_penalty: float = 0.0


PURCHASE_CFG = DatasetConfig(
    name="purchase100",
    num_classes=100,
    train_size=12000,
    val_size=2000,
    test_size=2000,
    ref_size=8000,
    batch_size=512,
    epochs=16,
    warmup_epochs=3,
    refresh_epochs=(7, 11, 15),
    intervention_budget=0.10,
    num_workers=4,
    amp=True,
    lr=1e-3,
    weight_decay=1e-4,
    optimizer="adam",
    mixup_alpha=0.2,
    targeted_penalty=0.02,
)

CIFAR_CFG = DatasetConfig(
    name="cifar10",
    num_classes=10,
    train_size=15000,
    val_size=5000,
    test_size=10000,
    ref_size=20000,
    batch_size=256,
    epochs=16,
    warmup_epochs=3,
    refresh_epochs=(7, 11, 15),
    intervention_budget=0.10,
    num_workers=4,
    amp=True,
    lr=0.1,
    weight_decay=5e-4,
    optimizer="sgd",
    momentum=0.9,
    mixup_alpha=0.4,
    targeted_penalty=0.01,
)


DATASET_CONFIGS = {
    "purchase100": PURCHASE_CFG,
    "cifar10": CIFAR_CFG,
}


def ensure_output_dirs() -> None:
    for path in [
        OUTPUT_ROOT / "configs",
        OUTPUT_ROOT / "checkpoints",
        OUTPUT_ROOT / "traces",
        OUTPUT_ROOT / "metrics",
        OUTPUT_ROOT / "attacks",
        OUTPUT_ROOT / "plots",
        OUTPUT_ROOT / "tables",
        OUTPUT_ROOT / "logs",
        PILOT_ROOT / "configs",
        PILOT_ROOT / "checkpoints",
        PILOT_ROOT / "traces",
        PILOT_ROOT / "metrics",
        PILOT_ROOT / "attacks",
        PILOT_ROOT / "logs",
        PILOT_ROOT / "tables",
        FIGURE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def run_id(dataset: str, method: str, seed: int) -> str:
    return f"{dataset}_{method}_{seed}"


def config_to_dict(cfg: DatasetConfig) -> dict:
    return asdict(cfg)
