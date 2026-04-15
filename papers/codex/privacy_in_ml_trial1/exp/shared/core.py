from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXP_DIR = ROOT / "exp"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

TARGET_SEEDS = [7, 17, 27]
RETRAIN_SEEDS = [7, 17]
SHADOW_COUNT = 8
NUM_WORKERS = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    for path in [DATA_DIR, FIGURES_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha1_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ensure_logs_dir(path: Path) -> Path:
    log_dir = path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def current_rss_mb() -> float:
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.uname().sysname == "Darwin":
            return float(rss_kb / (1024 ** 2))
        return float(rss_kb / 1024.0)
    except Exception:
        return 0.0


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    optimizer: str
    momentum: float = 0.0
    patience: int = 0


CONFIG: dict[str, Any] = {
    "purchase100": {
        "target": TrainConfig(epochs=35, batch_size=512, lr=1e-3, weight_decay=1e-4, optimizer="adam", patience=5),
        "shadow": TrainConfig(epochs=35, batch_size=512, lr=1e-3, weight_decay=1e-4, optimizer="adam", patience=5),
        "retrain": TrainConfig(epochs=35, batch_size=512, lr=1e-3, weight_decay=1e-4, optimizer="adam", patience=5),
        "rank_cap": 16,
        "beta_anchor": 0.05,
        "forget_lr": 5e-4,
        "retain_lr": 2.5e-4,
        "scope": "classifier_head",
    },
    "cifar10": {
        "target": TrainConfig(epochs=25, batch_size=256, lr=0.1, weight_decay=5e-4, optimizer="sgd", momentum=0.9, patience=0),
        "shadow": TrainConfig(epochs=25, batch_size=256, lr=0.1, weight_decay=5e-4, optimizer="sgd", momentum=0.9, patience=0),
        "retrain": TrainConfig(epochs=25, batch_size=256, lr=0.1, weight_decay=5e-4, optimizer="sgd", momentum=0.9, patience=0),
        "rank_cap": 32,
        "beta_anchor": 0.02,
        "forget_lr": 1e-4,
        "retain_lr": 5e-5,
        "scope": "final_block_plus_classifier",
    },
    "tau": 0.90,
    "lambda_proj": 1.0,
    "q_select": 0.25,
    "steps": 20,
    "checkpoint_steps": [5, 10, 15, 20],
    "pilot_target_runs": [
        {"dataset": "purchase100", "regime": "random_1pct", "seed": 7, "method": "base_ft"},
        {"dataset": "cifar10", "regime": "random_1pct", "seed": 7, "method": "base_ft"},
    ],
}


class IndexedTensorDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.features[idx], self.labels[idx], idx


class IndexedVisionDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray, train_aug: bool):
        self.images = images
        self.labels = labels.astype(np.int64)
        normalize = transforms.Normalize(mean.tolist(), std.tolist())
        ops: list[Any] = [transforms.ToPILImage()]
        if train_aug:
            ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        ops.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = self.transform(self.images[idx])
        label = int(self.labels[idx])
        return image, torch.tensor(label, dtype=torch.long), idx


class PurchaseMLP(nn.Module):
    def __init__(self, input_dim: int = 600, num_classes: int = 100):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        features = torch.flatten(x, 1)
        logits = m.fc(features)
        if return_features:
            return logits, features
        return logits


def build_model(dataset_name: str) -> nn.Module:
    if dataset_name == "purchase100":
        return PurchaseMLP()
    if dataset_name == "cifar10":
        return CIFARResNet18()
    raise ValueError(dataset_name)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_scoped_named_parameters(model: nn.Module, dataset_name: str) -> list[tuple[str, nn.Parameter]]:
    if dataset_name == "purchase100":
        prefixes = ("fc.",)
    elif dataset_name == "cifar10":
        prefixes = ("model.layer4.", "model.fc.")
    else:
        raise ValueError(dataset_name)
    return [(name, param) for name, param in model.named_parameters() if name.startswith(prefixes)]


def get_scoped_parameters(model: nn.Module, dataset_name: str) -> list[nn.Parameter]:
    return [param for _, param in get_scoped_named_parameters(model, dataset_name)]


def scoped_parameter_dim(model: nn.Module, dataset_name: str) -> int:
    return int(sum(param.numel() for param in get_scoped_parameters(model, dataset_name)))


def load_purchase100() -> tuple[np.ndarray, np.ndarray]:
    cached = Path("/home/zz865/.cache/huggingface/hub/datasets--TDDBench--purchase100/snapshots/d7eaad068444a9b1d3366c698a2bddfd59dec4d6/data/train-00000-of-00001.parquet")
    if cached.exists():
        frame = pd.read_parquet(cached)
        features = np.asarray(frame["feature"].tolist(), dtype=np.float32)
        labels = frame["label"].to_numpy(dtype=np.int64)
        return features, labels
    dataset = load_dataset("TDDBench/purchase100")["train"]
    features = np.asarray(dataset["feature"], dtype=np.float32)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    return features, labels


def load_cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    extracted = DATA_DIR / "cifar-10-batches-py"
    download = not extracted.exists()
    train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=download)
    test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=download)
    return np.asarray(train.data), np.asarray(train.targets), np.asarray(test.data), np.asarray(test.targets)


def stratified_split(indices: np.ndarray, labels: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels[indices],
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def build_base_pools(dataset_name: str, seed: int) -> dict[str, Any]:
    if dataset_name == "purchase100":
        x, y = load_purchase100()
        all_idx = np.arange(len(y))
        d_target, rest = stratified_split(all_idx, y, 0.30, seed)
        remaining_labels = y[rest]
        a_pool, rest2 = train_test_split(rest, test_size=2 / 3, random_state=seed + 1, stratify=remaining_labels)
        v_pool, t_eval = train_test_split(
            np.asarray(rest2),
            test_size=0.50,
            random_state=seed + 2,
            stratify=y[np.asarray(rest2)],
        )
        mean = x[d_target].mean(axis=0)
        std = x[d_target].std(axis=0) + 1e-6
        return {
            "kind": "tabular",
            "features": x,
            "labels": y,
            "d_target": np.asarray(d_target),
            "a_pool": np.asarray(a_pool),
            "v_pool": np.asarray(v_pool),
            "t_eval": np.asarray(t_eval),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }

    train_x, train_y, test_x, test_y = load_cifar10()
    train_idx = np.arange(len(train_y))
    a_pool, d_target = train_test_split(train_idx, test_size=0.70, random_state=seed, stratify=train_y)
    test_idx = np.arange(len(test_y))
    v_pool, t_eval = train_test_split(test_idx, test_size=0.60, random_state=seed + 1, stratify=test_y)
    d_target_images = train_x[d_target].astype(np.float32) / 255.0
    mean = d_target_images.mean(axis=(0, 1, 2))
    std = d_target_images.std(axis=(0, 1, 2)) + 1e-6
    return {
        "kind": "vision",
        "train_images": train_x,
        "train_labels": train_y.astype(np.int64),
        "test_images": test_x,
        "test_labels": test_y.astype(np.int64),
        "d_target": np.asarray(d_target),
        "a_pool": np.asarray(a_pool),
        "v_pool": np.asarray(v_pool),
        "t_eval": np.asarray(t_eval),
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def match_class_hist(source_indices: np.ndarray, source_labels: np.ndarray, target_labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    remaining = set(source_indices.tolist())
    for label, count in zip(*np.unique(target_labels, return_counts=True)):
        candidates = np.asarray([idx for idx in source_indices if idx in remaining and source_labels[idx] == label])
        if len(candidates) < count:
            raise RuntimeError(f"Insufficient candidates for label {label}: need {count}, have {len(candidates)}")
        chosen = rng.choice(candidates, size=count, replace=False)
        selected.extend(chosen.tolist())
        for idx in chosen.tolist():
            remaining.remove(int(idx))
    return np.asarray(selected, dtype=np.int64), np.asarray(sorted(remaining), dtype=np.int64)


def dataset_labels(pools: dict[str, Any], partition: str) -> np.ndarray:
    if pools["kind"] == "tabular":
        return pools["labels"][pools[partition]]
    if partition in {"v_pool", "t_eval"}:
        return pools["test_labels"][pools[partition]]
    return pools["train_labels"][pools[partition]]


def build_experiment_splits(dataset_name: str, regime: str, seed: int, base_model_path: Path | None = None) -> dict[str, Any]:
    pools = build_base_pools(dataset_name, seed)
    if pools["kind"] == "tabular":
        labels = pools["labels"]
        d_target = pools["d_target"]
        forget_fraction = 0.01
        forget, retain = train_test_split(d_target, test_size=1 - forget_fraction, random_state=seed + 10, stratify=labels[d_target])
        forget = np.asarray(forget)
        retain = np.asarray(retain)
    else:
        labels = pools["train_labels"]
        d_target = pools["d_target"]
        if regime == "random_1pct":
            forget_fraction = 0.01
            forget, retain = train_test_split(d_target, test_size=1 - forget_fraction, random_state=seed + 10, stratify=labels[d_target])
            forget = np.asarray(forget)
            retain = np.asarray(retain)
        else:
            if base_model_path is None:
                raise RuntimeError("Base model path required for high-loss split")
            losses = compute_cached_losses_for_indices(dataset_name, seed, regime="random_1pct", model_path=base_model_path, indices=d_target)
            count = max(1, int(round(0.05 * len(d_target))))
            order = np.argsort(losses)[::-1]
            forget = np.asarray(d_target[order[:count]])
            retain = np.asarray(d_target[order[count:]])

    r_train, remainder = train_test_split(retain, test_size=0.20, random_state=seed + 11, stratify=labels[retain])
    r_select, remainder = train_test_split(np.asarray(remainder), test_size=0.50, random_state=seed + 12, stratify=labels[np.asarray(remainder)])
    r_audit_a, r_audit_b = train_test_split(np.asarray(remainder), test_size=0.50, random_state=seed + 13, stratify=labels[np.asarray(remainder)])

    if pools["kind"] == "tabular":
        v_labels = pools["labels"]
    else:
        v_labels = pools["test_labels"]
    v_audit_a, remaining_v = match_class_hist(pools["v_pool"], v_labels, labels[np.asarray(r_audit_a)], seed + 14)
    v_audit_b, remaining_v = match_class_hist(remaining_v, v_labels, labels[np.asarray(r_audit_b)], seed + 15)

    splits = {
        "pools": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in pools.items() if k in {"d_target", "a_pool", "v_pool", "t_eval"}},
        "forget": np.asarray(forget, dtype=np.int64),
        "r_train": np.asarray(r_train, dtype=np.int64),
        "r_select": np.asarray(r_select, dtype=np.int64),
        "r_audit_a": np.asarray(r_audit_a, dtype=np.int64),
        "r_audit_b": np.asarray(r_audit_b, dtype=np.int64),
        "v_audit_a": np.asarray(v_audit_a, dtype=np.int64),
        "v_audit_b": np.asarray(v_audit_b, dtype=np.int64),
        "mean": pools["mean"],
        "std": pools["std"],
        "kind": pools["kind"],
    }
    return {"pools": pools, "splits": splits}


def _tabular_dataset(features: np.ndarray, labels: np.ndarray, indices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> IndexedTensorDataset:
    x = torch.from_numpy(((features[indices] - mean) / std).astype(np.float32))
    y = torch.from_numpy(labels[indices].astype(np.int64))
    return IndexedTensorDataset(x, y)


def build_dataset(
    dataset_name: str,
    pools: dict[str, Any],
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_aug: bool = False,
    source: str = "train",
) -> Dataset:
    if pools["kind"] == "tabular":
        return _tabular_dataset(pools["features"], pools["labels"], indices, mean, std)
    if source == "train":
        return IndexedVisionDataset(pools["train_images"][indices], pools["train_labels"][indices], mean, std, train_aug)
    return IndexedVisionDataset(pools["test_images"][indices], pools["test_labels"][indices], mean, std, train_aug)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, dataset_name: str, augmentations: int = 1) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            if dataset_name == "cifar10" and augmentations == 4:
                views = [x, torch.flip(x, dims=[3]), torch.roll(x, shifts=1, dims=2), torch.roll(x, shifts=1, dims=3)]
                logits = torch.stack([model(v) for v in views], dim=0).mean(dim=0)
            else:
                logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
            total_loss += float(loss.sum().item())
            total_acc += float((logits.argmax(dim=1) == y).sum().item())
            total_count += int(y.shape[0])
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_acc / max(total_count, 1),
        "logits": logits_np,
        "labels": labels_np,
    }


def train_model(
    dataset_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    cfg: TrainConfig,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device()
    model = build_model(dataset_name).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = None
    if dataset_name == "cifar10" and cfg.optimizer == "sgd":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True) if val_dataset else None

    best_val = -1.0
    patience_left = cfg.patience
    history: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.shape[0])
            correct += int((logits.argmax(dim=1) == y).sum().item())
            count += int(y.shape[0])
        if scheduler:
            scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(count, 1),
            "train_accuracy": correct / max(count, 1),
        }
        if val_loader:
            val_metrics = evaluate_model(model, val_loader, device, dataset_name)
            record["val_loss"] = val_metrics["loss"]
            record["val_accuracy"] = val_metrics["accuracy"]
            metric = val_metrics["accuracy"]
            if metric > best_val:
                best_val = metric
                patience_left = cfg.patience
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                patience_left -= 1
                if cfg.patience and patience_left < 0:
                    break
        else:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        history.append(record)
    runtime = time.time() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    metrics = {
        "history": history,
        "runtime_seconds": runtime,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
        "model_path": str(model_path),
    }
    save_json(out_dir / "train_metrics.json", metrics)
    return metrics


def build_train_val_split(indices: np.ndarray, labels: np.ndarray, seed: int, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 20:
        return indices, indices[:0]
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, stratify=labels[indices])
    return np.asarray(train_idx), np.asarray(val_idx)


def load_model_from_path(dataset_name: str, model_path: Path) -> nn.Module:
    model = build_model(dataset_name)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    return model


def compute_scores(model: nn.Module, dataset: Dataset, dataset_name: str, batch_size: int = 256) -> np.ndarray:
    device = get_device()
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    metrics = evaluate_model(model, loader, device, dataset_name, augmentations=4 if dataset_name == "cifar10" else 1)
    logits = torch.from_numpy(metrics["logits"])
    labels = torch.from_numpy(metrics["labels"]).long()
    loss = F.cross_entropy(logits, labels, reduction="none")
    return loss.numpy()


def compute_cached_losses_for_indices(dataset_name: str, seed: int, regime: str, model_path: Path, indices: np.ndarray) -> np.ndarray:
    pools = build_base_pools(dataset_name, seed)
    if pools["kind"] != "vision":
        raise RuntimeError("High-loss deletion is only used for CIFAR-10")
    model = load_model_from_path(dataset_name, model_path)
    dataset = build_dataset(dataset_name, pools, indices, pools["mean"], pools["std"], train_aug=False, source="train")
    return compute_scores(model, dataset, dataset_name, batch_size=256)


def flatten_gradient_list(grad_list: list[torch.Tensor | None], params: list[nn.Parameter]) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grad_list, params):
        if grad is None:
            chunks.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
        else:
            chunks.append(grad.reshape(-1))
    return torch.cat(chunks, dim=0)


def get_scope_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor, dataset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    params = get_scoped_parameters(model, dataset_name)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
    return flatten_gradient_list(list(grads), params), logits


def get_per_example_scope_gradients(model: nn.Module, x: torch.Tensor, y: torch.Tensor, dataset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    params = get_scoped_parameters(model, dataset_name)
    batch_grads = []
    logits_all = []
    for idx in range(x.shape[0]):
        logits = model(x[idx : idx + 1])
        loss = F.cross_entropy(logits, y[idx : idx + 1])
        grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False, allow_unused=True)
        batch_grads.append(flatten_gradient_list(list(grads), params))
        logits_all.append(logits.detach())
    return torch.stack(batch_grads, dim=0), torch.cat(logits_all, dim=0)


def covariance_scores(select_grads: np.ndarray, forget_grads: np.ndarray) -> np.ndarray:
    centered = forget_grads - forget_grads.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(forget_grads), 1)
    return np.einsum("bi,ij,bj->b", select_grads, cov, select_grads)


def basis_from_gradients(gradients: np.ndarray, weights: np.ndarray | None, tau: float, rank_cap: int) -> tuple[np.ndarray, dict[str, Any]]:
    if weights is None:
        weights = np.ones(len(gradients), dtype=np.float64) / max(len(gradients), 1)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.clip(weights, a_min=1e-8, a_max=None)
        weights = weights / max(weights.sum(), 1e-12)
    gram = (gradients.T * weights) @ gradients
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = np.maximum(eigvals, 0.0)
    total = max(float(positive.sum()), 1e-12)
    cumsum = np.cumsum(positive) / total
    rank = min(rank_cap, int(np.searchsorted(cumsum, tau) + 1))
    basis = eigvecs[:, :rank].astype(np.float32)
    return basis, {
        "eigenvalues": positive[: min(32, len(positive))].tolist(),
        "rank": int(rank),
        "energy_at_rank": float(cumsum[rank - 1]) if rank > 0 else 0.0,
    }


def project_update(update: np.ndarray, basis: np.ndarray, coeff: float) -> np.ndarray:
    if basis.size == 0:
        return update
    projected = basis @ (basis.T @ update)
    return update - coeff * projected


def assign_scope_gradient(model: nn.Module, dataset_name: str, grad_vector: np.ndarray) -> None:
    params = get_scoped_parameters(model, dataset_name)
    offset = 0
    grad_tensor = torch.from_numpy(grad_vector)
    for param in params:
        count = param.numel()
        chunk = grad_tensor[offset : offset + count].view_as(param).to(param.device, dtype=param.dtype)
        if param.grad is None:
            param.grad = chunk.clone()
        else:
            param.grad.copy_(chunk)
        offset += count


def zero_non_scope_grads(model: nn.Module, dataset_name: str) -> None:
    scoped_ids = {id(param) for param in get_scoped_parameters(model, dataset_name)}
    for param in model.parameters():
        if id(param) not in scoped_ids:
            param.grad = None


def collect_gradient_matrix(model: nn.Module, dataset: Dataset, batch_size: int = 256, max_items: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    device = get_device()
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    grads: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        grad, _ = get_per_example_scope_gradients(model, x, y, "purchase100" if isinstance(model, PurchaseMLP) else "cifar10")
        grads.append(grad.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        if max_items and sum(len(g) for g in grads) >= max_items:
            break
    grad_np = np.concatenate(grads, axis=0)
    label_np = np.concatenate(labels, axis=0)
    if max_items:
        grad_np = grad_np[:max_items]
        label_np = label_np[:max_items]
    return grad_np.astype(np.float32), label_np.astype(np.int64)


def fit_shadow_attack(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict[str, Any]:
    member_scores = np.asarray(member_scores, dtype=np.float64)
    nonmember_scores = np.asarray(nonmember_scores, dtype=np.float64)
    mu_in = float(member_scores.mean())
    sigma_in = float(member_scores.std(ddof=0) + 1e-6)
    mu_out = float(nonmember_scores.mean())
    sigma_out = float(nonmember_scores.std(ddof=0) + 1e-6)
    member_llr = gaussian_logpdf(member_scores, mu_in, sigma_in) - gaussian_logpdf(member_scores, mu_out, sigma_out)
    nonmember_llr = gaussian_logpdf(nonmember_scores, mu_in, sigma_in) - gaussian_logpdf(nonmember_scores, mu_out, sigma_out)
    threshold = float(np.quantile(nonmember_llr, 0.99))
    return {
        "threshold_1pct_fpr": threshold,
        "mu_in": mu_in,
        "sigma_in": sigma_in,
        "mu_out": mu_out,
        "sigma_out": sigma_out,
        "shadow_member_llr_mean": float(member_llr.mean()),
        "shadow_nonmember_llr_mean": float(nonmember_llr.mean()),
    }


def gaussian_logpdf(values: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sigma = max(float(sigma), 1e-6)
    return -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((values - mu) / sigma) ** 2


def compute_attack_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray, attack_model: dict[str, Any]) -> dict[str, float]:
    member_scores = np.asarray(member_scores, dtype=np.float64)
    nonmember_scores = np.asarray(nonmember_scores, dtype=np.float64)
    member_llr = gaussian_logpdf(member_scores, attack_model["mu_in"], attack_model["sigma_in"]) - gaussian_logpdf(
        member_scores, attack_model["mu_out"], attack_model["sigma_out"]
    )
    nonmember_llr = gaussian_logpdf(
        nonmember_scores, attack_model["mu_in"], attack_model["sigma_in"]
    ) - gaussian_logpdf(nonmember_scores, attack_model["mu_out"], attack_model["sigma_out"])
    y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    y_score = np.concatenate([member_llr, nonmember_llr])
    preds = (member_llr >= attack_model["threshold_1pct_fpr"]).astype(np.float32)
    tpr = float(preds.mean()) if len(preds) else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {
        "tpr_at_1pct_fpr": tpr,
        "roc_auc": auc,
        "member_llr_mean": float(member_llr.mean()) if len(member_llr) else float("nan"),
        "nonmember_llr_mean": float(nonmember_llr.mean()) if len(nonmember_llr) else float("nan"),
    }


def bootstrap_ci(differences: list[float], seed: int = 0, rounds: int = 2000) -> list[float]:
    if not differences:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    diffs = np.asarray(differences, dtype=np.float64)
    boots = []
    for _ in range(rounds):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boots.append(float(sample.mean()))
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_main_comparison(summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, regime in zip(axes, summary["dataset_regime"].unique()):
        sub = summary[summary["dataset_regime"] == regime]
        for method in sub["method"].unique():
            row = sub[sub["method"] == method].iloc[0]
            ax.errorbar(
                row["retained_tpr_mean"],
                row["test_accuracy_mean"],
                xerr=row["retained_tpr_std"],
                yerr=row["test_accuracy_std"],
                fmt="o",
                capsize=4,
                label=method,
            )
        ax.set_title(regime)
        ax.set_xlabel("Retained TPR@1%FPR")
        ax.set_ylabel("Test Accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "main_comparison.png", dpi=160)
    fig.savefig(FIGURES_DIR / "main_comparison.pdf")
    plt.close(fig)


def plot_ablation(ablation_df: pd.DataFrame) -> None:
    if ablation_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("retained_tpr_mean", "Retained TPR@1%FPR"),
        ("retained_auc_mean", "Retained ROC-AUC"),
        ("runtime_ratio_mean", "Runtime Ratio"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(ablation_df["method"], ablation_df[column], yerr=ablation_df[column.replace("_mean", "_std")], capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation_study.png", dpi=160)
    fig.savefig(FIGURES_DIR / "ablation_study.pdf")
    plt.close(fig)
