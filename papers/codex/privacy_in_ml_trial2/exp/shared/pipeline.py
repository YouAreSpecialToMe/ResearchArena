from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights

from .config import (
    ARTIFACTS_DIR,
    AUDIT_POOL_SIZE,
    CACHE_VERSION,
    CANARYS_PER_KIND,
    DATASET_CONFIGS,
    DELTA,
    EPSILON_TARGETS,
    FIGURES_DIR,
    LAMBDA_GRID,
    ORDERING_SEEDS,
    PRIMARY_K,
    QUERY_BUDGETS,
    ROOT,
    SEEDS,
    combo_name,
    config_dict,
    ensure_dirs,
)
from .metrics import audit_metrics
from .models import build_model
from .utils import (
    bootstrap_ci,
    bootstrap_mean_and_ci,
    choose_device,
    count_parameters,
    elapsed_minutes,
    entropy_from_probs,
    json_dump,
    json_load,
    now,
    safe_float,
    set_num_threads,
    set_seed,
    softmax_np,
)


STAGE_NAMES = [
    "environment",
    "data_preparation",
    "train_targets",
    "single_view_baselines",
    "weak_view_main",
    "ablations",
    "visualization",
]


def stage_dir(stage: str):
    path = ROOT / "exp" / stage
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(parents=True, exist_ok=True)
    return path


def stage_log(stage: str, name: str, payload: dict) -> None:
    json_dump(payload, stage_dir(stage) / "logs" / name)


def write_stage_result(stage: str, payload: dict) -> None:
    json_dump(payload, stage_dir(stage) / "results.json")


def write_stage_skip(stage: str, reason: str) -> None:
    path = stage_dir(stage) / "SKIPPED.md"
    path.write_text(reason.strip() + "\n", encoding="utf-8")


def clear_stage_skip(stage: str) -> None:
    path = stage_dir(stage) / "SKIPPED.md"
    if path.exists():
        path.unlink()


def guardrail_epochs(dataset: str) -> int:
    path = ARTIFACTS_DIR / "results" / "runtime_guardrails" / f"{dataset}.json"
    if path.exists():
        return int(json_load(path)["epochs"])
    return DATASET_CONFIGS[dataset].epochs


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, indices: list[int], transform: Callable | None = None) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.indices[idx]


class TensorDatasetWithIds(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, ids: list[str]) -> None:
        self.images = images
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], int(self.labels[idx].item()), self.ids[idx]


def dataset_transforms(dataset: str):
    if dataset == "fashion_mnist":
        mean, std = (0.5,), (0.5,)
        train_transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=0, translate=(2 / 28, 2 / 28)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        to_pil = None
    else:
        mean = ResNet18_Weights.DEFAULT.transforms().mean
        std = ResNet18_Weights.DEFAULT.transforms().std
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        to_pil = None
    return train_transform, eval_transform, to_pil


def weak_transform_pool(dataset: str):
    if dataset == "fashion_mnist":
        return {
            "translate": transforms.RandomAffine(degrees=0, translate=(2 / 28, 2 / 28)),
            "crop": transforms.RandomResizedCrop(28, scale=(0.9, 1.0)),
            "noise_001": "gaussian_0.01",
            "noise_003": "gaussian_0.03",
            "brightness": transforms.ColorJitter(brightness=0.05),
        }
    return {
        "crop2": transforms.RandomCrop(32, padding=2),
        "flip": transforms.RandomHorizontalFlip(p=1.0),
        "jitter": transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        "noise_0005": "gaussian_0.005",
        "noise_001": "gaussian_0.01",
    }


def load_raw_dataset(dataset: str):
    root = ROOT / "data"
    if dataset == "fashion_mnist":
        train = datasets.FashionMNIST(root=root, train=True, download=True)
        test = datasets.FashionMNIST(root=root, train=False, download=True)
    else:
        train = datasets.CIFAR10(root=root, train=True, download=True)
        test = datasets.CIFAR10(root=root, train=False, download=True)
    return train, test


def split_indices(dataset: str, seed: int) -> dict:
    cfg = DATASET_CONFIGS[dataset]
    path = ARTIFACTS_DIR / "results" / "splits" / f"{dataset}_seed{seed}.json"
    if path.exists():
        return json_load(path)
    train, _ = load_raw_dataset(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train)).tolist()
    splits = {}
    start = 0
    for key, size in [
        ("private_base", cfg.private_train_size),
        ("public_calibration", cfg.calibration_size),
        ("public_proxy", cfg.proxy_size),
        ("public_screening", cfg.screening_size),
        ("audit_reserve", cfg.reserve_size),
    ]:
        splits[key] = perm[start : start + size]
        start += size
    json_dump(splits, path)
    stage_log("data_preparation", f"{dataset}_seed{seed}_splits.json", splits)
    return splits


def compute_tensor_batch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_list, labels_list, ids_list, feat_norms = [], [], [], []
    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu())
            labels_list.append(torch.as_tensor(labels))
            ids_list.extend(list(ids))
            feat_norms.append(model.feature_norm(images).detach().cpu())
    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    feat_norm = torch.cat(feat_norms).numpy()
    return logits, labels, ids_list, feat_norm


def train_non_private_model(dataset: str, split_name: str, indices: list[int], seed: int) -> Path:
    out_path = ARTIFACTS_DIR / "checkpoints" / f"{dataset}_{split_name}_seed{seed}_public.pt"
    if out_path.exists():
        return out_path
    set_seed(seed)
    device = choose_device()
    cfg = DATASET_CONFIGS[dataset]
    train_ds_raw, test_ds_raw = load_raw_dataset(dataset)
    train_tf, eval_tf, _ = dataset_transforms(dataset)
    train_ds = IndexedDataset(train_ds_raw, indices, transform=train_tf)
    test_ds = IndexedDataset(test_ds_raw, list(range(len(test_ds_raw))), transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, 512), shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    model = build_model(dataset).to(device)
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=5)
    for _ in range(5):
        model.train()
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    metrics = evaluate_model(model, test_loader, device)
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, out_path)
    return out_path


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return {"accuracy": correct / total, "loss": loss_sum / total}


def fgsm_from_surrogate(model: nn.Module, image: torch.Tensor, label: int, epsilon: float) -> torch.Tensor:
    device = next(model.parameters()).device
    x = image.unsqueeze(0).to(device).clone().detach().requires_grad_(True)
    y = torch.tensor([label], device=device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    adv = x + epsilon * x.grad.sign()
    return adv.detach().squeeze(0).cpu()


def build_audit_pool(dataset: str, seed: int) -> dict:
    path = ARTIFACTS_DIR / "results" / "audit_pairs" / f"{dataset}_seed{seed}.json"
    if path.exists():
        return json_load(path)
    set_seed(seed)
    cfg = DATASET_CONFIGS[dataset]
    splits = split_indices(dataset, seed)
    train_ds_raw, _ = load_raw_dataset(dataset)
    _, eval_tf, _ = dataset_transforms(dataset)
    public_train = splits["public_calibration"] + splits["public_proxy"]
    surrogate_path = train_non_private_model(dataset, "surrogate", public_train, seed)
    surrogate = build_model(dataset)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=False)["model_state"])
    surrogate.to(choose_device()).eval()
    reserve = splits["audit_reserve"]
    labels = {idx: int(train_ds_raw[idx][1]) for idx in reserve}
    by_class = {klass: [] for klass in range(cfg.num_classes)}
    for idx in reserve:
        by_class[labels[idx]].append(idx)
    rng = np.random.default_rng(seed)
    rand_indices = rng.choice(reserve, size=CANARYS_PER_KIND, replace=False).tolist()
    remaining = [idx for idx in reserve if idx not in rand_indices]
    fgsm_indices = rng.choice(remaining, size=CANARYS_PER_KIND, replace=False).tolist()
    used_reserves = set()
    pairs = []
    for kind, chosen in [("random", rand_indices), ("fgsm", fgsm_indices)]:
        for src_idx in chosen:
            label = labels[src_idx]
            candidates = [idx for idx in by_class[label] if idx != src_idx and idx not in used_reserves]
            reserve_idx = int(rng.choice(candidates))
            used_reserves.add(reserve_idx)
            image, _ = train_ds_raw[src_idx]
            tensor = eval_tf(image)
            if kind == "fgsm":
                tensor = fgsm_from_surrogate(surrogate, tensor, label, cfg.fgsm_epsilon)
            pairs.append(
                {
                    "pair_id": f"{kind}_{src_idx}_{reserve_idx}",
                    "kind": kind,
                    "source_index": int(src_idx),
                    "reserve_index": int(reserve_idx),
                    "label": int(label),
                }
            )
    json_dump({"pairs": pairs}, path)
    canary_dir = ARTIFACTS_DIR / "results" / "audit_pairs"
    canary_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            pair["pair_id"]: fgsm_from_surrogate(
                surrogate,
                eval_tf(train_ds_raw[pair["source_index"]][0]),
                pair["label"],
                cfg.fgsm_epsilon,
            )
            if pair["kind"] == "fgsm"
            else eval_tf(train_ds_raw[pair["source_index"]][0])
            for pair in pairs
        },
        canary_dir / f"{dataset}_seed{seed}_canaries.pt",
    )
    stage_log(
        "data_preparation",
        f"{dataset}_seed{seed}_audit_pool.json",
        {"dataset": dataset, "seed": seed, "num_pairs": len(pairs), "pairs_preview": pairs[:10]},
    )
    return {"pairs": pairs}


def membership_mask(dataset: str, epsilon: float, seed: int) -> dict:
    path = ARTIFACTS_DIR / "results" / "membership_masks" / f"{combo_name(dataset, epsilon, seed)}.json"
    if path.exists():
        return json_load(path)
    rng = np.random.default_rng(int(seed + 100 * epsilon))
    mask = rng.integers(0, 2, size=AUDIT_POOL_SIZE).tolist()
    json_dump({"mask": mask}, path)
    return {"mask": mask}


def make_private_indices(dataset: str, epsilon: float, seed: int) -> tuple[list[int], list[dict]]:
    splits = split_indices(dataset, seed)
    pairs = build_audit_pool(dataset, seed)["pairs"]
    mask = membership_mask(dataset, epsilon, seed)["mask"]
    private = list(splits["private_base"])
    chosen = []
    for pair, keep in zip(pairs, mask):
        private.append(pair["source_index"] if keep else pair["reserve_index"])
        chosen.append({**pair, "is_member": int(keep)})
    return private, chosen


def find_noise_multiplier(target_epsilon: float, epochs: int, batch_size: int, dataset_size: int) -> float:
    return float(
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=DELTA,
            sample_rate=batch_size / dataset_size,
            epochs=epochs,
            accountant="prv",
            epsilon_tolerance=0.05,
        )
    )


def train_dp_target(dataset: str, epsilon: float, seed: int) -> dict:
    out_path = ARTIFACTS_DIR / "checkpoints" / f"{combo_name(dataset, epsilon, seed)}.pt"
    if out_path.exists():
        return torch.load(out_path, map_location="cpu", weights_only=False)
    set_seed(seed)
    set_num_threads()
    device = choose_device()
    cfg = DATASET_CONFIGS[dataset]
    epochs = guardrail_epochs(dataset)
    train_ds_raw, test_ds_raw = load_raw_dataset(dataset)
    train_tf, eval_tf, _ = dataset_transforms(dataset)
    private_indices, audit_meta = make_private_indices(dataset, epsilon, seed)
    train_ds = IndexedDataset(train_ds_raw, private_indices, transform=train_tf)
    test_ds = IndexedDataset(test_ds_raw, list(range(len(test_ds_raw))), transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)
    model = build_model(dataset).to(device)
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if seed == SEEDS[0]:
        noise_multiplier = find_noise_multiplier(epsilon, epochs, cfg.batch_size, len(train_ds))
        json_dump({"noise_multiplier": noise_multiplier}, ARTIFACTS_DIR / "results" / "noise_multipliers" / f"{dataset}_eps{int(epsilon)}.json")
    else:
        noise_multiplier = json_load(ARTIFACTS_DIR / "results" / "noise_multipliers" / f"{dataset}_eps{int(epsilon)}.json")["noise_multiplier"]
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=cfg.max_grad_norm,
        poisson_sampling=False,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    start = now()
    history = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            count += labels.size(0)
        scheduler.step()
        metrics = evaluate_model(model, test_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(count, 1),
                "test_accuracy": metrics["accuracy"],
                "epsilon": privacy_engine.get_epsilon(DELTA),
            }
        )
    final_metrics = evaluate_model(model, test_loader, device)
    payload = {
        "model_state": model._module.state_dict() if hasattr(model, "_module") else model.state_dict(),
        "history": history,
        "epsilon_realized": history[-1]["epsilon"],
        "test_metrics": final_metrics,
        "audit_meta": audit_meta,
        "runtime_minutes": elapsed_minutes(start),
        "parameter_count": count_parameters(model),
        "epochs": epochs,
    }
    torch.save(payload, out_path)
    if seed == SEEDS[0]:
        threshold = 9.0 if dataset == "fashion_mnist" else 15.0
        reduced_epochs = int(np.ceil(cfg.epochs * 0.8))
        guardrail_path = ARTIFACTS_DIR / "results" / "runtime_guardrails" / f"{dataset}.json"
        guardrail_payload = {
            "dataset": dataset,
            "initial_runtime_minutes": payload["runtime_minutes"],
            "threshold_minutes": threshold,
            "epochs": reduced_epochs if payload["runtime_minutes"] > threshold else cfg.epochs,
            "triggered": bool(payload["runtime_minutes"] > threshold),
        }
        json_dump(guardrail_payload, guardrail_path)
    stage_log(
        "train_targets",
        f"{combo_name(dataset, epsilon, seed)}_history.json",
        {
            "dataset": dataset,
            "epsilon_target": epsilon,
            "seed": seed,
            "history": history,
            "test_metrics": final_metrics,
            "runtime_minutes": payload["runtime_minutes"],
        },
    )
    return payload


def load_checkpoint_model(dataset: str, epsilon: float, seed: int, device: torch.device) -> tuple[nn.Module, dict]:
    payload = train_dp_target(dataset, epsilon, seed)
    model = build_model(dataset).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def standardize_public_features(
    dataset: str,
    target_model: nn.Module,
    surrogate_model: nn.Module,
    reference_model: nn.Module,
    indices: list[int],
    transform,
    device,
):
    train_ds_raw, _ = load_raw_dataset(dataset)
    subset = IndexedDataset(train_ds_raw, indices, transform=transform)
    loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=2)
    logits, labels, ids, _ = compute_tensor_batch(target_model, loader, device)
    surrogate_logits, _, _, _ = compute_tensor_batch(surrogate_model, loader, device)
    ref_logits, _, _, ref_feat = compute_tensor_batch(reference_model, loader, device)
    probs = softmax_np(logits)
    surrogate_probs = softmax_np(surrogate_logits)
    loss = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
    rows = pd.DataFrame(
        {
            "index": ids,
            "label": labels,
            "surrogate_loss": -np.log(np.clip(surrogate_probs[np.arange(len(labels)), labels], 1e-12, 1.0)),
            "surrogate_entropy": entropy_from_probs(surrogate_probs),
            "surrogate_max_prob": surrogate_probs.max(axis=1),
            "ref_feature_norm": ref_feat,
            "clean_neg_loss": -loss,
        }
    )
    return rows


def fit_quantile_models(features: pd.DataFrame, target_name: str) -> dict:
    use_cols = ["surrogate_loss", "surrogate_entropy", "surrogate_max_prob", "label", "ref_feature_norm"]
    models = {}
    X = features[use_cols].copy()
    X["label"] = X["label"].astype(float)
    target = features[target_name].to_numpy()
    for alpha in [0.5, 0.9, 0.95]:
        reg = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=0, n_estimators=150, max_depth=3)
        reg.fit(X, target)
        models[str(alpha)] = reg
    denom_floor = max(float(np.quantile(target, 0.95) - np.quantile(target, 0.5)), float(np.std(target) * 0.5), 0.05 if "agree_rate" in target_name else 1e-3)
    return {"target": target_name, "features": use_cols, "models": models, "denom_floor": denom_floor}


def apply_quantile_residual(models: dict, features: pd.DataFrame) -> np.ndarray:
    X = features[models["features"]].copy()
    X["label"] = X["label"].astype(float)
    q50 = models["models"]["0.5"].predict(X)
    q90 = models["models"]["0.9"].predict(X)
    q95 = models["models"]["0.95"].predict(X)
    target = features[models["target"]].to_numpy()
    denom = np.maximum(q95 - q50, models.get("denom_floor", 1e-3))
    return (target - q90) / denom


def screen_transforms(dataset: str) -> list[str]:
    canonical_seed = SEEDS[0]
    path = ARTIFACTS_DIR / "results" / "screening" / f"{dataset}.json"
    if path.exists():
        return json_load(path)["selected"]
    device = choose_device()
    cfg = DATASET_CONFIGS[dataset]
    splits = split_indices(dataset, canonical_seed)
    public_train = splits["public_calibration"] + splits["public_proxy"]
    ref_path = train_non_private_model(dataset, "reference", public_train, canonical_seed + 1)
    surrogate_path = train_non_private_model(dataset, "surrogate", public_train, canonical_seed)
    ref_model = build_model(dataset).to(device)
    ref_model.load_state_dict(torch.load(ref_path, map_location="cpu", weights_only=False)["model_state"])
    ref_model.eval()
    surrogate = build_model(dataset).to(device)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=False)["model_state"])
    surrogate.eval()
    train_ds_raw, _ = load_raw_dataset(dataset)
    _, eval_tf, _ = dataset_transforms(dataset)
    screening_indices = splits["public_screening"][:1000]
    fgsm_indices = splits["public_screening"][:500]
    pool = weak_transform_pool(dataset)
    selected = []
    details = []
    base_subset = IndexedDataset(train_ds_raw, screening_indices, transform=eval_tf)
    base_loader = DataLoader(base_subset, batch_size=256, shuffle=False, num_workers=2)
    base_logits, base_labels, _, _ = compute_tensor_batch(ref_model, base_loader, device)
    base_preds = base_logits.argmax(axis=1)
    base_acc = (base_preds == base_labels).mean()
    fgsm_tensors = []
    fgsm_labels = []
    for idx in fgsm_indices:
        image, label = train_ds_raw[idx]
        fgsm_tensors.append(fgsm_from_surrogate(surrogate, eval_tf(image), int(label), cfg.fgsm_epsilon))
        fgsm_labels.append(int(label))
    fgsm_stack = torch.stack(fgsm_tensors)
    fgsm_labels = np.asarray(fgsm_labels)
    for name, transform_obj in pool.items():
        clean_images = []
        clean_labels = []
        for idx in screening_indices:
            image, label = train_ds_raw[idx]
            transformed = apply_single_transform(eval_tf(image), dataset, transform_obj)
            clean_images.append(transformed)
            clean_labels.append(int(label))
        clean_ds = TensorDatasetWithIds(torch.stack(clean_images), torch.tensor(clean_labels), [str(i) for i in screening_indices])
        clean_logits, clean_y, _, _ = compute_tensor_batch(ref_model, DataLoader(clean_ds, batch_size=256, shuffle=False), device)
        clean_preds = clean_logits.argmax(axis=1)
        clean_acc = (clean_preds == clean_y).mean()
        clean_agree = (clean_preds == base_preds).mean()
        fgsm_views = torch.stack([apply_single_transform(x, dataset, transform_obj) for x in fgsm_stack])
        fgsm_ds = TensorDatasetWithIds(fgsm_views, torch.tensor(fgsm_labels), [f"fgsm_{i}" for i in range(len(fgsm_labels))])
        fgsm_logits, _, _, _ = compute_tensor_batch(ref_model, DataLoader(fgsm_ds, batch_size=256, shuffle=False), device)
        fgsm_preds = fgsm_logits.argmax(axis=1)
        fgsm_base_ds = TensorDatasetWithIds(fgsm_stack, torch.tensor(fgsm_labels), [f"base_{i}" for i in range(len(fgsm_labels))])
        fgsm_base_logits, _, _, _ = compute_tensor_batch(ref_model, DataLoader(fgsm_base_ds, batch_size=256, shuffle=False), device)
        fgsm_base_preds = fgsm_base_logits.argmax(axis=1)
        fgsm_acc = (fgsm_preds == fgsm_labels).mean()
        fgsm_base_acc = (fgsm_base_preds == fgsm_labels).mean()
        fgsm_agree = (fgsm_preds == fgsm_base_preds).mean()
        passed = (base_acc - clean_acc) <= 0.01 and (fgsm_base_acc - fgsm_acc) <= 0.01 and clean_agree >= 0.98 and fgsm_agree >= 0.98
        details.append(
            {
                "transform": name,
                "clean_acc_drop": float(base_acc - clean_acc),
                "fgsm_acc_drop": float(fgsm_base_acc - fgsm_acc),
                "clean_agree": float(clean_agree),
                "fgsm_agree": float(fgsm_agree),
                "passed": bool(passed),
            }
        )
        if passed:
            selected.append(name)
    stress_case = len(selected) < 2
    json_dump(
        {
            "selected": selected,
            "all_candidates": list(pool.keys()),
            "stress_case": stress_case,
            "canonical_seed": canonical_seed,
            "details": details,
        },
        path,
    )
    stage_log("data_preparation", f"{dataset}_screening.json", json_load(path))
    return selected


def apply_single_transform(tensor_image: torch.Tensor, dataset: str, transform_obj):
    x = tensor_image.clone()
    if isinstance(transform_obj, str) and transform_obj.startswith("gaussian_"):
        std = float(transform_obj.split("_")[1])
        x = x + std * torch.randn_like(x)
        return x.clamp(x.min().item(), x.max().item())
    if dataset == "fashion_mnist":
        mean, std = 0.5, 0.5
        unnorm = x * std + mean
        out = transform_obj(transforms.ToPILImage()(unnorm))
        out = transforms.ToTensor()(out)
        out = transforms.Normalize((mean,), (std,))(out)
        return out
    mean = torch.tensor(ResNet18_Weights.DEFAULT.transforms().mean).view(3, 1, 1)
    std = torch.tensor(ResNet18_Weights.DEFAULT.transforms().std).view(3, 1, 1)
    unnorm = x * std + mean
    out = transform_obj(transforms.ToPILImage()(unnorm))
    out = transforms.ToTensor()(out)
    out = transforms.Normalize(tuple(mean.view(-1).tolist()), tuple(std.view(-1).tolist()))(out)
    return out


def score_checkpoint(dataset: str, epsilon: float, seed: int, k_values: list[int] | None = None) -> dict:
    if k_values is None:
        k_values = [2, 4]
    score_path = ARTIFACTS_DIR / "scores" / f"{combo_name(dataset, epsilon, seed)}.parquet"
    meta_path = ARTIFACTS_DIR / "scores" / f"{combo_name(dataset, epsilon, seed)}_meta.json"
    if score_path.exists() and meta_path.exists():
        meta = json_load(meta_path)
        if meta.get("cache_version") == CACHE_VERSION:
            return {"score_path": str(score_path), "meta_path": str(meta_path)}
    device = choose_device()
    _, eval_tf, _ = dataset_transforms(dataset)
    model, ckpt = load_checkpoint_model(dataset, epsilon, seed, device)
    splits = split_indices(dataset, seed)
    public_train = splits["public_calibration"] + splits["public_proxy"]
    ref_path = train_non_private_model(dataset, "reference", public_train, seed + 1)
    surrogate_path = train_non_private_model(dataset, "surrogate", public_train, seed)
    reference_model = build_model(dataset).to(device)
    reference_model.load_state_dict(torch.load(ref_path, map_location="cpu", weights_only=False)["model_state"])
    reference_model.eval()
    surrogate = build_model(dataset).to(device)
    surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=False)["model_state"])
    surrogate.eval()
    selected_transforms = screen_transforms(dataset)
    if not selected_transforms:
        selected_transforms = list(weak_transform_pool(dataset).keys())[:2]
    transform_pool = weak_transform_pool(dataset)
    calibration = standardize_public_features(dataset, model, surrogate, reference_model, splits["public_calibration"], eval_tf, device)
    canaries = torch.load(ARTIFACTS_DIR / "results" / "audit_pairs" / f"{dataset}_seed{seed}_canaries.pt", weights_only=False)
    rows = []
    train_ds_raw, _ = load_raw_dataset(dataset)
    for item in ckpt["audit_meta"]:
        pair_id = item["pair_id"]
        if item["kind"] == "fgsm":
            image = canaries[pair_id]
        else:
            image = eval_tf(train_ds_raw[item["source_index"]][0])
        label = item["label"]
        clean_logits = model(image.unsqueeze(0).to(device)).detach().cpu().numpy()
        clean_probs = softmax_np(clean_logits)
        clean_neg_loss = float(np.log(np.clip(clean_probs[0, label], 1e-12, 1.0)))
        surrogate_logits = surrogate(image.unsqueeze(0).to(device)).detach().cpu().numpy()
        surrogate_probs = softmax_np(surrogate_logits)
        base_features = {
            "label": label,
            "surrogate_loss": float(-np.log(np.clip(surrogate_probs[0, label], 1e-12, 1.0))),
            "surrogate_entropy": float(entropy_from_probs(surrogate_probs)[0]),
            "surrogate_max_prob": float(surrogate_probs.max()),
            "ref_feature_norm": float(reference_model.feature_norm(image.unsqueeze(0).to(device)).detach().cpu().numpy()[0]),
        }
        row = {
            "pair_id": pair_id,
            "is_member": item["is_member"],
            "kind": item["kind"],
            "label": label,
            "clean_neg_loss": clean_neg_loss,
            **base_features,
        }
        for k in k_values:
            weak_losses = []
            weak_preds = []
            for i in range(k):
                tname = selected_transforms[i % len(selected_transforms)]
                view = apply_single_transform(image, dataset, transform_pool[tname])
                logits = model(view.unsqueeze(0).to(device)).detach().cpu().numpy()
                probs = softmax_np(logits)
                weak_losses.append(float(np.log(np.clip(probs[0, label], 1e-12, 1.0))))
                weak_preds.append(int(np.argmax(probs[0])))
            row[f"mean_neg_loss_k{k}"] = float(np.mean(weak_losses))
            row[f"worst_neg_loss_k{k}"] = float(np.min(weak_losses))
            row[f"var_neg_loss_k{k}"] = float(np.var(weak_losses))
            row[f"agree_rate_k{k}"] = float(np.mean(np.asarray(weak_preds) == weak_preds[0]))
        rows.append(row)
    score_df = pd.DataFrame(rows)
    quantile_payload = {}
    calibration_stats = {}
    for k in k_values:
        transformed = []
        for idx in splits["public_calibration"]:
            image, label = train_ds_raw[idx]
            clean = eval_tf(image)
            surrogate_logits = surrogate(clean.unsqueeze(0).to(device)).detach().cpu().numpy()
            surrogate_probs = softmax_np(surrogate_logits)
            feats = {
                "index": idx,
                "label": int(label),
                "surrogate_loss": float(-np.log(np.clip(surrogate_probs[0, label], 1e-12, 1.0))),
                "surrogate_entropy": float(entropy_from_probs(surrogate_probs)[0]),
                "surrogate_max_prob": float(surrogate_probs.max()),
                "ref_feature_norm": float(reference_model.feature_norm(clean.unsqueeze(0).to(device)).detach().cpu().numpy()[0]),
                "clean_neg_loss": float(np.log(np.clip(softmax_np(model(clean.unsqueeze(0).to(device)).detach().cpu().numpy())[0, label], 1e-12, 1.0))),
            }
            weak_losses = []
            weak_preds = []
            for i in range(k):
                tname = selected_transforms[i % len(selected_transforms)]
                view = apply_single_transform(clean, dataset, transform_pool[tname])
                logits = model(view.unsqueeze(0).to(device)).detach().cpu().numpy()
                probs = softmax_np(logits)
                weak_losses.append(float(np.log(np.clip(probs[0, label], 1e-12, 1.0))))
                weak_preds.append(int(np.argmax(probs[0])))
            feats.update(
                {
                    f"mean_neg_loss_k{k}": float(np.mean(weak_losses)),
                    f"worst_neg_loss_k{k}": float(np.min(weak_losses)),
                    f"var_neg_loss_k{k}": float(np.var(weak_losses)),
                    f"agree_rate_k{k}": float(np.mean(np.asarray(weak_preds) == weak_preds[0])),
                }
            )
            transformed.append(feats)
        cal_k = pd.DataFrame(transformed)
        calibration_stats[f"k{k}"] = {
            "clean_neg_loss_mean": float(calibration["clean_neg_loss"].mean()),
            "clean_neg_loss_std": float(calibration["clean_neg_loss"].std(ddof=0) + 1e-6),
            "worst_neg_loss_mean": float(cal_k[f"worst_neg_loss_k{k}"].mean()),
            "worst_neg_loss_std": float(cal_k[f"worst_neg_loss_k{k}"].std(ddof=0) + 1e-6),
            "agree_rate_mean": float(cal_k[f"agree_rate_k{k}"].mean()),
            "agree_rate_std": float(cal_k[f"agree_rate_k{k}"].std(ddof=0) + 1e-6),
        }
        for column in ["clean_neg_loss", f"mean_neg_loss_k{k}", f"worst_neg_loss_k{k}", f"var_neg_loss_k{k}", f"agree_rate_k{k}"]:
            source_df = calibration if column == "clean_neg_loss" else cal_k
            quantile_payload[f"{column}"] = fit_quantile_models(source_df, column)
        score_df[f"r_clean_k{k}"] = apply_quantile_residual(quantile_payload["clean_neg_loss"], score_df)
        for stat in ["mean_neg_loss", "worst_neg_loss", "var_neg_loss", "agree_rate"]:
            score_df[f"r_{stat.split('_')[0]}_k{k}"] = apply_quantile_residual(quantile_payload[f"{stat}_k{k}"], score_df)
    score_df["raw_loss_score"] = score_df["clean_neg_loss"]
    table = pa.Table.from_pandas(score_df)
    pq.write_table(table, score_path)
    screening = json_load(ARTIFACTS_DIR / "results" / "screening" / f"{dataset}.json")
    meta_payload = {
        "cache_version": CACHE_VERSION,
        "dataset": dataset,
        "epsilon_target": epsilon,
        "seed": seed,
        "selected_transforms": selected_transforms,
        "stress_case": screening.get("stress_case", False),
        "calibration_stats": calibration_stats,
        "quantile_denom_floors": {
            key: float(value["denom_floor"]) for key, value in quantile_payload.items()
        },
        "num_pairs": int(len(score_df)),
        "screening_canonical_seed": screening.get("canonical_seed", SEEDS[0]),
    }
    json_dump(meta_payload, meta_path)
    stage_log("single_view_baselines", f"{combo_name(dataset, epsilon, seed)}_scores_meta.json", meta_payload)
    stage_log(
        "single_view_baselines",
        f"{combo_name(dataset, epsilon, seed)}_score_preview.json",
        {
            "dataset": dataset,
            "epsilon_target": epsilon,
            "seed": seed,
            "columns": score_df.columns.tolist(),
            "preview": score_df.head(5).to_dict(orient="records"),
        },
    )
    return {"score_path": str(score_path), "meta_path": str(meta_path)}


def proxy_lambda_selection(dataset: str, epsilon: float, seed: int, k: int = PRIMARY_K) -> dict:
    path = ARTIFACTS_DIR / "results" / "lambda_selection" / f"{combo_name(dataset, epsilon, seed)}.json"
    if path.exists():
        payload = json_load(path)
        if payload.get("cache_version") == CACHE_VERSION:
            return payload
    splits = split_indices(dataset, seed)
    device = choose_device()
    _, eval_tf, _ = dataset_transforms(dataset)
    proxy_indices = np.asarray(splits["public_proxy"])
    summaries = []
    public_train = splits["public_calibration"] + splits["public_proxy"]
    reference_path = train_non_private_model(dataset, "reference", public_train, seed + 1)
    reference_model = build_model(dataset).to(device)
    reference_model.load_state_dict(torch.load(reference_path, map_location="cpu", weights_only=False)["model_state"])
    reference_model.eval()
    train_ds_raw, _ = load_raw_dataset(dataset)
    selected = screen_transforms(dataset) or list(weak_transform_pool(dataset).keys())[:2]
    transform_pool = weak_transform_pool(dataset)
    proxy_indices = np.asarray(splits["public_proxy"])
    rng = np.random.default_rng(seed + 4242)
    fixed_proxy_order = rng.permutation(proxy_indices)
    chunk_size = len(fixed_proxy_order) // 6
    fixed_chunks = [fixed_proxy_order[i * chunk_size : (i + 1) * chunk_size].tolist() for i in range(6)]
    for split_id in range(3):
        members = fixed_chunks[2 * split_id]
        nonmembers = fixed_chunks[2 * split_id + 1]
        model_path = train_non_private_model(dataset, f"proxy_{split_id}", members, seed + 50 + split_id)
        proxy_model = build_model(dataset).to(device)
        proxy_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False)["model_state"])
        proxy_model.eval()
        calibration_rows = []
        for idx in splits["public_calibration"]:
            image, label = train_ds_raw[idx]
            clean = eval_tf(image)
            surrogate_logits = reference_model(clean.unsqueeze(0).to(device)).detach().cpu().numpy()
            surrogate_probs = softmax_np(surrogate_logits)
            logits = proxy_model(clean.unsqueeze(0).to(device)).detach().cpu().numpy()
            probs = softmax_np(logits)
            weak_losses = []
            weak_preds = []
            for i in range(k):
                view = apply_single_transform(clean, dataset, transform_pool[selected[i % len(selected)]])
                v_logits = proxy_model(view.unsqueeze(0).to(device)).detach().cpu().numpy()
                v_probs = softmax_np(v_logits)
                weak_losses.append(float(np.log(np.clip(v_probs[0, label], 1e-12, 1.0))))
                weak_preds.append(int(np.argmax(v_probs[0])))
            calibration_rows.append(
                {
                    "label": int(label),
                    "surrogate_loss": float(-np.log(np.clip(surrogate_probs[0, label], 1e-12, 1.0))),
                    "surrogate_entropy": float(entropy_from_probs(surrogate_probs)[0]),
                    "surrogate_max_prob": float(surrogate_probs.max()),
                    "ref_feature_norm": float(reference_model.feature_norm(clean.unsqueeze(0).to(device)).detach().cpu().numpy()[0]),
                    "clean_neg_loss": float(np.log(np.clip(probs[0, label], 1e-12, 1.0))),
                    f"worst_neg_loss_k{k}": float(np.min(weak_losses)),
                    f"mean_neg_loss_k{k}": float(np.mean(weak_losses)),
                    f"agree_rate_k{k}": float(np.mean(np.asarray(weak_preds) == weak_preds[0])),
                }
            )
        calibration_df = pd.DataFrame(calibration_rows)
        clean_q = fit_quantile_models(calibration_df, "clean_neg_loss")
        worst_q = fit_quantile_models(calibration_df, f"worst_neg_loss_k{k}")
        agree_q = fit_quantile_models(calibration_df, f"agree_rate_k{k}")
        rows = []
        for is_member, indices in [(1, members), (0, nonmembers)]:
            for idx in indices[:AUDIT_POOL_SIZE // 2]:
                image, label = train_ds_raw[idx]
                clean = eval_tf(image)
                surrogate_logits = reference_model(clean.unsqueeze(0).to(device)).detach().cpu().numpy()
                surrogate_probs = softmax_np(surrogate_logits)
                logits = proxy_model(clean.unsqueeze(0).to(device)).detach().cpu().numpy()
                probs = softmax_np(logits)
                weak_losses = []
                weak_preds = []
                for i in range(k):
                    view = apply_single_transform(clean, dataset, transform_pool[selected[i % len(selected)]])
                    v_logits = proxy_model(view.unsqueeze(0).to(device)).detach().cpu().numpy()
                    v_probs = softmax_np(v_logits)
                    weak_losses.append(float(np.log(np.clip(v_probs[0, label], 1e-12, 1.0))))
                    weak_preds.append(int(np.argmax(v_probs[0])))
                rows.append(
                    {
                        "is_member": is_member,
                        "label": int(label),
                        "surrogate_loss": float(-np.log(np.clip(surrogate_probs[0, label], 1e-12, 1.0))),
                        "surrogate_entropy": float(entropy_from_probs(surrogate_probs)[0]),
                        "surrogate_max_prob": float(surrogate_probs.max()),
                        "ref_feature_norm": float(reference_model.feature_norm(clean.unsqueeze(0).to(device)).detach().cpu().numpy()[0]),
                        "clean_neg_loss": float(np.log(np.clip(probs[0, label], 1e-12, 1.0))),
                        f"worst_neg_loss_k{k}": float(np.min(weak_losses)),
                        f"mean_neg_loss_k{k}": float(np.mean(weak_losses)),
                        f"agree_rate_k{k}": float(np.mean(np.asarray(weak_preds) == weak_preds[0])),
                    }
                )
        df = pd.DataFrame(rows)
        df["r_clean"] = apply_quantile_residual(clean_q, df)
        df["r_worst"] = apply_quantile_residual(worst_q, df)
        df["r_agree"] = apply_quantile_residual(agree_q, df)
        for lam in LAMBDA_GRID:
            scores = np.maximum.reduce([df["r_clean"].to_numpy(), df["r_worst"].to_numpy(), lam * df["r_agree"].to_numpy()])
            for order_seed in ORDERING_SEEDS:
                order = np.random.default_rng(1000 * split_id + order_seed).permutation(len(df))
                m = min(len(df), 200 // (1 + k))
                idx = order[:m]
                metrics = audit_metrics(df["is_member"].to_numpy()[idx], scores[idx], delta=DELTA)
                summaries.append({"split": split_id, "ordering_seed": order_seed, "lambda": lam, "query_budget": 200, **metrics})
    summary_df = pd.DataFrame(summaries)
    eps_summary = summary_df.groupby("lambda")["eps_lb"].median()
    auc_summary = summary_df.groupby("lambda")["auc"].median()
    result = {
        "cache_version": CACHE_VERSION,
        "selected_lambda_eps": safe_float(eps_summary.idxmax()),
        "selected_lambda_auc": safe_float(auc_summary.idxmax()),
        "proxy_split_sizes": {
            "members_per_split": int(len(fixed_chunks[0])),
            "nonmembers_per_split": int(len(fixed_chunks[1])),
        },
        "proxy_zero_eps_fraction_by_lambda": summary_df.groupby("lambda")["eps_lb"].apply(lambda s: float((s <= 0.0).mean())).to_dict(),
        "summary": summary_df.to_dict(orient="records"),
        "summary_by_lambda": summary_df.groupby("lambda")[["eps_lb", "auc"]].median().reset_index().to_dict(orient="records"),
    }
    json_dump(result, path)
    stage_log("weak_view_main", f"{combo_name(dataset, epsilon, seed)}_lambda_selection.json", result)
    return result


def evaluate_all_methods(dataset: str, epsilon: float, seed: int) -> dict:
    path = ARTIFACTS_DIR / "results" / "evaluations" / f"{combo_name(dataset, epsilon, seed)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json_load(path)
        if payload.get("cache_version") == CACHE_VERSION:
            return payload
    score_info = score_checkpoint(dataset, epsilon, seed, k_values=[2, 4])
    df = pq.read_table(ARTIFACTS_DIR / "scores" / f"{combo_name(dataset, epsilon, seed)}.parquet").to_pandas()
    meta = json_load(Path(score_info["meta_path"]))
    lambda_info = proxy_lambda_selection(dataset, epsilon, seed)
    lam = lambda_info["selected_lambda_eps"]
    methods = {
        "raw_loss": df["raw_loss_score"].to_numpy(),
        "clean_quantile": df["r_clean_k4"].to_numpy(),
        "mean_only": df["r_mean_k4"].to_numpy(),
        "structured": np.maximum.reduce([df["r_clean_k4"].to_numpy(), df["r_worst_k4"].to_numpy(), lam * df["r_agree_k4"].to_numpy()]),
        "no_agreement": np.maximum(df["r_clean_k4"].to_numpy(), df["r_worst_k4"].to_numpy()),
        "no_worst": np.maximum(df["r_clean_k4"].to_numpy(), lam * df["r_agree_k4"].to_numpy()),
        "k2_structured": np.maximum.reduce([df["r_clean_k2"].to_numpy(), df["r_worst_k2"].to_numpy(), lam * df["r_agree_k2"].to_numpy()]),
    }
    z_stats = meta["calibration_stats"]["k4"]
    z_df = pd.DataFrame(
        {
            "clean_neg_loss": (df["clean_neg_loss"] - z_stats["clean_neg_loss_mean"]) / z_stats["clean_neg_loss_std"],
            "worst_neg_loss_k4": (df["worst_neg_loss_k4"] - z_stats["worst_neg_loss_mean"]) / z_stats["worst_neg_loss_std"],
            "agree_rate_k4": (df["agree_rate_k4"] - z_stats["agree_rate_mean"]) / z_stats["agree_rate_std"],
        }
    )
    methods["no_residualization"] = np.maximum.reduce([z_df["clean_neg_loss"].to_numpy(), z_df["worst_neg_loss_k4"].to_numpy(), lam * z_df["agree_rate_k4"].to_numpy()])
    y = df["is_member"].to_numpy()
    kind = df["kind"].to_numpy()
    per_example = df[["pair_id", "is_member", "kind", "label"]].copy()
    for method, scores in methods.items():
        per_example[method] = scores
    per_example.to_csv(path.parent / f"{combo_name(dataset, epsilon, seed)}_per_example_scores.csv", index=False)
    matched_rows = []
    same_candidate_rows = []
    full_rows = []
    for method, scores in methods.items():
        query_cost = 1 if method in ["raw_loss", "clean_quantile"] else (3 if method == "k2_structured" else 5)
        full = audit_metrics(y, scores, delta=DELTA)
        full_rows.append({"method": method, "query_cost_per_candidate": query_cost, **full})
        for order_seed in ORDERING_SEEDS:
            order = np.random.default_rng(order_seed).permutation(len(df))
            for budget in QUERY_BUDGETS:
                m_matched = min(len(df), budget // query_cost)
                used = m_matched * query_cost
                if m_matched < 20:
                    continue
                idx = order[:m_matched]
                metrics = audit_metrics(y[idx], scores[idx], delta=DELTA)
                matched_rows.append(
                    {
                        "mode": "matched_budget",
                        "method": method,
                        "ordering_seed": order_seed,
                        "budget": budget,
                        "candidates_used": m_matched,
                        "queries_used": used,
                        **metrics,
                    }
                )
                same_m = min(len(df), budget)
                same_used = same_m * query_cost
                idx_same = order[:same_m]
                same_metrics = audit_metrics(y[idx_same], scores[idx_same], delta=DELTA)
                same_candidate_rows.append(
                    {
                        "mode": "same_candidate",
                        "method": method,
                        "ordering_seed": order_seed,
                        "budget": budget,
                        "candidates_used": same_m,
                        "queries_used": same_used,
                        **same_metrics,
                    }
                )
    full_df = pd.DataFrame(full_rows)
    matched_df = pd.DataFrame(matched_rows)
    same_candidate_df = pd.DataFrame(same_candidate_rows)
    kind_rows = []
    for method, scores in methods.items():
        for family in ["random", "fgsm"]:
            mask = kind == family
            metrics = audit_metrics(y[mask], scores[mask], delta=DELTA)
            kind_rows.append({"method": method, "kind": family, **metrics})
    result = {
        "cache_version": CACHE_VERSION,
        "full_metrics": full_df.to_dict(orient="records"),
        "matched_budget_metrics": matched_df.to_dict(orient="records"),
        "same_candidate_metrics": same_candidate_df.to_dict(orient="records"),
        "kind_metrics": kind_rows,
        "selected_lambda_eps": lam,
        "selected_lambda_auc": lambda_info["selected_lambda_auc"],
        "stress_case": meta["stress_case"],
    }
    json_dump(result, path)
    matched_df.to_csv(path.parent / f"{combo_name(dataset, epsilon, seed)}_matched_budget.csv", index=False)
    same_candidate_df.to_csv(path.parent / f"{combo_name(dataset, epsilon, seed)}_same_candidate.csv", index=False)
    stage_log(
        "ablations",
        f"{combo_name(dataset, epsilon, seed)}_evaluation_summary.json",
        {
            "dataset": dataset,
            "epsilon_target": epsilon,
            "seed": seed,
            "selected_lambda_eps": lam,
            "selected_lambda_auc": lambda_info["selected_lambda_auc"],
            "full_metrics": result["full_metrics"],
            "stress_case": meta["stress_case"],
        },
    )
    return result


def aggregate_results() -> dict:
    settings_rows = []
    matched_curve_rows = []
    same_candidate_rows = []
    lambda_rows = []
    kind_rows = []
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            per_seed = []
            training_rows = []
            for seed in SEEDS:
                ckpt = train_dp_target(dataset, epsilon, seed)
                evals = evaluate_all_methods(dataset, epsilon, seed)
                full = pd.DataFrame(evals["full_metrics"])
                full["seed"] = seed
                per_seed.append(full)
                training_rows.append(
                    {
                        "dataset": dataset,
                        "epsilon_target": epsilon,
                        "seed": seed,
                        "epsilon_realized": ckpt["epsilon_realized"],
                        "test_accuracy": ckpt["test_metrics"]["accuracy"],
                        "runtime_minutes": ckpt["runtime_minutes"],
                    }
                )
                matched = pd.DataFrame(evals["matched_budget_metrics"])
                matched["dataset"] = dataset
                matched["epsilon_target"] = epsilon
                matched["seed"] = seed
                matched_curve_rows.append(matched)
                same_candidate = pd.DataFrame(evals["same_candidate_metrics"])
                same_candidate["dataset"] = dataset
                same_candidate["epsilon_target"] = epsilon
                same_candidate["seed"] = seed
                same_candidate_rows.append(same_candidate)
                kind_df = pd.DataFrame(evals["kind_metrics"])
                kind_df["dataset"] = dataset
                kind_df["epsilon_target"] = epsilon
                kind_df["seed"] = seed
                kind_rows.append(kind_df)
                lambda_rows.append(
                    {
                        "dataset": dataset,
                        "epsilon_target": epsilon,
                        "seed": seed,
                        "selected_lambda_eps": evals["selected_lambda_eps"],
                        "selected_lambda_auc": evals["selected_lambda_auc"],
                    }
                )
            all_seed_df = pd.concat(per_seed, ignore_index=True)
            training_df = pd.DataFrame(training_rows)
            for method in ["raw_loss", "clean_quantile", "mean_only", "structured"]:
                row = all_seed_df[all_seed_df["method"] == method]
                settings_rows.append(
                    {
                        "dataset": dataset,
                        "epsilon_target": epsilon,
                        "method": method,
                        "eps_lb_mean": float(row["eps_lb"].mean()),
                        "eps_lb_std": float(row["eps_lb"].std(ddof=1)),
                        "auc_mean": float(row["auc"].mean()),
                        "auc_std": float(row["auc"].std(ddof=1)),
                        "tpr_1_mean": float(row["tpr@1%"].mean()),
                        "epsilon_realized_mean": float(training_df["epsilon_realized"].mean()),
                        "test_accuracy_mean": float(training_df["test_accuracy"].mean()),
                    }
                )
    settings_df = pd.DataFrame(settings_rows)
    matched_curves_df = pd.concat(matched_curve_rows, ignore_index=True)
    same_candidate_df = pd.concat(same_candidate_rows, ignore_index=True)
    lambda_df = pd.DataFrame(lambda_rows)
    kind_df = pd.concat(kind_rows, ignore_index=True)
    settings_df.to_csv(ARTIFACTS_DIR / "results" / "main_results.csv", index=False)
    matched_curves_df.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "figure1_matched_budget_eps.csv", index=False)
    matched_curves_df.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "figure2_matched_budget_auc.csv", index=False)
    same_candidate_df.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "same_candidate_curves.csv", index=False)
    lambda_df.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "lambda_selection.csv", index=False)
    kind_df.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "figure3_kind_breakdown.csv", index=False)
    training_df = collect_training_metrics()
    training_df.to_csv(ARTIFACTS_DIR / "results" / "training_metrics.csv", index=False)
    summary = build_summary_json(settings_df, matched_curves_df, same_candidate_df, lambda_df, training_df, kind_df)
    json_dump(summary, ROOT / "results.json")
    stage_log(
        "visualization",
        "aggregate_summary.json",
        {
            "settings_rows": len(settings_df),
            "matched_curve_rows": len(matched_curves_df),
            "same_candidate_rows": len(same_candidate_df),
            "lambda_rows": len(lambda_df),
            "kind_rows": len(kind_df),
        },
    )
    return summary


def collect_training_metrics() -> pd.DataFrame:
    rows = []
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                payload = train_dp_target(dataset, epsilon, seed)
                rows.append(
                    {
                        "dataset": dataset,
                        "epsilon_target": epsilon,
                        "seed": seed,
                        "epsilon_realized": payload["epsilon_realized"],
                        "test_accuracy": payload["test_metrics"]["accuracy"],
                        "runtime_minutes": payload["runtime_minutes"],
                    }
                )
    return pd.DataFrame(rows)


def _summarize_paired_curve(curves_df: pd.DataFrame, dataset: str, epsilon: float, metric: str) -> list[dict]:
    subset = curves_df[
        (curves_df["dataset"] == dataset)
        & (curves_df["epsilon_target"] == epsilon)
        & (curves_df["method"].isin(["clean_quantile", "structured"]))
    ].copy()
    per_seed = (
        subset.groupby(["seed", "method", "queries_used"])[metric]
        .mean()
        .reset_index()
        .pivot_table(index=["seed", "queries_used"], columns="method", values=metric)
        .dropna()
        .reset_index()
    )
    rows = []
    for query_budget, group in per_seed.groupby("queries_used"):
        diff = group["structured"].to_numpy() - group["clean_quantile"].to_numpy()
        stats = bootstrap_mean_and_ci(diff, n_boot=2000, seed=123 + int(query_budget))
        rows.append(
            {
                "queries_used": int(query_budget),
                "paired_seed_differences": [float(x) for x in diff.tolist()],
                "difference_mean": stats["mean"],
                "difference_std": stats["std"],
                "difference_ci95": stats["ci95"],
            }
        )
    return sorted(rows, key=lambda row: row["queries_used"])


def _find_sustained_regions(curve_rows: list[dict], min_width: int = 120) -> list[dict]:
    if not curve_rows:
        return []
    regions = []
    active = []
    for row in curve_rows:
        if row["difference_mean"] > 0.0:
            if active and row["queries_used"] != active[-1]["queries_used"] + 40:
                active = []
            active.append(row)
        else:
            if active:
                width = active[-1]["queries_used"] - active[0]["queries_used"]
                if width >= min_width:
                    regions.append(active)
            active = []
    if active:
        width = active[-1]["queries_used"] - active[0]["queries_used"]
        if width >= min_width:
            regions.append(active)
    summarized = []
    for region in regions:
        diffs = np.array([row["difference_mean"] for row in region], dtype=float)
        ci_lows = [row["difference_ci95"][0] for row in region]
        ci_highs = [row["difference_ci95"][1] for row in region]
        summarized.append(
            {
                "budget_start": int(region[0]["queries_used"]),
                "budget_end": int(region[-1]["queries_used"]),
                "width_queries": int(region[-1]["queries_used"] - region[0]["queries_used"]),
                "mean_difference_over_region": float(diffs.mean()),
                "min_ci95_low": float(min(ci_lows)),
                "max_ci95_high": float(max(ci_highs)),
                "ci_positive_throughout": bool(all(low > 0.0 for low in ci_lows)),
            }
        )
    return summarized


def _proxy_lambda_diagnostics(dataset: str, epsilon: float) -> dict:
    zero_eps_rates = []
    lambda_choices = []
    for seed in SEEDS:
        payload = json_load(ARTIFACTS_DIR / "results" / "lambda_selection" / f"{combo_name(dataset, epsilon, seed)}.json")
        lambda_choices.append(payload["selected_lambda_eps"])
        seed_summary = pd.DataFrame(payload["summary"])
        if not seed_summary.empty:
            zero_eps_rates.append(float((seed_summary["eps_lb"] <= 0.0).mean()))
    zero_eps_mean = float(np.mean(zero_eps_rates)) if zero_eps_rates else 0.0
    collapsed_to_zero_rate = float(np.mean(np.asarray(lambda_choices) == 0.0)) if lambda_choices else 0.0
    diagnosis = []
    if collapsed_to_zero_rate >= 0.5:
        diagnosis.append("selected_lambda_eps collapses to 0.0 in most seeds")
    if zero_eps_mean >= 0.5:
        diagnosis.append("proxy audits often produce eps_lb = 0, weakening lambda discrimination")
    if not diagnosis:
        diagnosis.append("proxy lambda selection produced non-degenerate medians in most seeds")
    return {
        "mean_zero_eps_fraction": zero_eps_mean,
        "selected_lambda_eps_values": [float(v) for v in lambda_choices],
        "collapsed_to_zero_rate": collapsed_to_zero_rate,
        "diagnosis": diagnosis,
    }


def build_summary_json(
    settings_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    same_candidate_df: pd.DataFrame,
    lambda_df: pd.DataFrame,
    training_df: pd.DataFrame,
    kind_df: pd.DataFrame,
) -> dict:
    results = []
    overall_negative = True
    for (dataset, epsilon), group in settings_df.groupby(["dataset", "epsilon_target"]):
        structured = group[group["method"] == "structured"].iloc[0]
        clean = group[group["method"] == "clean_quantile"].iloc[0]
        mean_only = group[group["method"] == "mean_only"].iloc[0]
        diff_values = []
        for seed in SEEDS:
            evals = json_load(ARTIFACTS_DIR / "results" / "evaluations" / f"{combo_name(dataset, epsilon, seed)}.json")
            full = pd.DataFrame(evals["full_metrics"]).set_index("method")
            diff_values.append(full.loc["structured", "eps_lb"] - full.loc["clean_quantile", "eps_lb"])
        diff_values = np.asarray(diff_values)
        full_pool_stats = bootstrap_mean_and_ci(diff_values)
        eps_curve = _summarize_paired_curve(curves_df, dataset, epsilon, "eps_lb")
        auc_curve = _summarize_paired_curve(curves_df, dataset, epsilon, "auc")
        sustained_regions = _find_sustained_regions(eps_curve, min_width=120)
        if sustained_regions or float(diff_values.mean()) > 0.0:
            overall_negative = False
        same_candidate_summary = _summarize_paired_curve(same_candidate_df, dataset, epsilon, "eps_lb")
        kind_subset = kind_df[(kind_df["dataset"] == dataset) & (kind_df["epsilon_target"] == epsilon)]
        results.append(
            {
                "dataset": dataset,
                "epsilon_target": epsilon,
                "structured_eps_lb_mean": float(structured["eps_lb_mean"]),
                "structured_eps_lb_std": float(structured["eps_lb_std"]),
                "clean_quantile_eps_lb_mean": float(clean["eps_lb_mean"]),
                "clean_quantile_eps_lb_std": float(clean["eps_lb_std"]),
                "mean_only_eps_lb_mean": float(mean_only["eps_lb_mean"]),
                "mean_only_eps_lb_std": float(mean_only["eps_lb_std"]),
                "difference_mean": full_pool_stats["mean"],
                "difference_std": full_pool_stats["std"],
                "difference_ci95": full_pool_stats["ci95"],
                "lambda_agreement_rate": float(
                    (
                        lambda_df[(lambda_df["dataset"] == dataset) & (lambda_df["epsilon_target"] == epsilon)]["selected_lambda_eps"]
                        == lambda_df[(lambda_df["dataset"] == dataset) & (lambda_df["epsilon_target"] == epsilon)]["selected_lambda_auc"]
                    ).mean()
                ),
                "matched_budget_eps_curve_difference": eps_curve,
                "matched_budget_auc_curve_difference": auc_curve,
                "matched_budget_sustained_positive_regions": sustained_regions,
                "same_candidate_eps_curve_difference": same_candidate_summary,
                "proxy_lambda_diagnostics": _proxy_lambda_diagnostics(dataset, epsilon),
                "kind_breakdown": kind_subset[["method", "kind", "eps_lb", "auc"]].to_dict(orient="records"),
                "negative_result_for_setting": bool(full_pool_stats["mean"] <= 0.0 and len(sustained_regions) == 0),
                "interpretation": (
                    "Negative result: structured weak-view auditing does not improve over the clean quantile baseline "
                    "for this setting under matched total query accounting."
                    if full_pool_stats["mean"] <= 0.0 and len(sustained_regions) == 0
                    else "Mixed result: inspect matched-budget regions and seed-level intervals before claiming improvement."
                ),
            }
        )
    return {
        "config": config_dict(),
        "environment": json_load(ARTIFACTS_DIR / "environment" / "system_info.json"),
        "training": training_df.to_dict(orient="records"),
        "study_outcome": "negative_result" if overall_negative else "mixed_or_partially_positive",
        "reporting_status": {
            "final_status": "completed_negative_result" if overall_negative else "completed_mixed_result",
            "primary_metric": "empirical epsilon lower bound at delta=1e-5",
            "optional_plan_items_skipped": [
                "true-label margin sensitivity",
                "K=8 weak-view ablation",
            ],
            "protocol_deviations": [
                "Environment differs from the preregistered stack; see environment.version_note.",
                "proposal.md still states 3 seeds, but the executed plan and results use 4 seeds [7, 17, 27, 37].",
            ],
        },
        "settings": results,
    }


def generate_figures() -> None:
    sns.set_theme(style="whitegrid")
    settings_df = pd.read_csv(ARTIFACTS_DIR / "results" / "main_results.csv")
    curves_df = pd.read_csv(ARTIFACTS_DIR / "figures" / "source_data" / "figure1_matched_budget_eps.csv")
    lambda_df = pd.read_csv(ARTIFACTS_DIR / "figures" / "source_data" / "lambda_selection.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (dataset, epsilon) in zip(axes.flatten(), [(d, e) for d in DATASET_CONFIGS for e in EPSILON_TARGETS]):
        subset = curves_df[(curves_df["dataset"] == dataset) & (curves_df["epsilon_target"] == epsilon)]
        plot_df = subset[subset["method"].isin(["clean_quantile", "mean_only", "structured"])]
        agg = plot_df.groupby(["method", "queries_used"])["eps_lb"].agg(["mean", "std"]).reset_index()
        for method in ["clean_quantile", "mean_only", "structured"]:
            data = agg[agg["method"] == method]
            ax.plot(data["queries_used"], data["mean"], label=method)
            ax.fill_between(data["queries_used"], data["mean"] - data["std"], data["mean"] + data["std"], alpha=0.2)
        ax.set_title(f"{dataset} eps~{int(epsilon)}")
        ax.set_xlabel("Consumed Queries")
        ax.set_ylabel("Empirical Epsilon LB")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_matched_budget_eps.png", dpi=200)
    fig.savefig(FIGURES_DIR / "figure1_matched_budget_eps.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (dataset, epsilon) in zip(axes.flatten(), [(d, e) for d in DATASET_CONFIGS for e in EPSILON_TARGETS]):
        subset = curves_df[(curves_df["dataset"] == dataset) & (curves_df["epsilon_target"] == epsilon)]
        plot_df = subset[subset["method"].isin(["clean_quantile", "mean_only", "structured"])]
        agg = plot_df.groupby(["method", "queries_used"])["auc"].agg(["mean", "std"]).reset_index()
        for method in ["clean_quantile", "mean_only", "structured"]:
            data = agg[agg["method"] == method]
            ax.plot(data["queries_used"], data["mean"], label=method)
            ax.fill_between(data["queries_used"], data["mean"] - data["std"], data["mean"] + data["std"], alpha=0.2)
        ax.set_title(f"{dataset} eps~{int(epsilon)}")
        ax.set_xlabel("Consumed Queries")
        ax.set_ylabel("ROC AUC")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_matched_budget_auc.png", dpi=200)
    fig.savefig(FIGURES_DIR / "figure2_matched_budget_auc.pdf")
    plt.close(fig)

    ablation_rows = []
    kind_rows = []
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                evals = json_load(ARTIFACTS_DIR / "results" / "evaluations" / f"{combo_name(dataset, epsilon, seed)}.json")
                full = pd.DataFrame(evals["full_metrics"])
                full["dataset"] = dataset
                full["epsilon_target"] = epsilon
                ablation_rows.append(full)
                kind_df = pd.DataFrame(evals["kind_metrics"])
                kind_df["dataset"] = dataset
                kind_df["epsilon_target"] = epsilon
                kind_rows.append(kind_df)
    ablation_df = pd.concat(ablation_rows, ignore_index=True)
    kind_df = pd.concat(kind_rows, ignore_index=True)
    ablation_df.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "figure3_ablations.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=ablation_df[ablation_df["method"].isin(["structured", "no_agreement", "no_worst", "no_residualization", "k2_structured"])],
        x="method",
        y="eps_lb",
        hue="epsilon_target",
        ax=axes[0],
    )
    axes[0].set_title("Ablations")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=kind_df[kind_df["method"].isin(["clean_quantile", "mean_only", "structured"])], x="method", y="eps_lb", hue="kind", ax=axes[1])
    axes[1].set_title("Canary Family Breakdown")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_ablations.png", dpi=200)
    fig.savefig(FIGURES_DIR / "figure3_ablations.pdf")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    proxy_rows = []
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                payload = json_load(ARTIFACTS_DIR / "results" / "lambda_selection" / f"{combo_name(dataset, epsilon, seed)}.json")
                proxy_rows.extend([{**row, "dataset": dataset, "epsilon_target": epsilon, "seed": seed} for row in payload["summary"]])
    proxy_df = pd.DataFrame(proxy_rows)
    agg = proxy_df.groupby("lambda")[["eps_lb", "auc"]].median().reset_index()
    agg.to_csv(ARTIFACTS_DIR / "figures" / "source_data" / "figure4_lambda_selection.csv", index=False)
    ax1.plot(agg["lambda"], agg["eps_lb"], marker="o", label="Proxy eps LB")
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Proxy Epsilon LB")
    ax2 = ax1.twinx()
    ax2.plot(agg["lambda"], agg["auc"], marker="s", color="tab:orange", label="Proxy AUC")
    ax2.set_ylabel("Proxy AUC")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_lambda_selection.png", dpi=200)
    fig.savefig(FIGURES_DIR / "figure4_lambda_selection.pdf")
    plt.close(fig)
    stage_log(
        "visualization",
        "figure_inventory.json",
        {
            "figures": [
                "figure1_matched_budget_eps",
                "figure2_matched_budget_auc",
                "figure3_ablations",
                "figure4_lambda_selection",
            ],
            "source_csvs": [
                "figure1_matched_budget_eps.csv",
                "figure2_matched_budget_auc.csv",
                "figure3_ablations.csv",
                "figure3_kind_breakdown.csv",
                "figure4_lambda_selection.csv",
                "same_candidate_curves.csv",
                "lambda_selection.csv",
            ],
            "settings_rows": len(settings_df),
            "curve_rows": len(curves_df),
        },
    )


def write_environment_info() -> None:
    ensure_dirs()
    python_bin = "/home/zz865/.conda/envs/ar/bin/python"
    pip_freeze = subprocess.run([python_bin, "-m", "pip", "freeze"], check=True, capture_output=True, text=True).stdout.splitlines()
    nvidia = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True).stdout
    nproc = subprocess.run(["nproc"], check=True, capture_output=True, text=True).stdout.strip()
    free_h = subprocess.run(["free", "-h"], check=True, capture_output=True, text=True).stdout
    info = {
        "python": subprocess.run([python_bin, "--version"], check=True, capture_output=True, text=True).stdout.strip(),
        "python_bin": python_bin,
        "pip_freeze": pip_freeze,
        "nvidia_smi": nvidia,
        "nproc": nproc,
        "free_h": free_h,
        "version_note": "Plan requested Python 3.10 with torch==2.2.*, torchvision==0.17.*, opacus==1.4.*. This run used the preexisting /home/zz865/.conda/envs/ar environment instead; notable deviations include Python 3.11, opacus 1.5.4, NumPy 2.x, and a CIFAR-10 preprocessing correction to ImageNet normalization for the pretrained frozen ResNet-18 backbone.",
    }
    json_dump(info, ARTIFACTS_DIR / "environment" / "system_info.json")
    stage_log("environment", "system_info.json", info)


def run_all() -> dict:
    write_environment_info()
    write_stage_result(
        "environment",
        {
            "experiment": "environment",
            "status": "completed",
            "artifacts": ["artifacts/environment/system_info.json"],
            "deviation_documented": True,
        },
    )
    for dataset in DATASET_CONFIGS:
        for seed in SEEDS:
            split_indices(dataset, seed)
            build_audit_pool(dataset, seed)
        screen_transforms(dataset)
    write_stage_result(
        "data_preparation",
        {
            "experiment": "data_preparation",
            "status": "completed",
            "datasets": list(DATASET_CONFIGS),
            "seeds": SEEDS,
            "artifacts": [
                "artifacts/results/splits",
                "artifacts/results/audit_pairs",
                "artifacts/results/screening",
            ],
        },
    )
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                train_dp_target(dataset, epsilon, seed)
    training_df = collect_training_metrics()
    write_stage_result(
        "train_targets",
        {
            "experiment": "train_targets",
            "status": "completed",
            "metrics": training_df.groupby(["dataset", "epsilon_target"])[["test_accuracy", "epsilon_realized", "runtime_minutes"]].mean().reset_index().to_dict(orient="records"),
            "artifacts": ["artifacts/checkpoints", "artifacts/results/training_metrics.csv"],
        },
    )
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                score_checkpoint(dataset, epsilon, seed, k_values=[2, 4])
                evaluate_all_methods(dataset, epsilon, seed)
    write_stage_result(
        "single_view_baselines",
        {
            "experiment": "single_view_baselines",
            "status": "completed",
            "artifacts": ["artifacts/scores", "artifacts/results/evaluations/*_per_example_scores.csv"],
        },
    )
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                proxy_lambda_selection(dataset, epsilon, seed)
    write_stage_result(
        "weak_view_main",
        {
            "experiment": "weak_view_main",
            "status": "completed",
            "artifacts": ["artifacts/results/lambda_selection"],
            "optional_items_not_run": ["true-label margin sensitivity"],
        },
    )
    for dataset in DATASET_CONFIGS:
        for epsilon in EPSILON_TARGETS:
            for seed in SEEDS:
                evaluate_all_methods(dataset, epsilon, seed)
    write_stage_result(
        "ablations",
        {
            "experiment": "ablations",
            "status": "completed",
            "artifacts": ["artifacts/results/evaluations"],
            "optional_items_not_run": ["K=8 weak-view ablation"],
        },
    )
    summary = aggregate_results()
    generate_figures()
    write_stage_result(
        "visualization",
        {
            "experiment": "visualization",
            "status": "completed",
            "artifacts": ["results.json", "figures", "artifacts/figures/source_data"],
        },
    )
    clear_stage_skip("weak_view_main")
    clear_stage_skip("ablations")
    return summary
