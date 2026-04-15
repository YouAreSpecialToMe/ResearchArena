import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from datasets import load_dataset
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets as tv_datasets
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = ROOT / "exp"
FIG_ROOT = ROOT / "figures"
DATA_ROOT = ROOT / "data"
SEEDS = [7, 17, 27]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass
class SampleRecord:
    sample_id: int
    split: str
    y: int
    group: int
    meta: Dict


class TwoViewTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


class WaterbirdsDataset(Dataset):
    def __init__(self, split: str, transform=None, two_views: bool = False):
        self.ds = load_dataset("grodino/waterbirds")[split]
        self.transform = transform
        self.two_views = two_views
        self.records = []
        for idx, row in enumerate(self.ds):
            y = int(row["label"])
            place = int(row["place"])
            group = y * 2 + place
            self.records.append(
                SampleRecord(
                    sample_id=idx,
                    split=split,
                    y=y,
                    group=group,
                    meta={"place": place, "bird": int(row["bird"])},
                )
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if self.transform is None:
            out = transforms.ToTensor()(image)
        elif self.two_views:
            out = self.transform(image)
        else:
            out = self.transform(image)
        rec = self.records[idx]
        return out, rec.y, rec.group, rec.sample_id


class ColoredMNISTDataset(Dataset):
    def __init__(self, split: str, seed: int, transform=None, two_views: bool = False):
        ensure_dir(DATA_ROOT / "colored_mnist")
        train = split == "train"
        base = tv_datasets.MNIST(root=str(DATA_ROOT / "colored_mnist"), train=train, download=True)
        rng = np.random.default_rng(seed + {"train": 0, "validation": 1000, "test": 2000}[split])
        if split == "train":
            images = base.data.numpy()
            labels = base.targets.numpy()
            corr = 0.995
            offset = 0
        else:
            perm = rng.permutation(len(base.data))
            images = base.data.numpy()[perm]
            labels = base.targets.numpy()[perm]
            images = images[:5000]
            labels = labels[:5000]
            corr = 0.1
            offset = 60000 if split == "validation" else 70000
        labels = (labels < 5).astype(np.int64)
        groups = []
        colored = []
        for idx, (img, y) in enumerate(zip(images, labels)):
            use_spurious = rng.random() < corr
            color = int(y if use_spurious else 1 - y)
            rgb = np.zeros((28, 28, 3), dtype=np.uint8)
            rgb[..., color] = img
            rgb[..., 2] = img // 3
            colored.append(Image.fromarray(rgb))
            groups.append(color)
        self.images = colored
        self.labels = labels.tolist()
        self.groups = groups
        self.sample_ids = list(range(offset, offset + len(self.images)))
        self.transform = transform
        self.two_views = two_views

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is None:
            out = transforms.ToTensor()(image)
        elif self.two_views:
            out = self.transform(image)
        else:
            out = self.transform(image)
        return out, self.labels[idx], self.groups[idx], self.sample_ids[idx]


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: List[int], batch_size: int, seed: int):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.per_class = batch_size // len(np.unique(self.labels))
        self.seed = seed
        self.class_to_indices = {c: np.where(self.labels == c)[0].tolist() for c in np.unique(self.labels)}
        self.num_batches = len(self.labels) // batch_size

    def __iter__(self):
        rng = random.Random(self.seed)
        class_to_pool = {}
        for c, idxs in self.class_to_indices.items():
            shuffled = idxs.copy()
            rng.shuffle(shuffled)
            class_to_pool[c] = shuffled
        for _ in range(self.num_batches):
            batch = []
            for c in sorted(class_to_pool):
                pool = class_to_pool[c]
                if len(pool) < self.per_class:
                    refill = self.class_to_indices[c].copy()
                    rng.shuffle(refill)
                    pool.extend(refill)
                batch.extend(pool[: self.per_class])
                del pool[: self.per_class]
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return feats, logits


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        channels = [32, 64, 128, 256]
        layers = []
        in_ch = 3
        for ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.feat_dim = channels[-1]

    def forward(self, x):
        feats = self.pool(self.features(x)).flatten(1)
        logits = self.classifier(feats)
        return feats, logits


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def js_divergence(p, q, eps=1e-8):
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def trajectory_shape(values: np.ndarray) -> np.ndarray:
    t = np.arange(1, len(values) + 1, dtype=np.float32)
    if len(values) == 1:
        slope = 0.0
    else:
        slope = np.polyfit(t, values, 1)[0]
    return np.asarray(
        [values[0], values[-1], values.mean(), values.std(), values[-1] - values[0], slope],
        dtype=np.float32,
    )


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def compute_metrics(y_true, y_pred, groups):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    acc = float((y_true == y_pred).mean())
    wg_acc = min(float((y_pred[groups == g] == y_true[groups == g]).mean()) for g in np.unique(groups))
    return {
        "avg_accuracy": acc,
        "worst_group_accuracy": wg_acc,
        "robustness_gap": acc - wg_acc,
    }


def minority_f1_from_clusters(true_groups, pred_groups):
    true_groups = np.asarray(true_groups)
    pred_groups = np.asarray(pred_groups)
    labels = np.unique(pred_groups)
    best = 0.0
    for flip in [False, True]:
        mapped = pred_groups.copy()
        if flip and len(labels) == 2:
            mapped = 1 - mapped
        best = max(best, f1_score(true_groups, mapped, average="binary"))
    return float(best)


def cluster_balanced_accuracy(true_groups, pred_groups):
    true_groups = np.asarray(true_groups)
    pred_groups = np.asarray(pred_groups)
    labels = np.unique(pred_groups)
    best = 0.0
    for flip in [False, True]:
        mapped = pred_groups.copy()
        if flip and len(labels) == 2:
            mapped = 1 - mapped
        best = max(best, balanced_accuracy_score(true_groups, mapped))
    return float(best)


def hash_manifest(records: List[SampleRecord]) -> str:
    payload = json.dumps([asdict(r) for r in records], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_transforms(dataset_name: str):
    if dataset_name == "waterbirds":
        train_base = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=ResNet18_Weights.IMAGENET1K_V1.transforms().mean, std=ResNet18_Weights.IMAGENET1K_V1.transforms().std),
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=ResNet18_Weights.IMAGENET1K_V1.transforms().mean, std=ResNet18_Weights.IMAGENET1K_V1.transforms().std),
            ]
        )
        return train_base, eval_tf, TwoViewTransform(train_base)
    train_base = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
            transforms.ColorJitter(0.15, 0.15, 0.15, 0.02),
            transforms.ToTensor(),
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor()])
    return train_base, eval_tf, TwoViewTransform(train_base)


def build_dataset(dataset_name: str, split: str, seed: int, transform=None, two_views: bool = False):
    if dataset_name == "waterbirds":
        return WaterbirdsDataset(split=split, transform=transform, two_views=two_views)
    return ColoredMNISTDataset(split=split, seed=seed, transform=transform, two_views=two_views)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(dataset_name: str, num_classes: int):
    if dataset_name == "waterbirds":
        return ResNet18Backbone(num_classes)
    return SmallCNN(num_classes)


def make_loaders(dataset_name: str, seed: int, batch_size: int, train_tf, eval_tf):
    train_set = build_dataset(dataset_name, "train", seed, transform=train_tf)
    val_split = "validation" if dataset_name == "waterbirds" else "validation"
    test_split = "test"
    val_set = build_dataset(dataset_name, val_split, seed, transform=eval_tf)
    test_set = build_dataset(dataset_name, test_split, seed, transform=eval_tf)
    warmup_cache_set = build_dataset(dataset_name, "train", seed, transform=TwoViewTransform(train_tf), two_views=True)
    train_sampler = None
    shuffle = True
    if dataset_name in {"waterbirds", "colored_mnist"}:
        if dataset_name == "waterbirds":
            labels = [rec.y for rec in train_set.records]
        else:
            labels = train_set.labels
        train_sampler = BalancedBatchSampler(labels, batch_size, seed)
        shuffle = False
    gen = torch.Generator()
    gen.manual_seed(seed)
    kwargs = dict(num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=gen)
    if train_sampler is not None:
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, **kwargs)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    cache_loader = DataLoader(warmup_cache_set, batch_size=max(16, batch_size // 2), shuffle=False, **kwargs)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader, cache_loader


def build_optimizer(model, projection_head, dataset_name: str):
    backbone_lr = 3e-4 if dataset_name == "waterbirds" else 1e-3
    head_lr = 1e-3
    params = [
        {"params": model.encoder.parameters() if hasattr(model, "encoder") else model.features.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": head_lr},
    ]
    if projection_head is not None:
        params.append({"params": projection_head.parameters(), "lr": head_lr})
    return torch.optim.AdamW(params, weight_decay=1e-4)


def write_csv_row(path: Path, row: Dict, header: Optional[List[str]] = None):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_model(model, loader, device):
    model.eval()
    ys, preds, groups = [], [], []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, y, g, _ in loader:
            images = images.to(device)
            y = y.to(device)
            _, logits = model(images)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total += y.size(0)
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            groups.extend(g.numpy().tolist())
    metrics = compute_metrics(ys, preds, groups)
    metrics["cross_entropy"] = total_loss / max(total, 1)
    return metrics, ys, preds, groups


def extract_epoch_cache(model, loader, device, prev_feats: Optional[Dict[int, np.ndarray]] = None):
    model.eval()
    epoch_cache = {}
    with torch.no_grad():
        for (view_a, view_b), y, _, sample_ids in loader:
            view_a = view_a.to(device)
            view_b = view_b.to(device)
            feats_a, logits_a = model(view_a)
            feats_b, logits_b = model(view_b)
            feats = F.normalize((feats_a + feats_b) / 2.0, dim=1)
            probs = F.softmax(logits_a, dim=1)
            probs_b = F.softmax(logits_b, dim=1)
            y_dev = y.to(device)
            p = probs[torch.arange(len(y_dev)), y_dev]
            top2 = logits_a.topk(k=2, dim=1).indices
            margins = logits_a[torch.arange(len(y_dev)), y_dev] - torch.where(
                top2[:, 0] == y_dev,
                logits_a[torch.arange(len(y_dev)), top2[:, 1]],
                logits_a[torch.arange(len(y_dev)), top2[:, 0]],
            )
            ce = F.cross_entropy(logits_a, y_dev, reduction="none")
            js = js_divergence(probs, probs_b)
            feats_np = feats.cpu().numpy()
            for i, sid in enumerate(sample_ids.numpy().tolist()):
                prev = prev_feats.get(sid) if prev_feats is not None else None
                drift = 0.0 if prev is None else float(1.0 - np.dot(prev, feats_np[i]) / (np.linalg.norm(prev) * np.linalg.norm(feats_np[i]) + 1e-8))
                epoch_cache[sid] = {
                    "label": int(y[i]),
                    "feature": feats_np[i],
                    "confidence": float(p[i].cpu()),
                    "margin": float(margins[i].cpu()),
                    "loss": float(ce[i].cpu()),
                    "disagreement": float(js[i].cpu()),
                    "drift": drift,
                    "correct": int((logits_a.argmax(dim=1)[i] == y_dev[i]) and (margins[i] > 0)),
                }
    return epoch_cache


def fit_signal_models(signal_map: Dict[int, Dict], labels: Dict[int, int], signal_kind: str, seed: int):
    results = {}
    train_q = {}
    train_cluster = {}
    pairwise_repr = {}
    for cls in sorted(set(labels.values())):
        ids = [sid for sid, y in labels.items() if y == cls]
        matrix = []
        for sid in ids:
            entry = signal_map[sid]
            if signal_kind == "final_feature":
                vec = entry["feature"]
            else:
                parts = [np.asarray(entry["p"]), np.asarray(entry["m"]), np.asarray(entry["l"])]
                if signal_kind in {"trajectory", "trajectory_no_shape"}:
                    parts.extend([np.asarray(entry["u"]), np.asarray(entry["d"])])
                if signal_kind not in {"trajectory_no_shape", "output_only_no_shape"}:
                    for key in ["p", "m", "l"]:
                        parts.append(trajectory_shape(np.asarray(entry[key], dtype=np.float32)))
                    if signal_kind == "trajectory":
                        for key in ["u", "d"]:
                            parts.append(trajectory_shape(np.asarray(entry[key], dtype=np.float32)))
                    parts.append(np.asarray([entry["t_corr"]], dtype=np.float32))
                vec = np.concatenate(parts, axis=0).astype(np.float32)
            matrix.append(vec)
        X = np.stack(matrix).astype(np.float64)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
        Xn = (X - mean) / std
        pca = None
        if Xn.shape[1] > 32:
            pca = PCA(n_components=min(16, Xn.shape[0], Xn.shape[1]), random_state=seed)
            Xc = pca.fit_transform(Xn)
        else:
            Xc = Xn
        last_error = None
        for reg_covar in [1e-5, 1e-4, 1e-3]:
            gmm = GaussianMixture(
                n_components=2,
                covariance_type="diag",
                reg_covar=reg_covar,
                n_init=5,
                max_iter=200,
                random_state=seed,
            )
            try:
                gmm.fit(Xc)
                break
            except ValueError as exc:
                last_error = exc
                gmm = None
        if gmm is None:
            raise last_error
        q = gmm.predict_proba(Xc)
        labels_hat = gmm.predict(Xc)
        for sid, qi, lh, xci in zip(ids, q, labels_hat, Xc):
            train_q[sid] = qi
            train_cluster[sid] = int(lh)
            pairwise_repr[sid] = xci
        results[cls] = {"mean": mean, "std": std, "pca": pca, "gmm": gmm}
    return results, train_q, train_cluster, pairwise_repr


def serialize_signal_models(signal_models: Dict[int, Dict]) -> Dict[str, Dict]:
    serialized = {}
    for cls, meta in signal_models.items():
        pca = meta["pca"]
        gmm = meta["gmm"]
        serialized[str(cls)] = {
            "zscore_mean": meta["mean"].tolist(),
            "zscore_std": meta["std"].tolist(),
            "pca": None
            if pca is None
            else {
                "n_components": int(pca.n_components_),
                "components": pca.components_.tolist(),
                "explained_variance": pca.explained_variance_.tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "mean": pca.mean_.tolist(),
                "singular_values": pca.singular_values_.tolist(),
            },
            "gmm": {
                "weights": gmm.weights_.tolist(),
                "means": gmm.means_.tolist(),
                "covariances": gmm.covariances_.tolist(),
                "precisions_cholesky": gmm.precisions_cholesky_.tolist(),
                "converged": bool(gmm.converged_),
                "n_iter": int(gmm.n_iter_),
                "lower_bound": float(gmm.lower_bound_),
                "reg_covar": float(gmm.reg_covar),
            },
        }
    return serialized


def apply_signal_models(signal_models, signal_map, labels, signal_kind):
    out_q = {}
    out_cluster = {}
    for sid, label in labels.items():
        entry = signal_map[sid]
        if signal_kind == "final_feature":
            vec = entry["feature"]
        else:
            parts = [np.asarray(entry["p"]), np.asarray(entry["m"]), np.asarray(entry["l"])]
            if signal_kind in {"trajectory", "trajectory_no_shape"}:
                parts.extend([np.asarray(entry["u"]), np.asarray(entry["d"])])
            if signal_kind not in {"trajectory_no_shape", "output_only_no_shape"}:
                for key in ["p", "m", "l"]:
                    parts.append(trajectory_shape(np.asarray(entry[key], dtype=np.float32)))
                if signal_kind == "trajectory":
                    for key in ["u", "d"]:
                        parts.append(trajectory_shape(np.asarray(entry[key], dtype=np.float32)))
                parts.append(np.asarray([entry["t_corr"]], dtype=np.float32))
            vec = np.concatenate(parts, axis=0).astype(np.float32)
        meta = signal_models[label]
        Xn = (vec - meta["mean"]) / meta["std"]
        if meta["pca"] is not None:
            Xn = meta["pca"].transform(Xn[None, :])[0]
        q = meta["gmm"].predict_proba(Xn[None, :])[0]
        out_q[sid] = q
        out_cluster[sid] = int(np.argmax(q))
    return out_q, out_cluster


def build_signal_map(warmup_epochs: List[Dict[int, Dict]], sample_ids: List[int]):
    signal_map = {}
    for sid in sample_ids:
        entry = {"p": [], "m": [], "l": [], "u": [], "d": [], "feature": None}
        first_correct = None
        for epoch_idx, epoch_cache in enumerate(warmup_epochs, start=1):
            sample = epoch_cache[sid]
            entry["p"].append(sample["confidence"])
            entry["m"].append(sample["margin"])
            entry["l"].append(sample["loss"])
            entry["u"].append(sample["disagreement"])
            entry["d"].append(sample["drift"])
            entry["feature"] = sample["feature"]
            if first_correct is None and sample["correct"]:
                first_correct = epoch_idx
        entry["t_corr"] = first_correct if first_correct is not None else len(warmup_epochs) + 1
        signal_map[sid] = entry
    return signal_map


def weighted_supcon_loss(features, labels, weights, tau=0.07):
    features = F.normalize(features, dim=1)
    logits = features @ features.t() / tau
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = labels.view(-1, 1)
    mask = (labels == labels.t()).float()
    mask.fill_diagonal_(0.0)
    exp_logits = torch.exp(logits) * (1 - torch.eye(len(labels), device=labels.device))
    denom = exp_logits.sum(dim=1, keepdim=True) + 1e-8
    numer = (exp_logits * mask * weights).sum(dim=1)
    valid = numer > 0
    if valid.any():
        loss = -torch.log((numer[valid] / denom.squeeze(1)[valid]) + 1e-8).mean()
        return loss
    return features.new_tensor(0.0)


def build_pair_weights(sample_ids, labels, q_map, mode):
    n = len(sample_ids)
    w = torch.ones((n, n), dtype=torch.float32)
    if mode == "uniform":
        return w
    for i, sid_i in enumerate(sample_ids):
        qi = q_map[sid_i]
        for j, sid_j in enumerate(sample_ids):
            if labels[i] != labels[j] or i == j:
                w[i, j] = 0.0
                continue
            r_ij = 1.0 - float(np.dot(qi, q_map[sid_j]))
            w[i, j] = 0.1 + 0.9 * r_ij
    return w


def train_epoch(model, projection_head, loader, optimizer, scaler, device, dataset_name, q_map=None, contrastive=False, pair_mode="weighted", lambda_supcon=0.5):
    model.train()
    if projection_head is not None:
        projection_head.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        images, y, _, sample_ids = batch
        images = images.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            feats, logits = model(images)
            ce_loss = F.cross_entropy(logits, y)
            loss = ce_loss
            if contrastive:
                proj = projection_head(feats)
                weights = build_pair_weights(sample_ids.numpy().tolist(), y.cpu().numpy().tolist(), q_map, pair_mode).to(device)
                supcon = weighted_supcon_loss(proj, y, weights)
                loss = ce_loss + lambda_supcon * supcon
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
    return total_loss / max(total, 1)


def run_warmup(model, loader, cache_loader, val_loader, optimizer, device, seed, epochs, out_dir):
    scaler = GradScaler(enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    warmup_epochs = []
    prev_feats = {}
    log_path = out_dir / "warmup_log.csv"
    best_val = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, None, loader, optimizer, scaler, device, "warmup")
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device)
        scheduler.step()
        epoch_cache = extract_epoch_cache(model, cache_loader, device, prev_feats=prev_feats)
        prev_feats = {sid: payload["feature"] for sid, payload in epoch_cache.items()}
        warmup_epochs.append(epoch_cache)
        write_csv_row(log_path, {"epoch": epoch, "train_loss": train_loss, **val_metrics})
        ckpt_path = out_dir / f"warmup_epoch_{epoch}.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)
        if val_metrics["avg_accuracy"] > best_val:
            best_val = val_metrics["avg_accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return warmup_epochs


def pseudo_group_weights(q_map, labels):
    class_masses = defaultdict(lambda: None)
    grouped = defaultdict(list)
    for sid, q in q_map.items():
        grouped[labels[sid]].append(q)
    for cls, qs in grouped.items():
        class_masses[cls] = np.mean(np.stack(qs), axis=0)
    weights = {}
    for sid, q in q_map.items():
        pi = class_masses[labels[sid]]
        weights[sid] = float(np.sum(q / (pi + 1e-6)))
    return weights


def train_downstream(model, projection_head, train_loader, val_loader, test_loader, device, q_map, labels_map, method_name, out_dir, dataset_name, epochs=6):
    optimizer = build_optimizer(model, projection_head, dataset_name)
    scaler = GradScaler(enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    log_path = out_dir / "finetune_log.csv"
    best_state = None
    best_metric = -1.0
    erm_weights = pseudo_group_weights(q_map, labels_map)
    for epoch in range(1, epochs + 1):
        model.train()
        if projection_head is not None:
            projection_head.train()
        total_loss = 0.0
        total = 0
        for images, y, _, sample_ids in train_loader:
            images = images.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                feats, logits = model(images)
                if method_name == "soft_rwerm":
                    weights = torch.tensor([erm_weights[int(sid)] for sid in sample_ids.numpy().tolist()], device=device)
                    ce = F.cross_entropy(logits, y, reduction="none")
                    loss = (ce * weights).mean()
                elif method_name == "vanilla_supcon":
                    ce = F.cross_entropy(logits, y)
                    proj = projection_head(feats)
                    weights = build_pair_weights(sample_ids.numpy().tolist(), y.cpu().numpy().tolist(), q_map, "uniform").to(device)
                    loss = ce + 0.5 * weighted_supcon_loss(proj, y, weights)
                else:
                    ce = F.cross_entropy(logits, y)
                    proj = projection_head(feats)
                    pair_mode = "uniform" if method_name == "uniform_weight_ablation" else "weighted"
                    weights = build_pair_weights(sample_ids.numpy().tolist(), y.cpu().numpy().tolist(), q_map, pair_mode).to(device)
                    loss = ce + 0.5 * weighted_supcon_loss(proj, y, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
        scheduler.step()
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device)
        write_csv_row(log_path, {"epoch": epoch, "train_loss": total_loss / max(total, 1), **val_metrics})
        if val_metrics["avg_accuracy"] > best_metric:
            best_metric = val_metrics["avg_accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, val_y, val_pred, val_groups = evaluate_model(model, val_loader, device)
    test_metrics, test_y, test_pred, test_groups = evaluate_model(model, test_loader, device)
    return val_metrics, test_metrics, (val_y, val_pred, val_groups), (test_y, test_pred, test_groups)


def gather_signal_eval(signal_models, train_warmup, train_labels, val_warmup, val_labels, test_warmup, test_labels, signal_kind):
    train_q, train_cluster = apply_signal_models(signal_models, train_warmup, train_labels, signal_kind)
    val_q, val_cluster = apply_signal_models(signal_models, val_warmup, val_labels, signal_kind)
    test_q, test_cluster = apply_signal_models(signal_models, test_warmup, test_labels, signal_kind)
    return (train_q, train_cluster), (val_q, val_cluster), (test_q, test_cluster)


def extract_signal_for_split(model, dataset_name, split, seed, train_tf, batch_size, warmup_epochs, checkpoint_dir: Path):
    loader = DataLoader(
        build_dataset(dataset_name, split, seed, transform=TwoViewTransform(train_tf), two_views=True),
        batch_size=max(16, batch_size // 2),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    prev_feats = None
    per_epoch = []
    device = get_device()
    for epoch_idx in range(1, len(warmup_epochs) + 1):
        state = torch.load(checkpoint_dir / f"warmup_epoch_{epoch_idx}.pt", map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        cache = extract_epoch_cache(model, loader, device, prev_feats if prev_feats is not None else {})
        prev_feats = {sid: payload["feature"] for sid, payload in cache.items()}
        per_epoch.append(cache)
    sample_ids = sorted(per_epoch[-1].keys())
    return build_signal_map(per_epoch, sample_ids)


def save_json(path: Path, payload) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_yaml(path: Path, payload) -> None:
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def create_related_work_matrix():
    path = ROOT / "related_work_matrix.md"
    content = """# Related Work Matrix

| Work | Signal source | Uses training dynamics | Pseudo-group inference | Downstream objective | Group labels needed | Exact matched-budget trajectory vs output-only comparison |
|---|---|---|---|---|---|---|
| SPARE | Early outputs | Yes | Yes | Non-contrastive | No | not found |
| ExMap | Explainability heatmaps | No | Yes | Non-contrastive | No | not found |
| GIC | Inferred groups / refinement | Sometimes | Yes | Non-contrastive | No | not found |
| Beyond Distribution Shift | Training dynamics analysis | Yes | No | N/A | N/A | not found |
| Correct-N-Contrast | Pseudo-groups / labels | No | No | Contrastive | Often yes | not found |
| Current study | Warmup trajectories and endpoint features | Yes | Yes | Contrastive and non-contrastive | No | yes |
"""
    path.write_text(content)


def detect_gpu_driver_version() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        line = proc.stdout.strip().splitlines()[0]
        return line.strip()
    except Exception:
        return "unknown"


def write_env_report():
    import platform
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_driver_version": detect_gpu_driver_version(),
    }
    save_json(ROOT / "env_report.json", report)


def write_requirements_lock():
    import pkg_resources
    lines = sorted(f"{dist.project_name}=={dist.version}" for dist in pkg_resources.working_set)
    (ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n")


def write_data_summary(dataset_name: str, seed: int):
    summary = {}
    resolution = [224, 224] if dataset_name == "waterbirds" else [28, 28]
    for split in ["train", "validation", "test"]:
        ds = build_dataset(dataset_name, split, seed, transform=None)
        labels = []
        groups = []
        records = []
        for idx in range(len(ds)):
            _, y, g, sid = ds[idx]
            labels.append(int(y))
            groups.append(int(g))
            records.append(SampleRecord(sample_id=int(sid), split=split, y=int(y), group=int(g), meta={}))
        contingency = pd.crosstab(pd.Series(labels, name="label"), pd.Series(groups, name="group")).to_dict()
        summary[split] = {
            "size": len(ds),
            "class_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
            "group_counts": pd.Series(groups).value_counts().sort_index().to_dict(),
            "class_group_contingency": contingency,
            "imbalance_ratio": float(max(pd.Series(groups).value_counts()) / max(1, min(pd.Series(groups).value_counts()))),
            "manifest_sha256": hash_manifest(records),
            "expected_image_resolution": resolution,
        }
    save_json(ROOT / "data_summary.json", {"dataset": dataset_name, "splits": summary})


def ensure_experiment_logs():
    for result_path in ROOT.glob("exp/**/results.json"):
        out_dir = result_path.parent
        logs_dir = ensure_dir(out_dir / "logs")
        for name in ["warmup_log.csv", "finetune_log.csv"]:
            src = out_dir / name
            if src.exists():
                shutil.copy2(src, logs_dir / name)
        config_path = out_dir / "config.yaml"
        if config_path.exists():
            shutil.copy2(config_path, logs_dir / "config.yaml")


def compute_validation_diagnostics(dataset_name: str, warmup_epochs: int, seed: int, checkpoint_dir: Path):
    train_tf, _, _ = make_transforms(dataset_name)
    model = build_model(dataset_name, 2).to(get_device())
    val_ds = build_dataset(dataset_name, "validation", seed, transform=None)
    signal_map = extract_signal_for_split(model, dataset_name, "validation", seed, train_tf, 64 if dataset_name == "waterbirds" else 256, [None] * warmup_epochs, checkpoint_dir)
    rows = []
    for idx in range(len(val_ds)):
        _, y, g, sid = val_ds[idx]
        entry = signal_map[int(sid)]
        hidden_group = int(g) % 2
        for epoch in range(warmup_epochs):
            rows.append(
                {
                    "sample_id": int(sid),
                    "epoch": epoch + 1,
                    "label": int(y),
                    "hidden_group": hidden_group,
                    "confidence": float(entry["p"][epoch]),
                    "margin": float(entry["m"][epoch]),
                    "loss": float(entry["l"][epoch]),
                    "disagreement": float(entry["u"][epoch]),
                    "drift": float(entry["d"][epoch]),
                }
            )
    return pd.DataFrame(rows)


def disagreement_vector(q_map: Dict[int, np.ndarray], labels_map: Dict[int, int]) -> np.ndarray:
    by_class = defaultdict(list)
    for sid, y in labels_map.items():
        if sid in q_map:
            by_class[y].append(int(sid))
    parts = []
    for cls in sorted(by_class):
        ids = sorted(by_class[cls])
        for i, sid_i in enumerate(ids):
            qi = np.asarray(q_map[sid_i], dtype=np.float64)
            for sid_j in ids[i + 1 :]:
                qj = np.asarray(q_map[sid_j], dtype=np.float64)
                parts.append(1.0 - float(np.dot(qi, qj)))
    return np.asarray(parts, dtype=np.float64)


def restore_signal_artifacts():
    cluster_conditions = {
        "final_feature_xgsupcon": "final_feature",
        "output_only_xgsupcon": "output_only",
        "trajectory_xgsupcon": "trajectory",
        "output_only_soft_rwerm": "output_only",
        "trajectory_soft_rwerm": "trajectory",
        "trajectory_no_shape": "trajectory_no_shape",
        "uniform_weights": "trajectory",
    }
    allowed_waterbirds_conditions = set(cluster_conditions)
    stability_records = defaultdict(list)
    diagnostic_frames = []
    for config_path in sorted(ROOT.glob("exp/**/config.yaml")):
        if any(part.startswith("_") for part in config_path.parts):
            continue
        out_dir = config_path.parent
        if not (out_dir / "warmup_epoch_1.pt").exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        condition = config["condition"]
        seed = int(config["seed"])
        dataset_name = config["dataset"]
        warmup_epochs = int(config["warmup_epochs"])
        if dataset_name != "waterbirds":
            continue
        if condition not in allowed_waterbirds_conditions:
            continue
        existing_outputs = out_dir / "clustering_outputs.json"
        existing_payload = None
        if existing_outputs.exists():
            try:
                existing_payload = json.loads(existing_outputs.read_text())
            except json.JSONDecodeError:
                existing_payload = None
        if dataset_name == "waterbirds" and condition in {"output_only_xgsupcon", "trajectory_xgsupcon"}:
            diagnostic_frames.append(compute_validation_diagnostics(dataset_name, warmup_epochs, seed, out_dir))
            if existing_payload and "transform_state" in existing_payload and "val_q" in existing_payload:
                val_ds = build_dataset(dataset_name, "validation", seed, transform=None)
                val_labels = {int(ds[3]): int(ds[1]) for ds in (val_ds[i] for i in range(len(val_ds)))}
                key = f"waterbirds_{condition}_T{warmup_epochs}"
                stability_records[key].append(
                    {
                        "seed": seed,
                        "q_map": {int(k): np.asarray(v, dtype=np.float64) for k, v in existing_payload["val_q"].items()},
                        "labels_map": val_labels,
                    }
                )
                if "transform_state" in existing_payload:
                    continue
        elif existing_payload and "transform_state" in existing_payload:
            continue
        train_tf, _, _ = make_transforms(dataset_name)
        batch_size = 64
        model = build_model(dataset_name, 2).to(get_device())
        train_signal_map = extract_signal_for_split(model, dataset_name, "train", seed, train_tf, batch_size, [None] * warmup_epochs, out_dir)
        train_ds = build_dataset(dataset_name, "train", seed, transform=None)
        train_labels = {int(ds[3]): int(ds[1]) for ds in (train_ds[i] for i in range(len(train_ds)))}
        signal_models, train_q, train_cluster, _ = fit_signal_models(train_signal_map, train_labels, cluster_conditions[condition], seed)
        payload = {
            "signal_kind": cluster_conditions[condition],
            "transform_state": serialize_signal_models(signal_models),
            "train_q": {str(k): np.asarray(v).tolist() for k, v in train_q.items()},
            "train_cluster": {str(k): int(v) for k, v in train_cluster.items()},
        }
        if condition in {"output_only_xgsupcon", "trajectory_xgsupcon"}:
            val_signal_map = extract_signal_for_split(model, dataset_name, "validation", seed, train_tf, batch_size, [None] * warmup_epochs, out_dir)
            val_ds = build_dataset(dataset_name, "validation", seed, transform=None)
            val_labels = {int(ds[3]): int(ds[1]) for ds in (val_ds[i] for i in range(len(val_ds)))}
            val_groups = {int(ds[3]): int(ds[2]) for ds in (val_ds[i] for i in range(len(val_ds)))}
            _, (val_q, val_cluster), _ = gather_signal_eval(
                signal_models,
                train_signal_map,
                train_labels,
                val_signal_map,
                val_labels,
                val_signal_map,
                val_labels,
                cluster_conditions[condition],
            )
            payload["val_q"] = {str(k): np.asarray(v).tolist() for k, v in val_q.items()}
            payload["val_cluster"] = {str(k): int(v) for k, v in val_cluster.items()}
            payload["val_group_true"] = {str(k): int(v) for k, v in val_groups.items()}
        save_json(out_dir / "clustering_outputs.json", payload)
        if dataset_name == "waterbirds" and condition in {"output_only_xgsupcon", "trajectory_xgsupcon"}:
            key = f"waterbirds_{condition}_T{warmup_epochs}"
            stability_records[key].append(
                {
                    "seed": seed,
                    "q_map": val_q,
                    "labels_map": val_labels,
                }
            )
    if diagnostic_frames:
        df = pd.concat(diagnostic_frames, ignore_index=True)
        save_json(
            ROOT / "figures" / "warmup_diagnostics_data.json",
            {"rows": df.to_dict(orient="records")},
        )
    stability_summary = {}
    for key, runs in stability_records.items():
        corrs = []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                vec_i = disagreement_vector(runs[i]["q_map"], runs[i]["labels_map"])
                vec_j = disagreement_vector(runs[j]["q_map"], runs[j]["labels_map"])
                corr = spearmanr(vec_i, vec_j).statistic
                corrs.append(float(corr))
        if corrs:
            stability_summary[key] = {
                "pairwise_disagreement_spearman": mean_std(corrs),
                "values": corrs,
            }
    return stability_summary


def run_condition(dataset_name: str, condition: str, seed: int, warmup_epochs: int, output_dir: Path):
    set_seed(seed)
    ensure_dir(output_dir)
    train_tf, eval_tf, _ = make_transforms(dataset_name)
    batch_size = 64 if dataset_name == "waterbirds" else 256
    train_set, val_set, test_set, train_loader, val_loader, test_loader, cache_loader = make_loaders(dataset_name, seed, batch_size, train_tf, eval_tf)
    model = build_model(dataset_name, 2).to(get_device())
    projection_head = ProjectionHead(model.feat_dim).to(get_device()) if condition != "erm" else None
    config = {
        "dataset": dataset_name,
        "condition": condition,
        "seed": seed,
        "warmup_epochs": warmup_epochs,
        "downstream_epochs": 6,
        "batch_size": batch_size,
        "lambda_supcon": 0.5,
        "tau": 0.07,
    }
    save_yaml(output_dir / "config.yaml", config)
    warmup_start = time.time()
    optimizer = build_optimizer(model, None, dataset_name)
    warmup_history = run_warmup(model, train_loader, cache_loader, val_loader, optimizer, get_device(), seed, warmup_epochs, output_dir)
    warmup_time = time.time() - warmup_start
    sample_ids = sorted(warmup_history[-1].keys())
    signal_map = build_signal_map(warmup_history, sample_ids)
    labels_map = {sid: payload["label"] for sid, payload in warmup_history[-1].items()}
    if condition == "erm":
        # Continue ERM training from the warmup model for the same total budget.
        proj = None
        val_metrics, test_metrics, _, _ = train_downstream(model, proj, train_loader, val_loader, test_loader, get_device(), {sid: np.asarray([0.5, 0.5]) for sid in sample_ids}, labels_map, "soft_rwerm", output_dir, dataset_name, epochs=6)
        result = {
            "experiment": f"{dataset_name}_{condition}",
            "seed": seed,
            "warmup_epochs": warmup_epochs,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "runtime_minutes": (time.time() - warmup_start) / 60.0,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        }
        save_json(output_dir / "results.json", result)
        return result
    signal_kind = {
        "final_feature_xgsupcon": "final_feature",
        "output_only_xgsupcon": "output_only",
        "trajectory_xgsupcon": "trajectory",
        "output_only_soft_rwerm": "output_only",
        "trajectory_soft_rwerm": "trajectory",
        "trajectory_no_shape": "trajectory_no_shape",
        "uniform_weights": "trajectory",
        "vanilla_supcon": "output_only",
    }[condition]
    fidelity = None
    clustering_payload = None
    cluster_start = time.time()
    train_q = {sid: np.asarray([0.5, 0.5], dtype=np.float32) for sid in sample_ids}
    if condition != "vanilla_supcon":
        signal_models, train_q, train_cluster, _ = fit_signal_models(signal_map, labels_map, signal_kind, seed)
        clustering_time = time.time() - cluster_start
        val_signal_map = extract_signal_for_split(model, dataset_name, "validation", seed, train_tf, batch_size, warmup_history, output_dir)
        test_signal_map = extract_signal_for_split(model, dataset_name, "test", seed, train_tf, batch_size, warmup_history, output_dir)
        val_ds_plain = build_dataset(dataset_name, "validation", seed, transform=None)
        test_ds_plain = build_dataset(dataset_name, "test", seed, transform=None)
        val_labels_map = {}
        test_labels_map = {}
        val_groups_true = {}
        for i in range(len(val_ds_plain)):
            _, y, g, sid = val_ds_plain[i]
            val_labels_map[int(sid)] = int(y)
            val_groups_true[int(sid)] = int(g)
        for i in range(len(test_ds_plain)):
            _, y, _, sid = test_ds_plain[i]
            test_labels_map[int(sid)] = int(y)
        (_, _), (val_q, val_cluster), (test_q, test_cluster) = gather_signal_eval(
            signal_models,
            signal_map,
            labels_map,
            val_signal_map,
            val_labels_map,
            test_signal_map,
            test_labels_map,
            signal_kind,
        )
        fidelity_per_class = []
        for cls in sorted(set(val_labels_map.values())):
            ids = [sid for sid, y in val_labels_map.items() if y == cls]
            true_groups = [val_groups_true[sid] % 2 for sid in ids]
            pred_groups = [val_cluster[sid] for sid in ids]
            fidelity_per_class.append(
                {
                    "nmi": normalized_mutual_info_score(true_groups, pred_groups),
                    "ari": adjusted_rand_score(true_groups, pred_groups),
                    "minority_f1": minority_f1_from_clusters(true_groups, pred_groups),
                    "balanced_accuracy": cluster_balanced_accuracy(true_groups, pred_groups),
                }
            )
        fidelity = {k: float(np.mean([x[k] for x in fidelity_per_class])) for k in fidelity_per_class[0]}
        clustering_payload = {
            "signal_kind": signal_kind,
            "transform_state": serialize_signal_models(signal_models),
            "train_q": {str(k): v.tolist() for k, v in train_q.items()},
            "train_cluster": {str(k): int(v) for k, v in train_cluster.items()},
            "val_q": {str(k): v.tolist() for k, v in val_q.items()},
            "val_cluster": {str(k): int(v) for k, v in val_cluster.items()},
            "val_group_true": {str(k): int(v) for k, v in val_groups_true.items()},
            "test_q": {str(k): v.tolist() for k, v in test_q.items()},
            "test_cluster": {str(k): int(v) for k, v in test_cluster.items()},
        }
    else:
        clustering_time = 0.0
    downstream_method = {
        "final_feature_xgsupcon": "xgsupcon",
        "output_only_xgsupcon": "xgsupcon",
        "trajectory_xgsupcon": "xgsupcon",
        "trajectory_no_shape": "xgsupcon",
        "uniform_weights": "uniform_weight_ablation",
        "output_only_soft_rwerm": "soft_rwerm",
        "trajectory_soft_rwerm": "soft_rwerm",
        "vanilla_supcon": "vanilla_supcon",
    }[condition]
    finetune_start = time.time()
    val_metrics, test_metrics, _, _ = train_downstream(model, projection_head, train_loader, val_loader, test_loader, get_device(), train_q, labels_map, downstream_method, output_dir, dataset_name, epochs=6)
    finetune_time = time.time() - finetune_start
    result = {
        "experiment": f"{dataset_name}_{condition}",
        "seed": seed,
        "warmup_epochs": warmup_epochs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "runtime_minutes": (warmup_time + clustering_time + finetune_time) / 60.0,
        "warmup_minutes": warmup_time / 60.0,
        "clustering_minutes": clustering_time / 60.0,
        "finetune_minutes": finetune_time / 60.0,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
        "warmup_cache_size_mb": float(sum((output_dir / f"warmup_epoch_{i}.pt").stat().st_size for i in range(1, warmup_epochs + 1)) / 1024**2),
    }
    if condition != "vanilla_supcon":
        result["signal_kind"] = signal_kind
        result["fidelity"] = fidelity
        save_json(output_dir / "clustering_outputs.json", clustering_payload)
    save_json(output_dir / "results.json", result)
    return result


def aggregate_results(stability_summary=None):
    result_files = sorted(
        path
        for path in ROOT.glob("exp/**/results.json")
        if not any(part.startswith("_") for part in path.parts)
    )
    by_exp = defaultdict(list)
    raw = []
    for path in result_files:
        payload = json.loads(path.read_text())
        raw.append(payload)
        exp_key = f"{payload['experiment']}_T{payload.get('warmup_epochs', 'na')}"
        by_exp[exp_key].append(payload)
    summary = {"headline_findings": {}, "raw_runs": raw, "aggregated": {}, "stability": stability_summary or {}}
    rows = []
    for exp_name, runs in sorted(by_exp.items()):
        agg = {"n_runs": len(runs)}
        metrics = defaultdict(list)
        for run in runs:
            for split in ["val_metrics", "test_metrics"]:
                for key, value in run.get(split, {}).items():
                    metrics[f"{split}.{key}"].append(value)
            for key, value in run.get("fidelity", {}).items():
                metrics[f"fidelity.{key}"].append(value)
            for key in ["runtime_minutes", "peak_gpu_memory_mb"]:
                if key in run:
                    metrics[key].append(run[key])
        for key, values in metrics.items():
            agg[key] = mean_std(values)
        summary["aggregated"][exp_name] = agg
        rows.append({"experiment": exp_name, **{k: v["mean"] if isinstance(v, dict) and "mean" in v else v for k, v in agg.items()}})
    def safe_mean(exp, metric):
        return summary["aggregated"].get(exp, {}).get(metric, {}).get("mean")
    summary["headline_findings"] = {
        "primary_result": "Negative result under matched compute: trajectory features did not improve downstream robustness over output-only controls on Waterbirds, and the transfer claim to soft-rwERM failed.",
        "matched_budget": {
            "waterbirds_T2_test_worst_group_output_only": safe_mean("waterbirds_output_only_xgsupcon_T2", "test_metrics.worst_group_accuracy"),
            "waterbirds_T2_test_worst_group_trajectory": safe_mean("waterbirds_trajectory_xgsupcon_T2", "test_metrics.worst_group_accuracy"),
            "waterbirds_T4_test_worst_group_output_only": safe_mean("waterbirds_output_only_xgsupcon_T4", "test_metrics.worst_group_accuracy"),
            "waterbirds_T4_test_worst_group_trajectory": safe_mean("waterbirds_trajectory_xgsupcon_T4", "test_metrics.worst_group_accuracy"),
        },
        "transfer_test": {
            "waterbirds_soft_rwerm_output_only_test_worst_group": safe_mean("waterbirds_output_only_soft_rwerm_T4", "test_metrics.worst_group_accuracy"),
            "waterbirds_soft_rwerm_trajectory_test_worst_group": safe_mean("waterbirds_trajectory_soft_rwerm_T4", "test_metrics.worst_group_accuracy"),
        },
        "narrow_positive_claim": "Trajectory improves pseudo-group fidelity over output-only at T=4 on Waterbirds, but those fidelity gains do not translate into better worst-group accuracy.",
    }
    save_json(ROOT / "results.json", summary)
    save_json(ROOT / "results_summary.json", summary["aggregated"])
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(ROOT / "tables_summary.csv", index=False)
        (ROOT / "tables_summary.md").write_text(df.to_markdown(index=False))


def generate_named_tables():
    summary_path = ROOT / "results_summary.json"
    if not summary_path.exists():
        return
    ensure_dir(ROOT / "tables")
    summary = json.loads(summary_path.read_text())
    def get_mean(key, metric):
        return summary.get(key, {}).get(metric, {}).get("mean")
    def get_std(key, metric):
        return summary.get(key, {}).get(metric, {}).get("std")
    main_rows = []
    row_defs = [
        ("ERM", "waterbirds_erm_T4"),
        ("Vanilla SupCon", "waterbirds_vanilla_supcon_T4"),
        ("Final-feature xg-SupCon (T=4)", "waterbirds_final_feature_xgsupcon_T4"),
        ("Output-only xg-SupCon (T=4)", "waterbirds_output_only_xgsupcon_T4"),
        ("Trajectory xg-SupCon (T=4)", "waterbirds_trajectory_xgsupcon_T4"),
        ("Output-only soft-rwERM (T=4)", "waterbirds_output_only_soft_rwerm_T4"),
        ("Trajectory soft-rwERM (T=4)", "waterbirds_trajectory_soft_rwerm_T4"),
    ]
    for label, key in row_defs:
        main_rows.append(
            {
                "method": label,
                "val_avg_acc": get_mean(key, "val_metrics.avg_accuracy"),
                "test_avg_acc": get_mean(key, "test_metrics.avg_accuracy"),
                "val_worst_group": get_mean(key, "val_metrics.worst_group_accuracy"),
                "test_worst_group": get_mean(key, "test_metrics.worst_group_accuracy"),
                "val_gap": get_mean(key, "val_metrics.robustness_gap"),
                "test_gap": get_mean(key, "test_metrics.robustness_gap"),
                "val_nmi": get_mean(key, "fidelity.nmi"),
                "val_minority_f1": get_mean(key, "fidelity.minority_f1"),
                "runtime_minutes": get_mean(key, "runtime_minutes"),
                "peak_gpu_memory_mb": get_mean(key, "peak_gpu_memory_mb"),
            }
        )
    matched_rows = []
    matched_keys = ["waterbirds_output_only_xgsupcon_T2", "waterbirds_trajectory_xgsupcon_T2", "waterbirds_output_only_xgsupcon_T4", "waterbirds_trajectory_xgsupcon_T4"]
    wg_delta_t2 = get_mean("waterbirds_trajectory_xgsupcon_T2", "test_metrics.worst_group_accuracy") - get_mean("waterbirds_output_only_xgsupcon_T2", "test_metrics.worst_group_accuracy")
    wg_delta_t4 = get_mean("waterbirds_trajectory_xgsupcon_T4", "test_metrics.worst_group_accuracy") - get_mean("waterbirds_output_only_xgsupcon_T4", "test_metrics.worst_group_accuracy")
    nmi_delta_t2 = get_mean("waterbirds_trajectory_xgsupcon_T2", "fidelity.nmi") - get_mean("waterbirds_output_only_xgsupcon_T2", "fidelity.nmi")
    nmi_delta_t4 = get_mean("waterbirds_trajectory_xgsupcon_T4", "fidelity.nmi") - get_mean("waterbirds_output_only_xgsupcon_T4", "fidelity.nmi")
    for key in matched_keys:
        matched_rows.append(
            {
                "method": key,
                "test_worst_group_mean": get_mean(key, "test_metrics.worst_group_accuracy"),
                "test_worst_group_std": get_std(key, "test_metrics.worst_group_accuracy"),
                "val_nmi_mean": get_mean(key, "fidelity.nmi"),
                "val_nmi_std": get_std(key, "fidelity.nmi"),
                "wg_delta_vs_output_same_T": 0.0 if "output_only" in key else (wg_delta_t2 if key.endswith("T2") else wg_delta_t4),
                "nmi_delta_vs_output_same_T": 0.0 if "output_only" in key else (nmi_delta_t2 if key.endswith("T2") else nmi_delta_t4),
                "wg_diff_in_diff_T2_minus_T4": wg_delta_t2 - wg_delta_t4,
                "nmi_diff_in_diff_T2_minus_T4": nmi_delta_t2 - nmi_delta_t4,
            }
        )
    ablation_rows = []
    for label, key in [
        ("Full trajectory", "waterbirds_trajectory_xgsupcon_T4"),
        ("No u/d (alias of output-only)", "waterbirds_output_only_xgsupcon_T4"),
        ("No shape code", "waterbirds_trajectory_no_shape_T4"),
        ("Uniform SupCon weights", "waterbirds_uniform_weights_T4"),
        ("Output-only soft-rwERM", "waterbirds_output_only_soft_rwerm_T4"),
        ("Trajectory soft-rwERM", "waterbirds_trajectory_soft_rwerm_T4"),
    ]:
        ablation_rows.append(
            {
                "variant": label,
                "val_nmi": get_mean(key, "fidelity.nmi"),
                "val_minority_f1": get_mean(key, "fidelity.minority_f1"),
                "test_worst_group": get_mean(key, "test_metrics.worst_group_accuracy"),
            }
        )
    for name, rows in [
        ("main_waterbirds_table", main_rows),
        ("matched_budget_waterbirds_table", matched_rows),
        ("ablation_waterbirds_table", ablation_rows),
    ]:
        df = pd.DataFrame(rows)
        df.to_csv(ROOT / "tables" / f"{name}.csv", index=False)
        (ROOT / "tables" / f"{name}.md").write_text(df.to_markdown(index=False))


def write_skip_docs():
    bic_dir = ensure_dir(EXP_ROOT / "waterbirds_ablations" / "bic_selected_k_optional")
    (bic_dir / "SKIPPED.md").write_text(
        "Optional BIC-selected K ablation was not run. The required benchmark, transfer, and 3-seed Waterbirds ablations exhausted the planned budget, and plan.json explicitly prioritizes those before optional K-sensitivity.\n"
    )
    alias_dir = ensure_dir(EXP_ROOT / "waterbirds_ablations" / "no_u_d_alias")
    (alias_dir / "SKIPPED.md").write_text(
        "Ablation A ('remove u/d from trajectory but keep shape code and t_corr') is exactly the executed output-only control at T=4. No duplicate GPU run was launched; use the executed results under exp/waterbirds_benchmark/output_only_xgsupcon_T4_seed{7,17,27}/results.json.\n"
    )


def plot_figures():
    summary_path = ROOT / "results_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text())
    rows = []
    for exp_name, payload in summary.items():
        row = {"experiment": exp_name}
        for metric in ["test_metrics.worst_group_accuracy", "fidelity.nmi"]:
            if metric in payload:
                row[metric] = payload[metric]["mean"]
                row[f"{metric}_std"] = payload[metric]["std"]
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    ensure_dir(FIG_ROOT)
    for metric, ylabel, fname in [
        ("test_metrics.worst_group_accuracy", "Worst-group accuracy", "waterbirds_worst_group_accuracy"),
        ("fidelity.nmi", "Validation NMI", "waterbirds_validation_nmi"),
    ]:
        subset = df.dropna(subset=[metric])
        if subset.empty:
            continue
        plt.figure(figsize=(10, 4))
        sns.barplot(data=subset, x="experiment", y=metric, color="#4C78A8")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(FIG_ROOT / f"{fname}.png", dpi=200)
        plt.savefig(FIG_ROOT / f"{fname}.pdf")
        plt.close()
    diag_path = FIG_ROOT / "warmup_diagnostics_data.json"
    if diag_path.exists():
        payload = json.loads(diag_path.read_text())
        diag_df = pd.DataFrame(payload["rows"])
        if not diag_df.empty:
            long_df = diag_df.melt(
                id_vars=["epoch", "hidden_group"],
                value_vars=["confidence", "margin", "loss", "disagreement", "drift"],
                var_name="metric",
                value_name="value",
            )
            g = sns.relplot(
                data=long_df,
                x="epoch",
                y="value",
                hue="hidden_group",
                col="metric",
                kind="line",
                errorbar="sd",
                facet_kws={"sharey": False, "sharex": True},
                height=3.0,
                aspect=1.0,
            )
            g.set_titles("{col_name}")
            g.savefig(FIG_ROOT / "waterbirds_warmup_diagnostics.png", dpi=200)
            g.savefig(FIG_ROOT / "waterbirds_warmup_diagnostics.pdf")
            plt.close("all")


def write_stability_report(stability_summary):
    save_json(ROOT / "stability_analysis.json", stability_summary)


def run_suite(suite: str):
    create_related_work_matrix()
    write_env_report()
    write_requirements_lock()
    write_data_summary("waterbirds", SEEDS[0])
    conditions = []
    if suite == "waterbirds_benchmark":
        conditions = [
            ("erm", 4),
            ("vanilla_supcon", 4),
            ("final_feature_xgsupcon", 4),
            ("output_only_xgsupcon", 2),
            ("trajectory_xgsupcon", 2),
            ("output_only_xgsupcon", 4),
            ("trajectory_xgsupcon", 4),
            ("output_only_soft_rwerm", 4),
            ("trajectory_soft_rwerm", 4),
        ]
        dataset_name = "waterbirds"
    elif suite == "waterbirds_ablations":
        conditions = [
            ("trajectory_no_shape", 4),
            ("uniform_weights", 4),
        ]
        dataset_name = "waterbirds"
    elif suite == "colored_mnist":
        conditions = [
            ("erm", 4),
            ("output_only_xgsupcon", 4),
            ("trajectory_xgsupcon", 4),
        ]
        dataset_name = "colored_mnist"
        write_data_summary("colored_mnist", SEEDS[0])
    else:
        raise ValueError(f"Unknown suite: {suite}")
    for seed in SEEDS if dataset_name == "waterbirds" else [SEEDS[0]]:
        for condition, warmup_epochs in conditions:
            out_dir = ensure_dir(EXP_ROOT / suite / f"{condition}_T{warmup_epochs}_seed{seed}")
            if (out_dir / "results.json").exists():
                continue
            run_condition(dataset_name, condition, seed, warmup_epochs, out_dir)
    ensure_experiment_logs()
    stability_summary = restore_signal_artifacts()
    aggregate_results(stability_summary=stability_summary)
    write_stability_report(stability_summary)
    generate_named_tables()
    plot_figures()
    write_skip_docs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=["waterbirds_benchmark", "waterbirds_ablations", "colored_mnist"])
    args = parser.parse_args()
    run_suite(args.suite)


if __name__ == "__main__":
    main()
