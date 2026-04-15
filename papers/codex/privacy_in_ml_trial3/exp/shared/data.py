from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import DATA_ROOT, OUTPUT_ROOT, DatasetConfig, DATASET_CONFIGS


PURCHASE_URL = "https://raw.githubusercontent.com/xehartnort/Purchase100-Texas100-datasets/master/purchase100.npz"


class IndexedArrayDataset(Dataset):
    def __init__(self, x, y, indices, transform=None) -> None:
        self.x = x
        self.y = y.astype(np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        x = self.x[idx]
        y = int(self.y[idx])
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        return x, y, idx


def _download_purchase() -> Path:
    path = DATA_ROOT / "purchase100.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(PURCHASE_URL, path)
    return path


def _load_purchase_arrays():
    payload = np.load(_download_purchase())
    if {"features", "labels"}.issubset(payload.files):
        x = payload["features"].astype(np.float32)
        y = payload["labels"].astype(np.int64)
    else:
        keys = payload.files
        x = payload[keys[0]].astype(np.float32)
        y = payload[keys[1]].astype(np.int64)
    if y.ndim > 1:
        y = np.argmax(y, axis=1)
    y = y.reshape(-1)
    if y.min() == 1:
        y = y - 1
    return x, y


def _make_split(indices: np.ndarray, y: np.ndarray, sizes: list[int], seed: int):
    remaining = indices
    remaining_y = y[remaining]
    parts = []
    for size in sizes[:-1]:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=seed + len(parts))
        selected_idx, rest_idx = next(sss.split(np.zeros(len(remaining)), remaining_y))
        chosen = remaining[selected_idx]
        parts.append(chosen)
        remaining = remaining[rest_idx]
        remaining_y = y[remaining]
    parts.append(remaining[: sizes[-1]])
    return parts


def prepare_splits(dataset: str, seed: int) -> dict:
    cfg = DATASET_CONFIGS[dataset]
    split_path = OUTPUT_ROOT / "metrics" / f"{dataset}_split_seed_{seed}.json"
    if split_path.exists():
        return json.loads(split_path.read_text())
    if dataset == "purchase100":
        x, y = _load_purchase_arrays()
        all_indices = np.arange(len(y))
        train_idx, val_idx, test_idx, ref_idx = _make_split(
            all_indices,
            y,
            [cfg.train_size, cfg.val_size, cfg.test_size, cfg.ref_size],
            seed,
        )
    else:
        train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True)
        y = np.asarray(train_set.targets, dtype=np.int64)
        all_indices = np.arange(len(y))
        train_idx, val_idx, ref_idx = _make_split(
            all_indices,
            y,
            [cfg.train_size, cfg.val_size, cfg.ref_size],
            seed,
        )
        test_idx = list(range(cfg.test_size))
    split = {
        "dataset": dataset,
        "seed": seed,
        "train": np.sort(np.asarray(train_idx)).tolist(),
        "val": np.sort(np.asarray(val_idx)).tolist(),
        "test": np.sort(np.asarray(test_idx)).tolist(),
        "reference": np.sort(np.asarray(ref_idx)).tolist(),
    }
    split_path.write_text(json.dumps(split, indent=2))
    return split


def prepare_dataset_summary() -> None:
    rows = []
    preprocessing_notes = {
        "purchase100": "train-only z-score standardization",
        "cifar10": "train: random crop padding=4 + horizontal flip + per-channel normalization; eval: per-channel normalization only",
    }
    for dataset in ["purchase100", "cifar10"]:
        cfg = DATASET_CONFIGS[dataset]
        if dataset == "purchase100":
            _, y = _load_purchase_arrays()
        else:
            train = datasets.CIFAR10(DATA_ROOT, train=True, download=True)
            y = np.asarray(train.targets, dtype=np.int64)
        for seed in [11, 22, 33]:
            split = prepare_splits(dataset, seed)
            row = asdict(cfg)
            row["seed"] = seed
            row["split_path"] = str((OUTPUT_ROOT / "metrics" / f"{dataset}_split_seed_{seed}.json").resolve())
            row["train_class_hist"] = json.dumps(np.bincount(y[split["train"]], minlength=cfg.num_classes).tolist())
            row["val_class_hist"] = json.dumps(np.bincount(y[split["val"]], minlength=cfg.num_classes).tolist())
            row["test_class_hist"] = json.dumps(np.bincount(y[split["test"]], minlength=cfg.num_classes).tolist())
            row["reference_class_hist"] = json.dumps(np.bincount(y[split["reference"]], minlength=cfg.num_classes).tolist())
            row["preprocessing"] = preprocessing_notes[dataset]
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "tables" / "dataset_summary.csv", index=False)


def make_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    def seed_worker(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        import random

        random.seed(worker_seed)

    return seed_worker


def get_datasets(dataset: str, seed: int):
    cfg = DATASET_CONFIGS[dataset]
    split = prepare_splits(dataset, seed)
    if dataset == "purchase100":
        x, y = _load_purchase_arrays()
        train_x = x[split["train"]]
        mean = train_x.mean(axis=0, keepdims=True)
        std = train_x.std(axis=0, keepdims=True) + 1e-6
        x = (x - mean) / std
        datasets_dict = {
            "train": IndexedArrayDataset(x, y, split["train"]),
            "val": IndexedArrayDataset(x, y, split["val"]),
            "test": IndexedArrayDataset(x, y, split["test"]),
            "reference": IndexedArrayDataset(x, y, split["reference"]),
        }
    else:
        norm = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        train_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
        eval_tf = transforms.Compose([transforms.ToTensor(), norm])
        train_base = datasets.CIFAR10(DATA_ROOT, train=True, download=True)
        test_base = datasets.CIFAR10(DATA_ROOT, train=False, download=True)
        train_x = np.asarray(train_base.data)
        train_y = np.asarray(train_base.targets, dtype=np.int64)
        test_x = np.asarray(test_base.data)
        test_y = np.asarray(test_base.targets, dtype=np.int64)
        datasets_dict = {
            "train": IndexedArrayDataset(train_x, train_y, split["train"], transform=train_tf),
            "val": IndexedArrayDataset(train_x, train_y, split["val"], transform=eval_tf),
            "reference": IndexedArrayDataset(train_x, train_y, split["reference"], transform=eval_tf),
            "test": IndexedArrayDataset(test_x, test_y, split["test"], transform=eval_tf),
        }
    return cfg, split, datasets_dict
