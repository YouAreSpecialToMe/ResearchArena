from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets
from torchvision import transforms

from .config import DATA_DIR, REFERENCE_MODELS


class IndexedTensorDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], idx


class IndexedVisionSubset(Dataset):
    def __init__(self, dataset, indices: np.ndarray, transform=None) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = int(self.indices[idx])
        x, y = self.dataset[base_idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, base_idx


@dataclass
class DatasetBundle:
    name: str
    train_dataset: Dataset
    eval_dataset: Dataset
    labels_all: np.ndarray
    target_test_labels: np.ndarray
    input_dim: int | None = None
    num_classes: int = 0


def _stratified_split(indices: np.ndarray, labels: np.ndarray, sizes: List[float], seed: int) -> List[np.ndarray]:
    remaining_idx = indices.copy()
    remaining_y = labels.copy()
    out = []
    rng_seed = seed
    for frac in sizes[:-1]:
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=rng_seed)
        chosen, rest = next(splitter.split(remaining_idx, remaining_y))
        out.append(remaining_idx[chosen])
        remaining_idx = remaining_idx[rest]
        remaining_y = remaining_y[rest]
        rng_seed += 1
    out.append(remaining_idx)
    return out


def load_cifar10() -> DatasetBundle:
    DATA_DIR.mkdir(exist_ok=True)
    train_base = tv_datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test_base = tv_datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)
    eval_tf = transforms.Compose([transforms.ToTensor()])
    train_dataset = IndexedVisionSubset(train_base, np.arange(len(train_base)), transform=eval_tf)
    test_dataset = IndexedVisionSubset(test_base, np.arange(len(test_base)), transform=eval_tf)
    labels_all = np.asarray(train_base.targets)
    target_test_labels = np.asarray(test_base.targets)
    return DatasetBundle(
        name="cifar10",
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        labels_all=labels_all,
        target_test_labels=target_test_labels,
        num_classes=10,
    )


def load_purchase100() -> DatasetBundle:
    DATA_DIR.mkdir(exist_ok=True)
    arrow_path = next((DATA_DIR / "purchase100_cache").rglob("purchase100-train.arrow"))
    with pa.memory_map(str(arrow_path), "r") as source:
        table = ipc.RecordBatchStreamReader(source).read_all()
    features = np.asarray(table["feature"].to_pylist(), dtype=np.float32)
    labels = table["label"].to_numpy().astype(np.int64)
    dataset = IndexedTensorDataset(features, labels)
    return DatasetBundle(
        name="purchase100",
        train_dataset=dataset,
        eval_dataset=dataset,
        labels_all=labels,
        target_test_labels=labels,
        input_dim=features.shape[1],
        num_classes=int(labels.max() + 1),
    )


def make_splits(bundle: DatasetBundle, seed: int, out_dir: Path) -> Dict[str, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "split_indices.json"
    if split_path.exists():
        raw = json.loads(split_path.read_text())
        parts = {k: np.asarray(v, dtype=np.int64) if k != "reference_subsets" else [np.asarray(x, dtype=np.int64) for x in v] for k, v in raw.items()}
        verify_non_overlap(parts, separate_test_namespace=(bundle.name == "cifar10"))
        return parts

    indices = np.arange(len(bundle.labels_all))
    labels = bundle.labels_all
    if bundle.name == "cifar10":
        splitter_tv = StratifiedShuffleSplit(n_splits=1, train_size=25000, random_state=seed)
        target_train_sub, remainder = next(splitter_tv.split(indices, labels))
        target_train = indices[target_train_sub]
        rem_idx = indices[remainder]
        rem_labels = labels[rem_idx]
        splitter_ref = StratifiedShuffleSplit(n_splits=1, train_size=20000, random_state=seed + 1)
        ref_sub, val_sub = next(splitter_ref.split(rem_idx, rem_labels))
        ref_pool = rem_idx[ref_sub]
        target_val = rem_idx[val_sub]
        target_test_nm = np.arange(len(bundle.target_test_labels), dtype=np.int64)
    else:
        first = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=seed)
        train_pool_idx, target_test_nm = next(first.split(indices, labels))
        train_pool_labels = labels[train_pool_idx]
        second = StratifiedShuffleSplit(n_splits=1, train_size=5 / 9, random_state=seed + 1)
        target_train_sub, remain_sub = next(second.split(train_pool_idx, train_pool_labels))
        target_train = train_pool_idx[target_train_sub]
        remain_idx = train_pool_idx[remain_sub]
        remain_labels = labels[remain_idx]
        third = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=seed + 2)
        ref_sub, val_sub = next(third.split(remain_idx, remain_labels))
        ref_pool = remain_idx[ref_sub]
        target_val = remain_idx[val_sub]

    tv_labels = labels[target_val]
    splitter_cal = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 3)
    cal_sub, select_sub = next(splitter_cal.split(target_val, tv_labels))
    defense_cal = target_val[cal_sub]
    defense_select = target_val[select_sub]

    ref_labels = labels[ref_pool]
    reference_subsets = []
    for ref_id in range(REFERENCE_MODELS):
        splitter_ref_model = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 10 + ref_id)
        chosen, _ = next(splitter_ref_model.split(ref_pool, ref_labels))
        reference_subsets.append(ref_pool[chosen])

    split_dict = {
        "target_train": target_train.tolist(),
        "ref_pool": ref_pool.tolist(),
        "target_val": target_val.tolist(),
        "defense_cal": defense_cal.tolist(),
        "defense_select": defense_select.tolist(),
        "target_test_nm": target_test_nm.tolist(),
        "reference_subsets": [x.tolist() for x in reference_subsets],
    }
    split_path.write_text(json.dumps(split_dict, indent=2))

    parts = {
        "target_train": target_train,
        "ref_pool": ref_pool,
        "target_val": target_val,
        "defense_cal": defense_cal,
        "defense_select": defense_select,
        "target_test_nm": target_test_nm,
        "reference_subsets": reference_subsets,
    }
    verify_non_overlap(parts, separate_test_namespace=(bundle.name == "cifar10"))
    return parts


def verify_non_overlap(parts: Dict[str, np.ndarray | List[np.ndarray]], separate_test_namespace: bool = False) -> None:
    names = ["target_train", "ref_pool", "target_val", "target_test_nm"]
    arrays = {name: set(map(int, parts[name])) for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if separate_test_namespace and "target_test_nm" in {left, right}:
                continue
            if arrays[left].intersection(arrays[right]):
                raise RuntimeError(f"Data leakage between {left} and {right}")


def make_loader(dataset: Dataset, indices: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def fit_purchase_scaler(dataset: IndexedTensorDataset, train_indices: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(dataset.x[train_indices].numpy())
    return scaler


def apply_purchase_scaler(dataset: IndexedTensorDataset, scaler: StandardScaler) -> IndexedTensorDataset:
    x = scaler.transform(dataset.x.numpy()).astype(np.float32)
    y = dataset.y.numpy().astype(np.int64)
    return IndexedTensorDataset(x, y)
