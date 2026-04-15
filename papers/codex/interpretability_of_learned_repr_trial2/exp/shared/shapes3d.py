from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


FACTOR_NAMES = [
    "floor_hue",
    "wall_hue",
    "object_hue",
    "scale",
    "shape",
    "orientation",
]
FACTOR_SIZES = [10, 10, 10, 8, 4, 15]
FACTOR_TO_ID = {name: i for i, name in enumerate(FACTOR_NAMES)}


@dataclass
class Shapes3DPaths:
    h5_path: Path
    split_csv: Path
    pair_metadata: Path


class Shapes3DIndex:
    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            raw_labels = np.asarray(f["labels"], dtype=np.float32)
        self.raw_labels = raw_labels
        self.factor_values = [np.unique(raw_labels[:, i]) for i in range(raw_labels.shape[1])]
        self.labels = np.stack(
            [np.searchsorted(self.factor_values[i], raw_labels[:, i]).astype(np.int64) for i in range(raw_labels.shape[1])],
            axis=1,
        )
        self.length = int(self.labels.shape[0])
        self.lookup = {tuple(int(x) for x in row.tolist()): idx for idx, row in enumerate(self.labels)}

    def tuple_to_index(self, factors) -> int:
        return self.lookup[tuple(int(x) for x in factors)]

    def counterfactual_index(self, base_idx: int, factor_id: int, new_value: int) -> int:
        factors = self.labels[base_idx].copy()
        factors[factor_id] = new_value
        return self.tuple_to_index(factors)

    def get_labels(self, indices) -> np.ndarray:
        return self.labels[np.asarray(indices, dtype=np.int64)]

    def load_images(self, indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        unique_indices, inverse = np.unique(indices, return_inverse=True)
        with h5py.File(self.h5_path, "r") as f:
            images = np.asarray(f["images"][unique_indices], dtype=np.uint8)
        return images[inverse]


def _target_level_counts(split_size: int) -> list[np.ndarray]:
    targets = []
    for size in FACTOR_SIZES:
        base = split_size // size
        remainder = split_size % size
        counts = np.full(size, base, dtype=np.int64)
        counts[:remainder] += 1
        targets.append(counts)
    return targets


def _greedy_factor_balanced_sample(labels: np.ndarray, indices: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    targets = _target_level_counts(n)
    current = [np.zeros_like(t) for t in targets]
    selected = []
    for idx in order.tolist():
        row = labels[idx]
        if all(current[f][row[f]] < targets[f][row[f]] for f in range(len(FACTOR_NAMES))):
            selected.append(idx)
            for f in range(len(FACTOR_NAMES)):
                current[f][row[f]] += 1
            if len(selected) == n:
                return np.asarray(selected, dtype=np.int64)
    selected_set = set(selected)
    remaining = [idx for idx in order.tolist() if idx not in selected_set]
    selected.extend(remaining[: max(0, n - len(selected))])
    return np.asarray(selected[:n], dtype=np.int64)


def sample_stratified_split(index: Shapes3DIndex, seed: int, train_n=18000, val_n=3000, test_n=3000) -> pd.DataFrame:
    labels = index.labels
    all_indices = np.arange(index.length, dtype=np.int64)
    train_idx = _greedy_factor_balanced_sample(labels, all_indices, train_n, seed)
    remaining = np.setdiff1d(all_indices, train_idx, assume_unique=False)
    val_idx = _greedy_factor_balanced_sample(labels, remaining, val_n, seed + 1)
    remaining = np.setdiff1d(remaining, val_idx, assume_unique=False)
    test_idx = _greedy_factor_balanced_sample(labels, remaining, test_n, seed + 2)
    rows = []
    for split, ids in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        for idx in ids.tolist():
            row = {"image_id": int(idx), "split": split}
            for name, value in zip(FACTOR_NAMES, labels[idx].tolist()):
                row[name] = int(value)
            rows.append(row)
    return pd.DataFrame(rows)


def sample_balanced_subset(split_df: pd.DataFrame, split_sizes: dict[str, int], seed: int) -> pd.DataFrame:
    subset_parts = []
    for split, size in split_sizes.items():
        frame = split_df[split_df["split"] == split].reset_index(drop=True)
        labels = frame[FACTOR_NAMES].to_numpy(dtype=np.int64)
        chosen_local = _greedy_factor_balanced_sample(labels, np.arange(len(frame), dtype=np.int64), size, seed + len(subset_parts))
        subset_parts.append(frame.iloc[chosen_local].copy())
    return pd.concat(subset_parts, ignore_index=True)


def build_counterfactual_pairs(index: Shapes3DIndex, source_ids, split: str, n_pairs: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    rows = []
    per_factor = n_pairs // len(FACTOR_NAMES)
    remainder = n_pairs % len(FACTOR_NAMES)
    for factor_id, factor_name in enumerate(FACTOR_NAMES):
        count = per_factor + int(factor_id < remainder)
        chosen_sources = rng.choice(source_ids, size=count, replace=True)
        for src in chosen_sources.tolist():
            src = int(src)
            src_labels = index.labels[src].copy()
            choices = [v for v in range(FACTOR_SIZES[factor_id]) if v != int(src_labels[factor_id])]
            new_value = int(rng.choice(choices))
            tgt = index.counterfactual_index(src, factor_id, new_value)
            tgt_labels = index.labels[tgt]
            row = {
                "split": split,
                "pair_type": "counterfactual",
                "source_id": src,
                "target_id": int(tgt),
                "changed_factor": factor_name,
                "aug_seed": None,
                "aug_kind": None,
            }
            for name, value in zip(FACTOR_NAMES, src_labels.tolist()):
                row[f"source_{name}"] = int(value)
            for name, value in zip(FACTOR_NAMES, tgt_labels.tolist()):
                row[f"target_{name}"] = int(value)
            rows.append(row)
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def build_nuisance_pairs(source_ids, labels, split: str, n_pairs: int, seed: int, include_crop: bool, aug_kinds=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    rows = []
    if aug_kinds is None:
        aug_kinds = ["photo_full", "crop_photo"] if include_crop else ["photo_full"]
    chosen = rng.choice(source_ids, size=n_pairs, replace=True)
    for idx, src in enumerate(chosen.tolist()):
        synthetic_id = -int(seed * 1_000_000 + idx + 1)
        src = int(src)
        src_labels = labels[src]
        row = {
            "split": split,
            "pair_type": "nuisance",
            "source_id": src,
            "target_id": synthetic_id,
            "changed_factor": "none",
            "aug_seed": int(seed * 10_000 + idx),
            "aug_kind": str(rng.choice(aug_kinds)),
        }
        for name, value in zip(FACTOR_NAMES, src_labels.tolist()):
            row[f"source_{name}"] = int(value)
            row[f"target_{name}"] = int(value)
        rows.append(row)
    return pd.DataFrame(rows)


def save_pair_metadata(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
