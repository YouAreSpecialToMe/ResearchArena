from __future__ import annotations

import io
import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageFilter

from .config import DATASETS, PAIR_CAPS, PAIR_ROOT, PROCESSED_ROOT, RAW_ROOT, SPLIT_SEED
from .utils import ensure_dir, write_json


@dataclass
class DatasetBundle:
    name: str
    images: np.ndarray
    factors: np.ndarray
    factor_names: list[str]
    factor_sizes: list[int]


def download_file(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def load_dsprites(raw_path: Path) -> DatasetBundle:
    blob = np.load(raw_path, allow_pickle=False, encoding="latin1")
    images = (blob["imgs"] * 255).astype(np.uint8)
    factors = blob["latents_classes"][:, 1:].astype(np.int64)
    meta = DATASETS["dsprites"]
    return DatasetBundle("dsprites", images, factors, meta["factors"], meta["factor_sizes"])


def load_shapes3d(raw_path: Path) -> DatasetBundle:
    meta = DATASETS["shapes3d"]
    with h5py.File(raw_path, "r") as handle:
        images = np.array(handle["images"], dtype=np.uint8)
        raw_labels = np.array(handle["labels"])
    columns = []
    for col in range(raw_labels.shape[1]):
        values = raw_labels[:, col]
        unique = np.unique(values)
        lookup = {float(value): idx for idx, value in enumerate(sorted(unique.tolist()))}
        columns.append(np.asarray([lookup[float(v)] for v in values], dtype=np.int64))
    labels = np.stack(columns, axis=1)
    return DatasetBundle("shapes3d", images, labels, meta["factors"], meta["factor_sizes"])


def ensure_raw_dataset(dataset_name: str) -> Path:
    meta = DATASETS[dataset_name]
    raw_path = RAW_ROOT / meta["raw_name"]
    download_file(meta["download_url"], raw_path)
    return raw_path


@lru_cache(maxsize=4)
def load_dataset(dataset_name: str) -> DatasetBundle:
    raw_path = ensure_raw_dataset(dataset_name)
    if dataset_name == "dsprites":
        return load_dsprites(raw_path)
    if dataset_name == "shapes3d":
        return load_shapes3d(raw_path)
    raise ValueError(dataset_name)


def _split_tuple_indices(factors: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    indices = np.arange(len(factors))
    discrete_keys = factors[:, 0]
    train, val, test = [], [], []
    for key in np.unique(discrete_keys):
        block = indices[discrete_keys == key]
        rng.shuffle(block)
        n = len(block)
        n_train = int(round(n * 0.7))
        n_val = int(round(n * 0.15))
        n_test = n - n_train - n_val
        train.append(block[:n_train])
        val.append(block[n_train:n_train + n_val])
        test.append(block[n_train + n_val:n_train + n_val + n_test])
    return {
        "train": np.concatenate(train),
        "val": np.concatenate(val),
        "test": np.concatenate(test),
    }


def prepare_splits(dataset_name: str) -> dict[str, Any]:
    out_dir = ensure_dir(PROCESSED_ROOT / dataset_name)
    if (out_dir / "pair_stats.json").exists() and (out_dir / "dataset_stats.json").exists():
        return read_json(out_dir / "dataset_stats.json")
    bundle = load_dataset(dataset_name)
    split_indices = _split_tuple_indices(bundle.factors)
    tuple_to_index = {tuple(row.tolist()): idx for idx, row in enumerate(bundle.factors)}
    split_payload = {}
    for split, idx in split_indices.items():
        payload = {
            "indices": idx.tolist(),
            "count": int(len(idx)),
        }
        write_json(out_dir / f"{split}_tuples.json", payload)
        split_payload[split] = idx
    stats = {
        "dataset": dataset_name,
        "num_images": int(len(bundle.images)),
        "num_tuples": int(len(bundle.factors)),
        "split_sizes": {split: int(len(idx)) for split, idx in split_indices.items()},
    }
    write_json(out_dir / "dataset_stats.json", stats)
    build_pairs(bundle, split_payload, tuple_to_index, out_dir)
    return stats


def build_pairs(
    bundle: DatasetBundle,
    split_indices: dict[str, np.ndarray],
    tuple_to_index: dict[tuple[int, ...], int],
    out_dir: Path,
) -> None:
    pair_root = ensure_dir(PAIR_ROOT / bundle.name)
    rng = np.random.default_rng(SPLIT_SEED)
    for split, indices in split_indices.items():
        split_set = set(indices.tolist())
        split_dir = ensure_dir(pair_root / split)
        factor_pairs: dict[str, list[list[int]]] = {name: [] for name in bundle.factor_names}
        for factor_idx, factor_name in enumerate(bundle.factor_names):
            candidates: list[list[int]] = []
            for idx in indices:
                src = bundle.factors[idx].copy()
                choices = [v for v in range(bundle.factor_sizes[factor_idx]) if v != src[factor_idx]]
                target_value = choices[rng.integers(0, len(choices))]
                src[factor_idx] = target_value
                j = tuple_to_index.get(tuple(src.tolist()))
                if j is not None and j in split_set:
                    candidates.append([int(idx), int(j)])
            rng.shuffle(candidates)
            cap = PAIR_CAPS[split]["counterfactual"] // len(bundle.factor_names)
            factor_pairs[factor_name] = candidates[:cap]
        nuisance_pairs = []
        noise_cap = min(PAIR_CAPS[split]["nuisance"], len(indices))
        chosen = rng.choice(indices, size=noise_cap, replace=False)
        for idx in chosen.tolist():
            nuisance_pairs.append(
                {
                    "index": int(idx),
                    "view1": sample_corruption_params(rng),
                    "view2": sample_corruption_params(rng),
                }
            )
        payload = {
            "factor_pairs": factor_pairs,
            "nuisance_pairs": nuisance_pairs,
        }
        write_json(split_dir / "pairs.json", payload)
    stats = {
        split: {
            "counterfactual_pairs_per_factor": {
                factor: len(read_json(pair_root / split / "pairs.json")["factor_pairs"][factor])
                for factor in bundle.factor_names
            },
            "nuisance_pairs": len(read_json(pair_root / split / "pairs.json")["nuisance_pairs"]),
        }
        for split in split_indices
    }
    write_json(out_dir / "pair_stats.json", stats)


def sample_corruption_params(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "gaussian_noise_sigma": float(rng.uniform(0.01, 0.05)),
        "gaussian_blur_sigma": float(rng.uniform(0.4, 1.0)),
        "jpeg_quality": int(rng.integers(50, 96)),
        "dead_pixel_frac": float(rng.uniform(0.005, 0.02)),
        "seed": int(rng.integers(0, 2**31 - 1)),
    }


def render_image(bundle: DatasetBundle, index: int, corruption: dict[str, Any] | None = None) -> Image.Image:
    image = bundle.images[index]
    if bundle.name == "dsprites":
        image = np.repeat(image[..., None], 3, axis=2)
    pil = Image.fromarray(image)
    if corruption is None:
        return pil
    rng = np.random.default_rng(corruption["seed"])
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr += rng.normal(0.0, corruption["gaussian_noise_sigma"], size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=corruption["gaussian_blur_sigma"]))
    dead = np.asarray(pil).copy()
    num_dead = max(1, int(dead.shape[0] * dead.shape[1] * corruption["dead_pixel_frac"]))
    flat = dead.reshape(-1, dead.shape[-1])
    dead_idx = rng.choice(flat.shape[0], size=num_dead, replace=False)
    flat[dead_idx] = 0
    pil = Image.fromarray(dead)
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=corruption["jpeg_quality"])
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
