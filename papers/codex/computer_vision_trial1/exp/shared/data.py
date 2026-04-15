import csv
import io
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100

from exp.shared.common import (
    BATCH_SIZE,
    CACHE_DIR,
    CORRUPTION_FAMILIES,
    DATA_DIR,
    MODEL_NAME,
    NUM_WORKERS,
    PRETRAINED_NAME,
    ROOT,
    SEEDS,
    ensure_dir,
    load_json,
    save_json,
    set_seed,
)


CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
CIFAR100C_URL = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
ALT_DATA_DIR = ROOT.parent / "idea_01" / "data"

CIFAR10_CLASSNAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR100_CLASSNAMES = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear", "pickup truck", "pine tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
]


def _maybe_symlink(src: Path, dst: Path) -> bool:
    if dst.exists() or not src.exists():
        return False
    ensure_dir(dst.parent)
    os.symlink(src.resolve(), dst)
    return True


def bootstrap_existing_data() -> None:
    if ALT_DATA_DIR.exists():
        for name in ["CIFAR-100-C", "CIFAR-100-C.tar", "cifar-100-python", "cifar-100-python.tar.gz"]:
            _maybe_symlink(ALT_DATA_DIR / name, DATA_DIR / name)


def _download(url: str, destination: Path) -> None:
    ensure_dir(destination.parent)
    urllib.request.urlretrieve(url, destination)


def _extract_tar(archive_path: Path, destination: Path) -> None:
    ensure_dir(destination)
    with tarfile.open(archive_path) as handle:
        handle.extractall(destination)


def ensure_cifar10c() -> Path:
    target = DATA_DIR / "CIFAR-10-C"
    if target.exists():
        return target
    archive = DATA_DIR / "CIFAR-10-C.tar"
    if not archive.exists():
        _download(CIFAR10C_URL, archive)
    _extract_tar(archive, DATA_DIR)
    return target


def ensure_cifar100c() -> Path:
    target = DATA_DIR / "CIFAR-100-C"
    if target.exists():
        return target
    archive = DATA_DIR / "CIFAR-100-C.tar"
    if not archive.exists():
        _download(CIFAR100C_URL, archive)
    _extract_tar(archive, DATA_DIR)
    return target


def ensure_clean_datasets() -> None:
    ensure_dir(DATA_DIR)
    CIFAR10(root=DATA_DIR, train=True, download=True)
    CIFAR10(root=DATA_DIR, train=False, download=True)
    CIFAR100(root=DATA_DIR, train=False, download=True)


def ensure_all_datasets() -> None:
    bootstrap_existing_data()
    ensure_clean_datasets()
    ensure_cifar10c()
    ensure_cifar100c()


def _gaussian_noise(arr: np.ndarray, severity: int = 3, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    std = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]
    noisy = arr + rng.normal(0.0, std * 255.0, arr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _motion_blur(image: Image.Image, severity: int = 3) -> Image.Image:
    radius = [1.0, 1.8, 2.5, 3.2, 4.0][severity - 1]
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _fog(arr: np.ndarray, severity: int = 3) -> np.ndarray:
    strength = [0.12, 0.18, 0.24, 0.30, 0.38][severity - 1]
    haze = np.full_like(arr, 255, dtype=np.float32)
    out = arr.astype(np.float32) * (1.0 - strength) + haze * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _jpeg_compression(image: Image.Image, severity: int = 3) -> Image.Image:
    quality = [50, 35, 25, 18, 10][severity - 1]
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_corruption(image: Image.Image, corruption: str, severity: int = 3, rng: np.random.Generator | None = None) -> Image.Image:
    image = image.convert("RGB")
    if corruption == "gaussian_noise":
        return Image.fromarray(_gaussian_noise(np.asarray(image), severity=severity, rng=rng))
    if corruption == "motion_blur":
        return _motion_blur(image, severity=severity)
    if corruption == "fog":
        return Image.fromarray(_fog(np.asarray(image), severity=severity))
    if corruption == "jpeg_compression":
        return _jpeg_compression(image, severity=severity)
    raise ValueError(f"Unknown corruption: {corruption}")


@dataclass
class FeatureBundle:
    features: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    metadata: list[dict]


class NumpyImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, preprocess, metadata: list[dict] | None = None):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.preprocess = preprocess
        self.metadata = metadata or [{} for _ in range(len(images))]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index]).convert("RGB")
        return {
            "image": self.preprocess(image),
            "label": int(self.labels[index]),
            "index": index,
            "metadata": self.metadata[index],
        }


class PilotCorruptionDataset(Dataset):
    def __init__(self, base_images: np.ndarray, labels: np.ndarray, preprocess, seed: int, families: Iterable[str], severity: int = 3):
        self.base_images = base_images
        self.labels = labels.astype(np.int64)
        self.preprocess = preprocess
        self.items = []
        for family in families:
            for idx in range(len(base_images)):
                self.items.append((idx, family))
        self.seed = seed
        self.severity = severity

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_index, family = self.items[index]
        image = Image.fromarray(self.base_images[image_index]).convert("RGB")
        rng = np.random.default_rng(self.seed * 100000 + image_index)
        corrupted = apply_corruption(image, family, severity=self.severity, rng=rng)
        return {
            "image": self.preprocess(corrupted),
            "label": int(self.labels[image_index]),
            "index": image_index,
            "metadata": {"family": family},
        }


def dataloader_for(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def save_feature_bundle(path: Path, bundle: FeatureBundle) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "features": bundle.features.half().cpu(),
            "labels": bundle.labels.cpu(),
            "indices": bundle.indices.cpu(),
            "metadata": bundle.metadata,
        },
        path,
    )


def load_feature_bundle(path: Path) -> FeatureBundle:
    payload = torch.load(path, map_location="cpu")
    return FeatureBundle(
        features=payload["features"].float(),
        labels=payload["labels"].long(),
        indices=payload["indices"].long(),
        metadata=payload["metadata"],
    )


def encode_dataset(model, dataset: Dataset, device: torch.device) -> FeatureBundle:
    loader = dataloader_for(dataset)
    feature_chunks = []
    labels = []
    indices = []
    metadata = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            features = model.encode_image(images)
        features = torch.nn.functional.normalize(features.float(), dim=-1).cpu()
        feature_chunks.append(features)
        labels.append(batch["label"].long().cpu())
        indices.append(batch["index"].long().cpu())
        batch_meta = batch["metadata"]
        if isinstance(batch_meta, list):
            metadata.extend(batch_meta)
        elif isinstance(batch_meta, dict):
            keys = list(batch_meta.keys())
            length = len(batch["label"])
            for sample_idx in range(length):
                metadata.append(
                    {
                        key: batch_meta[key][sample_idx]
                        if isinstance(batch_meta[key], list)
                        else batch_meta[key]
                        for key in keys
                    }
                )
        else:
            metadata.extend([{} for _ in range(len(batch["label"]))])
    return FeatureBundle(
        features=torch.cat(feature_chunks, dim=0),
        labels=torch.cat(labels, dim=0),
        indices=torch.cat(indices, dim=0),
        metadata=metadata,
    )


def _feature_path(*parts: str) -> Path:
    return CACHE_DIR.joinpath(*parts)


def prepare_proxy_splits() -> dict[str, dict]:
    ensure_dir(DATA_DIR / "proxy_splits")
    dataset = CIFAR10(root=DATA_DIR, train=True, download=True)
    images = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets)
    summary = {}
    for seed in SEEDS:
        split_path = DATA_DIR / "proxy_splits" / f"seed_{seed}.json"
        meta_csv = DATA_DIR / "proxy_splits" / f"seed_{seed}_metadata.csv"
        if split_path.exists() and meta_csv.exists():
            summary[str(seed)] = load_json(split_path)
            continue
        set_seed(seed)
        rng = np.random.default_rng(seed)
        picks = rng.choice(np.arange(len(images)), size=2000, replace=False)
        rng.shuffle(picks)
        pilot_indices = picks[:500]
        proxy_indices = picks[500:]
        payload = {
            "seed": seed,
            "pilot_holdout_indices": pilot_indices.tolist(),
            "proxy_indices": proxy_indices.tolist(),
            "severity": 3,
            "families": CORRUPTION_FAMILIES,
        }
        save_json(split_path, payload)
        with meta_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["image_id", "seed", "source_index", "corruption_family", "severity", "split"],
            )
            writer.writeheader()
            for split_name, indices in [("pilot_holdout", pilot_indices), ("proxy", proxy_indices)]:
                for source_index in indices:
                    writer.writerow(
                        {
                            "image_id": int(source_index),
                            "seed": seed,
                            "source_index": int(source_index),
                            "corruption_family": "clean",
                            "severity": 0,
                            "split": split_name,
                        }
                    )
                    for family in CORRUPTION_FAMILIES:
                        writer.writerow(
                            {
                                "image_id": int(source_index),
                                "seed": seed,
                                "source_index": int(source_index),
                                "corruption_family": family,
                                "severity": 3,
                                "split": split_name,
                            }
                        )
        summary[str(seed)] = payload
    return summary


def cache_clean_features(model, preprocess, device: torch.device) -> dict[str, str]:
    manifest = {}
    dataset_specs = [
        ("cifar10_test_clean", CIFAR10(root=DATA_DIR, train=False, download=True), "cifar10", 10),
        ("cifar100_test_clean", CIFAR100(root=DATA_DIR, train=False, download=True), "cifar100", 100),
    ]
    for cache_name, dataset, dataset_key, n_classes in dataset_specs:
        path = _feature_path(f"{cache_name}.pt")
        manifest[cache_name] = str(path)
        if path.exists():
            continue
        bundle = encode_dataset(
            model,
            NumpyImageDataset(np.asarray(dataset.data), np.asarray(dataset.targets), preprocess),
            device=device,
        )
        save_feature_bundle(path, bundle)
        save_json(
            _feature_path(f"{cache_name}.meta.json"),
            {"dataset": dataset_key, "split": "test", "classes": n_classes, "num_images": len(dataset)},
        )
    return manifest


def cache_cifar_c_features(model, preprocess, device: torch.device) -> dict[str, str]:
    manifest = {}
    roots = {
        "cifar10": ensure_cifar10c(),
        "cifar100": ensure_cifar100c(),
    }
    label_files = {
        "cifar10": np.load(roots["cifar10"] / "labels.npy"),
        "cifar100": np.load(roots["cifar100"] / "labels.npy"),
    }
    for dataset_name, root in roots.items():
        labels = label_files[dataset_name]
        for family in CORRUPTION_FAMILIES:
            all_images = np.load(root / f"{family}.npy")
            for severity in [1, 2, 3, 4, 5]:
                start = (severity - 1) * 10000
                end = severity * 10000
                path = _feature_path(f"{dataset_name}_{family}_severity_{severity}.pt")
                manifest[f"{dataset_name}_{family}_severity_{severity}"] = str(path)
                if path.exists():
                    continue
                metadata = [{"family": family, "severity": severity} for _ in range(10000)]
                bundle = encode_dataset(
                    model,
                    NumpyImageDataset(all_images[start:end], labels, preprocess, metadata=metadata),
                    device=device,
                )
                save_feature_bundle(path, bundle)
    return manifest


def cache_pilot_features(model, preprocess, device: torch.device) -> dict[str, str]:
    manifest = {}
    split_summary = prepare_proxy_splits()
    dataset = CIFAR10(root=DATA_DIR, train=True, download=True)
    images = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets)
    for seed in SEEDS:
        split = split_summary[str(seed)]
        selected = np.asarray(split["pilot_holdout_indices"] + split["proxy_indices"], dtype=np.int64)
        clean_path = _feature_path("pilot", f"seed_{seed}_clean.pt")
        manifest[f"pilot_seed_{seed}_clean"] = str(clean_path)
        if not clean_path.exists():
            metadata = [{"family": "clean"} for _ in range(len(selected))]
            bundle = encode_dataset(
                model,
                NumpyImageDataset(images[selected], labels[selected], preprocess, metadata=metadata),
                device=device,
            )
            bundle.indices = torch.as_tensor(selected, dtype=torch.long)
            save_feature_bundle(clean_path, bundle)
        corr_path = _feature_path("pilot", f"seed_{seed}_corrupted.pt")
        manifest[f"pilot_seed_{seed}_corrupted"] = str(corr_path)
        if not corr_path.exists():
            bundle = encode_dataset(
                model,
                PilotCorruptionDataset(
                    base_images=images[selected],
                    labels=labels[selected],
                    preprocess=preprocess,
                    seed=seed,
                    families=CORRUPTION_FAMILIES,
                    severity=3,
                ),
                device=device,
            )
            expanded_indices = []
            expanded_meta = []
            for family in CORRUPTION_FAMILIES:
                for source_index in selected:
                    expanded_indices.append(int(source_index))
                    expanded_meta.append({"family": family, "severity": 3})
            bundle.indices = torch.as_tensor(expanded_indices, dtype=torch.long)
            bundle.metadata = expanded_meta
            save_feature_bundle(corr_path, bundle)
    return manifest


def prepare_all_features(model, preprocess, device: torch.device) -> dict[str, str]:
    ensure_all_datasets()
    ensure_dir(CACHE_DIR)
    manifest = {}
    manifest.update(cache_clean_features(model, preprocess, device))
    manifest.update(cache_cifar_c_features(model, preprocess, device))
    manifest.update(cache_pilot_features(model, preprocess, device))
    save_json(CACHE_DIR / "manifest.json", manifest)
    return manifest
