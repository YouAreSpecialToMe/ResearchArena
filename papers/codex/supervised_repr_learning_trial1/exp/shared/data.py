import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .utils import data_root, ensure_dir, write_json


CIFAR100_COARSE = np.array([
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13
], dtype=np.int64)


@dataclass
class DatasetBundle:
    dataset_name: str
    train_records: List[Dict]
    val_records: List[Dict]
    test_records: List[Dict]
    num_coarse_classes: int
    num_fine_classes: int
    input_size: int
    coarse_names: List[str]
    fine_names: List[str]


class RecordDataset(Dataset):
    def __init__(self, records: List[Dict], transform=None, dual_view_transform=None) -> None:
        self.records = records
        self.transform = transform
        self.dual_view_transform = dual_view_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = record["image"]
        if self.dual_view_transform is not None:
            v1 = self.dual_view_transform(image)
            v2 = self.dual_view_transform(image)
            eval_img = self.transform(image) if self.transform is not None else image
        else:
            eval_img = self.transform(image) if self.transform is not None else image
            v1 = eval_img
            v2 = eval_img
        return {
            "image": eval_img,
            "view1": v1,
            "view2": v2,
            "coarse_label": int(record["coarse_label"]),
            "fine_label": int(record["fine_label"]),
            "sample_id": int(record["sample_id"]),
            "coarse_name": record["coarse_name"],
            "fine_name": record["fine_name"],
        }


def imagenet_norm():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def cifar_eval_transform():
    return transforms.Compose([transforms.ToTensor(), imagenet_norm()])


def cifar_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_norm(),
    ])


def cifar_dual_view_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        imagenet_norm(),
    ])


def pet_eval_transform():
    return transforms.Compose([
        transforms.Resize(192),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        imagenet_norm(),
    ])


def pet_train_transform():
    return transforms.Compose([
        transforms.Resize(192),
        transforms.RandomCrop(160),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        imagenet_norm(),
    ])


def pet_dual_view_transform():
    return transforms.Compose([
        transforms.Resize(192),
        transforms.RandomCrop(160),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        imagenet_norm(),
    ])


def _strip_image(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return transforms.ToPILImage()(img)


def _save_split(split_path: Path, payload: Dict) -> None:
    write_json(split_path, payload)


def load_or_create_cifar100_splits() -> DatasetBundle:
    root = data_root()
    split_path = root / "cifar100_splits.json"
    cifar_present = (root / "cifar-100-python").exists()
    train_ds = datasets.CIFAR100(root=root, train=True, download=not cifar_present)
    test_ds = datasets.CIFAR100(root=root, train=False, download=False)

    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
    else:
        labels = CIFAR100_COARSE[np.array(train_ds.targets)]
        indices = np.arange(len(train_ds))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=5000,
            random_state=0,
            stratify=labels,
        )
        split = {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        }
        _save_split(split_path, split)

    def build_records(ds, indices, sample_offset=0):
        records = []
        for i in indices:
            fine = int(ds.targets[i])
            coarse = int(CIFAR100_COARSE[fine])
            records.append({
                "image": _strip_image(ds.data[i]),
                "coarse_label": coarse,
                "fine_label": fine,
                "sample_id": sample_offset + int(i),
                "coarse_name": ds.classes[CIFAR100_COARSE == coarse][0] if False else f"superclass_{coarse}",
                "fine_name": ds.classes[fine],
            })
        return records

    train_records = build_records(train_ds, train_idx, sample_offset=0)
    val_records = build_records(train_ds, val_idx, sample_offset=0)
    test_records = build_records(test_ds, range(len(test_ds)), sample_offset=100000)
    coarse_names = [f"superclass_{i}" for i in range(20)]
    return DatasetBundle(
        dataset_name="cifar100",
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        num_coarse_classes=20,
        num_fine_classes=100,
        input_size=32,
        coarse_names=coarse_names,
        fine_names=list(train_ds.classes),
    )


def load_or_create_oxford_pet_splits() -> DatasetBundle:
    root = data_root()
    split_path = root / "oxford_pet_splits.json"
    pet_present = any((root / candidate).exists() for candidate in ["oxford-iiit-pet", "oxford-iiit-pet.tar.gz"])
    trainval = datasets.OxfordIIITPet(root=root, split="trainval", target_types="category", download=not pet_present)
    test_ds = datasets.OxfordIIITPet(root=root, split="test", target_types="category", download=False)
    fine_names = trainval.classes

    def coarse_from_breed(name: str) -> Tuple[int, str]:
        return (0, "cat") if name[0].isupper() else (1, "dog")

    trainval_fine = np.array(trainval._labels)
    if split_path.exists():
        split = json.loads(split_path.read_text())
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
    else:
        indices = np.arange(len(trainval))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=max(1, int(0.15 * len(indices))),
            random_state=0,
            stratify=trainval_fine,
        )
        _save_split(split_path, {"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()})

    def build_records(ds, indices, sample_offset=0):
        records = []
        for i in indices:
            image, fine = ds[i]
            fine = int(fine)
            breed = fine_names[fine]
            coarse, coarse_name = coarse_from_breed(breed)
            records.append({
                "image": _strip_image(image),
                "coarse_label": coarse,
                "fine_label": fine,
                "sample_id": sample_offset + int(i),
                "coarse_name": coarse_name,
                "fine_name": breed,
            })
        return records

    train_records = build_records(trainval, train_idx, sample_offset=0)
    val_records = build_records(trainval, val_idx, sample_offset=0)
    test_records = build_records(test_ds, range(len(test_ds)), sample_offset=100000)
    return DatasetBundle(
        dataset_name="oxford_pet",
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        num_coarse_classes=2,
        num_fine_classes=len(fine_names),
        input_size=160,
        coarse_names=["cat", "dog"],
        fine_names=list(fine_names),
    )


def get_dataset_bundle(name: str) -> DatasetBundle:
    if name == "cifar100":
        return load_or_create_cifar100_splits()
    if name == "oxford_pet":
        return load_or_create_oxford_pet_splits()
    raise ValueError(f"Unknown dataset: {name}")


def build_datasets(bundle: DatasetBundle):
    if bundle.dataset_name == "cifar100":
        eval_t, train_t, dual_t = cifar_eval_transform(), cifar_train_transform(), cifar_dual_view_transform()
    else:
        eval_t, train_t, dual_t = pet_eval_transform(), pet_train_transform(), pet_dual_view_transform()
    return {
        "train_dual": RecordDataset(bundle.train_records, transform=eval_t, dual_view_transform=dual_t),
        "train_single": RecordDataset(bundle.train_records, transform=train_t, dual_view_transform=None),
        "train_eval": RecordDataset(bundle.train_records, transform=eval_t, dual_view_transform=None),
        "val_eval": RecordDataset(bundle.val_records, transform=eval_t, dual_view_transform=None),
        "test_eval": RecordDataset(bundle.test_records, transform=eval_t, dual_view_transform=None),
    }
