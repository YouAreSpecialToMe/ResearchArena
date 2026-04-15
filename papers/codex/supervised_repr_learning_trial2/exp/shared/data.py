import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from exp.shared.utils import ROOT, ensure_dir, save_json


SEEDS = [11, 22, 33]


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform, dataset_name, split_name, indices=None):
        self.hf_dataset = hf_dataset if indices is None else hf_dataset.select(indices)
        self.transform = transform
        self.dataset_name = dataset_name
        self.split_name = split_name

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[int(idx)]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        item = {
            "pixel_values": self.transform(image.convert("RGB")),
            "label": int(row["label"]),
            "sample_id": f"{self.dataset_name}_{self.split_name}_{idx:06d}",
        }
        if self.dataset_name == "waterbirds":
            place = int(row["place"])
            label = int(row["label"])
            item["group"] = label * 2 + place
            item["place"] = place
        else:
            item["group"] = -1
        return item


@dataclass
class PreparedDataset:
    name: str
    num_classes: int
    train_ds: object
    val_ds: object
    test_ds: object
    group_available: bool


def load_prepared_dataset(dataset_name: str, data_root: Path) -> PreparedDataset:
    from datasets import load_dataset

    data_root = Path(data_root)
    if dataset_name == "waterbirds":
        ds = load_dataset("grodino/waterbirds", cache_dir=str(data_root / "hf_cache"))
        num_classes = ds["train"].features["label"].num_classes
        return PreparedDataset(dataset_name, num_classes, ds["train"], ds["validation"], ds["test"], True)
    if dataset_name == "cub":
        ds = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", cache_dir=str(data_root / "hf_cache"))
        labels = np.asarray(ds["train"]["label"])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=123)
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
        save_json(ROOT / "cache" / "splits" / "cub_val_ids.json", [f"cub_train_{i:06d}" for i in val_idx.tolist()])
        return PreparedDataset(
            dataset_name,
            ds["train"].features["label"].num_classes,
            ds["train"].select(train_idx.tolist()),
            ds["train"].select(val_idx.tolist()),
            ds["test"],
            False,
        )
    raise ValueError(dataset_name)


def build_transforms():
    _, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=preprocess.transforms[-1].mean, std=preprocess.transforms[-1].std),
        ]
    )
    return preprocess, aug


def _extract_for_split(model, dataset, batch_size=128, num_workers=4, device="cuda"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, labels, sample_ids, groups = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            out = model.encode_image(pixels)
            out = torch.nn.functional.normalize(out, dim=-1).cpu().to(torch.float16)
            feats.append(out)
            labels.append(batch["label"].to(torch.int64).cpu())
            groups.append(batch["group"].to(torch.int64).cpu())
            sample_ids.extend(batch["sample_id"])
    return {
        "features": torch.cat(feats, dim=0),
        "labels": torch.cat(labels, dim=0),
        "groups": torch.cat(groups, dim=0),
        "ids": sample_ids,
    }


def cache_features_for_dataset(dataset_name: str, data_root: Path, cache_root: Path, device="cuda"):
    prepared = load_prepared_dataset(dataset_name, data_root)
    preprocess, aug = build_transforms()
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    model = model.eval().to(device)

    base_dir = ensure_dir(cache_root / "features" / dataset_name / "clip_vit_b16")
    split_defs = {
        "train_base": HFDatasetWrapper(prepared.train_ds, preprocess, dataset_name, "train"),
        "train_aug1": HFDatasetWrapper(prepared.train_ds, aug, dataset_name, "train"),
        "train_aug2": HFDatasetWrapper(prepared.train_ds, aug, dataset_name, "train"),
        "val": HFDatasetWrapper(prepared.val_ds, preprocess, dataset_name, "val"),
        "test": HFDatasetWrapper(prepared.test_ds, preprocess, dataset_name, "test"),
    }
    stats = {"dataset": dataset_name, "num_classes": prepared.num_classes, "splits": {}, "checksums": {}}
    seen_ids = set()
    for split_name, wrapped in split_defs.items():
        payload = _extract_for_split(model, wrapped, device=device)
        out_path = base_dir / f"{split_name}.pt"
        torch.save(payload, out_path)
        ids = payload["ids"]
        overlap = len(seen_ids.intersection(ids))
        if split_name in {"train_aug1", "train_aug2"}:
            overlap = 0
        else:
            seen_ids.update(ids)
        stats["splits"][split_name] = {
            "size": len(ids),
            "feature_dim": int(payload["features"].shape[-1]),
            "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(payload["labels"].numpy(), return_counts=True))},
            "group_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(payload["groups"].numpy(), return_counts=True))},
            "id_overlap_with_previous": overlap,
        }
        stats["checksums"][split_name] = hashlib.sha256(payload["features"].numpy().tobytes()).hexdigest()
    save_json(cache_root / "features" / dataset_name / "stats.json", stats)
    return stats


def load_cached_split(dataset_name: str, split_name: str):
    return torch.load(ROOT / "cache" / "features" / dataset_name / "clip_vit_b16" / f"{split_name}.pt", map_location="cpu")
