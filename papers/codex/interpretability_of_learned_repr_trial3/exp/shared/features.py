from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import open_clip
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import BACKBONES, FEATURE_ROOT, PROCESSED_ROOT
from .data import load_dataset, render_image
from .utils import ensure_dir, preferred_amp_dtype, select_device, write_json


class ImageDataset(Dataset):
    def __init__(self, dataset_name: str, indices: np.ndarray, backbone_name: str, nuisance: list[dict] | None = None) -> None:
        self.bundle = load_dataset(dataset_name)
        self.indices = indices
        self.nuisance = nuisance
        meta = BACKBONES[backbone_name]
        self.transform = transforms.Compose(
            [
                transforms.Resize((meta["image_size"], meta["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=meta["mean"], std=meta["std"]),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        index = int(self.indices[idx])
        corruption = None if self.nuisance is None else self.nuisance[idx]
        image = render_image(self.bundle, index, corruption=corruption)
        return self.transform(image), index


def build_backbone(backbone_name: str) -> tuple[torch.nn.Module, callable]:
    meta = BACKBONES[backbone_name]
    if meta["kind"] == "timm":
        model = timm.create_model(meta["model_name"], pretrained=True, num_classes=0, img_size=meta["image_size"])
        model.eval()
        return model, lambda x: model(x)
    model, _, _ = open_clip.create_model_and_transforms(meta["model_name"], pretrained=meta["pretrained"])
    model.eval()
    return model, lambda x: model.encode_image(x)


def extract_split_features(
    dataset_name: str,
    backbone_name: str,
    split: str,
    nuisance_views: list[dict] | None = None,
    tag: str = "clean",
    pair_indices: np.ndarray | None = None,
) -> Path:
    if pair_indices is None:
        split_payload = json_load(PROCESSED_ROOT / dataset_name / f"{split}_tuples.json")
        indices = np.asarray(split_payload["indices"], dtype=np.int64)
    else:
        indices = np.asarray(pair_indices, dtype=np.int64)
    return extract_indexed_features(dataset_name, backbone_name, indices, nuisance_views=nuisance_views, tag=f"{split}_{tag}")


def extract_all_clean_features(dataset_name: str, backbone_name: str) -> Path:
    bundle = load_dataset(dataset_name)
    indices = np.arange(len(bundle.images), dtype=np.int64)
    return extract_indexed_features(dataset_name, backbone_name, indices, nuisance_views=None, tag="all_clean")


def extract_indexed_features(dataset_name: str, backbone_name: str, indices: np.ndarray, nuisance_views: list[dict] | None, tag: str) -> Path:
    out_dir = ensure_dir(FEATURE_ROOT / dataset_name / backbone_name)
    mm_path = out_dir / f"{tag}.npy"
    index_path = out_dir / f"{tag}_index.json"
    if mm_path.exists() and index_path.exists():
        return mm_path
    dataset = ImageDataset(dataset_name, indices, backbone_name, nuisance=nuisance_views)
    meta = BACKBONES[backbone_name]
    loader = DataLoader(dataset, batch_size=meta["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    model, encode = build_backbone(backbone_name)
    device = select_device()
    model.to(device)
    amp_dtype = preferred_amp_dtype()
    features = []
    ordered_indices = []
    with torch.no_grad():
        for images, batch_indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                batch_features = encode(images)
            batch_features = batch_features.float().cpu().numpy()
            features.append(batch_features)
            ordered_indices.append(batch_indices.numpy())
    feature_array = np.concatenate(features, axis=0).astype(np.float16)
    index_array = np.concatenate(ordered_indices, axis=0).astype(np.int64)
    np.save(mm_path, feature_array)
    write_json(index_path, {"indices": index_array.tolist(), "shape": list(feature_array.shape)})
    return mm_path


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text())
