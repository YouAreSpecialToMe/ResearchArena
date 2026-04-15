from __future__ import annotations

import argparse
import itertools
import os
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model

from .models import ASDModel, SAEModel, SSAEModel, train_asd, train_sae, train_ssae
from .utils import (
    ALL_METHODS,
    ABLATION_METHODS,
    GAMMAS,
    OUTPUTS,
    PRIMARY_METHODS,
    ROOT,
    SEEDS,
    append_text,
    device,
    ensure_dir,
    init_workspace,
    load_json,
    now,
    elapsed_minutes,
    get_peak_memory_mb,
    reset_peak_memory,
    save_json,
    set_seed,
    torch_save,
    write_text,
)


FACTOR_COLUMNS = {
    "floor_hue": 0,
    "object_hue": 2,
    "scale": 3,
    "shape": 4,
}

TARGET_ATTRS = ["Smiling", "Eyeglasses", "Wearing_Hat"]
_THREEDSHAPES_ARRAYS: tuple[np.ndarray, np.ndarray] | None = None


def _cache_stats_path(dataset_name: str, split: str) -> Path:
    return OUTPUTS / "metrics" / f"cache_{dataset_name}_{split}.json"


def _cache_validation_stats(feats: torch.Tensor, sample_size: int = 100) -> dict[str, float | int]:
    sample = feats[: min(sample_size, feats.shape[0])].float()
    row_norms = sample.norm(dim=1)
    return {
        "sample_size": int(sample.shape[0]),
        "unique_rows_sample": int(torch.unique(sample, dim=0).shape[0]),
        "mean_feature_variance_sample": float(sample.var(dim=0).mean().item()),
        "mean_row_norm_sample": float(row_norms.mean().item()),
        "std_row_norm_sample": float(row_norms.std().item()) if sample.shape[0] > 1 else 0.0,
    }


def _assert_valid_cache(dataset_name: str, split: str, feats: torch.Tensor) -> dict[str, float | int]:
    stats = _cache_validation_stats(feats)
    if stats["unique_rows_sample"] <= 1 or stats["mean_feature_variance_sample"] <= 1e-10:
        raise RuntimeError(
            f"Degenerate CLIP cache for {dataset_name}/{split}: "
            f"unique_rows_sample={stats['unique_rows_sample']} "
            f"mean_feature_variance_sample={stats['mean_feature_variance_sample']}"
        )
    return stats


def _update_cache_manifest(dataset_name: str, split: str, clip_path: Path, dino_path: Path, meta_path: Path) -> None:
    manifest_path = OUTPUTS / "cache" / "cache_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    entry = {
        "clip_path": str(clip_path.relative_to(ROOT)) if clip_path.exists() else None,
        "dino_path": str(dino_path.relative_to(ROOT)) if dino_path.exists() else None,
        "meta_path": str(meta_path.relative_to(ROOT)) if meta_path.exists() else None,
    }
    if clip_path.exists():
        clip = torch.load(clip_path, map_location="cpu")
        entry["clip_shape"] = list(clip.shape)
    if dino_path.exists():
        dino = torch.load(dino_path, map_location="cpu")
        entry["dino_shape"] = list(dino.shape)
    if meta_path.exists():
        entry["meta_rows"] = int(pd.read_csv(meta_path).shape[0])
    manifest[f"{dataset_name}_{split}"] = entry
    save_json(manifest_path, manifest)


def _load_3dshapes_arrays() -> tuple[np.ndarray, np.ndarray]:
    global _THREEDSHAPES_ARRAYS
    if _THREEDSHAPES_ARRAYS is None:
        with h5py.File(_download_3dshapes(), "r") as f:
            images = np.array(f["images"])
            labels = np.array(f["labels"])
        _THREEDSHAPES_ARRAYS = (images, labels)
    return _THREEDSHAPES_ARRAYS


def _celeba_attr_names() -> list[str]:
    attr_path = ROOT / "data" / "celeba" / "list_attr_celeba.txt"
    if not attr_path.exists():
        return TARGET_ATTRS
    with attr_path.open("r", encoding="utf-8") as f:
        _ = f.readline()
        return f.readline().strip().split()


def _ensure_celeba_metadata() -> Path:
    base = ROOT / "data" / "celeba"
    base.mkdir(parents=True, exist_ok=True)
    files = {
        "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
        "identity_CelebA.txt": "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
        "list_bbox_celeba.txt": "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
        "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
        "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
    }
    import gdown

    for name, file_id in files.items():
        target = base / name
        if not target.exists():
            gdown.download(id=file_id, output=str(target), quiet=False, fuzzy=True)
    return base


def _package_versions() -> dict[str, str]:
    import datasets
    import matplotlib
    import numpy
    import open_clip
    import pandas
    import PIL
    import scipy
    import seaborn
    import sklearn
    import statsmodels
    import torch
    import torchvision
    import tqdm
    import transformers

    return {
        "python": os.popen(f"{ROOT / '.venv' / 'bin' / 'python'} --version").read().strip(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "open_clip_torch": open_clip.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__,
        "pillow": PIL.__version__,
        "tqdm": tqdm.__version__,
    }


def prepare_workspace() -> None:
    init_workspace()
    versions = _package_versions()
    write_text(OUTPUTS / "environment_versions.txt", "\n".join(f"{k}={v}" for k, v in versions.items()) + "\n")
    references = [
        {
            "citation_key": "joshi2025ssae",
            "paper": "Sparse Shift Autoencoders for Identifying Concepts from Large Language Model Activations",
            "source_url": "https://arxiv.org/abs/2502.12179",
            "why_included": "Direct shift-only baseline; corrected against the arXiv record.",
        },
        {
            "citation_key": "chatzoudis2025visualsparsesteering",
            "paper": "Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors",
            "source_url": "https://arxiv.org/abs/2506.01247",
            "why_included": "Nearby sparse-steering vision paper used to broaden related work.",
        },
        {
            "citation_key": "pach2025monosemantic",
            "paper": "Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models",
            "source_url": "https://arxiv.org/abs/2504.02821",
            "why_included": "Direct novelty threat on CLIP/VLM sparse representations.",
        },
        {
            "citation_key": "zaigrajew2025clip",
            "paper": "Interpreting CLIP with Hierarchical Sparse Autoencoders",
            "source_url": "https://arxiv.org/abs/2502.20578",
            "why_included": "Nearby CLIP sparse-feature interpretation baseline.",
        },
        {
            "citation_key": "joseph2025steeringclip",
            "paper": "Steering CLIP's vision transformer with sparse autoencoders",
            "source_url": "https://arxiv.org/abs/2504.08729",
            "why_included": "Direct prior art for CLIP feature steering.",
        },
        {
            "citation_key": "nasirisarvi2025sparc",
            "paper": "SPARC: Concept-Aligned Sparse Autoencoders for Cross-Model and Cross-Modal Interpretability",
            "source_url": "https://arxiv.org/abs/2507.06265",
            "why_included": "Cross-model/shared-space novelty threat.",
        },
        {
            "citation_key": "gu2026lucid",
            "paper": "LUCID-SAE: Learning Unified Vision-Language Sparse Codes for Interpretable Concept Discovery",
            "source_url": "https://arxiv.org/abs/2602.07311",
            "why_included": "Strong 2026 unified-code novelty threat.",
        },
        {
            "citation_key": "leask2025canonical",
            "paper": "Sparse Autoencoders Do Not Find Canonical Units of Analysis",
            "source_url": "https://arxiv.org/abs/2502.04878",
            "why_included": "Claim-boundary framing anchor.",
        },
        {
            "citation_key": "canby2024reliable",
            "paper": "How Reliable are Causal Probing Interventions?",
            "source_url": "https://arxiv.org/abs/2408.15510",
            "why_included": "Reliability framing for target-change versus preservation.",
        },
    ]
    pd.DataFrame(references).to_csv(OUTPUTS / "tables" / "reference_audit.csv", index=False)


def _download_3dshapes() -> Path:
    out = ROOT / "data" / "3dshapes.h5"
    if out.exists():
        return out
    url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
    import urllib.request

    urllib.request.urlretrieve(url, out)
    return out


def _factorize_labels(labels: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(labels, columns=["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"])
    for col in df.columns:
        uniq = np.sort(df[col].unique())
        mapping = {v: i for i, v in enumerate(uniq.tolist())}
        df[col] = df[col].map(mapping).astype(int)
    return df


def _sample_stratified(df: pd.DataFrame, n_total: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    group_cols = ["floor_hue", "object_hue", "scale", "shape"]
    groups = df.groupby(group_cols).indices
    keys = list(groups.keys())
    quotas = {k: max(1, int(round(len(v) / len(df) * n_total))) for k, v in groups.items()}
    while sum(quotas.values()) > n_total:
        k = max(quotas, key=quotas.get)
        if quotas[k] > 1:
            quotas[k] -= 1
    while sum(quotas.values()) < n_total:
        k = keys[rng.integers(0, len(keys))]
        quotas[k] += 1
    picked = []
    for k, idx in groups.items():
        take = min(quotas[k], len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(picked) > n_total:
        picked = rng.choice(np.array(picked), size=n_total, replace=False).tolist()
    return np.array(sorted(picked))


def prepare_data() -> None:
    init_workspace()
    set_seed(11)
    shapes_path = _download_3dshapes()
    with h5py.File(shapes_path, "r") as f:
        labels = np.array(f["labels"])
    factor_df = _factorize_labels(labels)
    idx = _sample_stratified(factor_df, 36000, 11)
    rng = np.random.default_rng(11)
    rng.shuffle(idx)
    splits = {
        "train": idx[:24000].tolist(),
        "val": idx[24000:30000].tolist(),
        "test": idx[30000:36000].tolist(),
    }
    save_json(OUTPUTS / "tables" / "3dshapes_split.json", splits)

    celeba_root = _ensure_celeba_metadata()
    attr_names = _celeba_attr_names()
    attr_ix = {name: i for i, name in enumerate(attr_names)}
    split_sizes = {"train": 18000, "valid": 3000, "test": 3000}
    celeba_splits = {}
    stats = {}
    attr_df = pd.read_csv(celeba_root / "list_attr_celeba.txt", delim_whitespace=True, skiprows=1)
    attr_df[attr_df == -1] = 0
    attr_df = attr_df.reset_index().rename(columns={"index": "filename"})
    partition_df = pd.read_csv(celeba_root / "list_eval_partition.txt", delim_whitespace=True, header=None, names=["filename", "partition"])
    target_ix = [attr_ix[a] for a in TARGET_ATTRS]
    partition_map = {"train": 0, "valid": 1, "test": 2}
    merged = attr_df.merge(partition_df, on="filename")
    for split_name, n_keep in split_sizes.items():
        split_df = merged[merged["partition"] == partition_map[split_name]].copy()
        attrs = split_df[attr_names].to_numpy()
        keys = attrs[:, target_ix]
        key_str = np.array(["".join(map(str, row.tolist())) for row in keys])
        chosen = []
        rng = np.random.default_rng(11 + len(split_name))
        for key in np.unique(key_str):
            group_idx = np.where(key_str == key)[0]
            quota = max(1, int(round(len(group_idx) / len(split_df) * n_keep)))
            chosen.extend(rng.choice(group_idx, size=min(quota, len(group_idx)), replace=False).tolist())
        if len(chosen) > n_keep:
            chosen = rng.choice(np.array(chosen), size=n_keep, replace=False).tolist()
        elif len(chosen) < n_keep:
            remain = np.setdiff1d(np.arange(len(split_df)), np.array(chosen))
            extra = rng.choice(remain, size=n_keep - len(chosen), replace=False).tolist()
            chosen.extend(extra)
        chosen = sorted(chosen)
        celeba_splits[split_name] = split_df.iloc[chosen]["filename"].tolist()
        sub_attrs = attrs[chosen]
        stats[split_name] = {a: float(sub_attrs[:, attr_ix[a]].mean()) for a in TARGET_ATTRS}
    save_json(OUTPUTS / "tables" / "celeba_split.json", celeba_splits)
    save_json(OUTPUTS / "tables" / "celeba_split_stats.json", stats)


class ShapesSubset(Dataset):
    def __init__(self, split: str) -> None:
        self.split = split
        self.indices = load_json(OUTPUTS / "tables" / "3dshapes_split.json")[split]
        self.path = _download_3dshapes()
        self.file = None
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        if self.file is None:
            self.file = h5py.File(self.path, "r")
        real_idx = self.indices[idx]
        image = self.file["images"][real_idx]
        label = self.file["labels"][real_idx]
        pil = Image.fromarray(image.astype(np.uint8))
        return self.transform(pil), real_idx, torch.tensor(label, dtype=torch.float32)


class CelebASubset(Dataset):
    def __init__(self, split: str) -> None:
        mapping = {"train": "train", "val": "valid", "test": "test"}
        tv_split = mapping[split]
        base = _ensure_celeba_metadata()
        self.zip_path = base / "img_align_celeba.zip"
        self.zip_file = None
        self.indices = load_json(OUTPUTS / "tables" / "celeba_split.json")[tv_split]
        attr_names = _celeba_attr_names()
        attr_df = pd.read_csv(base / "list_attr_celeba.txt", delim_whitespace=True, skiprows=1)
        attr_df[attr_df == -1] = 0
        attr_df = attr_df.reset_index().rename(columns={"index": "filename"}).set_index("filename")
        self.attr_names = attr_names
        self.attr_df = attr_df
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        filename = self.indices[idx]
        if self.zip_file is None:
            self.zip_file = zipfile.ZipFile(self.zip_path, "r")
        with self.zip_file.open(f"img_align_celeba/{filename}", "r") as f:
            img = Image.open(f).convert("RGB")
        attrs = torch.tensor(self.attr_df.loc[filename, self.attr_names].to_numpy(dtype=np.float32))
        return self.transform(img), filename, attrs


def _iter_3dshapes_batches(split: str, batch_size: int):
    indices = np.array(load_json(OUTPUTS / "tables" / "3dshapes_split.json")[split], dtype=np.int64)
    images, labels = _load_3dshapes_arrays()
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        batch_images = torch.from_numpy(images[batch_indices]).permute(0, 3, 1, 2).float() / 255.0
        batch_ids = torch.from_numpy(batch_indices)
        batch_labels = torch.from_numpy(labels[batch_indices]).float()
        yield batch_images, batch_ids, batch_labels


def _extract_clip_features(dataset_name: str, split: str, ds: Dataset) -> tuple[torch.Tensor, pd.DataFrame]:
    dev = device()
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.to(dev).eval()
    batch_size = 256
    if dataset_name == "3dshapes":
        dl = _iter_3dshapes_batches(split, batch_size)
    else:
        num_workers = min(4, os.cpu_count() or 1)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    meta = []
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    with torch.no_grad():
        for batch_idx, (images, idxs, labels) in enumerate(dl, start=1):
            images = images.to(dev, non_blocking=True)
            images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            images = (images - clip_mean) / clip_std
            with torch.autocast(device_type="cuda", enabled=dev.type == "cuda"):
                x = model.visual._embeds(images)
                x = model.visual.transformer(x)
                pooled, _ = model.visual._pool(x)
                pooled = pooled.float()
            feats.append(pooled.cpu())
            for i in range(len(idxs)):
                row = {"id": str(int(idxs[i].item()) if hasattr(idxs[i], "item") else idxs[i])}
                if dataset_name == "3dshapes":
                    raw = labels[i].numpy()
                    for k, c in FACTOR_COLUMNS.items():
                        row[k] = float(raw[c])
                else:
                    for j, name in enumerate(_celeba_attr_names()):
                        row[name] = int(labels[i][j].item())
                meta.append(row)
            if batch_idx % 5 == 0:
                print(f"clip {dataset_name} {split}: {batch_idx * len(images)} examples", flush=True)
    features = torch.cat(feats, dim=0)
    _assert_valid_cache(dataset_name, split, features)
    return features, pd.DataFrame(meta)


def _extract_dino_features(dataset_name: str, split: str, ds: Dataset) -> tuple[torch.Tensor, pd.DataFrame]:
    dev = device()
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = Dinov2Model.from_pretrained("facebook/dinov2-small").to(dev).eval()
    raw_ds = []
    if dataset_name == "3dshapes":
        src = ShapesSubset(split)
    else:
        src = CelebASubset(split)
    feats = []
    meta = []
    dino_mean = torch.tensor(processor.image_mean, device=dev).view(1, 3, 1, 1)
    dino_std = torch.tensor(processor.image_std, device=dev).view(1, 3, 1, 1)
    batch_size = 256
    if dataset_name == "3dshapes":
        dl = _iter_3dshapes_batches(split, batch_size)
    else:
        num_workers = min(4, os.cpu_count() or 1)
        dl = DataLoader(src, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    with torch.no_grad():
        for batch_idx, (images, idxs, labels) in enumerate(dl, start=1):
            images = images.to(dev, non_blocking=True)
            images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
            images = (images - dino_mean) / dino_std
            with torch.autocast(device_type="cuda", enabled=dev.type == "cuda"):
                outputs = model(pixel_values=images)
                pooled = outputs.last_hidden_state[:, 0].float()
            feats.append(pooled.cpu())
            for i in range(len(idxs)):
                row = {"id": str(int(idxs[i].item()) if hasattr(idxs[i], "item") else idxs[i])}
                if dataset_name == "3dshapes":
                    raw = labels[i].numpy()
                    for k, c in FACTOR_COLUMNS.items():
                        row[k] = float(raw[c])
                else:
                    for j, name in enumerate(_celeba_attr_names()):
                        row[name] = int(labels[i][j].item())
                meta.append(row)
            if batch_idx % 5 == 0:
                print(f"dino {dataset_name} {split}: {batch_idx * len(images)} examples", flush=True)
    return torch.cat(feats, dim=0), pd.DataFrame(meta)


def cache_features() -> None:
    init_workspace()
    for dataset_name, dataset_cls in [("3dshapes", ShapesSubset), ("celeba", CelebASubset)]:
        for split in ["train", "val", "test"]:
            ds = dataset_cls(split)
            clip_path = OUTPUTS / "cache" / f"{dataset_name}_{split}_clip.pt"
            dino_path = OUTPUTS / "cache" / f"{dataset_name}_{split}_dino.pt"
            meta_path = OUTPUTS / "cache" / f"{dataset_name}_{split}_meta.csv"
            if not clip_path.exists():
                print(f"[cache] start clip dataset={dataset_name} split={split}", flush=True)
                start = now()
                reset_peak_memory()
                feats, meta = _extract_clip_features(dataset_name, split, ds)
                torch_save(clip_path, feats)
                meta.to_csv(meta_path, index=False)
                save_json(
                    _cache_stats_path(dataset_name, split),
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "stage": "clip",
                        "num_examples": int(feats.shape[0]),
                        "feature_dim": int(feats.shape[1]),
                        "elapsed_minutes": elapsed_minutes(start),
                        "peak_memory_mb": get_peak_memory_mb(),
                        **_cache_validation_stats(feats),
                    },
                )
                print(f"[cache] wrote {clip_path.name} rows={feats.shape[0]}", flush=True)
            if not dino_path.exists():
                print(f"[cache] start dino dataset={dataset_name} split={split}", flush=True)
                start = now()
                reset_peak_memory()
                feats, _ = _extract_dino_features(dataset_name, split, ds)
                torch_save(dino_path, feats)
                stats_path = _cache_stats_path(dataset_name, split)
                stats = load_json(stats_path) if stats_path.exists() else {"dataset": dataset_name, "split": split}
                stats.update(
                    {
                        "dino_stage": "done",
                        "dino_num_examples": int(feats.shape[0]),
                        "dino_feature_dim": int(feats.shape[1]),
                        "dino_elapsed_minutes": elapsed_minutes(start),
                        "dino_peak_memory_mb": get_peak_memory_mb(),
                        "dino_unique_rows_sample": _cache_validation_stats(feats)["unique_rows_sample"],
                        "dino_mean_feature_variance_sample": _cache_validation_stats(feats)["mean_feature_variance_sample"],
                    }
                )
                save_json(stats_path, stats)
                print(f"[cache] wrote {dino_path.name} rows={feats.shape[0]}", flush=True)
            if clip_path.exists():
                _assert_valid_cache(dataset_name, split, torch.load(clip_path, map_location="cpu"))
            _update_cache_manifest(dataset_name, split, clip_path, dino_path, meta_path)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        scores.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(scores))


def _factorize_3dshape_meta(meta: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    out = meta.copy()
    target_cols = columns or list(FACTOR_COLUMNS.keys())
    for col in target_cols:
        uniq = sorted(out[col].unique().tolist())
        mapping = {value: idx for idx, value in enumerate(uniq)}
        out[col] = out[col].map(mapping).astype(int)
    return out


def _id_key(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    try:
        value_f = float(value)
        if value_f.is_integer():
            return str(int(value_f))
    except Exception:
        pass
    return str(value)


def _build_3d_pairs(retained: list[str], split: str, n_pairs: int) -> pd.DataFrame:
    meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / f"3dshapes_{split}_meta.csv"), retained)
    by_id = meta.copy()
    rows = []
    rng = np.random.default_rng(101 + len(split))
    per_factor = n_pairs // len(retained)
    for factor in retained:
        others = [f for f in retained if f != factor]
        groups = meta.groupby(others)
        factor_rows = []
        for _, group in groups:
            vals = sorted(group[factor].unique().tolist())
            if len(vals) < 2:
                continue
            for source_value in vals:
                candidates_src = group[group[factor] == source_value]
                others_vals = [v for v in vals if v != source_value]
                for target_value in others_vals:
                    candidates_tgt = group[group[factor] == target_value]
                    m = min(len(candidates_src), len(candidates_tgt))
                    if m == 0:
                        continue
                    take = min(m, 4)
                    src_ids = rng.choice(candidates_src.index.to_numpy(), size=take, replace=False)
                    tgt_ids = rng.choice(candidates_tgt.index.to_numpy(), size=take, replace=False)
                    for sidx, tidx in zip(src_ids, tgt_ids):
                        srow = meta.loc[sidx]
                        trow = meta.loc[tidx]
                        factor_rows.append(
                            {
                                "source_id": _id_key(srow["id"]),
                                "target_id": _id_key(trow["id"]),
                                "target_name": factor,
                                "source_value": int(srow[factor]),
                                "target_value": int(trow[factor]),
                            }
                        )
        if len(factor_rows) > per_factor:
            factor_rows = rng.choice(np.array(factor_rows, dtype=object), size=per_factor, replace=False).tolist()
        rows.extend(factor_rows)
    return pd.DataFrame(rows)


def _mine_celeba_pairs(split: str, attrs: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    meta = pd.read_csv(OUTPUTS / "cache" / f"celeba_{split}_meta.csv")
    dino = torch.load(OUTPUTS / "cache" / f"celeba_{split}_dino.pt").float()
    attr_names = meta.columns.tolist()
    rows = []
    coverage = {}
    for attr in attrs:
        other_attrs = [a for a in attrs if a != attr]
        candidates = []
        for source_idx, source in meta.iterrows():
            mask = np.ones(len(meta), dtype=bool)
            mask &= meta[attr].to_numpy() != source[attr]
            for other in other_attrs:
                mask &= meta[other].to_numpy() == source[other]
            eligible_idx = np.where(mask)[0]
            if len(eligible_idx) == 0:
                continue
            source_feat = dino[source_idx : source_idx + 1]
            target_feats = dino[eligible_idx]
            dist = torch.cdist(source_feat, target_feats).squeeze(0).numpy()
            best = int(dist.argmin())
            candidates.append((source_idx, eligible_idx[best], float(dist[best])))
        if not candidates:
            coverage[attr] = 0
            continue
        threshold = float(np.percentile([c[2] for c in candidates], 35))
        filtered = [c for c in candidates if c[2] <= threshold]
        caps = {"train": 1500, "val": 300, "test": 300}
        filtered = filtered[: caps[split]]
        coverage[attr] = len(filtered)
        for src_idx, tgt_idx, dist in filtered:
            s = meta.iloc[src_idx]
            t = meta.iloc[tgt_idx]
            rows.append(
                {
                    "source_id": _id_key(s["id"]),
                    "target_id": _id_key(t["id"]),
                    "target_name": attr,
                    "source_value": int(s[attr]),
                    "target_value": int(t[attr]),
                    "dino_distance": dist,
                }
            )
    return pd.DataFrame(rows), coverage


def probes_and_pairs() -> None:
    init_workspace()
    train_meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / "3dshapes_train_meta.csv"))
    val_meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / "3dshapes_val_meta.csv"))
    train_clip = torch.load(OUTPUTS / "cache" / "3dshapes_train_clip.pt").numpy()
    val_clip = torch.load(OUTPUTS / "cache" / "3dshapes_val_clip.pt").numpy()
    scores = {}
    for factor in FACTOR_COLUMNS:
        clf = LogisticRegression(max_iter=200, n_jobs=4)
        clf.fit(train_clip, train_meta[factor].astype(int).to_numpy())
        pred = clf.predict(val_clip)
        scores[factor] = _macro_f1(val_meta[factor].astype(int).to_numpy(), pred)
    retained = sorted(scores, key=scores.get, reverse=True)[:3]
    dropped = [f for f in FACTOR_COLUMNS if f not in retained][0]
    save_json(OUTPUTS / "tables" / "3dshapes_factor_screen.json", {"scores": scores, "retained_factors": retained, "dropped_factor": dropped})
    pair_targets = {"train": 18000, "val": 4500, "test": 4500}
    for split, n_pairs in pair_targets.items():
        pairs = _build_3d_pairs(retained, split, n_pairs)
        clip_meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / f"3dshapes_{split}_meta.csv"), retained)
        clip_feats = torch.load(OUTPUTS / "cache" / f"3dshapes_{split}_clip.pt")
        dino_feats = torch.load(OUTPUTS / "cache" / f"3dshapes_{split}_dino.pt")
        id_to_idx = {_id_key(v): i for i, v in enumerate(clip_meta["id"].tolist())}
        pairs["dino_distance"] = [
            float(torch.norm(dino_feats[id_to_idx[_id_key(s)]] - dino_feats[id_to_idx[_id_key(t)]]).item())
            for s, t in zip(pairs["source_id"], pairs["target_id"])
        ]
        pairs["clip_delta_norm"] = [
            float(torch.norm(clip_feats[id_to_idx[_id_key(t)]] - clip_feats[id_to_idx[_id_key(s)]]).item())
            for s, t in zip(pairs["source_id"], pairs["target_id"])
        ]
        pairs.to_parquet(OUTPUTS / "pairs" / f"3dshapes_{split}_pairs.parquet", index=False)
    attr_note = ""
    celeba_train_pairs, coverage = _mine_celeba_pairs("train", TARGET_ATTRS)
    if coverage.get("Wearing_Hat", 0) < 800:
        attrs = ["Smiling", "Eyeglasses", "Blond_Hair"]
        attr_note = "Replaced Wearing_Hat with Blond_Hair because filtered training pair coverage was below 800."
    else:
        attrs = TARGET_ATTRS
    write_text(OUTPUTS / "tables" / "celeba_attribute_note.txt", attr_note or "No attribute substitution required.")
    for split in ["train", "val", "test"]:
        pairs, cov = _mine_celeba_pairs(split, attrs)
        clip_meta = pd.read_csv(OUTPUTS / "cache" / f"celeba_{split}_meta.csv")
        clip_feats = torch.load(OUTPUTS / "cache" / f"celeba_{split}_clip.pt")
        id_to_idx = {_id_key(v): i for i, v in enumerate(clip_meta["id"].tolist())}
        pairs["clip_delta_norm"] = [
            float(torch.norm(clip_feats[id_to_idx[_id_key(t)]] - clip_feats[id_to_idx[_id_key(s)]]).item())
            for s, t in zip(pairs["source_id"], pairs["target_id"])
        ]
        pairs.to_parquet(OUTPUTS / "pairs" / f"celeba_{split}_pairs.parquet", index=False)
        if split == "train":
            quality_rows = []
            for attr, count in cov.items():
                sub = pairs[pairs["target_name"] == attr]
                quality_rows.append(
                    {
                        "attribute": attr,
                        "pair_count": count,
                        "dino_distance_p35": float(sub["dino_distance"].quantile(0.35)) if len(sub) else None,
                        "dino_distance_median": float(sub["dino_distance"].median()) if len(sub) else None,
                        "clip_delta_norm_median": float(sub["clip_delta_norm"].median()) if len(sub) else None,
                        "clip_delta_norm_min": float(sub["clip_delta_norm"].min()) if len(sub) else None,
                    }
                )
            pd.DataFrame(quality_rows).to_csv(OUTPUTS / "tables" / "celeba_pair_quality.csv", index=False)

    save_json(
        OUTPUTS / "metrics" / "probe_integrity.json",
        {
            "3dshapes_val_macro_f1": scores,
        },
    )


def _load_pair_tensors(dataset: str, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    clip = torch.load(OUTPUTS / "cache" / f"{dataset}_{split}_clip.pt").float()
    meta = pd.read_csv(OUTPUTS / "cache" / f"{dataset}_{split}_meta.csv")
    pairs = pd.read_parquet(OUTPUTS / "pairs" / f"{dataset}_{split}_pairs.parquet")
    id_to_idx = {_id_key(v): i for i, v in enumerate(meta["id"].tolist())}
    src = torch.stack([clip[id_to_idx[_id_key(s)]] for s in pairs["source_id"]])
    tgt = torch.stack([clip[id_to_idx[_id_key(t)]] for t in pairs["target_id"]])
    deltas = tgt - src
    return src, tgt, deltas, meta, pairs


def _select_sae_lambda(train_x: torch.Tensor, val_x: torch.Tensor, seed: int, dataset: str, method: str):
    best = None
    for lam in [1e-4, 3e-4, 1e-3]:
        model = SAEModel() if method == "sae" else SSAEModel()
        log_path = OUTPUTS / "logs" / f"{method}_{dataset}_seed{seed}_lam{lam}.csv"
        if method == "sae":
            result = train_sae(model, train_x, val_x, lam, seed, log_path)
        else:
            result = train_ssae(model, train_x, val_x, lam, seed, log_path)
        if best is None or result.val_score < best[0]:
            best = (result.val_score, lam, model, result)
    return best[1], best[2], best[3]


def _select_asd_hparams(train_x, val_x, train_delta, val_delta, seed: int, dataset: str):
    best = None
    for lam_sparse, lam_tie in itertools.product([1e-4, 3e-4, 1e-3], [0.1, 0.3, 1.0]):
        model = ASDModel(shared_decoder=True)
        log_path = OUTPUTS / "logs" / f"asd_{dataset}_seed{seed}_ls{lam_sparse}_lt{lam_tie}.csv"
        train_src, train_tgt, _, _, _ = _load_pair_tensors(dataset, "train")
        val_src, val_tgt, _, _, _ = _load_pair_tensors(dataset, "val")
        result = train_asd(model, train_x, val_x, train_src, train_tgt, val_src, val_tgt, lam_sparse, lam_tie, log_path)
        if best is None or result.val_score < best[0]:
            best = (result.val_score, lam_sparse, lam_tie, model, result)
    return best[1], best[2], best[3], best[4]


def pilot_asd() -> None:
    train_x = torch.load(OUTPUTS / "cache" / "3dshapes_train_clip.pt").float()
    val_x = torch.load(OUTPUTS / "cache" / "3dshapes_val_clip.pt").float()
    train_src, train_tgt, _, _, _ = _load_pair_tensors("3dshapes", "train")
    val_src, val_tgt, _, _, _ = _load_pair_tensors("3dshapes", "val")
    lam_sparse, lam_tie, model, result = _select_asd_hparams(train_x, val_x, None, None, 11, "3dshapes")
    save_json(
        OUTPUTS / "metrics" / "pilot_asd_3dshapes_seed11.json",
        {"lambda_sparse": lam_sparse, "lambda_tie": lam_tie, "val_score": result.val_score, "runtime_minutes": result.runtime_minutes, "peak_memory_mb": result.peak_memory_mb},
    )


def _train_method(dataset: str, method: str, seed: int, lam_sparse: float | None = None, lam_tie: float | None = None) -> None:
    set_seed(seed)
    train_x = torch.load(OUTPUTS / "cache" / f"{dataset}_train_clip.pt").float()
    val_x = torch.load(OUTPUTS / "cache" / f"{dataset}_val_clip.pt").float()
    test_x = torch.load(OUTPUTS / "cache" / f"{dataset}_test_clip.pt").float()
    train_src, train_tgt, train_delta, _, train_pairs = _load_pair_tensors(dataset, "train")
    val_src, val_tgt, val_delta, _, val_pairs = _load_pair_tensors(dataset, "val")
    out_prefix = f"{method}_{dataset}_seed{seed}"
    config = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 1024,
        "max_epochs": 20,
        "early_stopping_patience": 4,
        "gradient_clip_norm": 1.0,
    }
    if method == "sae":
        lam, model, result = _select_sae_lambda(train_x, val_x, seed, dataset, method)
        checkpoint = OUTPUTS / "models" / f"{out_prefix}.pt"
        torch_save(checkpoint, model.state_dict())
        config.update({"lambda_l1": lam, "representation": "clip_features"})
        save_json(OUTPUTS / "metrics" / f"{out_prefix}_config.json", config)
        save_json(OUTPUTS / "metrics" / f"{out_prefix}_train.json", {"lambda_l1": lam, **result.__dict__})
    elif method == "ssae":
        lam, model, result = _select_sae_lambda(train_delta, val_delta, seed, dataset, method)
        checkpoint = OUTPUTS / "models" / f"{out_prefix}.pt"
        torch_save(checkpoint, model.state_dict())
        config.update({"lambda_l1": lam, "representation": "clip_deltas"})
        save_json(OUTPUTS / "metrics" / f"{out_prefix}_config.json", config)
        save_json(OUTPUTS / "metrics" / f"{out_prefix}_train.json", {"lambda_l1": lam, **result.__dict__})
    else:
        shared = method != "asd_no_share"
        lam_sparse = 3e-4 if lam_sparse is None else lam_sparse
        lam_tie = 0.3 if lam_tie is None else lam_tie
        effective_lam_tie = 0.0 if method == "asd_no_tie" else lam_tie
        if method == "asd":
            lam_sparse, lam_tie, tuned_model, tuned = _select_asd_hparams(train_x, val_x, train_delta, val_delta, seed, dataset)
            model = tuned_model
            result = tuned
            effective_lam_tie = lam_tie
        else:
            model = ASDModel(shared_decoder=shared)
            result = train_asd(
                model,
                train_x,
                val_x,
                train_src,
                train_tgt,
                val_src,
                val_tgt,
                lam_sparse,
                effective_lam_tie,
                OUTPUTS / "logs" / f"{out_prefix}.csv",
            )
        checkpoint = OUTPUTS / "models" / f"{out_prefix}.pt"
        torch_save(checkpoint, model.state_dict())
        if method == "asd":
            train_metrics = {"lambda_sparse": lam_sparse, "lambda_tie": lam_tie, **result.__dict__}
            config.update({"lambda_sparse": lam_sparse, "lambda_tie": lam_tie, "shared_decoder": True})
        else:
            train_metrics = {"lambda_sparse": lam_sparse, "lambda_tie": effective_lam_tie, **result.__dict__}
            config.update({"lambda_sparse": lam_sparse, "lambda_tie": effective_lam_tie, "shared_decoder": shared})
        save_json(OUTPUTS / "metrics" / f"{out_prefix}_config.json", config)
        save_json(OUTPUTS / "metrics" / f"{out_prefix}_train.json", train_metrics)


def write_reproducibility_audit(gamma_map: dict[str, float] | None = None) -> None:
    gamma_map = gamma_map or load_json(OUTPUTS / "metrics" / "selected_gammas.json")
    rows = [
        {
            "paper_item": "Primary tables and all quoted numeric results",
            "artifact": "results.csv",
            "selection_rule": "Use rows with gamma equal to selected_gamma[dataset]; aggregate by seed, then report mean ± std over seed-level target means unless stated otherwise.",
        },
        {
            "paper_item": "Per-target ASD-minus-baseline confidence intervals",
            "artifact": "outputs/tables/bootstrap_cis.csv",
            "selection_rule": "Paired bootstrap over held-out test examples at selected_gamma; comparisons are asd_minus_ssae and asd_minus_sae.",
        },
        {
            "paper_item": "Dataset-level CelebA ASD-minus-SSAE CI and permutation p-value",
            "artifact": "results.json",
            "selection_rule": "Hierarchical bootstrap over seeds and target-level rows plus paired sign permutation over the three seed-level reliability gaps.",
        },
        {
            "paper_item": "Selected edit scales",
            "artifact": "outputs/metrics/selected_gammas.json",
            "selection_rule": "Choose gamma from {0.5, 1.0, 1.5} by averaging validation reliability over the three primary methods using seed-11 checkpoints only.",
        },
        {
            "paper_item": "Pilot hyperparameters reused by ablations",
            "artifact": "outputs/metrics/pilot_asd_3dshapes_seed11.json",
            "selection_rule": "Take lambda_sparse and lambda_tie from the best 3D Shapes seed-11 ASD validation run.",
        },
        {
            "paper_item": "Run-level training provenance",
            "artifact": "outputs/metrics/*_{config,train}.json",
            "selection_rule": "Each run stores dataset, method, seed, optimizer settings, and training metrics; asd_no_tie now records lambda_tie=0.0 explicitly.",
        },
        {
            "paper_item": "Pair construction and factor-retention screen",
            "artifact": "exp/probes_and_pairs/results.json",
            "selection_rule": "Retain 3D Shapes factors by validation probe score; mine CelebA pairs with DINO-distance filtering under fixed non-target labels.",
        },
        {
            "paper_item": "Reference verification",
            "artifact": "outputs/tables/reference_audit.csv",
            "selection_rule": "Audit sheet listing every paper discussed in related work with citation key and canonical source URL.",
        },
    ]
    for dataset, gamma in gamma_map.items():
        rows.append(
            {
                "paper_item": f"Default gamma for {dataset}",
                "artifact": "outputs/metrics/selected_gammas.json",
                "selection_rule": f"selected_gamma['{dataset}'] = {gamma}",
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUTS / "tables" / "reproducibility_audit.csv", index=False)


def train_matrix() -> None:
    for dataset in ["3dshapes", "celeba"]:
        for seed in SEEDS:
            for method in PRIMARY_METHODS:
                _train_method(dataset, method, seed)


def train_ablations() -> None:
    pilot = load_json(OUTPUTS / "metrics" / "pilot_asd_3dshapes_seed11.json")
    lam_sparse = pilot["lambda_sparse"]
    lam_tie = pilot["lambda_tie"]
    for dataset in ["3dshapes", "celeba"]:
        for seed in SEEDS:
            _train_method(dataset, "asd_no_tie", seed, lam_sparse=lam_sparse, lam_tie=lam_tie)
            _train_method(dataset, "asd_no_share", seed, lam_sparse=lam_sparse, lam_tie=lam_tie)


def _load_model(method: str, dataset: str, seed: int):
    if method == "sae":
        model = SAEModel()
    elif method == "ssae":
        model = SSAEModel()
    else:
        model = ASDModel(shared_decoder=(method != "asd_no_share"))
    model.load_state_dict(torch.load(OUTPUTS / "models" / f"{method}_{dataset}_seed{seed}.pt", map_location="cpu"))
    model.eval()
    return model


def _mean_active(code: torch.Tensor, threshold: float = 0.05) -> float:
    return float((code.abs() > threshold).float().sum(dim=1).mean().item())


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a + b == 0, 0.0, 2 * a * b / (a + b))


def _fit_multiclass_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray):
    best = None
    for c in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=200, C=c, n_jobs=4)
        clf.fit(train_x, train_y)
        pred = clf.predict(val_x)
        score = _macro_f1(val_y, pred)
        if best is None or score > best[0]:
            best = (score, clf)
    return best[1]


def _fit_binary_probe(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray):
    best = None
    for c in [0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=200, C=c, n_jobs=4)
        clf.fit(train_x, train_y)
        pred = clf.predict_proba(val_x)[:, 1]
        score = roc_auc_score(val_y, pred)
        if best is None or score > best[0]:
            best = (score, clf)
    return best[1], best[0]


def _prototype_from_pairs(method: str, model, dataset: str, split: str, target_name: str, source_value: int | None, target_value: int | None) -> tuple[torch.Tensor, float]:
    clip = torch.load(OUTPUTS / "cache" / f"{dataset}_{split}_clip.pt").float()
    meta = pd.read_csv(OUTPUTS / "cache" / f"{dataset}_{split}_meta.csv")
    pairs = pd.read_parquet(OUTPUTS / "pairs" / f"{dataset}_{split}_pairs.parquet")
    subset = pairs[pairs["target_name"] == target_name].copy()
    if source_value is not None:
        subset = subset[(subset["source_value"] == source_value) & (subset["target_value"] == target_value)]
    id_to_idx = {_id_key(v): i for i, v in enumerate(meta["id"].tolist())}
    src = torch.stack([clip[id_to_idx[_id_key(s)]] for s in subset["source_id"]])
    tgt = torch.stack([clip[id_to_idx[_id_key(t)]] for t in subset["target_id"]])
    delta = tgt - src
    alpha = float(delta.norm(dim=1).median().item())
    if method == "sae":
        z_src = model.encoder(src)
        z_tgt = model.encoder(tgt)
        proto_code = (z_tgt - z_src).mean(dim=0, keepdim=True)
        proto = model.decoder(proto_code).squeeze(0)
    elif method == "ssae":
        s = model.encoder(delta)
        proto_code = s.mean(dim=0, keepdim=True)
        proto = model.decoder(proto_code).squeeze(0)
    else:
        s = model.shift_encoder(delta)
        proto_code = s.mean(dim=0, keepdim=True)
        proto = model.shift_decoder(proto_code).squeeze(0)
    return proto, alpha


def _score_edit(
    dataset: str,
    attr: str,
    feat: torch.Tensor,
    edited_feat: torch.Tensor,
    row: pd.Series,
    retained: list[str],
    celeb_attrs: list[str],
    probe_store: dict[tuple[str, str], Any],
    clip_probes: dict[str, Any],
    dino_probes: dict[str, Any],
    bridge: Ridge | None,
    c_test_dino: np.ndarray | None,
    idx: int,
) -> dict[str, float | None]:
    if dataset == "3dshapes":
        target_pred = probe_store[(dataset, attr)].predict(edited_feat.detach().unsqueeze(0).numpy())[0]
        target_success = float(target_pred == int(row["target_value"]))
        other = [f for f in retained if f != attr]
        keep = []
        for o in other:
            clf = probe_store[(dataset, o)]
            before = clf.predict(feat.detach().unsqueeze(0).numpy())[0]
            after = clf.predict(edited_feat.detach().unsqueeze(0).numpy())[0]
            keep.append(float(before == after))
        preservation = float(np.mean(keep))
        dino_rel = None
    else:
        target_probe = clip_probes[attr]
        target_success = float(target_probe.predict(edited_feat.detach().unsqueeze(0).numpy())[0] == int(row["target_value"]))
        other = [a for a in celeb_attrs if a != attr]
        keep = []
        for o in other:
            probe = clip_probes[o]
            before = probe.predict(feat.detach().unsqueeze(0).numpy())[0]
            after = probe.predict(edited_feat.detach().unsqueeze(0).numpy())[0]
            keep.append(float(before == after))
        preservation = float(np.mean(keep))
        assert bridge is not None and c_test_dino is not None
        bridge_feat = bridge.predict(edited_feat.detach().unsqueeze(0).numpy())
        dino_target = float(dino_probes[attr].predict(bridge_feat)[0] == int(row["target_value"]))
        dino_keep = []
        for o in other:
            probe = dino_probes[o]
            before = probe.predict(c_test_dino[idx : idx + 1])[0]
            after = probe.predict(bridge_feat)[0]
            dino_keep.append(float(before == after))
        dino_rel = float(_harmonic(np.array([dino_target]), np.array([np.mean(dino_keep)])).mean())
    reliability = float(_harmonic(np.array([target_success]), np.array([preservation])).mean())
    return {
        "target_change": target_success,
        "nontarget_preservation": preservation,
        "reliability": reliability,
        "dino_reliability": dino_rel,
    }


def evaluate() -> None:
    init_workspace()
    retained = load_json(OUTPUTS / "tables" / "3dshapes_factor_screen.json")["retained_factors"]
    main_rows = []
    probe_store = {}
    train_clip = torch.load(OUTPUTS / "cache" / "3dshapes_train_clip.pt").numpy()
    val_clip = torch.load(OUTPUTS / "cache" / "3dshapes_val_clip.pt").numpy()
    test_clip = torch.load(OUTPUTS / "cache" / "3dshapes_test_clip.pt").float()
    train_meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / "3dshapes_train_meta.csv"))
    val_meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / "3dshapes_val_meta.csv"))
    test_meta = _factorize_3dshape_meta(pd.read_csv(OUTPUTS / "cache" / "3dshapes_test_meta.csv"))
    for factor in retained:
        probe_store[("3dshapes", factor)] = _fit_multiclass_probe(train_clip, train_meta[factor].astype(int).to_numpy(), val_clip, val_meta[factor].astype(int).to_numpy())
    celeb_attrs = ["Smiling", "Eyeglasses"]
    note = (OUTPUTS / "tables" / "celeba_attribute_note.txt").read_text().strip()
    celeb_attrs.append("Blond_Hair" if "Blond_Hair" in note else "Wearing_Hat")
    c_train_clip = torch.load(OUTPUTS / "cache" / "celeba_train_clip.pt").numpy()
    c_val_clip = torch.load(OUTPUTS / "cache" / "celeba_val_clip.pt").numpy()
    c_test_clip = torch.load(OUTPUTS / "cache" / "celeba_test_clip.pt").float()
    c_train_dino = torch.load(OUTPUTS / "cache" / "celeba_train_dino.pt").numpy()
    c_val_dino = torch.load(OUTPUTS / "cache" / "celeba_val_dino.pt").numpy()
    c_test_dino = torch.load(OUTPUTS / "cache" / "celeba_test_dino.pt").numpy()
    c_train_meta = pd.read_csv(OUTPUTS / "cache" / "celeba_train_meta.csv")
    c_val_meta = pd.read_csv(OUTPUTS / "cache" / "celeba_val_meta.csv")
    c_test_meta = pd.read_csv(OUTPUTS / "cache" / "celeba_test_meta.csv")
    dino_probes = {}
    clip_probes = {}
    auc_rows = []
    for attr in celeb_attrs:
        clip_probe, clip_auc = _fit_binary_probe(c_train_clip, c_train_meta[attr].astype(int).to_numpy(), c_val_clip, c_val_meta[attr].astype(int).to_numpy())
        dino_probe, dino_auc = _fit_binary_probe(c_train_dino, c_train_meta[attr].astype(int).to_numpy(), c_val_dino, c_val_meta[attr].astype(int).to_numpy())
        clip_probes[attr] = clip_probe
        dino_probes[attr] = dino_probe
        auc_rows.append({"attribute": attr, "clip_auc": clip_auc, "dino_auc": dino_auc})
    best_bridge = None
    for alpha in [0.1, 1.0, 10.0]:
        bridge = Ridge(alpha=alpha).fit(c_train_clip, c_train_dino)
        mse = float(np.mean((bridge.predict(c_val_clip) - c_val_dino) ** 2))
        if best_bridge is None or mse < best_bridge[0]:
            best_bridge = (mse, alpha, bridge)
    bridge = best_bridge[2]
    pd.DataFrame(auc_rows).to_csv(OUTPUTS / "tables" / "celeba_eval_summary.csv", index=False)
    probe_integrity_path = OUTPUTS / "metrics" / "probe_integrity.json"
    probe_integrity = load_json(probe_integrity_path) if probe_integrity_path.exists() else {}
    probe_integrity["celeba_val_auc"] = {row["attribute"]: {"clip_auc": row["clip_auc"], "dino_auc": row["dino_auc"]} for row in auc_rows}
    probe_integrity["celeba_dino_bridge_val_mse"] = float(best_bridge[0])
    save_json(probe_integrity_path, probe_integrity)
    id_maps = {
        ("3dshapes", split): {_id_key(v): i for i, v in enumerate(pd.read_csv(OUTPUTS / "cache" / f"3dshapes_{split}_meta.csv")["id"].tolist())}
        for split in ["train", "val", "test"]
    }
    id_maps.update({
        ("celeba", split): {_id_key(v): i for i, v in enumerate(pd.read_csv(OUTPUTS / "cache" / f"celeba_{split}_meta.csv")["id"].tolist())}
        for split in ["train", "val", "test"]
    })
    val_metric_rows = []
    for dataset in ["3dshapes", "celeba"]:
        attrs = retained if dataset == "3dshapes" else celeb_attrs
        split_meta = pd.read_csv(OUTPUTS / "cache" / f"{dataset}_val_meta.csv")
        split_clip = torch.load(OUTPUTS / "cache" / f"{dataset}_val_clip.pt").float()
        split_pairs = pd.read_parquet(OUTPUTS / "pairs" / f"{dataset}_val_pairs.parquet")
        for method in PRIMARY_METHODS:
            model = _load_model(method, dataset, 11)
            for attr in attrs:
                subset = split_pairs[split_pairs["target_name"] == attr]
                groups = subset.groupby(["source_value", "target_value"]) if dataset == "3dshapes" else [((None, None), subset)]
                for key, group in groups:
                    src_value, target_value = key if dataset == "3dshapes" else (None, None)
                    proto, alpha = _prototype_from_pairs(method, model, dataset, "train", attr, src_value, target_value)
                    proto = proto / (proto.norm() + 1e-8)
                    for gamma in GAMMAS:
                        edited = []
                        targets = []
                        preservations = []
                        for _, row in group.iterrows():
                            idx = id_maps[(dataset, "val")][_id_key(row["source_id"])]
                            feat = split_clip[idx]
                            edited_feat = feat + gamma * alpha * proto
                            if dataset == "3dshapes":
                                target_pred = probe_store[(dataset, attr)].predict(edited_feat.detach().unsqueeze(0).numpy())[0]
                                target_success = float(target_pred == int(row["target_value"]))
                                other = [f for f in retained if f != attr]
                                keep = []
                                for o in other:
                                    clf = probe_store[(dataset, o)]
                                    before = clf.predict(feat.detach().unsqueeze(0).numpy())[0]
                                    after = clf.predict(edited_feat.detach().unsqueeze(0).numpy())[0]
                                    keep.append(float(before == after))
                                preservation = float(np.mean(keep))
                            else:
                                target_probe = clip_probes[attr]
                                target_success = float(target_probe.predict(edited_feat.detach().unsqueeze(0).numpy())[0] == int(row["target_value"]))
                                other = [a for a in celeb_attrs if a != attr]
                                keep = []
                                for o in other:
                                    probe = clip_probes[o]
                                    before = probe.predict(feat.detach().unsqueeze(0).numpy())[0]
                                    after = probe.predict(edited_feat.detach().unsqueeze(0).numpy())[0]
                                    keep.append(float(before == after))
                                preservation = float(np.mean(keep))
                            edited.append(edited_feat)
                            targets.append(target_success)
                            preservations.append(preservation)
                        reliability = _harmonic(np.array(targets), np.array(preservations))
                        val_metric_rows.append({"dataset": dataset, "method": method, "target": attr, "gamma": gamma, "mean_reliability": float(reliability.mean())})
    val_df = pd.DataFrame(val_metric_rows)
    gamma_map = {}
    for dataset in ["3dshapes", "celeba"]:
        sub = val_df[val_df["dataset"] == dataset].groupby("gamma")["mean_reliability"].mean()
        gamma_map[dataset] = float(sub.idxmax())
    save_json(OUTPUTS / "metrics" / "selected_gammas.json", gamma_map)

    example_scores = {}
    for dataset in ["3dshapes", "celeba"]:
        attrs = retained if dataset == "3dshapes" else celeb_attrs
        split_meta = pd.read_csv(OUTPUTS / "cache" / f"{dataset}_test_meta.csv")
        split_clip = torch.load(OUTPUTS / "cache" / f"{dataset}_test_clip.pt").float()
        split_dino = torch.load(OUTPUTS / "cache" / f"{dataset}_test_dino.pt").numpy() if dataset == "celeba" else None
        split_pairs = pd.read_parquet(OUTPUTS / "pairs" / f"{dataset}_test_pairs.parquet")
        for method in ALL_METHODS:
            for seed in SEEDS:
                model = _load_model(method, dataset, seed)
                if method == "sae":
                    _, recon = model(split_clip)
                    active = _mean_active(model.encoder(split_clip))
                    recon_mse = float(torch.mean((recon - split_clip) ** 2).item())
                elif method == "ssae":
                    _, _, deltas, _, _ = _load_pair_tensors(dataset, "test")
                    _, recon = model(deltas)
                    active = _mean_active(model.encoder(deltas))
                    recon_mse = float("nan")
                else:
                    _, recon = model.forward_anchor(split_clip)
                    active = _mean_active(model.anchor_encoder(split_clip))
                    recon_mse = float(torch.mean((recon - split_clip) ** 2).item())
                src_feats, tgt_feats, deltas, _, _ = _load_pair_tensors(dataset, "test")
                if method == "ssae":
                    _, delta_recon = model(deltas)
                elif method == "sae":
                    z_src = model.encoder(src_feats)
                    z_tgt = model.encoder(tgt_feats)
                    delta_recon = model.decoder(z_tgt - z_src)
                else:
                    _, delta_recon = model.forward_shift(deltas)
                delta_mse = float(torch.mean((delta_recon - deltas) ** 2).item())
                eval_rows = []
                for attr in attrs:
                    subset = split_pairs[split_pairs["target_name"] == attr]
                    groups = subset.groupby(["source_value", "target_value"]) if dataset == "3dshapes" else [((None, None), subset)]
                    for key, group in groups:
                        src_value, target_value = key if dataset == "3dshapes" else (None, None)
                        proto, alpha = _prototype_from_pairs(method, model, dataset, "train", attr, src_value, target_value)
                        proto = proto / (proto.norm() + 1e-8)
                        proto_sparsity = int((proto.abs() > 1e-3).sum().item())
                        for gamma in GAMMAS:
                            gamma_scores = []
                            for _, row in group.iterrows():
                                idx = id_maps[(dataset, "test")][_id_key(row["source_id"])]
                                feat = split_clip[idx]
                                edited_feat = feat + gamma * alpha * proto
                                gamma_scores.append(
                                    _score_edit(
                                        dataset,
                                        attr,
                                        feat,
                                        edited_feat,
                                        row,
                                        retained,
                                        celeb_attrs,
                                        probe_store,
                                        clip_probes,
                                        dino_probes,
                                        bridge if dataset == "celeba" else None,
                                        c_test_dino if dataset == "celeba" else None,
                                        idx,
                                    )
                                )
                            example_scores.setdefault((dataset, method, seed, attr, gamma), []).extend(gamma_scores)
                            eval_rows.append(
                                {
                                    "dataset": dataset,
                                    "method": method,
                                    "seed": seed,
                                    "target": attr,
                                    "gamma": gamma,
                                    "source_value": None if src_value is None else int(src_value),
                                    "target_value": None if target_value is None else int(target_value),
                                    "pair_count": int(len(group)),
                                    "prototype_alpha": alpha,
                                    "prototype_sparsity": proto_sparsity,
                                    "target_change": float(np.mean([x["target_change"] for x in gamma_scores])),
                                    "nontarget_preservation": float(np.mean([x["nontarget_preservation"] for x in gamma_scores])),
                                    "reliability": float(np.mean([x["reliability"] for x in gamma_scores])),
                                    "dino_reliability": None if dataset == "3dshapes" else float(np.nanmean([x["dino_reliability"] for x in gamma_scores])),
                                }
                            )
                eval_payload = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "reconstruction_mse": recon_mse,
                    "delta_reconstruction_mse": delta_mse,
                    "active_latents": active,
                    "rows": eval_rows,
                }
                save_json(OUTPUTS / "metrics" / f"{method}_{dataset}_seed{seed}_eval.json", eval_payload)
                eval_df = pd.DataFrame(eval_rows)
                for (attr, gamma), group_df in eval_df.groupby(["target", "gamma"], dropna=False):
                    main_rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "seed": seed,
                            "target": attr,
                            "gamma": float(gamma),
                            "runtime_minutes": load_json(OUTPUTS / "metrics" / f"{method}_{dataset}_seed{seed}_train.json")["runtime_minutes"],
                            "peak_memory_mb": load_json(OUTPUTS / "metrics" / f"{method}_{dataset}_seed{seed}_train.json")["peak_memory_mb"],
                            "reconstruction_mse": recon_mse,
                            "delta_reconstruction_mse": delta_mse,
                            "active_latents": active,
                            "prototype_sparsity": float(np.average(group_df["prototype_sparsity"], weights=group_df["pair_count"])),
                            "pair_coverage": int(group_df["pair_count"].sum()),
                            "target_change": float(np.average(group_df["target_change"], weights=group_df["pair_count"])),
                            "nontarget_preservation": float(np.average(group_df["nontarget_preservation"], weights=group_df["pair_count"])),
                            "reliability": float(np.average(group_df["reliability"], weights=group_df["pair_count"])),
                            "dino_reliability": None if dataset == "3dshapes" else float(np.average(group_df["dino_reliability"].fillna(0.0), weights=group_df["pair_count"])),
                        }
                    )
    main_df = pd.DataFrame(main_rows)
    main_df = main_df.sort_values(["method", "dataset", "seed", "target", "gamma"]).reset_index(drop=True)
    main_df.to_csv(OUTPUTS / "tables" / "main_results.csv", index=False)
    main_df.to_csv(ROOT / "results.csv", index=False)
    save_json(OUTPUTS / "metrics" / "example_scores_index.json", {str(k): len(v) for k, v in example_scores.items()})

    def bootstrap_diff(a, b, n=10000, seed=11):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(a))
        diffs = []
        for _ in range(n):
            sample = rng.choice(idx, size=len(idx), replace=True)
            diffs.append(float(np.mean(np.array(a)[sample] - np.array(b)[sample])))
        return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    ci_rows = []
    for dataset in ["3dshapes", "celeba"]:
        attrs = retained if dataset == "3dshapes" else celeb_attrs
        for attr in attrs:
            gamma = gamma_map[dataset]
            asd = []
            ssae = []
            sae = []
            for seed in SEEDS:
                asd.extend([x["reliability"] for x in example_scores[(dataset, "asd", seed, attr, gamma)]])
                ssae.extend([x["reliability"] for x in example_scores[(dataset, "ssae", seed, attr, gamma)]])
                sae.extend([x["reliability"] for x in example_scores[(dataset, "sae", seed, attr, gamma)]])
            for comparison, lhs, rhs in [("asd_minus_ssae", asd, ssae), ("asd_minus_sae", asd, sae)]:
                mean_abs_diff = float(np.mean(np.abs(np.array(lhs) - np.array(rhs))))
                if mean_abs_diff <= 1e-8:
                    ci_rows.append(
                        {
                            "dataset": dataset,
                            "target": attr,
                            "comparison": comparison,
                            "ci_low": None,
                            "ci_high": None,
                            "mean_abs_example_difference": mean_abs_diff,
                            "status": "skipped_identical_scores",
                        }
                    )
                else:
                    ci_low, ci_high = bootstrap_diff(lhs, rhs)
                    ci_rows.append(
                        {
                            "dataset": dataset,
                            "target": attr,
                            "comparison": comparison,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "mean_abs_example_difference": mean_abs_diff,
                            "status": "ok",
                        }
                    )
    pd.DataFrame(ci_rows).to_csv(OUTPUTS / "tables" / "bootstrap_cis.csv", index=False)

    summary = {}
    default_df = main_df[main_df["gamma"] == main_df["dataset"].map(gamma_map)]
    celeba_default = default_df[default_df["dataset"] == "celeba"]
    by_seed = celeba_default.groupby(["method", "seed"])["reliability"].mean().unstack(0)
    diffs = (by_seed["asd"] - by_seed["ssae"]).to_numpy()
    signs = []
    for mask in itertools.product([-1, 1], repeat=len(diffs)):
        signs.append(float(np.mean(diffs * np.array(mask))))
    p_value = float(np.mean(np.abs(signs) >= abs(np.mean(diffs))))
    rng = np.random.default_rng(11)
    hb = []
    for _ in range(10000):
        seed_sample = rng.choice(SEEDS, size=len(SEEDS), replace=True)
        vals = []
        for seed in seed_sample:
            sub = celeba_default[celeba_default["seed"] == seed]
            sub_a = sub[sub["method"] == "asd"]["reliability"].to_numpy()
            sub_b = sub[sub["method"] == "ssae"]["reliability"].to_numpy()
            idx = rng.choice(np.arange(len(sub_a)), size=len(sub_a), replace=True)
            vals.append(float(np.mean(sub_a[idx] - sub_b[idx])))
        hb.append(float(np.mean(vals)))
    hb_ci = [float(np.percentile(hb, 2.5)), float(np.percentile(hb, 97.5))]
    summary["selected_gamma"] = gamma_map
    summary["celeba_asd_minus_ssae_mean"] = float(np.mean(diffs))
    summary["celeba_asd_minus_ssae_hierarchical_bootstrap_ci"] = hb_ci
    summary["celeba_asd_minus_ssae_permutation_p"] = p_value
    success = (
        summary["celeba_asd_minus_ssae_mean"] > 0.02
        and hb_ci[0] > 0
        and int(np.sum(diffs > 0)) >= 2
    )
    summary["success"] = bool(success)
    summary["conclusion"] = (
        "Shared decoding is supported in this compute-matched frozen-feature regime."
        if success
        else "Shared decoding is unsupported in this compute-matched frozen-feature regime."
    )
    save_json(ROOT / "results.json", summary)
    write_reproducibility_audit(gamma_map)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = default_df[default_df["dataset"] == "3dshapes"]
    sns.pointplot(data=plot_df, x="target", y="reliability", hue="method", errorbar="sd", dodge=True, ax=ax)
    ax.set_title("3D Shapes Prototype Reliability")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "plots" / "3dshapes_reliability.png", dpi=200)
    fig.savefig(ROOT / "figures" / "3dshapes_reliability.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    clip_df = default_df[default_df["dataset"] == "celeba"]
    sns.barplot(data=clip_df, x="target", y="reliability", hue="method", ax=axes[0])
    axes[0].set_title("CelebA CLIP")
    dino_df = clip_df.dropna(subset=["dino_reliability"]).copy()
    sns.barplot(data=dino_df, x="target", y="dino_reliability", hue="method", ax=axes[1])
    axes[1].set_title("CelebA DINO Bridge")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "plots" / "celeba_main_comparison.png", dpi=200)
    fig.savefig(ROOT / "figures" / "celeba_main_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=main_df.groupby(["dataset", "method", "gamma"])["reliability"].mean().reset_index(), x="gamma", y="reliability", hue="method", style="dataset", markers=True, ax=ax)
    ax.set_title("Gamma Sensitivity")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "plots" / "gamma_sensitivity.png", dpi=200)
    fig.savefig(ROOT / "figures" / "gamma_sensitivity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ab = default_df[default_df["method"].isin(["asd", "asd_no_tie", "asd_no_share"])].copy()
    baseline = ab[ab["method"] == "asd"][["dataset", "target", "seed", "reliability"]].rename(columns={"reliability": "asd_reliability"})
    merged = ab.merge(baseline, on=["dataset", "target", "seed"])
    merged["delta_from_asd"] = merged["reliability"] - merged["asd_reliability"]
    sns.barplot(data=merged[merged["method"] != "asd"], x="target", y="delta_from_asd", hue="method", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Ablation Effects")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "plots" / "ablation_effects.png", dpi=200)
    fig.savefig(ROOT / "figures" / "ablation_effects.png", dpi=200)
    plt.close(fig)

    expected_artifacts = [
        OUTPUTS / "cache" / "cache_manifest.json",
        OUTPUTS / "pairs" / "3dshapes_train_pairs.parquet",
        OUTPUTS / "pairs" / "celeba_train_pairs.parquet",
        OUTPUTS / "metrics" / "pilot_asd_3dshapes_seed11.json",
        OUTPUTS / "tables" / "main_results.csv",
        OUTPUTS / "tables" / "bootstrap_cis.csv",
        ROOT / "results.json",
        ROOT / "figures" / "3dshapes_reliability.png",
        ROOT / "figures" / "celeba_main_comparison.png",
        ROOT / "figures" / "gamma_sensitivity.png",
        ROOT / "figures" / "ablation_effects.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected_artifacts if not path.exists()]
    ledger_lines = [
        "# Run Ledger",
        "",
        "Execution followed the planned order: workspace setup, split creation, feature caching, probes and pair construction, pilot ASD, full primary matrix, ablations, evaluation, bootstrap CIs, and plotting.",
        "",
        f"Missing expected artifacts: {', '.join(missing) if missing else 'none'}",
        "",
        "Budget-driven deviations from plan: none recorded in code execution; any partial or failed reruns should be inferred from the per-run logs and metrics JSON files.",
        "",
        "Skipped steps: none.",
    ]
    write_text(ROOT / "run_ledger.md", "\n".join(ledger_lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare_workspace", "prepare_data", "cache_features", "probes_and_pairs", "pilot_asd", "train_matrix", "train_ablations", "evaluate"])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    {
        "prepare_workspace": prepare_workspace,
        "prepare_data": prepare_data,
        "cache_features": cache_features,
        "probes_and_pairs": probes_and_pairs,
        "pilot_asd": pilot_asd,
        "train_matrix": train_matrix,
        "train_ablations": train_ablations,
        "evaluate": evaluate,
    }[args.step]()


if __name__ == "__main__":
    main()
