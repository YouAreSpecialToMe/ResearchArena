from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open_clip
import torch
import torchvision
from PIL import Image
from skimage.segmentation import felzenszwalb, slic

from .common import (
    DOMAIN_ORDER_A,
    DOMAIN_ORDER_B,
    VOC_CLASSES,
    apply_domain_shift,
    bbox_iou,
    ensure_dir,
    image_to_array,
    label_to_array,
    save_json,
)


@dataclass
class SampleSpec:
    sample_id: str
    base_id: str
    domain: str
    image_path: str
    label_path: str
    split: str


def _digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_voc(root: Path):
    extracted = (root / "VOCdevkit/VOC2012/JPEGImages").exists()
    train = torchvision.datasets.VOCSegmentation(root=str(root), year="2012", image_set="train", download=not extracted)
    val = torchvision.datasets.VOCSegmentation(root=str(root), year="2012", image_set="val", download=False)
    return train, val


def _eligible_indices(dataset, min_fg_pixels: int, max_items: int) -> List[int]:
    out = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        arr = np.array(label)
        fg = ((arr != 0) & (arr != 255)).sum()
        if fg >= min_fg_pixels:
            out.append(idx)
        if len(out) >= max_items:
            break
    return out


def create_splits(root: Path, benchmark_size_per_domain: int = 24, reduced_size_per_domain: int = 12, calibration_size: int = 24) -> Dict:
    data_root = root / "data"
    train_ds, val_ds = load_voc(data_root)
    calibration_idx = _eligible_indices(train_ds, min_fg_pixels=3000, max_items=calibration_size)
    eval_idx = _eligible_indices(val_ds, min_fg_pixels=4000, max_items=benchmark_size_per_domain)
    reduced_idx = eval_idx[:reduced_size_per_domain]

    def build_specs(dataset, indices, split_name, domains):
        samples = []
        for domain in domains:
            for pos, idx in enumerate(indices):
                image_path = dataset.images[idx]
                label_path = dataset.masks[idx]
                base_id = Path(image_path).stem
                sample_id = f"{base_id}_{domain}_{pos:02d}"
                samples.append(
                    SampleSpec(
                        sample_id=sample_id,
                        base_id=base_id,
                        domain=domain,
                        image_path=image_path,
                        label_path=label_path,
                        split=split_name,
                    )
                )
        return samples

    calibration = [
        SampleSpec(
            sample_id=f"{Path(train_ds.images[idx]).stem}_clean_cal_{i:02d}",
            base_id=Path(train_ds.images[idx]).stem,
            domain="clean",
            image_path=train_ds.images[idx],
            label_path=train_ds.masks[idx],
            split="calibration",
        )
        for i, idx in enumerate(calibration_idx)
    ]
    benchmark = build_specs(val_ds, eval_idx, "main", DOMAIN_ORDER_A)
    reduced = build_specs(val_ds, reduced_idx, "reduced", DOMAIN_ORDER_A)

    payload = {
        "proxy_note": "Pascal VOC 2012 proxy benchmark with synthetic domain shifts. Original CAT-Seg/Cityscapes/ACDC plan was infeasible in this workspace.",
        "calibration": [s.__dict__ for s in calibration],
        "main": [s.__dict__ for s in benchmark],
        "reduced": [s.__dict__ for s in reduced],
        "orders": {"A": DOMAIN_ORDER_A, "B": DOMAIN_ORDER_B},
        "proxy_domains": {
            "clean": "Pascal VOC clean",
            "fog": "synthetic fog",
            "gaussian_noise": "synthetic gaussian noise",
            "snow": "synthetic snow",
            "dusk": "synthetic dusk",
            "night": "synthetic night",
        },
        "counts": {
            "calibration": len(calibration),
            "main_total": len(benchmark),
            "reduced_total": len(reduced),
            "per_domain_main": benchmark_size_per_domain,
            "per_domain_reduced": reduced_size_per_domain,
        },
    }
    save_json(root / "exp/01_data_prep/splits.json", payload)
    return payload


def render_sample(spec: Dict, size=(256, 256)) -> Tuple[Image.Image, np.ndarray]:
    image = Image.open(spec["image_path"]).convert("RGB")
    label = Image.open(spec["label_path"])
    shifted = apply_domain_shift(image, spec["domain"], seed=int(_digest(spec["sample_id"]), 16) % (2**31))
    return shifted.resize(size, Image.BILINEAR), label_to_array(label, size=size)


def _mask_to_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _nms(proposals: List[Dict], iou_thresh: float = 0.8, max_keep: int = 16) -> List[Dict]:
    kept: List[Dict] = []
    for prop in sorted(proposals, key=lambda p: p["area"], reverse=True):
        if any(bbox_iou(prop["bbox"], prev["bbox"]) > iou_thresh for prev in kept):
            continue
        kept.append(prop)
        if len(kept) >= max_keep:
            break
    return kept


def _segment_proposals(image_np: np.ndarray) -> List[Dict]:
    seg = felzenszwalb(image_np, scale=160, sigma=0.8, min_size=180)
    proposals = []
    image_area = image_np.shape[0] * image_np.shape[1]
    for region_id in np.unique(seg):
        mask = seg == region_id
        area = mask.sum()
        ratio = area / image_area
        if ratio < 0.005 or ratio > 0.35:
            continue
        bbox = _mask_to_bbox(mask)
        if bbox is None:
            continue
        proposals.append(
            {
                "region_id": int(region_id),
                "area": int(area),
                "area_ratio": float(ratio),
                "bbox": bbox,
                "mask": mask.astype(np.uint8),
            }
        )
    return _nms(proposals)


def _slic_regions(image_np: np.ndarray) -> List[Dict]:
    seg = slic(image_np, n_segments=16, compactness=10.0, start_label=0)
    regions = []
    image_area = image_np.shape[0] * image_np.shape[1]
    for region_id in np.unique(seg):
        mask = seg == region_id
        area = mask.sum()
        bbox = _mask_to_bbox(mask)
        if bbox is None:
            continue
        regions.append(
            {
                "region_id": int(region_id),
                "area": int(area),
                "area_ratio": float(area / image_area),
                "bbox": bbox,
                "mask": mask.astype(np.uint8),
            }
        )
    return regions


def build_caches(root: Path, splits: Dict, device: str = "cuda") -> Dict:
    cache_root = root / "data/proxy_cache"
    ensure_dir(cache_root)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text_tokens = tokenizer(VOC_CLASSES).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    np.save(cache_root / "clip_text_features.npy", text_features.detach().cpu().numpy())

    summaries = {"proposal_files_created": 0, "slic_files_created": 0}
    all_specs = splits["calibration"] + splits["main"] + splits["reduced"]
    unique_specs = {spec["sample_id"]: spec for spec in all_specs}

    for idx, spec in enumerate(unique_specs.values(), start=1):
        image, _ = render_sample(spec)
        image_np = np.array(image)
        proposal_path = cache_root / f"{spec['sample_id']}_proposals.npz"
        slic_path = cache_root / f"{spec['sample_id']}_slic.npz"
        if not proposal_path.exists():
            proposals = _segment_proposals(image_np)
            arrays = {"count": np.array([len(proposals)], dtype=np.int32)}
            for i, prop in enumerate(proposals):
                crop = image.crop((prop["bbox"][0], prop["bbox"][1], prop["bbox"][2] + 1, prop["bbox"][3] + 1))
                crop_tensor = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(crop_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).squeeze(0).detach().cpu().numpy()
                arrays[f"mask_{i}"] = prop["mask"]
                arrays[f"bbox_{i}"] = np.asarray(prop["bbox"], dtype=np.int32)
                arrays[f"logits_{i}"] = logits.astype(np.float32)
                arrays[f"area_ratio_{i}"] = np.asarray([prop["area_ratio"]], dtype=np.float32)
            np.savez_compressed(proposal_path, **arrays)
            summaries["proposal_files_created"] += 1
        if not slic_path.exists():
            regions = _slic_regions(image_np)
            arrays = {"count": np.array([len(regions)], dtype=np.int32)}
            for i, prop in enumerate(regions):
                crop = image.crop((prop["bbox"][0], prop["bbox"][1], prop["bbox"][2] + 1, prop["bbox"][3] + 1))
                crop_tensor = preprocess(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(crop_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = (100.0 * image_features @ text_features.T).squeeze(0).detach().cpu().numpy()
                arrays[f"mask_{i}"] = prop["mask"]
                arrays[f"bbox_{i}"] = np.asarray(prop["bbox"], dtype=np.int32)
                arrays[f"logits_{i}"] = logits.astype(np.float32)
                arrays[f"area_ratio_{i}"] = np.asarray([prop["area_ratio"]], dtype=np.float32)
            np.savez_compressed(slic_path, **arrays)
            summaries["slic_files_created"] += 1
        if idx % 20 == 0 or idx == len(unique_specs):
            print(f"[cache] processed {idx}/{len(unique_specs)} samples", flush=True)
    report = {
        **summaries,
        "unique_samples_expected": len(unique_specs),
        "proposal_files_present": len(list(cache_root.glob("*_proposals.npz"))),
        "slic_files_present": len(list(cache_root.glob("*_slic.npz"))),
        "clip_text_features_present": int((cache_root / "clip_text_features.npy").exists()),
    }
    save_json(root / "exp/01_data_prep/cache_report.json", report)
    return report


def load_cached_regions(cache_root: Path, sample_id: str, region_type: str) -> List[Dict]:
    suffix = "proposals" if region_type == "proposal" else "slic"
    data = np.load(cache_root / f"{sample_id}_{suffix}.npz", allow_pickle=False)
    count = int(data["count"][0])
    regions = []
    for i in range(count):
        regions.append(
            {
                "mask": data[f"mask_{i}"].astype(bool),
                "bbox": data[f"bbox_{i}"].astype(int).tolist(),
                "clip_logits": data[f"logits_{i}"].astype(np.float32),
                "area_ratio": float(data[f"area_ratio_{i}"][0]),
            }
        )
    return regions
