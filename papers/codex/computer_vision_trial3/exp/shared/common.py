from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


VOC_CLASSES: List[str] = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]

DOMAIN_ORDER_A = ["clean", "fog", "gaussian_noise", "snow", "dusk", "night"]
DOMAIN_ORDER_B = ["night", "dusk", "snow", "gaussian_noise", "fog", "clean"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))


def reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def append_log(path: Path, message: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_to_array(image: Image.Image, size=(256, 256)) -> np.ndarray:
    return np.array(image.resize(size, Image.BILINEAR))


def label_to_array(label: Image.Image, size=(256, 256)) -> np.ndarray:
    return np.array(label.resize(size, Image.NEAREST)).astype(np.int64)


def apply_domain_shift(image: Image.Image, domain: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    if domain == "clean":
        return image
    if domain == "fog":
        fog = Image.new("RGB", image.size, (230, 230, 230))
        return Image.blend(image, fog, alpha=0.38)
    if domain == "gaussian_noise":
        arr = np.asarray(image).astype(np.float32)
        noise = np.random.default_rng(seed).normal(0, 22, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if domain == "snow":
        arr = np.asarray(image).astype(np.float32)
        snow = np.random.default_rng(seed + 7).normal(235, 18, size=arr.shape)
        mask = np.random.default_rng(seed + 9).uniform(size=arr.shape[:2]) > 0.88
        arr[mask] = 0.6 * arr[mask] + 0.4 * snow[mask]
        arr = np.clip(arr * 1.05, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=0.5))
    if domain == "dusk":
        image = ImageEnhance.Brightness(image).enhance(0.58)
        image = ImageEnhance.Color(image).enhance(0.85)
        return image.filter(ImageFilter.GaussianBlur(radius=0.6))
    if domain == "night":
        image = ImageEnhance.Brightness(image).enhance(0.28)
        image = ImageEnhance.Contrast(image).enhance(1.25)
        image = ImageEnhance.Color(image).enhance(0.7)
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    raise ValueError(f"Unknown domain {domain}")


def compute_confusion(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    idx = target * num_classes + pred
    conf = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return conf


def miou_from_confusion(conf: np.ndarray, drop_absent: bool = False) -> Dict[str, float]:
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    fg_iou = iou[1:]
    if drop_absent:
        present_fg = denom[1:] > 0
        miou = float(fg_iou[present_fg].mean()) if present_fg.any() else 0.0
    else:
        miou = float(fg_iou.mean())
    return {
        "miou": miou,
        "background_iou": float(iou[0]),
        "per_class_iou": iou.tolist(),
        "present_foreground_classes": int((denom[1:] > 0).sum()),
    }


def weak_augment_tensor(x: torch.Tensor, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    scale = 0.9 + 0.2 * torch.rand(1, device=x.device)
    bias = (torch.rand(3, device=x.device) - 0.5) * 0.08
    out = x * scale + bias.view(1, 3, 1, 1)
    if torch.rand(1, device=x.device).item() < 0.2:
        out = torch.nn.functional.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
    return out.clamp(0.0, 1.0)


def bbox_iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    inter = inter_w * inter_h
    area_a = max(0, xa2 - xa1 + 1) * max(0, ya2 - ya1 + 1)
    area_b = max(0, xb2 - xb1 + 1) * max(0, yb2 - yb1 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def quantile(values: List[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def stderr(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def mean_std_ci95(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "mean": 0.0,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": 0,
        }
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "stderr": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "n": 1,
        }
    std = float(np.std(vals, ddof=1))
    se = float(std / math.sqrt(len(vals)))
    t_critical = {
        2: 12.7062047364,
        3: 4.30265272975,
        4: 3.18244630528,
        5: 2.7764451052,
    }.get(len(vals), 1.95996398454)
    delta = t_critical * se
    return {
        "mean": mean,
        "std": std,
        "stderr": se,
        "ci95_low": float(mean - delta),
        "ci95_high": float(mean + delta),
        "n": len(vals),
    }
