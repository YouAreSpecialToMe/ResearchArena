from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import ImageReward as RM
import lpips
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor, pipeline

from exp.shared.common import CLIP_MODEL_ID, DETECTOR_MODEL_ID, DINO_MODEL_ID, IMAGE_REWARD_MODEL_ID, PromptRecord


PERSON_WORDS = {"boy", "girl", "man", "woman"}
OBJECT_ALIASES = {
    "airplane": "airplane",
    "boat": "boat",
    "book": "book",
    "clock": "clock",
    "goldfish": "fish",
    "microwave": "microwave",
    "painting": "picture",
    "person": "person",
    "refrigerator": "refrigerator",
    "ship": "boat",
    "sofa": "couch",
    "television": "tv",
}
IRREGULAR_SINGULARS = {
    "women": "woman",
    "men": "man",
    "people": "person",
    "mice": "mouse",
    "geese": "goose",
    "children": "child",
}


def _normalize_object_name(name: str) -> str:
    lowered = name.lower().strip()
    lowered = IRREGULAR_SINGULARS.get(lowered, lowered)
    if lowered.endswith("s") and lowered not in {"glasses"}:
        lowered = lowered[:-1]
    if lowered in PERSON_WORDS:
        return "person"
    return OBJECT_ALIASES.get(lowered, lowered)


def _iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    xa1, ya1, xa2, ya2 = box_a["xmin"], box_a["ymin"], box_a["xmax"], box_a["ymax"]
    xb1, yb1, xb2, yb2 = box_b["xmin"], box_b["ymin"], box_b["xmax"], box_b["ymax"]
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    return inter / max(area_a + area_b - inter, 1e-6)


class CLIPScorer:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
        self.model.eval()

    def _load_image(self, image_or_path: Path | Image.Image) -> Image.Image:
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")
        return Image.open(image_or_path).convert("RGB")

    def image_text_score(self, image_or_path: Path | Image.Image, text: str) -> float:
        image = self._load_image(image_or_path)
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        image_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((image_emb * text_emb).sum().item())

    def image_image_score(self, image_a: Path, image_b: Path) -> float:
        images = [self._load_image(image_a), self._load_image(image_b)]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return float((image_emb[0] * image_emb[1]).sum().item())

    def best_text_label(self, image_or_path: Path | Image.Image, texts: list[str]) -> str:
        scores = [(text, self.image_text_score(image_or_path, text)) for text in texts]
        return max(scores, key=lambda item: item[1])[0]


class LPIPSScorer:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.metric = lpips.LPIPS(net="alex").to(device)

    def score(self, image_a: Path, image_b: Path) -> float:
        def _load(path: Path) -> torch.Tensor:
            arr = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
            ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return ten * 2 - 1

        with torch.no_grad():
            return float(self.metric(_load(image_a), _load(image_b)).item())


class DINOScorer:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
        self.model = AutoModel.from_pretrained(DINO_MODEL_ID).to(device)
        self.model.eval()
        self._cache: dict[str, torch.Tensor] = {}

    def _load_image(self, image_or_path: Path | Image.Image) -> Image.Image:
        if isinstance(image_or_path, Image.Image):
            return image_or_path.convert("RGB")
        return Image.open(image_or_path).convert("RGB")

    def image_feature(self, image_or_path: Path | Image.Image) -> torch.Tensor:
        cache_key = str(image_or_path) if isinstance(image_or_path, Path) else None
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        image = self._load_image(image_or_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        feature = outputs.last_hidden_state[:, 0]
        feature = feature / feature.norm(dim=-1, keepdim=True)
        if cache_key:
            self._cache[cache_key] = feature
        return feature

    def image_image_score(self, image_a: Path, image_b: Path) -> float:
        feat_a = self.image_feature(image_a)
        feat_b = self.image_feature(image_b)
        return float((feat_a * feat_b).sum().item())


class ImageRewardScorer:
    def __init__(self) -> None:
        self.model = RM.load(IMAGE_REWARD_MODEL_ID)

    def image_text_score(self, image_path: Path, text: str) -> float:
        return float(self.model.score(text, str(image_path)))


class SlotEvaluator:
    def __init__(self, clip: CLIPScorer, device: str = "cuda") -> None:
        self.clip = clip
        self.detector = pipeline(
            task="zero-shot-object-detection",
            model=DETECTOR_MODEL_ID,
            device=0 if device == "cuda" else -1,
        )

    def _detect(self, image: Image.Image, labels: list[str], threshold: float = 0.12) -> dict[str, list[dict[str, Any]]]:
        detections = self.detector(image, candidate_labels=labels, threshold=threshold)
        grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
        for det in detections:
            label = det["label"]
            if label not in grouped:
                continue
            keep = True
            for existing in grouped[label]:
                if _iou(existing["box"], det["box"]) > 0.6:
                    if existing["score"] >= det["score"]:
                        keep = False
                        break
            if keep:
                grouped[label].append(det)
        for label in grouped:
            grouped[label] = sorted(grouped[label], key=lambda item: item["score"], reverse=True)
        return grouped

    def _crop(self, image: Image.Image, box: dict[str, float]) -> Image.Image:
        left = max(0, int(box["xmin"]))
        upper = max(0, int(box["ymin"]))
        right = min(image.size[0], int(box["xmax"]))
        lower = min(image.size[1], int(box["ymax"]))
        return image.crop((left, upper, right, lower))

    def _relation_score(self, relation: str, left_box: dict[str, float], right_box: dict[str, float]) -> float:
        left_center = ((left_box["xmin"] + left_box["xmax"]) / 2.0, (left_box["ymin"] + left_box["ymax"]) / 2.0)
        right_center = ((right_box["xmin"] + right_box["xmax"]) / 2.0, (right_box["ymin"] + right_box["ymax"]) / 2.0)
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        if relation in {"next to", "near", "on side of"}:
            distance = math.sqrt(dx * dx + dy * dy)
            return 1.0 if distance < 180.0 else max(0.0, 180.0 / max(distance, 1.0))
        if relation == "on the left of":
            return 1.0 if dx > 25 and abs(dx) > abs(dy) else 0.0
        if relation == "on the right of":
            return 1.0 if dx < -25 and abs(dx) > abs(dy) else 0.0
        if relation == "on the top of":
            return 1.0 if dy > 25 and abs(dy) > abs(dx) else 0.0
        if relation == "on the bottom of":
            return 1.0 if dy < -25 and abs(dy) > abs(dx) else 0.0
        return 0.0

    def evaluate(self, image_path: Path, record: PromptRecord, prompt_text: str) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        labels = [_normalize_object_name(record.object_1)]
        if record.object_2:
            labels.append(_normalize_object_name(record.object_2))
        labels = list(dict.fromkeys(labels))
        detected = self._detect(image, labels)

        obj1_label = _normalize_object_name(record.object_1)
        obj2_label = _normalize_object_name(record.object_2) if record.object_2 else ""
        obj1_dets = detected.get(obj1_label, [])
        obj2_dets = detected.get(obj2_label, []) if obj2_label else []
        obj1_present = bool(obj1_dets)
        obj2_present = bool(obj2_dets) if record.object_2 else True

        attr1_correct = True
        attr2_correct = True
        if record.attribute_1 and obj1_dets:
            crop = self._crop(image, obj1_dets[0]["box"])
            candidates = [f"a {record.attribute_1} {record.object_1}"]
            if record.attribute_2:
                candidates.append(f"a {record.attribute_2} {record.object_1}")
            attr1_correct = self.clip.best_text_label(crop, candidates) == candidates[0]
        elif record.attribute_1:
            attr1_correct = False
        if record.attribute_2 and obj2_dets:
            crop = self._crop(image, obj2_dets[0]["box"])
            candidates = [f"a {record.attribute_2} {record.object_2}"]
            if record.attribute_1:
                candidates.append(f"a {record.attribute_1} {record.object_2}")
            attr2_correct = self.clip.best_text_label(crop, candidates) == candidates[0]
        elif record.attribute_2:
            attr2_correct = False

        relation_correct = True
        if record.relation:
            relation_correct = False
            if obj1_dets and obj2_dets:
                best = 0.0
                for left in obj1_dets[:3]:
                    for right in obj2_dets[:3]:
                        best = max(best, self._relation_score(record.relation, left["box"], right["box"]))
                relation_correct = best >= 0.99

        count_1_target = record.count_1 or "one"
        count_1_correct = True
        if record.category == "numeracy":
            target = 1 if count_1_target == "one" else int(
                {
                    "two": 2,
                    "three": 3,
                    "four": 4,
                    "five": 5,
                    "six": 6,
                    "seven": 7,
                    "eight": 8,
                }[count_1_target]
            )
            count_1_correct = len(obj1_dets) == target

        overall_success = obj1_present and obj2_present and attr1_correct and attr2_correct and relation_correct and count_1_correct
        if record.category == "attribute_binding":
            category_score = float(np.mean([obj1_present, obj2_present, attr1_correct, attr2_correct]))
        elif record.category == "relations":
            category_score = float(np.mean([obj1_present, obj2_present, relation_correct]))
        else:
            category_score = float(np.mean([obj1_present, count_1_correct]))

        return {
            "prompt_text": prompt_text,
            "object_1_present": bool(obj1_present),
            "object_2_present": bool(obj2_present),
            "attribute_1_correct": bool(attr1_correct),
            "attribute_2_correct": bool(attr2_correct),
            "relation_correct": bool(relation_correct),
            "count_1_correct": bool(count_1_correct),
            "overall_success": bool(overall_success),
            "category_score": category_score,
            "detected_count_object_1": len(obj1_dets),
            "detected_count_object_2": len(obj2_dets),
        }


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}


def bootstrap_ci(values: list[float], num_bootstrap: int = 1000, seed: int = 0) -> dict[str, float]:
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float32)
    samples = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, len(arr), len(arr))
        samples.append(float(np.mean(arr[idx])))
    return {"low": float(np.percentile(samples, 2.5)), "high": float(np.percentile(samples, 97.5))}


def paired_bootstrap_delta(a: list[float], b: list[float], num_bootstrap: int = 1000, seed: int = 0) -> dict[str, float]:
    if not a or len(a) != len(b):
        return {"mean_delta": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    deltas = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, len(a_arr), len(a_arr))
        deltas.append(float(np.mean(a_arr[idx] - b_arr[idx])))
    return {
        "mean_delta": float(np.mean(a_arr - b_arr)),
        "low": float(np.percentile(deltas, 2.5)),
        "high": float(np.percentile(deltas, 97.5)),
    }


def paired_permutation_test(a: list[float], b: list[float], num_permutations: int = 2000, seed: int = 0) -> dict[str, float]:
    if not a or len(a) != len(b):
        return {"observed_delta": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    observed = float(np.mean(diffs))
    permuted = []
    for _ in range(num_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        permuted.append(float(np.mean(diffs * signs)))
    p_value = float(np.mean(np.abs(permuted) >= abs(observed)))
    return {"observed_delta": observed, "p_value": p_value}


def pairwise_lpips(paths: list[Path], scorer: LPIPSScorer) -> float:
    if len(paths) < 2:
        return 0.0
    scores = [scorer.score(a, b) for a, b in itertools.combinations(paths, 2)]
    return float(np.mean(scores))
