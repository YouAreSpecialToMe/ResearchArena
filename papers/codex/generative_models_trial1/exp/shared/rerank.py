from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from exp.shared.attention import upsample_heatmap
from exp.shared.models import detect_boxes, siglip_score_image_text
from exp.shared.utils import ensure_dir, read_json, write_json


def crop_image(image: Image.Image, box: list[float]) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def zscores(values: list[float]) -> list[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-6:
        return [0.0 for _ in values]
    mean = float(arr.mean())
    return [float((v - mean) / std) for v in values]


def relation_score(box_a: list[float], box_b: list[float], relation: str, tau: float = 0.05) -> float:
    ax = 0.5 * (box_a[0] + box_a[2])
    ay = 0.5 * (box_a[1] + box_a[3])
    bx = 0.5 * (box_b[0] + box_b[2])
    by = 0.5 * (box_b[1] + box_b[3])
    if relation == "left_of":
        margin = (bx - ax) / max(1.0, abs(bx - ax) + abs(by - ay))
    elif relation == "right_of":
        margin = (ax - bx) / max(1.0, abs(bx - ax) + abs(by - ay))
    elif relation == "above":
        margin = (by - ay) / max(1.0, abs(bx - ax) + abs(by - ay))
    else:
        margin = (ay - by) / max(1.0, abs(bx - ax) + abs(by - ay))
    return sigmoid(margin / tau)


def count_score(matched_count: int, requested_count: int, extra_boxes: int) -> float:
    return math.exp(-abs(matched_count - requested_count)) * math.exp(-0.5 * extra_boxes)


def aggregate_scores(scores: list[float], mode: str = "geometric") -> float:
    if not scores:
        return 0.0
    eps = 1e-4
    if mode == "mean":
        return float(sum(scores) / len(scores))
    return float(math.exp(sum(math.log(eps + s) for s in scores) / len(scores)))


def box_attention_mass(heatmap_path: Path | None, phrase: str, box: list[float], image_size: tuple[int, int]) -> float:
    if heatmap_path is None or not heatmap_path.exists():
        return 0.0
    data = np.load(heatmap_path, allow_pickle=False)
    if phrase not in data:
        return 0.0
    up = upsample_heatmap(data[phrase], image_size[::-1])
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(image_size[0] - 1, x1))
    x2 = max(x1 + 1, min(image_size[0], x2))
    y1 = max(0, min(image_size[1] - 1, y1))
    y2 = max(y1 + 1, min(image_size[1], y2))
    return float(up[y1:y2, x1:x2].sum())


def noun_phrase(group: dict) -> str:
    parts = list(group.get("attrs", [])) + [group["noun"]]
    return " ".join(parts)


def build_negative_phrases(all_groups: list[dict], target_group: dict) -> list[str]:
    negatives = []
    for group in all_groups:
        if group["group_id"] == target_group["group_id"]:
            continue
        negatives.append(" ".join(group.get("attrs", []) + [target_group["noun"]]))
        if target_group.get("attrs"):
            negatives.append(" ".join(target_group["attrs"] + [group["noun"]]))
    return [n.strip() for n in negatives if n.strip()]


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_box(group: dict, image: Image.Image, item: dict) -> dict:
    box = item["box"]
    crop = crop_image(image, box)
    positive_phrase = noun_phrase(group)
    positive_siglip = siglip_score_image_text(crop, positive_phrase)
    size = ((box[2] - box[0]) * (box[3] - box[1])) / float(image.width * image.height)
    size_prior = -1.0 if size < 0.01 or size > 0.60 else 0.0
    return {
        "box": box,
        "det_conf": float(item["score"]),
        "size_prior": float(size_prior),
        "positive_siglip": float(positive_siglip),
    }


def rank_boxes(features: list[dict], assignment_source: str, group: dict, heatmap_path: Path | None, image_size: tuple[int, int]) -> list[dict]:
    if not features:
        return []
    augmented = []
    for item in features:
        daam_mass = box_attention_mass(heatmap_path, group["noun"], item["box"], image_size) if heatmap_path is not None else 0.0
        augmented.append({**item, "daam_mass": float(daam_mass)})
    det_z = zscores([item["det_conf"] for item in augmented])
    daam_z = zscores([item["daam_mass"] for item in augmented])
    size_z = zscores([item["size_prior"] for item in augmented])
    siglip_z = zscores([item["positive_siglip"] for item in augmented])
    ranked = []
    for idx, item in enumerate(augmented):
        if assignment_source == "detector":
            compat = det_z[idx] + size_z[idx]
        elif assignment_source == "crop_siglip":
            compat = siglip_z[idx] + 0.25 * det_z[idx] + 0.1 * size_z[idx]
        elif assignment_source == "detector_daam":
            compat = det_z[idx] + daam_z[idx] + size_z[idx]
        else:
            raise ValueError(f"Unsupported assignment source: {assignment_source}")
        ranked.append({**item, "compat": float(compat)})
    ranked.sort(key=lambda item: item["compat"], reverse=True)
    return ranked


def attribute_score(group: dict, groups: list[dict], crop: Image.Image, use_counterfactual: bool) -> float:
    positive_score = siglip_score_image_text(crop, noun_phrase(group))
    if use_counterfactual:
        negatives = build_negative_phrases(groups, group)
        negative_score = max((siglip_score_image_text(crop, text) for text in negatives), default=0.0)
        return sigmoid((positive_score - negative_score) / 0.05)
    return sigmoid(positive_score / 0.05)


def is_available(box: list[float], used_boxes: list[list[float]], iou_threshold: float = 0.5) -> bool:
    return all(compute_iou(box, used_box) < iou_threshold for used_box in used_boxes)


def feature_cache_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    idx = parts.index("candidate_cache")
    parts[idx] = "box_features"
    parts[-1] = image_path.stem + ".json"
    return Path(*parts)


def load_or_compute_phrase_boxes(groups: list[dict], image: Image.Image, image_path: Path, heatmap_path: Path | None) -> dict[str, list[dict]]:
    cache_path = feature_cache_path(image_path)
    if cache_path.exists():
        cached = read_json(cache_path)
        group_features = cached.get("group_features", {})
        if all(group["group_id"] in group_features for group in groups):
            return group_features

    group_features = {}
    for group in groups:
        boxes = detect_boxes(image, group["noun"])
        boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)[:4]
        group_features[group["group_id"]] = [score_box(group, image, item) for item in boxes]

    write_json(cache_path, {"image_path": str(image_path), "group_features": group_features})
    return group_features


def assign_slots(
    groups: list[dict],
    image: Image.Image,
    image_path: Path,
    heatmap_path: Path | None,
    assignment_source: str,
    use_counterfactual: bool,
    force_non_null: bool = False,
):
    raw_features = load_or_compute_phrase_boxes(groups, image, image_path, heatmap_path)
    phrase_boxes = {
        group["group_id"]: rank_boxes(raw_features[group["group_id"]], assignment_source, group, heatmap_path, image.size)
        for group in groups
    }

    assignments = {}
    group_counts = {}
    atomic_scores = []
    used_boxes: list[list[float]] = []
    for group in groups:
        boxes = phrase_boxes[group["group_id"]]
        matched = []
        for demand_idx in range(group["count"]):
            choice = None
            for item in boxes:
                if not is_available(item["box"], used_boxes):
                    continue
                choice = item
                break
            remaining = [item for item in boxes if choice is None or item["box"] != choice["box"]]
            runner_up = next((item for item in remaining if is_available(item["box"], used_boxes)), None)
            if choice is None or (not force_non_null and (choice["compat"] < -0.25 or not boxes)):
                assignments[f"{group['group_id']}#{demand_idx}"] = {"box": None, "margin": 0.0, "compat": [], "runner_up_box": None}
            else:
                used_boxes.append(choice["box"])
                margin = choice["compat"] - (runner_up["compat"] if runner_up else -1.0)
                assignments[f"{group['group_id']}#{demand_idx}"] = {
                    "box": choice["box"],
                    "margin": float(margin),
                    "compat": {
                        "det_conf": choice["det_conf"],
                        "daam_mass": choice["daam_mass"],
                        "size_prior": choice["size_prior"],
                        "positive_siglip": choice["positive_siglip"],
                        "compat": choice["compat"],
                        "assignment_source": assignment_source,
                    },
                    "runner_up_box": runner_up["box"] if runner_up else None,
                }
                matched.append(choice)

                if group["attrs"]:
                    crop = crop_image(image, choice["box"])
                    atomic_scores.append(attribute_score(group, groups, crop, use_counterfactual))

        extra_boxes = max(0, len(boxes) - len(matched)) if not force_non_null else 0
        cscore = count_score(len(matched), group["count"], extra_boxes)
        group_counts[group["group_id"]] = {"requested": group["count"], "matched": len(matched), "extra_boxes": extra_boxes, "score": cscore}
        atomic_scores.append(cscore)
    return assignments, group_counts, atomic_scores, phrase_boxes


def score_candidate(
    record: dict,
    image_path: Path,
    heatmap_path: Path | None,
    method: str,
    assignment_source: str,
    use_counterfactual: bool,
    aggregation: str = "geometric",
    force_non_null: bool = False,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    groups = record["parse"]["object_groups"]
    assignments, group_counts, atomic_scores, phrase_boxes = assign_slots(
        groups=groups,
        image=image,
        image_path=image_path,
        heatmap_path=heatmap_path,
        assignment_source=assignment_source,
        use_counterfactual=use_counterfactual,
        force_non_null=force_non_null,
    )
    for rel in record["parse"]["relation_atoms"]:
        left = assignments.get(f"{rel['lhs_group_id']}#0", {})
        right = assignments.get(f"{rel['rhs_group_id']}#0", {})
        if left.get("box") is None or right.get("box") is None:
            score = 0.0
        else:
            score = relation_score(left["box"], right["box"], rel["relation"])
        atomic_scores.append(score)
    final_score = aggregate_scores(atomic_scores, mode=aggregation)
    all_atoms_pass = all(score >= 0.5 for score in atomic_scores) if atomic_scores else False
    return {
        "final_score": final_score,
        "atomic_scores": atomic_scores,
        "group_counts": group_counts,
        "slot_assignments": assignments,
        "phrase_boxes": phrase_boxes,
        "all_atoms_pass": all_atoms_pass,
    }
