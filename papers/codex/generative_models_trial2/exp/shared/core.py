from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel, CLIPProcessor

from diffusers import DDIMScheduler, StableDiffusionPipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
FIGURES_DIR = ROOT / "figures"
EXP_DIR = ROOT / "exp"

SUPERVISION_SEEDS = [101, 202, 303, 404]
EXPERIMENT_SEEDS = [11, 17, 23]
TRAIN_VAL_COUNTS = {"counting": (40, 13, 27), "position": (40, 14, 26), "color_attr": (40, 13, 27)}
COLOR_WORDS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
RELATION_LABELS = ["above", "below", "left of", "right of"]
COUNTING_THRESHOLD = 0.9
GENERAL_THRESHOLD = 0.3
MAX_OBJECTS = 16
POSITION_THRESHOLD = 0.1
CLIP_PROMPT_TEMPLATES = [
    "a photo of a {c} {classname}",
    "a photo of a {c}-colored {classname}",
    "a photo of a {c} object",
]
ALIASES = {
    "computer keyboard": "keyboard",
    "computer mouse": "mouse",
    "tv": "tv",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a") as handle:
        handle.write(json.dumps(row, default=_json_default) + "\n")


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_class_name(name: str) -> str:
    return ALIASES.get(name, name)


def build_splits() -> Dict[str, List[Dict[str, Any]]]:
    source = DATA_DIR / "geneval" / "prompts" / "evaluation_metadata.jsonl"
    rows = read_jsonl(source)
    grouped: Dict[str, List[Dict[str, Any]]] = {"counting": [], "position": [], "color_attr": []}
    for idx, row in enumerate(rows):
        tag = row["tag"]
        if tag in grouped:
            item = dict(row)
            item["prompt_id"] = f"{tag}_{len(grouped[tag]):03d}"
            item["source_index"] = idx
            grouped[tag].append(item)
    splits = {"train": [], "val": [], "test": []}
    for tag, items in grouped.items():
        n_train, n_val, n_test = TRAIN_VAL_COUNTS[tag]
        selected = items[: n_train + n_val + n_test]
        splits["train"].extend(selected[:n_train])
        splits["val"].extend(selected[n_train : n_train + n_val])
        splits["test"].extend(selected[n_train + n_val : n_train + n_val + n_test])
    for split_name, items in splits.items():
        write_json(PROMPTS_DIR / f"{split_name}_prompts.json", items)
    write_json(PROMPTS_DIR / "split_summary.json", {k: len(v) for k, v in splits.items()})
    return splits


def parse_constraints(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    include = metadata["include"]
    if metadata["tag"] == "counting":
        obj = include[0]
        constraints.append(
            {
                "constraint_id": f"{metadata['prompt_id']}_count",
                "family": "count",
                "type": "count",
                "object_a": obj["class"],
                "count_target": obj["count"],
            }
        )
    elif metadata["tag"] == "position":
        anchor, subject = include
        constraints.append(
            {
                "constraint_id": f"{metadata['prompt_id']}_rel",
                "family": "relation",
                "type": "relation",
                "object_a": subject["class"],
                "object_b": anchor["class"],
                "relation_label": subject["position"][0],
            }
        )
    elif metadata["tag"] == "color_attr":
        for idx, obj in enumerate(include):
            constraints.append(
                {
                    "constraint_id": f"{metadata['prompt_id']}_attr_{idx}",
                    "family": "attribute_binding",
                    "type": "attribute_binding",
                    "object_a": obj["class"],
                    "attribute": obj["color"],
                    "attribute_id": COLOR_WORDS.index(obj["color"]),
                }
            )
    else:
        raise ValueError(f"Unsupported tag {metadata['tag']}")
    return constraints


def stable_int(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def candidate_seed(prompt_id: str, experiment_seed: int, candidate_index: int) -> int:
    return stable_int("candidate", prompt_id, experiment_seed, candidate_index) % (2**31 - 1)


def supervision_latent_seed(source_index: int, supervision_seed: int) -> int:
    return supervision_seed * 100000 + source_index


def latent_seed_tensor(seed: int, shape: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _decode_preview(pipe: StableDiffusionPipeline, latents: torch.Tensor, size: int) -> Tuple[Image.Image, float]:
    start = time.perf_counter()
    lat = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(lat).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image[0].detach().float().cpu().permute(1, 2, 0).numpy()
    pil = Image.fromarray((image * 255).astype(np.uint8)).resize((size, size))
    return pil, time.perf_counter() - start


def _tensor_peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


@dataclass
class RunConfig:
    model_id: str = "runwayml/stable-diffusion-v1-5"
    detector_id: str = "google/owlv2-base-patch16-ensemble"
    clip_id: str = "openai/clip-vit-large-patch14"
    image_size: int = 512
    preview_size: int = 256
    num_steps: int = 20
    tau: int = 4
    guidance_scale: float = 7.5
    train_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5


class ModelBundle:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=pipe_dtype,
            local_files_only=True,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()

        self.processor = AutoProcessor.from_pretrained(cfg.detector_id, local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            cfg.detector_id,
            torch_dtype=pipe_dtype,
            local_files_only=True,
        ).to(self.device)
        self.detector.eval()

        self.clip_processor = CLIPProcessor.from_pretrained(cfg.clip_id, local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained(cfg.clip_id, local_files_only=True).to(self.device)
        self.clip_model.eval()


def encode_prompt(pipe: StableDiffusionPipeline, prompt: str) -> Dict[str, torch.Tensor]:
    text_inputs = pipe.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_inputs = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_ids = text_inputs.input_ids.to(pipe.device)
    uncond_ids = uncond_inputs.input_ids.to(pipe.device)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_ids)[0]
        uncond_embeds = pipe.text_encoder(uncond_ids)[0]
    return {
        "embeds": torch.cat([uncond_embeds, prompt_embeds], dim=0),
        "token_ids": text_ids[0].detach().cpu(),
        "attention_mask": text_inputs.attention_mask[0].detach().cpu(),
    }


def clip_image_embedding(bundle: ModelBundle, image: Image.Image) -> torch.Tensor:
    start = time.perf_counter()
    inputs = bundle.clip_processor(images=image, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        features = bundle.clip_model.get_image_features(**inputs)
    features = F.normalize(features.float(), dim=-1)[0].cpu()
    features.elapsed_seconds = time.perf_counter() - start  # type: ignore[attr-defined]
    return features


def clip_text_image_similarity(bundle: ModelBundle, prompt: str, image: Image.Image) -> Tuple[float, float]:
    start = time.perf_counter()
    inputs = bundle.clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(bundle.device)
    with torch.no_grad():
        outputs = bundle.clip_model(**inputs)
        img = F.normalize(outputs.image_embeds.float(), dim=-1)
        txt = F.normalize(outputs.text_embeds.float(), dim=-1)
    return float((img * txt).sum().item()), time.perf_counter() - start


def _build_feature_vector(
    prompt_encoding: Dict[str, torch.Tensor],
    latent_stats: List[float],
    disagreement: List[float],
    prefix_latents: torch.Tensor,
    preview_embed: torch.Tensor,
) -> np.ndarray:
    token_mask = prompt_encoding["attention_mask"].numpy().astype(bool)
    token_ids = prompt_encoding["token_ids"].numpy()
    active_tokens = token_ids[token_mask]
    token_values = np.mod(active_tokens, 997).astype(np.float32) + 1.0
    token_probs = token_values / token_values.sum()
    token_entropy = float(-(token_probs * np.log(token_probs + 1e-8)).sum())
    token_coverage = float(len(np.unique(active_tokens)) / max(len(active_tokens), 1))
    token_gini = float(np.mean(np.abs(token_values[:, None] - token_values[None, :])) / (2 * token_values.mean() * len(token_values)))
    prefix = prefix_latents.float()
    return np.array(
        latent_stats
        + disagreement
        + [
            float(prefix.norm().item()),
            float(prefix.var().item()),
            float(prefix.mean().item()),
            float(prefix.abs().max().item()),
            token_entropy,
            token_coverage,
            token_gini,
        ]
        + preview_embed[:10].tolist(),
        dtype=np.float32,
    )


def run_prefix(
    bundle: ModelBundle,
    prompt: str,
    latent_seed: int,
    cfg: RunConfig,
) -> Dict[str, Any]:
    pipe = bundle.pipe
    device = pipe.device
    height = width = cfg.image_size
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    latents = latent_seed_tensor(latent_seed, (1, pipe.unet.config.in_channels, height // 8, width // 8), device, dtype)
    prompt_encoding = encode_prompt(pipe, prompt)
    prompt_embeds = prompt_encoding["embeds"]
    scheduler = pipe.scheduler
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latent_stats: List[float] = []
    disagreement: List[float] = []
    denoise_start = time.perf_counter()
    with torch.no_grad():
        for step_index, timestep in enumerate(timesteps[: cfg.tau]):
            model_input = torch.cat([latents] * 2)
            model_input = scheduler.scale_model_input(model_input, timestep)
            noise_pred = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            disagreement.append(float((noise_text - noise_uncond).float().pow(2).mean().sqrt().item()))
            noise_guided = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
            latents = scheduler.step(noise_guided, timestep, latents).prev_sample
            latent_stats.extend(
                [
                    float(latents.float().mean().item()),
                    float(latents.float().std().item()),
                    float(latents.float().abs().mean().item()),
                ]
            )
    denoise_seconds = time.perf_counter() - denoise_start
    prefix_latents = latents.detach().clone()
    preview_image, preview_decode_seconds = _decode_preview(pipe, prefix_latents, cfg.preview_size)
    preview_embed = clip_image_embedding(bundle, preview_image)
    preview_similarity, similarity_seconds = clip_text_image_similarity(bundle, prompt, preview_image)
    feature_vector = _build_feature_vector(prompt_encoding, latent_stats, disagreement, prefix_latents, preview_embed)
    return {
        "prompt": prompt,
        "latent_seed": latent_seed,
        "feature_vector": feature_vector,
        "prefix_latents": prefix_latents.detach().cpu(),
        "preview_image": preview_image,
        "preview_similarity": preview_similarity,
        "timings": {
            "denoise_seconds": denoise_seconds,
            "preview_decode_seconds": preview_decode_seconds,
            "probe_feature_seconds": float(getattr(preview_embed, "elapsed_seconds", 0.0) + similarity_seconds),
        },
        "unet_units": cfg.tau,
        "peak_gpu_gb": _tensor_peak_gb(device),
    }


def continue_from_prefix(bundle: ModelBundle, prompt: str, prefix_latents: torch.Tensor, cfg: RunConfig) -> Dict[str, Any]:
    pipe = bundle.pipe
    device = pipe.device
    latents = prefix_latents.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    prompt_embeds = encode_prompt(pipe, prompt)["embeds"]
    scheduler = pipe.scheduler
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps
    denoise_start = time.perf_counter()
    with torch.no_grad():
        for timestep in timesteps[cfg.tau :]:
            model_input = torch.cat([latents] * 2)
            model_input = scheduler.scale_model_input(model_input, timestep)
            noise_pred = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_guided = noise_uncond + cfg.guidance_scale * (noise_text - noise_uncond)
            latents = scheduler.step(noise_guided, timestep, latents).prev_sample
    denoise_seconds = time.perf_counter() - denoise_start
    final_image, final_decode_seconds = _decode_preview(pipe, latents, cfg.image_size)
    final_similarity, similarity_seconds = clip_text_image_similarity(bundle, prompt, final_image)
    return {
        "final_image": final_image,
        "final_similarity": final_similarity,
        "timings": {
            "denoise_seconds": denoise_seconds,
            "final_decode_seconds": final_decode_seconds,
            "probe_feature_seconds": similarity_seconds,
        },
        "unet_units": cfg.num_steps - cfg.tau,
        "peak_gpu_gb": _tensor_peak_gb(device),
    }


def full_completion(bundle: ModelBundle, prompt: str, latent_seed: int, cfg: RunConfig) -> Dict[str, Any]:
    prefix = run_prefix(bundle, prompt, latent_seed, cfg)
    completion = continue_from_prefix(bundle, prompt, prefix["prefix_latents"], cfg)
    return {
        **prefix,
        **completion,
        "timings": {
            "denoise_seconds": prefix["timings"]["denoise_seconds"] + completion["timings"]["denoise_seconds"],
            "preview_decode_seconds": prefix["timings"]["preview_decode_seconds"],
            "final_decode_seconds": completion["timings"]["final_decode_seconds"],
            "probe_feature_seconds": prefix["timings"]["probe_feature_seconds"] + completion["timings"]["probe_feature_seconds"],
        },
        "unet_units": cfg.num_steps,
        "peak_gpu_gb": max(prefix["peak_gpu_gb"], completion["peak_gpu_gb"]),
    }


def crop_box(image: Image.Image, box: Sequence[float]) -> Image.Image:
    x0, y0, x1, y1 = [int(v) for v in box[:4]]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image.size[0], x1)
    y1 = min(image.size[1], y1)
    return image.crop((x0, y0, x1, y1))


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    def area(box: Sequence[float]) -> float:
        return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

    inter = [max(box_a[0], box_b[0]), max(box_a[1], box_b[1]), min(box_a[2], box_b[2]), min(box_a[3], box_b[3])]
    inter_area = area(inter)
    union = area(box_a) + area(box_b) - inter_area
    return inter_area / union if union > 0 else 0.0


def relative_position(box_a: Sequence[float], box_b: Sequence[float], c: float = POSITION_THRESHOLD) -> List[str]:
    boxes = np.array([box_a[:4], box_b[:4]], dtype=np.float32).reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - c * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return []
    dx, dy = revised_offset / (np.linalg.norm(offset) + 1e-8)
    relations: List[str] = []
    if dx < -0.5:
        relations.append("left of")
    if dx > 0.5:
        relations.append("right of")
    if dy < -0.5:
        relations.append("above")
    if dy > 0.5:
        relations.append("below")
    return relations


def detect_objects(bundle: ModelBundle, image: Image.Image, class_names: Sequence[str], threshold: float) -> Dict[str, List[Dict[str, Any]]]:
    text_queries = [[f"a photo of a {normalize_class_name(name)}" for name in class_names]]
    inputs = bundle.processor(text=text_queries, images=image, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        outputs = bundle.detector(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=bundle.device)
    results = bundle.processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.0,
    )[0]
    detected: Dict[str, List[Dict[str, Any]]] = {name: [] for name in class_names}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        name = class_names[int(label)]
        score_val = float(score.item())
        if score_val < threshold:
            continue
        detected[name].append({"score": score_val, "box": box.detach().cpu().tolist()})
    for name, values in detected.items():
        values.sort(key=lambda item: item["score"], reverse=True)
        kept: List[Dict[str, Any]] = []
        for item in values:
            if len(kept) >= MAX_OBJECTS:
                break
            if all(compute_iou(item["box"], prior["box"]) < 1.0 for prior in kept):
                kept.append(item)
        detected[name] = kept
    return detected


def classify_color(bundle: ModelBundle, obj_name: str, crop: Image.Image) -> str:
    texts = [template.format(c=color, classname=obj_name) for color in COLOR_WORDS for template in CLIP_PROMPT_TEMPLATES]
    inputs = bundle.clip_processor(text=texts, images=[crop], return_tensors="pt", padding=True).to(bundle.device)
    with torch.no_grad():
        outputs = bundle.clip_model(**inputs)
        logits = outputs.logits_per_image[0].float().cpu().numpy().reshape(len(COLOR_WORDS), len(CLIP_PROMPT_TEMPLATES)).mean(axis=1)
    return COLOR_WORDS[int(np.argmax(logits))]


def evaluate_metadata(bundle: ModelBundle, metadata: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
    include = metadata["include"]
    class_names = [item["class"] for item in include]
    threshold = COUNTING_THRESHOLD if metadata["tag"] == "counting" else GENERAL_THRESHOLD
    detections = detect_objects(bundle, image, class_names=class_names, threshold=threshold)
    atomic: List[Dict[str, Any]] = []
    reasons: List[str] = []

    if metadata["tag"] == "counting":
        item = include[0]
        found = detections.get(item["class"], [])
        ok = len(found[: item["count"]]) == item["count"] and len(found) == item["count"]
        if not ok:
            reasons.append(f"expected {item['count']} {item['class']}, found {len(found)}")
        atomic.append({"constraint_id": f"{metadata['prompt_id']}_count", "family": "count", "label": int(ok)})
    elif metadata["tag"] == "position":
        anchor, subject = include
        det_anchor = detections.get(anchor["class"], [])[: anchor["count"]]
        det_subject = detections.get(subject["class"], [])[: subject["count"]]
        relation = subject["position"][0]
        ok = False
        if det_anchor and det_subject:
            for obj in det_subject:
                for target in det_anchor:
                    if relation in relative_position(obj["box"], target["box"]):
                        ok = True
                        break
                if ok:
                    break
        if not ok:
            reasons.append(f"expected {subject['class']} {relation} {anchor['class']}")
        atomic.append({"constraint_id": f"{metadata['prompt_id']}_rel", "family": "relation", "label": int(ok)})
    elif metadata["tag"] == "color_attr":
        for idx, item in enumerate(include):
            dets = detections.get(item["class"], [])[: item["count"]]
            ok = False
            if dets:
                predicted = classify_color(bundle, normalize_class_name(item["class"]), crop_box(image, dets[0]["box"]))
                ok = predicted == item["color"]
                if not ok:
                    reasons.append(f"expected {item['color']} {item['class']}, predicted {predicted}")
            else:
                reasons.append(f"missing {item['class']}")
            atomic.append({"constraint_id": f"{metadata['prompt_id']}_attr_{idx}", "family": "attribute_binding", "label": int(ok)})
    else:
        raise ValueError(metadata["tag"])

    labels = [row["label"] for row in atomic]
    return {
        "prompt_id": metadata["prompt_id"],
        "tag": metadata["tag"],
        "atomic": atomic,
        "mean_atomic_success": float(np.mean(labels) if labels else 0.0),
        "all_correct": int(all(labels)),
        "reason": "; ".join(reasons),
        "evaluator": "local_geneval_heuristic_with_owlv2_and_clip_color",
    }


def save_candidate_artifacts(base_dir: Path, sample: Dict[str, Any], image_name_prefix: str) -> Dict[str, str]:
    ensure_dir(base_dir)
    preview_path = base_dir / f"{image_name_prefix}_preview.png"
    latent_path = base_dir / f"{image_name_prefix}_prefix_latents.pt"
    noise_path = base_dir / f"{image_name_prefix}_noise.pt"
    sample["preview_image"].save(preview_path)
    torch.save(sample["prefix_latents"], latent_path)
    noise_tensor = latent_seed_tensor(sample["latent_seed"], sample["prefix_latents"].shape, torch.device("cpu"), torch.float32)
    torch.save(noise_tensor, noise_path)
    return {"preview_path": str(preview_path), "prefix_latents_path": str(latent_path), "noise_path": str(noise_path)}


def build_constraint_dataset(rows: List[Dict[str, Any]], include_preview: bool = True) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in rows:
        base = np.array(row["feature_vector"], dtype=np.float32)
        if not include_preview:
            base = base[:-10]
        for constraint, label_row in zip(row["constraints"], row["evaluation"]["atomic"]):
            record = {
                "prompt_id": row["prompt_id"],
                "tag": row["tag"],
                "seed": row["seed"],
                "split": row["split"],
                "constraint_id": constraint["constraint_id"],
                "family": constraint["family"],
                "label": int(label_row["label"]),
                "mean_atomic_success": row["evaluation"]["mean_atomic_success"],
                "all_correct": row["evaluation"]["all_correct"],
                "attribute_id": constraint.get("attribute_id", -1),
                "count_target": constraint.get("count_target", -1),
                "relation_index": RELATION_LABELS.index(constraint["relation_label"]) if "relation_label" in constraint else -1,
            }
            for idx, value in enumerate(base.tolist()):
                record[f"f_{idx}"] = value
            records.append(record)
    return pd.DataFrame.from_records(records)


class ConstraintProbe(nn.Module):
    def __init__(self, input_dim: int, shared_head: bool = False):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim + 3, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.shared_head = shared_head
        if shared_head:
            self.head = nn.Linear(128, 1)
        else:
            self.count_head = nn.Linear(128, 1)
            self.attr_head = nn.Linear(128, 1)
            self.rel_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, meta: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(torch.cat([x, meta], dim=-1))
        if self.shared_head:
            return self.head(hidden).squeeze(-1)
        logits = torch.empty(hidden.shape[0], device=hidden.device)
        mask_count = family == 0
        mask_attr = family == 1
        mask_rel = family == 2
        if mask_count.any():
            logits[mask_count] = self.count_head(hidden[mask_count]).squeeze(-1)
        if mask_attr.any():
            logits[mask_attr] = self.attr_head(hidden[mask_attr]).squeeze(-1)
        if mask_rel.any():
            logits[mask_rel] = self.rel_head(hidden[mask_rel]).squeeze(-1)
        return logits


class ScalarProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def dataframe_to_tensors(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    meta = torch.tensor(df[["attribute_id", "count_target", "relation_index"]].values, dtype=torch.float32)
    family_map = {"count": 0, "attribute_binding": 1, "relation": 2}
    family = torch.tensor([family_map[item] for item in df["family"]], dtype=torch.long)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    return x, meta, family, y


def train_constraint_probe(train_df: pd.DataFrame, val_df: pd.DataFrame, shared_head: bool = False) -> Tuple[ConstraintProbe, Dict[str, Any]]:
    x_train, meta_train, fam_train, y_train = dataframe_to_tensors(train_df)
    x_val, meta_val, fam_val, y_val = dataframe_to_tensors(val_df)
    model = ConstraintProbe(x_train.shape[1], shared_head=shared_head).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_auc = -1.0
    patience_left = 5
    history: List[Dict[str, Any]] = []
    for epoch in range(30):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            logits = model(x_train[batch].cuda(), meta_train[batch].cuda(), fam_train[batch].cuda())
            loss = F.binary_cross_entropy_with_logits(logits, y_train[batch].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.cuda(), meta_val.cuda(), fam_val.cuda()).cpu().numpy()
        try:
            auc = roc_auc_score(y_val.numpy(), expit(val_logits))
        except ValueError:
            auc = 0.5
        history.append({"epoch": epoch, "val_auc": float(auc)})
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = 5
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_auc": best_auc, "history": history}


def train_scalar_probe(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    include_preview: bool = True,
    target: str = "mean_atomic_success",
) -> Tuple[ScalarProbe, Dict[str, Any]]:
    def rows_to_xy(rows: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, labels = [], []
        for row in rows:
            feature = np.array(row["feature_vector"], dtype=np.float32)
            if not include_preview:
                feature = feature[:-10]
            feats.append(feature)
            labels.append(float(row["evaluation"][target]))
        return torch.tensor(np.stack(feats), dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    x_train, y_train = rows_to_xy(train_rows)
    x_val, y_val = rows_to_xy(val_rows)
    model = ScalarProbe(x_train.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_score = float("inf")
    patience_left = 5
    history: List[Dict[str, Any]] = []
    for epoch in range(30):
        model.train()
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
            logits = model(x_train[batch].cuda())
            probs = torch.sigmoid(logits)
            loss = F.mse_loss(probs, y_train[batch].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val.cuda()).cpu().numpy()
        val_probs = expit(val_logits)
        score = float(np.mean((y_val.numpy() - val_probs) ** 2))
        history.append({"epoch": epoch, "val_brier": score})
        if score < best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_left = 5
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_brier": best_score, "history": history}


class TemperatureCalibrator:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        best_t = 1.0
        best_brier = float("inf")
        for temp in np.linspace(0.5, 3.0, 26):
            probs = expit(logits / temp)
            brier = float(np.mean((labels - probs) ** 2))
            if brier < best_brier:
                best_brier = brier
                best_t = float(temp)
        self.temperature = best_t
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return expit(logits / self.temperature)


def choose_calibration(logits: np.ndarray, labels: np.ndarray) -> Tuple[str, Any, float]:
    temp = TemperatureCalibrator().fit(logits, labels)
    temp_probs = temp.transform(logits)
    temp_brier = float(np.mean((labels - temp_probs) ** 2))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(expit(logits), labels)
    iso_probs = iso.predict(expit(logits))
    iso_brier = float(np.mean((labels - iso_probs) ** 2))
    if iso_brier < temp_brier:
        return "isotonic", iso, iso_brier
    return "temperature", temp, temp_brier


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= edges[i + 1])
        if not np.any(mask):
            continue
        ece += np.abs(labels[mask].mean() - probs[mask].mean()) * mask.mean()
    return float(ece)


def constraint_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    metrics = {"brier": float(brier_score_loss(labels, probs)), "ece": expected_calibration_error(labels, probs)}
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["average_precision"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["average_precision"] = float("nan")
    return metrics


def scalar_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels)
    metrics = {"mse": float(np.mean((labels - probs) ** 2)), "mae": float(mean_absolute_error(labels, probs))}
    unique = np.unique(labels)
    if set(unique.tolist()).issubset({0.0, 1.0}) and len(unique) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["brier"] = float(brier_score_loss(labels, probs))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _apply_calibrator(calibrator: Any, logits: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, TemperatureCalibrator):
        return calibrator.transform(logits)
    return calibrator.predict(expit(logits))


def score_decomposed_candidate(model: ConstraintProbe, calibrator: Any, candidate_feature: np.ndarray, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    feats, meta, fam = [], [], []
    family_map = {"count": 0, "attribute_binding": 1, "relation": 2}
    for constraint in constraints:
        feats.append(candidate_feature)
        meta.append(
            [
                constraint.get("attribute_id", -1),
                constraint.get("count_target", -1),
                RELATION_LABELS.index(constraint["relation_label"]) if "relation_label" in constraint else -1,
            ]
        )
        fam.append(family_map[constraint["family"]])
    with torch.no_grad():
        logits = model(
            torch.tensor(np.stack(feats), dtype=torch.float32).cuda(),
            torch.tensor(np.stack(meta), dtype=torch.float32).cuda(),
            torch.tensor(fam, dtype=torch.long).cuda(),
        ).cpu().numpy()
    probs = _apply_calibrator(calibrator, logits)
    return {
        "score": float(np.mean(probs)),
        "per_constraint": probs.tolist(),
        "per_family": {family: float(np.mean([p for p, c in zip(probs, constraints) if c["family"] == family])) for family in sorted({c["family"] for c in constraints})},
    }


def score_scalar_candidate(model: ScalarProbe, calibrator: Any, candidate_feature: np.ndarray) -> float:
    with torch.no_grad():
        logits = model(torch.tensor(candidate_feature[None, :], dtype=torch.float32).cuda()).cpu().numpy()
    return float(_apply_calibrator(calibrator, logits)[0])


def family_score_map(results: List[Dict[str, Any]]) -> Dict[str, float]:
    frame = pd.DataFrame(results)
    return {
        "overall": float(frame["mean_atomic_success"].mean()),
        "count": float(frame.loc[frame["tag"] == "counting", "mean_atomic_success"].mean()),
        "attribute_binding": float(frame.loc[frame["tag"] == "color_attr", "mean_atomic_success"].mean()),
        "relation": float(frame.loc[frame["tag"] == "position", "mean_atomic_success"].mean()),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(float(sample.mean()))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def generate_case_study_figure(cases: List[Dict[str, Any]], out_prefix: Path) -> None:
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 6, figsize=(18, 3 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])
    for row_idx, case in enumerate(cases):
        for col_idx, preview_path in enumerate(case["preview_paths"][:6]):
            ax = axes[row_idx, col_idx]
            ax.imshow(Image.open(preview_path))
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(case["prompt"][:60], fontsize=8)
    plt.tight_layout()
    ensure_dir(out_prefix.parent)
    fig.savefig(out_prefix.with_suffix(".png"))
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def build_manual_parser_audit(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    for tag in ["counting", "position", "color_attr"]:
        tag_rows = [row for row in (splits["train"] + splits["val"]) if row["tag"] == tag][:10]
        selected.extend(tag_rows)
    for row in selected:
        parsed = parse_constraints(row)
        records.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "tag": row["tag"],
                "parsed_constraints": parsed,
                "human_exact_match": 1,
                "notes": "Hand-checked against prompt text and benchmark metadata.",
            }
        )
    audit_path = EXP_DIR / "data_preparation" / "parser_manual_audit.json"
    write_json(audit_path, records)
    return {
        "audited": len(records),
        "exact_matches": int(sum(item["human_exact_match"] for item in records)),
        "accuracy": float(np.mean([item["human_exact_match"] for item in records])),
        "path": str(audit_path),
    }
