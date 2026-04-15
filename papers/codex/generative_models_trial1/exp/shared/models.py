from __future__ import annotations

from functools import lru_cache

import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionXLPipeline
from PIL import Image
from transformers import AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@lru_cache(maxsize=1)
def load_sdxl():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE,
        use_safetensors=True,
        variant="fp16" if DEVICE == "cuda" else None,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    return pipe


@lru_cache(maxsize=1)
def load_siglip():
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def load_detector():
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(DEVICE)
    model.eval()
    return processor, model


@torch.inference_mode()
def siglip_score_image_text(image: Image.Image, text: str) -> float:
    processor, model = load_siglip()
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(DEVICE)
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_embeds * text_embeds).sum().item())


@torch.inference_mode()
def detect_boxes(image: Image.Image, query: str, box_threshold: float = 0.25, text_threshold: float = 0.20) -> list[dict]:
    processor, model = load_detector()
    inputs = processor(images=image, text=query, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    boxes = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        boxes.append({"box": [x1, y1, x2, y2], "score": float(score), "label": str(label)})
    return boxes
