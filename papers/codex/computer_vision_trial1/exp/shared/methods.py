from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import open_clip
import torch
import torch.nn.functional as F

from exp.shared.common import CORRUPTION_FAMILIES, MODEL_NAME


DISPLAY_FAMILY = {
    "gaussian_noise": "gaussian noise",
    "motion_blur": "motion blur",
    "fog": "fog",
    "jpeg_compression": "jpeg compression artifacts",
}

Q_PROMPTS = {
    "base": {
        "gaussian_noise": ["an image with gaussian noise corruption"],
        "motion_blur": ["an image with motion blur corruption"],
        "fog": ["an image with fog corruption"],
        "jpeg_compression": ["an image with jpeg compression artifacts"],
    },
    "strong": {
        "gaussian_noise": [
            "an image with gaussian noise corruption",
            "a photo ruined by sensor noise",
            "a grainy noisy image",
        ],
        "motion_blur": [
            "an image with motion blur corruption",
            "a photo blurred by camera motion",
            "an image smeared by movement",
        ],
        "fog": [
            "an image with fog corruption",
            "a hazy low-visibility scene",
            "a photo obscured by fog",
        ],
        "jpeg_compression": [
            "an image with jpeg compression artifacts",
            "a blocky over-compressed photo",
            "an image distorted by compression artifacts",
        ],
    },
}

RESIDUAL_PROMPTS = {
    "base": {
        "gaussian_noise": ["a photo of a {class_name} with gaussian noise"],
        "motion_blur": ["a photo of a {class_name} with motion blur"],
        "fog": ["a photo of a {class_name} with fog"],
        "jpeg_compression": ["a photo of a {class_name} with jpeg compression artifacts"],
    },
    "strong": {
        "gaussian_noise": [
            "a photo of a {class_name} with gaussian noise",
            "a grainy noisy photo of a {class_name}",
            "a photo of a {class_name} ruined by sensor noise",
        ],
        "motion_blur": [
            "a photo of a {class_name} with motion blur",
            "a camera-motion blurred photo of a {class_name}",
            "a smeared moving photo of a {class_name}",
        ],
        "fog": [
            "a photo of a {class_name} with fog",
            "a hazy photo of a {class_name}",
            "a low-visibility foggy photo of a {class_name}",
        ],
        "jpeg_compression": [
            "a photo of a {class_name} with jpeg compression artifacts",
            "a blocky over-compressed photo of a {class_name}",
            "a photo of a {class_name} distorted by compression artifacts",
        ],
    },
}

CLEAN_ENSEMBLE_TEMPLATES = [
    "a photo of a {class_name}",
    "an image of a {class_name}",
    "a close-up photo of a {class_name}",
    "a centered photo of a {class_name}",
    "a blurry photo of a {class_name}",
]


@dataclass
class ModelBundle:
    model: Any
    preprocess: Any
    tokenizer: Any
    device: torch.device


def load_openclip(device: torch.device) -> ModelBundle:
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained="laion2b_s34b_b79k")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return ModelBundle(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)


def _encode_texts(bundle: ModelBundle, prompts: list[str]) -> torch.Tensor:
    tokenized = bundle.tokenizer(prompts).to(bundle.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bundle.device.type == "cuda"):
        features = bundle.model.encode_text(tokenized)
    return F.normalize(features.float(), dim=-1).cpu()


@lru_cache(maxsize=8)
def _cached_single_template(class_key: tuple[str, ...], device_type: str) -> torch.Tensor:
    raise RuntimeError("Single-template cache must be populated via build_text_bank.")


def build_text_bank(bundle: ModelBundle, classnames: list[str]) -> dict[str, Any]:
    clean_single = _encode_texts(bundle, [f"a photo of a {name}" for name in classnames])
    clean_clear = _encode_texts(bundle, [f"a clear photo of a {name}" for name in classnames])
    ensemble = []
    for name in classnames:
        prompts = [template.format(class_name=name) for template in CLEAN_ENSEMBLE_TEMPLATES]
        features = _encode_texts(bundle, prompts)
        ensemble.append(F.normalize(features.mean(dim=0), dim=0))
    ensemble = torch.stack(ensemble, dim=0)

    naive = []
    for family in CORRUPTION_FAMILIES:
        prompts = [f"a photo of a {name} with {DISPLAY_FAMILY[family]}" for name in classnames]
        naive.append(_encode_texts(bundle, prompts))
    naive = torch.stack(naive, dim=0)

    generic_low_quality = _encode_texts(bundle, [f"a low-quality photo of a {name}" for name in classnames])
    generic_residual = F.normalize((generic_low_quality - clean_clear).mean(dim=0), dim=0)

    q_bank = {}
    residual_bank = {}
    for variant in ["base", "strong"]:
        q_vectors = []
        residuals = {}
        for family in CORRUPTION_FAMILIES:
            q_prompts = Q_PROMPTS[variant][family]
            q_feat = _encode_texts(bundle, q_prompts)
            q_vectors.append(F.normalize(q_feat.mean(dim=0), dim=0))

            per_class_diffs = []
            for class_name, clear_vec in zip(classnames, clean_clear):
                corr_prompts = [template.format(class_name=class_name) for template in RESIDUAL_PROMPTS[variant][family]]
                corr_feat = _encode_texts(bundle, corr_prompts)
                corr_mean = F.normalize(corr_feat.mean(dim=0), dim=0)
                per_class_diffs.append(corr_mean - clear_vec)
            residuals[family] = F.normalize(torch.stack(per_class_diffs, dim=0).mean(dim=0), dim=0)
        q_bank[variant] = torch.stack(q_vectors, dim=0)
        residual_bank[variant] = residuals

    return {
        "clean_single": clean_single,
        "clean_ensemble": ensemble,
        "clean_clear": clean_clear,
        "naive": naive,
        "generic_residual": generic_residual,
        "q_bank": q_bank,
        "residual_bank": residual_bank,
        "selected_prompts": {
            "clean_ensemble": CLEAN_ENSEMBLE_TEMPLATES,
            "q_prompts": Q_PROMPTS,
            "residual_prompts": RESIDUAL_PROMPTS,
        },
    }


def family_posterior(image_features: torch.Tensor, q_vectors: torch.Tensor, beta: float) -> torch.Tensor:
    logits = image_features @ q_vectors.T
    return torch.softmax(beta * logits, dim=1)


def score_zero_shot(image_features: torch.Tensor, clean_single: torch.Tensor) -> torch.Tensor:
    return image_features @ clean_single.T


def score_clean_ensemble(image_features: torch.Tensor, clean_ensemble: torch.Tensor) -> torch.Tensor:
    return image_features @ clean_ensemble.T


def score_naive(
    image_features: torch.Tensor,
    q: torch.Tensor,
    naive_prototypes: torch.Tensor,
) -> torch.Tensor:
    family_scores = torch.einsum("nd,fcd->nfc", image_features, naive_prototypes)
    return torch.einsum("nf,nfc->nc", q, family_scores)


def score_generic_residual(
    image_features: torch.Tensor,
    clean_single: torch.Tensor,
    generic_residual: torch.Tensor,
    alpha: float,
    lam: float,
) -> torch.Tensor:
    calibrated = F.normalize(clean_single + alpha * generic_residual.unsqueeze(0), dim=1)
    base = image_features @ clean_single.T
    adjusted = image_features @ calibrated.T
    return base + lam * (adjusted - base)


def score_family_residual(
    image_features: torch.Tensor,
    q: torch.Tensor,
    clean_single: torch.Tensor,
    residual_bank: dict[str, torch.Tensor],
    alpha: float,
    lam: float,
    family_order: list[str] | None = None,
) -> torch.Tensor:
    family_order = family_order or CORRUPTION_FAMILIES
    calibrated = []
    for family in family_order:
        calibrated.append(F.normalize(clean_single + alpha * residual_bank[family].unsqueeze(0), dim=1))
    calibrated = torch.stack(calibrated, dim=0)
    base = image_features @ clean_single.T
    family_scores = torch.einsum("nd,fcd->nfc", image_features, calibrated)
    mixed = torch.einsum("nf,nfc->nc", q, family_scores)
    return base + lam * (mixed - base)

