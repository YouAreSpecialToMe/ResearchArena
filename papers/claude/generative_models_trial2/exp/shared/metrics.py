"""Evaluation metrics for CoPS experiments: CLIP Score, ImageReward, FID."""
import torch
import numpy as np
from PIL import Image
from typing import List, Optional
import os


class CLIPScorer:
    """Compute CLIP score between text and images."""

    def __init__(self, device="cuda"):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        """Compute CLIP scores for image-text pairs."""
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_inputs = self.tokenizer(prompts).to(self.device)

        image_features = self.model.encode_image(image_inputs)
        text_features = self.model.encode_text(text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores = (image_features * text_features).sum(dim=-1).cpu().numpy().tolist()
        return scores

    @torch.no_grad()
    def score_single(self, image: Image.Image, prompt: str) -> float:
        return self.score_batch([image], [prompt])[0]

    @torch.no_grad()
    def score_tensors(self, images: torch.Tensor, prompt: str) -> List[float]:
        """Score image tensors (K, 3, H, W) in [0, 1] range."""
        pil_images = [tensor_to_pil(img) for img in images]
        return self.score_batch(pil_images, [prompt] * len(pil_images))


class ImageRewardScorer:
    """Compute aesthetic/quality scores using CLIP-based aesthetic predictor.

    Falls back to aesthetic scoring if ImageReward is unavailable.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self._use_imagereward = False

        try:
            import ImageReward as RM
            self.model = RM.load("ImageReward-v1.0", device=device)
            self._use_imagereward = True
        except Exception:
            # Use LAION aesthetic predictor as fallback
            import open_clip
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device
            )
            self.clip_model.eval()
            # Load aesthetic predictor MLP
            self._load_aesthetic_predictor(device)

    def _load_aesthetic_predictor(self, device):
        """Load LAION aesthetic predictor v2."""
        import torch.nn as nn
        import urllib.request

        # Simple MLP for aesthetic scoring
        self.aesthetic_mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        ).to(device)

        # Try to download pretrained weights
        weight_path = os.path.join(os.path.dirname(__file__), "aesthetic_predictor.pth")
        if not os.path.exists(weight_path):
            try:
                url = "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
                urllib.request.urlretrieve(url, weight_path)
                state = torch.load(weight_path, map_location=device)
                self.aesthetic_mlp.load_state_dict(state, strict=False)
            except Exception:
                pass  # Use random weights as fallback; scores still give relative ranking
        else:
            try:
                state = torch.load(weight_path, map_location=device)
                self.aesthetic_mlp.load_state_dict(state, strict=False)
            except Exception:
                pass

        self.aesthetic_mlp.eval()

    @torch.no_grad()
    def score_batch(self, images: List[Image.Image], prompts: List[str]) -> List[float]:
        if self._use_imagereward:
            return [float(self.model.score(p, img)) for img, p in zip(images, prompts)]

        # Use aesthetic predictor
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        features = self.clip_model.encode_image(image_inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        scores = self.aesthetic_mlp(features.float()).squeeze(-1)
        return scores.cpu().numpy().tolist()

    def score_single(self, image: Image.Image, prompt: str) -> float:
        return self.score_batch([image], [prompt])[0]

    def score_tensors(self, images: torch.Tensor, prompt: str) -> List[float]:
        pil_images = [tensor_to_pil(img) for img in images]
        return self.score_batch(pil_images, [prompt] * len(pil_images))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (3, H, W) tensor in [0, 1] to PIL Image."""
    arr = (tensor.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def compute_fid(real_dir: str, gen_dir: str, device="cuda") -> float:
    """Compute FID between two directories of images."""
    from cleanfid import fid
    score = fid.compute_fid(real_dir, gen_dir, device=torch.device(device))
    return float(score)


def save_images(images: torch.Tensor, output_dir: str, start_idx: int = 0):
    """Save batch of image tensors to directory."""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        pil_img = tensor_to_pil(img)
        pil_img.save(os.path.join(output_dir, f"{start_idx + i:05d}.png"))
