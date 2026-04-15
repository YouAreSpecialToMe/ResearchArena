from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionPipeline

from exp.shared.common import CFG_SCALE, GUIDED_STEPS, IMAGE_SIZE, MODEL_ID, NUM_STEPS, elapsed_seconds, latent_hash, timer


@dataclass
class EncodedPrompt:
    prompt: str
    input_ids: torch.Tensor
    embeddings: torch.Tensor
    slot_token_map: dict[str, list[int]]


@dataclass
class MethodConfig:
    name: str
    alpha_schedule: list[float]
    gate_c: float = 1.0
    guided_steps: list[int] | None = None
    use_gate: bool = False
    use_slot_redistribution: bool = False
    constant_alpha: float | None = None
    paraphrase_mode: str = "approved"


class SDRunner:
    def __init__(self, device: str = "cuda") -> None:
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        if device == "cuda":
            self.pipe.enable_attention_slicing()
        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler

    def _find_slot_token_indices(self, input_ids: list[int], aliases: list[str]) -> list[int]:
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            alias_ids = self.tokenizer(
                alias,
                add_special_tokens=False,
            )["input_ids"]
            if not alias_ids:
                continue
            window = len(alias_ids)
            for start in range(len(input_ids) - window + 1):
                if input_ids[start : start + window] == alias_ids:
                    return list(range(start, start + window))
        return []

    def _encode(self, prompt: str, slot_alias_map: dict[str, list[str]] | None = None) -> EncodedPrompt:
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        slot_token_map: dict[str, list[int]] = {}
        if slot_alias_map:
            input_id_list = input_ids[0].tolist()
            for slot_name, aliases in slot_alias_map.items():
                slot_token_map[slot_name] = self._find_slot_token_indices(input_id_list, aliases)
        with torch.no_grad():
            encoded = self.text_encoder(input_ids.to(self.device))[0]
        return EncodedPrompt(
            prompt=prompt,
            input_ids=input_ids.to(self.device),
            embeddings=encoded,
            slot_token_map=slot_token_map,
        )

    def _predict_cfg_eps(self, latents: torch.Tensor, timestep: torch.Tensor, prompt_emb: torch.Tensor, uncond_emb: torch.Tensor) -> torch.Tensor:
        latent_input = self.scheduler.scale_model_input(latents, timestep)
        latent_input = torch.cat([latent_input, latent_input], dim=0)
        emb = torch.cat([uncond_emb, prompt_emb], dim=0)
        with torch.no_grad():
            noise_pred = self.unet(latent_input, timestep, encoder_hidden_states=emb).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        return noise_uncond + CFG_SCALE * (noise_text - noise_uncond)

    def _build_slot_conditioned_embedding(
        self,
        encoded_prompts: list[EncodedPrompt],
        gate_value: float,
        alpha: float,
    ) -> tuple[torch.Tensor, dict[str, dict[str, float]]]:
        original = encoded_prompts[0]
        blended = original.embeddings.clone()
        slot_scores: dict[str, float] = {}
        slot_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for slot_name, original_positions in original.slot_token_map.items():
            if not original_positions:
                continue
            slot_vectors = []
            for encoded in encoded_prompts:
                positions = encoded.slot_token_map.get(slot_name, [])
                if not positions:
                    continue
                slot_vectors.append(encoded.embeddings[:, positions, :].mean(dim=1))
            if len(slot_vectors) < 2:
                continue
            stacked = torch.cat(slot_vectors, dim=0)
            consensus = stacked.mean(dim=0, keepdim=True)
            disagreement = torch.norm(stacked - consensus, dim=-1).mean().item()
            slot_scores[slot_name] = disagreement
            slot_stats[slot_name] = (
                original.embeddings[:, original_positions, :].mean(dim=1, keepdim=True),
                consensus.unsqueeze(1),
            )

        if not slot_stats:
            return blended, {}

        total_score = sum(slot_scores.values())
        if total_score <= 1e-8:
            normalized = {slot_name: 1.0 / len(slot_stats) for slot_name in slot_stats}
        else:
            normalized = {slot_name: score / total_score for slot_name, score in slot_scores.items()}

        diagnostics: dict[str, dict[str, float]] = {}
        for slot_name, (orig_mean, consensus_mean) in slot_stats.items():
            positions = original.slot_token_map[slot_name]
            slot_weight = normalized[slot_name]
            slot_delta = slot_weight * (consensus_mean - orig_mean)
            blended[:, positions, :] = blended[:, positions, :] + slot_delta
            diagnostics[slot_name] = {
                "weight": float(slot_weight),
                "disagreement": float(slot_scores[slot_name]),
                "embedding_delta_norm": float(torch.norm(alpha * gate_value * slot_delta.squeeze(1)).item()),
            }
        return blended, diagnostics

    def generate(
        self,
        prompt_id: str,
        prompts: list[str],
        seed: int,
        method: MethodConfig,
        output_path: Path,
        slot_alias_map: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        guided_steps = method.guided_steps or GUIDED_STEPS
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn((1, self.unet.config.in_channels, IMAGE_SIZE // 8, IMAGE_SIZE // 8), generator=generator, device=self.device, dtype=self.pipe.unet.dtype)
        uncond_emb = self._encode("").embeddings
        encoded_prompts = [self._encode(prompt, slot_alias_map=slot_alias_map) for prompt in prompts]
        self.scheduler.set_timesteps(NUM_STEPS, device=self.device)
        start = timer()
        gate_trace = []
        torch.cuda.reset_peak_memory_stats() if self.device == "cuda" else None
        for idx, timestep in enumerate(self.scheduler.timesteps):
            eps0 = self._predict_cfg_eps(latents, timestep, encoded_prompts[0].embeddings, uncond_emb)
            eps = eps0
            gate_value = 0.0
            slot_diagnostics: dict[str, dict[str, float]] = {}
            if idx in guided_steps and len(encoded_prompts) > 1:
                eps_list = [eps0]
                for encoded in encoded_prompts[1:]:
                    eps_list.append(self._predict_cfg_eps(latents, timestep, encoded.embeddings, uncond_emb))
                stacked = torch.stack(eps_list, dim=0)
                consensus = stacked.mean(dim=0)
                if method.constant_alpha is not None:
                    alpha = method.constant_alpha
                else:
                    alpha = method.alpha_schedule[guided_steps.index(idx)]
                if method.use_gate:
                    variance = stacked.var(dim=0, unbiased=False).mean().item()
                    gate_value = float(min(1.0, method.gate_c * variance * 10.0))
                else:
                    gate_value = 1.0
                if method.use_slot_redistribution:
                    blended_prompt_emb, slot_diagnostics = self._build_slot_conditioned_embedding(encoded_prompts, gate_value=gate_value, alpha=alpha)
                    redistributed = self._predict_cfg_eps(latents, timestep, blended_prompt_emb, uncond_emb)
                    eps = eps0 + alpha * gate_value * (redistributed - eps0)
                else:
                    eps = eps0 + alpha * gate_value * (consensus - eps0)
            latents = self.scheduler.step(eps, timestep, latents).prev_sample
            gate_trace.append(
                {
                    "step_index": idx,
                    "timestep": int(timestep.item()),
                    "gate": gate_value,
                    "slot_corrections": slot_diagnostics,
                }
            )
        with torch.no_grad():
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        pil = Image.fromarray((image * 255).astype(np.uint8))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(output_path)
        return {
            "runtime_seconds": elapsed_seconds(start),
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if self.device == "cuda" else 0.0,
            "latent_hash": latent_hash(seed, prompt_id),
            "gate_trace": gate_trace,
            "output_path": str(output_path),
        }
