from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class HeatmapRecord:
    phrase: str
    token_indices: list[int]
    heatmap: np.ndarray


class TraceCollector:
    def __init__(self, tokenizer, prompt: str, phrases: list[str], keep_last_steps: int = 10) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.phrases = [p for p in phrases if p]
        self.keep_last_steps = keep_last_steps
        self.step_maps: list[dict[str, torch.Tensor]] = []
        self.current_step: dict[str, list[torch.Tensor]] = defaultdict(list)
        tokenized = tokenizer(prompt, return_tensors="pt")
        self.input_ids = tokenized["input_ids"][0].tolist()
        decoded = [tokenizer.decode([tok]).strip().lower() for tok in self.input_ids]
        self.phrase_token_indices: dict[str, list[int]] = {}
        for phrase in self.phrases:
            parts = phrase.lower().split()
            indices = []
            for idx, piece in enumerate(decoded):
                normalized = piece.replace("</w>", "").strip()
                if normalized in parts:
                    indices.append(idx)
            self.phrase_token_indices[phrase] = indices

    def add(self, phrase_to_map: dict[str, torch.Tensor]) -> None:
        self.step_maps.append({k: v.detach().cpu() for k, v in phrase_to_map.items()})
        if len(self.step_maps) > self.keep_last_steps:
            self.step_maps.pop(0)

    def finalize(self) -> list[HeatmapRecord]:
        if not self.step_maps:
            return []
        merged: dict[str, list[torch.Tensor]] = defaultdict(list)
        for step in self.step_maps:
            for phrase, value in step.items():
                merged[phrase].append(value)
        out = []
        for phrase, tensors in merged.items():
            heatmap = torch.stack(tensors, dim=0).mean(dim=0).numpy().astype(np.float16)
            out.append(HeatmapRecord(phrase=phrase, token_indices=self.phrase_token_indices.get(phrase, []), heatmap=heatmap))
        return out


class RecordingAttnProcessor:
    def __init__(self, layer_name: str, collector: TraceCollector | None = None) -> None:
        self.layer_name = layer_name
        self.collector = collector

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channels, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channels, height * width).transpose(1, 2)
        else:
            batch, _, _ = hidden_states.shape
            height = width = int(math.sqrt(hidden_states.shape[1]))

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, hidden_states.shape[1], batch)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        scale = attn.scale if hasattr(attn, "scale") else 1.0 / math.sqrt(query.shape[-1])
        attention_scores = torch.bmm(query, key.transpose(1, 2)) * scale
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = attention_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not hidden_states and self.collector is not None and "up_blocks" in self.layer_name:
            heads = attention_probs.shape[0] // batch
            probs = attention_probs.view(batch, heads, attention_probs.shape[1], attention_probs.shape[2]).mean(dim=1)
            phrase_to_map = {}
            for phrase, token_indices in self.collector.phrase_token_indices.items():
                valid = [idx for idx in token_indices if idx < probs.shape[-1]]
                if not valid:
                    continue
                token_map = probs[:, :, valid].mean(dim=-1)
                side = int(math.sqrt(token_map.shape[1]))
                if side * side != token_map.shape[1]:
                    continue
                token_map = token_map[0].reshape(side, side)
                phrase_to_map[phrase] = token_map
            if phrase_to_map:
                self.collector.add(phrase_to_map)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channels, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_recording_processors(pipe, collector: TraceCollector) -> dict[str, object]:
    original = dict(pipe.unet.attn_processors)
    processors = {}
    for name in pipe.unet.attn_processors.keys():
        processors[name] = RecordingAttnProcessor(name, collector)
    pipe.unet.set_attn_processor(processors)
    return original


def restore_processors(pipe, original: dict[str, object]) -> None:
    pipe.unet.set_attn_processor(original)


def upsample_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(heatmap.astype(np.float32))[None, None]
    up = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)[0, 0]
    arr = up.numpy()
    denom = arr.sum()
    if denom > 0:
        arr = arr / denom
    return arr.astype(np.float32)
