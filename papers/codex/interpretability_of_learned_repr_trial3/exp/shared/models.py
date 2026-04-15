from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_sparsify(z: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(z, k=min(k, z.shape[1]), dim=1)
    mask = torch.zeros_like(z)
    mask.scatter_(1, indices, 1.0)
    return z * mask, mask


@dataclass
class BlockSpec:
    inv_slice: slice
    factor_slices: list[slice]
    residual_slice: slice


def build_block_spec(latent_dim: int, num_factors: int) -> BlockSpec:
    inv = int(round(0.2 * latent_dim))
    residual = int(round(0.2 * latent_dim))
    factor_total = latent_dim - inv - residual
    per_factor = factor_total // num_factors
    cursor = 0
    inv_slice = slice(cursor, cursor + inv)
    cursor += inv
    factor_slices = []
    for _ in range(num_factors):
        factor_slices.append(slice(cursor, cursor + per_factor))
        cursor += per_factor
    residual_slice = slice(cursor, latent_dim)
    return BlockSpec(inv_slice, factor_slices, residual_slice)


class LinearSAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, topk: int, method: str, anchor_bank: torch.Tensor | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.topk = topk
        self.method = method
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        if method == "ra_sae":
            if anchor_bank is None:
                raise ValueError("RA-SAE requires anchor bank")
            self.register_buffer("anchor_bank", anchor_bank)
            self.anchor_logits = nn.Parameter(torch.randn(latent_dim, anchor_bank.shape[0]) * 0.01)
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(latent_dim, input_dim)
            with torch.no_grad():
                self.decoder.weight.copy_(self.encoder.weight.t())
                nn.init.zeros_(self.decoder.bias)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.method == "ra_sae":
            weights = torch.softmax(self.anchor_logits, dim=1)
            decoder_weight = weights @ self.anchor_bank
            return z @ decoder_weight + self.decoder_bias
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        pre = F.relu(self.encoder(x) + self.bias)
        sparse, mask = topk_sparsify(pre, self.topk)
        recon = self.decode(sparse)
        return {"pre": pre, "z": sparse, "mask": mask, "recon": recon}
