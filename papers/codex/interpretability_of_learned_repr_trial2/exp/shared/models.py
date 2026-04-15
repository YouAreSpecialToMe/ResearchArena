from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainOutput:
    metrics: dict
    history: list


class BaseSAE(nn.Module):
    def __init__(self, d_in: int, width: int = 1024):
        super().__init__()
        self.encoder = nn.Linear(d_in, width)
        self.decoder_bias = nn.Parameter(torch.zeros(d_in))
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

    @property
    def decoder_weight(self):
        return self.encoder.weight.t()

    def _apply_topk(self, z, topk: int | None = None):
        if topk is None or topk <= 0 or topk >= z.size(1):
            return z
        values, indices = z.topk(k=topk, dim=1)
        masked = torch.zeros_like(z)
        masked.scatter_(1, indices, values)
        return masked

    def encode_pre(self, x):
        return self.encoder(x)

    def encode(self, x, topk: int | None = None):
        z = F.relu(self.encode_pre(x))
        return self._apply_topk(z, topk=topk)

    def decode(self, z):
        return F.linear(z, self.decoder_weight, self.decoder_bias)

    def forward(self, x, topk: int | None = None):
        z = self.encode(x, topk=topk)
        x_hat = self.decode(z)
        return z, x_hat

    def active_count(self, z, threshold=1e-3):
        return (z > threshold).float().sum(dim=1).mean()

    def dead_fraction(self, z, threshold=1e-3):
        return ((z > threshold).float().sum(dim=0) == 0).float().mean()


class VanillaSAE(BaseSAE):
    pass


class ArchetypalSAE(BaseSAE):
    def coherence_penalty(self):
        w = F.normalize(self.encoder.weight, dim=1)
        sim = w @ w.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        sim = sim - eye
        return F.relu(sim.abs() - 0.1).pow(2).mean()


class MPSAE(BaseSAE):
    def __init__(self, d_in: int, width: int = 1024, k: int = 16):
        super().__init__(d_in=d_in, width=width)
        self.k = k

    def encode(self, x, topk: int | None = None):
        # Greedy non-negative matching pursuit with tied decoder atoms.
        weight = F.normalize(self.encoder.weight, dim=1)
        residual = x
        code = torch.zeros(x.size(0), weight.size(0), device=x.device, dtype=x.dtype)
        steps = self.k if topk is None else topk
        for _ in range(min(steps, weight.size(0))):
            scores = residual @ weight.t()
            scores = torch.clamp(scores, min=0.0)
            chosen = scores.argmax(dim=1)
            coeff = scores.gather(1, chosen.unsqueeze(1)).squeeze(1)
            active = coeff > 1e-8
            if not active.any():
                break
            code[active, chosen[active]] = code[active, chosen[active]] + coeff[active]
            residual = residual - coeff.unsqueeze(1) * weight[chosen]
        return code


class PairSAE(BaseSAE):
    def __init__(self, d_in: int, width: int = 1024, num_factors: int = 6):
        super().__init__(d_in=d_in, width=width)
        self.register_buffer("running_mu", torch.zeros(num_factors, width))
        self.register_buffer("running_count", torch.zeros(num_factors))

    def update_centroids(self, factor_ids, signed_delta, momentum=0.05):
        for factor_id in factor_ids.unique():
            mask = factor_ids == factor_id
            if mask.any():
                mean_delta = signed_delta[mask].mean(dim=0)
                self.running_mu[factor_id] = (1 - momentum) * self.running_mu[factor_id] + momentum * mean_delta.detach()
                self.running_count[factor_id] += mask.sum()

    def pair_losses(self, z_a, z_b, factor_ids):
        d = z_b - z_a
        delta = d.abs()
        p = delta / (delta.sum(dim=1, keepdim=True) + 1e-8)
        m = delta.size(1)
        concentration = (m * (p.pow(2).sum(dim=1)) - 1) / max(m - 1, 1)
        l_conc = (1 - concentration).mean()
        mu = self.running_mu[factor_ids]
        cos = F.cosine_similarity(d, mu, dim=1, eps=1e-8)
        l_align = (1 - cos).mean()
        mus = F.normalize(self.running_mu + 1e-8, dim=1)
        sim = mus @ mus.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=sim.dtype)
        l_sep = F.relu(sim - eye).pow(2).sum() / max((sim.numel() - sim.size(0)), 1)
        return {
            "signed_delta": d,
            "concentration": concentration.mean(),
            "l_conc": l_conc,
            "l_align": l_align,
            "l_sep": l_sep,
        }
