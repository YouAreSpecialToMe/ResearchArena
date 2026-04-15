from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class ForwardBundle:
    logits: torch.Tensor
    aux_logits: dict[int, torch.Tensor]
    hidden_states: tuple[torch.Tensor, ...]


class LateBindModel(nn.Module):
    def __init__(self, model_name: str = "roberta-base", aux_layers: tuple[int, int] = (4, 8), num_labels: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        dropout = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.aux_layers = tuple(aux_layers)
        self.aux_heads = nn.ModuleDict({str(layer): nn.Linear(hidden, num_labels) for layer in aux_layers})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ForwardBundle:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        pooled = self.dropout(hidden_states[-1][:, 0, :])
        logits = self.classifier(pooled)
        aux_logits: dict[int, torch.Tensor] = {}
        for layer in self.aux_layers:
            aux_hidden = self.dropout(hidden_states[layer][:, 0, :])
            aux_logits[layer] = self.aux_heads[str(layer)](aux_hidden)
        return ForwardBundle(logits=logits, aux_logits=aux_logits, hidden_states=hidden_states)


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1).clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)


def latebind_loss(
    bundle: ForwardBundle,
    labels: torch.Tensor,
    risk: torch.Tensor,
    masked_bundle: ForwardBundle | None,
    lambda_late: float,
    lambda_inv: float,
    entropy_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(bundle.logits, labels)
    late = torch.zeros_like(ce)
    if lambda_late > 0.0:
        penalties = []
        for logits in bundle.aux_logits.values():
            penalties.append(torch.relu(entropy_floor - predictive_entropy(logits)))
        late = risk * torch.stack(penalties, dim=0).sum(dim=0)
        late = late.mean()
    inv = torch.zeros_like(ce)
    if lambda_inv > 0.0 and masked_bundle is not None:
        kl = F.kl_div(
            F.log_softmax(masked_bundle.logits, dim=-1),
            F.softmax(bundle.logits.detach(), dim=-1),
            reduction="none",
        ).sum(dim=-1)
        inv = (risk * kl).mean()
    total = ce + lambda_late * late + lambda_inv * inv
    return total, {
        "loss_ce": float(ce.detach().item()),
        "loss_late": float(late.detach().item()),
        "loss_inv": float(inv.detach().item()),
        "loss_total": float(total.detach().item()),
    }


def load_tokenizer(model_name: str = "roberta-base"):
    return AutoTokenizer.from_pretrained(model_name)

