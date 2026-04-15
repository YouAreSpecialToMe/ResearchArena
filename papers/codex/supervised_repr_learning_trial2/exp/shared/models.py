from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim) * 0.02)

    def forward(self, z):
        return z @ F.normalize(self.weight, dim=-1).t()


class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@dataclass
class PrototypeDiagnostics:
    active_counts: dict
    occupancy_entropy: dict
    occupancies: dict


class PrototypeHead(nn.Module):
    def __init__(self, num_classes, dim, max_k):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.max_k = max_k
        self.prototypes = nn.Parameter(torch.randn(num_classes, max_k, dim) * 0.02)
        self.active_k = {c: 1 for c in range(num_classes)}

    def set_active_k(self, mapping):
        self.active_k = {int(k): int(v) for k, v in mapping.items()}

    def active_mask(self, device):
        mask = torch.zeros(self.num_classes, self.max_k, dtype=torch.bool, device=device)
        for c, k in self.active_k.items():
            mask[c, :k] = True
        return mask

    def normalized_prototypes(self):
        return F.normalize(self.prototypes, dim=-1)

    def class_logits(self, z):
        protos = self.normalized_prototypes()
        sims = torch.einsum("bd,ckd->bck", z, protos)
        mask = self.active_mask(z.device)
        sims = sims.masked_fill(~mask.unsqueeze(0), -1e9)
        class_logits = torch.logsumexp(sims / 0.07, dim=-1)
        return class_logits, sims

    def true_class_assignments(self, z, y, tau=0.07):
        protos = self.normalized_prototypes()
        true_sims = (z.unsqueeze(1) * protos[y]).sum(-1)
        mask_rows = []
        for cls in y.tolist():
            row = torch.zeros(self.max_k, dtype=torch.bool, device=z.device)
            row[: self.active_k[int(cls)]] = True
            mask_rows.append(row)
        mask = torch.stack(mask_rows, dim=0)
        true_sims = true_sims.masked_fill(~mask, -1e9)
        q = torch.softmax(true_sims / tau, dim=-1)
        return q, true_sims

    def predict_subclasses(self, z, y):
        _, sims = self.class_logits(z)
        assignments = []
        for idx, cls in enumerate(y.tolist()):
            assignments.append(int(torch.argmax(sims[idx, cls, : self.active_k[int(cls)]], dim=-1)))
        return assignments
