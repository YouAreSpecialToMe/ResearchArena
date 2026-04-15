from typing import Optional

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    batch_size, n_views, dim = features.shape
    features = F.normalize(features, dim=-1)
    features = features.reshape(batch_size * n_views, dim)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(logits) - torch.eye(batch_size * n_views, device=features.device)
    mask = mask.repeat(n_views, n_views) * logits_mask
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return -mean_log_prob_pos.mean()


def pairwise_cosine(anchor: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,bkd->bk", F.normalize(anchor, dim=-1), F.normalize(neighbors, dim=-1))


def relational_mse_loss(
    anchor_features: torch.Tensor,
    neighbor_features: torch.Tensor,
    teacher_sims: torch.Tensor,
    tau_n: float,
) -> torch.Tensor:
    anchor_features = anchor_features.float()
    neighbor_features = neighbor_features.float()
    teacher_sims = teacher_sims.float()
    sims = pairwise_cosine(anchor_features, neighbor_features)
    target = teacher_sims
    return F.mse_loss(sims, target)


def nest_kl_loss(
    anchor_features: torch.Tensor,
    neighbor_features: torch.Tensor,
    teacher_probs: torch.Tensor,
    tau_n: float,
) -> torch.Tensor:
    anchor_features = anchor_features.float()
    neighbor_features = neighbor_features.float()
    teacher_probs = teacher_probs.float()
    teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sims = pairwise_cosine(anchor_features, neighbor_features) / tau_n
    log_q = F.log_softmax(sims, dim=-1)
    return F.kl_div(log_q, teacher_probs, reduction="batchmean")


def maskcon_loss(
    proj1: torch.Tensor,
    proj2: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    teacher_neighbors: Optional[torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    features = torch.cat([proj1, proj2], dim=0)
    labels_2 = torch.cat([labels, labels], dim=0)
    ids_2 = torch.cat([sample_ids, sample_ids], dim=0)
    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(logits) - torch.eye(logits.shape[0], device=logits.device)
    coarse_mask = (labels_2.unsqueeze(0) == labels_2.unsqueeze(1)).float()
    pos_mask = coarse_mask * logits_mask

    if teacher_neighbors is not None:
        teacher_neighbors_2 = torch.cat([teacher_neighbors, teacher_neighbors], dim=0)
        candidate_ids = ids_2.unsqueeze(0).expand(ids_2.shape[0], -1)
        neigh_mask = (teacher_neighbors_2.unsqueeze(1) == candidate_ids.unsqueeze(-1)).any(dim=-1).float()
        pos_mask = torch.where(neigh_mask > 0, pos_mask, pos_mask * 0.25)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1.0)
    return -mean_log_prob_pos.mean()
