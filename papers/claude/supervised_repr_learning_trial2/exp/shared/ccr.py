"""Calibrated Collapse Regularizer (CCR) — fully vectorized, no Python loops on GPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCR(nn.Module):
    """Calibrated Collapse Regularizer.

    Prevents excessive neural collapse (NC1) by maintaining minimum within-class spread.
    Variants: fixed, adaptive, soft, spectral.
    """

    def __init__(self, num_classes, feat_dim, variant='adaptive',
                 tau=1.0, gamma=0.1, momentum=0.999):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.variant = variant
        self.tau = tau
        self.gamma = gamma
        self.momentum = momentum

        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('prototype_counts', torch.zeros(num_classes))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer('_adaptive_tau_cache', torch.tensor(tau))

    @torch.no_grad()
    def update_prototypes(self, features, labels):
        """Update running class prototypes with EMA — fully vectorized, no Python loops."""
        batch_sums = torch.zeros(self.num_classes, self.feat_dim, device=features.device)
        batch_counts = torch.zeros(self.num_classes, device=features.device)
        batch_sums.scatter_add_(0, labels.unsqueeze(1).expand_as(features), features)
        batch_counts.scatter_add_(0, labels, torch.ones(labels.size(0), device=labels.device))

        active = batch_counts > 0

        # Compute batch means only for active classes
        batch_means = torch.zeros_like(self.prototypes)
        active_counts = batch_counts[active].unsqueeze(1)
        batch_means[active] = batch_sums[active] / active_counts

        # Initialize new classes (first time seen)
        new_classes = active & ~self.initialized
        if new_classes.any():
            self.prototypes[new_classes] = batch_means[new_classes]
            self.initialized[new_classes] = True

        # EMA update for existing classes
        existing = active & self.initialized
        if existing.any():
            self.prototypes[existing] = (self.momentum * self.prototypes[existing]
                                         + (1 - self.momentum) * batch_means[existing])

        self.prototype_counts[active] += batch_counts[active]

    def compute_spreads_and_loss(self, features, labels, threshold):
        """Compute per-class spreads and loss in one pass — fully vectorized."""
        proto_for_sample = self.prototypes[labels]  # (N, D)
        dists_sq = ((features - proto_for_sample) ** 2).sum(dim=1)  # (N,)

        # Per-class mean spread using scatter
        class_sums = torch.zeros(self.num_classes, device=features.device)
        class_counts = torch.zeros(self.num_classes, device=features.device)
        class_sums.scatter_add_(0, labels, dists_sq)
        class_counts.scatter_add_(0, labels, torch.ones_like(dists_sq))

        valid = class_counts >= 2
        if not valid.any():
            return torch.tensor(0.0, device=features.device), 0.0, 0.0, 0.0

        spreads = class_sums[valid] / class_counts[valid]
        mean_spread = spreads.mean().item()

        if self.variant == 'soft':
            loss = F.softplus(threshold - spreads).mean()
        else:
            loss = torch.clamp(threshold - spreads, min=0).mean()

        return loss, mean_spread, spreads.min().item(), spreads.max().item()

    @torch.no_grad()
    def compute_adaptive_tau(self):
        """Compute adaptive threshold: tau = gamma * min inter-class distance^2."""
        if self.initialized.sum() < 2:
            return self.tau
        active = self.prototypes[self.initialized]
        # Efficient pairwise distance: only need min
        dists = torch.cdist(active.unsqueeze(0), active.unsqueeze(0)).squeeze(0)
        dists.fill_diagonal_(float('inf'))
        min_dist = dists.min()
        tau = self.gamma * (min_dist ** 2)
        self._adaptive_tau_cache = tau
        return tau

    def compute_spectral_loss(self, features, labels):
        """Spectral CCR: negative mean effective rank of within-class covariance."""
        classes_in_batch = labels.unique()
        eff_ranks = []
        for c in classes_in_batch[:15]:  # Limit for speed
            mask = labels == c
            if mask.sum() < 3:
                continue
            class_features = features[mask]
            centered = class_features - class_features.mean(dim=0, keepdim=True)
            try:
                s = torch.linalg.svdvals(centered)
                s = s + 1e-10
                p = s / s.sum()
                eff_rank = torch.exp(-torch.sum(p * torch.log(p)))
                eff_ranks.append(eff_rank)
            except RuntimeError:
                continue

        if not eff_ranks:
            return torch.tensor(0.0, device=features.device)
        return -torch.stack(eff_ranks).mean()

    def forward(self, features, labels):
        """Compute CCR loss."""
        self.update_prototypes(features.detach(), labels)

        metrics = {}

        if self.variant == 'spectral':
            spectral_loss = self.compute_spectral_loss(features, labels)
            _, mean_spread, _, _ = self.compute_spreads_and_loss(
                features, labels, torch.tensor(0.0, device=features.device))
            metrics['mean_spread'] = mean_spread
            metrics['spectral_loss'] = spectral_loss.item()
            return spectral_loss, metrics

        # Get threshold
        if self.variant in ('adaptive', 'soft'):
            threshold = self.compute_adaptive_tau()
        else:
            threshold = self.tau

        if torch.is_tensor(threshold):
            metrics['adaptive_tau'] = threshold.item()
        else:
            metrics['adaptive_tau'] = threshold

        loss, mean_spread, min_spread, max_spread = self.compute_spreads_and_loss(
            features, labels, threshold)

        metrics['mean_spread'] = mean_spread
        metrics['ccr_loss'] = loss.item()
        metrics['min_spread'] = min_spread
        metrics['max_spread'] = max_spread

        return loss, metrics
