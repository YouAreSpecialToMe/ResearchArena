"""Optimized contrastive loss implementations.

Key optimization: vectorized centroid update and weight computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = labels.shape[0]
        labels_2n = labels.repeat(2).view(-1, 1)  # [2N, 1]
        mask = torch.eq(labels_2n, labels_2n.T).float()  # [2N, 2N]

        n = 2 * batch_size
        sim = torch.matmul(features, features.T) / self.temperature
        logits_mask = 1.0 - torch.eye(n, device=device)
        mask = mask * logits_mask

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = mask.sum(1).clamp(min=1)
        mean_log_prob = (mask * log_prob).sum(1) / pos_count
        return -mean_log_prob.mean()


class HSCLLoss(nn.Module):
    """Hard-negative Supervised Contrastive Loss (Jiang et al., 2022)."""
    def __init__(self, temperature=0.1, beta_hard=1.0):
        super().__init__()
        self.temperature = temperature
        self.beta_hard = beta_hard

    def forward(self, features, labels):
        device = features.device
        batch_size = labels.shape[0]
        labels_2n = labels.repeat(2).view(-1, 1)
        mask = torch.eq(labels_2n, labels_2n.T).float()
        n = 2 * batch_size

        sim = torch.matmul(features, features.T) / self.temperature
        logits_mask = 1.0 - torch.eye(n, device=device)
        pos_mask = mask * logits_mask
        neg_mask = (1.0 - mask) * logits_mask

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits)
        neg_weights = (exp_logits ** self.beta_hard) * neg_mask
        weighted_exp = exp_logits * pos_mask + neg_weights
        log_prob = logits - torch.log(weighted_exp.sum(1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(1).clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(1) / pos_count
        return -mean_log_prob.mean()


class CCSupConLoss(nn.Module):
    """Optimized Confusion-Calibrated Supervised Contrastive Loss.

    Key optimizations over original:
    - Vectorized centroid update using scatter_add (no Python for-loop)
    - Pre-computed weight matrix indexed by labels
    """
    def __init__(self, num_classes=100, feat_dim=128, temperature=0.1,
                 tau_c=0.1, alpha=0.99, beta=1.0, warmup_epochs=0):
        super().__init__()
        self.temperature = temperature
        self.tau_c = tau_c
        self.alpha = alpha
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.register_buffer('centroids', torch.zeros(num_classes, feat_dim))
        self.register_buffer('centroid_counts', torch.zeros(num_classes))
        self.register_buffer('confusability', torch.zeros(num_classes, num_classes))
        self.register_buffer('all_initialized', torch.tensor(False))

    @torch.no_grad()
    def update_centroids(self, features, labels):
        """Vectorized centroid update using scatter operations."""
        features = features.detach().float()  # Force float32
        C, D = self.num_classes, self.feat_dim

        # Count samples per class and sum features per class
        # Use one_hot for efficient aggregation
        one_hot = torch.zeros(features.shape[0], C, device=features.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)  # [N, C]

        counts = one_hot.sum(0)  # [C]
        present = counts > 0

        # Class means: [C, D] = [C, N] @ [N, D]
        class_sums = one_hot.T @ features  # [C, D]

        # Compute means only for present classes
        if present.any():
            means = class_sums[present] / counts[present].unsqueeze(1)
            means = F.normalize(means, dim=1)

            # First-time init vs EMA update
            new_init = present & (self.centroid_counts == 0)
            existing = present & (self.centroid_counts > 0)

            if new_init.any():
                # Extract means for newly initialized classes
                # Map from present-mask indices to new_init indices
                present_idx = torch.where(present)[0]
                new_init_in_present = new_init[present]
                self.centroids[present_idx[new_init_in_present]] = means[new_init_in_present]

            if existing.any():
                present_idx = torch.where(present)[0]
                existing_in_present = existing[present]
                old = self.centroids[present_idx[existing_in_present]]
                new = means[existing_in_present]
                updated = F.normalize(self.alpha * old + (1 - self.alpha) * new, dim=1)
                self.centroids[present_idx[existing_in_present]] = updated

            self.centroid_counts[present] += counts[present]

            # Check if all classes have been seen
            if not self.all_initialized and (self.centroid_counts > 0).all():
                self.all_initialized.fill_(True)

    @torch.no_grad()
    def compute_confusability(self):
        """Compute symmetrized confusability matrix from centroids."""
        if not self.all_initialized:
            return

        # Force float32 to avoid overflow with mixed precision
        with torch.amp.autocast('cuda', enabled=False):
            centroids = self.centroids.float()
            sim = torch.matmul(centroids, centroids.T) / self.tau_c
            sim.fill_diagonal_(-1e9)
            conf = F.softmax(sim, dim=1)
            conf.fill_diagonal_(0)
            self.confusability = 0.5 * (conf + conf.T)

    def forward(self, features, labels, current_epoch=0):
        device = features.device
        batch_size = labels.shape[0]
        labels_2n = labels.repeat(2)  # [2N]
        n = 2 * batch_size

        # Update centroids and confusability
        self.update_centroids(features, labels_2n)
        self.compute_confusability()

        # Same-class mask
        mask = torch.eq(labels_2n.unsqueeze(1), labels_2n.unsqueeze(0)).float()
        logits_mask = 1.0 - torch.eye(n, device=device)
        pos_mask = mask * logits_mask

        # Similarity
        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Compute confusability weights
        effective_beta = self.beta if current_epoch >= self.warmup_epochs else 0.0

        if effective_beta > 0.0 and self.all_initialized:
            # Index confusability by label pairs: [2N, 2N]
            conf_weights = self.confusability[labels_2n][:, labels_2n]
            neg_mask = (1.0 - mask) * logits_mask
            weights = 1.0 + effective_beta * conf_weights * neg_mask
            # For positive pairs and self: weight = 1 (already 1 since neg_mask is 0)
            weights = weights + pos_mask + torch.eye(n, device=device)  # ensure pos/self = 1
            # Actually simpler: weights already has 1 everywhere from the "1.0 +" part
            # except diagonal. Let me redo this cleanly:
            weights = logits_mask * (mask + (1.0 - mask) * (1.0 + effective_beta * conf_weights))
        else:
            weights = logits_mask

        exp_logits = torch.exp(logits) * weights
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(1).clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(1) / pos_count
        return -mean_log_prob.mean()

    def get_confusability_matrix(self):
        return self.confusability.cpu().numpy()
