"""Loss functions for contrastive learning experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [2N, D] L2-normalized features (two views concatenated)
            labels: [2N] labels (repeated for two views)
        """
        device = features.device
        batch_size = features.shape[0]

        # Compute similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # [2N, 2N]

        # Mask: same class (excluding self)
        labels = labels.unsqueeze(1)
        mask_pos = (labels == labels.T).float()  # [2N, 2N]
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self  # exclude self

        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Log-sum-exp of all negatives + positives (excluding self)
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean of log-prob over positives
        pos_count = mask_pos.sum(dim=1, keepdim=True).clamp(min=1)
        log_prob = (logits - log_sum_exp) * mask_pos
        loss = -(log_prob.sum(dim=1) / pos_count.squeeze(1)).mean()

        return loss


class HardNegLoss(nn.Module):
    """Hard Negative Contrastive Loss (Robinson et al., ICLR 2021)."""

    def __init__(self, temperature=0.07, beta=1.0):
        super().__init__()
        self.temperature = temperature
        self.beta = beta

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Raw similarities for importance weights (not temperature-scaled)
        sim_raw = torch.matmul(features, features.T)  # [2N, 2N]
        # Temperature-scaled logits for the contrastive loss
        sim = sim_raw / self.temperature

        labels_col = labels.unsqueeze(1)
        mask_pos = (labels_col == labels_col.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self
        mask_neg = 1 - (labels_col == labels_col.T).float()

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits)

        # Importance weights on RAW similarities (not temperature-scaled)
        # This avoids effective concentration of beta/tau which is too sharp
        raw_max, _ = sim_raw.max(dim=1, keepdim=True)
        raw_shifted = sim_raw - raw_max.detach()
        neg_weights = torch.exp(self.beta * raw_shifted) * mask_neg
        neg_weights = neg_weights / (neg_weights.sum(dim=1, keepdim=True) + 1e-12) * mask_neg.sum(dim=1, keepdim=True)

        weighted_neg = exp_logits * neg_weights
        pos_exp = exp_logits * mask_pos

        denominator = weighted_neg.sum(dim=1, keepdim=True) + pos_exp.sum(dim=1, keepdim=True)
        log_prob = (logits - torch.log(denominator + 1e-12)) * mask_pos
        pos_count = mask_pos.sum(dim=1).clamp(min=1)
        loss = -(log_prob.sum(dim=1) / pos_count).mean()

        return loss


class TCLLoss(nn.Module):
    """Tuned Contrastive Learning (inspired by Animesh & Chandraker, WACV 2025).

    Uses learnable temperature multipliers for positives and negatives.
    Key fix: compute a unified logit tensor so the stability constant M
    is consistent across numerator and denominator, forming a valid log-softmax.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        # Learnable log-scale multipliers (exp(0)=1.0 → starts at base temperature)
        self.log_tau_pos = nn.Parameter(torch.tensor(0.0))
        self.log_tau_neg = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        sim = torch.matmul(features, features.T)  # raw cosine similarities
        labels_col = labels.unsqueeze(1)
        mask_pos = (labels_col == labels_col.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self
        mask_neg = 1.0 - (labels_col == labels_col.T).float()

        # Learnable temperatures (clamped for stability)
        tau_pos = self.temperature * torch.exp(self.log_tau_pos).clamp(0.5, 2.0)
        tau_neg = self.temperature * torch.exp(self.log_tau_neg).clamp(0.5, 2.0)

        # Build unified logit tensor: positives scaled by tau_pos, negatives by tau_neg
        logits = (sim / tau_pos) * (mask_pos + mask_self) + (sim / tau_neg) * mask_neg
        # Mask out self-similarity
        logits = logits - mask_self * 1e9

        # Numerical stability with shared M
        M = logits.max(dim=1, keepdim=True)[0].detach()
        logits_stable = logits - M

        exp_all = torch.exp(logits_stable) * (1 - mask_self)
        log_denom = torch.log(exp_all.sum(dim=1, keepdim=True) + 1e-12)

        # Log-prob for positive pairs (using the positive-scaled logits from logits_stable)
        log_prob = (logits_stable - log_denom) * mask_pos
        pos_count = mask_pos.sum(dim=1).clamp(min=1)
        loss = -(log_prob.sum(dim=1) / pos_count).mean()

        return loss


class ReweightedSupConLoss(nn.Module):
    """SupCon with confusion-based pairwise reweighting (MACL-style)."""

    def __init__(self, temperature=0.07, beta_rw=2.0):
        super().__init__()
        self.temperature = temperature
        self.beta_rw = beta_rw

    def forward(self, features, labels, confusion_matrix=None):
        device = features.device
        batch_size = features.shape[0]

        sim = torch.matmul(features, features.T) / self.temperature
        labels_col = labels.unsqueeze(1)
        mask_pos = (labels_col == labels_col.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self
        mask_neg = 1 - (labels_col == labels_col.T).float()

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits)

        # Reweight negatives by confusion
        if confusion_matrix is not None:
            # Get pairwise confusion weights for this batch
            lab = labels.long()
            conf_weights = confusion_matrix[lab][:, lab]  # [2N, 2N]
            neg_weights = (1 + self.beta_rw * conf_weights) * mask_neg
        else:
            neg_weights = mask_neg

        weighted_exp = exp_logits * neg_weights + exp_logits * mask_pos
        log_sum_exp = torch.log(weighted_exp.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask_pos.sum(dim=1).clamp(min=1)
        log_prob = (logits - log_sum_exp) * mask_pos
        loss = -(log_prob.sum(dim=1) / pos_count).mean()

        return loss


class VarConTLoss(nn.Module):
    """SupCon with per-sample adaptive temperature (VarCon-T style)."""

    def __init__(self, temperature=0.07, gamma_v=2.0):
        super().__init__()
        self.temperature = temperature
        self.gamma_v = gamma_v

    def forward(self, features, labels, confidences=None):
        device = features.device
        batch_size = features.shape[0]

        # Per-sample temperature
        if confidences is not None:
            # tau(x) = tau_base / (1 + gamma * (1 - max_c p(c|x)))
            uncertainty = 1 - confidences  # [2N]
            tau = self.temperature / (1 + self.gamma_v * uncertainty)
            tau = tau.unsqueeze(1)  # [2N, 1]
        else:
            tau = self.temperature

        sim = torch.matmul(features, features.T) / tau  # [2N, 2N]
        labels_col = labels.unsqueeze(1)
        mask_pos = (labels_col == labels_col.T).float()
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = mask_pos.sum(dim=1).clamp(min=1)
        log_prob = (logits - log_sum_exp) * mask_pos
        loss = -(log_prob.sum(dim=1) / pos_count).mean()

        return loss


class CGALoss(nn.Module):
    """Confusion-Geometric Alignment loss (sample-to-prototype formulation).

    Instead of computing CGA on noisy batch prototypes, this formulation:
    1. Uses stable EMA prototypes (detached) as reference points
    2. Pushes each sample's cross-class similarities toward CGA targets
    3. Gradients flow through the sample embeddings (not prototypes)

    This avoids the noisy batch-prototype problem (only ~5 samples/class
    with batch_size=512 and 100 classes) while maintaining gradient signal.
    """

    def __init__(self, num_classes, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    def forward(self, z, labels, ema_prototypes, confusion_norm):
        """
        Args:
            z: [N, D] L2-normalized sample embeddings (requires grad)
            labels: [N] class labels
            ema_prototypes: [C, D] L2-normalized EMA prototypes (detached)
            confusion_norm: [C, C] symmetric normalized confusion, values in [0, 1]
        """
        C = self.num_classes
        baseline = -1.0 / (C - 1)

        # Target cross-class similarities: more negative for confused pairs
        # s*(i,j) = baseline - alpha * c_norm(i,j)
        target_sim = baseline - self.alpha * confusion_norm
        target_sim = target_sim.clamp(min=-0.95, max=0.95)

        # Zero diagonal (same-class target handled separately)
        diag_mask = 1 - torch.eye(C, device=z.device)
        target_sim = target_sim * diag_mask

        # Compute sample-to-prototype similarities: [N, C]
        sim = torch.matmul(z, ema_prototypes.T)

        # For each sample, get its class label and the target similarities
        # to all OTHER class prototypes
        lab = labels.long()
        per_sample_targets = target_sim[lab]  # [N, C]

        # Mask: only penalize cross-class similarities (not same-class)
        cross_mask = torch.ones(z.shape[0], C, device=z.device)
        cross_mask[torch.arange(z.shape[0], device=z.device), lab] = 0

        # Weighted MSE: emphasize confused pairs
        conf_weights = confusion_norm[lab]  # [N, C]
        weights = (1.0 + conf_weights) * cross_mask

        diff = sim - per_sample_targets
        loss = (weights * diff ** 2).sum() / (weights.sum() + 1e-8)

        return loss
