"""Loss functions: SupCon loss and Curriculum loss with consistency regularization."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2-normalized embeddings
            labels: [B] integer labels
        Returns:
            scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Mask: same label = positive pair
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]

        # Remove self-similarity from denominator and positives
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask  # positives excluding self

        # For numerical stability
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        # Log-sum-exp over all negatives + positives (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean of log-probability over positive pairs
        # Only consider anchors that have at least one positive
        pos_per_anchor = mask.sum(dim=1)
        valid = pos_per_anchor > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (pos_per_anchor + 1e-12)
        loss = -mean_log_prob[valid].mean()
        return loss


class CurriculumLoss(nn.Module):
    """Curriculum loss: primary SupCon + consistency regularization from coarser levels."""
    def __init__(self, temperature=0.1, consistency_weight=0.5):
        super().__init__()
        self.primary_loss = SupConLoss(temperature=temperature)
        self.consistency_weight = consistency_weight

    def forward(self, features, primary_labels, consistency_labels_list=None):
        """
        Args:
            features: [B, D] L2-normalized embeddings
            primary_labels: [B] labels at current EC level
            consistency_labels_list: list of [B] labels at coarser EC levels (for regularization)
        Returns:
            total_loss, primary_loss_val, consistency_loss_val
        """
        primary_loss = self.primary_loss(features, primary_labels)

        if consistency_labels_list is None or len(consistency_labels_list) == 0 or self.consistency_weight == 0:
            return primary_loss, primary_loss.item(), 0.0

        consistency_loss = torch.tensor(0.0, device=features.device, requires_grad=True)
        for coarse_labels in consistency_labels_list:
            consistency_loss = consistency_loss + self.primary_loss(features, coarse_labels)

        # Normalize by number of consistency terms to prevent overwhelming primary loss
        n_terms = len(consistency_labels_list)
        consistency_loss_normalized = consistency_loss / n_terms

        total_loss = primary_loss + self.consistency_weight * consistency_loss_normalized
        return total_loss, primary_loss.item(), consistency_loss.item() / n_terms
