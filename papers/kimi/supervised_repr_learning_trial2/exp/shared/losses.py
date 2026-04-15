"""
Loss functions for contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss (Khosla et al., 2020)."""
    
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels, weights=None, return_per_sample=False):
        """
        Args:
            features: normalized feature vectors (batch_size, n_views, feat_dim)
            labels: ground truth labels (batch_size,)
            weights: sample weights (batch_size,) - optional
            return_per_sample: whether to return per-sample losses
        """
        device = features.device
        batch_size = features.shape[0]
        
        if len(features.shape) < 3:
            # Single view case
            features = features.unsqueeze(1)
        
        contrast_count = features.shape[1]
        # Concatenate all views: (batch_size * n_views, feat_dim)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # Expand labels for all views
        # labels: (batch_size,) -> (batch_size * n_views,)
        labels_expanded = labels.repeat(contrast_count)
        
        # Compute logits (anchor @ all_features.T / temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask for positive samples (same label)
        labels_expanded = labels_expanded.contiguous().view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        
        # Mask out self-contrasts
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        
        # For each anchor, compute exp(logits) for all samples
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        mask = mask * logits_mask  # Remove self from positives
        
        # Handle case where some samples have no positives
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum < 1e-8, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        
        # Reshape loss to (contrast_count, batch_size) and average over views
        loss = loss.view(contrast_count, batch_size).mean(dim=0)
        
        # Apply sample weights if provided
        if weights is not None:
            loss = loss * weights
        
        if return_per_sample:
            return loss
        
        return loss.mean()


class GCSCLoss(nn.Module):
    """
    Gradient-Confusion Aware Supervised Contrastive Learning Loss.
    
    This loss combines:
    1. Base SupCon loss
    2. Per-sample gradient alignment computation
    3. Learning velocity tracking
    4. Adaptive sample weighting based on utility scores
    """
    
    def __init__(self, temperature=0.5, gamma=2.0, velocity_temp=0.1, 
                 velocity_momentum=0.9, curriculum_epochs=500):
        super(GCSCLoss, self).__init__()
        self.temperature = temperature
        self.gamma = gamma
        self.velocity_temp = velocity_temp
        self.velocity_momentum = velocity_momentum
        self.curriculum_epochs = curriculum_epochs
        
        # For tracking per-sample statistics
        self.sample_loss_ema = {}  # sample_idx -> EMA of loss
        self.epoch = 0
        
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def compute_gradient_alignment(self, features, labels, encoder, return_features=False):
        """
        Compute per-sample gradient alignment with batch gradient.
        
        Args:
            features: encoder features (batch_size, feat_dim)
            labels: labels (batch_size,)
            encoder: encoder network for computing gradients
            
        Returns:
            alignment_scores: (batch_size,) - cosine similarity with batch gradient
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Normalize features for contrastive loss
        features_norm = F.normalize(features, dim=1)
        
        # Compute pairwise similarity matrix
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # Create label mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0)  # Remove self
        
        # Per-sample gradient approximation
        # For each sample i: look at its similarity to positives vs negatives
        pos_sim = (similarity_matrix * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        neg_sim = (similarity_matrix * (1 - mask)).sum(dim=1) / ((1 - mask).sum(dim=1) + 1e-8)
        
        # Gradient alignment approximation: samples with high pos_sim and low neg_sim
        # align well with the learning direction
        alignment = torch.sigmoid((pos_sim - neg_sim) / self.temperature)
        
        if return_features:
            return alignment, features
        return alignment
    
    def compute_utility_scores(self, per_sample_loss, indices, alignment_scores):
        """
        Compute utility scores combining gradient alignment and learning velocity.
        
        Args:
            per_sample_loss: (batch_size,) per-sample losses
            indices: (batch_size,) sample indices in dataset
            alignment_scores: (batch_size,) gradient alignment scores
            
        Returns:
            utility_scores: (batch_size,) utility scores for weighting
        """
        batch_size = per_sample_loss.shape[0]
        device = per_sample_loss.device
        
        # Update EMA of loss for each sample
        velocities = []
        for i, idx in enumerate(indices.cpu().numpy()):
            idx = int(idx)
            current_loss = per_sample_loss[i].item()
            
            if idx in self.sample_loss_ema:
                prev_ema = self.sample_loss_ema[idx]
                # Compute velocity (improvement rate)
                velocity = prev_ema - current_loss
                # Update EMA
                self.sample_loss_ema[idx] = self.velocity_momentum * prev_ema + \
                                           (1 - self.velocity_momentum) * current_loss
            else:
                # First time seeing this sample
                velocity = 0.0
                self.sample_loss_ema[idx] = current_loss
            
            velocities.append(velocity)
        
        velocities = torch.tensor(velocities, device=device)
        
        # Compute utility score: u_i = (1 + alpha_i)^gamma * sigmoid(v_i / T)
        gradient_term = torch.pow(1 + alignment_scores, self.gamma)
        velocity_term = torch.sigmoid(velocities / self.velocity_temp)
        
        utility_scores = gradient_term * velocity_term
        
        return utility_scores
    
    def compute_weights(self, utility_scores, use_curriculum=True):
        """
        Compute final sample weights from utility scores.
        
        Args:
            utility_scores: (batch_size,) utility scores
            use_curriculum: whether to use curriculum scheduling
            
        Returns:
            weights: (batch_size,) normalized weights
        """
        if use_curriculum and self.curriculum_epochs > 0:
            # Curriculum: lambda increases from 0 to 1 over curriculum_epochs
            lambda_param = min(1.0, self.epoch / self.curriculum_epochs)
        else:
            lambda_param = 1.0
        
        # Apply temperature to utility scores
        weights = torch.pow(utility_scores + 1e-8, lambda_param)
        
        # Normalize to have mean 1
        weights = weights / (weights.mean() + 1e-8)
        
        return weights
    
    def forward(self, features, labels, indices, encoder, return_stats=False):
        """
        Compute GC-SCL loss with adaptive sample weighting.
        
        Args:
            features: (batch_size, n_views, feat_dim) normalized features
            labels: (batch_size,) ground truth labels
            indices: (batch_size,) sample indices
            encoder: encoder network for gradient computation
            return_stats: whether to return statistics for analysis
            
        Returns:
            loss: scalar loss value
            stats: dict of statistics (if return_stats=True)
        """
        # First compute base SupCon loss per sample
        supcon = SupConLoss(temperature=self.temperature)
        per_sample_loss = supcon(features, labels, weights=None, return_per_sample=True)
        
        # Get features for gradient alignment (use first view)
        if len(features.shape) == 3:
            feat_for_grad = features[:, 0, :]
        else:
            feat_for_grad = features
        
        # Compute gradient alignment scores
        alignment_scores = self.compute_gradient_alignment(feat_for_grad, labels, encoder)
        
        # Compute utility scores
        utility_scores = self.compute_utility_scores(per_sample_loss, indices, alignment_scores)
        
        # Compute final weights
        weights = self.compute_weights(utility_scores, use_curriculum=True)
        
        # Apply weights to loss
        weighted_loss = (per_sample_loss * weights).mean()
        
        if return_stats:
            stats = {
                'alignment_mean': alignment_scores.mean().item(),
                'alignment_std': alignment_scores.std().item(),
                'utility_mean': utility_scores.mean().item(),
                'utility_std': utility_scores.std().item(),
                'weights_mean': weights.mean().item(),
                'weights_std': weights.std().item(),
                'weights_min': weights.min().item(),
                'weights_max': weights.max().item(),
            }
            return weighted_loss, stats
        
        return weighted_loss
