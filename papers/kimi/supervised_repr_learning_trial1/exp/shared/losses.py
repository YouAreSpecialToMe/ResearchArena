"""
Loss functions for contrastive learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Standard Supervised Contrastive Loss (Khosla et al. 2020)."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: (2*batch_size, projection_dim) normalized embeddings
            labels: (2*batch_size,) labels
            
        Returns:
            loss: Scalar loss
        """
        batch_size = features.shape[0] // 2
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label, not same sample)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positives
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)  # Avoid division by zero
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


class FDSupervisedContrastiveLoss(nn.Module):
    """Feature-Diversity-Aware Supervised Contrastive Loss (FD-SCL)."""
    
    def __init__(self, temperature=0.1, weight_temperature=0.5, 
                 activation_threshold=0.0, epsilon=1e-6):
        super().__init__()
        self.temperature = temperature
        self.weight_temperature = weight_temperature
        self.activation_threshold = activation_threshold
        self.epsilon = epsilon
        
    def compute_diversity_weights(self, features, labels):
        """
        Compute diversity-aware weights for positive pairs.
        
        Args:
            features: (2*batch_size, projection_dim) normalized embeddings
            labels: (2*batch_size,) labels
            
        Returns:
            weights: (2*batch_size, 2*batch_size) weight matrix for positive pairs
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. Compute feature activation frequency across batch
        # features are normalized, so we use the threshold directly
        activations = (features > self.activation_threshold).float()  # (batch, dim)
        freq = activations.mean(dim=0, keepdim=True)  # (1, dim)
        
        # Clamp frequency to avoid extreme rarity weights
        freq = torch.clamp(freq, min=0.01, max=0.99)
        
        # 2. Create mask for positive pairs (same label, not same sample)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.ones_like(pos_mask)
        logits_mask.fill_diagonal_(0)
        pos_mask = pos_mask * logits_mask
        
        # 3. Compute pairwise rarity scores
        # For each pair (i, p), r_ip = sum_d [indicator(z_id > eta) * indicator(z_pd > eta)] / (freq_d + epsilon)
        # Use log-space computation for numerical stability
        rarity_per_dim = 1.0 / (freq + self.epsilon)  # (1, dim)
        
        # Clamp rarity to prevent extreme values
        rarity_per_dim = torch.clamp(rarity_per_dim, max=100.0)
        
        weighted_activations = activations * rarity_per_dim  # (batch, dim)
        rarity_scores = torch.matmul(activations, weighted_activations.T)  # (batch, batch)
        
        # Clamp rarity scores to prevent overflow in exp
        rarity_scores = torch.clamp(rarity_scores, min=-50.0, max=50.0)
        
        # 4. Convert rarity scores to weights via softmax
        # For each anchor i, compute softmax over positive pairs
        # w_ip = softmax(r_ip / tau_w) * |P(i)|
        
        # Apply softmax only over positive pairs
        # We need to mask out non-positive pairs with large negative before softmax
        rarity_exp = torch.exp(rarity_scores / self.weight_temperature)
        rarity_exp = rarity_exp * pos_mask  # Zero out non-positive pairs
        
        # Compute softmax: exp(r_ip) / sum_{p in P(i)} exp(r_ip)
        rarity_sum = rarity_exp.sum(dim=1, keepdim=True) + self.epsilon
        weights = rarity_exp / rarity_sum
        
        # Scale by number of positives to preserve loss scale
        num_positives = pos_mask.sum(dim=1, keepdim=True)
        num_positives = torch.clamp(num_positives, min=1.0)
        weights = weights * num_positives
        
        # Store rarity scores for analysis (keep on same device during forward)
        self.last_rarity_scores = rarity_scores.detach()
        self.last_weights = weights.detach()
        
        return weights, pos_mask
        
    def forward(self, features, labels, return_stats=False):
        """
        Args:
            features: (2*batch_size, projection_dim) normalized embeddings
            labels: (2*batch_size,) labels
            return_stats: If True, return additional statistics
            
        Returns:
            loss: Scalar loss
            stats: Dict of statistics (if return_stats=True)
        """
        batch_size = features.shape[0]
        
        # Compute diversity-aware weights
        weights, pos_mask = self.compute_diversity_weights(features, labels)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Compute log_prob
        logits_mask = torch.ones_like(pos_mask)
        logits_mask.fill_diagonal_(0)
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute weighted mean of log-likelihood over positives
        mask_sum = pos_mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        # Weighted loss
        weighted_log_prob = weights * log_prob * pos_mask
        mean_log_prob_pos = weighted_log_prob.sum(1) / mask_sum
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        if return_stats:
            pos_mask_bool = pos_mask > 0
            stats = {
                'mean_weight': weights[pos_mask_bool].mean().item(),
                'std_weight': weights[pos_mask_bool].std().item(),
                'mean_rarity': self.last_rarity_scores[pos_mask_bool].mean().item(),
                'weight_entropy': -((weights[pos_mask_bool] / weights[pos_mask_bool].sum()) * 
                                   torch.log(weights[pos_mask_bool] / weights[pos_mask_bool].sum() + 1e-8)).sum().item()
            }
            return loss, stats
        
        return loss


class FDSCLUniformAblation(nn.Module):
    """Ablation: FD-SCL with uniform weights (equivalent to standard SCL)."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """Same as standard SCL."""
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        
        return loss


class FDSCLRandomAblation(nn.Module):
    """Ablation: FD-SCL with random weights."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """SCL with random weights."""
        batch_size = features.shape[0]
        device = features.device
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Random weights for positive pairs
        weights = torch.rand(batch_size, batch_size, device=device)
        weights = weights * mask
        
        # Normalize weights per anchor
        weight_sum = weights.sum(1, keepdim=True) + 1e-8
        weights = weights / weight_sum
        num_positives = mask.sum(dim=1, keepdim=True)
        weights = weights * num_positives
        
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        
        weighted_log_prob = weights * log_prob * mask
        mean_log_prob_pos = weighted_log_prob.sum(1) / mask_sum
        loss = -mean_log_prob_pos.mean()
        
        return loss
