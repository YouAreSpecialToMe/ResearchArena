"""
Loss functions for CAG-HNM experiments.
Includes SupCon, JD-CCL fixed selection, and CAG-HNM curriculum loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupConLoss(nn.Module):
    """Standard Supervised Contrastive Learning Loss (Khosla et al., 2020)."""
    
    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Args:
            features: normalized feature projections (batch_size, projection_dim)
            labels: class labels (batch_size,)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-similarity
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)
        
        # Mask for positive pairs (excluding self)
        pos_mask = mask * logits_mask
        
        # Compute log probabilities
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


class JDCCLFixedLoss(nn.Module):
    """
    JD-CCL style fixed attribute-guided hard negative mining.
    Uses fixed top-k selection based on Jaccard similarity.
    """
    
    def __init__(self, temperature=0.1, lambda_weight=2.0, top_k_percent=0.25):
        super(JDCCLFixedLoss, self).__init__()
        self.temperature = temperature
        self.lambda_weight = lambda_weight
        self.top_k_percent = top_k_percent
    
    def forward(self, features, labels, attr_similarity):
        """
        Args:
            features: normalized feature projections (batch_size, projection_dim)
            labels: class labels (batch_size,)
            attr_similarity: pre-computed attribute similarity matrix (num_classes, num_classes)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Get attribute similarity for labels in batch
        # labels: (batch_size,) -> get similarity between each pair
        label_sim = attr_similarity[labels.cpu().numpy()][:, labels.cpu().numpy()]
        label_sim = torch.from_numpy(label_sim).float().to(device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels_expanded = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        
        # Remove self-similarity
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)
        
        # Mask for positive pairs (excluding self)
        pos_mask = mask * logits_mask
        
        # Hard negative selection based on attribute similarity
        # Higher similarity = harder negative
        num_negatives = batch_size - 1  # excluding self
        top_k = max(1, int(num_negatives * self.top_k_percent))
        
        # Create hard negative mask
        # For each anchor, select top-k hardest negatives
        neg_mask = 1 - mask  # Different classes are negatives
        neg_mask.fill_diagonal_(0)
        
        # Apply fixed weighting based on attribute similarity
        weights = torch.exp(self.lambda_weight * label_sim)
        
        # Weight the similarity scores
        weighted_similarity = similarity_matrix + torch.log(weights + 1e-8)
        
        # Compute log probabilities with weights
        exp_logits = torch.exp(weighted_similarity) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss


class CAGHNMLoss(nn.Module):
    """
    Curriculum Attribute-Guided Hard Negative Mining Loss.
    Progressively increases emphasis on hard negatives based on training progress.
    """
    
    def __init__(self, temperature=0.1, lambda_min=0.1, lambda_max=2.0, gamma=2.0):
        super(CAGHNMLoss, self).__init__()
        self.temperature = temperature
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.gamma = gamma
        self.current_epoch = 0
        self.total_epochs = 100
    
    def set_epoch(self, epoch, total_epochs):
        """Update curriculum progress."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def get_curriculum_param(self):
        """
        Compute curriculum parameter lambda(t).
        lambda(t) = lambda_min + (lambda_max - lambda_min) * t^gamma
        where t is normalized training progress [0, 1]
        """
        t = self.current_epoch / max(1, self.total_epochs - 1)
        lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * (t ** self.gamma)
        return lambda_t
    
    def forward(self, features, labels, attr_similarity):
        """
        Args:
            features: normalized feature projections (batch_size, projection_dim)
            labels: class labels (batch_size,)
            attr_similarity: pre-computed attribute similarity matrix (num_classes, num_classes)
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Get current curriculum parameter
        lambda_t = self.get_curriculum_param()
        
        # Get attribute similarity for labels in batch
        label_sim = attr_similarity[labels.cpu().numpy()][:, labels.cpu().numpy()]
        label_sim = torch.from_numpy(label_sim).float().to(device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels_expanded = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_expanded, labels_expanded.T).float().to(device)
        
        # Remove self-similarity
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)
        
        # Mask for positive pairs (excluding self)
        pos_mask = mask * logits_mask
        
        # Compute curriculum-based weights
        # w_n(t) = exp(lambda(t) * s_attr)
        # Only apply to negatives (different classes)
        neg_mask = 1 - mask
        neg_mask.fill_diagonal_(0)
        
        # Weights: 1.0 for positives, exp(lambda * sim) for negatives
        weights = torch.ones_like(similarity_matrix)
        weights = weights + neg_mask * (torch.exp(lambda_t * label_sim) - 1)
        
        # Weighted similarity for denominator
        exp_logits = torch.exp(similarity_matrix) * logits_mask * weights
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positives
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss, lambda_t


def create_loss(loss_type='supcon', temperature=0.1, **kwargs):
    """
    Create loss function based on type.
    
    Args:
        loss_type: 'supcon', 'jdccl_fixed', or 'caghnm'
        temperature: temperature parameter
        **kwargs: additional arguments for specific loss types
    """
    if loss_type == 'supcon':
        return SupConLoss(temperature=temperature)
    elif loss_type == 'jdccl_fixed':
        return JDCCLFixedLoss(
            temperature=temperature,
            lambda_weight=kwargs.get('lambda_weight', 2.0),
            top_k_percent=kwargs.get('top_k_percent', 0.25)
        )
    elif loss_type == 'caghnm':
        return CAGHNMLoss(
            temperature=temperature,
            lambda_min=kwargs.get('lambda_min', 0.1),
            lambda_max=kwargs.get('lambda_max', 2.0),
            gamma=kwargs.get('gamma', 2.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test losses
    batch_size = 32
    projection_dim = 128
    num_classes = 100
    
    features = F.normalize(torch.randn(batch_size, projection_dim), dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Create dummy attribute similarity
    attr_sim = np.random.rand(num_classes, num_classes).astype(np.float32)
    
    print("Testing SupCon Loss...")
    loss_fn = SupConLoss(temperature=0.1)
    loss = loss_fn(features, labels)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nTesting JD-CCL Fixed Loss...")
    loss_fn = JDCCLFixedLoss(temperature=0.1)
    loss = loss_fn(features, labels, attr_sim)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nTesting CAG-HNM Loss...")
    loss_fn = CAGHNMLoss(temperature=0.1, lambda_min=0.1, lambda_max=2.0, gamma=2.0)
    loss_fn.set_epoch(50, 100)
    loss, lambda_t = loss_fn(features, labels, attr_sim)
    print(f"Loss: {loss.item():.4f}, lambda(t): {lambda_t:.4f}")
