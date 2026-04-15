"""
Token-level uncertainty computation.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute predictive entropy from logits.
    
    Args:
        logits: Model logits of shape (..., C)
        dim: Dimension along which to compute entropy
    
    Returns:
        Entropy of shape (...)
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = F.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def compute_margin(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute margin uncertainty (1 - (p_max - p_second)).
    Lower margin = higher uncertainty.
    
    Args:
        logits: Model logits of shape (..., C)
        dim: Dimension along which to compute margin
    
    Returns:
        Margin of shape (...)
    """
    probs = F.softmax(logits, dim=dim)
    
    # Get top 2 probabilities
    top2_probs, _ = torch.topk(probs, k=2, dim=dim)
    
    # Margin = p_max - p_second
    margin = top2_probs[..., 0] - top2_probs[..., 1]
    
    # Convert to uncertainty (lower margin = higher uncertainty)
    uncertainty = 1.0 - margin
    
    return uncertainty


def compute_confidence(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute maximum probability (confidence).
    
    Args:
        logits: Model logits of shape (..., C)
        dim: Dimension along which to compute confidence
    
    Returns:
        Confidence of shape (...)
    """
    probs = F.softmax(logits, dim=dim)
    confidence, _ = probs.max(dim=dim)
    return confidence


def compute_token_uncertainty(logits: torch.Tensor, method: str = 'entropy') -> torch.Tensor:
    """
    Compute token-level uncertainty using specified method.
    
    Args:
        logits: Token-level logits of shape (B, N, C)
        method: Uncertainty method ('entropy', 'margin', 'confidence')
    
    Returns:
        Uncertainty scores of shape (B, N)
    """
    if method == 'entropy':
        return compute_entropy(logits, dim=-1)
    elif method == 'margin':
        return compute_margin(logits, dim=-1)
    elif method == 'confidence':
        # Higher confidence = lower uncertainty
        return 1.0 - compute_confidence(logits, dim=-1)
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


def normalize_uncertainty(uncertainty: torch.Tensor, 
                          method: str = 'minmax') -> torch.Tensor:
    """
    Normalize uncertainty scores.
    
    Args:
        uncertainty: Uncertainty scores of shape (...)
        method: Normalization method ('minmax', 'zscore', 'softmax')
    
    Returns:
        Normalized uncertainty scores
    """
    if method == 'minmax':
        min_val = uncertainty.min(dim=-1, keepdim=True)[0]
        max_val = uncertainty.max(dim=-1, keepdim=True)[0]
        return (uncertainty - min_val) / (max_val - min_val + 1e-10)
    
    elif method == 'zscore':
        mean = uncertainty.mean(dim=-1, keepdim=True)
        std = uncertainty.std(dim=-1, keepdim=True)
        return (uncertainty - mean) / (std + 1e-10)
    
    elif method == 'softmax':
        return F.softmax(uncertainty, dim=-1)
    
    else:
        return uncertainty


def compute_uncertainty_stats(uncertainty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute statistics of uncertainty scores.
    
    Args:
        uncertainty: Uncertainty scores of shape (B, N)
    
    Returns:
        mean: Mean uncertainty per batch (B,)
        std: Std uncertainty per batch (B,)
    """
    mean = uncertainty.mean(dim=-1)
    std = uncertainty.std(dim=-1)
    return mean, std


if __name__ == '__main__':
    # Test uncertainty computation
    print("Testing uncertainty computation...")
    
    batch_size = 4
    num_tokens = 196
    num_classes = 1000
    
    # Create dummy logits
    logits = torch.randn(batch_size, num_tokens, num_classes)
    
    # Test entropy
    print("\nEntropy:")
    entropy = compute_entropy(logits)
    print(f"  Shape: {entropy.shape}")
    print(f"  Range: [{entropy.min():.3f}, {entropy.max():.3f}]")
    
    # Test margin
    print("\nMargin:")
    margin = compute_margin(logits)
    print(f"  Shape: {margin.shape}")
    print(f"  Range: [{margin.min():.3f}, {margin.max():.3f}]")
    
    # Test token uncertainty
    print("\nToken uncertainty (entropy):")
    uncertainty = compute_token_uncertainty(logits, method='entropy')
    print(f"  Shape: {uncertainty.shape}")
    
    # Test normalization
    print("\nNormalized uncertainty (minmax):")
    norm_unc = normalize_uncertainty(uncertainty, method='minmax')
    print(f"  Shape: {norm_unc.shape}")
    print(f"  Range: [{norm_unc.min():.3f}, {norm_unc.max():.3f}]")
    
    # Test stats
    print("\nUncertainty stats:")
    mean, std = compute_uncertainty_stats(uncertainty)
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    
    print("\nUncertainty computation test passed!")
