"""
Token selection strategies based on uncertainty.
"""

import torch
from typing import Tuple, Optional


def dynamic_threshold_selection(uncertainty: torch.Tensor, 
                                 alpha: float = 0.5) -> torch.Tensor:
    """
    Select tokens using dynamic threshold based on mean and std of uncertainty.
    
    Selection criterion: U_i > mu_U + alpha * sigma_U
    
    Args:
        uncertainty: Token uncertainty scores of shape (B, N)
        alpha: Threshold parameter (default: 0.5)
    
    Returns:
        Binary mask of shape (B, N) where True indicates selected tokens
    """
    # Compute statistics per sample
    mean_unc = uncertainty.mean(dim=-1, keepdim=True)
    std_unc = uncertainty.std(dim=-1, keepdim=True)
    
    # Dynamic threshold
    threshold = mean_unc + alpha * std_unc
    
    # Select tokens above threshold
    mask = uncertainty > threshold
    
    return mask


def fixed_ratio_selection(uncertainty: torch.Tensor,
                          ratio: float = 0.5) -> torch.Tensor:
    """
    Select top-k% tokens by uncertainty.
    
    Args:
        uncertainty: Token uncertainty scores of shape (B, N)
        ratio: Fraction of tokens to select (0.0 to 1.0)
    
    Returns:
        Binary mask of shape (B, N)
    """
    batch_size, num_tokens = uncertainty.shape
    k = max(1, int(num_tokens * ratio))
    
    # Get indices of top-k uncertain tokens
    _, top_k_indices = torch.topk(uncertainty, k=k, dim=-1)
    
    # Create mask
    mask = torch.zeros_like(uncertainty, dtype=torch.bool)
    batch_indices = torch.arange(batch_size, device=uncertainty.device).unsqueeze(-1)
    mask[batch_indices, top_k_indices] = True
    
    return mask


def fixed_threshold_selection(uncertainty: torch.Tensor,
                               threshold: float) -> torch.Tensor:
    """
    Select tokens with uncertainty above fixed threshold.
    
    Args:
        uncertainty: Token uncertainty scores of shape (B, N)
        threshold: Fixed threshold value
    
    Returns:
        Binary mask of shape (B, N)
    """
    return uncertainty > threshold


def adaptive_selection(uncertainty: torch.Tensor,
                        min_ratio: float = 0.2,
                        max_ratio: float = 0.8,
                        alpha: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive selection that ensures minimum and maximum selection ratios.
    
    Args:
        uncertainty: Token uncertainty scores of shape (B, N)
        min_ratio: Minimum fraction of tokens to select
        max_ratio: Maximum fraction of tokens to select
        alpha: Threshold parameter
    
    Returns:
        mask: Binary mask of shape (B, N)
        stats: Dictionary with selection statistics
    """
    batch_size, num_tokens = uncertainty.shape
    
    # Start with dynamic threshold
    mask = dynamic_threshold_selection(uncertainty, alpha)
    
    # Get selection ratio per sample
    selection_ratio = mask.float().mean(dim=-1)
    
    # Adjust for samples with too few or too many selected tokens
    for i in range(batch_size):
        if selection_ratio[i] < min_ratio:
            # Select more tokens
            mask[i] = fixed_ratio_selection(uncertainty[i:i+1], min_ratio)[0]
        elif selection_ratio[i] > max_ratio:
            # Select fewer tokens
            mask[i] = fixed_ratio_selection(uncertainty[i:i+1], max_ratio)[0]
    
    # Compute statistics
    stats = {
        'mean_ratio': mask.float().mean().item(),
        'min_ratio': mask.float().mean(dim=-1).min().item(),
        'max_ratio': mask.float().mean(dim=-1).max().item(),
    }
    
    return mask, stats


def random_selection(num_tokens: int,
                     ratio: float,
                     batch_size: int,
                     device: torch.device) -> torch.Tensor:
    """
    Random token selection (for ablation baseline).
    
    Args:
        num_tokens: Number of tokens per sample
        ratio: Fraction of tokens to select
        batch_size: Batch size
        device: Device
    
    Returns:
        Binary mask of shape (B, N)
    """
    k = max(1, int(num_tokens * ratio))
    
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        indices = torch.randperm(num_tokens, device=device)[:k]
        mask[i, indices] = True
    
    return mask


def compute_selection_ratio(mask: torch.Tensor) -> float:
    """
    Compute the fraction of selected tokens.
    
    Args:
        mask: Binary mask of shape (B, N)
    
    Returns:
        Selection ratio (fraction of True values)
    """
    return mask.float().mean().item()


if __name__ == '__main__':
    # Test token selection
    print("Testing token selection...")
    
    batch_size = 4
    num_tokens = 196
    
    # Create dummy uncertainty
    torch.manual_seed(42)
    uncertainty = torch.randn(batch_size, num_tokens).abs()
    
    # Test dynamic threshold
    print("\nDynamic threshold selection (alpha=0.5):")
    mask = dynamic_threshold_selection(uncertainty, alpha=0.5)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Selection ratio: {compute_selection_ratio(mask):.3f}")
    
    # Test fixed ratio
    print("\nFixed ratio selection (ratio=0.5):")
    mask = fixed_ratio_selection(uncertainty, ratio=0.5)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Selection ratio: {compute_selection_ratio(mask):.3f}")
    
    # Test adaptive selection
    print("\nAdaptive selection:")
    mask, stats = adaptive_selection(uncertainty, min_ratio=0.2, max_ratio=0.8, alpha=0.5)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Stats: {stats}")
    
    # Test random selection
    print("\nRandom selection (ratio=0.5):")
    mask = random_selection(num_tokens, ratio=0.5, batch_size=batch_size, 
                            device=torch.device('cpu'))
    print(f"  Mask shape: {mask.shape}")
    print(f"  Selection ratio: {compute_selection_ratio(mask):.3f}")
    
    print("\nToken selection test passed!")
