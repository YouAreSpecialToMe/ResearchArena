"""
Lightweight auxiliary projection head for token-level predictions.
"""

import torch
import torch.nn as nn
from typing import Optional


class UncertaintyHead(nn.Module):
    """
    Lightweight auxiliary projection head that maps token features to class logits.
    Attached to intermediate layer to obtain token-level predictions.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_classes: int,
                 head_type: str = 'mlp',
                 hidden_dim: Optional[int] = None):
        """
        Args:
            embed_dim: Dimension of token features
            num_classes: Number of output classes
            head_type: 'linear' or 'mlp'
            hidden_dim: Hidden dimension for MLP (default: embed_dim // 4)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.head_type = head_type
        
        if hidden_dim is None:
            hidden_dim = embed_dim // 4
        
        if head_type == 'linear':
            self.head = nn.Linear(embed_dim, num_classes)
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize head weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Token features of shape (B, N, D) or (B*N, D)
        
        Returns:
            Class logits of shape (B, N, C) or (B*N, C)
        """
        return self.head(x)
    
    def get_num_params(self) -> int:
        """Get number of parameters in the head."""
        return sum(p.numel() for p in self.parameters())


class MultiLayerUncertaintyHead(nn.Module):
    """
    Uncertainty heads attached to multiple layers for multi-scale uncertainty.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 layer_indices: list,
                 head_type: str = 'mlp'):
        super().__init__()
        self.layer_indices = layer_indices
        
        # Create head for each layer
        self.heads = nn.ModuleDict({
            str(idx): UncertaintyHead(embed_dim, num_classes, head_type)
            for idx in layer_indices
        })
    
    def forward(self, features_dict: dict) -> dict:
        """
        Forward pass for multiple layers.
        
        Args:
            features_dict: Dict mapping layer index to features
        
        Returns:
            Dict mapping layer index to logits
        """
        logits_dict = {}
        for idx in self.layer_indices:
            if str(idx) in features_dict:
                logits_dict[str(idx)] = self.heads[str(idx)](features_dict[str(idx)])
        return logits_dict


if __name__ == '__main__':
    # Test uncertainty head
    print("Testing UncertaintyHead...")
    
    batch_size = 4
    num_tokens = 196  # 14x14 patches for 224x224 images with patch size 16
    embed_dim = 768
    num_classes = 1000
    
    # Test linear head
    print("\nLinear head:")
    head_linear = UncertaintyHead(embed_dim, num_classes, head_type='linear')
    x = torch.randn(batch_size, num_tokens, embed_dim)
    logits = head_linear(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {head_linear.get_num_params():,}")
    
    # Test MLP head
    print("\nMLP head:")
    head_mlp = UncertaintyHead(embed_dim, num_classes, head_type='mlp', hidden_dim=192)
    logits = head_mlp(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters: {head_mlp.get_num_params():,}")
    
    print("\nUncertaintyHead test passed!")
