"""
Model configurations and utilities for loading pretrained ViTs.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, List, Optional, Tuple


# Model configuration dictionary
MODEL_CONFIGS = {
    'deit_small_patch16_224': {
        'name': 'deit_small_patch16_224',
        'embed_dim': 384,
        'num_layers': 12,
        'num_heads': 6,
        'patch_size': 16,
        'img_size': 224,
        'num_params': '22M',
        'head_layer_for_aux': 8,  # Recommended layer for auxiliary head
    },
    'vit_base_patch16_224': {
        'name': 'vit_base_patch16_224',
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'patch_size': 16,
        'img_size': 224,
        'num_params': '86M',
        'head_layer_for_aux': 8,
    },
    'vit_small_patch16_224': {
        'name': 'vit_small_patch16_224',
        'embed_dim': 384,
        'num_layers': 12,
        'num_heads': 6,
        'patch_size': 16,
        'img_size': 224,
        'num_params': '22M',
        'head_layer_for_aux': 8,
    },
}


def get_model_config(model_name: str) -> Dict:
    """Get configuration for a model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]


def load_model(model_name: str, num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
    """Load a pretrained ViT model."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def configure_model_for_tta(model: nn.Module) -> nn.Module:
    """
    Configure model for TTA by enabling gradients on normalization layers.
    This is the standard approach used in TENT and other TTA methods.
    """
    model.train()
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients for normalization layers
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
            for param in m.parameters():
                param.requires_grad = True
            # Use eval mode for BN to use test-time statistics
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    return model


def get_norm_layer_params(model: nn.Module) -> List[nn.Parameter]:
    """Get parameters of normalization layers."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
            params.extend(list(m.parameters()))
    return params


def copy_model_state(model: nn.Module) -> Dict:
    """Create a copy of model state for resetting."""
    return {name: param.clone() for name, param in model.named_parameters()}


def load_model_state(model: nn.Module, state_dict: Dict):
    """Load model state from a saved dictionary."""
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name])


class ModelWithIntermediateFeatures(nn.Module):
    """
    Wrapper that extracts intermediate layer features from a ViT.
    Useful for attaching auxiliary heads.
    """
    
    def __init__(self, model: nn.Module, layer_idx: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.intermediate_features = None
        
        # Register hook to capture intermediate features
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook to capture features at specified layer."""
        def hook_fn(module, input, output):
            self.intermediate_features = output
        
        # Find the target layer
        if hasattr(self.model, 'blocks'):
            # Deit/ViT style
            target_layer = self.model.blocks[self.layer_idx]
            target_layer.register_forward_hook(hook_fn)
        elif hasattr(self.model, 'layers'):
            # Swin style
            target_layer = self.model.layers[self.layer_idx]
            target_layer.register_forward_hook(hook_fn)
        else:
            raise ValueError("Unknown model architecture")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass returning both output and intermediate features."""
        output = self.model(x)
        return output, self.intermediate_features
    
    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.model, name)


def get_num_classes(dataset: str) -> int:
    """Get number of classes for a dataset."""
    if dataset == 'cifar10':
        return 10
    elif dataset == 'cifar100':
        return 100
    elif dataset in ['imagenet', 'imagenet-c', 'imagenet-r', 'imagenet-a']:
        return 1000
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == '__main__':
    # Test model loading
    print("Testing model loading...")
    
    for model_name in ['deit_small_patch16_224', 'vit_base_patch16_224']:
        print(f"\nLoading {model_name}...")
        model = load_model(model_name, num_classes=1000, pretrained=False)
        config = get_model_config(model_name)
        
        print(f"  Embed dim: {config['embed_dim']}")
        print(f"  Num layers: {config['num_layers']}")
        print(f"  Num params: {config['num_params']}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        print(f"  Output shape: {out.shape}")
    
    print("\nModel config test passed!")
