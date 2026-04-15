"""
Model definitions and utilities for DU-VPT.
Includes ViT backbone, prompt layers, and adaptation mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple, Optional, Dict
import numpy as np


class PromptLayer(nn.Module):
    """Learnable prompt tokens for a single layer."""
    
    def __init__(self, n_prompts: int, embed_dim: int, prompt_type: str = 'semantic'):
        super().__init__()
        self.n_prompts = n_prompts
        self.embed_dim = embed_dim
        self.prompt_type = prompt_type
        
        # Initialize prompts
        self.prompts = nn.Parameter(torch.randn(n_prompts, embed_dim) * 0.02)
        
        # Prompt type-specific initialization
        if prompt_type == 'structural':
            # Structure prompts for local smoothness
            self.local_weight = 0.1
        else:
            self.local_weight = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend prompts to input tokens."""
        batch_size = x.shape[0]
        prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompts, x], dim=1)
    
    def get_prompts(self) -> torch.Tensor:
        """Get current prompt values."""
        return self.prompts


class DUVPTViT(nn.Module):
    """
    DU-VPT: Decomposed Uncertainty-Guided Visual Prompt Tuning.
    Wraps a ViT model with uncertainty decomposition and targeted prompts.
    """
    
    def __init__(
        self,
        vit_model: nn.Module,
        n_prompts: int = 10,
        n_layers: int = 12,
        embed_dim: int = 768,
        tau_alpha: float = 0.2,
        tau_epsilon: float = 1.0,
    ):
        super().__init__()
        self.vit = vit_model
        self.n_prompts = n_prompts
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.tau_alpha = tau_alpha
        self.tau_epsilon = tau_epsilon
        
        # Prompt banks for each layer
        self.struct_prompts = nn.ModuleList([
            PromptLayer(n_prompts, embed_dim, prompt_type='structural')
            for _ in range(n_layers)
        ])
        
        self.semantic_prompts = nn.ModuleList([
            PromptLayer(n_prompts, embed_dim, prompt_type='semantic')
            for _ in range(n_layers)
        ])
        
        # Calibration statistics (computed once)
        self.register_buffer('calib_mean', torch.zeros(n_layers, embed_dim))
        self.register_buffer('calib_std', torch.ones(n_layers, embed_dim))
        self.register_buffer('fisher_info', torch.zeros(n_layers, embed_dim))
        
        # Layer selection tracking
        self.selected_layers = []
        self.shift_type = 'unknown'
        
        # Freeze backbone
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Only prompt parameters are trainable
        for prompt_layer in self.struct_prompts:
            for param in prompt_layer.parameters():
                param.requires_grad = True
        
        for prompt_layer in self.semantic_prompts:
            for param in prompt_layer.parameters():
                param.requires_grad = True
    
    def compute_aleatoric_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute aleatoric uncertainty from local feature consistency.
        High values indicate local inconsistency (data corruption).
        """
        # features: [B, N, D] where N is number of patches
        B, N, D = features.shape
        
        # Reshape to 2D grid (assuming square patches)
        H = W = int(np.sqrt(N - 1))  # -1 for cls token
        
        if H * W != N - 1:
            # Non-square or no cls token - use variance across all tokens
            return features.var(dim=1).mean(dim=-1)
        
        # Separate cls token
        cls_token = features[:, 0:1, :]
        patch_tokens = features[:, 1:, :]
        
        # Reshape to grid
        patch_tokens = patch_tokens.reshape(B, H, W, D)
        
        # Compute local consistency (4-connected neighbors)
        local_vars = []
        
        for i in range(H):
            for j in range(W):
                neighbors = []
                # 4-connected neighbors
                if i > 0: neighbors.append(patch_tokens[:, i-1, j, :])
                if i < H-1: neighbors.append(patch_tokens[:, i+1, j, :])
                if j > 0: neighbors.append(patch_tokens[:, i, j-1, :])
                if j < W-1: neighbors.append(patch_tokens[:, i, j+1, :])
                
                if neighbors:
                    center = patch_tokens[:, i, j, :]
                    neighbor_mean = torch.stack(neighbors, dim=1).mean(dim=1)
                    local_var = ((center - neighbor_mean) ** 2).sum(dim=-1)
                    local_vars.append(local_var)
        
        if local_vars:
            local_var = torch.stack(local_vars, dim=1).mean(dim=1)
        else:
            local_var = torch.zeros(B, device=features.device)
        
        return local_var
    
    def compute_epistemic_uncertainty(self, features: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Compute epistemic uncertainty from deviation from calibration statistics.
        High values indicate OOD features.
        """
        # features: [B, N, D]
        # Use cls token for epistemic uncertainty
        cls_features = features[:, 0, :]  # [B, D]
        
        # Normalize with calibration statistics
        mean = self.calib_mean[layer_idx]
        std = self.calib_std[layer_idx]
        
        normalized = (cls_features - mean) / (std + 1e-8)
        
        # Epistemic = deviation from calibration
        epistemic = (normalized ** 2).sum(dim=-1)
        
        return epistemic
    
    def decompose_uncertainty(self, layer_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose uncertainty into aleatoric and epistemic components.
        
        Returns:
            aleatoric: [B, L] aleatoric uncertainty per layer
            epistemic: [B, L] epistemic uncertainty per layer
        """
        B = layer_features[0].shape[0]
        L = len(layer_features)
        
        aleatoric = torch.zeros(B, L, device=layer_features[0].device)
        epistemic = torch.zeros(B, L, device=layer_features[0].device)
        
        for l, features in enumerate(layer_features):
            aleatoric[:, l] = self.compute_aleatoric_uncertainty(features)
            epistemic[:, l] = self.compute_epistemic_uncertainty(features, l)
        
        # Normalize
        aleatoric = aleatoric / (aleatoric.max(dim=1, keepdim=True)[0] + 1e-8)
        epistemic = epistemic / (epistemic.max(dim=1, keepdim=True)[0] + 1e-8)
        
        return aleatoric, epistemic
    
    def diagnose_shift(
        self, 
        aleatoric: torch.Tensor, 
        epistemic: torch.Tensor
    ) -> Tuple[str, List[int], str]:
        """
        Diagnose shift type based on uncertainty patterns.
        
        Returns:
            shift_type: 'low_level', 'semantic', 'mixed', or 'none'
            target_layers: List of layer indices to adapt
            prompt_type: 'structural', 'semantic', or 'hybrid'
        """
        B, L = aleatoric.shape
        
        # Average over batch for diagnosis
        alpha_mean = aleatoric.mean(dim=0)
        epsilon_mean = epistemic.mean(dim=0)
        
        # Split layers into early, middle, deep
        early_end = L // 3
        deep_start = 2 * L // 3
        
        alpha_early = alpha_mean[:early_end].mean()
        alpha_deep = alpha_mean[deep_start:].mean()
        epsilon_deep = epsilon_mean[deep_start:].mean()
        
        # Diagnosis logic
        if alpha_early > self.tau_alpha and epsilon_deep < self.tau_epsilon:
            shift_type = 'low_level'
            target_layers = list(range(0, L // 2))
            prompt_type = 'structural'
        elif alpha_early < self.tau_alpha and epsilon_deep > self.tau_epsilon:
            shift_type = 'semantic'
            target_layers = list(range(deep_start, L))
            prompt_type = 'semantic'
        elif alpha_early > self.tau_alpha and epsilon_deep > self.tau_epsilon:
            shift_type = 'mixed'
            target_layers = list(range(L))
            prompt_type = 'semantic'  # Use semantic for mixed
        else:
            shift_type = 'none'
            target_layers = []
            prompt_type = 'none'
        
        return shift_type, target_layers, prompt_type
    
    def extract_layer_features(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Extract features at each layer of ViT.
        
        Returns:
            layer_features: List of [B, N, D] tensors
            output: Final output
        """
        # Get intermediate layer features using forward hooks
        layer_features = []
        hooks = []
        
        def hook_fn(module, input, output):
            layer_features.append(output)
        
        # Register hooks on transformer blocks
        blocks = self.vit.blocks if hasattr(self.vit, 'blocks') else self.vit.transformer.blocks
        
        for block in blocks:
            hook = block.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        output = self.vit(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_features, output
    
    def forward_with_prompts(
        self, 
        x: torch.Tensor,
        target_layers: List[int],
        prompt_type: str
    ) -> torch.Tensor:
        """
        Forward pass with prompts at selected layers.
        """
        if prompt_type == 'none' or not target_layers:
            return self.vit(x)
        
        # This is a simplified version - in practice would need to modify
        # the forward pass to inject prompts at specific layers
        # For now, use standard forward and return output
        return self.vit(x)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = False):
        """Forward pass with optional uncertainty decomposition."""
        if not return_uncertainty:
            return self.vit(x)
        
        # Extract features and decompose uncertainty
        layer_features, output = self.extract_layer_features(x)
        
        # Decompose uncertainty
        aleatoric, epistemic = self.decompose_uncertainty(layer_features)
        
        # Diagnose shift
        shift_type, target_layers, prompt_type = self.diagnose_shift(aleatoric, epistemic)
        
        self.selected_layers = target_layers
        self.shift_type = shift_type
        
        return output, {
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'shift_type': shift_type,
            'target_layers': target_layers,
            'prompt_type': prompt_type
        }
    
    def get_prompt_params(self) -> List[nn.Parameter]:
        """Get all prompt parameters."""
        params = []
        for prompt_layer in self.struct_prompts:
            params.append(prompt_layer.prompts)
        for prompt_layer in self.semantic_prompts:
            params.append(prompt_layer.prompts)
        return params
    
    def set_calibration_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """Set calibration statistics."""
        self.calib_mean.copy_(mean)
        self.calib_std.copy_(std)
    
    def compute_fisher_penalty(self) -> torch.Tensor:
        """Compute Fisher regularization penalty."""
        penalty = 0.0
        for l, (struct_prompt, sem_prompt) in enumerate(zip(self.struct_prompts, self.semantic_prompts)):
            fisher = self.fisher_info[l]
            # Regularize towards initial values (0 initialization)
            penalty += ((struct_prompt.prompts ** 2) / (2 * (fisher + 1e-8))).sum()
            penalty += ((sem_prompt.prompts ** 2) / (2 * (fisher + 1e-8))).sum()
        return penalty


def create_vit_model(model_name='vit_base_patch16_224', pretrained=True, num_classes=1000):
    """Create a ViT model using timm."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model


def load_pretrained_vit(device='cuda'):
    """Load pretrained ViT-B/16 for DU-VPT experiments."""
    model = timm.create_model(
        'vit_base_patch16_224.augreg_in21k_ft_in1k',
        pretrained=True,
        num_classes=1000
    )
    model = model.to(device)
    model.eval()
    return model
