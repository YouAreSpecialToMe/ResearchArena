"""Attention Entropy Profiling (AEP) - Core implementation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


class AEPExtractor:
    """Extracts Attention Entropy Profile from Vision Transformers."""

    def __init__(self, model, num_layers: int = 12):
        self.model = model
        self.num_layers = num_layers
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on attention modules to capture attention weights."""
        self.hooks = []
        for name, module in self.model.named_modules():
            # timm ViT: attention is in blocks[i].attn
            if hasattr(module, 'attn_drop') and 'attn' in name and 'attn_drop' not in name:
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)

    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights after softmax.

        In timm's Attention module, we need to capture the attention weights.
        We'll use a pre-hook on attn_drop to get the weights.
        """
        pass  # We'll use a different approach

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def extract_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention maps from all layers for input x.

        Returns list of tensors, each of shape (batch, heads, tokens, tokens).
        """
        attention_maps = []

        def make_hook(storage):
            def hook_fn(module, input, output):
                # In timm's Attention, we need to intercept the attention weights
                # The qkv projection happens, then attention is computed
                pass
            return hook_fn

        # For timm ViTs, we need to monkey-patch the forward method
        # to capture attention weights
        return self._extract_with_monkey_patch(x)

    def _extract_with_monkey_patch(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Extract attention maps by temporarily modifying the attention forward pass."""
        attention_maps = []
        original_forwards = []

        # Get all attention modules
        attn_modules = []
        for name, module in self.model.named_modules():
            if type(module).__name__ == 'Attention' and hasattr(module, 'qkv'):
                attn_modules.append(module)

        # Monkey-patch each attention module
        for attn in attn_modules:
            original_forwards.append(attn.forward)

            def make_new_forward(original_forward, attn_mod):
                def new_forward(x):
                    B, N, C = x.shape
                    qkv = attn_mod.qkv(x).reshape(B, N, 3, attn_mod.num_heads,
                                                     C // attn_mod.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)

                    # Handle different attention scaling
                    head_dim = C // attn_mod.num_heads
                    scale = attn_mod.scale if hasattr(attn_mod, 'scale') else head_dim ** -0.5

                    attn_weights = (q @ k.transpose(-2, -1)) * scale
                    attn_weights = attn_weights.softmax(dim=-1)

                    # Store attention weights (detached, on CPU to save GPU memory)
                    attention_maps.append(attn_weights.detach().cpu())

                    attn_weights = attn_mod.attn_drop(attn_weights)
                    out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
                    out = attn_mod.proj(out)
                    out = attn_mod.proj_drop(out)
                    return out
                return new_forward

            attn.forward = make_new_forward(attn.forward, attn)

        # Forward pass
        with torch.no_grad():
            logits = self.model(x)

        # Restore original forwards
        for attn, orig_fwd in zip(attn_modules, original_forwards):
            attn.forward = orig_fwd

        return logits, attention_maps


def compute_cls_entropy(attn: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute entropy of CLS-to-patch attention distribution.

    Args:
        attn: (batch, heads, tokens, tokens) attention weights
    Returns:
        (batch, heads) entropy values
    """
    # CLS token attention to all other tokens (excluding self)
    cls_attn = attn[:, :, 0, 1:]  # (batch, heads, num_patches)
    # Renormalize after removing self-attention
    cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + eps)
    entropy = -(cls_attn * torch.log(cls_attn + eps)).sum(dim=-1)  # (batch, heads)
    return entropy


def compute_avg_token_entropy(attn: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute average entropy across all query tokens.

    Args:
        attn: (batch, heads, tokens, tokens)
    Returns:
        (batch, heads) average entropy
    """
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (batch, heads, tokens)
    return entropy.mean(dim=-1)  # (batch, heads)


def compute_concentration_ratio(attn: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Compute fraction of attention on top-k tokens, averaged over queries.

    Args:
        attn: (batch, heads, tokens, tokens)
        k: number of top tokens
    Returns:
        (batch, heads) concentration ratios
    """
    topk_vals, _ = attn.topk(min(k, attn.shape[-1]), dim=-1)
    concentration = topk_vals.sum(dim=-1)  # (batch, heads, tokens)
    return concentration.mean(dim=-1)  # (batch, heads)


def compute_head_agreement(attn: torch.Tensor) -> torch.Tensor:
    """Compute mean pairwise cosine similarity of CLS attention across heads.

    Args:
        attn: (batch, heads, tokens, tokens)
    Returns:
        (batch,) agreement scores
    """
    cls_attn = attn[:, :, 0, 1:]  # (batch, heads, num_patches)
    # Normalize
    cls_attn_norm = F.normalize(cls_attn, dim=-1)  # (batch, heads, num_patches)
    # Pairwise cosine similarity
    sim = torch.bmm(cls_attn_norm, cls_attn_norm.transpose(-1, -2))  # (batch, heads, heads)
    num_heads = attn.shape[1]
    # Mean of upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(num_heads, num_heads, device=sim.device), diagonal=1).bool()
    agreement = sim[:, mask].mean(dim=-1)  # (batch,)
    return agreement


def compute_aep_profile(attention_maps: List[torch.Tensor]) -> np.ndarray:
    """Compute AEP profile from list of attention maps.

    For each layer: [cls_entropy_mean, cls_entropy_std, avg_token_entropy,
                     concentration_ratio, head_agreement]
    Total dimension: 5 * num_layers

    Args:
        attention_maps: list of (batch, heads, tokens, tokens) tensors
    Returns:
        (batch, 5*num_layers) numpy array
    """
    profiles = []
    batch_size = attention_maps[0].shape[0]

    for attn in attention_maps:
        cls_ent = compute_cls_entropy(attn)  # (batch, heads)
        avg_ent = compute_avg_token_entropy(attn)  # (batch, heads)
        conc = compute_concentration_ratio(attn)  # (batch, heads)
        agreement = compute_head_agreement(attn)  # (batch,)

        layer_profile = torch.stack([
            cls_ent.mean(dim=-1),   # mean CLS entropy across heads
            cls_ent.std(dim=-1),    # std CLS entropy across heads
            avg_ent.mean(dim=-1),   # mean avg token entropy across heads
            conc.mean(dim=-1),      # mean concentration across heads
            agreement,              # head agreement
        ], dim=-1)  # (batch, 5)
        profiles.append(layer_profile)

    # Stack all layers: (batch, num_layers, 5) -> (batch, 5*num_layers)
    profiles = torch.stack(profiles, dim=1)  # (batch, num_layers, 5)
    profiles = profiles.reshape(batch_size, -1)  # (batch, 5*num_layers)
    return profiles.numpy()


def compute_id_statistics(profiles: np.ndarray, reg: float = 1e-4) -> Dict:
    """Compute in-distribution reference statistics.

    Args:
        profiles: (N, D) array of AEP profiles
        reg: regularization for covariance matrix
    Returns:
        dict with 'mean', 'cov', 'cov_inv'
    """
    mean = profiles.mean(axis=0)
    cov = np.cov(profiles, rowvar=False)
    cov += reg * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    return {'mean': mean, 'cov': cov, 'cov_inv': cov_inv}


def compute_mahalanobis_scores(profiles: np.ndarray, stats: Dict) -> np.ndarray:
    """Compute Mahalanobis distance scores.

    Args:
        profiles: (N, D) array
        stats: dict with 'mean', 'cov_inv'
    Returns:
        (N,) array of scores
    """
    diff = profiles - stats['mean']
    scores = np.sum(diff @ stats['cov_inv'] * diff, axis=1)
    return scores


def compute_aep_profile_subset(attention_maps: List[torch.Tensor],
                                feature_indices: Optional[List[int]] = None,
                                layer_indices: Optional[List[int]] = None) -> np.ndarray:
    """Compute AEP profile with optional feature/layer subsetting for ablations.

    Args:
        attention_maps: list of attention maps
        feature_indices: which of the 5 features to include (0-4)
        layer_indices: which layers to include
    Returns:
        (batch, D) numpy array
    """
    if layer_indices is not None:
        attention_maps = [attention_maps[i] for i in layer_indices]

    profiles = []
    batch_size = attention_maps[0].shape[0]

    for attn in attention_maps:
        all_features = []
        cls_ent = compute_cls_entropy(attn)
        avg_ent = compute_avg_token_entropy(attn)
        conc = compute_concentration_ratio(attn)
        agreement = compute_head_agreement(attn)

        feature_list = [
            cls_ent.mean(dim=-1),
            cls_ent.std(dim=-1),
            avg_ent.mean(dim=-1),
            conc.mean(dim=-1),
            agreement,
        ]

        if feature_indices is not None:
            feature_list = [feature_list[i] for i in feature_indices]

        layer_profile = torch.stack(feature_list, dim=-1)
        profiles.append(layer_profile)

    profiles = torch.stack(profiles, dim=1)
    profiles = profiles.reshape(batch_size, -1)
    return profiles.numpy()
