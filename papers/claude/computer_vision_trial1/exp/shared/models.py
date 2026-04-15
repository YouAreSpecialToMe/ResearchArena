import torch
import torch.nn as nn
import timm
import numpy as np


def load_model(name, pretrained=True):
    """Load a DeiT model from timm."""
    model_names = {
        'deit_small': 'deit_small_patch16_224',
        'deit_base': 'deit_base_patch16_224',
        'deit_s': 'deit_small_patch16_224',
        'deit_b': 'deit_base_patch16_224',
    }
    model_name = model_names.get(name, name)
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    return model


class AttentionHook:
    """Registers hooks on all attention layers to capture attention weights."""

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for i, block in enumerate(self.model.blocks):
            hook = block.attn.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # timm Attention: we need to get attention weights
            # We'll compute them from qkv
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            self.attention_maps[layer_idx] = attn.detach()
        return hook_fn

    def get_cls_row_entropy(self, layer_idx):
        """Compute CLS-row attention entropy. O(N) per head.
        Returns: [batch] tensor of mean entropy across heads.
        """
        attn = self.attention_maps[layer_idx]  # [B, H, N, N]
        cls_attn = attn[:, :, 0, :]  # [B, H, N] - CLS attending to all tokens
        entropy = -(cls_attn * torch.log(cls_attn + 1e-10)).sum(dim=-1)  # [B, H]
        return entropy.mean(dim=-1)  # [B]

    def get_per_head_cls_entropy(self, layer_idx):
        """Get per-head CLS entropy. Returns [B, H] tensor."""
        attn = self.attention_maps[layer_idx]
        cls_attn = attn[:, :, 0, :]
        entropy = -(cls_attn * torch.log(cls_attn + 1e-10)).sum(dim=-1)
        return entropy

    def get_full_entropy(self, layer_idx):
        """Compute full attention matrix entropy. O(N^2) per head.
        Returns: [batch] tensor of mean entropy across all tokens and heads.
        """
        attn = self.attention_maps[layer_idx]  # [B, H, N, N]
        entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)  # [B, H, N]
        return entropy.mean(dim=(1, 2))  # [B]

    def clear(self):
        self.attention_maps = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
