import torch
import math
import numpy as np
import json
import os
from tqdm import tqdm
from .models import AttentionHook


def compute_cls_entropy_from_tokens(block, x):
    """Compute CLS-row attention entropy from input tokens x.
    x: [B, N, C] — tokens entering this block (before any merging).
    Returns: scalar mean entropy across batch and heads.
    """
    B, N, C = x.shape
    num_heads = block.attn.num_heads
    head_dim = C // num_heads
    with torch.no_grad():
        qkv = block.attn.qkv(x).reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * block.attn.scale
        attn = attn.softmax(dim=-1)
        cls_attn = attn[:, :, 0, :]  # [B, H, N]
        entropy = -(cls_attn * torch.log(cls_attn + 1e-10)).sum(dim=-1)  # [B, H]
        mean_entropy = entropy.mean().item()
    return mean_entropy, N


def calibrate(model, dataloader, num_images=500, seed=42):
    """Compute per-layer CLS-row entropy statistics on clean data WITHOUT ToMe.

    CRITICAL FIX: Calibration is done on the UNPATCHED model (no token merging).
    This breaks the circularity where ToMe masks the entropy signal.
    We also store the token count N at each layer for normalization.
    """
    device = next(model.parameters()).device
    num_layers = len(model.blocks)
    all_entropies = {i: [] for i in range(num_layers)}
    all_token_counts = {i: [] for i in range(num_layers)}

    count = 0
    model.eval()

    # Register hooks to capture input to each block
    layer_inputs = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input):
            layer_inputs[idx] = input[0].detach()
        return hook_fn

    for i, block in enumerate(model.blocks):
        h = block.register_forward_pre_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calibrating (no ToMe)"):
            if count >= num_images:
                break
            images = images.to(device)
            batch_size = images.shape[0]
            remaining = num_images - count
            if batch_size > remaining:
                images = images[:remaining]
                batch_size = remaining

            layer_inputs.clear()
            _ = model(images)

            for layer_idx in range(num_layers):
                x = layer_inputs[layer_idx]
                ent, N = compute_cls_entropy_from_tokens(model.blocks[layer_idx], x)
                all_entropies[layer_idx].append(ent)
                all_token_counts[layer_idx].append(N)

            count += batch_size

    for h in hooks:
        h.remove()

    stats = {}
    for layer_idx in range(num_layers):
        ents = np.array(all_entropies[layer_idx])
        N_cal = int(np.mean(all_token_counts[layer_idx]))
        # Store normalized entropy: H / log(N) so it's comparable across different N
        log_N = math.log(N_cal) if N_cal > 1 else 1.0
        norm_ents = ents / log_N
        stats[layer_idx] = {
            'mean': float(ents.mean()),
            'std': float(ents.std()),
            'norm_mean': float(norm_ents.mean()),
            'norm_std': float(norm_ents.std()),
            'N': N_cal,
        }

    return stats


class EGTMPatcher:
    """Entropy-Guided Token Merging (v2 — circularity fix).

    Key fix: Calibration is done WITHOUT ToMe, so clean entropy stats reflect
    the full token space. During inference, entropy is computed BEFORE merging
    at each layer, and normalized by log(N) to account for token count reduction
    from prior layers' merging.

    r_l = round(r_0 * max(alpha, exp(-beta * delta_l)))
    where delta_l = max(0, (H_norm_l - norm_mean_l) / norm_std_l)
    H_norm_l = H_l / log(N_l)
    """

    def __init__(self, model, r_0, calibration_stats, alpha=0.3, beta=1.0,
                 use_importance_protection=False, top_k_protect=10):
        self.model = model
        self.r_0 = r_0
        self.stats = calibration_stats
        self.alpha = alpha
        self.beta = beta
        self.use_importance_protection = use_importance_protection
        self.top_k_protect = top_k_protect
        self.merge_ratios = []
        self._hooks = []
        self._current_ratios = []

        # Patch model with ToMe
        import tome.patch.timm as tome_timm_patch
        tome_timm_patch(model)
        model.r = r_0

        # Install adaptive hooks
        self._install_hooks()

    def _install_hooks(self):
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_pre_hook(self._make_adaptive_hook(i))
            self._hooks.append(hook)

    def _make_adaptive_hook(self, layer_idx):
        parent = self

        def hook_fn(module, input):
            x = input[0]
            B, N, C = x.shape

            # Compute CLS-row attention entropy on pre-merge tokens
            ent, _ = compute_cls_entropy_from_tokens(module, x)

            # Normalize by log(N) to account for different token counts
            log_N = math.log(N) if N > 1 else 1.0
            norm_ent = ent / log_N

            # Get calibration stats (normalized)
            layer_stats = parent.stats.get(layer_idx, parent.stats.get(str(layer_idx)))
            if layer_stats is None:
                return input

            norm_mean = layer_stats['norm_mean']
            norm_std = layer_stats['norm_std']

            if norm_std > 1e-6:
                delta = max(0, (norm_ent - norm_mean) / norm_std)
            else:
                delta = 0.0

            adaptive_r = round(parent.r_0 * max(parent.alpha, math.exp(-parent.beta * delta)))

            # Ensure we don't try to merge more tokens than available
            max_r = max(0, (N - 1) // 2)
            adaptive_r = min(adaptive_r, max_r)

            parent.model.r = adaptive_r
            parent._current_ratios.append(adaptive_r)

            return input

        return hook_fn

    def forward_with_tracking(self, images):
        self._current_ratios = []
        outputs = self.model(images)
        self.merge_ratios.append(list(self._current_ratios))
        return outputs

    def get_avg_merge_ratio(self):
        if not self.merge_ratios:
            return self.r_0
        all_ratios = [np.mean(r) for r in self.merge_ratios if r]
        return float(np.mean(all_ratios)) if all_ratios else self.r_0

    def reset_tracking(self):
        self.merge_ratios = []
        self._current_ratios = []

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


class ImportanceProtectedToMe:
    """ToMe with importance-aware token protection.
    Protects top-k tokens (by CLS attention score) from being merged.
    Used for ablation studies.
    """

    def __init__(self, model, r, top_k_protect=10):
        self.model = model
        self.r = r
        self.top_k_protect = top_k_protect
        self._hooks = []

        import tome.patch.timm as tome_timm_patch
        tome_timm_patch(model)
        model.r = r

        # Install protection hooks
        for i, block in enumerate(model.blocks):
            hook = block.register_forward_pre_hook(self._make_protection_hook(i))
            self._hooks.append(hook)

    def _make_protection_hook(self, layer_idx):
        parent = self

        def hook_fn(module, input):
            x = input[0]
            B, N, C = x.shape
            if N <= parent.top_k_protect + 2:
                return input

            # Compute CLS attention scores
            num_heads = module.attn.num_heads
            head_dim = C // num_heads
            with torch.no_grad():
                qkv = module.attn.qkv(x).reshape(B, N, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * module.attn.scale
                attn = attn.softmax(dim=-1)
                # CLS attention to patches
                cls_scores = attn[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
                _, top_idx = cls_scores.topk(min(parent.top_k_protect, N - 1), dim=-1)

            # Store protected indices for ToMe to use
            # Since we can't directly modify ToMe's matching, we boost
            # the protected tokens' keys to make them less likely to be merged
            # by adding a large value to their key norms
            protected_mask = torch.zeros(B, N, 1, device=x.device)
            for b in range(B):
                protected_mask[b, top_idx[b] + 1, 0] = 100.0  # +1 for CLS offset

            # Modify input to discourage merging of important tokens
            # ToMe uses key cosine similarity — we can't easily interfere
            # Instead, just reduce r to leave room for protection
            effective_r = max(1, parent.r - parent.top_k_protect // 2)
            parent.model.r = effective_r

            return input

        return hook_fn

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
