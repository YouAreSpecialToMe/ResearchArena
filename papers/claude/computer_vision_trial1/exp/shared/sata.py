"""SATA: Spatial Autocorrelation Token Analysis baseline.

Simplified re-implementation of Nikzad et al., CVPR 2025.
Uses spatial autocorrelation to smooth token features before FFN via
local spatial averaging weighted by cosine similarity.

Simplifications vs. original paper:
- Uses cosine-similarity-weighted local averaging instead of full Moran's I + grouping
- Averages with 4-connected spatial neighbors above a similarity threshold
- Fully vectorized (no per-sample Python loops) for GPU efficiency

This is a training-free method that does NOT reduce token count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SATAPatcher:
    """Patches a ViT with SATA-style spatial smoothing before each block."""

    def __init__(self, model, sim_threshold=0.7, grid_size=14):
        self.model = model
        self.sim_threshold = sim_threshold
        self.grid_size = grid_size
        self._hooks = []
        self._neighbor_offsets = None
        self._patch()

    def _patch(self):
        for i, block in enumerate(self.model.blocks):
            hook = block.register_forward_pre_hook(self._make_sata_hook(i))
            self._hooks.append(hook)

    def _make_sata_hook(self, layer_idx):
        parent = self

        def hook_fn(module, input):
            x = input[0]  # [B, N, C]
            B, N, C = x.shape

            num_patches = N - 1
            H = W = int(math.sqrt(num_patches))
            if H * W != num_patches or num_patches < 4:
                return input

            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]  # [B, H*W, C]

            # Reshape to spatial grid
            grid = patch_tokens.reshape(B, H, W, C)

            # Compute similarity-weighted spatial average using 2D convolution-like approach
            # Pad grid for neighbor access
            padded = F.pad(grid.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=0)
            # padded: [B, C, H+2, W+2]

            # Extract center and 4 neighbors
            center = grid.permute(0, 3, 1, 2)  # [B, C, H, W]
            up    = padded[:, :, 0:H, 1:W+1]   # [B, C, H, W]
            down  = padded[:, :, 2:H+2, 1:W+1]
            left  = padded[:, :, 1:H+1, 0:W]
            right = padded[:, :, 1:H+1, 2:W+2]

            neighbors = torch.stack([up, down, left, right], dim=2)  # [B, C, 4, H, W]

            # Compute cosine similarity with center
            center_norm = F.normalize(center, dim=1)  # [B, C, H, W]
            neighbors_norm = F.normalize(neighbors, dim=1)  # [B, C, 4, H, W]

            # Similarity: [B, 4, H, W]
            sim = (center_norm.unsqueeze(2) * neighbors_norm).sum(dim=1)  # [B, 4, H, W]

            # Create validity mask (border pixels have fewer neighbors)
            valid = torch.ones(B, 4, H, W, device=x.device)
            valid[:, 0, 0, :] = 0   # up invalid for top row
            valid[:, 1, -1, :] = 0  # down invalid for bottom row
            valid[:, 2, :, 0] = 0   # left invalid for left col
            valid[:, 3, :, -1] = 0  # right invalid for right col

            # Apply threshold: only average with similar neighbors
            mask = (sim > parent.sim_threshold) * valid  # [B, 4, H, W]

            # Weighted average: center + similar neighbors
            # Weight: 1 for center, mask for each neighbor
            weighted_sum = center.clone()  # [B, C, H, W]
            count = torch.ones(B, 1, H, W, device=x.device)

            for i in range(4):
                w = mask[:, i:i+1, :, :]  # [B, 1, H, W]
                weighted_sum = weighted_sum + neighbors[:, :, i, :, :] * w
                count = count + w

            smoothed = weighted_sum / count.clamp(min=1)  # [B, C, H, W]
            smoothed_flat = smoothed.permute(0, 2, 3, 1).reshape(B, num_patches, C)

            x_new = torch.cat([cls_token, smoothed_flat], dim=1)
            return (x_new,)

        return hook_fn

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
