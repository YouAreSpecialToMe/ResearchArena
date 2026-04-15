"""
CASS-ViM: Content-Adaptive Selective Scanning for Vision State Space Models
Lightweight implementation using PyTorch native operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from einops import rearrange


class SimplifiedSSM(nn.Module):
    """
    Simplified SSM module using efficient gating mechanism.
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        # Conv1d for local context
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        
        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_inner, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        
    def forward(self, x):
        """
        Args:
            x: [B, L, D] where L is sequence length
        """
        batch, seq_len, dim = x.shape
        
        # Input projection
        xz = self.in_proj(x)
        x_conv, z = xz.chunk(2, dim=-1)
        
        # Conv1d
        x_conv = rearrange(x_conv, 'b l d -> b d l')
        x_conv = self.conv(x_conv)
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # Simplified SSM with selective gating
        x_ssm = self.x_proj(x_conv)
        B_ssm, C_ssm = x_ssm.chunk(2, dim=-1)
        delta = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log.float())
        
        # Fast selective scan approximation
        y = self.fast_selective_scan(x_conv, delta, A, B_ssm, C_ssm)
        y = y * F.silu(z)
        
        return self.out_proj(y)
    
    def fast_selective_scan(self, x, delta, A, B, C):
        """Optimized selective scan using parallel scan."""
        batch, seq_len, d_inner = x.shape
        d_state = A.size(1)
        
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        deltaB_x = torch.einsum('bld,bln,bld->bldn', delta, B, x)
        
        # Parallel scan (simplified)
        hs = []
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            hs.append(y)
        
        y = torch.stack(hs, dim=1)
        return y + self.D * x


class PatchEmbed(nn.Module):
    """Patch embedding layer."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class DirectionScanner(nn.Module):
    """Base class for scanning images in different directions."""
    DIRECTIONS_4 = ['horizontal', 'vertical', 'diag_down', 'diag_up']
    DIRECTIONS_8 = DIRECTIONS_4 + ['horizontal_rev', 'vertical_rev', 'diag_down_rev', 'diag_up_rev']
    
    def __init__(self, img_size=32, patch_size=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_side = img_size // patch_size
        
    def scan(self, x, direction):
        """Scan patches in the specified direction."""
        B, H, W, C = x.shape
        
        if direction == 'horizontal':
            return rearrange(x, 'b h w c -> b (h w) c')
        elif direction == 'horizontal_rev':
            x = torch.flip(x, dims=[2])
            return rearrange(x, 'b h w c -> b (h w) c')
        elif direction == 'vertical':
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'vertical_rev':
            x = torch.flip(x, dims=[1])
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction in ['diag_down', 'diag_up', 'diag_down_rev', 'diag_up_rev']:
            # For diagonal scans, use flattened version as approximation
            return rearrange(x, 'b h w c -> b (h w) c')
        else:
            return rearrange(x, 'b h w c -> b (h w) c')
    
    def unscan(self, seq, direction, H, W):
        """Reverse the scanning process."""
        return rearrange(seq, 'b (h w) c -> b h w c', h=H, w=W)


class GradientDirectionSelector(nn.Module):
    """Gradient-based direction selection module."""
    def __init__(self, num_directions=4, window_sizes=[3, 3, 5, 7], 
                 topk=[1, 1, 2, 2], stage=0):
        super().__init__()
        self.num_directions = num_directions
        self.window_size = window_sizes[stage]
        self.topk = topk[stage]
        
        # Fixed convolution kernels for gradient computation
        self.register_buffer('kernel_h', torch.tensor([[-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 1, 3))
        self.register_buffer('kernel_v', torch.tensor([[-1], [0], [1]], dtype=torch.float32).reshape(1, 1, 3, 1))
        
        # Small MLP for direction scoring
        self.mlp = nn.Sequential(
            nn.Linear(num_directions, 32),
            nn.ReLU(),
            nn.Linear(32, num_directions)
        )
        
    def compute_gradients(self, x):
        """Compute spatial gradients using fixed kernels."""
        B, H, W, C = x.shape
        x_gray = x.mean(dim=-1, keepdim=True)
        x_gray = rearrange(x_gray, 'b h w c -> b c h w')
        x_padded = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        
        grad_h = torch.abs(F.conv2d(x_padded, self.kernel_h.expand(1, 1, -1, -1)))
        grad_v = torch.abs(F.conv2d(x_padded, self.kernel_v.expand(1, 1, -1, -1)))
        
        # Ensure same spatial dimensions
        if grad_h.shape != grad_v.shape:
            grad_h = F.interpolate(grad_h, size=(H, W), mode='bilinear', align_corners=False)
            grad_v = F.interpolate(grad_v, size=(H, W), mode='bilinear', align_corners=False)
        
        # Approximate diagonal gradients
        grad_diag = (grad_h + grad_v) / 2
        grad_antidiag = torch.abs(grad_h - grad_v)
        
        return {
            'horizontal': grad_h,
            'vertical': grad_v,
            'diag_down': grad_diag,
            'diag_up': grad_antidiag
        }
    
    def aggregate_scores(self, gradients):
        """Aggregate gradient magnitudes over local windows."""
        scores = {}
        for direction, grad in gradients.items():
            pooled = F.avg_pool2d(grad, self.window_size, stride=1, 
                                 padding=self.window_size//2)
            scores[direction] = pooled.mean(dim=[2, 3])
        return scores
    
    def forward(self, x):
        """Select directions based on gradient analysis."""
        B = x.size(0)
        gradients = self.compute_gradients(x)
        scores = self.aggregate_scores(gradients)
        
        if self.num_directions == 4:
            score_tensor = torch.cat([
                scores['horizontal'], scores['vertical'],
                scores['diag_down'], scores['diag_up']
            ], dim=1)
        else:
            score_tensor = torch.cat([
                scores['horizontal'], scores['vertical'],
                scores['diag_down'], scores['diag_up'],
                scores['horizontal'], scores['vertical'],
                scores['diag_down'], scores['diag_up']
            ], dim=1)
        
        logits = self.mlp(score_tensor)
        direction_probs = F.softmax(logits, dim=-1)
        selected_indices = torch.topk(direction_probs, self.topk, dim=-1).indices
        
        return selected_indices, direction_probs


class CASSViMBlock(nn.Module):
    """CASS-ViM block with content-adaptive scanning."""
    def __init__(self, dim, num_directions=4, d_state=16, expand=2,
                 window_size=3, topk=1, stage=0, selector_type='gradient'):
        super().__init__()
        self.dim = dim
        self.num_directions = num_directions
        self.topk = topk
        self.selector_type = selector_type
        
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        # Direction selector
        if selector_type == 'gradient':
            self.direction_selector = GradientDirectionSelector(
                num_directions=num_directions,
                window_sizes=[window_size] * 4,
                topk=[topk] * 4,
                stage=stage
            )
        else:
            self.direction_selector = None
            if selector_type == 'fixed':
                self.fixed_directions = list(range(min(topk, num_directions)))
        
        # SSM for each direction
        self.ssm = SimplifiedSSM(dim, d_state=d_state, expand=expand)
        
        if topk > 1:
            self.direction_merge = nn.Linear(dim * topk, dim)
        
    def forward(self, x, return_directions=False):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        # Select directions
        if self.selector_type == 'gradient':
            selected_indices, dir_probs = self.direction_selector(x_norm)
        elif self.selector_type == 'random':
            selected_indices = torch.randint(0, self.num_directions, (B, self.topk), device=x.device)
            dir_probs = None
        elif self.selector_type == 'fixed':
            selected_indices = torch.tensor([self.fixed_directions] * B, device=x.device)
            dir_probs = None
        else:
            selected_indices = torch.zeros(B, self.topk, device=x.device, dtype=torch.long)
            dir_probs = None
        
        # Process each selected direction
        direction_outputs = []
        for k in range(self.topk):
            outputs_batch = []
            for b in range(B):
                dir_idx = selected_indices[b, k].item()
                dirs = self.scanner.DIRECTIONS_4 if self.num_directions == 4 else self.scanner.DIRECTIONS_8
                direction = dirs[dir_idx % len(dirs)]
                
                seq = self.scanner.scan(x_norm[b:b+1], direction)
                processed = self.ssm(seq)
                out = self.scanner.unscan(processed, direction, H, W)
                outputs_batch.append(out)
            
            direction_outputs.append(torch.cat(outputs_batch, dim=0))
        
        # Merge directions
        if self.topk == 1:
            output = direction_outputs[0]
        else:
            merged = torch.cat(direction_outputs, dim=-1)
            output = self.direction_merge(merged)
        
        output = x + output
        
        if return_directions:
            return output, selected_indices, dir_probs
        return output


class CASSViM(nn.Module):
    """Full CASS-ViM architecture - Lightweight version."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 4, 2],
                 num_directions=4, d_state=16, expand=2,
                 window_sizes=[3, 3, 5, 7], topks=[1, 1, 2, 2],
                 selector_type='gradient'):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_directions = num_directions
        self.selector_type = selector_type
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]) * 0.02)
        
        # Stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    CASSViMBlock(
                        dim=dim, num_directions=num_directions,
                        d_state=d_state, expand=expand,
                        window_size=window_sizes[i], topk=topks[i],
                        stage=i, selector_type=selector_type
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            
            if i < len(embed_dims) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, embed_dims[i+1])
                    )
                )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x, return_directions=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        all_directions = []
        
        for i, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            for block in stage:
                if return_directions and hasattr(block, 'direction_selector') and block.direction_selector is not None:
                    x, dirs, probs = block(x, return_directions=True)
                    all_directions.append(dirs)
                else:
                    x = block(x)
            
            if i < len(self.downsamples):
                x = rearrange(x, 'b h w c -> b (h w) c')
                x = downsample(x)
                H = W = int(x.size(1) ** 0.5)
                x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.mean(dim=1)
        
        if return_directions:
            return x, all_directions
        return x
    
    def forward(self, x, return_directions=False):
        if return_directions:
            x, dirs = self.forward_features(x, return_directions=True)
            x = self.norm(x)
            logits = self.head(x)
            return logits, dirs
        
        x = self.forward_features(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


class VMamba(nn.Module):
    """VMamba baseline with fixed 4-direction cross-scan."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 4, 2],
                 d_state=16, expand=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]) * 0.02)
        
        self.scanner = DirectionScanner()
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(VMambaBlock(dim=dim, d_state=d_state, expand=expand))
            self.stages.append(nn.Sequential(*stage_blocks))
            
            if i < len(embed_dims) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, embed_dims[i+1])
                    )
                )
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        for i, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            if i < len(self.downsamples):
                x = rearrange(x, 'b h w c -> b (h w) c')
                x = downsample(x)
                H = W = int(x.size(1) ** 0.5)
                x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.mean(dim=1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


class VMambaBlock(nn.Module):
    """VMamba block with fixed 4-direction scanning."""
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        self.ssms = nn.ModuleList([
            SimplifiedSSM(dim, d_state=d_state, expand=expand)
            for _ in range(4)
        ])
        
        self.direction_merge = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        direction_outputs = []
        for i, (ssm, direction) in enumerate(zip(self.ssms, self.scanner.DIRECTIONS_4)):
            seq = self.scanner.scan(x_norm, direction)
            processed = ssm(seq)
            out = self.scanner.unscan(processed, direction, H, W)
            direction_outputs.append(out)
        
        merged = torch.cat(direction_outputs, dim=-1)
        output = self.direction_merge(merged)
        
        return x + output


class LocalMamba(nn.Module):
    """LocalMamba baseline with learned fixed per-layer directions."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[64, 128, 256, 512], depths=[2, 2, 4, 2],
                 d_state=16, expand=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.fixed_directions_per_stage = [
            [0, 1], [1, 2], [0, 2, 3], [0, 1, 2, 3]
        ]
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]) * 0.02)
        
        self.scanner = DirectionScanner()
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    LocalMambaBlock(
                        dim=dim, d_state=d_state, expand=expand,
                        fixed_directions=self.fixed_directions_per_stage[i]
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            
            if i < len(embed_dims) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, embed_dims[i+1])
                    )
                )
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        for i, (stage, downsample) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            if i < len(self.downsamples):
                x = rearrange(x, 'b h w c -> b (h w) c')
                x = downsample(x)
                H = W = int(x.size(1) ** 0.5)
                x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.mean(dim=1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


class LocalMambaBlock(nn.Module):
    """LocalMamba block with fixed per-layer directions."""
    def __init__(self, dim, d_state=16, expand=2, fixed_directions=[0, 1]):
        super().__init__()
        self.dim = dim
        self.fixed_directions = fixed_directions
        self.num_dirs = len(fixed_directions)
        
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        self.ssms = nn.ModuleList([
            SimplifiedSSM(dim, d_state=d_state, expand=expand)
            for _ in range(self.num_dirs)
        ])
        
        if self.num_dirs > 1:
            self.direction_merge = nn.Linear(dim * self.num_dirs, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        direction_outputs = []
        for ssm, dir_idx in zip(self.ssms, self.fixed_directions):
            direction = self.scanner.DIRECTIONS_4[dir_idx]
            seq = self.scanner.scan(x_norm, direction)
            processed = ssm(seq)
            out = self.scanner.unscan(processed, direction, H, W)
            direction_outputs.append(out)
        
        if self.num_dirs == 1:
            output = direction_outputs[0]
        else:
            merged = torch.cat(direction_outputs, dim=-1)
            output = self.direction_merge(merged)
        
        return x + output
