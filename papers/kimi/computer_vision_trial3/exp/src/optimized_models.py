"""
Optimized CASS-ViM implementation with fully batched operations.
Eliminates per-sample loops for efficient GPU utilization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class OptimizedSequenceBlock(nn.Module):
    """Optimized sequence modeling using GRU."""
    def __init__(self, dim, expand=2):
        super().__init__()
        d_inner = dim * expand
        self.in_proj = nn.Linear(dim, d_inner)
        self.gru = nn.GRU(d_inner, d_inner // 2, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(d_inner, dim)
        
    def forward(self, x):
        x = self.in_proj(x)
        x, _ = self.gru(x)
        x = self.out_proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b c h w -> b (h w) c')


class DirectionScanner:
    """Batched direction scanner."""
    DIRECTIONS_4 = ['horizontal', 'vertical', 'diag_down', 'diag_up']
    DIRECTIONS_8 = DIRECTIONS_4 + ['horizontal_rev', 'vertical_rev', 'diag_down_rev', 'diag_up_rev']
    
    def scan_batched(self, x, direction):
        """Batched scan: x is [B, H, W, C]"""
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
        elif direction == 'diag_down':
            # Diagonal: transpose then horizontal
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'diag_up':
            # Anti-diagonal: flip then transpose then horizontal
            x = torch.flip(x, dims=[2])
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        else:
            return rearrange(x, 'b h w c -> b (h w) c')
    
    def unscan_batched(self, seq, H, W):
        """Batched unscan: seq is [B, L, C]"""
        return rearrange(seq, 'b (h w) c -> b h w c', h=H, w=W)


class GradientDirectionSelector(nn.Module):
    """Optimized gradient-based direction selector."""
    def __init__(self, num_directions=4, window_size=3, topk=1):
        super().__init__()
        self.num_directions = num_directions
        self.window_size = window_size
        self.topk = topk
        
        # Fixed gradient kernels
        self.register_buffer('kernel_h', torch.tensor([[-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 1, 3))
        self.register_buffer('kernel_v', torch.tensor([[-1], [0], [1]], dtype=torch.float32).reshape(1, 1, 3, 1))
        
        # Scoring MLP
        self.mlp = nn.Sequential(
            nn.Linear(num_directions, 32),
            nn.ReLU(),
            nn.Linear(32, num_directions)
        )
        
    def forward(self, x):
        """Batched direction selection. x: [B, H, W, C]"""
        B, H, W, C = x.shape
        
        # Convert to grayscale [B, 1, H, W]
        x_gray = x.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        x_padded = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        
        # Compute gradients
        grad_h = torch.abs(F.conv2d(x_padded, self.kernel_h.expand(1, 1, -1, -1)))
        grad_v = torch.abs(F.conv2d(x_padded, self.kernel_v.expand(1, 1, -1, -1)))
        
        # Resize to match if needed
        if grad_h.shape[2:] != (H, W):
            grad_h = F.interpolate(grad_h, size=(H, W), mode='bilinear', align_corners=False)
            grad_v = F.interpolate(grad_v, size=(H, W), mode='bilinear', align_corners=False)
        
        # Pool gradients
        pool_size = self.window_size
        grad_h_pooled = F.avg_pool2d(grad_h, pool_size, stride=1, padding=pool_size//2)
        grad_v_pooled = F.avg_pool2d(grad_v, pool_size, stride=1, padding=pool_size//2)
        
        # Aggregate per-sample scores
        score_h = grad_h_pooled.mean(dim=[2, 3])
        score_v = grad_v_pooled.mean(dim=[2, 3])
        score_diag = (score_h + score_v) / 2
        score_antidiag = torch.abs(score_h - score_v)
        
        if self.num_directions == 4:
            scores = torch.cat([score_h, score_v, score_diag, score_antidiag], dim=1)
        else:
            # For 8 directions, duplicate the scores
            scores = torch.cat([score_h, score_v, score_diag, score_antidiag,
                               score_h, score_v, score_diag, score_antidiag], dim=1)
        
        logits = self.mlp(scores)
        probs = F.softmax(logits, dim=-1)
        selected = torch.topk(probs, self.topk, dim=-1).indices
        
        return selected, probs


class OptimizedCASSBlock(nn.Module):
    """Optimized CASS block with batched direction processing."""
    def __init__(self, dim, num_directions=4, expand=2, 
                 window_size=3, topk=1, selector_type='gradient'):
        super().__init__()
        self.dim = dim
        self.num_directions = num_directions
        self.topk = topk
        self.selector_type = selector_type
        
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        if selector_type == 'gradient':
            self.direction_selector = GradientDirectionSelector(
                num_directions=num_directions, window_size=window_size, topk=topk
            )
        else:
            self.direction_selector = None
            if selector_type == 'fixed':
                self.fixed_directions = list(range(min(topk, num_directions)))
        
        # Shared sequence model for all directions
        self.seq_model = OptimizedSequenceBlock(dim, expand=expand)
        
        if topk > 1:
            self.direction_merge = nn.Linear(dim * topk, dim)
    
    def forward(self, x, return_directions=False):
        """Fully batched forward pass."""
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
        
        # Process all directions in a batched manner
        dirs = self.scanner.DIRECTIONS_4 if self.num_directions == 4 else self.scanner.DIRECTIONS_8
        
        all_outputs = []
        for k in range(self.topk):
            dir_indices = selected_indices[:, k]  # [B]
            
            # Group samples by direction for efficiency
            output = torch.zeros_like(x_norm)
            for dir_idx in range(self.num_directions):
                mask = (dir_indices == dir_idx)
                if mask.any():
                    direction = dirs[dir_idx % len(dirs)]
                    x_subset = x_norm[mask]
                    seq = self.scanner.scan_batched(x_subset, direction)
                    processed = self.seq_model(seq)
                    out = self.scanner.unscan_batched(processed, H, W)
                    output[mask] = out
            
            all_outputs.append(output)
        
        # Merge
        if self.topk == 1:
            output = all_outputs[0]
        else:
            merged = torch.cat(all_outputs, dim=-1)
            output = self.direction_merge(merged)
        
        output = x + output
        
        if return_directions:
            return output, selected_indices, dir_probs
        return output


class OptimizedCASSViM(nn.Module):
    """Optimized CASS-ViM with fully batched operations."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2],
                 num_directions=4, expand=2,
                 window_sizes=[3, 3, 5, 5], topks=[1, 1, 2, 2],
                 selector_type='gradient'):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_directions = num_directions
        self.selector_type = selector_type
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dims[0]) * 0.02)
        
        # Stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    OptimizedCASSBlock(
                        dim=dim, num_directions=num_directions, expand=expand,
                        window_size=window_sizes[i], topk=topks[i],
                        selector_type=selector_type
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


class OptimizedVMamba(nn.Module):
    """Optimized VMamba baseline with fixed 4-direction scanning."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2], expand=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dims[0]) * 0.02)
        
        self.scanner = DirectionScanner()
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(OptimizedVMambaBlock(dim=dim, expand=expand))
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


class OptimizedVMambaBlock(nn.Module):
    """Optimized VMamba block with batched fixed 4-direction scanning."""
    def __init__(self, dim, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        # Use separate sequence models for each direction
        self.seq_models = nn.ModuleList([
            OptimizedSequenceBlock(dim, expand=expand) for _ in range(4)
        ])
        
        self.direction_merge = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        direction_outputs = []
        for ssm, direction in zip(self.seq_models, self.scanner.DIRECTIONS_4):
            seq = self.scanner.scan_batched(x_norm, direction)
            processed = ssm(seq)
            out = self.scanner.unscan_batched(processed, H, W)
            direction_outputs.append(out)
        
        merged = torch.cat(direction_outputs, dim=-1)
        output = self.direction_merge(merged)
        
        return x + output


class OptimizedLocalMamba(nn.Module):
    """Optimized LocalMamba baseline with fixed per-layer directions."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2], expand=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.fixed_directions_per_stage = [
            [0, 1], [1, 2], [0, 2, 3], [0, 1, 2, 3]
        ]
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dims[0]) * 0.02)
        
        self.scanner = DirectionScanner()
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    OptimizedLocalMambaBlock(
                        dim=dim, expand=expand,
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


class OptimizedLocalMambaBlock(nn.Module):
    """Optimized LocalMamba block with fixed per-layer directions."""
    def __init__(self, dim, expand=2, fixed_directions=[0, 1]):
        super().__init__()
        self.dim = dim
        self.fixed_directions = fixed_directions
        self.num_dirs = len(fixed_directions)
        
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        self.seq_models = nn.ModuleList([
            OptimizedSequenceBlock(dim, expand=expand) for _ in range(self.num_dirs)
        ])
        
        if self.num_dirs > 1:
            self.direction_merge = nn.Linear(dim * self.num_dirs, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        direction_outputs = []
        for ssm, dir_idx in zip(self.seq_models, self.fixed_directions):
            direction = self.scanner.DIRECTIONS_4[dir_idx]
            seq = self.scanner.scan_batched(x_norm, direction)
            processed = ssm(seq)
            out = self.scanner.unscan_batched(processed, H, W)
            direction_outputs.append(out)
        
        if self.num_dirs == 1:
            output = direction_outputs[0]
        else:
            merged = torch.cat(direction_outputs, dim=-1)
            output = self.direction_merge(merged)
        
        return x + output
