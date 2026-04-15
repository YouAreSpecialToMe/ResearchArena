"""
Optimized CASS-ViM implementation - v2 with smaller, faster models.
Designed for 25-30 minute training on CIFAR-100.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EfficientSeqBlock(nn.Module):
    """Efficient sequence block using 1D convolutions."""
    def __init__(self, dim, expand=2):
        super().__init__()
        d_inner = dim * expand
        self.fc1 = nn.Linear(dim, d_inner)
        # Use 1D conv for local interactions - much faster than GRU for short sequences
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.fc2 = nn.Linear(d_inner, dim)
        
    def forward(self, x):
        """x: [B, L, D]"""
        x = self.fc1(x)
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv(x)
        x = rearrange(x, 'b d l -> b l d')
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=48):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b c h w -> b (h w) c')


class DirectionScanner:
    """Batched direction scanner."""
    DIRECTIONS_4 = ['h', 'v', 'd', 'a']  # horizontal, vertical, diag, anti-diag
    DIRECTIONS_8 = DIRECTIONS_4 + ['hr', 'vr', 'dr', 'ar']  # + reversed
    
    def scan(self, x, direction):
        """Batched scan: x is [B, H, W, C]"""
        B, H, W, C = x.shape
        
        if direction == 'h':
            return rearrange(x, 'b h w c -> b (h w) c')
        elif direction == 'hr':
            x = torch.flip(x, dims=[2])
            return rearrange(x, 'b h w c -> b (h w) c')
        elif direction == 'v':
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'vr':
            x = torch.flip(x, dims=[1])
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'd':
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'dr':
            x = torch.flip(x, dims=[2])
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'a':
            x = torch.flip(x, dims=[2])
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        elif direction == 'ar':
            x = x.transpose(1, 2)
            return rearrange(x, 'b w h c -> b (w h) c')
        else:
            return rearrange(x, 'b h w c -> b (h w) c')
    
    def unscan(self, seq, H, W):
        """Batched unscan: seq is [B, L, C]"""
        return rearrange(seq, 'b (h w) c -> b h w c', h=H, w=W)


class SimpleGradientSelector(nn.Module):
    """Simplified gradient-based direction selector."""
    def __init__(self, num_directions=4, window_size=3, topk=1):
        super().__init__()
        self.num_directions = num_directions
        self.window_size = window_size
        self.topk = topk
        
        # Fixed kernels for gradient computation
        self.register_buffer('kernel_h', torch.tensor([[-1., 0., 1.]]).view(1, 1, 1, 3))
        self.register_buffer('kernel_v', torch.tensor([[-1.], [0.], [1.]]).view(1, 1, 3, 1))
        
        # Small MLP for scoring
        self.mlp = nn.Sequential(
            nn.Linear(num_directions, 16),
            nn.ReLU(),
            nn.Linear(16, num_directions)
        )
        
    def forward(self, x):
        """Batched direction selection. x: [B, H, W, C]"""
        B, H, W, C = x.shape
        
        # Convert to grayscale [B, 1, H, W]
        x_gray = x.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        
        # Pad for convolution
        x_pad = F.pad(x_gray, (1, 1, 1, 1), mode='reflect')
        
        # Compute gradients
        grad_h = torch.abs(F.conv2d(x_pad, self.kernel_h.expand(1, 1, -1, -1)))
        grad_v = torch.abs(F.conv2d(x_pad, self.kernel_v.expand(1, 1, -1, -1)))
        
        # Global average pooling
        score_h = grad_h.mean(dim=[2, 3])
        score_v = grad_v.mean(dim=[2, 3])
        
        # Derived scores
        score_d = (score_h + score_v) / 2  # Diagonal
        score_a = torch.abs(score_h - score_v)  # Anti-diagonal
        
        if self.num_directions == 4:
            scores = torch.cat([score_h, score_v, score_d, score_a], dim=1)
        else:
            # For 8 directions, duplicate
            scores = torch.cat([score_h, score_v, score_d, score_a] * 2, dim=1)
        
        logits = self.mlp(scores)
        probs = F.softmax(logits, dim=-1)
        selected = torch.topk(probs, self.topk, dim=-1).indices
        
        return selected, probs


class CASSBlock(nn.Module):
    """CASS block with efficient batched processing."""
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
            self.direction_selector = SimpleGradientSelector(
                num_directions=num_directions, window_size=window_size, topk=topk
            )
        else:
            self.direction_selector = None
            if selector_type == 'fixed':
                self.fixed_directions = list(range(min(topk, num_directions)))
        
        # Shared sequence model for all directions
        self.seq_model = EfficientSeqBlock(dim, expand=expand)
        
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
        
        # Process directions
        dirs = self.scanner.DIRECTIONS_4 if self.num_directions == 4 else self.scanner.DIRECTIONS_8
        
        all_outputs = []
        for k in range(self.topk):
            dir_indices = selected_indices[:, k]  # [B]
            
            # Process each sample based on its selected direction
            # For efficiency, we group by direction
            output = torch.zeros_like(x_norm)
            for dir_idx in range(self.num_directions):
                mask = (dir_indices == dir_idx)
                if mask.any():
                    direction = dirs[dir_idx % len(dirs)]
                    x_subset = x_norm[mask]
                    seq = self.scanner.scan(x_subset, direction)
                    processed = self.seq_model(seq)
                    out = self.scanner.unscan(processed, H, W)
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


class CASSViMTiny(nn.Module):
    """Tiny CASS-ViM for fast training on CIFAR-100."""
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
                    CASSBlock(
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


class VMambaBlock(nn.Module):
    """VMamba block with fixed 4 directions."""
    def __init__(self, dim, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        self.seq_models = nn.ModuleList([
            EfficientSeqBlock(dim, expand=expand) for _ in range(4)
        ])
        
        self.direction_merge = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        direction_outputs = []
        for ssm, direction in zip(self.seq_models, self.scanner.DIRECTIONS_4):
            seq = self.scanner.scan(x_norm, direction)
            processed = ssm(seq)
            out = self.scanner.unscan(processed, H, W)
            direction_outputs.append(out)
        
        merged = torch.cat(direction_outputs, dim=-1)
        output = self.direction_merge(merged)
        
        return x + output


class VMambaTiny(nn.Module):
    """Tiny VMamba baseline with fixed 4-direction scanning."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2], expand=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dims[0]) * 0.02)
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(VMambaBlock(dim=dim, expand=expand))
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
    def __init__(self, dim, expand=2, fixed_directions=[0, 1]):
        super().__init__()
        self.dim = dim
        self.fixed_directions = fixed_directions
        self.num_dirs = len(fixed_directions)
        
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        
        self.seq_models = nn.ModuleList([
            EfficientSeqBlock(dim, expand=expand) for _ in range(self.num_dirs)
        ])
        
        if self.num_dirs > 1:
            self.direction_merge = nn.Linear(dim * self.num_dirs, dim)
    
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        direction_outputs = []
        for ssm, dir_idx in zip(self.seq_models, self.fixed_directions):
            direction = self.scanner.DIRECTIONS_4[dir_idx]
            seq = self.scanner.scan(x_norm, direction)
            processed = ssm(seq)
            out = self.scanner.unscan(processed, H, W)
            direction_outputs.append(out)
        
        if self.num_dirs == 1:
            output = direction_outputs[0]
        else:
            merged = torch.cat(direction_outputs, dim=-1)
            output = self.direction_merge(merged)
        
        return x + output


class LocalMambaTiny(nn.Module):
    """Tiny LocalMamba baseline with fixed per-layer directions."""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=100,
                 embed_dims=[32, 64, 128, 256], depths=[2, 2, 2, 2], expand=2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Fixed directions per stage (simulating DARTS search result)
        self.fixed_directions_per_stage = [
            [0, 1], [1, 2], [0, 2, 3], [0, 1, 2, 3]
        ]
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dims[0])
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dims[0]) * 0.02)
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    LocalMambaBlock(
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
