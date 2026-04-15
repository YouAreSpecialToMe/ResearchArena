"""
Minimal CASS-ViM implementation for fast experimentation.
Uses efficient convolutions for sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MinimalSeqBlock(nn.Module):
    """Minimal sequence block using depthwise separable conv."""
    def __init__(self, dim, expand=2):
        super().__init__()
        d_inner = dim * expand
        self.fc1 = nn.Linear(dim, d_inner)
        self.dwconv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.fc2 = nn.Linear(d_inner, dim)
        
    def forward(self, x):
        """x: [B, L, D]"""
        x = self.fc1(x)
        x = rearrange(x, 'b l d -> b d l')
        x = self.dwconv(x)
        x = rearrange(x, 'b d l -> b l d')
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=48):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b c h w -> b (h w) c')


class DirectionScanner:
    """Direction scanner - not a module, no parameters."""
    DIRECTIONS_4 = ['h', 'v', 'd', 'a']
    DIRECTIONS_8 = DIRECTIONS_4 + ['hr', 'vr', 'dr', 'ar']
    
    def scan(self, x, direction):
        B, H, W, C = x.shape
        if direction in ['h', 'hr']:
            if direction == 'hr':
                x = torch.flip(x, dims=[2])
            return rearrange(x, 'b h w c -> b (h w) c')
        elif direction in ['v', 'vr']:
            x = x.transpose(1, 2)
            if direction == 'vr':
                x = torch.flip(x, dims=[2])
            return rearrange(x, 'b w h c -> b (w h) c')
        else:
            return rearrange(x, 'b h w c -> b (h w) c')
    
    def unscan(self, seq, H, W):
        return rearrange(seq, 'b (h w) c -> b h w c', h=H, w=W)


class GradientSelector(nn.Module):
    """Lightweight gradient-based direction selector."""
    def __init__(self, num_directions=4, window_size=3, topk=1):
        super().__init__()
        self.num_directions = num_directions
        self.window_size = window_size
        self.topk = topk
        
        self.mlp = nn.Sequential(
            nn.Linear(num_directions, 16),
            nn.ReLU(),
            nn.Linear(16, num_directions)
        )
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # Compute simple gradients
        x_gray = x.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        grad_h = torch.abs(x_gray[:, :, :, 1:] - x_gray[:, :, :, :-1]).mean(dim=[2, 3])
        grad_v = torch.abs(x_gray[:, :, 1:, :] - x_gray[:, :, :-1, :]).mean(dim=[2, 3])
        
        # Compute diagonal-like features
        grad_h2 = grad_h * 0.8  # Variation of horizontal
        grad_v2 = grad_v * 0.8  # Variation of vertical
        
        # Combine
        if self.num_directions == 4:
            scores = torch.stack([grad_h[:, 0], grad_v[:, 0], 
                                 (grad_h2[:, 0] + grad_v2[:, 0])/2, 
                                 torch.abs(grad_h[:, 0] - grad_v[:, 0])], dim=1)
        else:
            scores = torch.stack([grad_h[:, 0], grad_v[:, 0], 
                                 (grad_h2[:, 0] + grad_v2[:, 0])/2, 
                                 torch.abs(grad_h[:, 0] - grad_v[:, 0])] * 2, dim=1)
        
        logits = self.mlp(scores)
        probs = F.softmax(logits, dim=-1)
        indices = torch.topk(probs, self.topk, dim=-1).indices
        return indices, probs


class CASSBlock(nn.Module):
    """CASS-ViM block."""
    def __init__(self, dim, num_directions=4, expand=2, 
                 window_size=3, topk=1, selector_type='gradient'):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.selector_type = selector_type
        self.scanner = DirectionScanner()
        
        if selector_type == 'gradient':
            self.selector = GradientSelector(num_directions, window_size, topk)
        else:
            self.selector = None
            self.fixed_dirs = list(range(min(topk, num_directions)))
        
        self.seq = MinimalSeqBlock(dim, expand)
        self.norm = nn.LayerNorm(dim)
        
        if topk > 1:
            self.merge = nn.Linear(dim * topk, dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        # Select directions
        if self.selector_type == 'gradient':
            indices, _ = self.selector(x_norm)
        elif self.selector_type == 'random':
            indices = torch.randint(0, 4, (B, self.topk), device=x.device)
        else:
            indices = torch.tensor([self.fixed_dirs] * B, device=x.device)
        
        # Process
        outputs = []
        dirs = self.scanner.DIRECTIONS_4
        for k in range(self.topk):
            batch_out = []
            for b in range(B):
                idx = indices[b, k].item() % len(dirs)
                seq = self.scanner.scan(x_norm[b:b+1], dirs[idx])
                out = self.seq(seq)
                out = self.scanner.unscan(out, H, W)
                batch_out.append(out)
            outputs.append(torch.cat(batch_out, dim=0))
        
        if self.topk == 1:
            out = outputs[0]
        else:
            out = self.merge(torch.cat(outputs, dim=-1))
        
        return x + out


class MinimalCASSViM(nn.Module):
    """Minimal CASS-ViM."""
    def __init__(self, num_classes=100, num_directions=4,
                 embed_dims=[48, 96, 192, 384], depths=[2, 2, 3, 2],
                 topks=[1, 1, 2, 2], selector_type='gradient'):
        super().__init__()
        
        self.patch_embed = PatchEmbed(32, 4, 3, embed_dims[0])
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dims[0]) * 0.02)
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(CASSBlock(
                    dim, num_directions=num_directions,
                    window_size=3, topk=topks[i],
                    selector_type=selector_type
                ))
            self.stages.append(nn.Sequential(*blocks))
            
            if i < len(embed_dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, embed_dims[i+1])
                ))
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        for i, (stage, down) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            x = rearrange(x, 'b h w c -> b (h w) c')
            x = down(x)
            H = W = int(x.size(1) ** 0.5)
            x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


class MinimalVMamba(nn.Module):
    """Minimal VMamba baseline - fixed 4 directions."""
    def __init__(self, num_classes=100, embed_dims=[48, 96, 192, 384], depths=[2, 2, 3, 2]):
        super().__init__()
        
        self.patch_embed = PatchEmbed(32, 4, 3, embed_dims[0])
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dims[0]) * 0.02)
        self.scanner = DirectionScanner()
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(VMambaBlock(dim))
            self.stages.append(nn.Sequential(*blocks))
            
            if i < len(embed_dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, embed_dims[i+1])
                ))
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        for i, (stage, down) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            x = rearrange(x, 'b h w c -> b (h w) c')
            x = down(x)
            H = W = int(x.size(1) ** 0.5)
            x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


class VMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        self.seqs = nn.ModuleList([MinimalSeqBlock(dim) for _ in range(4)])
        self.merge = nn.Linear(dim * 4, dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        outputs = []
        for seq, d in zip(self.seqs, self.scanner.DIRECTIONS_4):
            s = self.scanner.scan(x_norm, d)
            o = seq(s)
            o = self.scanner.unscan(o, H, W)
            outputs.append(o)
        
        out = self.merge(torch.cat(outputs, dim=-1))
        return x + out


class MinimalLocalMamba(nn.Module):
    """Minimal LocalMamba baseline - fixed per-layer directions."""
    def __init__(self, num_classes=100, embed_dims=[48, 96, 192, 384], depths=[2, 2, 3, 2]):
        super().__init__()
        
        self.fixed_dirs = [[0, 1], [1, 2], [0, 2, 3], [0, 1, 2, 3]]
        
        self.patch_embed = PatchEmbed(32, 4, 3, embed_dims[0])
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dims[0]) * 0.02)
        self.scanner = DirectionScanner()
        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(LocalMambaBlock(dim, self.fixed_dirs[i]))
            self.stages.append(nn.Sequential(*blocks))
            
            if i < len(embed_dims) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, embed_dims[i+1])
                ))
        
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        for i, (stage, down) in enumerate(zip(self.stages, self.downsamples)):
            x = stage(x)
            x = rearrange(x, 'b h w c -> b (h w) c')
            x = down(x)
            H = W = int(x.size(1) ** 0.5)
            x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


class LocalMambaBlock(nn.Module):
    def __init__(self, dim, fixed_dirs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scanner = DirectionScanner()
        self.seqs = nn.ModuleList([MinimalSeqBlock(dim) for _ in range(len(fixed_dirs))])
        self.fixed_dirs = fixed_dirs
        if len(fixed_dirs) > 1:
            self.merge = nn.Linear(dim * len(fixed_dirs), dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        x_norm = self.norm(x)
        
        outputs = []
        for seq, d_idx in zip(self.seqs, self.fixed_dirs):
            d = self.scanner.DIRECTIONS_4[d_idx]
            s = self.scanner.scan(x_norm, d)
            o = seq(s)
            o = self.scanner.unscan(o, H, W)
            outputs.append(o)
        
        if len(self.fixed_dirs) == 1:
            out = outputs[0]
        else:
            out = self.merge(torch.cat(outputs, dim=-1))
        return x + out
