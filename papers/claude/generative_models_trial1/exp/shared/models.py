"""U-Net architecture for flow matching on CIFAR-10 (32x32)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None, dropout=0.1):
        super().__init__()
        out_channels = out_channels or channels
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip_conv = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        # Reshape to (B, 3*C, H*W) then split
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, heads, d, N)
        scale = (C // self.num_heads) ** -0.5
        # Attention: (B, heads, N, N) = (B, heads, N, d) @ (B, heads, d, N)
        attn = torch.bmm(
            q.reshape(B * self.num_heads, -1, H * W).transpose(1, 2),
            k.reshape(B * self.num_heads, -1, H * W)
        ) * scale  # (B*heads, N, N)
        attn = attn.softmax(dim=-1)
        h = torch.bmm(
            v.reshape(B * self.num_heads, -1, H * W),
            attn.transpose(1, 2)
        )  # (B*heads, d, N)
        h = h.reshape(B, C, H, W)
        return x + self.proj(h)


class UNet(nn.Module):
    """DDPM++ style U-Net for CIFAR-10 32x32. ~35M params."""
    def __init__(self, in_channels=3, out_channels=3,
                 model_channels=128, channel_mult=(1, 2, 2, 2),
                 num_res_blocks=2, attention_resolutions=(16, 8),
                 dropout=0.1, num_heads=4):
        super().__init__()
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        emb_ch = model_channels * 4

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Build encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = model_channels
        ds = 32
        # Track channel counts for skip connections: start with input_conv output
        self._skip_channels = [model_channels]

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, emb_ch, out_ch, dropout)])
                if ds in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                self.down_blocks.append(block)
                ch = out_ch
                self._skip_channels.append(ch)
            if level < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                self._skip_channels.append(ch)
                ds //= 2

        # Middle
        self.mid_block1 = ResBlock(ch, emb_ch, ch, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResBlock(ch, emb_ch, ch, dropout)

        # Build decoder (reverse order, popping skip channels)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        skip_chs = list(self._skip_channels)  # copy
        for level in reversed(range(len(channel_mult))):
            out_ch = model_channels * channel_mult[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                block = nn.ModuleList([ResBlock(ch + skip_ch, emb_ch, out_ch, dropout)])
                if ds in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                self.up_blocks.append(block)
                ch = out_ch
            if level > 0:
                self.up_samples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1)))
                ds *= 2

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t):
        emb = self.time_embed(timestep_embedding(t, self.model_channels))
        h = self.input_conv(x)
        hs = [h]

        # Down
        block_idx = 0
        sample_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                for layer in self.down_blocks[block_idx]:
                    h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
                hs.append(h)
                block_idx += 1
            if level < len(self.channel_mult) - 1:
                h = self.down_samples[sample_idx](h)
                hs.append(h)
                sample_idx += 1

        # Mid
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        # Up
        block_idx = 0
        sample_idx = 0
        for level in reversed(range(len(self.channel_mult))):
            for _ in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in self.up_blocks[block_idx]:
                    h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
                block_idx += 1
            if level > 0:
                h = self.up_samples[sample_idx](h)
                sample_idx += 1

        return self.out_conv(F.silu(self.out_norm(h)))
