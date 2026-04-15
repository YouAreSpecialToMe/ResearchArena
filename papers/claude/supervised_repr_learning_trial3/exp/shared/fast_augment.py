"""Vectorized GPU augmentations using grid_sample for massive speedup.

Replaces per-image Python loops with batched GPU operations.
"""

import math
import torch
import torch.nn.functional as F


CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
CIFAR100_STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)


class FastContrastiveAugment:
    """Vectorized contrastive augmentations on GPU using grid_sample."""

    def __init__(self, device='cuda'):
        self.device = device
        self.mean = CIFAR100_MEAN.to(device)
        self.std = CIFAR100_STD.to(device)

    def random_resized_crop(self, imgs):
        """Batched RandomResizedCrop(32, scale=(0.2, 1.0)) via affine_grid + grid_sample."""
        B, C, H, W = imgs.shape

        # Sample area fraction and compute crop size (square aspect ratio)
        area = torch.empty(B, device=self.device).uniform_(0.2, 1.0)
        # Also sample aspect ratio like torchvision: log-uniform in [3/4, 4/3]
        log_ratio = torch.empty(B, device=self.device).uniform_(
            math.log(3.0 / 4.0), math.log(4.0 / 3.0)
        )
        ratio = log_ratio.exp()

        crop_w = (area * ratio).sqrt().clamp(max=1.0)
        crop_h = (area / ratio).sqrt().clamp(max=1.0)

        # Fallback for invalid crops (when w or h > 1)
        invalid = (crop_w > 1.0) | (crop_h > 1.0)
        crop_w = torch.where(invalid, area.sqrt().clamp(max=1.0), crop_w)
        crop_h = torch.where(invalid, area.sqrt().clamp(max=1.0), crop_h)

        # Random position: top-left corner in [0, 1-crop_size]
        x1 = torch.rand(B, device=self.device) * (1.0 - crop_w).clamp(min=0)
        y1 = torch.rand(B, device=self.device) * (1.0 - crop_h).clamp(min=0)

        # Build affine transform: maps output [-1,1] to input crop region
        # Input coords: x in [2*x1-1, 2*(x1+crop_w)-1], y similar
        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = crop_w
        theta[:, 1, 1] = crop_h
        theta[:, 0, 2] = 2.0 * x1 + crop_w - 1.0
        theta[:, 1, 2] = 2.0 * y1 + crop_h - 1.0

        grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)
        return F.grid_sample(imgs, grid, mode='bilinear', padding_mode='reflection',
                             align_corners=False)

    def __call__(self, imgs):
        """Apply full contrastive augmentation pipeline.

        Args:
            imgs: [B, 3, 32, 32] float tensor in [0, 1] on GPU
        Returns:
            [B, 3, 32, 32] augmented and normalized tensor
        """
        B = imgs.shape[0]

        # 1. RandomResizedCrop
        imgs = self.random_resized_crop(imgs)

        # 2. RandomHorizontalFlip (p=0.5)
        flip_mask = torch.rand(B, device=self.device) < 0.5
        imgs[flip_mask] = imgs[flip_mask].flip(-1)

        # 3. ColorJitter(0.4, 0.4, 0.4, 0.1) with p=0.8
        jitter_mask = torch.rand(B, device=self.device) < 0.8
        n_jitter = jitter_mask.sum().item()
        if n_jitter > 0:
            j = imgs[jitter_mask]
            # Brightness: multiply by uniform(0.6, 1.4)
            b = torch.empty(n_jitter, 1, 1, 1, device=self.device).uniform_(0.6, 1.4)
            j = j * b
            # Contrast: lerp toward mean
            c = torch.empty(n_jitter, 1, 1, 1, device=self.device).uniform_(0.6, 1.4)
            m = j.mean(dim=[2, 3], keepdim=True)
            j = (j - m) * c + m
            # Saturation: lerp toward grayscale
            s = torch.empty(n_jitter, 1, 1, 1, device=self.device).uniform_(0.6, 1.4)
            gray = j.mean(dim=1, keepdim=True)
            j = (j - gray) * s + gray
            # Hue: small rotation (simplified)
            # Skip for speed - minimal impact
            imgs[jitter_mask] = j.clamp(0, 1)

        # 4. RandomGrayscale (p=0.2)
        gray_mask = torch.rand(B, device=self.device) < 0.2
        if gray_mask.any():
            g = imgs[gray_mask].mean(dim=1, keepdim=True)
            imgs[gray_mask] = g.expand(-1, 3, -1, -1)

        # 5. Normalize
        imgs = (imgs - self.mean) / self.std

        return imgs


class FastCEAugment:
    """Vectorized CE augmentations: RandomCrop(32, padding=4) + flip."""

    def __init__(self, device='cuda'):
        self.device = device
        self.mean = CIFAR100_MEAN.to(device)
        self.std = CIFAR100_STD.to(device)

    def __call__(self, imgs):
        B, C, H, W = imgs.shape

        # Pad with reflection
        padded = F.pad(imgs, [4, 4, 4, 4], mode='reflect')  # [B, 3, 40, 40]

        # Batched random crop: generate random offsets
        top = torch.randint(0, 9, (B,), device=self.device)
        left = torch.randint(0, 9, (B,), device=self.device)

        # Use affine_grid for batched crop (each image gets different offset)
        # Crop region in padded image: [top:top+32, left:left+32] out of 40x40
        # In normalized coords [-1, 1]: offset = 2*pos/39 - 1
        pH, pW = 40, 40
        crop_h = 32.0 / pH
        crop_w = 32.0 / pW

        y1 = top.float() / pH
        x1 = left.float() / pW

        theta = torch.zeros(B, 2, 3, device=self.device)
        theta[:, 0, 0] = crop_w
        theta[:, 1, 1] = crop_h
        theta[:, 0, 2] = 2.0 * x1 + crop_w - 1.0
        theta[:, 1, 2] = 2.0 * y1 + crop_h - 1.0

        grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)
        imgs = F.grid_sample(padded, grid, mode='bilinear', padding_mode='zeros',
                             align_corners=False)

        # RandomHorizontalFlip
        flip_mask = torch.rand(B, device=self.device) < 0.5
        imgs[flip_mask] = imgs[flip_mask].flip(-1)

        # Normalize
        imgs = (imgs - self.mean) / self.std
        return imgs
