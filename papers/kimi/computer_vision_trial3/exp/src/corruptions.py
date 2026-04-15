"""
Synthetic corruption generation for test-time adaptation experiments.
Based on ImageNet-C corruptions.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io


def gaussian_noise(x, severity=3):
    """Add Gaussian noise."""
    std = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0, 1)


def shot_noise(x, severity=3):
    """Add shot (Poisson) noise."""
    lam = [60, 25, 12, 5, 3][severity - 1]
    x_discrete = torch.poisson(x * lam) / lam
    return torch.clamp(x_discrete, 0, 1)


def impulse_noise(x, severity=3):
    """Add salt and pepper noise."""
    amount = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    
    # Convert to numpy for easier manipulation
    x_np = x.cpu().numpy()
    
    for i in range(x.shape[0]):
        # Salt
        num_salt = int(amount * x.shape[2] * x.shape[3])
        coords = [np.random.randint(0, s, num_salt) for s in x.shape[2:]]
        x_np[i, :, coords[0], coords[1]] = 1
        
        # Pepper
        num_pepper = int(amount * x.shape[2] * x.shape[3])
        coords = [np.random.randint(0, s, num_pepper) for s in x.shape[2:]]
        x_np[i, :, coords[0], coords[1]] = 0
    
    return torch.from_numpy(x_np).to(x.device)


def gaussian_blur(x, severity=3):
    """Apply Gaussian blur."""
    radius = [0.4, 0.6, 1.0, 1.5, 2.0][severity - 1]
    
    # Convert to PIL, apply blur, convert back
    x_np = x.cpu().numpy()
    blurred = []
    for i in range(x.shape[0]):
        img = (x_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
        img = np.array(pil_img).astype(np.float32) / 255.0
        blurred.append(torch.from_numpy(img.transpose(2, 0, 1)))
    
    return torch.stack(blurred).to(x.device)


def brightness(x, severity=3):
    """Adjust brightness."""
    factor = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    return torch.clamp(x + factor, 0, 1)


def contrast(x, severity=3):
    """Adjust contrast."""
    factor = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    mean = x.mean(dim=[2, 3], keepdim=True)
    return torch.clamp((x - mean) * factor + mean, 0, 1)


def jpeg_compression(x, severity=3):
    """Apply JPEG compression."""
    quality = [25, 18, 15, 10, 7][severity - 1]
    
    x_np = x.cpu().numpy()
    compressed = []
    for i in range(x.shape[0]):
        img = (x_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        
        # Save and load with JPEG compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        pil_img = Image.open(buffer)
        
        img = np.array(pil_img).astype(np.float32) / 255.0
        compressed.append(torch.from_numpy(img.transpose(2, 0, 1)))
    
    return torch.stack(compressed).to(x.device)


def pixelate(x, severity=3):
    """Pixelate image."""
    scales = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    
    h, w = x.shape[2:]
    new_h, new_w = int(h * scales), int(w * scales)
    
    x_down = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x_up = F.interpolate(x_down, size=(h, w), mode='nearest')
    
    return x_up


# Corruption registry
CORRUPTION_FUNCTIONS = {
    'gaussian_noise': gaussian_noise,
    'shot_noise': shot_noise,
    'impulse_noise': impulse_noise,
    'defocus_blur': gaussian_blur,
    'glass_blur': gaussian_blur,  # Approximation
    'motion_blur': gaussian_blur,  # Approximation
    'zoom_blur': gaussian_blur,  # Approximation
    'brightness': brightness,
    'contrast': contrast,
    'jpeg_compression': jpeg_compression,
    'pixelate': pixelate,
    'snow': gaussian_noise,  # Approximation
    'frost': gaussian_noise,  # Approximation
    'fog': gaussian_blur,  # Approximation
    'elastic_transform': gaussian_blur,  # Approximation
}


def apply_corruption(x, corruption_type, severity=3):
    """
    Apply a corruption to a batch of images.
    
    Args:
        x: Image tensor [B, C, H, W] in range [0, 1]
        corruption_type: Type of corruption
        severity: Severity level (1-5)
    
    Returns:
        Corrupted image tensor
    """
    if corruption_type not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption: {corruption_type}")
    
    return CORRUPTION_FUNCTIONS[corruption_type](x, severity)


class CorruptedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies corruptions on-the-fly."""
    
    def __init__(self, base_dataset, corruption_type, severity=3):
        """
        Args:
            base_dataset: Base dataset to wrap
            corruption_type: Type of corruption to apply
            severity: Severity level (1-5)
        """
        self.base_dataset = base_dataset
        self.corruption_type = corruption_type
        self.severity = severity
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # Denormalize to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        
        # Apply corruption
        img_denorm = img_denorm.unsqueeze(0)  # Add batch dimension
        img_corrupted = apply_corruption(img_denorm, self.corruption_type, self.severity)
        img_corrupted = img_corrupted.squeeze(0)
        
        # Renormalize
        img = (img_corrupted - mean) / std
        
        return img, label


if __name__ == "__main__":
    # Test corruptions
    x = torch.randn(2, 3, 224, 224).clamp(0, 1)
    
    for corruption in ['gaussian_noise', 'shot_noise', 'brightness', 'contrast']:
        x_corrupted = apply_corruption(x, corruption, severity=3)
        print(f"{corruption}: input range [{x.min():.3f}, {x.max():.3f}], "
              f"output range [{x_corrupted.min():.3f}, {x_corrupted.max():.3f}]")
