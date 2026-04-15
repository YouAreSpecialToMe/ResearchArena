"""
Augmentation utilities for test-time adaptation.
Includes operations used in MEMO and APAC-TTA.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# ============== Augmentation Operations ==============

AUGMENTATION_OPS = [
    'gaussian_noise',
    'shot_noise',
    'brightness',
    'contrast',
    'defocus_blur',
    'jpeg_compression',
    'pixelate',
    'saturate',
]


def apply_augmentation(image, operation, severity, dataset='cifar10'):
    """
    Apply a single augmentation operation with given severity.
    
    Args:
        image: PIL Image or torch Tensor
        operation: Augmentation operation name
        severity: Severity level (1-5) or float for continuous
        dataset: 'cifar10', 'cifar100', or 'imagenet'
    
    Returns:
        Augmented image (same type as input)
    """
    is_tensor = isinstance(image, torch.Tensor)
    
    if is_tensor:
        # Convert tensor to PIL
        if image.dim() == 4:  # [B, C, H, W]
            image = image[0]
        img_pil = TF.to_pil_image(image.cpu())
    else:
        img_pil = image.copy()
    
    # Apply augmentation based on operation
    if operation == 'gaussian_noise':
        img_pil = _gaussian_noise(img_pil, severity)
    elif operation == 'shot_noise':
        img_pil = _shot_noise(img_pil, severity)
    elif operation == 'brightness':
        img_pil = _brightness(img_pil, severity)
    elif operation == 'contrast':
        img_pil = _contrast(img_pil, severity)
    elif operation == 'defocus_blur':
        img_pil = _defocus_blur(img_pil, severity)
    elif operation == 'jpeg_compression':
        img_pil = _jpeg_compression(img_pil, severity)
    elif operation == 'pixelate':
        img_pil = _pixelate(img_pil, severity)
    elif operation == 'saturate':
        img_pil = _saturate(img_pil, severity)
    elif operation == 'motion_blur':
        img_pil = _motion_blur(img_pil, severity)
    elif operation == 'zoom_blur':
        img_pil = _zoom_blur(img_pil, severity)
    elif operation == 'fog':
        img_pil = _fog(img_pil, severity)
    elif operation == 'frost':
        img_pil = _frost(img_pil, severity)
    elif operation == 'snow':
        img_pil = _snow(img_pil, severity)
    elif operation == 'elastic_transform':
        img_pil = _elastic_transform(img_pil, severity)
    elif operation == 'impulse_noise':
        img_pil = _impulse_noise(img_pil, severity)
    elif operation == 'glass_blur':
        img_pil = _glass_blur(img_pil, severity)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    if is_tensor:
        # Convert back to tensor
        return TF.to_tensor(img_pil).to(image.device)
    return img_pil


# ============== Individual Augmentation Functions ==============

def _gaussian_noise(img, severity):
    """Add Gaussian noise"""
    w, h = img.size
    c = len(img.getbands())
    std = [0.08, 0.12, 0.18, 0.26, 0.38][int(severity) - 1] if isinstance(severity, (int, np.integer)) else severity * 0.3
    noise = np.random.normal(0, std * 255, (h, w, c))
    img_array = np.array(img).astype(np.float32) + noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def _shot_noise(img, severity):
    """Add shot (Poisson) noise"""
    w, h = img.size
    lam = [60, 25, 12, 5, 3][int(severity) - 1] if isinstance(severity, (int, np.integer)) else max(1, int(60 / severity))
    img_array = np.array(img).astype(np.float32)
    img_array = np.random.poisson(img_array * lam) / lam
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def _impulse_noise(img, severity):
    """Add salt and pepper noise"""
    amount = [0.03, 0.06, 0.09, 0.17, 0.27][int(severity) - 1] if isinstance(severity, (int, np.integer)) else severity * 0.1
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Salt
    num_salt = np.ceil(amount * img_array.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 1
    
    # Pepper
    num_pepper = np.ceil(amount * img_array.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape]
    img_array[coords[0], coords[1], :] = 0
    
    return Image.fromarray((img_array * 255).astype(np.uint8))


def _brightness(img, severity):
    """Adjust brightness"""
    factor = [0.1, 0.2, 0.3, 0.4, 0.5][int(severity) - 1] if isinstance(severity, (int, np.integer)) else severity * 0.2
    factor = 1.0 + factor * np.random.choice([-1, 1])
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def _contrast(img, severity):
    """Adjust contrast"""
    factor = [0.4, 0.3, 0.2, 0.15, 0.1][int(severity) - 1] if isinstance(severity, (int, np.integer)) else 0.5 - severity * 0.1
    factor = 0.5 + factor * np.random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def _saturate(img, severity):
    """Adjust saturation"""
    factor = [0.5, 0.75, 1.0, 1.5, 2.0][int(severity) - 1] if isinstance(severity, (int, np.integer)) else severity * 0.5
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def _defocus_blur(img, severity):
    """Apply defocus blur"""
    radius = [3, 4, 6, 8, 10][int(severity) - 1] if isinstance(severity, (int, np.integer)) else int(severity * 3)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _motion_blur(img, severity):
    """Apply motion blur"""
    size = [10, 15, 15, 15, 20][int(severity) - 1] if isinstance(severity, (int, np.integer)) else int(severity * 5)
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return img.filter(ImageFilter.Kernel((size, size), kernel.flatten()))


def _zoom_blur(img, severity):
    """Apply zoom blur"""
    severity = int(severity) if isinstance(severity, (int, np.integer)) else 3
    w, h = img.size
    img_array = np.array(img)
    
    # Simple zoom blur approximation
    result = np.zeros_like(img_array, dtype=np.float32)
    for i in range(1, severity + 1):
        scale = 1 + i * 0.05
        new_w, new_h = int(w * scale), int(h * scale)
        zoomed = img.resize((new_w, new_h), Image.BILINEAR)
        # Crop back to original size
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        zoomed = zoomed.crop((left, top, left + w, top + h))
        result += np.array(zoomed)
    
    result = result / severity
    return Image.fromarray(result.astype(np.uint8))


def _jpeg_compression(img, severity):
    """Apply JPEG compression"""
    quality = [25, 18, 15, 10, 7][int(severity) - 1] if isinstance(severity, (int, np.integer)) else max(5, int(30 - severity * 5))
    from io import BytesIO
    output = BytesIO()
    img.save(output, 'JPEG', quality=quality)
    output.seek(0)
    return Image.open(output)


def _pixelate(img, severity):
    """Apply pixelation"""
    factor = [0.6, 0.5, 0.4, 0.3, 0.25][int(severity) - 1] if isinstance(severity, (int, np.integer)) else max(0.1, 0.7 - severity * 0.1)
    w, h = img.size
    small_w, small_h = int(w * factor), int(h * factor)
    img_small = img.resize((small_w, small_h), Image.BILINEAR)
    return img_small.resize((w, h), Image.BILINEAR)


def _fog(img, severity):
    """Apply fog effect"""
    severity = int(severity) if isinstance(severity, (int, np.integer)) else 3
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Add fog (white overlay with varying intensity)
    fog_intensity = severity * 0.1
    fog = np.ones_like(img_array) * fog_intensity
    
    # Add some spatial variation
    h, w = img_array.shape[:2]
    y, x = np.ogrid[:h, :w]
    mask = np.sin(x / 20) * np.cos(y / 20) * 0.1 + 0.9
    mask = mask[:, :, np.newaxis]
    
    result = img_array * (1 - fog * mask) + fog * mask
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def _frost(img, severity):
    """Apply frost effect"""
    severity = int(severity) if isinstance(severity, (int, np.integer)) else 3
    img_array = np.array(img).astype(np.float32)
    
    # Add frost-like noise pattern
    h, w = img_array.shape[:2]
    noise = np.random.randn(h, w, 1) * severity * 20
    result = img_array + noise
    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def _snow(img, severity):
    """Apply snow effect"""
    severity = int(severity) if isinstance(severity, (int, np.integer)) else 3
    img_array = np.array(img).astype(np.float32)
    
    # Add white dots
    h, w = img_array.shape[:2]
    snow_mask = np.random.rand(h, w) < (severity * 0.02)
    img_array[snow_mask] = np.minimum(img_array[snow_mask] + 100, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))


def _elastic_transform(img, severity):
    """Apply elastic deformation"""
    severity = int(severity) if isinstance(severity, (int, np.integer)) else 3
    img_array = np.array(img)
    
    # Simplified elastic transform
    from scipy.ndimage import gaussian_filter, map_coordinates
    
    alpha = severity * 20
    sigma = severity * 3
    
    shape = img_array.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    result = np.zeros_like(img_array)
    for i in range(img_array.shape[2]):
        result[:, :, i] = map_coordinates(img_array[:, :, i], indices, order=1).reshape(shape)
    
    return Image.fromarray(result)


def _glass_blur(img, severity):
    """Apply glass blur effect"""
    severity = int(severity) if isinstance(severity, (int, np.integer)) else 3
    
    # First apply slight Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=severity))
    
    # Then apply random displacement
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    result = img_array.copy()
    for _ in range(severity * 2):
        dx = np.random.randint(-2, 3, size=(h, w))
        dy = np.random.randint(-2, 3, size=(h, w))
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        
        result = result[y_new, x_new]
    
    return Image.fromarray(result)


# ============== Augmentation Sets ==============

def get_augmentation_set(n_augmentations=8):
    """
    Get a set of augmentation transforms for MEMO-style augmentation.
    Returns a list of callable augmentation functions.
    """
    import torchvision.transforms as transforms
    
    aug_list = []
    
    # Define a set of standard augmentations
    base_transforms = [
        # Color/brightness variations
        transforms.Lambda(lambda x: TF.adjust_brightness(x, np.random.uniform(0.6, 1.4))),
        transforms.Lambda(lambda x: TF.adjust_contrast(x, np.random.uniform(0.6, 1.4))),
        transforms.Lambda(lambda x: TF.adjust_saturation(x, np.random.uniform(0.5, 1.5))),
        # Gaussian blur
        transforms.Lambda(lambda x: TF.gaussian_blur(x, kernel_size=3)),
        # Cropping/resizing
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=1.0),
        # Noise
        transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.1, 0, 1)),
        # JPEG-like compression (approximated)
        transforms.Lambda(lambda x: TF.adjust_sharpness(x, np.random.uniform(0.5, 2.0))),
    ]
    
    # Cycle through augmentations if n_augmentations > len(base_transforms)
    for i in range(n_augmentations):
        if i < len(base_transforms):
            aug_list.append(base_transforms[i])
        else:
            # Add random combinations
            aug_list.append(transforms.Compose([
                base_transforms[i % len(base_transforms)],
                base_transforms[(i + 1) % len(base_transforms)]
            ]))
    
    return aug_list


def get_memo_augmentations():
    """Get the set of 64 augmentations used in MEMO paper"""
    # Simplified version: 8 operations x 8 severity levels
    augmentations = []
    for op in AUGMENTATION_OPS:
        for sev in [1, 2, 3, 4, 5]:
            augmentations.append((op, sev))
    # Add some random combinations
    for _ in range(24):
        op = np.random.choice(AUGMENTATION_OPS)
        sev = np.random.uniform(1, 5)
        augmentations.append((op, sev))
    return augmentations


def apply_augmentations_batch(image, operations_with_severities, dataset='cifar10'):
    """
    Apply multiple augmentations to create a batch.
    
    Args:
        image: Single image (PIL or tensor)
        operations_with_severities: List of (operation, severity) tuples
        dataset: Dataset type
    
    Returns:
        Batch of augmented images as tensor [B, C, H, W]
    """
    augmented = []
    for op, sev in operations_with_severities:
        aug_img = apply_augmentation(image, op, sev, dataset)
        if not isinstance(aug_img, torch.Tensor):
            aug_img = TF.to_tensor(aug_img)
        augmented.append(aug_img)
    
    return torch.stack(augmented)


# ============== Gumbel-Softmax for Differentiable Selection ==============

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax sampling for differentiable discrete selection.
    
    Args:
        logits: [B, K] logit scores for K operations
        temperature: Temperature for softmax (lower = more discrete)
        hard: If True, use straight-through estimator
    
    Returns:
        [B, K] softmax weights
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    
    return y


def select_operations(policy_logits, num_ops=8, temperature=0.5):
    """
    Select augmentation operations based on policy logits.
    
    Args:
        policy_logits: [num_operations] logits from Meta-APN
        num_ops: Number of operations to select
        temperature: Gumbel-Softmax temperature
    
    Returns:
        selected_indices: List of selected operation indices
        probs: Selection probabilities
    """
    probs = gumbel_softmax_sample(policy_logits, temperature, hard=False)
    
    # Select top-k operations
    topk_probs, topk_indices = torch.topk(probs, k=min(num_ops, len(probs)))
    
    return topk_indices, topk_probs


def sample_augmentation_policy(meta_apn, features, prototype_distances, 
                                temperature=0.5, deterministic=False):
    """
    Sample augmentation policy from Meta-APN.
    
    Args:
        meta_apn: Meta-APN model
        features: Feature vector [feature_dim]
        prototype_distances: Distances to prototypes [num_classes]
        temperature: Gumbel-Softmax temperature
        deterministic: If True, use argmax instead of sampling
    
    Returns:
        dict with 'operations', 'severities', 'num_aug'
    """
    features = features.unsqueeze(0) if features.dim() == 1 else features
    prototype_distances = prototype_distances.unsqueeze(0) if prototype_distances.dim() == 1 else prototype_distances
    
    with torch.no_grad():
        policy_logits, severity_scale, num_aug_logits = meta_apn(features, prototype_distances)
    
    # Select operations
    if deterministic:
        op_probs = F.softmax(policy_logits / temperature, dim=-1)
        selected_ops = torch.topk(op_probs[0], k=8)[1].tolist()
    else:
        op_probs = gumbel_softmax_sample(policy_logits[0], temperature, hard=True)
        selected_ops = torch.topk(op_probs, k=8)[1].tolist()
    
    # Get severity scale
    base_severity = 3  # Medium severity
    severities = [max(1, min(5, int(base_severity * severity_scale.item())))] * len(selected_ops)
    
    # Determine number of augmentations (4, 8, or 16)
    num_aug_options = [4, 8, 16]
    if deterministic:
        num_aug = num_aug_options[int(torch.argmax(num_aug_logits[0]).item() % 3)]
    else:
        num_aug_probs = F.softmax(num_aug_logits[0], dim=-1)
        num_aug_idx = torch.multinomial(num_aug_probs, 1).item() % 3
        num_aug = num_aug_options[num_aug_idx]
    
    return {
        'operations': selected_ops,
        'severities': severities,
        'num_aug': num_aug
    }
