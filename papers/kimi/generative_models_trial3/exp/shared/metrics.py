"""
Evaluation metrics for CAD-DiT experiments.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
try:
    from torchvision.models import inception_v3, Inception_V3_Weights
except:
    from torchvision.models import inception_v3


class InceptionScore:
    """Compute Inception Score."""
    def __init__(self, device='cuda', batch_size=32, splits=10):
        self.device = device
        self.batch_size = batch_size
        self.splits = splits
        
        # Load inception model
        try:
            self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        except:
            self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get the classifier
        try:
            self.classifier = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except:
            self.classifier = inception_v3(pretrained=True)
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
    
    def compute(self, images):
        """
        Compute IS for a batch of images.
        
        Args:
            images: Tensor of shape [N, 3, H, W] in range [0, 1]
        
        Returns:
            mean, std of IS
        """
        preds = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size].to(self.device)
                # Resize to 299x299 for inception
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = nn.functional.interpolate(
                        batch, size=(299, 299), mode='bilinear', align_corners=False
                    )
                # Get predictions
                pred = self.classifier(batch)
                pred = nn.functional.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # Compute IS
        split_scores = []
        for k in range(self.splits):
            part = preds[k * (len(preds) // self.splits): (k + 1) * (len(preds) // self.splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)


class FIDScore:
    """Compute Fréchet Inception Distance."""
    def __init__(self, device='cuda', batch_size=32):
        self.device = device
        self.batch_size = batch_size
        
        # Load inception model
        try:
            self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        except:
            self.model = inception_v3(pretrained=True, transform_input=False)
        
        # Remove the final fc layer and use it as feature extractor
        self.model.fc = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
    
    def get_activations(self, images):
        """Get inception features for images."""
        activations = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size].to(self.device)
                # Resize to 299x299 for inception
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = nn.functional.interpolate(
                        batch, size=(299, 299), mode='bilinear', align_corners=False
                    )
                # Normalize to [-1, 1]
                batch = batch * 2 - 1
                
                feat = self.model(batch)
                activations.append(feat.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    def compute(self, images_real, images_fake):
        """
        Compute FID between real and fake images.
        
        Args:
            images_real: Tensor of shape [N, 3, H, W] in range [0, 1]
            images_fake: Tensor of shape [N, 3, H, W] in range [0, 1]
        
        Returns:
            FID score
        """
        act_real = self.get_activations(images_real)
        act_fake = self.get_activations(images_fake)
        
        # Compute statistics
        mu_real = np.mean(act_real, axis=0)
        sigma_real = np.cov(act_real, rowvar=False)
        
        mu_fake = np.mean(act_fake, axis=0)
        sigma_fake = np.cov(act_fake, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_fake
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma_real.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return float(fid)


def compute_flops_dit(num_layers, hidden_size, seq_len, active_layers=None):
    """
    Estimate FLOPs for DiT model.
    
    Args:
        num_layers: Total number of transformer layers
        hidden_size: Hidden dimension
        seq_len: Sequence length (number of tokens)
        active_layers: If provided, fraction of layers actually used per token
    
    Returns:
        Estimated FLOPs
    """
    # Per-layer FLOPs for transformer block
    # Self-attention: 4 * hidden_size^2 * seq_len + 2 * hidden_size * seq_len^2
    # FFN: 8 * hidden_size^2 * seq_len (assuming 4x expansion)
    
    attn_flops = 4 * hidden_size * hidden_size * seq_len + 2 * hidden_size * seq_len * seq_len
    ffn_flops = 8 * hidden_size * hidden_size * seq_len
    layer_flops = attn_flops + ffn_flops
    
    if active_layers is None:
        total_flops = num_layers * layer_flops
    else:
        # active_layers is average number of layers used
        total_flops = active_layers * layer_flops
    
    return total_flops


def compute_speedup(original_flops, optimized_flops):
    """Compute speedup ratio."""
    return original_flops / optimized_flops


def compute_flops_reduction(original_flops, optimized_flops):
    """Compute FLOPs reduction percentage."""
    return (original_flops - optimized_flops) / original_flops * 100
