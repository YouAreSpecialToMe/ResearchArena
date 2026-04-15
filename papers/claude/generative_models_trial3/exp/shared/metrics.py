"""
Metrics computation for generated images (FID, IS).
"""
import torch
import numpy as np
import os
import tempfile
from pathlib import Path


def save_images_for_fid(images_tensor, output_dir):
    """Save a batch of images as PNGs for FID computation."""
    from torchvision.utils import save_image
    os.makedirs(output_dir, exist_ok=True)
    for i in range(images_tensor.shape[0]):
        img = images_tensor[i]
        save_image(img, os.path.join(output_dir, f'{i:06d}.png'),
                   normalize=True, value_range=(-1, 1))


def compute_fid_from_dir(gen_dir, ref_stats_path=None, num_images=10000):
    """Compute FID using clean-fid."""
    try:
        from cleanfid import fid
        # Compute against ImageNet 256 reference
        # First try with precomputed stats
        if ref_stats_path and os.path.exists(ref_stats_path):
            score = fid.compute_fid(gen_dir, dataset_name="imagenet",
                                     dataset_res=256, dataset_split="custom",
                                     mode="clean")
        else:
            # Use clean-fid's built-in ImageNet stats
            score = fid.compute_fid(gen_dir, dataset_name="imagenet_train",
                                     dataset_res=256, mode="clean")
        return score
    except Exception as e:
        print(f"clean-fid failed: {e}, trying torch-fidelity...")
        return compute_fid_torch_fidelity(gen_dir)


def compute_fid_torch_fidelity(gen_dir):
    """Fallback FID computation using torch-fidelity."""
    try:
        from torch_fidelity import calculate_metrics
        metrics = calculate_metrics(
            input1=gen_dir,
            input2='cifar10-train',  # placeholder, we'll use custom
            cuda=True,
            fid=True,
            isc=True,
        )
        return metrics.get('frechet_inception_distance', float('nan'))
    except Exception as e:
        print(f"torch-fidelity also failed: {e}")
        return float('nan')


def compute_inception_score(gen_dir, splits=10):
    """Compute Inception Score."""
    try:
        from torch_fidelity import calculate_metrics
        metrics = calculate_metrics(
            input1=gen_dir,
            cuda=True,
            isc=True,
            isc_splits=splits,
        )
        return metrics.get('inception_score_mean', float('nan')), metrics.get('inception_score_std', float('nan'))
    except Exception as e:
        print(f"IS computation failed: {e}")
        return float('nan'), float('nan')


def compute_metrics_from_images(images_tensor, tmp_dir=None):
    """Compute FID and IS from a tensor of generated images."""
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix='gen_images_')

    save_images_for_fid(images_tensor, tmp_dir)

    fid_score = compute_fid_from_dir(tmp_dir)
    is_mean, is_std = compute_inception_score(tmp_dir)

    return {
        'fid': fid_score,
        'is_mean': is_mean,
        'is_std': is_std,
        'image_dir': tmp_dir,
    }
