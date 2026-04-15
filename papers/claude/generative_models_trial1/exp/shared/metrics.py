"""Evaluation metrics: FID, per-frequency MSE, LPIPS."""
import os
import torch
import numpy as np
from pathlib import Path


def save_images_for_fid(images, save_dir, start_idx=0):
    """Save tensor images as PNG files for FID computation.

    Args:
        images: (N, 3, H, W) tensor in [-1, 1]
        save_dir: directory to save images
    """
    from torchvision.utils import save_image
    os.makedirs(save_dir, exist_ok=True)
    images = (images + 1) / 2  # [-1,1] -> [0,1]
    images = images.clamp(0, 1)
    for i in range(images.shape[0]):
        save_image(images[i], os.path.join(save_dir, f'{start_idx + i:06d}.png'))


def compute_fid(gen_dir, dataset_name='cifar10', dataset_split='train', num_gen=None):
    """Compute FID using clean-fid library.

    Args:
        gen_dir: directory containing generated images
        dataset_name: 'cifar10' for CIFAR-10 stats
        dataset_split: 'train' or 'test'
    """
    from cleanfid import fid
    score = fid.compute_fid(gen_dir, dataset_name=dataset_name,
                            dataset_split=dataset_split, dataset_res=32,
                            mode='clean')
    return score


def compute_lpips_score(images1, images2, net='alex', batch_size=64):
    """Compute mean LPIPS between two sets of images.

    Args:
        images1, images2: (N, 3, H, W) tensors in [-1, 1]
    """
    import lpips
    loss_fn = lpips.LPIPS(net=net).cuda()
    loss_fn.eval()

    N = min(len(images1), len(images2))
    scores = []
    for i in range(0, N, batch_size):
        b1 = images1[i:i + batch_size].cuda()
        b2 = images2[i:i + batch_size].cuda()
        with torch.no_grad():
            d = loss_fn(b1, b2)
        scores.append(d.cpu())

    return torch.cat(scores).mean().item()


@torch.no_grad()
def generate_samples(model, num_samples, num_steps, batch_size=256,
                     image_shape=(3, 32, 32), device='cuda', seed=42):
    """Generate samples using Euler ODE solver.

    Args:
        model: velocity field model
        num_samples: total number of samples to generate
        num_steps: number of Euler steps
        batch_size: generation batch size
        image_shape: (C, H, W)
        device: device
        seed: random seed
    Returns:
        all_samples: (num_samples, C, H, W) tensor in [-1, 1]
    """
    from .flow_matching import euler_sample

    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    all_samples = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, *image_shape, device=device, generator=gen)
        samples = euler_sample(model, z, num_steps)
        all_samples.append(samples.cpu())
        remaining -= bs

    return torch.cat(all_samples, dim=0)[:num_samples]


@torch.no_grad()
def evaluate_model(model, num_steps_list, num_samples=50000,
                   batch_size=256, device='cuda', seed=42,
                   compute_spectral=True, num_freq_bands=4,
                   real_images=None):
    """Full evaluation pipeline.

    Args:
        model: velocity field or consistency model
        num_steps_list: list of step budgets to evaluate, e.g. [1, 2, 4]
        num_samples: number of samples for FID
        batch_size: generation batch size
        compute_spectral: whether to compute per-band MSE
        num_freq_bands: K for spectral analysis
        real_images: real images for spectral comparison (N, C, H, W) in [-1, 1]
    Returns:
        results dict
    """
    import tempfile, shutil
    from .spectral import create_fft_frequency_masks, fft_band_mse

    results = {}

    for num_steps in num_steps_list:
        step_key = f"{num_steps}_step"
        print(f"  Evaluating {num_steps}-step generation ({num_samples} samples)...")
        samples = generate_samples(model, num_samples, num_steps,
                                   batch_size=batch_size, device=device, seed=seed)

        # FID
        tmp_dir = tempfile.mkdtemp()
        try:
            save_images_for_fid(samples, tmp_dir)
            fid_score = compute_fid(tmp_dir)
        finally:
            shutil.rmtree(tmp_dir)

        step_results = {'fid': fid_score}

        # Per-frequency MSE
        if compute_spectral and real_images is not None:
            H, W = samples.shape[2], samples.shape[3]
            masks = create_fft_frequency_masks(H, W, num_freq_bands, device=device)
            n_eval = min(len(samples), len(real_images), 10000)
            gen_batch = samples[:n_eval].to(device)
            real_batch = real_images[:n_eval].to(device)
            band_mses = fft_band_mse(gen_batch, real_batch, masks)
            step_results['per_band_mse'] = [m.item() for m in band_mses]

        results[step_key] = step_results

    return results
