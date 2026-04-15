"""DCT-based spectral decomposition for SCD loss."""
import torch
import torch.nn.functional as F
import math


def dct_2d(x):
    """2D DCT of images using FFT. x: (B, C, H, W) -> DCT coefficients."""
    B, C, H, W = x.shape
    # Type-II DCT via FFT
    # DCT along H
    v_h = torch.cat([x[:, :, ::2, :], x[:, :, 1::2, :].flip(2)], dim=2)
    V_h = torch.fft.fft(v_h, dim=2)
    k_h = torch.arange(H, device=x.device, dtype=torch.float32)
    factor_h = 2 * torch.exp(-1j * math.pi * k_h / (2 * H))
    factor_h[0] = factor_h[0] / math.sqrt(2)
    factor_h = factor_h.view(1, 1, H, 1)
    X_h = (V_h * factor_h).real / math.sqrt(2 * H)

    # DCT along W
    v_w = torch.cat([X_h[:, :, :, ::2], X_h[:, :, :, 1::2].flip(3)], dim=3)
    V_w = torch.fft.fft(v_w, dim=3)
    k_w = torch.arange(W, device=x.device, dtype=torch.float32)
    factor_w = 2 * torch.exp(-1j * math.pi * k_w / (2 * W))
    factor_w[0] = factor_w[0] / math.sqrt(2)
    factor_w = factor_w.view(1, 1, 1, W)
    X = (V_w * factor_w).real / math.sqrt(2 * W)

    return X


def idct_2d(X):
    """2D inverse DCT. X: (B, C, H, W) -> spatial domain."""
    B, C, H, W = X.shape
    # Inverse DCT along W
    k_w = torch.arange(W, device=X.device, dtype=torch.float32)
    factor_w = math.sqrt(2 * W) * torch.exp(1j * math.pi * k_w / (2 * W))
    factor_w[0] = factor_w[0] * math.sqrt(2)
    factor_w = factor_w.view(1, 1, 1, W)
    V_w = X.to(torch.complex64) * factor_w
    v_w = torch.fft.ifft(V_w, dim=3).real
    x_w = torch.zeros_like(X)
    x_w[:, :, :, ::2] = v_w[:, :, :, :W // 2]
    x_w[:, :, :, 1::2] = v_w[:, :, :, W // 2:].flip(3)

    # Inverse DCT along H
    k_h = torch.arange(H, device=X.device, dtype=torch.float32)
    factor_h = math.sqrt(2 * H) * torch.exp(1j * math.pi * k_h / (2 * H))
    factor_h[0] = factor_h[0] * math.sqrt(2)
    factor_h = factor_h.view(1, 1, H, 1)
    V_h = x_w.to(torch.complex64) * factor_h
    v_h = torch.fft.ifft(V_h, dim=2).real
    x = torch.zeros_like(X)
    x[:, :, ::2, :] = v_h[:, :, :H // 2, :]
    x[:, :, 1::2, :] = v_h[:, :, H // 2:, :].flip(2)

    return x


def create_frequency_masks(H, W, K, device='cpu'):
    """Create K frequency band masks for HxW DCT coefficients.

    Bands are defined by L1 distance from DC component (0,0).
    Returns list of K masks, each shape (1, 1, H, W).
    """
    # Frequency magnitude: L1 distance from DC
    freq_y = torch.arange(H, device=device, dtype=torch.float32)
    freq_x = torch.arange(W, device=device, dtype=torch.float32)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_mag = fy + fx  # L1 distance from DC

    max_freq = freq_mag.max().item()
    boundaries = torch.linspace(0, max_freq + 1, K + 1, device=device)

    masks = []
    for k in range(K):
        mask = ((freq_mag >= boundaries[k]) & (freq_mag < boundaries[k + 1])).float()
        masks.append(mask.view(1, 1, H, W))

    return masks


def spectral_band_mse(pred, target, masks):
    """Compute per-band MSE in DCT domain.

    Args:
        pred: predicted images (B, C, H, W)
        target: target images (B, C, H, W)
        masks: list of K frequency band masks
    Returns:
        list of K per-band MSE values
    """
    diff_dct = dct_2d(pred - target)
    band_mses = []
    for mask in masks:
        masked = diff_dct * mask
        mse = (masked ** 2).sum() / (mask.sum() * pred.shape[0] * pred.shape[1] + 1e-8)
        band_mses.append(mse)
    return band_mses


def simple_spectral_loss(pred, target, masks, weights):
    """Simplified spectral loss using FFT-based frequency decomposition.

    Computes MSE in frequency domain with per-band weights.
    """
    # Use 2D FFT for simplicity and reliability
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    diff_power = diff_freq.real ** 2 + diff_freq.imag ** 2

    total_loss = torch.tensor(0.0, device=pred.device)
    for mask, w in zip(masks, weights):
        band_power = (diff_power * mask).mean()
        total_loss = total_loss + w * band_power

    return total_loss


def create_fft_frequency_masks(H, W, K, device='cpu'):
    """Create K frequency band masks for FFT coefficients.

    Uses radial distance in frequency space.
    """
    freq_y = torch.fft.fftfreq(H, device=device)
    freq_x = torch.fft.fftfreq(W, device=device)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing='ij')
    freq_mag = torch.sqrt(fy ** 2 + fx ** 2)

    max_freq = freq_mag.max().item()
    boundaries = torch.linspace(0, max_freq + 1e-6, K + 1, device=device)

    masks = []
    for k in range(K):
        mask = ((freq_mag >= boundaries[k]) & (freq_mag < boundaries[k + 1])).float()
        masks.append(mask.view(1, 1, H, W))

    return masks


def fft_band_mse(pred, target, masks):
    """Compute per-band MSE in FFT domain.

    Args:
        pred: predicted images (B, C, H, W)
        target: target images (B, C, H, W)
        masks: list of K FFT frequency band masks
    Returns:
        list of K per-band MSE values
    """
    diff = pred - target
    diff_freq = torch.fft.fft2(diff)
    diff_power = diff_freq.real ** 2 + diff_freq.imag ** 2

    band_mses = []
    for mask in masks:
        band_power = (diff_power * mask).mean()
        band_mses.append(band_power)
    return band_mses
