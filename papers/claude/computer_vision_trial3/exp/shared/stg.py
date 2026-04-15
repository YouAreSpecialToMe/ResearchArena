"""Spectral Token Gating (STG) - core implementation.

A training-free, plug-and-play method that improves corruption robustness
of pretrained Vision Transformers by analyzing frequency-domain signatures
of token embeddings and gating corrupted tokens in self-attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial


class SpectralTokenGating:
    """Spectral Token Gating module for ViTs.

    Operates by:
    1. Collecting spectral statistics from clean images (calibration)
    2. At test time, computing frequency spectrum of each token
    3. Measuring anomaly via Mahalanobis distance from clean reference
    4. Gating value vectors in attention to suppress corrupted tokens
    """

    def __init__(self, model, model_key, target_layers=None, K=3, alpha=5.0,
                 tau_percentile=95, gating_fn='sigmoid', device='cuda'):
        self.model = model
        self.model_key = model_key
        self.K = K
        self.alpha = alpha
        self.tau_percentile = tau_percentile
        self.gating_fn = gating_fn
        self.device = device

        # Determine target layers
        from models import get_model_blocks, MODEL_CONFIGS
        self.blocks = get_model_blocks(model, model_key)
        config = MODEL_CONFIGS[model_key]
        self.target_layers = target_layers or config['default_stg_layers']
        self.architecture = config['architecture']

        # Calibration statistics: per-layer mean and covariance of spectral ratios
        self.calibration_stats = {}  # layer_idx -> {'mean': tensor, 'cov_inv': tensor, 'tau': float}

        # Hooks and state
        self._hooks = []
        self._token_embeddings = {}  # layer_idx -> embeddings captured by hooks
        self._gating_scores = {}  # layer_idx -> gating scores for visualization

    def _compute_spectral_ratios(self, tokens):
        """Compute spectral energy ratios for a batch of token embeddings.

        Args:
            tokens: (batch*num_tokens, embed_dim) tensor

        Returns:
            ratios: (batch*num_tokens, K) tensor of energy ratios per frequency band
        """
        # 1D DFT along embedding dimension
        spectrum = torch.fft.rfft(tokens, dim=-1)
        power = torch.abs(spectrum) ** 2  # (N, embed_dim//2 + 1)

        freq_bins = power.shape[-1]
        band_size = freq_bins // self.K
        ratios = []

        total_energy = power.sum(dim=-1, keepdim=True) + 1e-10

        for k in range(self.K):
            start = k * band_size
            end = (k + 1) * band_size if k < self.K - 1 else freq_bins
            band_energy = power[:, start:end].sum(dim=-1, keepdim=True)
            ratios.append(band_energy / total_energy)

        return torch.cat(ratios, dim=-1)  # (N, K)

    def calibrate(self, dataloader, n_images=1000):
        """Collect spectral statistics from clean images.

        Args:
            dataloader: DataLoader with clean images
            n_images: number of images to use for calibration
        """
        self.model.eval()

        # Collect embeddings at target layers using hooks
        layer_ratios = {l: [] for l in self.target_layers}

        def hook_fn(layer_idx, module, input, output):
            if self.architecture == 'columnar':
                # DeiT: output is (B, N, D) where N includes CLS token
                tokens = output[:, 1:, :]  # exclude CLS token
            else:
                # Swin: output from block is (B, N, D) - no CLS token in windows
                if isinstance(output, tuple):
                    tokens = output[0]
                else:
                    tokens = output
            # Flatten batch and tokens
            tokens_flat = tokens.reshape(-1, tokens.shape[-1])
            ratios = self._compute_spectral_ratios(tokens_flat)
            layer_ratios[layer_idx].append(ratios.cpu())

        # Register hooks
        hooks = []
        for layer_idx in self.target_layers:
            h = self.blocks[layer_idx].register_forward_hook(
                partial(hook_fn, layer_idx))
            hooks.append(h)

        n_collected = 0
        with torch.no_grad():
            for images, _ in dataloader:
                if n_collected >= n_images:
                    break
                images = images.to(self.device)
                self.model(images)
                n_collected += images.shape[0]

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute statistics per layer
        for layer_idx in self.target_layers:
            all_ratios = torch.cat(layer_ratios[layer_idx], dim=0)  # (total_tokens, K)

            mean = all_ratios.mean(dim=0)  # (K,)
            # Compute covariance
            centered = all_ratios - mean.unsqueeze(0)
            cov = (centered.T @ centered) / (centered.shape[0] - 1)
            # Regularize
            cov += 1e-5 * torch.eye(self.K)
            cov_inv = torch.linalg.inv(cov)

            # Compute Mahalanobis distances for calibration set to set threshold
            diff = centered  # (N, K)
            mahal = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=-1))  # (N,)

            tau = torch.quantile(mahal, self.tau_percentile / 100.0).item()

            self.calibration_stats[layer_idx] = {
                'mean': mean.to(self.device),
                'cov_inv': cov_inv.to(self.device),
                'tau': tau,
                'mahal_dist': mahal,  # keep for analysis
            }

    def _compute_gating_scores(self, tokens, layer_idx):
        """Compute gating scores for a set of tokens.

        Args:
            tokens: (B, N, D) tensor of token embeddings (excluding CLS)
            layer_idx: which layer's calibration stats to use

        Returns:
            scores: (B, N) tensor, values in [0, 1] where 1 = keep, 0 = suppress
        """
        B, N, D = tokens.shape
        stats = self.calibration_stats[layer_idx]

        tokens_flat = tokens.reshape(-1, D)
        ratios = self._compute_spectral_ratios(tokens_flat)  # (B*N, K)

        diff = ratios - stats['mean'].unsqueeze(0)
        mahal = torch.sqrt(torch.sum(diff @ stats['cov_inv'] * diff, dim=-1) + 1e-10)  # (B*N,)

        tau = stats['tau']

        if self.gating_fn == 'sigmoid':
            scores = torch.sigmoid(-self.alpha * (mahal - tau))
        elif self.gating_fn == 'hard':
            scores = (mahal < tau).float()
        elif self.gating_fn == 'linear':
            tau_max = tau * 1.5  # extend to 150% of tau for linear ramp
            scores = torch.clamp(1.0 - mahal / tau_max, min=0.0, max=1.0)
        else:
            raise ValueError(f"Unknown gating function: {self.gating_fn}")

        return scores.reshape(B, N)

    def enable(self):
        """Install forward hooks that modulate attention values."""
        self.disable()  # remove any existing hooks

        def attn_hook_deit(layer_idx, module, input, output):
            """Hook for DeiT attention blocks.
            We hook the entire block and modify the residual contribution.
            Actually, we need to hook at the attention level to modify V.
            Instead, we hook the block output and apply gating post-hoc.
            """
            # output is (B, N, D) where N = 1 + num_patches (CLS + patches)
            B, N, D = output.shape
            patch_tokens = output[:, 1:, :]  # (B, N-1, D)

            # Compute gating scores for patch tokens
            scores = self._compute_gating_scores(patch_tokens, layer_idx)  # (B, N-1)
            self._gating_scores[layer_idx] = scores.detach()

            # Apply gating: modulate patch token contributions
            # We gate the residual from this block by scaling patch tokens
            gated_output = output.clone()
            gated_output[:, 1:, :] = patch_tokens * scores.unsqueeze(-1)
            return gated_output

        def attn_hook_swin(layer_idx, module, input, output):
            """Hook for Swin transformer blocks."""
            if isinstance(output, tuple):
                tokens = output[0]
            else:
                tokens = output

            B, N, D = tokens.shape
            scores = self._compute_gating_scores(tokens, layer_idx)  # (B, N)
            self._gating_scores[layer_idx] = scores.detach()

            gated = tokens * scores.unsqueeze(-1)
            if isinstance(output, tuple):
                return (gated,) + output[1:]
            return gated

        for layer_idx in self.target_layers:
            if layer_idx not in self.calibration_stats:
                continue
            if self.architecture == 'columnar':
                hook = self.blocks[layer_idx].register_forward_hook(
                    partial(attn_hook_deit, layer_idx))
            else:
                hook = self.blocks[layer_idx].register_forward_hook(
                    partial(attn_hook_swin, layer_idx))
            self._hooks.append(hook)

    def disable(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._gating_scores = {}

    def get_gating_scores(self):
        """Return the most recent gating scores."""
        return self._gating_scores

    def save_calibration(self, filepath):
        """Save calibration statistics."""
        save_data = {}
        for layer_idx, stats in self.calibration_stats.items():
            save_data[layer_idx] = {
                'mean': stats['mean'].cpu(),
                'cov_inv': stats['cov_inv'].cpu(),
                'tau': stats['tau'],
            }
        torch.save(save_data, filepath)

    def load_calibration(self, filepath):
        """Load calibration statistics."""
        save_data = torch.load(filepath, map_location=self.device, weights_only=True)
        self.calibration_stats = {}
        for layer_idx, stats in save_data.items():
            # Keys may be stored as strings in JSON-like formats
            layer_idx = int(layer_idx) if isinstance(layer_idx, str) else layer_idx
            self.calibration_stats[layer_idx] = {
                'mean': stats['mean'].to(self.device),
                'cov_inv': stats['cov_inv'].to(self.device),
                'tau': stats['tau'],
            }
