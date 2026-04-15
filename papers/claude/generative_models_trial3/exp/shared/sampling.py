"""
Shared sampling utilities for CSG experiments.
Implements Standard CFG, CSG, ESG, CSG-PL, and CSG-H sampling loops.
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DiT'))
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


def load_dit_model(checkpoint_path, device='cuda', image_size=256):
    """Load pretrained DiT-XL/2 model."""
    latent_size = image_size // 8
    model = DiT_models['DiT-XL/2'](input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_vae(device='cuda'):
    """Load the Stable Diffusion VAE for decoding latents."""
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    return vae


def decode_latents(vae, latents):
    """Decode latent representations to images using VAE."""
    with torch.no_grad():
        images = vae.decode(latents / 0.18215).sample
    return images


def generate_class_labels(num_images, num_classes=1000, seed=0):
    """Generate balanced class labels for FID computation."""
    rng = np.random.RandomState(seed)
    # First fill all classes equally, then random for remainder
    labels_per_class = num_images // num_classes
    remainder = num_images % num_classes
    labels = []
    for c in range(num_classes):
        labels.extend([c] * labels_per_class)
    if remainder > 0:
        extra = rng.choice(num_classes, remainder, replace=False)
        labels.extend(extra.tolist())
    rng.shuffle(labels)
    return labels


def get_noise_and_labels(num_images, latent_size=32, seed=0, device='cuda'):
    """Generate deterministic noise and labels for a given seed."""
    labels = generate_class_labels(num_images, seed=seed)
    # Generate noise in CPU for reproducibility, then move to device
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(num_images, 4, latent_size, latent_size, generator=generator)
    return noise, torch.tensor(labels)


# ============= Standard CFG (2-pass) =============

def forward_cfg(model, x, t_tensor, y_cond, y_uncond, cfg_scale):
    """Standard 2-pass CFG: run conditional and unconditional, then combine."""
    # Combine into single batch for efficiency
    x_combined = torch.cat([x, x], dim=0)
    t_combined = torch.cat([t_tensor, t_tensor], dim=0)
    y_combined = torch.cat([y_cond, y_uncond], dim=0)

    model_out = model(x_combined, t_combined, y_combined)

    eps_cond, eps_uncond = model_out.chunk(2, dim=0)
    # Apply CFG to noise prediction channels only (first 4 channels)
    eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    return eps_guided


# ============= CSG: Conditioning-Space Guidance =============

def forward_csg(model, x, t_tensor, y_cond, y_uncond, cfg_scale, per_layer_weights=None):
    """
    CSG: Compute AdaLN params for both conditions, extrapolate, single forward pass.
    per_layer_weights: optional list/tensor of per-layer guidance weights (for CSG-PL)
    """
    # Step 1: Compute conditioning embeddings
    t_emb = model.t_embedder(t_tensor)
    y_cond_emb = model.y_embedder(y_cond, train=False)
    y_uncond_emb = model.y_embedder(y_uncond, train=False)
    c_cond = t_emb + y_cond_emb
    c_uncond = t_emb + y_uncond_emb

    # Step 2: Patch embed input
    x_tokens = model.x_embedder(x) + model.pos_embed

    # Step 3: Forward through blocks with guided AdaLN params
    for i, block in enumerate(model.blocks):
        w = cfg_scale if per_layer_weights is None else per_layer_weights[i]

        # Compute AdaLN params for both conditions (cheap MLP)
        params_cond = block.adaLN_modulation(c_cond)
        params_uncond = block.adaLN_modulation(c_uncond)

        # Extrapolate in parameter space
        params_guided = params_uncond + w * (params_cond - params_uncond)

        # Unpack guided params
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params_guided.chunk(6, dim=1)

        # Apply block with guided params (inline the block forward)
        # Attention branch
        x_norm1 = block.norm1(x_tokens)
        x_mod1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x_tokens = x_tokens + gate_msa.unsqueeze(1) * block.attn(x_mod1)

        # MLP branch
        x_norm2 = block.norm2(x_tokens)
        x_mod2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x_tokens = x_tokens + gate_mlp.unsqueeze(1) * block.mlp(x_mod2)

    # Step 4: Final layer with guided params
    final_params_cond = model.final_layer.adaLN_modulation(c_cond)
    final_params_uncond = model.final_layer.adaLN_modulation(c_uncond)
    w_final = cfg_scale if per_layer_weights is None else per_layer_weights[-1] if len(per_layer_weights) > len(model.blocks) else cfg_scale
    final_params_guided = final_params_uncond + w_final * (final_params_cond - final_params_uncond)
    shift, scale = final_params_guided.chunk(2, dim=1)

    x_tokens = model.final_layer.norm_final(x_tokens)
    x_tokens = x_tokens * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x_tokens = model.final_layer.linear(x_tokens)

    # Unpatchify
    output = model.unpatchify(x_tokens)
    return output


# ============= ESG: Embedding-Space Guidance =============

def forward_esg(model, x, t_tensor, y_cond, y_uncond, cfg_scale):
    """
    ESG: Interpolate conditioning embedding BEFORE AdaLN MLPs, then single forward pass.
    """
    t_emb = model.t_embedder(t_tensor)
    y_cond_emb = model.y_embedder(y_cond, train=False)
    y_uncond_emb = model.y_embedder(y_uncond, train=False)
    c_cond = t_emb + y_cond_emb
    c_uncond = t_emb + y_uncond_emb

    # Interpolate in embedding space
    c_guided = c_uncond + cfg_scale * (c_cond - c_uncond)

    # Standard forward with guided embedding
    x_tokens = model.x_embedder(x) + model.pos_embed
    for block in model.blocks:
        x_tokens = block(x_tokens, c_guided)
    x_tokens = model.final_layer(x_tokens, c_guided)
    output = model.unpatchify(x_tokens)
    return output


# ============= Sampling Loops =============

def ddim_sample_step(x_t, eps_pred, t, t_prev, alphas_cumprod):
    """Single DDIM step."""
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1) if t_prev >= 0 else torch.ones_like(alpha_t)

    # Split model output into noise and variance predictions
    if eps_pred.shape[1] == 8:  # learned sigma
        eps, model_var = eps_pred.chunk(2, dim=1)
    else:
        eps = eps_pred

    # Predict x_0
    x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
    x_0_pred = torch.clamp(x_0_pred, -10, 10)

    # Compute x_{t-1}
    x_prev = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * eps
    return x_prev


def sample_images(model, diffusion, method, noise_batch, labels_batch, cfg_scale,
                  device='cuda', per_layer_weights=None, hybrid_ratio=0.0, hybrid_steps=None):
    """
    Generate images using specified method.
    method: 'cfg', 'csg', 'esg', 'no_guidance'
    Returns: latent predictions
    """
    batch_size = noise_batch.shape[0]
    x = noise_batch.to(device)
    y_cond = labels_batch.to(device)
    y_uncond = torch.full_like(y_cond, 1000)  # null class label

    # Get diffusion schedule
    timesteps = list(range(diffusion.num_timesteps))[::-1]  # reverse: T-1, T-2, ..., 0
    alphas_cumprod = torch.tensor(diffusion.alphas_cumprod, device=device, dtype=torch.float32)

    # Map to actual diffusion timestep indices
    timestep_map = diffusion.timestep_map if hasattr(diffusion, 'timestep_map') else list(range(diffusion.original_num_steps))

    for i, t_idx in enumerate(timesteps):
        t_actual = timestep_map[t_idx]
        t_tensor = torch.full((batch_size,), t_actual, device=device, dtype=torch.long)
        t_prev_idx = timesteps[i + 1] if i + 1 < len(timesteps) else -1

        with torch.no_grad():
            # Decide method for this step
            use_full_cfg = False
            if hybrid_steps is not None and t_idx in hybrid_steps:
                use_full_cfg = True

            if method == 'no_guidance':
                eps_pred = model(x, t_tensor, y_cond)
            elif method == 'cfg' or use_full_cfg:
                eps_pred = forward_cfg(model, x, t_tensor, y_cond, y_uncond, cfg_scale)
            elif method == 'csg':
                eps_pred = forward_csg(model, x, t_tensor, y_cond, y_uncond, cfg_scale, per_layer_weights)
            elif method == 'esg':
                eps_pred = forward_esg(model, x, t_tensor, y_cond, y_uncond, cfg_scale)
            elif method == 'csg_hybrid':
                if use_full_cfg:
                    eps_pred = forward_cfg(model, x, t_tensor, y_cond, y_uncond, cfg_scale)
                else:
                    eps_pred = forward_csg(model, x, t_tensor, y_cond, y_uncond, cfg_scale, per_layer_weights)

        # DDIM step - index alphas_cumprod by step index, not model timestep
        alpha_t = alphas_cumprod[t_idx]
        if t_prev_idx >= 0:
            alpha_prev = alphas_cumprod[t_prev_idx]
        else:
            alpha_prev = torch.tensor(1.0, device=device)

        # Handle learned sigma (DiT outputs 8 channels)
        if eps_pred.shape[1] == 8:
            eps, _ = eps_pred.chunk(2, dim=1)
        else:
            eps = eps_pred

        x_0_pred = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -10, 10)
        x = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * eps

    return x


def get_hybrid_steps(num_steps, ratio, position='middle'):
    """Get the timestep indices that should use full CFG in hybrid mode.

    Note: In the sampling loop, timesteps go from num_steps-1 (noisy) down to 0 (clean).
    The linearity analysis shows errors are highest at low step indices (clean end).

    position options:
      'middle': CFG steps centered in the middle of the step index range
      'early_clean': CFG at lowest step indices (clean timesteps, end of sampling, highest error)
      'early_noisy': CFG at highest step indices (noisy timesteps, start of sampling)
    """
    num_cfg_steps = max(1, int(num_steps * ratio))
    if position == 'middle':
        center = num_steps // 2
        start = max(0, center - num_cfg_steps // 2)
        end = min(num_steps, start + num_cfg_steps)
        return set(range(start, end))
    elif position == 'early_clean':
        # Use full CFG at step indices 0..num_cfg_steps-1 (clean end, highest error)
        return set(range(0, num_cfg_steps))
    elif position == 'early_noisy':
        # Use full CFG at step indices (num_steps-num_cfg_steps)..num_steps-1 (noisy end)
        return set(range(num_steps - num_cfg_steps, num_steps))
    return set()


def get_per_layer_weights(schedule, num_layers=28, mean_w=4.0):
    """Generate per-layer guidance weight schedules."""
    if schedule == 'uniform':
        return [mean_w] * num_layers
    elif schedule == 'decreasing':
        # Linear from 6.0 to 2.0
        weights = np.linspace(6.0, 2.0, num_layers).tolist()
    elif schedule == 'increasing':
        # Linear from 2.0 to 6.0
        weights = np.linspace(2.0, 6.0, num_layers).tolist()
    elif schedule == 'bell':
        # Bell curve centered at middle
        x = np.linspace(-2, 2, num_layers)
        bell = np.exp(-x**2)
        # Normalize to have mean = mean_w, range from 2.0 to 6.0
        bell = 2.0 + 4.0 * (bell - bell.min()) / (bell.max() - bell.min())
        # Adjust mean
        weights = (bell * mean_w / bell.mean()).tolist()
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    # Ensure mean is correct
    w_array = np.array(weights)
    w_array = w_array * (mean_w / w_array.mean())
    return w_array.tolist()
