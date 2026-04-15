"""
Coherent Particle Sampling (CoPS): Verifier-Free Inference-Time Scaling for Diffusion Models.

Core implementation of:
1. PCSTracker - tracks prediction coherence score across denoising steps
2. ParticleResampler - resamples particles proportional to exp(alpha * PCS)
3. CoPS pipeline wrapper - wraps diffusers pipeline for particle-based sampling
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class PCSTracker:
    """Tracks Prediction Coherence Score (PCS) for a set of particles.

    PCS measures how smoothly the model's denoised prediction x_hat_0
    evolves across timesteps. More coherent (smoother) predictions
    indicate lower ODE discretization error and typically higher quality.
    """
    num_particles: int
    distance_metric: str = "l2"  # "l2", "cosine"
    timestep_weights: str = "uniform"  # "uniform", "mid_emphasis", "early_emphasis", "late_emphasis"
    total_steps: int = 50

    # Internal state
    prev_predictions: Optional[torch.Tensor] = field(default=None, repr=False)
    pcs_scores: Optional[torch.Tensor] = field(default=None, repr=False)
    step_coherences: list = field(default_factory=list, repr=False)
    step_count: int = 0

    def reset(self):
        self.prev_predictions = None
        self.pcs_scores = None
        self.step_coherences = []
        self.step_count = 0

    def _get_weight(self, step_idx: int) -> float:
        """Get the weight for a given step index."""
        if self.timestep_weights == "uniform":
            return 1.0
        elif self.timestep_weights == "mid_emphasis":
            mid = self.total_steps / 2
            sigma = self.total_steps / 5
            return float(np.exp(-0.5 * ((step_idx - mid) / sigma) ** 2))
        elif self.timestep_weights == "early_emphasis":
            return float(np.exp(-step_idx / (self.total_steps / 5)))
        elif self.timestep_weights == "late_emphasis":
            return float(np.exp(-(self.total_steps - step_idx) / (self.total_steps / 5)))
        else:
            return 1.0

    def _compute_distance(self, pred_curr: torch.Tensor, pred_prev: torch.Tensor) -> torch.Tensor:
        """Compute distance between current and previous predictions for each particle.

        Args:
            pred_curr: (K, C, H, W) current denoised predictions
            pred_prev: (K, C, H, W) previous denoised predictions

        Returns:
            distances: (K,) distance for each particle
        """
        K = pred_curr.shape[0]
        if self.distance_metric == "l2":
            diff = (pred_curr - pred_prev).reshape(K, -1)
            return torch.norm(diff, dim=1)
        elif self.distance_metric == "cosine":
            curr_flat = pred_curr.reshape(K, -1)
            prev_flat = pred_prev.reshape(K, -1)
            cos_sim = F.cosine_similarity(curr_flat, prev_flat, dim=1)
            return 1.0 - cos_sim  # distance = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def update(self, denoised_predictions: torch.Tensor) -> torch.Tensor:
        """Update PCS with new denoised predictions.

        Args:
            denoised_predictions: (K, C, H, W) predicted x_0 for each particle

        Returns:
            pcs_scores: (K,) current cumulative PCS for each particle
        """
        K = denoised_predictions.shape[0]

        if self.pcs_scores is None:
            self.pcs_scores = torch.zeros(K, device=denoised_predictions.device)

        if self.prev_predictions is not None:
            distances = self._compute_distance(denoised_predictions, self.prev_predictions)
            weight = self._get_weight(self.step_count)
            # PCS is negative distance (higher = more coherent)
            coherence = -distances * weight
            self.pcs_scores += coherence
            self.step_coherences.append(coherence.detach().cpu())

        self.prev_predictions = denoised_predictions.detach().clone()
        self.step_count += 1

        return self.pcs_scores.clone()


class ParticleResampler:
    """Resamples particles proportional to exp(alpha * PCS)."""

    def __init__(self, alpha: float = 1.0, sigma_jitter: float = 0.01):
        self.alpha = alpha
        self.sigma_jitter = sigma_jitter

    def resample(self, latents: torch.Tensor, pcs_scores: torch.Tensor,
                 generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resample particles based on PCS scores.

        Args:
            latents: (K, C, H, W) current latent states
            pcs_scores: (K,) PCS scores for each particle
            generator: random generator for reproducibility

        Returns:
            resampled_latents: (K, C, H, W) resampled and jittered latents
            indices: (K,) which original particles were selected
        """
        K = latents.shape[0]

        # Compute resampling weights
        log_weights = self.alpha * pcs_scores
        log_weights = log_weights - log_weights.max()  # numerical stability
        weights = torch.softmax(log_weights, dim=0)

        # Multinomial resampling
        indices = torch.multinomial(weights, K, replacement=True, generator=generator)
        resampled = latents[indices].clone()

        # Add jitter to duplicated particles to maintain diversity
        unique_indices = indices.unique()
        if len(unique_indices) < K:
            # Find which resampled particles are duplicates
            counts = torch.zeros(K, device=latents.device)
            for idx in indices:
                counts[idx] += 1

            # Add noise to all but the first copy of each duplicated particle
            seen = set()
            for i in range(K):
                orig_idx = indices[i].item()
                if orig_idx in seen:
                    noise = torch.randn_like(resampled[i]) * self.sigma_jitter
                    resampled[i] = resampled[i] + noise
                else:
                    seen.add(orig_idx)

        return resampled, indices


def cops_sample(
    pipe,
    prompt: str,
    num_particles: int = 4,
    num_inference_steps: int = 50,
    resample_interval: int = 10,
    alpha: float = 1.0,
    sigma_jitter: float = 0.01,
    distance_metric: str = "l2",
    timestep_weights: str = "uniform",
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    return_all_particles: bool = False,
    custom_timesteps: Optional[List[int]] = None,
) -> Dict:
    """Run CoPS sampling with a diffusers pipeline.

    Generates K particles in parallel, tracking PCS and resampling at intervals.

    Args:
        pipe: StableDiffusionPipeline (must be on GPU, float16)
        prompt: text prompt
        num_particles: K, number of parallel trajectories
        num_inference_steps: steps per trajectory
        resample_interval: R, resample every R steps
        alpha: resampling temperature
        sigma_jitter: noise added to duplicated particles
        distance_metric: "l2" or "cosine"
        timestep_weights: "uniform", "mid_emphasis", etc.
        guidance_scale: CFG scale
        seed: random seed
        height, width: image dimensions
        return_all_particles: if True, return all K final images
        custom_timesteps: optional custom timestep schedule (for ASA)

    Returns:
        dict with keys: "image", "all_images", "pcs_scores", "pcs_trajectories",
                        "selected_index", "resampling_history"
    """
    device = pipe.device
    K = num_particles

    # Initialize PCS tracker and resampler
    pcs_tracker = PCSTracker(
        num_particles=K,
        distance_metric=distance_metric,
        timestep_weights=timestep_weights,
        total_steps=num_inference_steps,
    )
    resampler = ParticleResampler(alpha=alpha, sigma_jitter=sigma_jitter)

    # Encode prompt
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    # Unconditional embeddings for CFG
    uncond_inputs = pipe.tokenizer(
        "", padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]

    # Duplicate for K particles
    text_emb = text_embeddings.repeat(K, 1, 1)  # (K, seq_len, dim)
    uncond_emb = uncond_embeddings.repeat(K, 1, 1)

    # Set up scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    if custom_timesteps is not None:
        timesteps = torch.tensor(custom_timesteps, device=device)

    # Initialize K particles from different noise
    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (K, pipe.unet.config.in_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float16)
    latents = latents * pipe.scheduler.init_noise_sigma

    resampling_history = []
    pcs_trajectories = []

    # Denoising loop
    for step_idx, t in enumerate(timesteps):
        # Expand timestep for K particles
        t_batch = t.expand(K)

        # CFG: concatenate unconditional and conditional
        latent_model_input = torch.cat([latents, latents], dim=0)
        t_model = torch.cat([t_batch, t_batch])
        prompt_embeds = torch.cat([uncond_emb, text_emb], dim=0)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input, t_model,
                encoder_hidden_states=prompt_embeds
            ).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute denoised prediction x_hat_0
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t.long()].to(device)
        beta_prod_t = 1 - alpha_prod_t
        denoised_pred = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

        # Update PCS
        pcs_scores = pcs_tracker.update(denoised_pred)
        pcs_trajectories.append(pcs_scores.detach().cpu().numpy().copy())

        # Resample at intervals (not at first or last step)
        if (step_idx + 1) % resample_interval == 0 and step_idx > 0 and step_idx < len(timesteps) - 1:
            latents_before = latents.clone()
            latents, indices = resampler.resample(latents, pcs_scores, generator=generator)
            resampling_history.append({
                "step": step_idx,
                "timestep": t.item(),
                "pcs_scores": pcs_scores.detach().cpu().numpy().tolist(),
                "selected_indices": indices.cpu().numpy().tolist(),
            })
            # Reset PCS tracker after resampling (start fresh for next interval)
            # Actually, we keep accumulating PCS but update prev_predictions
            # for resampled particles
            pcs_tracker.prev_predictions = None  # Reset for clean tracking

        # Scheduler step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode all particles
    latents_decoded = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        images = pipe.vae.decode(latents_decoded).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    # Select best particle by final PCS
    best_idx = pcs_scores.argmax().item()
    best_image = images[best_idx]

    result = {
        "image": best_image,
        "pcs_scores": pcs_scores.detach().cpu().numpy().tolist(),
        "pcs_trajectories": np.array(pcs_trajectories).tolist(),
        "selected_index": best_idx,
        "resampling_history": resampling_history,
    }

    if return_all_particles:
        result["all_images"] = images

    return result


def generate_particles_batch(
    pipe,
    prompt: str,
    num_particles: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    track_pcs: bool = True,
    distance_metric: str = "l2",
) -> Dict:
    """Generate K particles WITHOUT resampling, tracking PCS for each.

    This is the shared generation function - particles are generated once,
    then different selection criteria (random, PCS, CLIP, ImageReward) are
    applied post-hoc.

    This is more efficient than running cops_sample because we can reuse
    the same particles for all selection methods.

    Args:
        pipe: StableDiffusionPipeline
        prompt: text prompt
        num_particles: K
        num_inference_steps: steps
        guidance_scale: CFG scale
        seed: random seed
        height, width: image dimensions
        track_pcs: whether to compute PCS
        distance_metric: for PCS computation

    Returns:
        dict with: "images" (K, 3, H, W), "pcs_scores" (K,),
                   "pcs_trajectories", "latents_per_step"
    """
    device = pipe.device
    K = num_particles

    pcs_tracker = PCSTracker(
        num_particles=K,
        distance_metric=distance_metric,
        total_steps=num_inference_steps,
    ) if track_pcs else None

    # Encode prompt
    text_inputs = pipe.tokenizer(
        prompt, padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]

    uncond_inputs = pipe.tokenizer(
        "", padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]

    text_emb = text_embeddings.repeat(K, 1, 1)
    uncond_emb = uncond_embeddings.repeat(K, 1, 1)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    generator = torch.Generator(device=device).manual_seed(seed)
    latent_shape = (K, pipe.unet.config.in_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float16)
    latents = latents * pipe.scheduler.init_noise_sigma

    pcs_trajectories = []
    step_denoised_preds = []  # For post-hoc PCS with different metrics

    for step_idx, t in enumerate(timesteps):
        t_batch = t.expand(K)
        latent_model_input = torch.cat([latents, latents], dim=0)
        t_model = torch.cat([t_batch, t_batch])
        prompt_embeds = torch.cat([uncond_emb, text_emb], dim=0)

        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input, t_model,
                encoder_hidden_states=prompt_embeds
            ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute denoised prediction
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t.long()].to(device)
        beta_prod_t = 1 - alpha_prod_t
        denoised_pred = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

        if pcs_tracker is not None:
            pcs_scores = pcs_tracker.update(denoised_pred)
            pcs_trajectories.append(pcs_scores.detach().cpu().numpy().copy())

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Decode
    latents_decoded = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        images = pipe.vae.decode(latents_decoded).sample
    images = (images / 2 + 0.5).clamp(0, 1)

    result = {
        "images": images,  # (K, 3, H, W) tensor
    }

    if pcs_tracker is not None:
        result["pcs_scores"] = pcs_tracker.pcs_scores.detach().cpu().numpy().tolist()
        result["pcs_trajectories"] = np.array(pcs_trajectories).tolist()
        result["step_coherences"] = [c.numpy().tolist() for c in pcs_tracker.step_coherences]

    return result


def cops_sample_with_resampling(
    pipe,
    prompt: str,
    num_particles: int = 4,
    num_inference_steps: int = 50,
    resample_interval: int = 10,
    alpha: float = 1.0,
    sigma_jitter: float = 0.01,
    distance_metric: str = "l2",
    timestep_weights: str = "uniform",
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
) -> Dict:
    """CoPS with active resampling during generation.

    Unlike generate_particles_batch (no resampling), this applies
    resampling during generation, which changes particle trajectories.
    """
    return cops_sample(
        pipe=pipe,
        prompt=prompt,
        num_particles=num_particles,
        num_inference_steps=num_inference_steps,
        resample_interval=resample_interval,
        alpha=alpha,
        sigma_jitter=sigma_jitter,
        distance_metric=distance_metric,
        timestep_weights=timestep_weights,
        guidance_scale=guidance_scale,
        seed=seed,
        height=height,
        width=width,
        return_all_particles=True,
    )
