"""
CAD-DiT: Consistency-Adaptive Depth for Diffusion Transformers
Model implementations with token-conditional adaptive depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from diffusers import DiTPipeline, DDIMScheduler
import numpy as np
from tqdm import tqdm


class PredictionConsistencyMetric:
    """
    Prediction Consistency Metric (PCM) for measuring token-level convergence.
    PCM_i^l = 1 - ||ε_i^l - ε_i^(l-1)|| / (||ε_i^(l-1)|| + δ)
    """
    def __init__(self, delta: float = 1e-6):
        self.delta = delta
        self.prev_predictions = None
    
    def reset(self):
        self.prev_predictions = None
    
    def compute(self, current_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency between current and previous predictions.
        
        Args:
            current_pred: [batch, seq_len, dim] current layer prediction
            
        Returns:
            consistency: [batch, seq_len] consistency scores in [0, 1]
        """
        if self.prev_predictions is None:
            # First layer - no consistency to compute
            self.prev_predictions = current_pred.clone()
            return torch.zeros(current_pred.shape[0], current_pred.shape[1], 
                              device=current_pred.device)
        
        # Compute normalized difference
        diff_norm = torch.norm(current_pred - self.prev_predictions, dim=-1)
        prev_norm = torch.norm(self.prev_predictions, dim=-1)
        
        # PCM = 1 - normalized difference
        normalized_diff = diff_norm / (prev_norm + self.delta)
        consistency = 1.0 - torch.clamp(normalized_diff, 0.0, 1.0)
        
        # Update previous prediction
        self.prev_predictions = current_pred.clone()
        
        return consistency


class CADDiTWrapper:
    """
    Wrapper for DiT model with Consistency-Adaptive Depth.
    """
    def __init__(
        self,
        model_name: str = "facebook/DiT-XL-2-256",
        device: str = "cuda",
        tau_base: float = 0.95,
        alpha: float = 0.3,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ):
        self.device = device
        self.tau_base = tau_base
        self.alpha = alpha
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Load model
        self.pipe = DiTPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Get model info
        self.transformer = self.pipe.transformer
        self.num_layers = len(self.transformer.transformer_blocks)
        self.hidden_size = self.transformer.config.hidden_size
        
        # Statistics tracking
        self.exit_stats = {
            'layer_exits': [],  # List of exit layers per token
            'timestep_flops': [],  # FLOPs per timestep
        }
    
    def get_threshold(self, timestep: int) -> float:
        """
        Compute timestep-aware threshold.
        τ(t) = τ_base * (1 - α * t/T)
        """
        t_normalized = timestep / self.num_inference_steps
        threshold = self.tau_base * (1 - self.alpha * t_normalized)
        return threshold
    
    def adaptive_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.Tensor,
        timestep_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Token-conditional forward pass with adaptive depth.
        
        Returns:
            output: Final hidden states
            info: Dictionary with statistics
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize tracking
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        exit_layers = torch.full((batch_size, seq_len), self.num_layers, 
                                 dtype=torch.long, device=device)
        
        # Initialize PCM
        pcm = PredictionConsistencyMetric()
        prev_predictions = None
        
        # Get threshold for this timestep
        threshold = self.get_threshold(timestep_idx)
        
        # Process through transformer blocks with conditional exit
        hidden = hidden_states
        
        for layer_idx, block in enumerate(self.transformer.transformer_blocks):
            # Only process active tokens through this layer
            if active_mask.any():
                # Process current layer
                hidden_active = block(
                    hidden[active_mask].unsqueeze(0),
                    timestep,
                    class_labels,
                )[0]
                
                # Update hidden states for active tokens
                hidden_flat = hidden.view(-1, hidden.shape[-1])
                active_indices = active_mask.view(-1).nonzero(as_tuple=True)[0]
                hidden_flat[active_indices] = hidden_active.view(-1, hidden_active.shape[-1])
                hidden = hidden_flat.view(batch_size, seq_len, -1)
                
                # Compute prediction consistency for active tokens
                if layer_idx > 0:
                    # Get prediction from current hidden state
                    current_pred = hidden[active_mask].unsqueeze(0)
                    
                    if prev_predictions is not None:
                        # Compute consistency
                        consistency = self.compute_consistency(
                            current_pred, prev_predictions[active_mask].unsqueeze(0)
                        )
                        
                        # Check which tokens should exit
                        converged = consistency.squeeze(0) > threshold
                        
                        # Update active mask and exit layers
                        active_flat = active_mask.view(-1)
                        active_indices = active_flat.nonzero(as_tuple=True)[0]
                        for i, idx in enumerate(active_indices):
                            if converged[i]:
                                active_flat[idx] = False
                                exit_layers.view(-1)[idx] = layer_idx
                    
                    # Update previous predictions
                    prev_predictions = hidden.clone()
        
        # Final output
        info = {
            'exit_layers': exit_layers,
            'avg_exit_layer': exit_layers.float().mean().item(),
            'active_ratio': active_mask.float().mean().item(),
        }
        
        return hidden, info
    
    def compute_consistency(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Compute normalized consistency between predictions."""
        delta = 1e-6
        diff_norm = torch.norm(current - previous, dim=-1)
        prev_norm = torch.norm(previous, dim=-1)
        normalized_diff = diff_norm / (prev_norm + delta)
        consistency = 1.0 - torch.clamp(normalized_diff, 0.0, 1.0)
        return consistency
    
    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_images: int = 1,
        seed: int = 42,
        track_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate images with adaptive depth.
        
        Returns:
            Dictionary with images and statistics
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Initialize latents
        latents = torch.randn(
            (num_images, 4, 32, 32),
            generator=generator,
            device=self.device,
            dtype=torch.float16,
        )
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare class embeddings
        class_embeddings = self.transformer.y_embedder(class_labels)
        
        # Denoising loop
        all_stats = []
        
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Get model prediction with adaptive depth
            latent_model_input = torch.cat([latents] * 2) if self.guidance_scale > 1.0 else latents
            
            # TODO: Implement full adaptive forward pass
            # For now, use standard forward
            timestep = t.expand(latent_model_input.shape[0])
            
            noise_pred = self.transformer(
                latent_model_input,
                timestep,
                class_labels if self.guidance_scale <= 1.0 else 
                torch.cat([class_labels, torch.zeros_like(class_labels)]),
            ).sample
            
            # CFG
            if self.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return {
            'images': images,
            'stats': all_stats,
        }


class DiTBaseline:
    """Standard DiT inference baseline."""
    def __init__(
        self,
        model_name: str = "facebook/DiT-XL-2-256",
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        self.pipe = DiTPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
    
    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_images: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate images using standard inference."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        images = []
        for i in range(num_images):
            img = self.pipe(
                class_labels=class_labels[i:i+1],
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]
            images.append(img)
        
        return images


class DeepCacheDiT:
    """DeepCache baseline for DiT."""
    def __init__(
        self,
        model_name: str = "facebook/DiT-XL-2-256",
        device: str = "cuda",
        num_inference_steps: int = 50,
        cache_interval: int = 5,
        guidance_scale: float = 1.0,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.cache_interval = cache_interval
        self.guidance_scale = guidance_scale
        
        self.pipe = DiTPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
    
    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_images: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate images with DeepCache."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Simple implementation: use standard generation
        # Full DeepCache requires more complex feature caching
        images = []
        for i in range(num_images):
            img = self.pipe(
                class_labels=class_labels[i:i+1],
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]
            images.append(img)
        
        return images


class DeltaDiT:
    """Δ-DiT baseline with fixed schedule layer skipping."""
    def __init__(
        self,
        model_name: str = "facebook/DiT-XL-2-256",
        device: str = "cuda",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        skip_front_layers: int = 7,
        skip_rear_layers: int = 7,
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.skip_front_layers = skip_front_layers
        self.skip_rear_layers = skip_rear_layers
        
        self.pipe = DiTPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.num_layers = len(self.pipe.transformer.transformer_blocks)
    
    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_images: int = 1,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate images with fixed layer skipping."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Use standard generation for now
        # Full implementation would modify transformer forward
        images = []
        for i in range(num_images):
            img = self.pipe(
                class_labels=class_labels[i:i+1],
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]
            images.append(img)
        
        return images
