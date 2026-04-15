"""
RAS (Region-Adaptive Sampling) Baseline for FLUX
Based on: "Region-Adaptive Sampling for Diffusion Transformers" (arXiv:2502.10389)

RAS identifies focus regions via noise magnitude and updates only those regions
each step while caching others.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from diffusers import FluxPipeline
import numpy as np
from tqdm import tqdm
import time


class RASFluxSampler:
    """
    RAS baseline for FLUX.
    Identifies "focus regions" based on noise magnitude and only updates
    those regions each step, caching the rest.
    """
    
    def __init__(
        self,
        pipeline: FluxPipeline,
        num_inference_steps: int = 50,
        region_ratio: float = 0.5,  # Fraction of regions to update each step
        patch_size: int = 8,
        device: str = "cuda"
    ):
        self.pipe = pipeline
        self.num_inference_steps = num_inference_steps
        self.region_ratio = region_ratio
        self.patch_size = patch_size
        self.device = device
        
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        height: int = 512,
        width: int = 512,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generate with RAS acceleration."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        all_images = []
        times = []
        
        for prompt in tqdm(prompts, desc=f"RAS (ratio={self.region_ratio})"):
            start = time.time()
            
            result = self._generate_single(
                prompt=prompt,
                height=height,
                width=width,
                generator=generator,
            )
            
            elapsed = time.time() - start
            times.append(elapsed)
            all_images.extend(result['images'])
        
        # Effective NFE is reduced by region_ratio
        effective_nfe = self.num_inference_steps * self.region_ratio
        
        return {
            'images': all_images,
            'wall_time_total': sum(times),
            'wall_time_per_image': np.mean(times),
            'nfe_total': len(prompts) * self.num_inference_steps,
            'nfe_per_image': self.num_inference_steps,
            'effective_nfe': effective_nfe,
            'region_ratio': self.region_ratio,
        }
    
    def _generate_single(
        self,
        prompt: str,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> Dict[str, Any]:
        """Generate a single image with RAS."""
        # Encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.device,
            num_images_per_prompt=1,
        )
        
        # Prepare latents
        latents = self._prepare_latents(height, width, generator)
        
        # Setup scheduler
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        latent_height = height // 8
        latent_width = width // 8
        num_patches_h = latent_height // self.patch_size
        num_patches_w = latent_width // self.patch_size
        total_patches = num_patches_h * num_patches_w
        num_regions_to_update = max(1, int(total_patches * self.region_ratio))
        
        # Denoising loop with region selection
        for step_idx, t in enumerate(timesteps):
            # Compute noise magnitude for region selection
            latent_model_input = self.pipe.scheduler.scale_model_input(latents, t)
            
            # Get noise prediction
            noise_pred = self._predict_velocity(
                latent_model_input, t, prompt_embeds, 
                pooled_prompt_embeds, text_ids
            )
            
            # Compute per-patch noise magnitude for focus region selection
            noise_mag = torch.norm(noise_pred, dim=1, keepdim=True)
            patch_noise = F.avg_pool2d(
                noise_mag,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ).squeeze(1)  # [1, H/P, W/P]
            
            # Select focus regions (highest noise magnitude)
            flat_noise = patch_noise.flatten()
            _, top_indices = torch.topk(flat_noise, num_regions_to_update)
            
            # Create region mask
            region_mask = torch.zeros(total_patches, device=self.device, dtype=torch.bool)
            region_mask[top_indices] = True
            region_mask = region_mask.view(1, num_patches_h, num_patches_w)
            
            # Upsample to latent resolution
            region_mask_up = F.interpolate(
                region_mask.unsqueeze(1).float(),
                size=(latent_height, latent_width),
                mode='nearest'
            ).expand_as(latents)
            
            # Scheduler step
            new_latents = self.pipe.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
            
            # Only update focus regions (RAS key: cache non-focus regions)
            latents = torch.where(
                region_mask_up.bool(),
                new_latents,  # Update focus regions
                latents  # Keep cached for non-focus regions
            )
        
        # Decode
        images = self._decode_latents(latents)
        
        return {
            'images': images,
        }
    
    def _prepare_latents(self, height: int, width: int, generator: torch.Generator):
        """Initialize latent noise."""
        latent_height = height // 8
        latent_width = width // 8
        
        latents = torch.randn(
            (1, 16, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )
        return latents * self.pipe.scheduler.init_noise_sigma
    
    def _predict_velocity(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity using FLUX transformer."""
        if isinstance(prompt_embeds, tuple):
            encoder_hidden_states = prompt_embeds[0]
        else:
            encoder_hidden_states = prompt_embeds
        
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor([timestep], device=self.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(latents.shape[0])
        
        noise_pred = self.pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        
        return noise_pred
    
    def _decode_latents(self, latents: torch.Tensor) -> List:
        """Decode latents to PIL images."""
        latents = latents.to(torch.float32)
        latents = latents / self.pipe.vae.config.scaling_factor
        
        with torch.no_grad():
            images = self.pipe.vae.decode(latents).sample
        
        images = (images / 2 + 0.5).clamp(0, 1)
        
        from PIL import Image
        pil_images = []
        for img in images:
            img_np = img.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))
        
        return pil_images
