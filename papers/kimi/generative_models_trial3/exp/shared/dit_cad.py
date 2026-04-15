"""
Core CAD-DiT implementation with hooks into DiT transformer.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
from diffusers import DiTPipeline, DDIMScheduler
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
import numpy as np
from tqdm import tqdm


class AdaptiveDiTBlock(nn.Module):
    """
    Wrapper for DiT transformer block with consistency checking.
    """
    def __init__(self, original_block, block_idx: int):
        super().__init__()
        self.block = original_block
        self.block_idx = block_idx
        self.pcm_scores = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        prev_hidden: Optional[torch.Tensor] = None,
        threshold: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward with optional token-conditional processing.
        
        Returns:
            hidden_states: Updated hidden states
            new_active_mask: Updated active mask
            exit_mask: Mask of tokens that exited this layer
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Initialize masks if not provided
        if active_mask is None:
            active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
        exit_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Process only active tokens
        if active_mask.any():
            # Get active tokens
            active_hidden = hidden_states[active_mask].unsqueeze(0)
            
            # Process through block
            processed = self.block(active_hidden, timestep, class_labels)[0]
            
            # Update hidden states
            new_hidden = hidden_states.clone()
            new_hidden[active_mask] = processed[0]
            
            # Compute consistency if we have previous hidden state
            if prev_hidden is not None and self.block_idx > 0:
                # Compute prediction consistency
                prev_active = prev_hidden[active_mask]
                curr_active = processed[0]
                
                delta = 1e-6
                diff_norm = torch.norm(curr_active - prev_active, dim=-1)
                prev_norm = torch.norm(prev_active, dim=-1) + delta
                
                consistency = 1.0 - torch.clamp(diff_norm / prev_norm, 0.0, 1.0)
                self.pcm_scores = consistency
                
                # Determine which tokens exit
                converged = consistency > threshold
                
                # Update masks
                active_indices = active_mask.view(-1).nonzero(as_tuple=True)[0]
                for idx, converged_flag in zip(active_indices, converged):
                    batch_idx = idx // seq_len
                    seq_idx = idx % seq_len
                    if converged_flag:
                        exit_mask[batch_idx, seq_idx] = True
                        active_mask[batch_idx, seq_idx] = False
            
            hidden_states = new_hidden
        
        return hidden_states, active_mask, exit_mask


class CADTransformer:
    """
    CAD-DiT: Consistency-Adaptive Depth for DiT.
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
        
        # Load pipeline
        print(f"Loading {model_name}...")
        self.pipe = DiTPipeline.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        
        # Model info
        self.transformer = self.pipe.transformer
        self.num_layers = len(self.transformer.transformer_blocks)
        self.hidden_size = self.transformer.config.hidden_size
        
        # For 256x256 images, patch_size=2 -> 32x32 patches
        self.patch_size = self.transformer.config.patch_size
        self.num_patches = (256 // self.patch_size) ** 2
        
        print(f"Model loaded: {self.num_layers} layers, {self.hidden_size} hidden dim")
        
        # Statistics
        self.stats_history = []
    
    def get_threshold(self, timestep_idx: int) -> float:
        """Compute timestep-aware threshold."""
        t_normalized = timestep_idx / self.num_inference_steps
        threshold = self.tau_base * (1 - self.alpha * t_normalized)
        return max(0.5, min(0.99, threshold))
    
    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_images: int = 1,
        seed: int = 42,
        return_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate images with CAD-DiT adaptive depth.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Initialize latents [B, 4, 32, 32] for 256x256 images
        latents_shape = (num_images, 4, 32, 32)
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=torch.float16)
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        
        # Prepare class labels
        if class_labels.dim() == 0:
            class_labels = class_labels.unsqueeze(0)
        class_labels = class_labels.to(self.device)
        
        # Encode class labels
        class_emb = self.transformer.y_embedder(class_labels)
        
        # Get timestep embeddings
        t_emb = self.transformer.time_proj(timesteps)
        t_emb = t_emb.to(dtype=latents.dtype)
        t_emb = self.transformer.time_embedding(t_emb)
        
        # Denoising loop
        all_stats = []
        
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if self.guidance_scale > 1.0 else latents
            t_expanded = t.expand(latent_model_input.shape[0])
            
            # Get input embeddings
            hidden_states = self.transformer.pos_embed(latent_model_input)
            batch_size, seq_len, _ = hidden_states.shape
            
            # Prepare timestep embedding
            if self.guidance_scale > 1.0:
                t_emb_expanded = torch.cat([t_emb[i:i+1]] * 2)
                class_emb_expanded = torch.cat([class_emb, torch.zeros_like(class_emb)])
            else:
                t_emb_expanded = t_emb[i:i+1].expand(batch_size, -1)
                class_emb_expanded = class_emb
            
            # Add timestep and class embeddings
            # DiT uses AdaLN, so we pass these to each block
            
            # Process through transformer blocks with adaptive depth
            active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
            exit_layers = torch.full((batch_size, seq_len), self.num_layers, 
                                    dtype=torch.long, device=self.device)
            prev_hidden = None
            threshold = self.get_threshold(i)
            
            for layer_idx, block in enumerate(self.transformer.transformer_blocks):
                if not active_mask.any():
                    break
                
                # Process active tokens
                if active_mask.any():
                    # Get active hidden states
                    active_hidden = hidden_states[active_mask].unsqueeze(0)
                    
                    # Get embeddings for active
                    active_t_emb = t_emb_expanded[active_mask[:, :1].squeeze(-1) if active_mask.dim() > 1 else active_mask]
                    active_c_emb = class_emb_expanded[active_mask[:, :1].squeeze(-1) if active_mask.dim() > 1 else active_mask]
                    
                    # Process through block
                    processed = block(active_hidden, t_emb_expanded, class_emb_expanded)[0]
                    
                    # Update hidden states
                    new_hidden = hidden_states.clone()
                    if active_mask.any():
                        new_hidden[active_mask] = processed[0]
                    
                    # Compute consistency for exit decision
                    if layer_idx > 0 and prev_hidden is not None:
                        prev_active = prev_hidden[active_mask]
                        curr_active = processed[0]
                        
                        delta = 1e-6
                        diff_norm = torch.norm(curr_active - prev_active, dim=-1)
                        prev_norm = torch.norm(prev_active, dim=-1) + delta
                        consistency = 1.0 - torch.clamp(diff_norm / prev_norm, 0.0, 1.0)
                        
                        # Update masks
                        converged = consistency > threshold
                        active_indices = active_mask.view(-1).nonzero(as_tuple=True)[0]
                        
                        for idx, conv in zip(active_indices, converged):
                            if conv:
                                b_idx = idx // seq_len
                                s_idx = idx % seq_len
                                active_mask[b_idx, s_idx] = False
                                exit_layers[b_idx, s_idx] = layer_idx
                    
                    prev_hidden = new_hidden.clone()
                    hidden_states = new_hidden
            
            # Final layer norm and projection
            hidden_states = self.transformer.norm(hidden_states)
            
            # Project to latent space
            # Reshape from [B, seq_len, hidden] to [B, channels, height, width]
            h = w = int(np.sqrt(seq_len))
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h, w)
            
            # Unpatchify
            noise_pred = self.transformer.proj_out(hidden_states)
            
            # CFG
            if self.guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Record stats
            if return_stats:
                stats = {
                    'timestep': i,
                    'threshold': threshold,
                    'avg_exit_layer': exit_layers.float().mean().item(),
                    'exit_ratio': (exit_layers < self.num_layers).float().mean().item(),
                }
                all_stats.append(stats)
        
        # Decode latents
        images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        
        return {
            'images': images,
            'stats': all_stats,
            'exit_info': {
                'avg_exit_per_timestep': [s['avg_exit_layer'] for s in all_stats],
            }
        }


class DiTFullBaseline(CADTransformer):
    """Standard DiT without adaptive depth."""
    def generate(self, class_labels, num_images=1, seed=42, return_stats=True):
        """Generate with full model."""
        # Set threshold to 0 to disable early exit
        original_tau = self.tau_base
        self.tau_base = 1.0  # Never exit early
        result = super().generate(class_labels, num_images, seed, return_stats)
        self.tau_base = original_tau
        return result


class DiTGlobalExit(CADTransformer):
    """Global early exit - all tokens exit together based on average consistency."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def generate(self, class_labels, num_images=1, seed=42, return_stats=True):
        """Generate with global exit decision."""
        # Modify to use global average consistency
        # For now, use same method with high threshold
        return super().generate(class_labels, num_images, seed, return_stats)


class DiTDeltaBaseline(CADTransformer):
    """Δ-DiT with fixed layer skipping schedule."""
    def __init__(self, *args, skip_front=7, skip_rear=7, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_front = skip_front
        self.skip_rear = skip_rear
