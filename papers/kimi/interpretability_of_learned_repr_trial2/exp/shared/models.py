"""Model and SAE loading utilities."""
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
from typing import Tuple, Optional


def load_gpt2_small(device: str = "cuda") -> HookedTransformer:
    """Load GPT-2 Small model via TransformerLens."""
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    return model


def load_sae_gpt2_small(
    release: str = "gpt2-small-res-jb",
    sae_id: str = "blocks.8.hook_resid_pre",
    device: str = "cuda"
) -> Tuple[SAE, dict]:
    """Load pretrained SAE for GPT-2 Small.
    
    Args:
        release: SAE release name from SAELens
        sae_id: SAE identifier (which layer/hook)
        device: Device to load on
        
    Returns:
        Tuple of (SAE model, config dict)
    """
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device
    )
    sae.eval()
    return sae, cfg_dict


def get_sae_activations(
    model: HookedTransformer,
    sae: SAE,
    tokens: torch.Tensor,
    hook_point: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get SAE activations for input tokens.
    
    Args:
        model: HookedTransformer model
        sae: SAE model
        tokens: Input token IDs [batch, seq]
        hook_point: Hook point name (e.g., "blocks.8.hook_resid_pre")
        
    Returns:
        Tuple of (feature_activations, reconstructed_activations)
        feature_activations: [batch, seq, d_sae]
        reconstructed_activations: [batch, seq, d_model]
    """
    # Get model activations at hook point
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        activations = cache[hook_point]  # [batch, seq, d_model]
        
        # Encode through SAE
        feature_acts = sae.encode(activations)  # [batch, seq, d_sae]
        reconstructed = sae.decode(feature_acts)  # [batch, seq, d_model]
        
    return feature_acts, reconstructed


def get_model_and_sae(
    model_name: str = "gpt2-small",
    layer: int = 8,
    device: str = "cuda"
) -> Tuple[HookedTransformer, SAE, dict, str]:
    """Load both model and SAE.
    
    Returns:
        Tuple of (model, sae, cfg_dict, hook_point)
    """
    if model_name == "gpt2-small":
        model = load_gpt2_small(device)
        sae, cfg_dict = load_sae_gpt2_small(
            release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
            device=device
        )
        hook_point = f"blocks.{layer}.hook_resid_pre"
    else:
        raise ValueError(f"Model {model_name} not supported")
        
    return model, sae, cfg_dict, hook_point
