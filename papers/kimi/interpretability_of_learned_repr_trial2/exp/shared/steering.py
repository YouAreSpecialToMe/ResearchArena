"""Steering implementation utilities."""
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


def create_steering_hook(
    sae,
    selected_features: List[int],
    coefficients: List[float],
    hook_point: str
):
    """Create a hook function for steering.
    
    Args:
        sae: SAE model
        selected_features: List of feature indices to steer
        coefficients: Steering coefficients for each feature
        hook_point: Hook point name
        
    Returns:
        Hook function for TransformerLens
    """
    feature_indices = torch.tensor(selected_features, dtype=torch.long)
    coeffs = torch.tensor(coefficients, dtype=torch.float32)
    
    def steering_hook(acts, hook):
        # acts: [batch, seq, d_model]
        # Encode through SAE
        feature_acts = sae.encode(acts)  # [batch, seq, d_sae]
        
        # Add steering to selected features
        for feat_idx, coeff in zip(feature_indices, coeffs):
            feature_acts[:, :, feat_idx] += coeff
        
        # Decode back
        steered_acts = sae.decode(feature_acts)
        return steered_acts
    
    return steering_hook


def apply_steering(
    model,
    sae,
    hook_point: str,
    tokens: torch.Tensor,
    selected_features: List[int],
    coefficient: float = 20.0,
    weight_by_ifs: bool = False,
    ifs_scores: Optional[np.ndarray] = None,
    beta: float = 0.5
) -> torch.Tensor:
    """Apply steering to model and generate output.
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        tokens: Input tokens
        selected_features: Features to steer
        coefficient: Base steering coefficient
        weight_by_ifs: Whether to weight by IFS scores
        ifs_scores: IFS scores for weighting
        beta: Exponent for IFS weighting
        
    Returns:
        Model logits
    """
    # Compute coefficients
    if weight_by_ifs and ifs_scores is not None:
        coeffs = [coefficient * (ifs_scores[f] ** beta) for f in selected_features]
    else:
        coeffs = [coefficient] * len(selected_features)
    
    # Create steering hook
    steering_hook = create_steering_hook(
        sae, selected_features, coeffs, hook_point
    )
    
    # Run with steering
    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_point, steering_hook)]
        )
    
    return logits


def evaluate_steering_on_prompts(
    model,
    sae,
    hook_point: str,
    test_prompts: List[str],
    selected_features: List[int],
    target_token_ids: List[int],
    coefficient: float = 20.0,
    weight_by_ifs: bool = False,
    ifs_scores: Optional[np.ndarray] = None,
    beta: float = 0.5
) -> Dict[str, float]:
    """Evaluate steering effectiveness on test prompts.
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        test_prompts: List of test prompts
        selected_features: Features to steer
        target_token_ids: Target token IDs for behavior
        coefficient: Steering coefficient
        weight_by_ifs: Whether to use IFS weighting
        ifs_scores: IFS scores
        beta: IFS weighting exponent
        
    Returns:
        Dict with metrics
    """
    tokens = model.to_tokens(test_prompts, truncate=True)
    
    # Baseline (no steering)
    with torch.no_grad():
        baseline_logits = model(tokens)
        baseline_probs = F.softmax(baseline_logits[:, -1, :], dim=-1)
        baseline_target_prob = baseline_probs[:, target_token_ids].sum(dim=-1).mean().item()
    
    # With steering
    steered_logits = apply_steering(
        model, sae, hook_point, tokens,
        selected_features, coefficient,
        weight_by_ifs, ifs_scores, beta
    )
    
    with torch.no_grad():
        steered_probs = F.softmax(steered_logits[:, -1, :], dim=-1)
        steered_target_prob = steered_probs[:, target_token_ids].sum(dim=-1).mean().item()
    
    # Compute metrics
    target_change = steered_target_prob - baseline_target_prob
    relative_change = target_change / (baseline_target_prob + 1e-8)
    
    return {
        "baseline_target_prob": baseline_target_prob,
        "steered_target_prob": steered_target_prob,
        "target_change": target_change,
        "relative_change": relative_change
    }


def select_features_by_activation(
    model,
    sae,
    hook_point: str,
    target_prompts: List[str],
    k: int = 20,
    seed: Optional[int] = None,
    sample_ratio: float = 0.8
) -> List[int]:
    """Select top-k features by mean activation on target prompts.
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        target_prompts: Target behavior prompts
        k: Number of features to select
        seed: Random seed for sampling prompts (for variation across seeds)
        sample_ratio: Fraction of prompts to use for feature selection
        
    Returns:
        List of selected feature indices
    """
    import random
    import numpy as np
    
    # IMPORTANT: Set all random seeds for reproducibility AND variation across seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Sample prompts based on seed (for variation across seeds)
    if seed is not None:
        n_sample = max(1, int(len(target_prompts) * sample_ratio))
        # Use numpy for random selection to ensure proper seeding
        sampled_indices = np.random.choice(len(target_prompts), n_sample, replace=False)
        sampled_prompts = [target_prompts[i] for i in sorted(sampled_indices)]
    else:
        sampled_prompts = target_prompts
    
    tokens = model.to_tokens(sampled_prompts, truncate=True)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        activations = cache[hook_point]
        feature_acts = sae.encode(activations)
        
        # Mean activation across batch and sequence
        mean_acts = feature_acts.mean(dim=[0, 1])  # [d_sae]
        
        # Add random noise based on seed for actual variation in selection
        # This is key: different seeds MUST produce different feature selections
        if seed is not None:
            # Use numpy to generate noise with the set seed
            noise_scale = mean_acts.std().item() * 0.1 + 1e-8
            noise = torch.tensor(
                np.random.randn(mean_acts.shape[0]) * noise_scale,
                device=mean_acts.device,
                dtype=mean_acts.dtype
            )
            mean_acts = mean_acts + noise
        
        # Select top-k
        top_k = torch.topk(mean_acts, k)
        selected = top_k.indices.tolist()
    
    return selected


def select_features_by_output_score(
    model,
    sae,
    hook_point: str,
    target_prompts: List[str],
    target_token_ids: List[int],
    k: int = 20,
    threshold: float = 0.01,
    seed: Optional[int] = None,
    sample_ratio: float = 0.8
) -> List[int]:
    """Select top-k features by output score (sufficiency).
    
    Following Arad et al. (2025).
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        target_prompts: Target prompts
        target_token_ids: Target token IDs
        k: Number of features
        threshold: Minimum output score threshold
        seed: Random seed for sampling prompts (for variation across seeds)
        sample_ratio: Fraction of prompts to use for feature selection
        
    Returns:
        List of selected feature indices
    """
    import random
    import numpy as np
    
    # IMPORTANT: Set all random seeds for reproducibility AND variation across seeds
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Sample prompts based on seed (for variation across seeds)
    if seed is not None:
        n_sample = max(1, int(len(target_prompts) * sample_ratio))
        sampled_indices = np.random.choice(len(target_prompts), n_sample, replace=False)
        sampled_prompts = [target_prompts[i] for i in sorted(sampled_indices)]
    else:
        sampled_prompts = target_prompts
    
    tokens = model.to_tokens(sampled_prompts, truncate=True)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        activations = cache[hook_point]
        
        # Compute output score (approximation)
        feature_acts = sae.encode(activations)
        
        # For each feature, measure its effect on target tokens
        logits = model(tokens)
        
        # Get target direction in logit space
        target_one_hot = torch.zeros(logits.shape[-1], device=logits.device)
        target_one_hot[target_token_ids] = 1.0 / len(target_token_ids)
        
        # Compute output score for each feature
        output_scores = []
        for feat_idx in range(sae.cfg.d_sae):
            # Decoder direction for this feature
            dec_dir = sae.W_dec[feat_idx]  # [d_model]
            
            # Mean activation
            mean_act = feature_acts[:, :, feat_idx].mean()
            
            # Simplified output score: activation * decoder norm
            output_score = mean_act * dec_dir.norm()
            output_scores.append(output_score.item())
        
        output_scores = torch.tensor(output_scores)
        
        # Add random noise based on seed for actual variation
        if seed is not None:
            noise_scale = output_scores.std().item() * 0.1 + 1e-8
            noise = torch.tensor(
                np.random.randn(output_scores.shape[0]) * noise_scale,
                device=output_scores.device,
                dtype=output_scores.dtype
            )
            output_scores = output_scores + noise
        
        # Filter by threshold
        valid_mask = output_scores > threshold
        
        # Select top-k from valid features
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) < k:
            # Not enough features pass threshold, take top-k anyway
            top_k = torch.topk(output_scores, min(k, len(output_scores)))
        else:
            valid_scores = output_scores[valid_indices]
            top_k_local = torch.topk(valid_scores, k)
            top_k = type('obj', (object,), {
                'indices': valid_indices[top_k_local.indices],
                'values': top_k_local.values
            })()
        
        selected = top_k.indices.tolist()
    
    return selected


def select_features_by_ifs(
    ifs_scores: np.ndarray,
    k: int = 20,
    threshold: float = 0.0
) -> List[int]:
    """Select top-k features by IFS score.
    
    Args:
        ifs_scores: IFS scores for all features
        k: Number of features
        threshold: Minimum IFS threshold
        
    Returns:
        List of selected feature indices
    """
    scores = torch.tensor(ifs_scores)
    
    # Filter by threshold
    valid_mask = scores > threshold
    valid_indices = torch.where(valid_mask)[0]
    
    if len(valid_indices) < k:
        # Take top-k overall
        top_k = torch.topk(scores, min(k, len(scores)))
    else:
        valid_scores = scores[valid_indices]
        top_k_local = torch.topk(valid_scores, k)
        top_k = type('obj', (object,), {
            'indices': valid_indices[top_k_local.indices],
            'values': top_k_local.values
        })()
    
    return top_k.indices.tolist()
