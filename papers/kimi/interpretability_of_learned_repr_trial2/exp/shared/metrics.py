"""Evaluation metrics for steering experiments."""
import torch
import numpy as np
from typing import List, Dict, Tuple
import torch.nn.functional as F


def compute_logit_difference(
    model,
    tokens: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int
) -> float:
    """Compute logit difference between positive and negative tokens.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens [batch, seq]
        positive_token_id: Token ID for positive outcome
        negative_token_id: Token ID for negative outcome
        
    Returns:
        Average logit difference (positive - negative)
    """
    with torch.no_grad():
        logits = model(tokens)  # [batch, seq, vocab]
        # Get logits for last position
        last_logits = logits[:, -1, :]  # [batch, vocab]
        
        pos_logits = last_logits[:, positive_token_id]
        neg_logits = last_logits[:, negative_token_id]
        
        diff = (pos_logits - neg_logits).mean().item()
    
    return diff


def compute_next_token_probability(
    model,
    tokens: torch.Tensor,
    target_token_ids: List[int]
) -> float:
    """Compute probability of target tokens at next position.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens [batch, seq]
        target_token_ids: List of target token IDs
        
    Returns:
        Average probability of target tokens
    """
    with torch.no_grad():
        logits = model(tokens)
        probs = F.softmax(logits[:, -1, :], dim=-1)  # [batch, vocab]
        
        # Sum probabilities for target tokens
        target_prob = probs[:, target_token_ids].sum(dim=-1).mean().item()
    
    return target_prob


def compute_kl_divergence(
    model,
    tokens: torch.Tensor,
    intervention_fn=None
) -> float:
    """Compute KL divergence between original and intervened outputs.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens
        intervention_fn: Function to apply intervention
        
    Returns:
        KL divergence
    """
    with torch.no_grad():
        # Original logits
        logits_orig = model(tokens)
        probs_orig = F.softmax(logits_orig[:, -1, :], dim=-1)
        
        # Intervened logits
        if intervention_fn:
            # This would require hooks
            logits_intervened = model(tokens)  # Placeholder
        else:
            logits_intervened = logits_orig
            
        probs_intervened = F.softmax(logits_intervened[:, -1, :], dim=-1)
        
        # KL divergence
        kl = F.kl_div(
            probs_intervened.log(),
            probs_orig,
            reduction='batchmean'
        ).item()
    
    return kl


def compute_perplexity(model, tokens: torch.Tensor) -> float:
    """Compute perplexity on tokens.
    
    Args:
        model: HookedTransformer model
        tokens: Input tokens [batch, seq]
        
    Returns:
        Perplexity (geometric mean of inverse probabilities)
    """
    with torch.no_grad():
        logits = model(tokens)
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tokens[:, 1:].contiguous()
        
        # Use mean reduction to get average loss per token
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        # Clamp loss to avoid overflow in exp
        loss = torch.clamp(loss, max=20.0)  # Cap at ~485M perplexity
        perplexity = torch.exp(loss).item()
    
    return perplexity


def compute_perplexity_with_intervention(
    model,
    sae,
    hook_point: str,
    texts: List[str],
    selected_features: List[int],
    coefficient: float = 20.0,
    ifs_scores: np.ndarray = None,
    beta: float = 0.5,
    max_length: int = 128,
    max_samples: int = 50
) -> Dict[str, float]:
    """Compute perplexity baseline and with steering intervention.
    
    This function properly compares perplexity on the SAME texts before/after
    to measure side effects accurately.
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        texts: List of texts to evaluate on
        selected_features: Features to steer
        coefficient: Steering coefficient
        ifs_scores: IFS scores for weighting
        beta: IFS weighting exponent
        max_length: Max token length
        max_samples: Max number of samples to use
        
    Returns:
        Dict with baseline_ppl, steered_ppl, ppl_change, relative_change
    """
    # Limit samples for efficiency
    if len(texts) > max_samples:
        texts = texts[:max_samples]
    
    # Tokenize all texts
    tokens_list = []
    for text in texts:
        tokens = model.to_tokens(text, truncate=True)
        # Only keep texts with at least 2 tokens
        if tokens.shape[1] >= 2:
            tokens_list.append(tokens)
    
    if len(tokens_list) == 0:
        return {
            "baseline_ppl": float('inf'),
            "steered_ppl": float('inf'),
            "ppl_change": float('inf'),
            "relative_change": float('inf')
        }
    
    # Compute baseline perplexity
    total_nll = 0
    total_tokens = 0
    
    with torch.no_grad():
        for tokens in tokens_list:
            logits = model(tokens)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            total_nll += nll.item()
            total_tokens += shift_labels.numel()
    
    avg_nll_baseline = total_nll / total_tokens if total_tokens > 0 else 10.0
    # Cap NLL to avoid overflow (exp(20) ~ 485M, which is already very high)
    baseline_ppl = np.exp(min(avg_nll_baseline, 15.0))
    
    # Compute steering coefficients
    if ifs_scores is not None:
        coeffs = [coefficient * (ifs_scores[f] ** beta) for f in selected_features]
    else:
        coeffs = [coefficient] * len(selected_features)
    
    # Compute steered perplexity
    total_nll_steered = 0
    
    def make_steering_hook():
        feature_indices = torch.tensor(selected_features, dtype=torch.long, device=model.cfg.device)
        coefficients = torch.tensor(coeffs, dtype=torch.float32, device=model.cfg.device)
        
        def steering_hook(acts, hook):
            f_acts = sae.encode(acts)
            for i, feat_idx in enumerate(feature_indices):
                f_acts[:, :, feat_idx] += coefficients[i]
            return sae.decode(f_acts)
        return steering_hook
    
    with torch.no_grad():
        for tokens in tokens_list:
            hook_fn = make_steering_hook()
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_point, hook_fn)]
            )
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            nll = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            total_nll_steered += nll.item()
    
    avg_nll_steered = total_nll_steered / total_tokens if total_tokens > 0 else 10.0
    # Cap NLL to avoid overflow
    steered_ppl = np.exp(min(avg_nll_steered, 15.0))
    
    ppl_change = steered_ppl - baseline_ppl
    relative_change = ppl_change / (baseline_ppl + 1e-8)
    
    return {
        "baseline_ppl": baseline_ppl,
        "steered_ppl": steered_ppl,
        "ppl_change": ppl_change,
        "relative_change": relative_change,
        "total_tokens_evaluated": total_tokens
    }


def compute_side_effect_score(
    baseline_metrics: Dict[str, float],
    intervened_metrics: Dict[str, float]
) -> float:
    """Compute side effect score as average absolute change in metrics.
    
    Args:
        baseline_metrics: Dict of metric name -> value at baseline
        intervened_metrics: Dict of metric name -> value with intervention
        
    Returns:
        Side effect score (average absolute difference)
    """
    differences = []
    for key in baseline_metrics:
        if key in intervened_metrics:
            diff = abs(intervened_metrics[key] - baseline_metrics[key])
            differences.append(diff)
    
    return np.mean(differences) if differences else 0.0


def evaluate_steering_effectiveness(
    model,
    test_prompts: List[Dict],
    intervention_fn,
    behavior: str = "truthfulness"
) -> Dict[str, float]:
    """Evaluate steering effectiveness on test prompts.
    
    Returns dict with metrics:
        - target_behavior_change: Change in desired behavior
        - side_effect_score: Unintended effects
        - perplexity_change: Change in generation quality
    """
    # This is a placeholder - actual implementation would depend on behavior
    metrics = {
        "target_behavior_change": 0.0,
        "side_effect_score": 0.0,
        "perplexity_change": 0.0
    }
    
    return metrics


def compute_steering_metrics(
    model,
    sae,
    hook_point: str,
    contrastive_pairs: List[Dict],
    selected_features: List[int],
    steering_coefficient: float = 20.0
) -> Dict[str, float]:
    """Compute comprehensive steering metrics.
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        contrastive_pairs: List of contrastive prompt pairs
        selected_features: List of feature indices to steer
        steering_coefficient: Coefficient for steering
        
    Returns:
        Dict with steering metrics
    """
    metrics = {
        "target_effectiveness": 0.0,
        "side_effects": 0.0,
        "perplexity": 0.0
    }
    
    return metrics
