"""Intervention Fidelity Score computation using efficient attribution patching."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_ifs(necessity, sufficiency, consistency):
    """Compute Intervention Fidelity Score from components."""
    nec = np.clip(necessity, 1e-8, None)
    suf = np.clip(sufficiency, 1e-8, None)
    cons = np.clip(consistency, 1e-8, 1.0)
    return np.sqrt(nec * suf) * cons


def compute_ifs_for_features_efficient(
    model,
    sae,
    hook_point: str,
    positive_prompts: List[str],
    neutral_prompts: List[str],
    target_token_ids: List[int],
    device: str = "cuda",
    n_top_candidates: int = 2000  # Only compute IFS for top candidates
) -> Dict[str, np.ndarray]:
    """Compute IFS efficiently using activation-based pre-filtering.
    
    Instead of computing IFS for all 24k features, we:
    1. Compute proxy scores for all features (fast)
    2. Select top candidates
    3. Compute actual IFS only for candidates
    
    Args:
        model: HookedTransformer
        sae: SAE model
        hook_point: Hook point name
        positive_prompts: Prompts showing target behavior
        neutral_prompts: Neutral prompts for sufficiency
        target_token_ids: Target token IDs
        device: Device
        n_top_candidates: Number of top candidates to compute full IFS for
        
    Returns:
        Dict with 'necessity', 'sufficiency', 'consistency', 'ifs' arrays
    """
    model.eval()
    sae.eval()
    
    d_sae = sae.cfg.d_sae
    print(f"  Computing proxy scores for all {d_sae} features...")
    
    # Step 1: Compute proxy scores for all features (fast)
    pos_tokens = model.to_tokens(positive_prompts[:50], truncate=True).to(device)
    neu_tokens = model.to_tokens(neutral_prompts[:50], truncate=True).to(device)
    
    with torch.no_grad():
        # Get activations
        _, cache_pos = model.run_with_cache(pos_tokens, names_filter=[hook_point])
        pos_acts = cache_pos[hook_point]
        pos_feature_acts = sae.encode(pos_acts)
        
        _, cache_neu = model.run_with_cache(neu_tokens, names_filter=[hook_point])
        neu_acts = cache_neu[hook_point]
        neu_feature_acts = sae.encode(neu_acts)
        
        # Mean activations
        pos_mean_act = pos_feature_acts.mean(dim=[0, 1])  # [d_sae]
        neu_mean_act = neu_feature_acts.mean(dim=[0, 1])  # [d_sae]
        
        # Decoder norms
        decoder_norms = sae.W_dec.norm(dim=-1)  # [d_sae]
        
        # Proxy necessity: activation on positive prompts * decoder norm
        proxy_necessity = pos_mean_act * decoder_norms
        
        # Proxy sufficiency: decoder norm * (1 - activation on neutral)
        proxy_sufficiency = decoder_norms * torch.clamp(1 - neu_mean_act, min=0.1)
        
        # Proxy IFS (geometric mean)
        proxy_ifs = torch.sqrt(proxy_necessity * proxy_sufficiency + 1e-8)
        
        # Select top candidates
        top_k = torch.topk(proxy_ifs, min(n_top_candidates, d_sae))
        candidate_indices = top_k.indices.cpu().numpy()
        
        print(f"    Selected top {len(candidate_indices)} candidates by proxy IFS")
    
    # Step 2: Compute actual IFS for candidates using direct patching
    print(f"  Computing actual IFS for top {len(candidate_indices)} candidates...")
    
    necessity_scores = np.zeros(d_sae)
    sufficiency_scores = np.zeros(d_sae)
    consistency_scores = np.zeros(d_sae)
    
    # Batch processing for efficiency
    batch_size = 16
    
    for i in range(0, len(candidate_indices), batch_size):
        batch_indices = candidate_indices[i:i+batch_size]
        
        for feat_idx in batch_indices:
            feat_idx = int(feat_idx)
            
            # Compute necessity (ablation effect on positive prompts)
            with torch.no_grad():
                sample_tokens = pos_tokens[:4]
                
                # Baseline
                logits_orig = model(sample_tokens)
                probs_orig = F.softmax(logits_orig[:, -1, :], dim=-1)
                baseline_prob = probs_orig[:, target_token_ids].sum(dim=-1).mean()
                
                # Ablated
                def ablate_hook(acts, hook):
                    f_acts = sae.encode(acts)
                    f_acts[:, :, feat_idx] = 0
                    return sae.decode(f_acts)
                
                logits_abl = model.run_with_hooks(
                    sample_tokens,
                    fwd_hooks=[(hook_point, ablate_hook)]
                )
                probs_abl = F.softmax(logits_abl[:, -1, :], dim=-1)
                ablated_prob = probs_abl[:, target_token_ids].sum(dim=-1).mean()
                
                necessity = max(0, (baseline_prob - ablated_prob).item())
                necessity_scores[feat_idx] = necessity
                
                # Compute sufficiency (injection effect on neutral prompts)
                sample_tokens_neu = neu_tokens[:4]
                
                logits_orig_neu = model(sample_tokens_neu)
                probs_orig_neu = F.softmax(logits_orig_neu[:, -1, :], dim=-1)
                baseline_prob_neu = probs_orig_neu[:, target_token_ids].sum(dim=-1).mean()
                
                def inject_hook(acts, hook):
                    f_acts = sae.encode(acts)
                    f_acts[:, :, feat_idx] += 10.0
                    return sae.decode(f_acts)
                
                logits_inj = model.run_with_hooks(
                    sample_tokens_neu,
                    fwd_hooks=[(hook_point, inject_hook)]
                )
                probs_inj = F.softmax(logits_inj[:, -1, :], dim=-1)
                injected_prob = probs_inj[:, target_token_ids].sum(dim=-1).mean()
                
                sufficiency = max(0, (injected_prob - baseline_prob_neu).item())
                sufficiency_scores[feat_idx] = sufficiency
        
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(candidate_indices):
            print(f"    Progress: {min(i+batch_size, len(candidate_indices))}/{len(candidate_indices)}")
    
    # Compute consistency as normalized inverse variance
    print(f"  Computing consistency scores...")
    for feat_idx in candidate_indices:
        feat_idx = int(feat_idx)
        
        # Measure effect variance across different prompt samples
        effects = []
        with torch.no_grad():
            for j in range(0, min(8, len(pos_tokens)), 2):
                sample = pos_tokens[j:j+1]
                
                logits_orig = model(sample)
                probs_orig = F.softmax(logits_orig[:, -1, :], dim=-1)
                baseline = probs_orig[:, target_token_ids].sum(dim=-1).mean().item()
                
                def inject_hook_var(acts, hook):
                    f_acts = sae.encode(acts)
                    f_acts[:, :, feat_idx] += 5.0
                    return sae.decode(f_acts)
                
                logits_inj = model.run_with_hooks(
                    sample,
                    fwd_hooks=[(hook_point, inject_hook_var)]
                )
                probs_inj = F.softmax(logits_inj[:, -1, :], dim=-1)
                injected = probs_inj[:, target_token_ids].sum(dim=-1).mean().item()
                
                effects.append(injected - baseline)
        
        if len(effects) > 1:
            variance = np.var(effects)
            consistency = 1.0 / (1.0 + variance * 100)
        else:
            consistency = 0.5
        
        consistency_scores[feat_idx] = consistency
    
    # Normalize scores
    if necessity_scores.max() > 0:
        necessity_scores = necessity_scores / (necessity_scores.max() + 1e-8)
    if sufficiency_scores.max() > 0:
        sufficiency_scores = sufficiency_scores / (sufficiency_scores.max() + 1e-8)
    
    # Compute IFS
    ifs_scores = compute_ifs(necessity_scores, sufficiency_scores, consistency_scores)
    
    print(f"\n  IFS Statistics:")
    print(f"    Necessity: mean={necessity_scores.mean():.4f}, max={necessity_scores.max():.4f}")
    print(f"    Sufficiency: mean={sufficiency_scores.mean():.4f}, max={sufficiency_scores.max():.4f}")
    print(f"    Consistency: mean={consistency_scores.mean():.4f}, max={consistency_scores.max():.4f}")
    print(f"    IFS: mean={ifs_scores.mean():.6f}, max={ifs_scores.max():.6f}")
    print(f"    Non-zero IFS: {(ifs_scores > 0).sum()}/{d_sae}")
    print(f"    Features with IFS > 0.01: {(ifs_scores > 0.01).sum()}")
    print(f"    Features with IFS > 0.1: {(ifs_scores > 0.1).sum()}")
    
    return {
        "necessity": necessity_scores,
        "sufficiency": sufficiency_scores,
        "consistency": consistency_scores,
        "ifs": ifs_scores,
        "candidate_indices": candidate_indices
    }


def compute_ifs_for_features(
    model,
    sae,
    hook_point: str,
    positive_prompts: List[str],
    neutral_prompts: List[str],
    target_token_ids: List[int],
    device: str = "cuda",
    **kwargs
) -> Dict[str, np.ndarray]:
    """Main entry point for IFS computation - uses efficient version."""
    return compute_ifs_for_features_efficient(
        model, sae, hook_point,
        positive_prompts, neutral_prompts,
        target_token_ids, device,
        n_top_candidates=kwargs.get('n_top_candidates', 2000)
    )
