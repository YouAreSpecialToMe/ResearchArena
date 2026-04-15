"""Compute Intervention Fidelity Score (IFS) components for all features."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets


def compute_necessity_scores(
    model,
    sae,
    hook_point,
    positive_prompts,
    target_token_ids,
    batch_size=8
):
    """Compute necessity scores using attribution patching.
    
    Necessity(f) = E[P(y|x) - P(y|x, f=0)]
    
    Uses gradient approximation to estimate effect of ablating each feature.
    """
    print(f"  Computing necessity on {len(positive_prompts)} prompts...")
    
    all_tokens = model.to_tokens(positive_prompts, truncate=True)
    n_batches = (len(all_tokens) + batch_size - 1) // batch_size
    
    necessity_scores = torch.zeros(sae.cfg.d_sae, device='cpu')
    
    for i in range(n_batches):
        batch_tokens = all_tokens[i*batch_size:(i+1)*batch_size]
        
        # Forward with gradient tracking
        activations = {}
        def cache_hook(acts, hook):
            activations[hook.name] = acts.detach().clone().requires_grad_(True)
            return acts
        
        with torch.enable_grad():
            logits = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(hook_point, cache_hook)]
            )
            
            # Target probability
            probs = F.softmax(logits[:, -1, :], dim=-1)
            target_prob = probs[:, target_token_ids].sum(dim=-1).mean()
            
            # Gradient w.r.t. activations
            acts = activations[hook_point]
            if acts.grad_fn is not None:
                grads = torch.autograd.grad(target_prob, acts, retain_graph=False)[0]
                
                # Approximate necessity: gradient * decoder weight * feature activation
                # This estimates how much removing the feature would decrease target prob
                with torch.no_grad():
                    feature_acts = sae.encode(acts.detach())
                    # Approximate effect of feature on activation
                    grad_norm = grads.norm(dim=-1, keepdim=True)  # [batch, seq, 1]
                    necessity = (grad_norm * feature_acts).mean(dim=[0, 1])  # [d_sae]
                    necessity_scores += necessity.cpu()
    
    necessity_scores /= n_batches
    return torch.clamp(necessity_scores, min=0).numpy()


def compute_sufficiency_scores(
    model,
    sae,
    hook_point,
    neutral_prompts,
    target_token_ids,
    batch_size=8
):
    """Compute sufficiency scores (similar to Arad et al. output score).
    
    Sufficiency(f) = E[P(y|x, f=alpha) - P(y|x)] on neutral contexts
    """
    print(f"  Computing sufficiency on {len(neutral_prompts)} prompts...")
    
    all_tokens = model.to_tokens(neutral_prompts, truncate=True)
    n_batches = (len(all_tokens) + batch_size - 1) // batch_size
    
    sufficiency_scores = torch.zeros(sae.cfg.d_sae, device='cpu')
    
    for i in range(n_batches):
        batch_tokens = all_tokens[i*batch_size:(i+1)*batch_size]
        
        activations = {}
        def cache_hook(acts, hook):
            activations[hook.name] = acts.detach().clone().requires_grad_(True)
            return acts
        
        with torch.enable_grad():
            logits = model.run_with_hooks(
                batch_tokens,
                fwd_hooks=[(hook_point, cache_hook)]
            )
            
            probs = F.softmax(logits[:, -1, :], dim=-1)
            target_prob = probs[:, target_token_ids].sum(dim=-1).mean()
            
            acts = activations[hook_point]
            if acts.grad_fn is not None:
                grads = torch.autograd.grad(target_prob, acts, retain_graph=False)[0]
                
                with torch.no_grad():
                    feature_acts = sae.encode(acts.detach())
                    # Sufficiency: how much activating feature increases target prob
                    grad_norm = grads.norm(dim=-1, keepdim=True)
                    sufficiency = (grad_norm * feature_acts).mean(dim=[0, 1])
                    sufficiency_scores += sufficiency.cpu()
    
    sufficiency_scores /= n_batches
    return torch.clamp(sufficiency_scores, min=0).numpy()


def compute_consistency_scores(
    necessity_scores_list,
    n_bootstrap=5
):
    """Compute consistency as stability of scores across bootstrap samples."""
    print(f"  Computing consistency from {len(necessity_scores_list)} samples...")
    
    # Stack scores
    scores = np.stack(necessity_scores_list)  # [n_samples, d_sae]
    
    # Variance across samples
    variance = scores.var(axis=0)  # [d_sae]
    max_var = variance.max()
    
    if max_var > 0:
        consistency = 1.0 - (variance / max_var)
    else:
        consistency = np.ones(scores.shape[1])
    
    return np.clip(consistency, 0, 1)


def compute_ifs(necessity, sufficiency, consistency):
    """Compute Intervention Fidelity Score."""
    # Normalize to [0, 1]
    nec_norm = np.clip(necessity / (necessity.max() + 1e-8), 0, 1)
    suf_norm = np.clip(sufficiency / (sufficiency.max() + 1e-8), 0, 1)
    
    # IFS = sqrt(Nec * Suf) * Cons
    ifs = np.sqrt(nec_norm * suf_norm + 1e-8) * consistency
    
    return ifs


def main():
    print("=" * 60)
    print("Computing Intervention Fidelity Score (IFS)")
    print("=" * 60)
    
    set_seed(42)
    device = "cuda"
    
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    # Use TruthfulQA pairs
    all_pairs = datasets["truthfulqa_pairs"]
    
    # Split: use positive (true) and neutral prompts
    positive_prompts = [p["positive"] for p in all_pairs[:200]]
    neutral_prompts = [p["neutral"] for p in all_pairs[:200]]
    
    print(f"  Positive prompts: {len(positive_prompts)}")
    print(f"  Neutral prompts: {len(neutral_prompts)}")
    
    # Target tokens
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    target_token_ids = true_tokens
    
    print("\nComputing IFS components...")
    start_time = time.time()
    
    # Compute necessity on positive prompts
    necessity = compute_necessity_scores(
        model, sae, hook_point, positive_prompts, target_token_ids
    )
    
    # Compute sufficiency on neutral prompts
    sufficiency = compute_sufficiency_scores(
        model, sae, hook_point, neutral_prompts, target_token_ids
    )
    
    # Compute consistency by splitting data
    print("  Computing consistency...")
    n_split = len(positive_prompts) // 2
    nec_1 = compute_necessity_scores(
        model, sae, hook_point, positive_prompts[:n_split], target_token_ids
    )
    nec_2 = compute_necessity_scores(
        model, sae, hook_point, positive_prompts[n_split:], target_token_ids
    )
    suf_1 = compute_sufficiency_scores(
        model, sae, hook_point, neutral_prompts[:n_split], target_token_ids
    )
    suf_2 = compute_sufficiency_scores(
        model, sae, hook_point, neutral_prompts[n_split:], target_token_ids
    )
    
    consistency_nec = compute_consistency_scores([nec_1, nec_2])
    consistency_suf = compute_consistency_scores([suf_1, suf_2])
    consistency = (consistency_nec + consistency_suf) / 2
    
    # Compute full IFS
    print("  Computing full IFS...")
    ifs = compute_ifs(necessity, sufficiency, consistency)
    
    runtime = time.time() - start_time
    
    # Save results
    results = {
        "experiment": "ifs_computation",
        "config": {
            "n_positive": len(positive_prompts),
            "n_neutral": len(neutral_prompts),
            "d_sae": sae.cfg.d_sae
        },
        "scores": {
            "necessity": necessity.tolist(),
            "sufficiency": sufficiency.tolist(),
            "consistency": consistency.tolist(),
            "ifs": ifs.tolist()
        },
        "statistics": {
            "necessity_mean": float(necessity.mean()),
            "necessity_std": float(necessity.std()),
            "sufficiency_mean": float(sufficiency.mean()),
            "sufficiency_std": float(sufficiency.std()),
            "consistency_mean": float(consistency.mean()),
            "consistency_std": float(consistency.std()),
            "ifs_mean": float(ifs.mean()),
            "ifs_std": float(ifs.std()),
            "ifs_max": float(ifs.max()),
            "top_10_ifs_indices": [int(i) for i in np.argsort(ifs)[-10:]],
            "top_10_ifs_values": [float(ifs[i]) for i in np.argsort(ifs)[-10:]]
        },
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/ifs_computation/results.json", results)
    print(f"\n✓ Results saved to exp/ifs_computation/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    print("\nIFS Statistics:")
    print(f"  Necessity:    {necessity.mean():.6f} ± {necessity.std():.6f}")
    print(f"  Sufficiency:  {sufficiency.mean():.6f} ± {sufficiency.std():.6f}")
    print(f"  Consistency:  {consistency.mean():.6f} ± {consistency.std():.6f}")
    print(f"  IFS:          {ifs.mean():.6f} ± {ifs.std():.6f}")
    print(f"  Max IFS:      {ifs.max():.6f}")
    print(f"  Features with IFS > 0.1: {(ifs > 0.1).sum()}")


if __name__ == "__main__":
    main()
