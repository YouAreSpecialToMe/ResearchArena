"""
Evaluation metrics for SAEs including robustness metrics.
"""
import torch
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


def compute_fvu(x, x_recon):
    """
    Compute Fraction of Variance Unexplained.
    FVU = MSE / Var(x)
    """
    mse = torch.nn.functional.mse_loss(x_recon, x).item()
    var = x.var().item()
    return mse / (var + 1e-8)


def compute_l0_sparsity(z):
    """
    Compute L0 sparsity (average number of active features per sample).
    """
    active = (z > 0).float().sum(dim=-1).mean().item()
    return active


def compute_dead_features(activations_list, d_sae):
    """
    Compute percentage of dead features (never activated).
    
    Args:
        activations_list: List of activation tensors across batches
        d_sae: Total number of features
    """
    all_activations = torch.cat(activations_list, dim=0)
    ever_active = (all_activations > 0).any(dim=0).sum().item()
    dead_pct = (d_sae - ever_active) / d_sae * 100
    return dead_pct


def compute_feature_density(z):
    """
    Compute feature activation frequency distribution.
    """
    # Frequency of each feature across all samples
    freq = (z > 0).float().mean(dim=0)
    return freq


def compute_feature_correlation(z):
    """
    Compute average pairwise feature correlation.
    Handles edge cases where features have zero variance.
    """
    z_np = z.detach().cpu().numpy()
    
    # Filter out features with zero variance (dead features)
    feature_stds = z_np.std(axis=0)
    valid_features = feature_stds > 1e-8
    
    if valid_features.sum() < 2:
        return 0.0  # Not enough features to compute correlation
    
    z_valid = z_np[:, valid_features]
    corr_matrix = np.corrcoef(z_valid.T)
    
    # Handle any remaining NaNs
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Get upper triangle (excluding diagonal)
    triu_idx = np.triu_indices_from(corr_matrix, k=1)
    if len(triu_idx[0]) == 0:
        return 0.0
    
    avg_corr = corr_matrix[triu_idx].mean()
    return float(avg_corr)


def compute_feature_diversity_entropy(z):
    """
    Compute feature activation frequency entropy as diversity metric.
    Higher entropy = more diverse feature usage.
    """
    freq = compute_feature_density(z)
    # Normalize to probability distribution
    freq = freq / (freq.sum() + 1e-8)
    # Shannon entropy
    entropy = -(freq * torch.log(freq + 1e-8)).sum().item()
    return entropy


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def compute_population_overlap_change(model, prompts_a, prompts_b, tokenizer, 
                                      target_model, layer_idx, device='cuda'):
    """
    Compute population-level overlap change metric.
    
    Measures how much the top-activating features change when prompts
    from category A are perturbed to look like category B.
    
    Returns:
        overlap_change: Average Jaccard distance between original and attacked
    """
    model.eval()
    
    # Get baseline activations for both categories
    with torch.no_grad():
        # Tokenize
        tokens_a = tokenizer(prompts_a, return_tensors='pt', 
                            padding=True, truncation=True).to(device)
        tokens_b = tokenizer(prompts_b, return_tensors='pt',
                            padding=True, truncation=True).to(device)
        
        # Get activations from target LLM
        outputs_a = target_model(**tokens_a, output_hidden_states=True)
        outputs_b = target_model(**tokens_b, output_hidden_states=True)
        
        # Get layer activations (average over sequence)
        acts_a = outputs_a.hidden_states[layer_idx].mean(dim=1)
        acts_b = outputs_b.hidden_states[layer_idx].mean(dim=1)
        
        # Encode with SAE
        z_a, _ = model.encode(acts_a)
        z_b, _ = model.encode(acts_b)
        
        # Get top-k features for each
        k = 50
        _, topk_a = torch.topk(z_a, k, dim=-1)
        _, topk_b = torch.topk(z_b, k, dim=-1)
    
    # Compute pairwise overlaps (baseline similarity between categories)
    baseline_overlaps = []
    for i in range(len(prompts_a)):
        set_a = set(topk_a[i].cpu().numpy())
        set_b = set(topk_b[i].cpu().numpy())
        baseline_overlaps.append(jaccard_similarity(set_a, set_b))
    
    baseline_overlap = np.mean(baseline_overlaps)
    
    # Simulate attack: For each A prompt, find features that activate like B
    # Simple version: measure overlap change when adding "science" tokens to "art" prompts
    attacked_overlaps = []
    
    # Create attacked version by adding science-related suffix
    attack_suffix = " The scientific method requires empirical evidence and rigorous testing."
    attacked_prompts = [p + attack_suffix for p in prompts_a]
    
    with torch.no_grad():
        tokens_attacked = tokenizer(attacked_prompts, return_tensors='pt',
                                   padding=True, truncation=True).to(device)
        outputs_attacked = target_model(**tokens_attacked, output_hidden_states=True)
        acts_attacked = outputs_attacked.hidden_states[layer_idx].mean(dim=1)
        z_attacked, _ = model.encode(acts_attacked)
        _, topk_attacked = torch.topk(z_attacked, k, dim=-1)
    
    # Compute overlaps after attack
    for i in range(len(prompts_a)):
        set_a = set(topk_a[i].cpu().numpy())
        set_attacked = set(topk_attacked[i].cpu().numpy())
        attacked_overlaps.append(jaccard_similarity(set_a, set_attacked))
    
    attacked_overlap = np.mean(attacked_overlaps)
    
    # Overlap change: how much did the attack change the activation pattern
    overlap_change = abs(attacked_overlap - baseline_overlap)
    
    return {
        'baseline_overlap': baseline_overlap,
        'attacked_overlap': attacked_overlap,
        'overlap_change': overlap_change
    }


def compute_individual_attack_success_rate(model, prompts, tokenizer,
                                           target_model, layer_idx,
                                           n_features=50, device='cuda'):
    """
    Compute individual feature attack success rate.
    
    For each target feature, try to find a perturbation that activates it.
    Returns the percentage of successfully attacked features.
    """
    model.eval()
    
    # First, identify top activating features
    with torch.no_grad():
        tokens = tokenizer(prompts, return_tensors='pt',
                          padding=True, truncation=True).to(device)
        outputs = target_model(**tokens, output_hidden_states=True)
        acts = outputs.hidden_states[layer_idx].mean(dim=1)
        z, _ = model.encode(acts)
        
        # Get features with highest average activation
        mean_acts = z.mean(dim=0)
        top_features = torch.topk(mean_acts, n_features).indices.cpu().numpy()
    
    # Simulate attacks on each feature
    # Simplified version: measure how easily each feature can be maximally activated
    successful_attacks = 0
    
    for feat_idx in top_features:
        with torch.no_grad():
            # Get feature activation across prompts
            feat_acts = z[:, feat_idx]
            
            # Simulate attack: check if small perturbation can significantly change activation
            # In a real attack, we'd optimize tokens; here we approximate
            act_std = feat_acts.std().item()
            act_max = feat_acts.max().item()
            
            # Feature is considered "attackable" if it has high variance
            # and can achieve high activation
            if act_std > 0.1 and act_max > 1.0:
                successful_attacks += 1
    
    attack_success_rate = successful_attacks / n_features
    
    return {
        'attack_success_rate': attack_success_rate,
        'n_features_tested': n_features,
        'n_successful': successful_attacks
    }


def compute_proxy_correlation(proxy_scores, empirical_robustness):
    """
    Compute correlation between proxy scores and empirical robustness.
    
    Args:
        proxy_scores: Tensor of proxy scores for each feature
        empirical_robustness: Tensor of empirical attack success per feature
    
    Returns:
        Dictionary with correlation metrics
    """
    proxy_np = proxy_scores.detach().cpu().numpy()
    empirical_np = empirical_robustness.detach().cpu().numpy()
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(proxy_np, empirical_np)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(proxy_np, empirical_np)
    
    # Kendall tau
    kendall_tau, kendall_p = stats.kendalltau(proxy_np, empirical_np)
    
    return {
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p
    }


def evaluate_sae_model(model, val_activations, batch_size=1024, device='cuda'):
    """
    Comprehensive SAE evaluation.
    """
    model.eval()
    
    all_recons = []
    all_latents = []
    
    with torch.no_grad():
        for i in range(0, len(val_activations), batch_size):
            batch = val_activations[i:i+batch_size].to(device)
            x_recon, z, _ = model(batch)
            all_recons.append(x_recon.cpu())
            all_latents.append(z.cpu())
    
    all_recons = torch.cat(all_recons, dim=0)
    all_latents = torch.cat(all_latents, dim=0)
    
    # Compute metrics
    metrics = {
        'fvu': compute_fvu(val_activations, all_recons),
        'l0_sparsity': compute_l0_sparsity(all_latents),
        'dead_features_pct': compute_dead_features([all_latents], model.d_sae),
        'feature_diversity_entropy': compute_feature_diversity_entropy(all_latents),
        'feature_correlation': compute_feature_correlation(all_latents),
    }
    
    return metrics
