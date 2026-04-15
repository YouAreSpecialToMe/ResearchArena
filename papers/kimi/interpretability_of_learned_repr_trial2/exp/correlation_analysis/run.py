"""Correlation Analysis: Metrics vs. IFS.

Measure correlation between traditional SAE metrics, output scores, and IFS
to quantify the causal-semantic disconnect.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets
from exp.shared.ifs import compute_ifs_for_features


def compute_traditional_metrics(model, sae, hook_point, prompts):
    """Compute traditional SAE metrics for each feature.
    
    Returns:
        Dict with mean_activation, l0_contribution, decoder_norm
    """
    print("  Computing traditional metrics...")
    
    tokens = model.to_tokens(prompts, truncate=True)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        acts = cache[hook_point]
        feature_acts = sae.encode(acts)  # [batch, seq, d_sae]
        
        # Mean activation
        mean_activation = feature_acts.mean(dim=[0, 1]).cpu().numpy()  # [d_sae]
        
        # L0 contribution (fraction of tokens where feature is active)
        l0_contribution = (feature_acts > 0).float().mean(dim=[0, 1]).cpu().numpy()
        
        # Decoder norm
        decoder_norm = sae.W_dec.norm(dim=-1).cpu().numpy()
        
        # Reconstruction MSE contribution (simplified)
        reconstructed = sae.decode(feature_acts)
        per_feature_mse = []
        for feat_idx in range(sae.cfg.d_sae):
            # Contribution of this feature to reconstruction
            feat_act = feature_acts[:, :, feat_idx:feat_idx+1]
            feat_dec = sae.W_dec[feat_idx:feat_idx+1, :]
            contribution = (feat_act @ feat_dec).norm(dim=-1).mean().item()
            per_feature_mse.append(contribution)
        reconstruction_contrib = np.array(per_feature_mse)
    
    return {
        "mean_activation": mean_activation,
        "l0_contribution": l0_contribution,
        "decoder_norm": decoder_norm,
        "reconstruction_contrib": reconstruction_contrib
    }


def compute_output_scores(model, sae, hook_point, prompts, target_token_ids):
    """Compute output scores (Arad et al. style sufficiency scores).
    
    Output score measures how much feature affects target token probabilities.
    """
    print("  Computing output scores...")
    
    tokens = model.to_tokens(prompts, truncate=True)
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
        acts = cache[hook_point]
        feature_acts = sae.encode(acts)
        
        # Get baseline logits
        logits = model(tokens)
        
        # Output score for each feature
        # Simplified: mean activation * alignment with target direction
        output_scores = []
        
        # Get target direction in logit space
        target_one_hot = torch.zeros(logits.shape[-1], device=logits.device)
        target_one_hot[target_token_ids] = 1.0 / len(target_token_ids)
        
        for feat_idx in range(sae.cfg.d_sae):
            # Decoder direction
            dec_dir = sae.W_dec[feat_idx]
            
            # Mean activation
            mean_act = feature_acts[:, :, feat_idx].mean().item()
            
            # Simplified output score: activation * decoder norm
            output_score = mean_act * dec_dir.norm().item()
            output_scores.append(output_score)
        
        output_scores = np.array(output_scores)
    
    return output_scores


def compute_correlation_matrix(metrics_dict):
    """Compute correlation matrix between all metrics."""
    metric_names = list(metrics_dict.keys())
    n_metrics = len(metric_names)
    
    # Spearman correlations
    spearman_corr = np.zeros((n_metrics, n_metrics))
    spearman_pval = np.zeros((n_metrics, n_metrics))
    
    # Pearson correlations  
    pearson_corr = np.zeros((n_metrics, n_metrics))
    pearson_pval = np.zeros((n_metrics, n_metrics))
    
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            x = metrics_dict[name_i]
            y = metrics_dict[name_j]
            
            # Spearman
            sr, sp = spearmanr(x, y)
            spearman_corr[i, j] = sr
            spearman_pval[i, j] = sp
            
            # Pearson
            pr, pp = pearsonr(x, y)
            pearson_corr[i, j] = pr
            pearson_pval[i, j] = pp
    
    return {
        "metric_names": metric_names,
        "spearman_correlation": spearman_corr.tolist(),
        "spearman_pvalue": spearman_pval.tolist(),
        "pearson_correlation": pearson_corr.tolist(),
        "pearson_pvalue": pearson_pval.tolist()
    }


def compute_precision_at_k(metric_a, metric_b, k_values=[10, 50, 100, 500]):
    """Compute Precision@k: fraction of top-k by metric A in top-k by metric B."""
    results = {}
    
    for k in k_values:
        top_k_a = set(np.argsort(metric_a)[-k:])
        top_k_b = set(np.argsort(metric_b)[-k:])
        
        precision = len(top_k_a & top_k_b) / k
        results[f"k={k}"] = float(precision)
    
    return results


def identify_disconnect_features(metrics_dict, ifs_scores, n_top=100):
    """Identify features with high traditional scores but low IFS."""
    
    # High activation, low IFS
    high_act_threshold = np.percentile(metrics_dict["mean_activation"], 90)
    low_ifs_threshold = np.percentile(ifs_scores, 50)
    
    high_act_low_ifs = np.where(
        (metrics_dict["mean_activation"] > high_act_threshold) & 
        (ifs_scores < low_ifs_threshold)
    )[0]
    
    # High output score, low IFS
    high_out_threshold = np.percentile(metrics_dict["output_score"], 90)
    high_out_low_ifs = np.where(
        (metrics_dict["output_score"] > high_out_threshold) & 
        (ifs_scores < low_ifs_threshold)
    )[0]
    
    return {
        "high_activation_low_ifs_count": int(len(high_act_low_ifs)),
        "high_activation_low_ifs_indices": high_act_low_ifs[:n_top].tolist(),
        "high_output_score_low_ifs_count": int(len(high_out_low_ifs)),
        "high_output_score_low_ifs_indices": high_out_low_ifs[:n_top].tolist(),
    }


def main():
    print("=" * 60)
    print("Correlation Analysis: Metrics vs. IFS")
    print("=" * 60)
    
    set_seed(42)
    device = "cuda"
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    all_pairs = datasets["truthfulqa_pairs"]
    prompts = [p["positive"] for p in all_pairs[:200]]
    
    print(f"  Using {len(prompts)} prompts")
    
    # Define target tokens
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    target_token_ids = true_tokens
    
    start_time = time.time()
    
    # Compute traditional metrics
    traditional = compute_traditional_metrics(model, sae, hook_point, prompts)
    
    # Compute output scores
    output_scores = compute_output_scores(model, sae, hook_point, prompts, target_token_ids)
    
    # Compute IFS components
    print("  Computing IFS components...")
    n_half = len(prompts) // 2
    ifs_results = compute_ifs_for_features(
        model, sae, hook_point,
        prompts[:n_half], prompts[n_half:],
        target_token_ids
    )
    
    necessity = ifs_results["necessity"]
    sufficiency = ifs_results["sufficiency"]
    consistency = ifs_results["consistency"]
    ifs = ifs_results["ifs"]
    
    print(f"    IFS: mean={ifs.mean():.6f}, max={ifs.max():.6f}")
    
    # Build metrics dictionary
    metrics_dict = {
        "mean_activation": traditional["mean_activation"],
        "l0_contribution": traditional["l0_contribution"],
        "decoder_norm": traditional["decoder_norm"],
        "reconstruction_contrib": traditional["reconstruction_contrib"],
        "output_score": output_scores,
        "necessity": necessity,
        "sufficiency": sufficiency,
        "consistency": consistency,
        "ifs": ifs
    }
    
    # Compute correlation matrix
    print("  Computing correlations...")
    correlation_matrix = compute_correlation_matrix(metrics_dict)
    
    # Compute Precision@k
    print("  Computing Precision@k...")
    precision_results = {
        "ifs_vs_mean_activation": compute_precision_at_k(ifs, traditional["mean_activation"]),
        "ifs_vs_output_score": compute_precision_at_k(ifs, output_scores),
        "ifs_vs_decoder_norm": compute_precision_at_k(ifs, traditional["decoder_norm"]),
        "output_score_vs_mean_activation": compute_precision_at_k(output_scores, traditional["mean_activation"]),
    }
    
    # Identify disconnect features
    print("  Identifying disconnect features...")
    metrics_dict["output_score"] = output_scores
    disconnect = identify_disconnect_features(metrics_dict, ifs)
    
    # Compute summary statistics
    print("  Computing summary statistics...")
    
    # Key correlations
    ifs_idx = correlation_matrix["metric_names"].index("ifs")
    output_idx = correlation_matrix["metric_names"].index("output_score")
    activation_idx = correlation_matrix["metric_names"].index("mean_activation")
    necessity_idx = correlation_matrix["metric_names"].index("necessity")
    sufficiency_idx = correlation_matrix["metric_names"].index("sufficiency")
    
    summary = {
        "ifs_vs_output_score_spearman": float(correlation_matrix["spearman_correlation"][ifs_idx][output_idx]),
        "ifs_vs_mean_activation_spearman": float(correlation_matrix["spearman_correlation"][ifs_idx][activation_idx]),
        "ifs_vs_necessity_spearman": float(correlation_matrix["spearman_correlation"][ifs_idx][necessity_idx]),
        "ifs_vs_sufficiency_spearman": float(correlation_matrix["spearman_correlation"][ifs_idx][sufficiency_idx]),
        "necessity_vs_sufficiency_spearman": float(correlation_matrix["spearman_correlation"][necessity_idx][sufficiency_idx]),
        "output_vs_activation_spearman": float(correlation_matrix["spearman_correlation"][output_idx][activation_idx]),
    }
    
    runtime = time.time() - start_time
    
    # Save results
    output = {
        "experiment": "correlation_analysis",
        "config": {
            "n_prompts": len(prompts),
            "d_sae": sae.cfg.d_sae
        },
        "correlation_matrix": correlation_matrix,
        "precision_at_k": precision_results,
        "disconnect_features": disconnect,
        "summary_statistics": summary,
        "metric_statistics": {
            name: {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max())
            }
            for name, vals in metrics_dict.items()
        },
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/correlation_analysis/results.json", output)
    print(f"\n✓ Results saved to exp/correlation_analysis/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    # Print summary
    print("\nKey Correlations (Spearman):")
    print(f"  IFS vs Output Score:     {summary['ifs_vs_output_score_spearman']:+.3f}")
    print(f"  IFS vs Mean Activation:  {summary['ifs_vs_mean_activation_spearman']:+.3f}")
    print(f"  IFS vs Necessity:        {summary['ifs_vs_necessity_spearman']:+.3f}")
    print(f"  IFS vs Sufficiency:      {summary['ifs_vs_sufficiency_spearman']:+.3f}")
    print(f"  Necessity vs Sufficiency:{summary['necessity_vs_sufficiency_spearman']:+.3f}")
    
    print(f"\nDisconnect Features:")
    print(f"  High activation, low IFS: {disconnect['high_activation_low_ifs_count']}")
    print(f"  High output score, low IFS: {disconnect['high_output_score_low_ifs_count']}")
    
    print(f"\nPrecision@100:")
    print(f"  IFS vs Activation: {precision_results['ifs_vs_mean_activation']['k=100']:.3f}")
    print(f"  IFS vs Output Score: {precision_results['ifs_vs_output_score']['k=100']:.3f}")


if __name__ == "__main__":
    main()
