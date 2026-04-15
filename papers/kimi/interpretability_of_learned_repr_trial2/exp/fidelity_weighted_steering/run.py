"""Fidelity-Weighted Steering using actual IFS scores."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import time
import numpy as np
import torch
import random

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets
from exp.shared.steering import evaluate_steering_on_prompts, select_features_by_ifs
from exp.shared.ifs import compute_ifs_for_features


def run_fidelity_weighted_steering(
    model,
    sae,
    hook_point,
    target_prompts,
    test_prompts_all,
    target_token_ids,
    k_values,
    seeds,
    coefficient,
    beta=0.5
):
    """Run fidelity-weighted steering experiment with seed-varying test sets."""
    results = []
    
    # Compute IFS once (deterministic - feature importance doesn't change with seed)
    print("  Computing IFS scores for all features...")
    
    # Use subset of prompts for IFS computation
    n_ifs_prompts = min(100, len(target_prompts))
    ifs_prompts_pos = target_prompts[:n_ifs_prompts]
    ifs_prompts_neu = target_prompts[n_ifs_prompts:2*n_ifs_prompts] if len(target_prompts) > n_ifs_prompts else target_prompts[:n_ifs_prompts]
    
    ifs_results = compute_ifs_for_features(
        model, sae, hook_point,
        ifs_prompts_pos, ifs_prompts_neu,
        target_token_ids,
        use_direct_patching=True
    )
    
    ifs_scores = ifs_results["ifs"]
    necessity_scores = ifs_results["necessity"]
    sufficiency_scores = ifs_results["sufficiency"]
    consistency_scores = ifs_results["consistency"]
    
    print(f"    IFS computed: mean={ifs_scores.mean():.6f}, max={ifs_scores.max():.6f}")
    print(f"    Necessity: mean={necessity_scores.mean():.4f}, max={necessity_scores.max():.4f}")
    print(f"    Sufficiency: mean={sufficiency_scores.mean():.4f}, max={sufficiency_scores.max():.4f}")
    print(f"    Consistency: mean={consistency_scores.mean():.4f}, max={consistency_scores.max():.4f}")
    
    # Features with high IFS
    high_ifs_count = (ifs_scores > 0.01).sum()
    print(f"    Features with IFS > 0.01: {high_ifs_count}")
    
    for seed in seeds:
        # IMPORTANT: Set seed for variation in test prompt sampling
        set_seed(seed)
        
        # Sample different test prompts for each seed (creates variation)
        test_sample_size = min(30, len(test_prompts_all))
        test_indices = np.random.choice(len(test_prompts_all), test_sample_size, replace=False)
        test_prompts = [test_prompts_all[i] for i in test_indices]
        
        print(f"  Seed {seed}: Using {len(test_prompts)} test prompts (sampled from {len(test_prompts_all)})")
        
        for k in k_values:
            print(f"    Running k={k}...")
            
            # Select features by IFS
            selected_features = select_features_by_ifs(ifs_scores, k=k, threshold=0.0)
            
            # Evaluate with IFS weighting
            metrics = evaluate_steering_on_prompts(
                model, sae, hook_point,
                test_prompts, selected_features,
                target_token_ids, coefficient,
                weight_by_ifs=True, ifs_scores=ifs_scores, beta=beta
            )
            
            results.append({
                "k": k,
                "seed": seed,
                "method": "fidelity_weighted",
                "selected_features": selected_features,
                "test_prompt_indices": test_indices.tolist(),
                **metrics
            })
    
    return results, ifs_results


def main():
    print("=" * 60)
    print("Fidelity-Weighted Steering (FWS)")
    print("=" * 60)
    
    # Hyperparameters
    seeds = [42, 43, 44]
    k_values = [5, 10, 20, 50, 100, 200]
    coefficient = 20.0
    beta = 0.5
    device = "cuda"
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    # Use truthfulqa pairs
    all_pairs = datasets["truthfulqa_pairs"]
    
    # Split: first 80% for feature selection/IFS computation, last 20% held-out for testing
    split_idx = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    target_prompts = [p["positive"] for p in train_pairs]
    test_prompts_all = [p["positive"] for p in test_pairs]
    
    print(f"  Target prompts (for IFS computation): {len(target_prompts)}")
    print(f"  Test prompts pool (sampled per seed): {len(test_prompts_all)}")
    
    # Define target tokens
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    target_token_ids = true_tokens
    
    print(f"\nRunning experiments...")
    print(f"  Seeds: {seeds}")
    print(f"  K values: {k_values}")
    print(f"  Beta: {beta}")
    
    start_time = time.time()
    
    results, ifs_results = run_fidelity_weighted_steering(
        model, sae, hook_point,
        target_prompts, test_prompts_all,
        target_token_ids, k_values, seeds, coefficient, beta
    )
    
    runtime = time.time() - start_time
    
    # Aggregate results by k
    aggregated = {}
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        if k_results:
            target_changes = [r["target_change"] for r in k_results]
            aggregated[f"k={k}"] = {
                "target_change_mean": float(np.mean(target_changes)),
                "target_change_std": float(np.std(target_changes)),
                "relative_change_mean": float(np.mean([r["relative_change"] for r in k_results])),
                "relative_change_std": float(np.std([r["relative_change"] for r in k_results])),
                "baseline_target_prob_mean": float(np.mean([r["baseline_target_prob"] for r in k_results])),
                "steered_target_prob_mean": float(np.mean([r["steered_target_prob"] for r in k_results])),
                "n_seeds": len(k_results)
            }
    
    # Save results
    output = {
        "experiment": "fidelity_weighted_steering",
        "method": "ifs_based_selection_with_ifs_weighting",
        "config": {
            "seeds": seeds,
            "k_values": k_values,
            "coefficient": coefficient,
            "beta": beta,
            "train_size": len(target_prompts),
            "test_size": len(test_prompts_all)
        },
        "ifs_statistics": {
            "necessity_mean": float(ifs_results["necessity"].mean()),
            "necessity_std": float(ifs_results["necessity"].std()),
            "necessity_max": float(ifs_results["necessity"].max()),
            "sufficiency_mean": float(ifs_results["sufficiency"].mean()),
            "sufficiency_std": float(ifs_results["sufficiency"].std()),
            "sufficiency_max": float(ifs_results["sufficiency"].max()),
            "consistency_mean": float(ifs_results["consistency"].mean()),
            "consistency_std": float(ifs_results["consistency"].std()),
            "ifs_mean": float(ifs_results["ifs"].mean()),
            "ifs_std": float(ifs_results["ifs"].std()),
            "ifs_max": float(ifs_results["ifs"].max()),
            "features_with_ifs_gt_0.01": int((ifs_results["ifs"] > 0.01).sum()),
            "features_with_ifs_gt_0.1": int((ifs_results["ifs"] > 0.1).sum()),
        },
        "raw_results": results,
        "aggregated": aggregated,
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/fidelity_weighted_steering/results.json", output)
    print(f"\n✓ Results saved to exp/fidelity_weighted_steering/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    # Print summary
    print("\nSummary (mean ± std across seeds):")
    for k in k_values:
        if f"k={k}" in aggregated:
            agg = aggregated[f"k={k}"]
            print(f"  k={k:3d}: target_change={agg['target_change_mean']:.4f} ± {agg['target_change_std']:.4f}")


if __name__ == "__main__":
    main()
