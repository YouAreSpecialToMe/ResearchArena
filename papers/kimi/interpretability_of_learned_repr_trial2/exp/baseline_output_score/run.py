"""Output Score Baseline (Arad et al., 2025)."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import time
import numpy as np
import torch

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets
from exp.shared.steering import evaluate_steering_on_prompts, select_features_by_output_score


def run_output_score_baseline(
    model,
    sae,
    hook_point,
    target_prompts,
    test_prompts,
    target_token_ids,
    k_values,
    seeds,
    coefficient
):
    """Run output score baseline with proper seed variation."""
    results = []
    
    for seed in seeds:
        # IMPORTANT: Set seed BEFORE feature selection for proper variation
        set_seed(seed)
        
        for k in k_values:
            print(f"  Running k={k}, seed={seed}...")
            
            # Select features by output score - now varies by seed
            selected_features = select_features_by_output_score(
                model, sae, hook_point, target_prompts, target_token_ids, k=k, seed=seed
            )
            
            # Evaluate steering
            metrics = evaluate_steering_on_prompts(
                model, sae, hook_point,
                test_prompts, selected_features,
                target_token_ids, coefficient
            )
            
            results.append({
                "k": k,
                "seed": seed,
                "method": "output_score_baseline",
                "selected_features": selected_features,
                **metrics
            })
    
    return results


def main():
    print("=" * 60)
    print("Output Score Baseline (Arad et al., 2025)")
    print("=" * 60)
    
    # Hyperparameters
    seeds = [42, 43, 44]
    k_values = [5, 10, 20, 50, 100, 200]
    coefficient = 20.0
    device = "cuda"
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    # Use truthfulqa pairs
    all_pairs = datasets["truthfulqa_pairs"]
    
    # Split: first 80% for feature selection, last 20% for testing
    split_idx = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    target_prompts = [p["positive"] for p in train_pairs]
    test_prompts = [p["positive"] for p in test_pairs]
    
    print(f"  Target prompts: {len(target_prompts)}")
    print(f"  Test prompts: {len(test_prompts)}")
    
    # Define target tokens (True/Yes for truthfulness)
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    target_token_ids = true_tokens
    
    print(f"\nRunning experiments...")
    print(f"  Seeds: {seeds}")
    print(f"  K values: {k_values}")
    
    start_time = time.time()
    
    results = run_output_score_baseline(
        model, sae, hook_point,
        target_prompts, test_prompts,
        target_token_ids, k_values, seeds, coefficient
    )
    
    runtime = time.time() - start_time
    
    # Aggregate results by k
    aggregated = {}
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        if k_results:
            aggregated[f"k={k}"] = {
                "target_change_mean": float(np.mean([r["target_change"] for r in k_results])),
                "target_change_std": float(np.std([r["target_change"] for r in k_results])),
                "relative_change_mean": float(np.mean([r["relative_change"] for r in k_results])),
                "relative_change_std": float(np.std([r["relative_change"] for r in k_results])),
                "baseline_target_prob_mean": float(np.mean([r["baseline_target_prob"] for r in k_results])),
                "steered_target_prob_mean": float(np.mean([r["steered_target_prob"] for r in k_results])),
                "n_seeds": len(k_results)
            }
    
    # Save results
    output = {
        "experiment": "output_score_baseline",
        "config": {
            "seeds": seeds,
            "k_values": k_values,
            "coefficient": coefficient,
            "train_size": len(target_prompts),
            "test_size": len(test_prompts)
        },
        "raw_results": results,
        "aggregated": aggregated,
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/baseline_output_score/results.json", output)
    print(f"\n✓ Results saved to exp/baseline_output_score/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    # Print summary
    print("\nSummary (mean ± std across seeds):")
    for k in k_values:
        if f"k={k}" in aggregated:
            agg = aggregated[f"k={k}"]
            print(f"  k={k:3d}: target_change={agg['target_change_mean']:.4f} ± {agg['target_change_std']:.4f}")


if __name__ == "__main__":
    main()
