"""Baseline 1: Random Feature Selection for Steering."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets
from exp.shared.steering import evaluate_steering_on_prompts, create_steering_hook


def run_random_steering_experiment(
    model,
    sae,
    hook_point,
    test_prompts,
    target_token_ids,
    k_values,
    seeds,
    coefficient
):
    """Run random feature steering experiment across seeds and k values."""
    results = []
    d_sae = sae.cfg.d_sae
    
    for seed in seeds:
        set_seed(seed)
        
        for k in k_values:
            print(f"  Running k={k}, seed={seed}...")
            
            # Random feature selection
            selected_features = random.sample(range(d_sae), k=k)
            
            # Evaluate steering
            metrics = evaluate_steering_on_prompts(
                model, sae, hook_point,
                test_prompts, selected_features,
                target_token_ids, coefficient
            )
            
            results.append({
                "k": k,
                "seed": seed,
                "method": "random",
                **metrics
            })
    
    return results


def main():
    print("=" * 60)
    print("Baseline 1: Random Feature Selection")
    print("=" * 60)
    
    # Hyperparameters
    seeds = [42, 43, 44]
    k_values = [10, 20, 50, 100]
    coefficient = 20.0
    device = "cuda"
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    # Use truthfulqa pairs for testing
    test_pairs = datasets["truthfulqa_pairs"][:100]  # Use subset for speed
    test_prompts = [p["positive"] for p in test_pairs]
    
    # Define target tokens (for truthfulness, we want true answers)
    # Simplified: use tokens associated with common true/false responses
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    false_tokens = [model.to_single_token("False"), model.to_single_token("No")]
    target_token_ids = true_tokens
    
    print(f"\nRunning experiments...")
    print(f"  Seeds: {seeds}")
    print(f"  K values: {k_values}")
    print(f"  Test prompts: {len(test_prompts)}")
    
    start_time = time.time()
    
    results = run_random_steering_experiment(
        model, sae, hook_point,
        test_prompts, target_token_ids,
        k_values, seeds, coefficient
    )
    
    runtime = time.time() - start_time
    
    # Aggregate results by k
    aggregated = {}
    for k in k_values:
        k_results = [r for r in results if r["k"] == k]
        aggregated[k] = {
            "target_change_mean": np.mean([r["target_change"] for r in k_results]),
            "target_change_std": np.std([r["target_change"] for r in k_results]),
            "relative_change_mean": np.mean([r["relative_change"] for r in k_results]),
            "relative_change_std": np.std([r["relative_change"] for r in k_results]),
        }
    
    # Save results
    output = {
        "experiment": "baseline_random",
        "method": "random_feature_selection",
        "config": {
            "seeds": seeds,
            "k_values": k_values,
            "coefficient": coefficient,
            "d_sae": sae.cfg.d_sae
        },
        "raw_results": results,
        "aggregated": aggregated,
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/baseline_random/results.json", output)
    print(f"\n✓ Results saved to exp/baseline_random/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    # Print summary
    print("\nSummary (mean ± std):")
    for k in k_values:
        agg = aggregated[k]
        print(f"  k={k:3d}: target_change={agg['target_change_mean']:.4f} ± {agg['target_change_std']:.4f}")


if __name__ == "__main__":
    main()
