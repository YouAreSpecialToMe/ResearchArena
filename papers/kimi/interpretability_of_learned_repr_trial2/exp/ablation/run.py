"""Component Ablation: Test Individual IFS Components."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import time
import numpy as np
import torch

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets
from exp.shared.steering import evaluate_steering_on_prompts
from exp.shared.ifs import compute_ifs_for_features, compute_ifs


def select_features_by_component(
    component_scores: np.ndarray,
    k: int = 20,
    threshold: float = 0.0
) -> list:
    """Select top-k features by a single component score."""
    import torch as th
    scores = th.tensor(component_scores)
    
    valid_mask = scores > threshold
    valid_indices = th.where(valid_mask)[0]
    
    if len(valid_indices) < k:
        top_k = th.topk(scores, min(k, len(scores)))
    else:
        valid_scores = scores[valid_indices]
        top_k_local = th.topk(valid_scores, k)
        top_k = type('obj', (object,), {
            'indices': valid_indices[top_k_local.indices],
            'values': top_k_local.values
        })()
    
    return top_k.indices.tolist()


def run_component_ablation(
    model,
    sae,
    hook_point,
    target_prompts,
    test_prompts,
    target_token_ids,
    k,
    seeds,
    coefficient
):
    """Run component ablation study with seed variation."""
    results = []
    
    # Compute IFS components (deterministic)
    print("  Computing IFS components...")
    n_ifs_prompts = min(100, len(target_prompts))
    ifs_prompts_pos = target_prompts[:n_ifs_prompts]
    ifs_prompts_neu = target_prompts[n_ifs_prompts:2*n_ifs_prompts] if len(target_prompts) > n_ifs_prompts else target_prompts[:n_ifs_prompts]
    
    ifs_results = compute_ifs_for_features(
        model, sae, hook_point,
        ifs_prompts_pos, ifs_prompts_neu,
        target_token_ids,
        use_direct_patching=True
    )
    
    nec_scores = ifs_results["necessity"]
    suf_scores = ifs_results["sufficiency"]
    cons_scores = ifs_results["consistency"]
    ifs_scores = ifs_results["ifs"]
    
    # Define component combinations
    components = {
        "sufficiency_only": suf_scores,
        "necessity_only": nec_scores,
        "consistency_only": cons_scores,
        "sufficiency_necessity": np.sqrt(np.clip(nec_scores * suf_scores, 0, None)),
        "sufficiency_consistency": suf_scores * cons_scores,
        "full_ifs": ifs_scores
    }
    
    for seed in seeds:
        # IMPORTANT: Set seed for test prompt sampling variation
        set_seed(seed)
        
        # Sample test prompts
        test_sample_size = min(30, len(test_prompts))
        test_indices = np.random.choice(len(test_prompts), test_sample_size, replace=False)
        test_prompts_sampled = [test_prompts[i] for i in test_indices]
        
        for comp_name, comp_scores in components.items():
            print(f"  Seed {seed}, Component: {comp_name}...")
            
            # Select features by this component
            selected_features = select_features_by_component(comp_scores, k=k)
            
            # Evaluate steering
            if "full_ifs" in comp_name:
                # Use IFS weighting for full IFS
                metrics = evaluate_steering_on_prompts(
                    model, sae, hook_point,
                    test_prompts_sampled, selected_features,
                    target_token_ids, coefficient,
                    weight_by_ifs=True, ifs_scores=ifs_scores, beta=0.5
                )
            else:
                # No IFS weighting for ablation components
                metrics = evaluate_steering_on_prompts(
                    model, sae, hook_point,
                    test_prompts_sampled, selected_features,
                    target_token_ids, coefficient
                )
            
            results.append({
                "component": comp_name,
                "k": k,
                "seed": seed,
                **metrics
            })
    
    return results, components


def main():
    print("=" * 60)
    print("Component Ablation Study")
    print("=" * 60)
    
    # Hyperparameters
    seeds = [42, 43, 44]
    k = 20  # Fixed k for ablation
    coefficient = 20.0
    device = "cuda"
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    all_pairs = datasets["truthfulqa_pairs"]
    
    # Split
    split_idx = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    test_pairs = all_pairs[split_idx:]
    
    target_prompts = [p["positive"] for p in train_pairs]
    test_prompts = [p["positive"] for p in test_pairs]
    
    print(f"  Target prompts: {len(target_prompts)}")
    print(f"  Test prompts: {len(test_prompts)}")
    
    # Define target tokens
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    target_token_ids = true_tokens
    
    print(f"\nRunning ablation experiments...")
    print(f"  k = {k}")
    print(f"  Seeds: {seeds}")
    
    start_time = time.time()
    
    results, components = run_component_ablation(
        model, sae, hook_point,
        target_prompts, test_prompts,
        target_token_ids, k, seeds, coefficient
    )
    
    runtime = time.time() - start_time
    
    # Aggregate results by component
    aggregated = {}
    component_names = list(components.keys())
    
    for comp_name in component_names:
        comp_results = [r for r in results if r["component"] == comp_name]
        if comp_results:
            target_changes = [r["target_change"] for r in comp_results]
            aggregated[comp_name] = {
                "target_change_mean": float(np.mean(target_changes)),
                "target_change_std": float(np.std(target_changes)),
                "relative_change_mean": float(np.mean([r["relative_change"] for r in comp_results])),
                "relative_change_std": float(np.std([r["relative_change"] for r in comp_results])),
                "baseline_target_prob_mean": float(np.mean([r["baseline_target_prob"] for r in comp_results])),
                "steered_target_prob_mean": float(np.mean([r["steered_target_prob"] for r in comp_results])),
                "n_seeds": len(comp_results)
            }
    
    # Save results
    output = {
        "experiment": "component_ablation",
        "config": {
            "k": k,
            "coefficient": coefficient,
            "seeds": seeds,
            "components": component_names
        },
        "aggregated": aggregated,
        "raw_results": results,
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/ablation/results.json", output)
    print(f"\n✓ Results saved to exp/ablation/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    # Print summary
    print("\nComponent Ablation Summary (mean ± std across seeds):")
    for comp_name in component_names:
        if comp_name in aggregated:
            agg = aggregated[comp_name]
            print(f"  {comp_name:25s}: target_change={agg['target_change_mean']:.4f} ± {agg['target_change_std']:.4f}")


if __name__ == "__main__":
    main()
