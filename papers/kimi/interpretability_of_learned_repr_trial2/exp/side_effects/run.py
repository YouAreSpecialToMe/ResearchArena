"""Side Effects Measurement on Held-Out Tasks.

Measure side effects of steering on general capabilities using proper perplexity calculation.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/interpretability_of_learned_representations/idea_01')

import json
import time
import numpy as np
import torch

from exp.shared import set_seed, save_results, get_model_and_sae
from exp.shared.data_loader import load_prepared_datasets
from exp.shared.steering import select_features_by_activation, select_features_by_output_score, select_features_by_ifs
from exp.shared.ifs import compute_ifs_for_features
from exp.shared.metrics import compute_perplexity_with_intervention


def load_hellaswag_texts(n_samples=100):
    """Load HellaSwag contexts as text for perplexity evaluation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hellaswag", split=f"validation[:{n_samples}]", trust_remote_code=True)
        
        texts = []
        for item in ds:
            context = item["ctx"]
            if len(context) > 50:  # Only keep reasonably long contexts
                texts.append(context)
        
        return texts
    except Exception as e:
        print(f"  Warning: Could not load HellaSwag: {e}")
        # Fallback to simple texts
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "The capital of France is Paris.",
            "Water boils at 100 degrees Celsius at sea level.",
        ] * 25


def run_side_effects_experiment(
    model,
    sae,
    hook_point,
    target_prompts,
    test_texts,
    target_token_ids,
    k_values,
    coefficient,
    seeds
):
    """Run side effects experiment comparing all methods with proper perplexity."""
    results = []
    
    # Compute IFS for FWS (deterministic - feature importance)
    print("  Computing IFS for Fidelity-Weighted Steering...")
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
    
    print(f"    IFS: mean={ifs_scores.mean():.6f}, max={ifs_scores.max():.6f}")
    
    for seed in seeds:
        # IMPORTANT: Set seed for proper variation
        set_seed(seed)
        
        # Sample different test texts for each seed
        test_sample_size = min(50, len(test_texts))
        test_indices = np.random.choice(len(test_texts), test_sample_size, replace=False)
        test_texts_sampled = [test_texts[i] for i in test_indices]
        
        print(f"  Seed {seed}: Evaluating on {len(test_texts_sampled)} texts")
        
        for k in k_values:
            print(f"    Running k={k}...")
            
            # 1. Activation baseline
            act_features = select_features_by_activation(
                model, sae, hook_point, target_prompts, k=k, seed=seed
            )
            act_results = compute_perplexity_with_intervention(
                model, sae, hook_point,
                test_texts_sampled, act_features,
                coefficient=coefficient
            )
            results.append({
                "method": "activation",
                "k": k,
                "seed": seed,
                **act_results
            })
            
            # 2. Output score baseline
            out_features = select_features_by_output_score(
                model, sae, hook_point, target_prompts, target_token_ids, k=k, seed=seed
            )
            out_results = compute_perplexity_with_intervention(
                model, sae, hook_point,
                test_texts_sampled, out_features,
                coefficient=coefficient
            )
            results.append({
                "method": "output_score",
                "k": k,
                "seed": seed,
                **out_results
            })
            
            # 3. Fidelity-Weighted Steering
            ifs_features = select_features_by_ifs(ifs_scores, k=k)
            ifs_results_eval = compute_perplexity_with_intervention(
                model, sae, hook_point,
                test_texts_sampled, ifs_features,
                coefficient=coefficient,
                ifs_scores=ifs_scores,
                beta=0.5
            )
            results.append({
                "method": "fidelity_weighted",
                "k": k,
                "seed": seed,
                **ifs_results_eval
            })
            
            print(f"      Activation: ppl_change={act_results['ppl_change']:+.2f}")
            print(f"      Output Score: ppl_change={out_results['ppl_change']:+.2f}")
            print(f"      Fidelity-Weighted: ppl_change={ifs_results_eval['ppl_change']:+.2f}")
    
    return results


def main():
    print("=" * 60)
    print("Side Effects Measurement (Proper Perplexity)")
    print("=" * 60)
    
    # Hyperparameters
    seeds = [42, 43, 44]
    k_values = [20, 50, 100]
    coefficient = 20.0
    device = "cuda"
    
    # Load model and SAE
    print("\nLoading model and SAE...")
    model, sae, cfg_dict, hook_point = get_model_and_sae(device=device)
    
    # Load datasets
    print("Loading datasets...")
    datasets = load_prepared_datasets()
    
    all_pairs = datasets["truthfulqa_pairs"]
    target_prompts = [p["positive"] for p in all_pairs[:200]]
    
    print(f"  Target prompts: {len(target_prompts)}")
    
    # Load HellaSwag texts
    print("Loading HellaSwag texts...")
    test_texts = load_hellaswag_texts(n_samples=200)
    print(f"  Test texts: {len(test_texts)}")
    
    # Define target tokens
    true_tokens = [model.to_single_token("True"), model.to_single_token("Yes")]
    target_token_ids = true_tokens
    
    print(f"\nRunning side effects experiment...")
    print(f"  k values: {k_values}")
    print(f"  seeds: {seeds}")
    
    start_time = time.time()
    
    results = run_side_effects_experiment(
        model, sae, hook_point,
        target_prompts, test_texts,
        target_token_ids, k_values, coefficient, seeds
    )
    
    runtime = time.time() - start_time
    
    # Aggregate results
    aggregated = {}
    methods = ["activation", "output_score", "fidelity_weighted"]
    
    for method in methods:
        for k in k_values:
            key = f"{method}_k={k}"
            method_k_results = [r for r in results if r["method"] == method and r["k"] == k]
            if method_k_results:
                ppl_changes = [r["ppl_change"] for r in method_k_results]
                aggregated[key] = {
                    "baseline_ppl_mean": float(np.mean([r["baseline_ppl"] for r in method_k_results])),
                    "baseline_ppl_std": float(np.std([r["baseline_ppl"] for r in method_k_results])),
                    "steered_ppl_mean": float(np.mean([r["steered_ppl"] for r in method_k_results])),
                    "steered_ppl_std": float(np.std([r["steered_ppl"] for r in method_k_results])),
                    "ppl_change_mean": float(np.mean(ppl_changes)),
                    "ppl_change_std": float(np.std(ppl_changes)),
                    "relative_change_mean": float(np.mean([r["relative_change"] for r in method_k_results])),
                    "relative_change_std": float(np.std([r["relative_change"] for r in method_k_results])),
                    "n_seeds": len(method_k_results)
                }
    
    # Compute side effect scores (absolute perplexity change)
    side_effects = {}
    for method in methods:
        for k in k_values:
            key = f"{method}_k={k}"
            if key in aggregated:
                side_effects[key] = abs(aggregated[key]["ppl_change_mean"])
    
    # Compare methods
    comparison = {}
    for k in k_values:
        act_key = f"activation_k={k}"
        out_key = f"output_score_k={k}"
        ifs_key = f"fidelity_weighted_k={k}"
        
        if act_key in side_effects and ifs_key in side_effects:
            comparison[f"k={k}_fws_vs_activation"] = side_effects[ifs_key] / (side_effects[act_key] + 1e-8)
        if out_key in side_effects and ifs_key in side_effects:
            comparison[f"k={k}_fws_vs_output"] = side_effects[ifs_key] / (side_effects[out_key] + 1e-8)
    
    # Save results
    output = {
        "experiment": "side_effects",
        "config": {
            "seeds": seeds,
            "k_values": k_values,
            "coefficient": coefficient,
            "n_test_texts": len(test_texts)
        },
        "raw_results": results,
        "aggregated": aggregated,
        "side_effects": side_effects,
        "comparison": comparison,
        "runtime_minutes": runtime / 60
    }
    
    save_results("exp/side_effects/results.json", output)
    print(f"\n✓ Results saved to exp/side_effects/results.json")
    print(f"  Runtime: {runtime/60:.1f} minutes")
    
    # Print summary
    print("\nSide Effects Summary (perplexity change, mean ± std):")
    for method in methods:
        for k in k_values:
            key = f"{method}_k={k}"
            if key in aggregated:
                agg = aggregated[key]
                print(f"  {method:20s} k={k:3d}: ppl_change={agg['ppl_change_mean']:+.2f} ± {agg['ppl_change_std']:.2f}")
    
    print(f"\nSide Effect Ratios (FWS / Baseline, lower is better):")
    for k in k_values:
        key_act = f"k={k}_fws_vs_activation"
        key_out = f"k={k}_fws_vs_output"
        if key_act in comparison:
            print(f"  k={k}: FWS vs Activation = {comparison[key_act]:.3f}")
        if key_out in comparison:
            print(f"  k={k}: FWS vs Output Score = {comparison[key_out]:.3f}")


if __name__ == "__main__":
    main()
