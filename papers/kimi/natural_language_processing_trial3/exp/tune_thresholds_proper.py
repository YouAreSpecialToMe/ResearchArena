"""
Proper threshold tuning for ESR using validation set.
"""

import torch
import json
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from shared.models import load_model, compute_entropy, compute_varentropy
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def analyze_uncertainty_distribution(model, tokenizer, data, device="cuda", max_tokens=512):
    """
    Analyze the distribution of entropy and varentropy on the validation set.
    This helps determine appropriate thresholds.
    """
    print("Analyzing uncertainty distribution on validation set...")
    
    all_entropy = []
    all_varentropy = []
    all_max_prob = []
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        
        with torch.no_grad():
            for step in range(max_tokens):
                outputs = model(input_ids=generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :]
                
                # Compute metrics
                probs = torch.softmax(next_logits[0], dim=-1)
                log_probs = torch.log_softmax(next_logits[0], dim=-1)
                entropy = -(probs * log_probs).sum().item()
                varentropy = (probs * (log_probs + entropy) ** 2).sum().item()
                max_prob = probs.max().item()
                
                all_entropy.append(entropy)
                all_varentropy.append(varentropy)
                all_max_prob.append(max_prob)
                
                # Greedy decode
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}")
    
    # Compute statistics
    stats = {
        "entropy": {
            "mean": np.mean(all_entropy),
            "std": np.std(all_entropy),
            "min": np.min(all_entropy),
            "max": np.max(all_entropy),
            "percentile_25": np.percentile(all_entropy, 25),
            "percentile_50": np.percentile(all_entropy, 50),
            "percentile_75": np.percentile(all_entropy, 75),
            "percentile_90": np.percentile(all_entropy, 90),
            "percentile_95": np.percentile(all_entropy, 95),
        },
        "varentropy": {
            "mean": np.mean(all_varentropy),
            "std": np.std(all_varentropy),
            "min": np.min(all_varentropy),
            "max": np.max(all_varentropy),
            "percentile_25": np.percentile(all_varentropy, 25),
            "percentile_50": np.percentile(all_varentropy, 50),
            "percentile_75": np.percentile(all_varentropy, 75),
            "percentile_90": np.percentile(all_varentropy, 90),
            "percentile_95": np.percentile(all_varentropy, 95),
        },
        "max_prob": {
            "mean": np.mean(all_max_prob),
            "std": np.std(all_max_prob),
            "min": np.min(all_max_prob),
            "max": np.max(all_max_prob),
        }
    }
    
    return stats


def test_threshold_combinations(model, tokenizer, data, threshold_grid, device="cuda"):
    """
    Test different threshold combinations on validation set.
    """
    print("\nTesting threshold combinations...")
    
    results = []
    
    for tau_h, tau_v in threshold_grid:
        # Count how many tokens would trigger revision
        trigger_count = 0
        total_tokens = 0
        
        for i, item in enumerate(data[:20]):  # Use subset for speed
            prompt = create_cot_prompt(item["question"])
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generated_ids = inputs["input_ids"]
            
            with torch.no_grad():
                for step in range(512):
                    outputs = model(input_ids=generated_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    probs = torch.softmax(next_logits[0], dim=-1)
                    log_probs = torch.log_softmax(next_logits[0], dim=-1)
                    entropy = -(probs * log_probs).sum().item()
                    varentropy = (probs * (log_probs + entropy) ** 2).sum().item()
                    
                    total_tokens += 1
                    
                    # Check if this would trigger revision
                    if entropy > tau_h and varentropy < tau_v and step > 10:
                        trigger_count += 1
                    
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            if (i + 1) % 5 == 0:
                print(f"  tau_h={tau_h:.2f}, tau_v={tau_v:.2f}: {i+1}/20 done")
        
        trigger_rate = trigger_count / total_tokens if total_tokens > 0 else 0
        
        results.append({
            "tau_h": tau_h,
            "tau_v": tau_v,
            "trigger_count": trigger_count,
            "total_tokens": total_tokens,
            "trigger_rate": trigger_rate
        })
        
        print(f"  tau_h={tau_h:.2f}, tau_v={tau_v:.2f}: trigger_rate={trigger_rate:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--val_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("="*70)
    print("Threshold Tuning for ESR")
    print("="*70)
    
    set_seed(args.seed)
    
    # Load validation data
    print("\nLoading validation data...")
    val_data_path = Path("exp/data/gsm8k_val_50.json")
    if val_data_path.exists():
        with open(val_data_path) as f:
            val_data = json.load(f)
        print(f"Loaded {len(val_data)} validation problems")
    else:
        # Create validation set
        all_data = load_gsm8k("train")
        val_data = random.sample(all_data, min(args.val_size, len(all_data)))
        val_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_data_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Created validation set with {len(val_data)} problems")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model)
    
    # Step 1: Analyze uncertainty distribution
    print("\n" + "="*70)
    print("Step 1: Analyzing Uncertainty Distribution")
    print("="*70)
    stats = analyze_uncertainty_distribution(model, tokenizer, val_data[:20])
    
    print("\nEntropy Statistics:")
    for key, value in stats["entropy"].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nVarentropy Statistics:")
    for key, value in stats["varentropy"].items():
        print(f"  {key}: {value:.4f}")
    
    # Step 2: Test threshold combinations
    print("\n" + "="*70)
    print("Step 2: Testing Threshold Combinations")
    print("="*70)
    
    # Use percentiles from distribution analysis to define grid
    entropy_p75 = stats["entropy"]["percentile_75"]
    entropy_p90 = stats["entropy"]["percentile_90"]
    varentropy_p25 = stats["varentropy"]["percentile_25"]
    varentropy_p50 = stats["varentropy"]["percentile_50"]
    
    print(f"\nUsing entropy range [{entropy_p75:.2f}, {entropy_p90:.2f}]")
    print(f"Using varentropy range [{varentropy_p25:.2f}, {varentropy_p50:.2f}]")
    
    threshold_grid = []
    for tau_h in [entropy_p75, entropy_p75 + 0.5, entropy_p90]:
        for tau_v in [varentropy_p25, varentropy_p50, varentropy_p50 + 0.5]:
            threshold_grid.append((tau_h, tau_v))
    
    threshold_results = test_threshold_combinations(
        model, tokenizer, val_data[:20], threshold_grid
    )
    
    # Find best thresholds (targeting 15-40% revision rate)
    print("\n" + "="*70)
    print("Step 3: Recommended Thresholds")
    print("="*70)
    
    target_range = (0.15, 0.40)
    suitable = [r for r in threshold_results 
                if target_range[0] <= r["trigger_rate"] <= target_range[1]]
    
    if suitable:
        # Pick the one with trigger rate closest to 25%
        best = min(suitable, key=lambda x: abs(x["trigger_rate"] - 0.25))
        print(f"\nRecommended thresholds (closest to 25% trigger rate):")
        print(f"  tau_h = {best['tau_h']:.2f}")
        print(f"  tau_v = {best['tau_v']:.2f}")
        print(f"  Expected trigger rate: {best['trigger_rate']:.2%}")
    else:
        # Pick the one with highest trigger rate below 50%
        below_50 = [r for r in threshold_results if r["trigger_rate"] < 0.50]
        if below_50:
            best = max(below_50, key=lambda x: x["trigger_rate"])
            print(f"\nRecommended thresholds (highest trigger rate below 50%):")
            print(f"  tau_h = {best['tau_h']:.2f}")
            print(f"  tau_v = {best['tau_v']:.2f}")
            print(f"  Expected trigger rate: {best['trigger_rate']:.2%}")
        else:
            print("\nWarning: All thresholds result in >50% trigger rate!")
            print("Consider using lower values.")
            best = min(threshold_results, key=lambda x: abs(x["trigger_rate"] - 0.25))
            print(f"\nBest compromise:")
            print(f"  tau_h = {best['tau_h']:.2f}")
            print(f"  tau_v = {best['tau_v']:.2f}")
    
    # Save results
    output = {
        "uncertainty_stats": stats,
        "threshold_results": threshold_results,
        "recommended": {
            "tau_h": best['tau_h'],
            "tau_v": best['tau_v']
        }
    }
    
    output_path = Path("exp/results/threshold_tuning.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
