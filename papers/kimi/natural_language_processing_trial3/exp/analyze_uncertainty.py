"""Analyze entropy and varentropy distributions to set proper thresholds."""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.data_loader import load_json, create_cot_prompt
from shared.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum().item()


def compute_varentropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    return (probs * (log_probs + entropy) ** 2).sum().item()


def analyze_problem(model, tokenizer, question):
    """Analyze uncertainty patterns for a single problem."""
    prompt = create_cot_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    generated_ids = inputs["input_ids"]
    
    uncertainties = []
    
    with torch.no_grad():
        for _ in range(256):
            outputs = model(input_ids=generated_ids, return_dict=True)
            next_logits = outputs.logits[:, -1, :]
            
            h = compute_entropy(next_logits[0])
            v = compute_varentropy(next_logits[0])
            uncertainties.append((h, v))
            
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return uncertainties


def main():
    set_seed(42)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    dataset = load_json("exp/data/gsm8k_test.json")[:30]  # Analyze 30 problems
    
    all_entropies = []
    all_varentropies = []
    problem_stats = []
    
    print("Analyzing uncertainty patterns...")
    for i, item in enumerate(dataset):
        print(f"  Problem {i+1}/{len(dataset)}")
        uncertainties = analyze_problem(model, tokenizer, item["question"])
        
        entropies = [h for h, v in uncertainties]
        varentropies = [v for h, v in uncertainties]
        
        all_entropies.extend(entropies)
        all_varentropies.extend(varentropies)
        
        problem_stats.append({
            "mean_h": sum(entropies) / len(entropies),
            "max_h": max(entropies),
            "min_h": min(entropies),
            "mean_v": sum(varentropies) / len(varentropies),
            "max_v": max(varentropies),
            "min_v": min(varentropies),
        })
    
    # Compute overall statistics
    import numpy as np
    
    print("\n=== Entropy Statistics ===")
    print(f"  Mean: {np.mean(all_entropies):.4f}")
    print(f"  Median: {np.median(all_entropies):.4f}")
    print(f"  Std: {np.std(all_entropies):.4f}")
    print(f"  Min: {np.min(all_entropies):.4f}")
    print(f"  Max: {np.max(all_entropies):.4f}")
    print(f"  75th percentile: {np.percentile(all_entropies, 75):.4f}")
    print(f"  90th percentile: {np.percentile(all_entropies, 90):.4f}")
    print(f"  95th percentile: {np.percentile(all_entropies, 95):.4f}")
    
    print("\n=== Varentropy Statistics ===")
    print(f"  Mean: {np.mean(all_varentropies):.4f}")
    print(f"  Median: {np.median(all_varentropies):.4f}")
    print(f"  Std: {np.std(all_varentropies):.4f}")
    print(f"  Min: {np.min(all_varentropies):.4f}")
    print(f"  Max: {np.max(all_varentropies):.4f}")
    print(f"  25th percentile: {np.percentile(all_varentropies, 25):.4f}")
    print(f"  10th percentile: {np.percentile(all_varentropies, 10):.4f}")
    
    # Recommend thresholds
    print("\n=== Recommended Thresholds ===")
    print(f"  tau_h (entropy): {np.percentile(all_entropies, 75):.2f} - {np.percentile(all_entropies, 85):.2f}")
    print(f"  tau_v (varentropy): {np.percentile(all_varentropies, 25):.2f} - {np.percentile(all_varentropies, 40):.2f}")
    
    # Save results
    results = {
        "entropy": {
            "mean": float(np.mean(all_entropies)),
            "median": float(np.median(all_entropies)),
            "std": float(np.std(all_entropies)),
            "min": float(np.min(all_entropies)),
            "max": float(np.max(all_entropies)),
            "p75": float(np.percentile(all_entropies, 75)),
            "p85": float(np.percentile(all_entropies, 85)),
            "p90": float(np.percentile(all_entropies, 90)),
        },
        "varentropy": {
            "mean": float(np.mean(all_varentropies)),
            "median": float(np.median(all_varentropies)),
            "std": float(np.std(all_varentropies)),
            "min": float(np.min(all_varentropies)),
            "max": float(np.max(all_varentropies)),
            "p25": float(np.percentile(all_varentropies, 25)),
            "p40": float(np.percentile(all_varentropies, 40)),
            "p10": float(np.percentile(all_varentropies, 10)),
        },
        "problem_stats": problem_stats
    }
    
    Path("exp/results").mkdir(parents=True, exist_ok=True)
    with open("exp/results/uncertainty_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to exp/results/uncertainty_analysis.json")


if __name__ == "__main__":
    main()
