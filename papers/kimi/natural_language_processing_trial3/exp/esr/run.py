"""ESR (Entropy-guided Stepwise Revision) full method runner."""

import torch
from typing import List, Dict, Any
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer
from esr_algorithm import ESRGenerator


def run_esr_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                       seed: int = 42, max_problems: int = None,
                       tau_h: float = 2.5, tau_v: float = 1.5, r_max: int = 3):
    """Run ESR full method experiment."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Running ESR on {dataset_name} with seed {seed}, tau_h={tau_h}, tau_v={tau_v}")
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Load data
    if dataset_name == "gsm8k":
        data = load_gsm8k("test")
    else:
        from shared.data_loader import load_math500
        data = load_math500()
    
    if max_problems:
        data = random.sample(data, min(max_problems, len(data)))
    
    # Initialize ESR generator
    generator = ESRGenerator(
        model, tokenizer, 
        tau_h=tau_h, tau_v=tau_v, r_max=r_max,
        max_new_tokens=1024
    )
    
    results = []
    correct = 0
    total_tokens = 0
    total_revisions = 0
    
    # Track revision outcomes for analysis
    revision_outcomes = {
        "true_positive": 0,   # Wrong -> Correct
        "false_positive": 0,  # Correct -> Wrong (harm)
        "false_negative": 0,  # Wrong -> Wrong (failed revision)
        "true_negative": 0    # Correct -> Correct (no revision needed)
    }
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        
        # Run ESR
        result = generator.generate(prompt, track_uncertainty=True)
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        total_revisions += result["revision_count"]
        
        # Track revision outcomes if revision occurred
        if result["revision_count"] > 0:
            # This is a simplified categorization - actual analysis requires first-pass output
            if is_correct:
                revision_outcomes["true_positive"] += 1
            else:
                revision_outcomes["false_negative"] += 1
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "revisions": result["revision_count"],
            "revision_history": result["revision_history"],
            "uncertainty_triggers": result["uncertainty_triggers"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}, "
                  f"Avg Revisions: {total_revisions/(i+1):.2f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    avg_revisions = total_revisions / len(data)
    revision_rate = sum(1 for r in results if r["revisions"] > 0) / len(data)
    
    output = {
        "method": "esr",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "tau_h": tau_h,
        "tau_v": tau_v,
        "r_max": r_max,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_revisions": avg_revisions,
        "revision_rate": revision_rate,
        "total_problems": len(data),
        "correct_count": correct,
        "revision_outcomes": revision_outcomes,
        "results": results
    }
    
    # Save results
    output_path = Path(f"exp/results/esr_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"ESR Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}, "
          f"Revision Rate={revision_rate:.3f}, Avg Revisions={avg_revisions:.2f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--tau_h", type=float, default=2.5)
    parser.add_argument("--tau_v", type=float, default=1.5)
    parser.add_argument("--r_max", type=int, default=3)
    args = parser.parse_args()
    
    run_esr_experiment(args.dataset, args.model, args.seed, args.max_problems,
                       args.tau_h, args.tau_v, args.r_max)
