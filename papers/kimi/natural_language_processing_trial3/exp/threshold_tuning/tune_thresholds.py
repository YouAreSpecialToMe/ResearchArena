"""Threshold tuning with cross-validation for ESR."""

import sys
import json
import numpy as np
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.data_loader import load_gsm8k, create_cot_prompt
from shared.metrics import compare_answers
from shared.models import load_model
from shared.utils import set_seed
from esr.esr_algorithm import ESRSimpleGenerator


def extract_answer_from_text(text: str):
    """Extract numeric answer from text."""
    import re
    if not text:
        return None
    
    if "####" in text:
        match = re.search(r"####\s*([-\d.,]+)", text)
        if match:
            return match.group(1)
    
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1)
    
    answer_match = re.search(r"(?:final answer|answer is|answer:)\s*([-\d.,]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    
    return text.strip()


def evaluate_threshold_pair(model, tokenizer, dataset, tau_h, tau_v, r_max=3):
    """Evaluate a single threshold pair."""
    generator = ESRSimpleGenerator(
        model=model,
        tokenizer=tokenizer,
        tau_h=tau_h,
        tau_v=tau_v,
        r_max=r_max,
        max_new_tokens=1024
    )
    
    correct = 0
    total_tokens = 0
    revision_count = 0
    
    for item in dataset:
        prompt = create_cot_prompt(item["question"])
        output = generator.generate(prompt)
        
        pred = extract_answer_from_text(output["output"])
        if compare_answers(pred, item["answer"]):
            correct += 1
        
        total_tokens += output["total_tokens"]
        revision_count += output["revision_count"]
    
    accuracy = correct / len(dataset)
    avg_tokens = total_tokens / len(dataset)
    avg_revisions = revision_count / len(dataset)
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_revisions": avg_revisions
    }


def cross_validate(model, tokenizer, dataset, tau_h_values, tau_v_values, n_folds=5):
    """Perform k-fold cross-validation."""
    # Split dataset into folds
    fold_size = len(dataset) // n_folds
    folds = [dataset[i*fold_size:(i+1)*fold_size] for i in range(n_folds-1)]
    folds.append(dataset[(n_folds-1)*fold_size:])  # Last fold gets remaining
    
    results = {}
    
    for tau_h in tau_h_values:
        for tau_v in tau_v_values:
            print(f"  Testing tau_h={tau_h}, tau_v={tau_v}")
            
            fold_accuracies = []
            
            for fold_idx in range(n_folds):
                # Use all folds except fold_idx for training (threshold selection)
                # Actually, for threshold selection, we just evaluate on the held-out fold
                val_data = folds[fold_idx]
                
                metrics = evaluate_threshold_pair(model, tokenizer, val_data, tau_h, tau_v)
                fold_accuracies.append(metrics["accuracy"])
            
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            results[(tau_h, tau_v)] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "fold_accuracies": fold_accuracies
            }
            
            print(f"    Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output", type=str, default="exp/results/threshold_tuning.json")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100, help="Limit tuning to N samples")
    args = parser.parse_args()
    
    set_seed(42)
    
    print("Loading model...")
    model, tokenizer = load_model(args.model)
    
    print("Loading GSM8K train set...")
    train_data = load_gsm8k("train")
    
    if args.limit:
        # Stratified sampling by answer length
        sorted_data = sorted(train_data, key=lambda x: len(str(x.get("answer_text", ""))))
        easy = sorted_data[:len(sorted_data)//3]
        medium = sorted_data[len(sorted_data)//3:2*len(sorted_data)//3]
        hard = sorted_data[2*len(sorted_data)//3:]
        
        sample_size = args.limit // 3
        dataset = (
            easy[:sample_size] + 
            medium[:sample_size] + 
            hard[:sample_size]
        )
    else:
        dataset = train_data
    
    print(f"Tuning dataset size: {len(dataset)}")
    
    # Define search space
    tau_h_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    tau_v_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    print(f"\nRunning {args.n_folds}-fold cross-validation...")
    print(f"Threshold search space: tau_H ∈ {tau_h_values}, tau_V ∈ {tau_v_values}")
    
    results = cross_validate(model, tokenizer, dataset, tau_h_values, tau_v_values, args.n_folds)
    
    # Find best thresholds
    best_pair = max(results.keys(), key=lambda k: results[k]["mean_accuracy"])
    best_tau_h, best_tau_v = best_pair
    
    print(f"\nBest thresholds: tau_H={best_tau_h}, tau_V={best_tau_v}")
    print(f"Best accuracy: {results[best_pair]['mean_accuracy']:.4f}")
    
    # Sensitivity analysis
    print("\nSensitivity analysis (±10%):")
    sensitivity_results = {}
    for scale in [0.9, 1.0, 1.1]:
        test_tau_h = best_tau_h * scale
        test_tau_v = best_tau_v * scale
        metrics = evaluate_threshold_pair(model, tokenizer, dataset[:20], test_tau_h, test_tau_v)
        sensitivity_results[f"scale_{scale:.1f}"] = {
            "tau_h": test_tau_h,
            "tau_v": test_tau_v,
            "metrics": metrics
        }
        print(f"  Scale {scale:.1f}: acc={metrics['accuracy']:.4f}")
    
    # Save results
    output_data = {
        "best_tau_h": best_tau_h,
        "best_tau_v": best_tau_v,
        "best_accuracy": results[best_pair]["mean_accuracy"],
        "cv_results": {f"h{k[0]}_v{k[1]}": v for k, v in results.items()},
        "sensitivity_analysis": sensitivity_results,
        "search_space": {
            "tau_h_values": tau_h_values,
            "tau_v_values": tau_v_values
        }
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
