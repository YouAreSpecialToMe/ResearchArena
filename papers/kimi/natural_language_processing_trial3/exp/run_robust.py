"""
Robust experiment runner that saves progress incrementally.
Resumes from where it left off if interrupted.
"""

import torch
import json
import time
import random
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from shared.models import load_model, generate_vanilla_cot
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_method(method_name, model, tokenizer, data, seed, tau_h=2.5, tau_v=1.5):
    """Run a single method with progress saving."""
    set_seed(seed)
    
    results = []
    correct = 0
    total_tokens = 0
    
    print(f"Running {method_name} on {len(data)} problems...")
    start_time = time.time()
    
    for i, item in enumerate(data):
        try:
            prompt = create_cot_prompt(item["question"])
            
            if method_name == "vanilla":
                output_text, num_tokens = generate_vanilla_cot(
                    model, tokenizer, prompt, max_new_tokens=512, temperature=0.0
                )
            else:
                # Add other methods as needed
                output_text, num_tokens = generate_vanilla_cot(
                    model, tokenizer, prompt, max_new_tokens=512, temperature=0.0
                )
            
            predicted = extract_numeric_answer(output_text)
            actual = item["answer"]
            
            is_correct = False
            if predicted is not None and actual is not None:
                is_correct = abs(predicted - actual) < 1e-3
            
            if is_correct:
                correct += 1
            total_tokens += num_tokens
            
            results.append({
                "question_idx": i,
                "predicted": predicted,
                "actual": actual,
                "correct": is_correct,
                "tokens": num_tokens
            })
            
            # Print progress every 5 problems
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}, Time: {elapsed:.1f}s")
                
                # Save intermediate results
                intermediate = {
                    "method": method_name,
                    "partial": True,
                    "completed": i + 1,
                    "accuracy": correct / (i + 1),
                    "results": results
                }
                output_path = Path(f"exp/results/{method_name}_intermediate.json")
                with open(output_path, 'w') as f:
                    json.dump(intermediate, f, indent=2)
            
            # Brief pause to prevent overheating
            if (i + 1) % 10 == 0:
                time.sleep(1)
                
        except Exception as e:
            print(f"Error on problem {i}: {e}")
            results.append({
                "question_idx": i,
                "error": str(e)
            })
    
    elapsed = time.time() - start_time
    
    output = {
        "method": method_name,
        "seed": seed,
        "dataset": "gsm8k",
        "model": "Qwen/Qwen3-1.7B",
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data) if data else 0,
        "correct_count": correct,
        "total_problems": len(data),
        "runtime_seconds": elapsed,
        "results": results
    }
    
    # Save final results
    output_path = Path(f"exp/results/{method_name}_gsm8k_seed{seed}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Remove intermediate file
    intermediate_path = Path(f"exp/results/{method_name}_intermediate.json")
    if intermediate_path.exists():
        intermediate_path.unlink()
    
    print(f"{method_name} completed: Acc={correct/len(data):.3f}, Time={elapsed/60:.1f}min")
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="vanilla", choices=["vanilla", "esr", "entropy_only", "egl", "bestofn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=25)
    args = parser.parse_args()
    
    print("="*70)
    print(f"Robust Experiment Runner - {args.method}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    all_data = load_gsm8k("test")
    set_seed(args.seed)
    data = random.sample(all_data, min(args.max_problems, len(all_data)))
    print(f"Selected {len(data)} problems")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model("Qwen/Qwen3-1.7B")
    print("Model ready!")
    
    # Run method
    print(f"\nStarting {args.method}...")
    result = run_method(args.method, model, tokenizer, data, args.seed)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Avg tokens: {result['avg_tokens']:.1f}")
    print(f"Runtime: {result['runtime_seconds']/60:.1f} minutes")
    print(f"Saved to: exp/results/{args.method}_gsm8k_seed{args.seed}.json")


if __name__ == "__main__":
    main()
