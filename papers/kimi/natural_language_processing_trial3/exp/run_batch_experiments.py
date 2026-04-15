"""
Batch experiment runner that processes problems in chunks and saves progress.
Optimized for running larger-scale experiments efficiently.
"""

import torch
import json
import time
import random
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from shared.models import load_model, generate_vanilla_cot
from shared.data_loader import load_gsm8k, load_math500, create_cot_prompt, extract_numeric_answer

# Import the generator classes from run_complete_experiments
from run_complete_experiments import ESRGenerator, EntropyOnlyGenerator, EGLGenerator, BestOfNGenerator, set_seed


def run_method_batch(method: str, model, tokenizer, data: List[Dict], seed: int,
                     tau_h: float = 2.5, tau_v: float = 1.5, 
                     batch_size: int = 50, resume: bool = True) -> Dict[str, Any]:
    """Run a method with batch processing and progress saving."""
    
    set_seed(seed)
    
    # Initialize generator based on method
    if method == "vanilla":
        generator = None
    elif method == "esr":
        generator = ESRGenerator(model, tokenizer, tau_h=tau_h, tau_v=tau_v, r_max=3, max_new_tokens=512)
    elif method == "entropy_only":
        generator = EntropyOnlyGenerator(model, tokenizer, tau_h=tau_h, r_max=3, max_new_tokens=512)
    elif method == "egl":
        generator = EGLGenerator(model, tokenizer, tau_h=tau_h, max_new_tokens=512)
    elif method == "bestofn":
        generator = BestOfNGenerator(model, tokenizer, n=4, temperature=0.7, max_new_tokens=512)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Check for existing progress
    progress_file = Path(f"exp/results/progress_{method}_seed{seed}.json")
    results = []
    start_idx = 0
    
    if resume and progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                results = progress.get("results", [])
                start_idx = len(results)
                print(f"Resuming from problem {start_idx}")
        except Exception as e:
            print(f"Could not load progress: {e}")
    
    correct = sum(1 for r in results if r.get("correct", False))
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_revisions = sum(r.get("details", {}).get("revision_count", 0) for r in results if "details" in r)
    
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(start_idx, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch = data[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start}-{batch_end-1} ({len(batch)} problems)...")
        
        for i, item in enumerate(batch):
            global_idx = batch_start + i
            
            try:
                prompt = create_cot_prompt(item["question"])
                
                # Generate based on method
                if method == "vanilla":
                    output_text, num_tokens = generate_vanilla_cot(
                        model, tokenizer, prompt, max_new_tokens=512, temperature=0.0
                    )
                    result_info = {"tokens": num_tokens}
                else:
                    gen_result = generator.generate(prompt)
                    output_text = gen_result["output"]
                    num_tokens = gen_result["total_tokens"]
                    result_info = gen_result
                
                predicted = extract_numeric_answer(output_text)
                actual = item["answer"]
                
                is_correct = False
                if predicted is not None and actual is not None:
                    is_correct = abs(predicted - actual) < 1e-3
                
                if is_correct:
                    correct += 1
                total_tokens += num_tokens
                
                if method in ["esr", "entropy_only"]:
                    total_revisions += result_info.get("revision_count", 0)
                
                results.append({
                    "question_idx": global_idx,
                    "predicted": predicted,
                    "actual": actual,
                    "correct": is_correct,
                    "tokens": num_tokens,
                    "details": result_info
                })
                
            except Exception as e:
                print(f"Error on problem {global_idx}: {e}")
                results.append({
                    "question_idx": global_idx,
                    "error": str(e)
                })
        
        # Save progress after each batch
        progress = {
            "method": method,
            "seed": seed,
            "completed": len(results),
            "total": len(data),
            "results": results
        }
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Print progress
        elapsed = time.time() - start_time
        current_acc = correct / len(results) if results else 0
        avg_tokens = total_tokens / len(results) if results else 0
        print(f"  Progress: {len(results)}/{len(data)}, Acc: {current_acc:.3f}, AvgTokens: {avg_tokens:.1f}, Time: {elapsed/60:.1f}min")
    
    elapsed = time.time() - start_time
    accuracy = correct / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    
    output = {
        "method": method,
        "seed": seed,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "correct_count": correct,
        "total_problems": len(results),
        "runtime_seconds": elapsed,
        "results": results
    }
    
    if method in ["esr", "entropy_only"]:
        output["avg_revisions"] = total_revisions / len(results) if results else 0
        output["revision_rate"] = sum(1 for r in results if r.get("details", {}).get("revision_count", 0) > 0) / len(results) if results else 0
    
    # Clean up progress file
    if progress_file.exists():
        progress_file.unlink()
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run batch experiments")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--tau_h", type=float, default=2.5)
    parser.add_argument("--tau_v", type=float, default=1.5)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    
    print("="*70)
    print(f"Batch Experiment Runner - {args.method}")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    if args.dataset == "gsm8k":
        data = load_gsm8k("test")
    else:
        data = load_math500()
    
    if args.max_problems:
        data = data[:args.max_problems]  # Take first N instead of random for reproducibility
    
    print(f"Loaded {len(data)} problems")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model)
    
    # Run experiment
    result = run_method_batch(
        args.method, model, tokenizer, data, args.seed,
        tau_h=args.tau_h, tau_v=args.tau_v,
        batch_size=args.batch_size, resume=not args.no_resume
    )
    
    # Save results
    output_path = Path(f"exp/results/{args.method}_{args.dataset}_seed{args.seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Avg tokens: {result['avg_tokens']:.1f}")
    print(f"Runtime: {result['runtime_seconds']/60:.1f} minutes")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
