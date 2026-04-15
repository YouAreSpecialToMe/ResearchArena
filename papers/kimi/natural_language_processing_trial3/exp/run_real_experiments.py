"""
Run REAL experiments with actual model inference.
This script runs all methods with proper inference and saves verifiable results.
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
from shared.data_loader import load_gsm8k, load_math500, create_cot_prompt, extract_numeric_answer
from esr.esr_algorithm import ESRGenerator


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_vanilla(model, tokenizer, data, seed):
    """Run Vanilla CoT baseline."""
    set_seed(seed)
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        output_text, num_tokens = generate_vanilla_cot(model, tokenizer, prompt, 
                                                        max_new_tokens=1024, temperature=0.0)
        
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
        
        if (i + 1) % 10 == 0:
            print(f"  Vanilla: {i+1}/{len(data)}, Acc: {correct/(i+1):.3f}")
    
    return {
        "method": "vanilla",
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_esr(model, tokenizer, data, seed, tau_h=2.5, tau_v=1.5, r_max=3):
    """Run ESR full method."""
    set_seed(seed)
    generator = ESRGenerator(model, tokenizer, tau_h=tau_h, tau_v=tau_v, r_max=r_max)
    
    results = []
    correct = 0
    total_tokens = 0
    total_revisions = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
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
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "revisions": result["revision_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  ESR: {i+1}/{len(data)}, Acc: {correct/(i+1):.3f}, AvgRev: {total_revisions/(i+1):.2f}")
    
    return {
        "method": "esr",
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "avg_revisions": total_revisions / len(data),
        "revision_rate": sum(1 for r in results if r["revisions"] > 0) / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_entropy_only(model, tokenizer, data, seed, tau_h=2.5):
    """Run Entropy-Only baseline (no varentropy)."""
    set_seed(seed)
    from entropy_only.run import EntropyOnlyGenerator
    
    generator = EntropyOnlyGenerator(model, tokenizer, tau_h=tau_h, r_max=3)
    
    results = []
    correct = 0
    total_tokens = 0
    total_revisions = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = generator.generate(prompt)
        
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        total_revisions += result["revision_count"]
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "revisions": result["revision_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Entropy-Only: {i+1}/{len(data)}, Acc: {correct/(i+1):.3f}")
    
    return {
        "method": "entropy_only",
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "avg_revisions": total_revisions / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_egl(model, tokenizer, data, seed, tau_h=2.5):
    """Run EGL post-hoc baseline."""
    set_seed(seed)
    from egl_posthoc.run import EGLPostHoc
    
    egl = EGLPostHoc(model, tokenizer, tau_h=tau_h)
    
    results = []
    correct = 0
    total_tokens = 0
    refined_count = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = egl.generate(prompt)
        
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        if result["refined"]:
            refined_count += 1
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "refined": result["refined"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  EGL: {i+1}/{len(data)}, Acc: {correct/(i+1):.3f}")
    
    return {
        "method": "egl",
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "refinement_rate": refined_count / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_bestofn(model, tokenizer, data, seed, n=4):
    """Run Best-of-N baseline."""
    set_seed(seed)
    from bestofn.run import BestOfN
    
    bon = BestOfN(model, tokenizer, n_samples=n, temperature=0.7)
    
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = bon.generate(prompt)
        
        predicted = result["best_answer"]
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Best-of-{n}: {i+1}/{len(data)}, Acc: {correct/(i+1):.3f}")
    
    return {
        "method": f"bestofn_{n}",
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=150)
    parser.add_argument("--methods", nargs="+", default=["vanilla", "esr", "entropy_only", "egl", "bestofn"])
    args = parser.parse_args()
    
    print("="*70)
    print(f"ESR Real Inference Experiments")
    print(f"Dataset: {args.dataset}, Model: {args.model}, Seed: {args.seed}")
    print(f"Max problems: {args.max_problems}, Methods: {args.methods}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    if args.dataset == "gsm8k":
        all_data = load_gsm8k("test")
    else:
        all_data = load_math500()
    
    # Sample subset
    set_seed(args.seed)
    data = random.sample(all_data, min(args.max_problems, len(all_data)))
    print(f"Loaded {len(data)} problems")
    
    # Load model once
    print("\nLoading model (this may take a minute)...")
    model, tokenizer = load_model(args.model)
    print("Model loaded successfully!")
    
    # Run each method
    all_results = {}
    
    for method in args.methods:
        print(f"\n{'='*70}")
        print(f"Running: {method}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        try:
            if method == "vanilla":
                result = run_vanilla(model, tokenizer, data, args.seed)
            elif method == "esr":
                result = run_esr(model, tokenizer, data, args.seed)
            elif method == "entropy_only":
                result = run_entropy_only(model, tokenizer, data, args.seed)
            elif method == "egl":
                result = run_egl(model, tokenizer, data, args.seed)
            elif method == "bestofn":
                result = run_bestofn(model, tokenizer, data, args.seed)
            else:
                print(f"Unknown method: {method}")
                continue
            
            elapsed = time.time() - start_time
            result["runtime_seconds"] = elapsed
            result["seed"] = args.seed
            result["dataset"] = args.dataset
            result["model"] = args.model
            
            all_results[method] = result
            
            # Save individual result
            output_path = Path(f"exp/results/{method}_{args.dataset}_seed{args.seed}.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n{method} completed in {elapsed/60:.1f} minutes")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  Avg tokens: {result['avg_tokens']:.1f}")
            print(f"  Saved to: {output_path}")
            
        except Exception as e:
            print(f"ERROR running {method}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save aggregate results
    aggregate_path = Path(f"exp/results/aggregate_{args.dataset}_seed{args.seed}.json")
    with open(aggregate_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for method, result in all_results.items():
        print(f"{method:20s}: Acc={result['accuracy']:.3f}, Tokens={result['avg_tokens']:.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
