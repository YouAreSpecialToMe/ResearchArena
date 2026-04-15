"""
Fixed comprehensive experiment runner.
Properly dispatches to method-specific generation code.
"""

import torch
import json
import time
import random
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from shared.models import load_model, generate_vanilla_cot
from shared.data_loader import load_gsm8k, load_math500, create_cot_prompt, extract_numeric_answer

# Import method-specific generators
from esr.esr_algorithm import ESRGenerator
from entropy_only.run import EntropyOnlyGenerator
from egl_posthoc.run import EGLPostHoc
from egb_beam.run import EGBBeamSearch
from bestofn.run import BestOfN


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_vanilla(model, tokenizer, data, seed, max_tokens=1024):
    """Run vanilla CoT baseline."""
    set_seed(seed)
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        output_text, num_tokens = generate_vanilla_cot(
            model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=0.0
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
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_entropy_only(model, tokenizer, data, seed, tau_h=2.5, max_tokens=1024):
    """Run entropy-only baseline."""
    set_seed(seed)
    generator = EntropyOnlyGenerator(model, tokenizer, tau_h=tau_h, r_max=3, max_new_tokens=max_tokens)
    
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
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}, Revisions: {total_revisions}")
    
    return {
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "avg_revisions": total_revisions / len(data),
        "revision_rate": sum(1 for r in results if r["revisions"] > 0) / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_esr(model, tokenizer, data, seed, tau_h=2.5, tau_v=1.5, max_tokens=1024):
    """Run full ESR method."""
    set_seed(seed)
    generator = ESRGenerator(model, tokenizer, tau_h=tau_h, tau_v=tau_v, r_max=3, max_new_tokens=max_tokens)
    
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
            "revisions": result["revision_count"],
            "revision_history": result.get("revision_history", []),
            "uncertainty_triggers": result.get("uncertainty_triggers", [])
        })
        
        if (i + 1) % 10 == 0:
            revision_rate = sum(1 for r in results if r["revisions"] > 0) / len(results)
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}, RevRate: {revision_rate:.3f}")
    
    revision_rate = sum(1 for r in results if r["revisions"] > 0) / len(data)
    
    return {
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "avg_revisions": total_revisions / len(data),
        "revision_rate": revision_rate,
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_egl_posthoc(model, tokenizer, data, seed, tau_h=2.5, max_tokens=1024):
    """Run EGL post-hoc baseline."""
    set_seed(seed)
    egl = EGLPostHoc(model, tokenizer, tau_h=tau_h, max_new_tokens=max_tokens)
    
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
            "refined": result["refined"],
            "avg_entropy": result["avg_entropy"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "refinement_rate": refined_count / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_egb_beam(model, tokenizer, data, seed, tau_h=2.5, max_tokens=1024):
    """Run EGB beam search baseline."""
    set_seed(seed)
    egb = EGBBeamSearch(model, tokenizer, tau_h=tau_h, k_beams=3, max_new_tokens=max_tokens)
    
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = egb.generate(prompt)
        
        predicted = extract_numeric_answer(result["output"])
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
            "tokens": result["total_tokens"],
            "branch_count": result["branch_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


def run_bestofn(model, tokenizer, data, seed, max_tokens=1024):
    """Run Best-of-N baseline."""
    set_seed(seed)
    bon = BestOfN(model, tokenizer, n_samples=4, temperature=0.7, max_new_tokens=max_tokens)
    
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
            "tokens": result["total_tokens"],
            "agreement": result["agreement_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {
        "accuracy": correct / len(data),
        "avg_tokens": total_tokens / len(data),
        "correct_count": correct,
        "total_problems": len(data),
        "results": results
    }


METHOD_RUNNERS = {
    "vanilla": run_vanilla,
    "entropy_only": run_entropy_only,
    "esr": run_esr,
    "egl_posthoc": run_egl_posthoc,
    "egb_beam": run_egb_beam,
    "bestofn": run_bestofn,
}


def main():
    parser = argparse.ArgumentParser(description="Run fixed experiments")
    parser.add_argument("--method", required=True, 
                       choices=["vanilla", "entropy_only", "esr", "egl_posthoc", "egb_beam", "bestofn", "all"],
                       help="Method to run")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"],
                       help="Dataset to use")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                       help="Model to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Maximum number of problems to evaluate")
    parser.add_argument("--tau_h", type=float, default=2.5,
                       help="Entropy threshold")
    parser.add_argument("--tau_v", type=float, default=1.5,
                       help="Varentropy threshold")
    parser.add_argument("--output_dir", default="exp/results",
                       help="Output directory")
    args = parser.parse_args()
    
    print("="*70)
    print(f"Fixed Experiment Runner")
    print("="*70)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Tau_H: {args.tau_h}, Tau_V: {args.tau_v}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    if args.dataset == "gsm8k":
        all_data = load_gsm8k("test")
    else:
        all_data = load_math500()
    
    # Use same data sample for all methods with same seed
    set_seed(args.seed)
    if args.max_problems:
        data = random.sample(all_data, min(args.max_problems, len(all_data)))
    else:
        data = all_data
    print(f"Loaded {len(data)} problems")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model)
    
    # Determine which methods to run
    if args.method == "all":
        methods_to_run = ["vanilla", "entropy_only", "esr", "egl_posthoc", "egb_beam", "bestofn"]
    else:
        methods_to_run = [args.method]
    
    # Run each method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method in methods_to_run:
        print(f"\n{'='*70}")
        print(f"Running {method}...")
        print('='*70)
        
        start_time = time.time()
        
        # Get the runner function
        runner = METHOD_RUNNERS[method]
        
        # Run with appropriate parameters
        if method in ["entropy_only", "egl_posthoc", "egb_beam"]:
            result = runner(model, tokenizer, data, args.seed, tau_h=args.tau_h)
        elif method == "esr":
            result = runner(model, tokenizer, data, args.seed, tau_h=args.tau_h, tau_v=args.tau_v)
        else:
            result = runner(model, tokenizer, data, args.seed)
        
        elapsed = time.time() - start_time
        
        # Add metadata
        result["method"] = method
        result["dataset"] = args.dataset
        result["model"] = args.model
        result["seed"] = args.seed
        result["runtime_seconds"] = elapsed
        
        if method == "esr":
            result["tau_h"] = args.tau_h
            result["tau_v"] = args.tau_v
        elif method in ["entropy_only", "egl_posthoc", "egb_beam"]:
            result["tau_h"] = args.tau_h
        
        # Save results
        output_file = output_dir / f"{method}_{args.dataset}_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{method} completed in {elapsed/60:.1f} minutes")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Avg Tokens: {result['avg_tokens']:.1f}")
        if "revision_rate" in result:
            print(f"  Revision Rate: {result['revision_rate']:.3f}")
        if "refinement_rate" in result:
            print(f"  Refinement Rate: {result['refinement_rate']:.3f}")
        print(f"  Saved to: {output_file}")
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)


if __name__ == "__main__":
    main()
