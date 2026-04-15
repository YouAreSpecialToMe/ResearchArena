"""Main experiment runner for ESR experiments."""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loader import load_gsm8k, load_math500, create_cot_prompt, extract_answer_from_text
from shared.metrics import compute_accuracy, compare_answers, categorize_revision_outcome, compute_revision_metrics
from shared.models import load_model, generate_vanilla_cot
from shared.utils import set_seed, save_results
from esr.esr_algorithm import ESRGenerator, ESRSimpleGenerator


def evaluate_esr(
    model,
    tokenizer,
    dataset: List[Dict],
    tau_h: float,
    tau_v: float,
    r_max: int,
    seed: int,
    method: str = "esr_full"
) -> Dict[str, Any]:
    """Evaluate ESR on a dataset."""
    set_seed(seed)
    
    results = []
    total_time = 0
    
    generator_class = ESRSimpleGenerator if method == "esr_simple" else ESRGenerator
    generator = generator_class(
        model=model,
        tokenizer=tokenizer,
        tau_h=tau_h,
        tau_v=tau_v,
        r_max=r_max,
        max_new_tokens=1024
    )
    
    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        question = item["question"]
        gold_answer = item["answer"]
        
        prompt = create_cot_prompt(question)
        
        start_time = time.time()
        output = generator.generate(prompt, track_uncertainty=True)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        predicted_answer = extract_answer_from_text(output["output"])
        correct = compare_answers(predicted_answer, gold_answer)
        
        result = {
            "question_id": i,
            "question": question,
            "reference_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "output": output["output"],
            "total_tokens": output["total_tokens"],
            "revision_count": output["revision_count"],
            "revision_triggered": output["revision_count"] > 0,
            "time": elapsed
        }
        
        if "initial_output" in output:
            initial_answer = extract_answer_from_text(output["initial_output"])
            initial_correct = compare_answers(initial_answer, gold_answer)
            result["initial_answer"] = initial_answer
            result["initial_correct"] = initial_correct
            result["final_correct"] = correct
        else:
            result["initial_correct"] = correct
            result["final_correct"] = correct
        
        results.append(result)
    
    # Compute metrics
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)
    avg_time = total_time / len(results)
    revision_rate = sum(1 for r in results if r["revision_triggered"]) / len(results)
    
    # Revision metrics
    rev_metrics = compute_revision_metrics(results)
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "revision_rate": revision_rate,
        "revision_metrics": rev_metrics,
        "results": results
    }


def evaluate_vanilla_cot(
    model,
    tokenizer,
    dataset: List[Dict],
    seed: int
) -> Dict[str, Any]:
    """Evaluate vanilla CoT baseline."""
    set_seed(seed)
    
    results = []
    total_time = 0
    
    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        question = item["question"]
        gold_answer = item["answer"]
        
        prompt = create_cot_prompt(question)
        
        start_time = time.time()
        output, num_tokens = generate_vanilla_cot(model, tokenizer, prompt, max_new_tokens=1024)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        predicted_answer = extract_answer_from_text(output)
        correct = compare_answers(predicted_answer, gold_answer)
        
        results.append({
            "question_id": i,
            "question": question,
            "reference_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "output": output,
            "total_tokens": num_tokens,
            "time": elapsed
        })
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "results": results
    }


def evaluate_entropy_only(
    model,
    tokenizer,
    dataset: List[Dict],
    tau_h: float,
    r_max: int,
    seed: int
) -> Dict[str, Any]:
    """Evaluate entropy-only baseline (revision triggered only by high entropy)."""
    # Use very high tau_v to effectively disable varentropy check
    # This makes should_revise() only depend on entropy > tau_h
    
    set_seed(seed)
    
    results = []
    total_time = 0
    
    generator = ESRGenerator(
        model=model,
        tokenizer=tokenizer,
        tau_h=tau_h,
        tau_v=0.0,  # Low threshold so varentropy check always passes
        r_max=r_max,
        max_new_tokens=1024
    )
    
    # Override should_revise to only check entropy
    original_should_revise = generator.should_revise
    generator.should_revise = lambda h, v: h > tau_h
    
    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        question = item["question"]
        gold_answer = item["answer"]
        
        prompt = create_cot_prompt(question)
        
        start_time = time.time()
        output = generator.generate(prompt, track_uncertainty=True)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        predicted_answer = extract_answer_from_text(output["output"])
        correct = compare_answers(predicted_answer, gold_answer)
        
        results.append({
            "question_id": i,
            "question": question,
            "reference_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "output": output["output"],
            "total_tokens": output["total_tokens"],
            "revision_count": output["revision_count"],
            "revision_triggered": output["revision_count"] > 0,
            "time": elapsed
        })
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)
    avg_time = total_time / len(results)
    revision_rate = sum(1 for r in results if r["revision_triggered"]) / len(results)
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "revision_rate": revision_rate,
        "results": results
    }


def evaluate_egl_posthoc(
    model,
    tokenizer,
    dataset: List[Dict],
    tau_h: float,
    seed: int
) -> Dict[str, Any]:
    """Evaluate EGL-style post-hoc refinement."""
    set_seed(seed)
    
    results = []
    total_time = 0
    
    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        question = item["question"]
        gold_answer = item["answer"]
        
        # First pass
        prompt = create_cot_prompt(question)
        start_time = time.time()
        output1, tokens1 = generate_vanilla_cot(model, tokenizer, prompt, max_new_tokens=1024)
        
        # Compute average entropy (simplified - use proxy based on output)
        # In full implementation, would compute actual entropy during generation
        # For now, use a heuristic: longer outputs with more uncertain words
        
        # Trigger refinement if output is long or contains uncertainty markers
        uncertainty_markers = ["maybe", "perhaps", "uncertain", "not sure", "think", "might"]
        has_uncertainty = any(m in output1.lower() for m in uncertainty_markers)
        
        if has_uncertainty or len(output1) > 400:
            # Post-hoc refinement
            refinement_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully. There may be an issue with my reasoning. Let me work through this again.\n"
            output2, tokens2 = generate_vanilla_cot(model, tokenizer, refinement_prompt, max_new_tokens=1024)
            
            predicted_answer = extract_answer_from_text(output2)
            total_tokens = tokens1 + tokens2
            refined = True
        else:
            predicted_answer = extract_answer_from_text(output1)
            total_tokens = tokens1
            refined = False
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        correct = compare_answers(predicted_answer, gold_answer)
        
        results.append({
            "question_id": i,
            "question": question,
            "reference_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "correct": correct,
            "output": output1 if not refined else output2,
            "total_tokens": total_tokens,
            "refined": refined,
            "time": elapsed
        })
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)
    avg_time = total_time / len(results)
    refinement_rate = sum(1 for r in results if r["refined"]) / len(results)
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "refinement_rate": refinement_rate,
        "results": results
    }


def evaluate_bestofn(
    model,
    tokenizer,
    dataset: List[Dict],
    n: int,
    seed: int
) -> Dict[str, Any]:
    """Evaluate Best-of-N baseline."""
    set_seed(seed)
    
    results = []
    total_time = 0
    
    for i, item in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        question = item["question"]
        gold_answer = item["answer"]
        
        prompt = create_cot_prompt(question)
        
        start_time = time.time()
        
        # Generate N samples
        outputs = []
        answers = []
        total_tokens = 0
        
        for _ in range(n):
            output, tokens = generate_vanilla_cot(
                model, tokenizer, prompt, 
                max_new_tokens=1024, 
                temperature=0.7
            )
            outputs.append(output)
            total_tokens += tokens
            
            pred_answer = extract_answer_from_text(output)
            answers.append(pred_answer)
        
        # Majority voting
        from collections import Counter
        answer_counts = Counter([str(a) for a in answers if a is not None])
        if answer_counts:
            majority_answer = answer_counts.most_common(1)[0][0]
            # Try to convert back to number
            try:
                majority_answer = float(majority_answer)
            except:
                pass
        else:
            majority_answer = answers[0] if answers else None
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        correct = compare_answers(majority_answer, gold_answer)
        
        results.append({
            "question_id": i,
            "question": question,
            "reference_answer": gold_answer,
            "predicted_answer": majority_answer,
            "all_answers": answers,
            "correct": correct,
            "total_tokens": total_tokens,
            "time": elapsed
        })
    
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    return {
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_time": avg_time,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                       choices=["esr", "vanilla", "entropy_only", "egl", "bestofn"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--tau_h", type=float, default=2.5)
    parser.add_argument("--tau_v", type=float, default=1.5)
    parser.add_argument("--r_max", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=4, help="N for Best-of-N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    
    print(f"Running {args.method} on {args.dataset} with model {args.model}")
    print(f"Seed: {args.seed}")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Load dataset
    if args.dataset == "gsm8k":
        dataset = load_gsm8k("test")
    else:
        dataset = load_math500()
    
    if args.limit:
        dataset = dataset[:args.limit]
    
    print(f"Dataset size: {len(dataset)}")
    
    # Run evaluation
    start_time = time.time()
    
    if args.method == "esr":
        results = evaluate_esr(model, tokenizer, dataset, args.tau_h, args.tau_v, args.r_max, args.seed)
    elif args.method == "vanilla":
        results = evaluate_vanilla_cot(model, tokenizer, dataset, args.seed)
    elif args.method == "entropy_only":
        results = evaluate_entropy_only(model, tokenizer, dataset, args.tau_h, args.r_max, args.seed)
    elif args.method == "egl":
        results = evaluate_egl_posthoc(model, tokenizer, dataset, args.tau_h, args.seed)
    elif args.method == "bestofn":
        results = evaluate_bestofn(model, tokenizer, dataset, args.n_samples, args.seed)
    
    runtime = time.time() - start_time
    results["runtime"] = runtime
    results["config"] = vars(args)
    
    # Save results
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Avg tokens: {results['avg_tokens']:.1f}")
    print(f"Runtime: {runtime:.1f}s")


if __name__ == "__main__":
    main()
