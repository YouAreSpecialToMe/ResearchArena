#!/usr/bin/env python3
"""
Baseline 2: Self-Consistency with 16 samples
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
import numpy as np
from collections import Counter
from shared.model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text


COT_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""


def run_self_consistency(
    model_name: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    num_samples: int = 16,
    temperature: float = 0.7,
    limit: int = None,
):
    """Run Self-Consistency baseline with 16 samples."""
    print(f"=" * 60)
    print(f"Self-Consistency Baseline: {model_name}, Seed: {seed}")
    print(f"Samples: {num_samples}, Temperature: {temperature}")
    print(f"=" * 60)
    
    np.random.seed(seed)
    
    # Load model
    print(f"Loading model {model_name}...")
    llm, tokenizer = load_model(model_name)
    model_wrapper = LLMWrapper(llm, tokenizer, model_name)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        problems = json.load(f)
    
    if limit:
        problems = problems[:limit]
    
    print(f"Running on {len(problems)} problems with {num_samples} samples each...")
    
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        # Build prompts
        prompt = COT_PROMPT.format(question=question)
        prompts = [prompt] * num_samples
        
        # Generate 16 samples in parallel
        gen_start = time.time()
        responses = model_wrapper.generate_batch(
            prompts,
            temperature=temperature,
            max_tokens=1024
        )
        gen_time = time.time() - gen_start
        
        # Extract answers from all samples
        answers = []
        for response in responses:
            pred = extract_answer_from_text(response, dataset_type)
            answers.append(pred)
        
        # Majority voting
        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            answer_counts = Counter(valid_answers)
            consensus_answer = answer_counts.most_common(1)[0][0]
        else:
            consensus_answer = None
        
        # Check if consensus is correct
        pred_normalized = normalize_answer(consensus_answer)
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_normalized == gold_normalized
        
        # Also check Pass@1 (any single sample correct)
        any_correct = any(
            normalize_answer(a) == gold_normalized 
            for a in valid_answers
        )
        
        if is_correct:
            correct += 1
        
        # Estimate tokens
        tokens_this_problem = sum(len(r.split()) for r in responses)
        total_tokens += tokens_this_problem
        
        results.append({
            'id': problem['id'],
            'question': question,
            'gold_answer': gold_answer,
            'consensus_answer': consensus_answer,
            'all_answers': answers,
            'correct': is_correct,
            'pass_at_1': any_correct,
            'tokens': tokens_this_problem,
            'latency': gen_time,
            'responses': responses,
        })
        
        if (i + 1) % 10 == 0:
            acc = correct / (i + 1)
            print(f"  Processed {i+1}/{len(problems)} | Accuracy: {acc:.3f}")
    
    total_time = time.time() - start_time
    
    # Compute metrics
    accuracy = correct / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    avg_latency = total_time / len(results) if results else 0
    
    print(f"\n{'=' * 60}")
    print(f"Results Summary")
    print(f"{'=' * 60}")
    print(f"Accuracy (Consensus): {accuracy:.4f} ({correct}/{len(results)})")
    print(f"Avg Tokens: {avg_tokens:.1f}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Total Time: {total_time:.1f}s")
    
    # Save results
    output = {
        'experiment': f'sc{num_samples}_{model_name}_seed{seed}',
        'model': model_name,
        'dataset': dataset_path,
        'seed': seed,
        'num_samples': num_samples,
        'metrics': {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'avg_tokens': avg_tokens,
            'avg_latency': avg_latency,
            'total_time': total_time,
        },
        'results': results,
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3.1-8b')
    parser.add_argument('--dataset', type=str, default='data/gsm8k.json')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        dataset_name = os.path.basename(args.dataset).replace('.json', '')
        args.output = f'results/baseline_sc16/{args.model}_{dataset_name}_seed{args.seed}.json'
    
    run_self_consistency(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed,
        num_samples=args.samples,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
