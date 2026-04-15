#!/usr/bin/env python3
"""
Baseline 1: Standard Chain-of-Thought
Single-pass CoT reasoning with greedy decoding.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
from pathlib import Path
import numpy as np
from shared.model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text


COT_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""


def run_cot_baseline(
    model_name: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    max_tokens: int = 2048,
    limit: int = None,
):
    """Run standard CoT baseline."""
    print(f"=" * 60)
    print(f"Baseline CoT: {model_name}, Seed: {seed}")
    print(f"=" * 60)
    
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
    
    print(f"Running on {len(problems)} problems...")
    
    # Run inference
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        # Build prompt
        prompt = COT_PROMPT.format(question=question)
        
        # Generate
        gen_start = time.time()
        response = model_wrapper.generate(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens
        )
        gen_time = time.time() - gen_start
        
        # Extract answer
        pred_answer = extract_answer_from_text(response, dataset_type)
        
        # Normalize and compare
        pred_normalized = normalize_answer(pred_answer)
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_normalized == gold_normalized
        
        if is_correct:
            correct += 1
        
        # Estimate tokens
        num_tokens = len(response.split())  # Rough estimate
        total_tokens += num_tokens
        
        results.append({
            'id': problem['id'],
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': pred_answer,
            'response': response,
            'correct': is_correct,
            'tokens': num_tokens,
            'latency': gen_time,
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
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    print(f"Avg Tokens: {avg_tokens:.1f}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Total Time: {total_time:.1f}s")
    
    # Save results
    output = {
        'experiment': f'baseline_cot_{model_name}_seed{seed}',
        'model': model_name,
        'dataset': dataset_path,
        'seed': seed,
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
    parser.add_argument('--model', type=str, default='llama-3.1-8b',
                        choices=['llama-3.1-8b', 'qwen2.5-7b', 'deepseek-r1-7b'])
    parser.add_argument('--dataset', type=str, default='data/gsm8k.json')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        dataset_name = os.path.basename(args.dataset).replace('.json', '')
        args.output = f'results/baseline_cot/{args.model}_{dataset_name}_seed{args.seed}.json'
    
    run_cot_baseline(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
