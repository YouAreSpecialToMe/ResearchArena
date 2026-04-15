#!/usr/bin/env python3
"""
Baseline: Chain of Mindset (CoM) - Simplified Implementation
CoM uses multiple reasoning mindsets (Spatial, Convergent, Divergent, Algorithmic)
with a Meta-Agent selecting between them.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import argparse
import numpy as np
from shared.model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text


# Define the four mindsets from CoM paper
MINDSET_PROMPTS = {
    'spatial': """Approach this problem using spatial and visual reasoning. 
Imagine the problem as a diagram or spatial layout. 
What visual representations can help solve this?""",
    
    'convergent': """Approach this problem using focused, logical deduction.
Identify the key facts and apply direct reasoning to reach the solution.
Work systematically toward the answer.""",
    
    'divergent': """Approach this problem by exploring multiple possibilities.
Consider different angles and approaches. What alternative strategies might work?
Generate several ideas before committing to one.""",
    
    'algorithmic': """Approach this problem using step-by-step procedures.
Break down the solution into clear, sequential steps.
Follow a methodical algorithm to solve this.""",
}

META_AGENT_PROMPT = """Given the following problem, select the best reasoning mindset:
- spatial: for geometry, visualization, layout problems
- convergent: for direct logical deduction problems  
- divergent: for open-ended, creative problems
- algorithmic: for step-by-step, procedural problems

Problem: {question}

Respond with only one word: spatial, convergent, divergent, or algorithmic."""


def select_mindset(model_wrapper: LLMWrapper, question: str) -> str:
    """Meta-Agent selects the best mindset for this problem."""
    prompt = META_AGENT_PROMPT.format(question=question)
    response = model_wrapper.generate(prompt, temperature=0.0, max_tokens=20)
    
    response_lower = response.lower().strip()
    for mindset in MINDSET_PROMPTS.keys():
        if mindset in response_lower:
            return mindset
    
    # Default to algorithmic if unclear
    return 'algorithmic'


def run_chain_of_mindset(
    model_name: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    limit: int = None,
):
    """Run Chain of Mindset baseline."""
    print(f"=" * 60)
    print(f"Chain of Mindset Baseline: {model_name}, Seed: {seed}")
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
    
    print(f"Running on {len(problems)} problems...")
    
    results = []
    correct = 0
    total_tokens = 0
    mindset_counts = {'spatial': 0, 'convergent': 0, 'divergent': 0, 'algorithmic': 0}
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        gen_start = time.time()
        
        # Step 1: Meta-Agent selects mindset
        mindset = select_mindset(model_wrapper, question)
        mindset_counts[mindset] += 1
        
        # Step 2: Generate reasoning with selected mindset
        mindset_prompt = MINDSET_PROMPTS[mindset]
        full_prompt = f"{mindset_prompt}\n\nProblem: {question}\n\nSolution:"
        
        response = model_wrapper.generate(
            full_prompt,
            temperature=0.0,
            max_tokens=1024
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
        num_tokens = len(response.split())
        total_tokens += num_tokens
        
        results.append({
            'id': problem['id'],
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': pred_answer,
            'selected_mindset': mindset,
            'correct': is_correct,
            'tokens': num_tokens,
            'latency': gen_time,
            'response': response,
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
    print(f"Mindset Distribution: {mindset_counts}")
    
    # Compute mindset entropy
    total_mindsets = sum(mindset_counts.values())
    if total_mindsets > 0:
        probabilities = [c / total_mindsets for c in mindset_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    else:
        entropy = 0.0
    print(f"Mindset Entropy: {entropy:.3f} bits")
    
    # Save results
    output = {
        'experiment': f'com_{model_name}_seed{seed}',
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
            'mindset_distribution': mindset_counts,
            'mindset_entropy': entropy,
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
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--retrieval_index', type=str, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        dataset_name = os.path.basename(args.dataset).replace('.json', '')
        args.output = f'results/baseline_com/{args.model}_{dataset_name}_seed{args.seed}.json'
    
    run_chain_of_mindset(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
