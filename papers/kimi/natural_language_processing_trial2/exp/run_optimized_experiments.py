#!/usr/bin/env python3
"""
Optimized CDHR Experiments with batching
Uses batch generation for faster throughput.
"""
import os
import sys
import json
import time
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, 'exp')
from shared.fixed_model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text

os.makedirs("results", exist_ok=True)


def extract_answer(text):
    import re
    patterns = [r'####\s*(-?\d+(?:\.\d+)?)', r'the answer is\s*(-?\d+(?:\.\d+)?)']
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return m.group(1)
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def run_batch_cot(model, problems, seed, batch_size=4):
    """Run CoT with batching."""
    print(f"\nCoT (seed={seed}, n={len(problems)})")
    np.random.seed(seed)
    
    results = [None] * len(problems)
    correct = 0
    total_tokens = 0
    start = time.time()
    
    # Process in batches
    for batch_start in range(0, len(problems), batch_size):
        batch_end = min(batch_start + batch_size, len(problems))
        batch = problems[batch_start:batch_end]
        
        prompts = [f"Solve step by step:\n{p['question']}\n\nSolution:" for p in batch]
        responses = model.generate_batch(prompts, temperature=0.0, max_tokens=350)
        
        for i, (prob, response) in enumerate(zip(batch, responses)):
            idx = batch_start + i
            pred = normalize_answer(extract_answer(response))
            gold = normalize_answer(prob['answer'])
            is_correct = pred == gold
            
            if is_correct:
                correct += 1
            
            tokens = len(model.tokenizer.encode(response))
            total_tokens += tokens
            
            results[idx] = {'id': prob['id'], 'correct': is_correct, 'tokens': tokens}
        
        if batch_end % 20 == 0 or batch_end == len(problems):
            print(f"  {batch_end}/{len(problems)}: Acc={correct/batch_end:.3f}", flush=True)
    
    elapsed = time.time() - start
    return {
        'accuracy': correct / len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
        'results': results
    }


def run_single_cdhr(model, prob, seed, theta_v, theta_sigma):
    """Run CDHR on a single problem."""
    np.random.seed(seed)
    
    conf_history = []
    strategy = 'linear'
    responses = []
    all_strategies = []
    
    prompts = {
        'linear': "Solve step by step:",
        'analogical': "Think of a similar problem:",
        'decomposition': "Break into parts:",
        'verification': "Verify your work:",
    }
    
    for step in range(4):
        prompt = f"{prompts[strategy]}\n{prob['question']}\n\nSolution:"
        response, logprobs = model.generate_with_logprobs(prompt, max_tokens=250)
        responses.append(response)
        
        conf = np.exp(np.mean(logprobs)) if logprobs else 0.5
        conf_history.append(conf)
        all_strategies.append(strategy)
        
        if len(conf_history) >= 3:
            recent = conf_history[-3:]
            vel = (recent[-1] - recent[0]) / 2
            var = np.var(recent)
            
            if var > theta_sigma:
                strategy = 'decomposition'
            elif vel > theta_v:
                strategy = 'linear'
            elif vel < -theta_v:
                strategy = 'verification'
            else:
                strategy = 'analogical'
        
        if 'answer is' in response.lower():
            break
    
    pred = normalize_answer(extract_answer(' '.join(responses)))
    gold = normalize_answer(prob['answer'])
    is_correct = pred == gold
    
    tokens = sum(len(model.tokenizer.encode(r)) for r in responses)
    
    return {
        'id': prob['id'],
        'correct': is_correct,
        'tokens': tokens,
        'strategies': all_strategies
    }


def run_cdhr(model, problems, seed, theta_v=0.05, theta_sigma=0.1):
    """Run CDHR."""
    print(f"\nCDHR (seed={seed}, n={len(problems)})")
    
    results = []
    all_strategies = []
    start = time.time()
    
    for i, prob in enumerate(problems):
        result = run_single_cdhr(model, prob, seed + i, theta_v, theta_sigma)
        results.append(result)
        all_strategies.extend(result['strategies'])
        
        if (i + 1) % 10 == 0:
            correct = sum(r['correct'] for r in results)
            print(f"  {i+1}/{len(problems)}: Acc={correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    
    strat_counts = Counter(all_strategies)
    total = sum(strat_counts.values())
    probs = [c/total for c in strat_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    return {
        'accuracy': sum(r['correct'] for r in results) / len(problems),
        'avg_tokens': sum(r['tokens'] for r in results) / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
        'strategy_entropy': entropy,
        'results': results
    }


def main():
    print("="*60)
    print("CDHR OPTIMIZED EXPERIMENTS")
    print("="*60)
    
    # Load data
    with open("data/gsm8k.json") as f:
        gsm8k = json.load(f)[:80]  # 80 problems
    
    print(f"\nDataset: GSM8K ({len(gsm8k)} problems)")
    
    # Load model
    print("\nLoading model...")
    backend = load_model("llama-3.1-8b", backend="transformers")
    model = LLMWrapper(backend, "llama-3.1-8b")
    print("Model loaded!\n")
    
    all_results = {}
    
    # Run experiments
    # CoT - 3 seeds
    for seed in [42, 123, 456]:
        result = run_batch_cot(model, gsm8k, seed, batch_size=4)
        all_results[f'cot_seed{seed}'] = result
        with open(f"results/cot_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: results/cot_seed{seed}.json")
    
    # CDHR - 3 seeds
    for seed in [42, 123, 456]:
        result = run_cdhr(model, gsm8k, seed)
        all_results[f'cdhr_seed{seed}'] = result
        with open(f"results/cdhr_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: results/cdhr_seed{seed}.json")
    
    # Ablations
    for name, tv, ts in [('cdhr_th03', 0.03, 0.075), ('cdhr_th07', 0.07, 0.125)]:
        result = run_cdhr(model, gsm8k[:50], 42, theta_v=tv, theta_sigma=ts)
        all_results[name] = result
        with open(f"results/{name}.json", 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: results/{name}.json")
    
    # Aggregate
    print("\n" + "="*60)
    print("AGGREGATING")
    print("="*60)
    
    def stats(keys):
        accs = [all_results[k]['accuracy'] for k in keys]
        tokens = [all_results[k]['avg_tokens'] for k in keys]
        return {
            'accuracy': {'mean': float(np.mean(accs)), 'std': float(np.std(accs))},
            'avg_tokens': {'mean': float(np.mean(tokens)), 'std': float(np.std(tokens))},
        }
    
    summary = {
        'cot': stats(['cot_seed42', 'cot_seed123', 'cot_seed456']),
        'cdhr': stats(['cdhr_seed42', 'cdhr_seed123', 'cdhr_seed456']),
    }
    
    final = {
        'summary': summary,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
