#!/usr/bin/env python3
"""
Minimal CDHR Experiments - Guaranteed to complete
Uses smaller problem sets and simpler inference.
"""
import os
import sys
import json
import time
import numpy as np
from collections import Counter

sys.path.insert(0, 'exp')
from shared.fixed_model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text

os.makedirs("results", exist_ok=True)


def extract_answer(text):
    """Extract numerical answer."""
    import re
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'the answer is\s*(-?\d+(?:\.\d+)?)',
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return m.group(1)
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def run_cot(model, problems, seed):
    """Run CoT baseline."""
    print(f"\nCoT (seed={seed}, n={len(problems)})")
    np.random.seed(seed)
    
    results = []
    correct = 0
    total_tokens = 0
    start = time.time()
    
    for i, prob in enumerate(problems):
        prompt = f"Solve step by step:\n{prob['question']}\n\nSolution:"
        response = model.generate(prompt, temperature=0.0, max_tokens=400)
        
        pred = normalize_answer(extract_answer(response))
        gold = normalize_answer(prob['answer'])
        is_correct = pred == gold
        
        if is_correct:
            correct += 1
        
        tokens = len(model.tokenizer.encode(response))
        total_tokens += tokens
        
        results.append({'id': prob['id'], 'correct': is_correct, 'tokens': tokens})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(problems)}: Acc={correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    return {
        'accuracy': correct / len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
        'results': results
    }


def run_cdhr(model, problems, seed, theta_v=0.05, theta_sigma=0.1):
    """Run CDHR."""
    print(f"\nCDHR (seed={seed}, n={len(problems)}, θv={theta_v}, θσ={theta_sigma})")
    np.random.seed(seed)
    
    results = []
    correct = 0
    total_tokens = 0
    all_strategies = []
    start = time.time()
    
    for i, prob in enumerate(problems):
        # CDHR with up to 4 steps
        conf_history = []
        strategy = 'linear'
        responses = []
        
        prompts = {
            'linear': "Solve step by step:",
            'analogical': "Think of a similar problem:",
            'decomposition': "Break into parts:",
            'verification': "Verify your work:",
        }
        
        for step in range(4):
            prompt = f"{prompts[strategy]}\n{prob['question']}\n\nSolution:"
            response, logprobs = model.generate_with_logprobs(prompt, max_tokens=300)
            responses.append(response)
            
            # Confidence from logprobs
            conf = np.exp(np.mean(logprobs)) if logprobs else 0.5
            conf_history.append(conf)
            all_strategies.append(strategy)
            
            # Strategy switch
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
        
        if is_correct:
            correct += 1
        
        tokens = sum(len(model.tokenizer.encode(r)) for r in responses)
        total_tokens += tokens
        
        results.append({'id': prob['id'], 'correct': is_correct, 'tokens': tokens})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(problems)}: Acc={correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    
    # Strategy entropy
    strat_counts = Counter(all_strategies)
    total = sum(strat_counts.values())
    probs = [c/total for c in strat_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    
    return {
        'accuracy': correct / len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
        'strategy_entropy': entropy,
        'results': results
    }


def main():
    print("="*60)
    print("CDHR MINIMAL EXPERIMENTS")
    print("="*60)
    
    # Load data
    with open("data/gsm8k.json") as f:
        gsm8k = json.load(f)[:100]  # Only 100 problems
    
    print(f"\nDataset: GSM8K ({len(gsm8k)} problems)")
    
    # Load model
    print("\nLoading model...")
    backend = load_model("llama-3.1-8b", backend="transformers")
    model = LLMWrapper(backend, "llama-3.1-8b")
    print("Model loaded!\n")
    
    all_results = {}
    
    # Run essential experiments
    # 1. CoT - 3 seeds
    for seed in [42, 123, 456]:
        result = run_cot(model, gsm8k, seed)
        all_results[f'cot_seed{seed}'] = result
        with open(f"results/cot_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # 2. CDHR - 3 seeds
    for seed in [42, 123, 456]:
        result = run_cdhr(model, gsm8k, seed)
        all_results[f'cdhr_seed{seed}'] = result
        with open(f"results/cdhr_seed{seed}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # 3. Ablations (1 seed, smaller sets)
    ablations = [
        ('cdhr_th03', 0.03, 0.075),
        ('cdhr_th07', 0.07, 0.125),
    ]
    
    for name, tv, ts in ablations:
        result = run_cdhr(model, gsm8k[:60], 42, theta_v=tv, theta_sigma=ts)
        all_results[name] = result
        with open(f"results/{name}.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Aggregate
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
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
        'ablations': {
            'th03': {'accuracy': all_results['cdhr_th03']['accuracy']},
            'th07': {'accuracy': all_results['cdhr_th07']['accuracy']},
        }
    }
    
    final = {
        'summary': summary,
        'all_results': {k: {'accuracy': v['accuracy'], 'avg_tokens': v['avg_tokens']} 
                       for k, v in all_results.items()},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    print("\nFINAL SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
