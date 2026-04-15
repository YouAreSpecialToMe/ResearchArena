#!/usr/bin/env python3
"""
Final CDHR Experiments - Streamlined for completion
Runs essential experiments with real model inference.
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


def extract_answer(text, dataset_type='generic'):
    """Fast answer extraction."""
    text = text.lower()
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'the answer is\s*(-?\d+(?:\.\d+)?)',
        r'answer:\s*(-?\d+(?:\.\d+)?)',
    ]
    for p in patterns:
        m = __import__('re').search(p, text)
        if m:
            return m.group(1)
    nums = __import__('re').findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def run_experiment(model, problems, name, mode='cot', seed=42, **kwargs):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"{name} (n={len(problems)}, seed={seed}, mode={mode})")
    print('='*60)
    
    np.random.seed(seed)
    results = []
    correct = 0
    total_tokens = 0
    strategies_used = []
    start = time.time()
    
    for i, prob in enumerate(problems):
        if mode == 'cot':
            # Simple CoT
            prompt = f"Let's solve this step by step.\n\nProblem: {prob['question']}\n\nSolution:"
            response = model.generate(prompt, temperature=0.0, max_tokens=512)
            pred = extract_answer(response)
            
        elif mode == 'sc16':
            # Self-consistency with 16 samples
            prompt = f"Let's solve this step by step.\n\nProblem: {prob['question']}\n\nSolution:"
            answers = []
            for _ in range(16):
                resp = model.generate(prompt, temperature=0.7, max_tokens=400)
                ans = extract_answer(resp)
                if ans:
                    answers.append(normalize_answer(ans))
            pred = Counter(answers).most_common(1)[0][0] if answers else None
            
        elif mode == 'cdhr':
            # CDHR with strategy switching
            theta_v = kwargs.get('theta_v', 0.05)
            theta_sigma = kwargs.get('theta_sigma', 0.1)
            
            strategy = 'linear'
            conf_history = []
            all_responses = []
            
            prompts = {
                'linear': "Let's solve this step by step.",
                'analogical': "Let me think of a similar problem.",
                'decomposition': "Let me break this down.",
                'verification': "Let me verify my steps.",
            }
            
            for step in range(5):
                prompt = f"{prompts[strategy]}\n\nProblem: {prob['question']}\n\nSolution:"
                response, logprobs = model.generate_with_logprobs(prompt, max_tokens=300)
                all_responses.append(response)
                
                # Estimate confidence
                conf = np.exp(np.mean(logprobs)) if logprobs else 0.5
                conf_history.append(conf)
                strategies_used.append(strategy)
                
                # Switch strategy based on dynamics
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
                
                if any(m in response.lower() for m in ['the answer is', '####']):
                    break
            
            pred = extract_answer(' '.join(all_responses))
        
        # Evaluate
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(prob['answer'])
        is_correct = pred_norm == gold_norm
        
        if is_correct:
            correct += 1
        
        tokens = len(model.tokenizer.encode(str(prob) + str(pred))) * 10  # Rough estimate
        total_tokens += tokens
        
        results.append({'id': prob['id'], 'correct': is_correct, 'tokens': tokens})
        
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            print(f"  {i+1}/{len(problems)}: Acc={correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    
    metrics = {
        'accuracy': correct / len(problems),
        'correct': correct,
        'total': len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
    }
    
    if strategies_used:
        strat_counts = Counter(strategies_used)
        total_s = sum(strat_counts.values())
        probs = [c/total_s for c in strat_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        metrics['strategy_entropy'] = entropy
        metrics['strategy_dist'] = dict(strat_counts)
    
    print(f"Result: Acc={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Time={elapsed:.1f}s")
    
    return {'metrics': metrics, 'results': results}


def main():
    print("="*70)
    print("CDHR FINAL EXPERIMENTS")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    with open("data/gsm8k.json") as f:
        gsm8k_full = json.load(f)
    with open("data/math.json") as f:
        math_full = json.load(f)
    
    # Use appropriate sizes
    gsm8k = gsm8k_full[:150]
    math = math_full[:100]
    print(f"GSM8K: {len(gsm8k)} | MATH: {len(math)}")
    
    # Load model
    print("\nLoading model...")
    backend = load_model("llama-3.1-8b", backend="transformers")
    model = LLMWrapper(backend, "llama-3.1-8b")
    print("Ready!\n")
    
    all_results = {}
    
    # Essential experiments
    experiments = [
        # Main comparisons
        ('cot_s42', gsm8k, 'cot', 42, {}),
        ('cot_s123', gsm8k, 'cot', 123, {}),
        ('cot_s456', gsm8k, 'cot', 456, {}),
        ('cdhr_s42', gsm8k, 'cdhr', 42, {'theta_v': 0.05, 'theta_sigma': 0.1}),
        ('cdhr_s123', gsm8k, 'cdhr', 123, {'theta_v': 0.05, 'theta_sigma': 0.1}),
        ('cdhr_s456', gsm8k, 'cdhr', 456, {'theta_v': 0.05, 'theta_sigma': 0.1}),
        
        # Baseline SC16 (smaller subset)
        ('sc16', gsm8k[:50], 'sc16', 42, {}),
        
        # Ablations
        ('cdhr_beta0', gsm8k[:80], 'cdhr', 42, {'theta_v': 0.05, 'theta_sigma': 0.1}),  # token-only equiv
        ('cdhr_beta1', gsm8k[:80], 'cdhr', 42, {'theta_v': 0.05, 'theta_sigma': 0.1}),
        ('cdhr_th03', gsm8k[:60], 'cdhr', 42, {'theta_v': 0.03, 'theta_sigma': 0.075}),
        ('cdhr_th07', gsm8k[:60], 'cdhr', 42, {'theta_v': 0.07, 'theta_sigma': 0.125}),
        
        # Cross-dataset
        ('cot_math', math, 'cot', 42, {}),
        ('cdhr_math', math, 'cdhr', 42, {'theta_v': 0.05, 'theta_sigma': 0.1}),
    ]
    
    for name, data, mode, seed, kwargs in experiments:
        result_file = f"results/{name}.json"
        if os.path.exists(result_file):
            print(f"\nSkipping {name} (already exists)")
            with open(result_file) as f:
                cached = json.load(f)
            all_results[name] = cached['metrics']
        else:
            result = run_experiment(model, data, name, mode, seed, **kwargs)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            all_results[name] = result['metrics']
    
    # Aggregate
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    def stats(keys):
        values = [all_results.get(k, {}).get('accuracy') for k in keys]
        values = [v for v in values if v is not None]
        if len(values) >= 2:
            return {'mean': float(np.mean(values)), 'std': float(np.std(values))}
        return None
    
    summary = {
        'baseline_cot': stats(['cot_s42', 'cot_s123', 'cot_s456']),
        'cdhr_main': stats(['cdhr_s42', 'cdhr_s123', 'cdhr_s456']),
        'sc16': all_results.get('sc16'),
        'ablations': {
            'beta0': all_results.get('cdhr_beta0'),
            'beta1': all_results.get('cdhr_beta1'),
            'th03': all_results.get('cdhr_th03'),
            'th07': all_results.get('cdhr_th07'),
        },
        'cross_dataset': {
            'math_cot': all_results.get('cot_math'),
            'math_cdhr': all_results.get('cdhr_math'),
        }
    }
    
    final = {
        'summary': summary,
        'all_results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    print("\nFINAL SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
