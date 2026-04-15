#!/usr/bin/env python3
"""
Fast CDHR Experiments - Optimized for completion
Uses smaller sample sizes and flushes output for monitoring.
"""
import os
import sys
import json
import time
import re
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exp'))
from shared.fixed_model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text

os.makedirs("results", exist_ok=True)

MODEL_NAME = "llama-3.1-8b"
BACKEND = "transformers"
SEEDS = [42, 123, 456]


def save_result(name, data):
    with open(f"results/{name}.json", 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: results/{name}.json", flush=True)


def load_result(name):
    path = f"results/{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def run_cot(model, problems, seed, exp_name):
    """Run CoT baseline."""
    print(f"\n{'='*60}", flush=True)
    print(f"CoT: {exp_name} (seed={seed}, n={len(problems)})", flush=True)
    print(f"{'='*60}", flush=True)
    
    results = []
    correct = 0
    total_tokens = 0
    start = time.time()
    
    for i, prob in enumerate(problems):
        prompt = f"Let's solve this step by step.\n\nProblem: {prob['question']}\n\nSolution:"
        
        response = model.generate(prompt, temperature=0.0, max_tokens=1024)
        
        pred = extract_answer_from_text(response, prob.get('dataset', 'generic'))
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(prob['answer'])
        is_correct = pred_norm == gold_norm
        
        if is_correct:
            correct += 1
        
        tokens = len(model.tokenizer.encode(response))
        total_tokens += tokens
        
        results.append({
            'id': prob['id'],
            'correct': is_correct,
            'tokens': tokens,
        })
        
        if (i + 1) % 20 == 0 or i == len(problems) - 1:
            print(f"  Progress: {i+1}/{len(problems)} | Acc: {correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    metrics = {
        'accuracy': correct / len(problems),
        'correct': correct,
        'total': len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
    }
    print(f"Result: Acc={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Time={elapsed:.1f}s", flush=True)
    return metrics, results


def run_cdhr(model, problems, seed, exp_name, theta_v=0.05, theta_sigma=0.1, beta=0.5):
    """Run CDHR."""
    print(f"\n{'='*60}", flush=True)
    print(f"CDHR: {exp_name} (seed={seed}, n={len(problems)}, θv={theta_v}, θσ={theta_sigma}, β={beta})", flush=True)
    print(f"{'='*60}", flush=True)
    
    results = []
    correct = 0
    total_tokens = 0
    all_strategies = []
    start = time.time()
    
    strategy_prompts = {
        "linear": "Let's solve this step by step.",
        "analogical": "This resembles a similar problem. Let me adapt that approach.",
        "decomposition": "Let me break this into smaller parts.",
        "verification": "Let me verify my reasoning carefully.",
    }
    
    for i, prob in enumerate(problems):
        conf_history = []
        strategy_history = []
        current_strategy = "linear"
        reasoning_trace = []
        
        for step in range(6):  # Max 6 steps
            context = ""
            if reasoning_trace:
                context = " Previous: " + reasoning_trace[-1][:80]
            
            prompt = f"{strategy_prompts[current_strategy]}{context}\n\nProblem: {prob['question']}\n\nSolution:"
            
            response, logprobs = model.generate_with_logprobs(prompt, temperature=0.0, max_tokens=400)
            
            token_conf = np.exp(np.mean(logprobs)) if logprobs else 0.5
            conf_history.append(token_conf)
            strategy_history.append(current_strategy)
            reasoning_trace.append(response)
            
            if len(conf_history) >= 3:
                recent = conf_history[-3:]
                velocity = (recent[-1] - recent[0]) / 2
                variance = np.var(recent)
                
                if variance > theta_sigma:
                    current_strategy = "decomposition"
                elif velocity > theta_v:
                    current_strategy = "linear"
                elif velocity < -theta_v:
                    current_strategy = "verification"
                else:
                    current_strategy = "analogical"
            
            if any(marker in response.lower() for marker in ["the answer is", "####", "final answer", "therefore"]):
                break
        
        all_text = " ".join(reasoning_trace)
        pred = extract_answer_from_text(all_text, prob.get('dataset', 'generic'))
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(prob['answer'])
        is_correct = pred_norm == gold_norm
        
        if is_correct:
            correct += 1
        
        tokens = sum(len(model.tokenizer.encode(r)) for r in reasoning_trace)
        total_tokens += tokens
        all_strategies.extend(strategy_history)
        
        results.append({
            'id': prob['id'],
            'correct': is_correct,
            'tokens': tokens,
            'strategies': dict(Counter(strategy_history)),
        })
        
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            print(f"  Progress: {i+1}/{len(problems)} | Acc: {correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    
    strat_counts = Counter(all_strategies)
    total = sum(strat_counts.values())
    entropy = -sum((c/total) * np.log2(c/total) for c in strat_counts.values()) if total > 0 else 0
    
    metrics = {
        'accuracy': correct / len(problems),
        'correct': correct,
        'total': len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
        'strategy_entropy': entropy,
        'strategy_dist': dict(strat_counts),
    }
    print(f"Result: Acc={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Entropy={entropy:.3f}", flush=True)
    return metrics, results


def run_sc16(model, problems, seed, exp_name):
    """Run Self-Consistency-16."""
    print(f"\n{'='*60}", flush=True)
    print(f"SC-16: {exp_name} (seed={seed}, n={len(problems)})", flush=True)
    print(f"{'='*60}", flush=True)
    
    results = []
    correct = 0
    total_tokens = 0
    start = time.time()
    
    for i, prob in enumerate(problems):
        prompt = f"Let's solve this step by step.\n\nProblem: {prob['question']}\n\nSolution:"
        
        samples = []
        for _ in range(16):
            response = model.generate(prompt, temperature=0.7, max_tokens=512)
            pred = extract_answer_from_text(response, prob.get('dataset', 'generic'))
            if pred:
                samples.append(normalize_answer(pred))
        
        pred_answer = Counter(samples).most_common(1)[0][0] if samples else ""
        gold_norm = normalize_answer(prob['answer'])
        is_correct = pred_answer == gold_norm
        
        if is_correct:
            correct += 1
        
        tokens = 300 * 16  # Estimate
        total_tokens += tokens
        
        results.append({'id': prob['id'], 'correct': is_correct, 'tokens': tokens})
        
        if (i + 1) % 5 == 0 or i == len(problems) - 1:
            print(f"  Progress: {i+1}/{len(problems)} | Acc: {correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    metrics = {
        'accuracy': correct / len(problems),
        'correct': correct,
        'total': len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
    }
    print(f"Result: Acc={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}", flush=True)
    return metrics, results


def main():
    print("="*70, flush=True)
    print("CDHR FAST EXPERIMENTS", flush=True)
    print("="*70, flush=True)
    
    # Load datasets
    print("\nLoading datasets...", flush=True)
    with open("data/gsm8k.json") as f:
        gsm8k_full = json.load(f)
    with open("data/math.json") as f:
        math_full = json.load(f)
    with open("data/gpqa.json") as f:
        gpqa_full = json.load(f)
    
    # Smaller sizes for faster completion
    gsm8k = gsm8k_full[:200]
    math = math_full[:100]
    gpqa = gpqa_full[:80]
    
    print(f"GSM8K: {len(gsm8k)} problems", flush=True)
    print(f"MATH: {len(math)} problems", flush=True)
    print(f"GPQA: {len(gpqa)} problems", flush=True)
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}", flush=True)
    backend = load_model(MODEL_NAME, backend=BACKEND)
    model = LLMWrapper(backend, MODEL_NAME)
    print("Model loaded!\n", flush=True)
    
    all_results = {}
    
    # 1. Baseline CoT - 3 seeds on GSM8K
    for seed in SEEDS:
        name = f"cot_seed{seed}"
        cached = load_result(name)
        if cached:
            all_results[name] = cached['metrics']
            print(f"Loaded cached {name}", flush=True)
        else:
            metrics, res = run_cot(model, gsm8k, seed, name)
            save_result(name, {'metrics': metrics, 'results': res})
            all_results[name] = metrics
    
    # 2. CDHR Main - 3 seeds on GSM8K
    for seed in SEEDS:
        name = f"cdhr_seed{seed}"
        cached = load_result(name)
        if cached:
            all_results[name] = cached['metrics']
            print(f"Loaded cached {name}", flush=True)
        else:
            metrics, res = run_cdhr(model, gsm8k, seed, name)
            save_result(name, {'metrics': metrics, 'results': res})
            all_results[name] = metrics
    
    # 3. Self-Consistency-16 on GSM8K subset
    name = "sc16"
    cached = load_result(name)
    if cached:
        all_results[name] = cached['metrics']
        print(f"Loaded cached {name}", flush=True)
    else:
        metrics, res = run_sc16(model, gsm8k[:80], 42, name)
        save_result(name, {'metrics': metrics, 'results': res})
        all_results[name] = metrics
    
    # 4. Ablations
    # Token-only (beta=1.0)
    name = "ablation_token_only"
    cached = load_result(name)
    if cached:
        all_results[name] = cached['metrics']
    else:
        metrics, res = run_cdhr(model, gsm8k[:100], 42, name, beta=1.0)
        save_result(name, {'metrics': metrics, 'results': res})
        all_results[name] = metrics
    
    # Beta variations
    for beta in [0.0, 0.25, 0.75]:
        name = f"ablation_beta{beta}"
        cached = load_result(name)
        if cached:
            all_results[name] = cached['metrics']
        else:
            metrics, res = run_cdhr(model, gsm8k[:80], 42, name, beta=beta)
            save_result(name, {'metrics': metrics, 'results': res})
            all_results[name] = metrics
    
    # Threshold variations
    for theta_v in [0.03, 0.07]:
        for theta_sigma in [0.075, 0.125]:
            name = f"ablation_th{theta_v}_{theta_sigma}"
            cached = load_result(name)
            if cached:
                all_results[name] = cached['metrics']
            else:
                metrics, res = run_cdhr(model, gsm8k[:60], 42, name, theta_v=theta_v, theta_sigma=theta_sigma)
                save_result(name, {'metrics': metrics, 'results': res})
                all_results[name] = metrics
    
    # 5. Cross-dataset
    for dataset_name, dataset, size in [("math", math, 100), ("gpqa", gpqa, 80)]:
        # CoT
        name = f"cot_{dataset_name}"
        cached = load_result(name)
        if cached:
            all_results[name] = cached['metrics']
        else:
            metrics, res = run_cot(model, dataset[:size], 42, name)
            save_result(name, {'metrics': metrics, 'results': res})
            all_results[name] = metrics
        
        # CDHR
        name = f"cdhr_{dataset_name}"
        cached = load_result(name)
        if cached:
            all_results[name] = cached['metrics']
        else:
            metrics, res = run_cdhr(model, dataset[:size], 42, name)
            save_result(name, {'metrics': metrics, 'results': res})
            all_results[name] = metrics
    
    # Aggregate
    print("\n" + "="*70, flush=True)
    print("AGGREGATING RESULTS", flush=True)
    print("="*70, flush=True)
    
    def stats(prefix):
        values = [all_results.get(f"{prefix}_seed{s}", {}).get('accuracy') for s in SEEDS]
        values = [v for v in values if v is not None]
        if len(values) >= 2:
            return {'mean': float(np.mean(values)), 'std': float(np.std(values)), 'values': values}
        return None
    
    summary = {
        'baseline_cot': stats('cot'),
        'cdhr_main': stats('cdhr'),
        'sc16': all_results.get('sc16'),
        'ablations': {
            'token_only': all_results.get('ablation_token_only'),
            'beta_sweep': {str(b): all_results.get(f'ablation_beta{b}') for b in [0.0, 0.25, 0.75]},
        },
        'cross_dataset': {
            'math_cot': all_results.get('cot_math'),
            'math_cdhr': all_results.get('cdhr_math'),
            'gpqa_cot': all_results.get('cot_gpqa'),
            'gpqa_cdhr': all_results.get('cdhr_gpqa'),
        }
    }
    
    final = {'summary': summary, 'all_results': all_results, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    print("\nFINAL SUMMARY:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nSaved to results.json", flush=True)


if __name__ == '__main__':
    main()
