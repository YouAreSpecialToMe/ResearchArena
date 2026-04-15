#!/usr/bin/env python3
"""
Paper Experiments - Focused on essential results
Runs key experiments with real model inference.
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
    import re
    patterns = [r'####\s*(-?\d+(?:\.\d+)?)', r'the answer is\s*(-?\d+(?:\.\d+)?)']
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return m.group(1)
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return nums[-1] if nums else None


def run_experiment(model, problems, name, config):
    """Generic experiment runner."""
    print(f"\n{'='*60}")
    print(f"{name} (n={len(problems)})")
    print('='*60)
    
    mode = config.get('mode', 'cot')
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    results = []
    correct = 0
    total_tokens = 0
    all_strategies = []
    start = time.time()
    
    for i, prob in enumerate(problems):
        if mode == 'cot':
            # Simple CoT
            prompt = f"Solve step by step:\n{prob['question']}\n\nSolution:"
            response = model.generate(prompt, temperature=0.0, max_tokens=400)
            pred = normalize_answer(extract_answer(response))
            tokens = len(model.tokenizer.encode(response))
            
        elif mode == 'sc8':
            # Self-consistency with 8 samples
            prompt = f"Solve step by step:\n{prob['question']}\n\nSolution:"
            answers = []
            for _ in range(8):
                resp = model.generate(prompt, temperature=0.7, max_tokens=350)
                ans = extract_answer(resp)
                if ans:
                    answers.append(normalize_answer(ans))
            pred = Counter(answers).most_common(1)[0][0] if answers else None
            tokens = 250 * 8  # Estimate
            
        elif mode == 'cdhr':
            # CDHR with dynamics
            theta_v = config.get('theta_v', 0.05)
            theta_sigma = config.get('theta_sigma', 0.1)
            
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
            tokens = sum(len(model.tokenizer.encode(r)) for r in responses)
        
        gold = normalize_answer(prob['answer'])
        is_correct = (pred == gold) if pred else False
        
        if is_correct:
            correct += 1
        total_tokens += tokens
        
        results.append({'id': prob['id'], 'correct': is_correct, 'tokens': tokens})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(problems)}: Acc={correct/(i+1):.3f}", flush=True)
    
    elapsed = time.time() - start
    
    metrics = {
        'accuracy': correct / len(problems),
        'avg_tokens': total_tokens / len(problems),
        'avg_latency': elapsed / len(problems),
        'total_time': elapsed,
        'results': results
    }
    
    if all_strategies:
        strat_counts = Counter(all_strategies)
        total = sum(strat_counts.values())
        probs = [c/total for c in strat_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        metrics['strategy_entropy'] = entropy
        metrics['strategy_dist'] = dict(strat_counts)
    
    print(f"Result: Acc={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Time={elapsed:.1f}s")
    return metrics


def main():
    print("="*60)
    print("CDHR PAPER EXPERIMENTS")
    print("="*60)
    
    # Load datasets
    with open("data/gsm8k.json") as f:
        gsm8k = json.load(f)[:100]  # 100 problems
    with open("data/math.json") as f:
        math = json.load(f)[:60]   # 60 problems
    
    print(f"\nDatasets: GSM8K={len(gsm8k)}, MATH={len(math)}")
    
    # Load model
    print("\nLoading model...")
    backend = load_model("llama-3.1-8b", backend="transformers")
    model = LLMWrapper(backend, "llama-3.1-8b")
    print("Model loaded!\n")
    
    all_results = {}
    
    # Essential experiments
    experiments = [
        # Main comparisons on GSM8K - 3 seeds
        ('cot_s42', gsm8k, {'mode': 'cot', 'seed': 42}),
        ('cot_s123', gsm8k, {'mode': 'cot', 'seed': 123}),
        ('cot_s456', gsm8k, {'mode': 'cot', 'seed': 456}),
        ('cdhr_s42', gsm8k, {'mode': 'cdhr', 'seed': 42, 'theta_v': 0.05, 'theta_sigma': 0.1}),
        ('cdhr_s123', gsm8k, {'mode': 'cdhr', 'seed': 123, 'theta_v': 0.05, 'theta_sigma': 0.1}),
        ('cdhr_s456', gsm8k, {'mode': 'cdhr', 'seed': 456, 'theta_v': 0.05, 'theta_sigma': 0.1}),
        
        # Self-consistency baseline
        ('sc8', gsm8k[:50], {'mode': 'sc8', 'seed': 42}),
        
        # Ablations on subset
        ('cdhr_th03', gsm8k[:60], {'mode': 'cdhr', 'seed': 42, 'theta_v': 0.03, 'theta_sigma': 0.075}),
        ('cdhr_th07', gsm8k[:60], {'mode': 'cdhr', 'seed': 42, 'theta_v': 0.07, 'theta_sigma': 0.125}),
        
        # Cross-dataset validation
        ('cot_math', math, {'mode': 'cot', 'seed': 42}),
        ('cdhr_math', math, {'mode': 'cdhr', 'seed': 42, 'theta_v': 0.05, 'theta_sigma': 0.1}),
    ]
    
    for name, data, config in experiments:
        result = run_experiment(model, data, name, config)
        all_results[name] = result
        with open(f"results/{name}.json", 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: results/{name}.json\n")
    
    # Aggregate results
    print("="*60)
    print("AGGREGATING RESULTS")
    print("="*60)
    
    def compute_stats(keys):
        accs = [all_results[k]['accuracy'] for k in keys]
        tokens = [all_results[k]['avg_tokens'] for k in keys]
        return {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'tokens_mean': float(np.mean(tokens)),
        }
    
    summary = {
        'cot_gsm8k': compute_stats(['cot_s42', 'cot_s123', 'cot_s456']),
        'cdhr_gsm8k': compute_stats(['cdhr_s42', 'cdhr_s123', 'cdhr_s456']),
        'sc8': {'accuracy': all_results['sc8']['accuracy'], 'avg_tokens': all_results['sc8']['avg_tokens']},
        'ablations': {
            'th03': {'accuracy': all_results['cdhr_th03']['accuracy']},
            'th07': {'accuracy': all_results['cdhr_th07']['accuracy']},
        },
        'cross_dataset': {
            'cot_math': {'accuracy': all_results['cot_math']['accuracy']},
            'cdhr_math': {'accuracy': all_results['cdhr_math']['accuracy']},
        }
    }
    
    # Add strategy entropy from main CDHR run
    if 'strategy_entropy' in all_results['cdhr_s42']:
        summary['cdhr_gsm8k']['strategy_entropy'] = all_results['cdhr_s42']['strategy_entropy']
    
    final_output = {
        'summary': summary,
        'all_results': {k: {'accuracy': v['accuracy'], 'avg_tokens': v['avg_tokens']} 
                       for k, v in all_results.items()},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print("\nFINAL SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
