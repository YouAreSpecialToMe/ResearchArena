#!/usr/bin/env python3
"""
Simple experiment runner using transformers (more reliable than vLLM for sequential runs).
"""
import json
import time
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


# Simple prompts
COT_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""

STRATEGY_PROMPTS = {
    'linear': """Let's solve this step by step.

Problem: {question}

Solution:""",
    'analogical': """I need to think about this differently. Let me use analogical reasoning.

Problem: {question}

Let me think of a similar problem and adapt its solution:""",
    'decomposition': """This is complex. Let me break it down into smaller parts.

Problem: {question}

Step-by-step decomposition:""",
    'verification': """Let me carefully verify my reasoning.

Problem: {question}

Careful verification:""",
}


def load_model_simple(model_name):
    """Load model with simpler settings."""
    model_paths = {
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'deepseek-r1-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    }
    path = model_paths.get(model_name, model_name)
    
    print(f"Loading {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=512):
    """Generate text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text


def extract_answer(text):
    """Extract numerical answer."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def normalize_answer(answer):
    """Normalize answer."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    answer = re.sub(r'^(?:the answer is|answer:|ans:)\s*', '', answer)
    try:
        num = float(answer.replace(',', ''))
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        pass
    return answer.strip()


def estimate_confidence(text):
    """Estimate confidence from text."""
    confidence = 0.5
    words = len(text.split())
    if words > 30:
        confidence += 0.1
    if words > 60:
        confidence += 0.1
    if re.search(r'[=+\-*/]', text):
        confidence += 0.05
    if any(w in text.lower() for w in ['therefore', 'thus', 'so', 'hence', 'answer is']):
        confidence += 0.1
    if any(w in text.lower() for w in ['maybe', 'perhaps', 'unclear', 'not sure']):
        confidence -= 0.15
    if re.search(r'\d+', text):
        confidence += 0.05
    return max(0.0, min(1.0, confidence))


def run_cot(model, tokenizer, question):
    """Run CoT baseline."""
    prompt = COT_PROMPT.format(question=question)
    start = time.time()
    response = generate(model, tokenizer, prompt, max_new_tokens=400)
    latency = time.time() - start
    
    return {
        'response': response,
        'answer': extract_answer(response),
        'tokens': len(response.split()),
        'latency': latency,
    }


def run_cdhr(model, tokenizer, question, theta_v=0.05, theta_sigma=0.1):
    """Run CDHR with strategy switching."""
    confidence_history = []
    strategy_history = []
    current_strategy = 'linear'
    reasoning_parts = []
    switch_count = 0
    max_switches = 5
    max_steps = 8
    window_size = 3
    
    start = time.time()
    
    for step in range(max_steps):
        # Build prompt with current strategy
        prompt = STRATEGY_PROMPTS[current_strategy].format(question=question)
        if reasoning_parts:
            prompt += "\n\nPrevious reasoning:\n" + reasoning_parts[-1][:100] + "..."
        
        response = generate(model, tokenizer, prompt, max_new_tokens=350)
        
        # Estimate confidence
        confidence = estimate_confidence(response)
        confidence_history.append(confidence)
        strategy_history.append(current_strategy)
        reasoning_parts.append(response)
        
        # Compute dynamics
        if len(confidence_history) >= window_size:
            recent = confidence_history[-window_size:]
            velocity = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
            variance = np.var(recent)
            
            # Select new strategy
            if variance > theta_sigma:
                new_strategy = 'decomposition'
            elif velocity > theta_v:
                new_strategy = 'linear'
            elif velocity < -theta_v:
                new_strategy = 'verification'
            else:
                new_strategy = 'analogical'
            
            if new_strategy != current_strategy and switch_count < max_switches:
                current_strategy = new_strategy
                switch_count += 1
        
        # Check completion
        if any(m in response.lower() for m in ["the answer is", "therefore", "####"]):
            break
    
    latency = time.time() - start
    all_text = " ".join(reasoning_parts)
    
    # Strategy distribution
    strategy_dist = {}
    for s in strategy_history:
        strategy_dist[s] = strategy_dist.get(s, 0) + 1
    
    return {
        'answer': extract_answer(all_text),
        'reasoning': '\n'.join([f"[{s}] {r[:100]}..." for s, r in zip(strategy_history, reasoning_parts)]),
        'steps': len(strategy_history),
        'switches': switch_count,
        'strategy_dist': strategy_dist,
        'confidence_traj': confidence_history,
        'tokens': len(all_text.split()),
        'latency': latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3.1-8b')
    parser.add_argument('--method', type=str, default='cot', choices=['cot', 'cdhr'])
    parser.add_argument('--dataset', type=str, default='data/gsm8k.json')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--theta_v', type=float, default=0.05)
    parser.add_argument('--theta_sigma', type=float, default=0.1)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.method} | Model: {args.model}")
    print(f"{'='*60}")
    
    # Load model
    model, tokenizer = load_model_simple(args.model)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, 'r') as f:
        problems = json.load(f)
    problems = problems[:args.limit]
    print(f"Running on {len(problems)} problems")
    
    # Run experiments
    results = []
    correct = 0
    
    for i, problem in enumerate(problems):
        if args.method == 'cot':
            output = run_cot(model, tokenizer, problem['question'])
        else:
            output = run_cdhr(model, tokenizer, problem['question'], args.theta_v, args.theta_sigma)
        
        pred = normalize_answer(output['answer'])
        gold = normalize_answer(problem['answer'])
        is_correct = pred == gold
        
        if is_correct:
            correct += 1
        
        results.append({
            'id': problem['id'],
            'gold': problem['answer'],
            'predicted': output['answer'],
            'correct': is_correct,
            **output
        })
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(problems)} | Acc: {correct/(i+1):.3f}")
    
    # Compute metrics
    accuracy = correct / len(results) if results else 0
    avg_tokens = sum(r['tokens'] for r in results) / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': len(results),
        'avg_tokens': avg_tokens,
        'avg_latency': avg_latency,
    }
    
    # Strategy entropy for CDHR
    if args.method == 'cdhr':
        all_dist = [r['strategy_dist'] for r in results]
        strat_counts = {}
        for d in all_dist:
            for k, v in d.items():
                strat_counts[k] = strat_counts.get(k, 0) + v
        total = sum(strat_counts.values())
        if total > 0:
            probs = [c/total for c in strat_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            metrics['strategy_entropy'] = entropy
            metrics['strategy_dist'] = strat_counts
    
    print(f"\n{'='*60}")
    print(f"Results: Acc={accuracy:.4f}, Tokens={avg_tokens:.1f}")
    print(f"{'='*60}")
    
    # Save
    output = {
        'experiment': f'{args.method}_{args.model}',
        'method': args.method,
        'model': args.model,
        'dataset': args.dataset,
        'limit': args.limit,
        'metrics': metrics,
        'results': results,
    }
    
    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
