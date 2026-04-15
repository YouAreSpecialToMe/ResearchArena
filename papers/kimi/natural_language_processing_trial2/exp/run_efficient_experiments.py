#!/usr/bin/env python3
"""
Efficient batch-based experiment runner for CDHR.
Uses vLLM's batching capabilities for faster inference.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import re
import argparse
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from shared.data_loader import normalize_answer


# ============ Prompts ============

COT_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""

LINEAR_PROMPT = """Let's solve this step by step.

Problem: {question}

Solution:"""

ANALOGICAL_PROMPT = """I need to think about this differently. Let me consider a similar problem.

Problem: {question}

Let me work through this using an analogous approach:"""

DECOMP_PROMPT = """This is complex. Let me break it down into smaller parts.

Problem: {question}

Let me decompose this:
1. First, identify what we know
2. Then, identify what we need to find
3. Finally, connect them step by step

Solution:"""

VERIFY_PROMPT = """Let me carefully verify my reasoning for this problem.

Problem: {question}

Let me check my work carefully:"""


# ============ CDHR Engine ============

class CDHREngine:
    """CDHR engine with strategy selection."""
    
    STRATEGY_PROMPTS = {
        'linear': LINEAR_PROMPT,
        'analogical': ANALOGICAL_PROMPT,
        'decomposition': DECOMP_PROMPT,
        'verification': VERIFY_PROMPT,
    }
    
    def __init__(self, theta_v=0.05, theta_sigma=0.1, window_size=3, max_switches=5, max_steps=8):
        self.theta_v = theta_v
        self.theta_sigma = theta_sigma
        self.window_size = window_size
        self.max_switches = max_switches
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        self.confidence_history = []
        self.strategy_history = []
        self.current_strategy = 'linear'
        self.switch_count = 0
        self.reasoning_parts = []
    
    def estimate_confidence(self, text: str) -> float:
        """Estimate confidence from text heuristics."""
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
        if any(w in text.lower() for w in ['maybe', 'perhaps', 'unclear', 'not sure', 'confused']):
            confidence -= 0.15
        if re.search(r'\d+', text):
            confidence += 0.05
        if 'error' in text.lower() or 'mistake' in text.lower():
            confidence -= 0.1
        return max(0.0, min(1.0, confidence))
    
    def compute_dynamics(self):
        """Compute confidence dynamics."""
        if len(self.confidence_history) < self.window_size:
            return 0.0, 0.0, "progressing"
        
        recent = self.confidence_history[-self.window_size:]
        velocity = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
        variance = np.var(recent)
        
        if variance > self.theta_sigma:
            trajectory = "oscillating"
        elif velocity > self.theta_v:
            trajectory = "progressing"
        elif velocity < -self.theta_v:
            trajectory = "declining"
        else:
            trajectory = "stagnant"
        
        return velocity, variance, trajectory
    
    def select_strategy(self, trajectory: str) -> str:
        """Select strategy based on trajectory."""
        mapping = {
            "progressing": "linear",
            "stagnant": "analogical",
            "oscillating": "decomposition",
            "declining": "verification",
        }
        return mapping.get(trajectory, "linear")
    
    def build_prompt(self, question: str) -> str:
        """Build prompt with current strategy."""
        base = self.STRATEGY_PROMPTS[self.current_strategy]
        context = ""
        if self.reasoning_parts and self.current_strategy != "linear":
            # Add brief context from previous steps
            recent = self.reasoning_parts[-1][:100] if self.reasoning_parts else ""
            context = f"\n\nPrevious insight: {recent}..."
        return base.format(question=question) + context + "\n\n"
    
    def is_complete(self, text: str) -> bool:
        """Check if reasoning is complete."""
        markers = ["the answer is", "therefore", "####", "final answer", "so the answer"]
        return any(m in text.lower() for m in markers)
    
    def extract_answer(self, text: str) -> Optional[str]:
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
    
    def solve(self, model, tokenizer, question: str) -> Dict:
        """Solve problem with CDHR strategy switching."""
        self.reset()
        full_reasoning = []
        start_time = time.time()
        
        for step in range(self.max_steps):
            prompt = self.build_prompt(question)
            
            # Generate
            sampling_params = SamplingParams(temperature=0.0, max_tokens=400)
            outputs = model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            
            # Estimate confidence
            confidence = self.estimate_confidence(response)
            self.confidence_history.append(confidence)
            self.strategy_history.append(self.current_strategy)
            self.reasoning_parts.append(response)
            full_reasoning.append(f"[{self.current_strategy}] {response[:150]}...")
            
            # Compute dynamics and switch if needed
            velocity, variance, trajectory = self.compute_dynamics()
            new_strategy = self.select_strategy(trajectory)
            
            if new_strategy != self.current_strategy and self.switch_count < self.max_switches:
                self.current_strategy = new_strategy
                self.switch_count += 1
            
            if self.is_complete(response):
                break
        
        latency = time.time() - start_time
        all_text = " ".join(self.reasoning_parts)
        
        # Strategy distribution
        strategy_dist = {}
        for s in self.strategy_history:
            strategy_dist[s] = strategy_dist.get(s, 0) + 1
        
        return {
            'answer': self.extract_answer(all_text),
            'reasoning': '\n'.join(full_reasoning),
            'steps': len(self.strategy_history),
            'switches': self.switch_count,
            'strategy_dist': strategy_dist,
            'confidence_traj': self.confidence_history,
            'tokens': len(all_text.split()),
            'latency': latency,
        }


def run_cot_baseline(model, tokenizer, problems: List[Dict]) -> List[Dict]:
    """Run CoT baseline on problems."""
    results = []
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    
    for i, problem in enumerate(problems):
        prompt = COT_PROMPT.format(question=problem['question'])
        
        start = time.time()
        outputs = model.generate([prompt], sampling_params)
        latency = time.time() - start
        
        response = outputs[0].outputs[0].text
        
        # Extract answer
        patterns = [r'####\s*(-?\d+(?:\.\d+)?)', r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)']
        pred_answer = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                pred_answer = match.group(1)
                break
        if not pred_answer:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
            pred_answer = numbers[-1] if numbers else None
        
        results.append({
            'id': problem['id'],
            'question': problem['question'],
            'gold': problem['answer'],
            'predicted': pred_answer,
            'response': response,
            'tokens': len(response.split()),
            'latency': latency,
        })
        
        if (i + 1) % 10 == 0:
            print(f"  CoT: {i+1}/{len(problems)}")
    
    return results


def run_cdhr(model, tokenizer, problems: List[Dict], theta_v=0.05, theta_sigma=0.1) -> List[Dict]:
    """Run CDHR on problems."""
    results = []
    engine = CDHREngine(theta_v=theta_v, theta_sigma=theta_sigma)
    
    for i, problem in enumerate(problems):
        output = engine.solve(model, tokenizer, problem['question'])
        
        results.append({
            'id': problem['id'],
            'question': problem['question'],
            'gold': problem['answer'],
            'predicted': output['answer'],
            **output
        })
        
        if (i + 1) % 10 == 0:
            print(f"  CDHR: {i+1}/{len(problems)}")
    
    return results


def evaluate(results: List[Dict]) -> Dict:
    """Evaluate results."""
    correct = 0
    total_tokens = 0
    total_latency = 0
    
    for r in results:
        pred = normalize_answer(r['predicted'])
        gold = normalize_answer(r['gold'])
        if pred == gold:
            correct += 1
        total_tokens += r.get('tokens', 0)
        total_latency += r.get('latency', 0)
    
    n = len(results)
    return {
        'accuracy': correct / n if n > 0 else 0,
        'correct': correct,
        'total': n,
        'avg_tokens': total_tokens / n if n > 0 else 0,
        'avg_latency': total_latency / n if n > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3.1-8b')
    parser.add_argument('--method', type=str, default='cot', choices=['cot', 'cdhr'])
    parser.add_argument('--dataset', type=str, default='data/gsm8k.json')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--theta_v', type=float, default=0.05)
    parser.add_argument('--theta_sigma', type=float, default=0.1)
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.method} | Model: {args.model}")
    print(f"{'='*60}")
    
    # Load model
    model_paths = {
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'deepseek-r1-7b': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    }
    model_path = model_paths.get(args.model, args.model)
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=8192, trust_remote_code=True)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    with open(args.dataset, 'r') as f:
        problems = json.load(f)
    problems = problems[:args.limit]
    print(f"Running on {len(problems)} problems")
    
    # Run experiment
    start_time = time.time()
    
    if args.method == 'cot':
        results = run_cot_baseline(model, tokenizer, problems)
    else:
        results = run_cdhr(model, tokenizer, problems, args.theta_v, args.theta_sigma)
    
    total_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate(results)
    metrics['total_time'] = total_time
    
    # Add strategy entropy for CDHR
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
    print(f"Results: Accuracy={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}")
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
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
