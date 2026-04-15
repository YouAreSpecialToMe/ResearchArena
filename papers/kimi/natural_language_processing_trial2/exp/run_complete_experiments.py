#!/usr/bin/env python3
"""
Complete CDHR Experiments - Real Model Inference
Runs all baselines and CDHR variants with actual model inference.
"""
import os
import sys
import json
import time
import re
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Add exp/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exp'))

from shared.fixed_model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text


# ============== Configuration ==============
SEEDS = [42, 123, 456]
MODELS = ["llama-3.1-8b"]  # Primary model for all experiments
ALL_MODELS = ["llama-3.1-8b", "qwen2.5-7b", "deepseek-r1-7b"]

DATASETS = {
    "gsm8k": "data/gsm8k.json",
    "math": "data/math.json",
    "gpqa": "data/gpqa.json",
    "aime": "data/aime.json",
}

# ============== Prompt Templates ==============
COT_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""

SELF_CONSISTENCY_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""

REFINEMENT_PROMPT = """Let's solve this problem step by step.

Problem: {question}

Solution:"""

REFINEMENT_RETRY_PROMPT = """I need to reconsider my approach to this problem.

Problem: {question}

Let me think more carefully and solve it correctly:

Solution:"""

LINEAR_PROMPT = """Let's solve this step by step.

Problem: {question}

Current progress: {context}

Continue solving:"""

ANALOGICAL_PROMPT = """I'm stuck on this problem. Let me think of a similar, simpler problem and adapt that solution.

Problem: {question}

Current progress: {context}

Similar problem approach: Solve a simpler version first, then apply the same method.

Let me apply this approach:"""

DECOMPOSITION_PROMPT = """This problem is complex. Let me break it down into smaller, manageable parts.

Problem: {question}

Current progress: {context}

Let me identify the key components and solve each part:
1. First, let's identify what we need to find
2. Then, identify the given information
3. Finally, work through step by step

Solution:"""

VERIFICATION_PROMPT = """I need to verify my reasoning so far. Let me check my previous steps carefully.

Problem: {question}

My previous work: {context}

Let me verify each step:
- Check calculations
- Verify assumptions
- Look for errors

Corrected solution:"""


# ============== Utility Functions ==============
def compute_token_confidence(logprobs: List[float]) -> float:
    """Compute confidence from token logprobs."""
    if not logprobs:
        return 0.5
    # Convert logprobs to probabilities and average
    probs = [np.exp(lp) if lp < 0 else lp for lp in logprobs]
    return float(np.mean(probs))


def compute_self_consistency_confidence(
    model: LLMWrapper, prompt: str, k: int = 5, temperature: float = 0.7
) -> float:
    """Compute self-consistency confidence by sampling k continuations."""
    samples = []
    for _ in range(k):
        response = model.generate(prompt, temperature=temperature, max_tokens=256)
        # Extract answer
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            samples.append(numbers[-1])
    
    if not samples:
        return 0.5
    
    # Compute agreement frequency for most common answer
    counter = Counter(samples)
    most_common = counter.most_common(1)[0]
    return most_common[1] / len(samples)


def compute_composite_confidence(
    token_confidence: float,
    consistency_confidence: float,
    beta: float = 0.5
) -> float:
    """Compute composite confidence score."""
    return beta * token_confidence + (1 - beta) * consistency_confidence


# ============== Baseline: Standard CoT ==============
def run_cot_baseline(
    model: LLMWrapper,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    max_tokens: int = 2048,
    limit: int = None,
):
    """Run standard CoT baseline with real inference."""
    print(f"\n{'='*60}")
    print(f"Baseline CoT - Seed: {seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        problems = json.load(f)
    
    if limit:
        problems = problems[:limit]
    
    print(f"Running on {len(problems)} problems...")
    
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        prompt = COT_PROMPT.format(question=question)
        
        gen_start = time.time()
        response = model.generate(prompt, temperature=0.0, max_tokens=max_tokens)
        gen_time = time.time() - gen_start
        
        pred_answer = extract_answer_from_text(response, dataset_type)
        pred_normalized = normalize_answer(pred_answer)
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_normalized == gold_normalized
        
        if is_correct:
            correct += 1
        
        # Estimate tokens
        num_tokens = len(model.tokenizer.encode(response))
        total_tokens += num_tokens
        
        results.append({
            'id': problem['id'],
            'correct': is_correct,
            'tokens': num_tokens,
            'latency': gen_time,
        })
        
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(problems)}] Accuracy: {acc:.3f}")
    
    total_time = time.time() - start_time
    
    metrics = {
        'accuracy': correct / len(results) if results else 0,
        'correct': correct,
        'total': len(results),
        'avg_tokens': total_tokens / len(results) if results else 0,
        'avg_latency': total_time / len(results) if results else 0,
        'total_time': total_time,
    }
    
    print(f"\nResults: Accuracy={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Time={total_time:.1f}s")
    
    output = {
        'experiment': f'baseline_cot_seed{seed}',
        'seed': seed,
        'metrics': metrics,
        'results': results,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============== Baseline: Self-Consistency-16 ==============
def run_self_consistency_baseline(
    model: LLMWrapper,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    num_samples: int = 16,
    max_tokens: int = 2048,
    limit: int = None,
):
    """Run Self-Consistency with N samples."""
    print(f"\n{'='*60}")
    print(f"Baseline Self-Consistency-{num_samples} - Seed: {seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
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
        
        prompt = SELF_CONSISTENCY_PROMPT.format(question=question)
        
        gen_start = time.time()
        
        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            response = model.generate(prompt, temperature=0.7, max_tokens=max_tokens)
            pred_answer = extract_answer_from_text(response, dataset_type)
            samples.append(pred_answer)
        
        gen_time = time.time() - gen_start
        
        # Majority voting
        valid_samples = [s for s in samples if s is not None]
        if valid_samples:
            counter = Counter([normalize_answer(s) for s in valid_samples])
            pred_answer = counter.most_common(1)[0][0]
        else:
            pred_answer = None
        
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_answer == gold_normalized
        
        if is_correct:
            correct += 1
        
        # Tokens for all samples
        num_tokens = sum(len(model.tokenizer.encode(s)) for s in samples if s)
        total_tokens += num_tokens
        
        results.append({
            'id': problem['id'],
            'correct': is_correct,
            'tokens': num_tokens,
            'latency': gen_time,
        })
        
        if (i + 1) % 5 == 0 or i == len(problems) - 1:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(problems)}] Accuracy: {acc:.3f}")
    
    total_time = time.time() - start_time
    
    metrics = {
        'accuracy': correct / len(results) if results else 0,
        'correct': correct,
        'total': len(results),
        'avg_tokens': total_tokens / len(results) if results else 0,
        'avg_latency': total_time / len(results) if results else 0,
        'total_time': total_time,
    }
    
    print(f"\nResults: Accuracy={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Time={total_time:.1f}s")
    
    output = {
        'experiment': f'baseline_sc{num_samples}_seed{seed}',
        'seed': seed,
        'num_samples': num_samples,
        'metrics': metrics,
        'results': results,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============== Baseline: Iterative Refinement (CoRefine-style) ==============
def run_refinement_baseline(
    model: LLMWrapper,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    confidence_threshold: float = 0.5,
    max_iterations: int = 3,
    max_tokens: int = 2048,
    limit: int = None,
):
    """Run iterative refinement baseline."""
    print(f"\n{'='*60}")
    print(f"Baseline Refinement - Seed: {seed}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    with open(dataset_path, 'r') as f:
        problems = json.load(f)
    
    if limit:
        problems = problems[:limit]
    
    print(f"Running on {len(problems)} problems...")
    
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        gen_start = time.time()
        
        # Iterative refinement
        final_response = None
        num_iterations = 0
        
        for iteration in range(max_iterations):
            num_iterations += 1
            
            if iteration == 0:
                prompt = REFINEMENT_PROMPT.format(question=question)
            else:
                prompt = REFINEMENT_RETRY_PROMPT.format(question=question)
            
            response, logprobs = model.generate_with_logprobs(prompt, temperature=0.0, max_tokens=max_tokens)
            confidence = compute_token_confidence(logprobs)
            
            final_response = response
            
            # Stop if confidence is high enough
            if confidence >= confidence_threshold:
                break
        
        gen_time = time.time() - gen_start
        
        pred_answer = extract_answer_from_text(final_response, dataset_type)
        pred_normalized = normalize_answer(pred_answer)
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_normalized == gold_normalized
        
        if is_correct:
            correct += 1
        
        num_tokens = len(model.tokenizer.encode(final_response)) * num_iterations
        total_tokens += num_tokens
        
        results.append({
            'id': problem['id'],
            'correct': is_correct,
            'tokens': num_tokens,
            'latency': gen_time,
            'iterations': num_iterations,
        })
        
        if (i + 1) % 10 == 0 or i == len(problems) - 1:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(problems)}] Accuracy: {acc:.3f}")
    
    total_time = time.time() - start_time
    
    metrics = {
        'accuracy': correct / len(results) if results else 0,
        'correct': correct,
        'total': len(results),
        'avg_tokens': total_tokens / len(results) if results else 0,
        'avg_latency': total_time / len(results) if results else 0,
        'total_time': total_time,
    }
    
    print(f"\nResults: Accuracy={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Time={total_time:.1f}s")
    
    output = {
        'experiment': f'baseline_refinement_seed{seed}',
        'seed': seed,
        'metrics': metrics,
        'results': results,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============== CDHR: Confidence-Dynamic Heterogeneous Reasoning ==============
class CDHREngine:
    """CDHR engine with real confidence estimation."""
    
    def __init__(
        self,
        model: LLMWrapper,
        theta_v: float = 0.05,
        theta_sigma: float = 0.1,
        beta: float = 0.5,
        window_size: int = 3,
        max_switches: int = 5,
        max_steps: int = 10,
        use_consistency: bool = True,
    ):
        self.model = model
        self.theta_v = theta_v
        self.theta_sigma = theta_sigma
        self.beta = beta
        self.window_size = window_size
        self.max_switches = max_switches
        self.max_steps = max_steps
        self.use_consistency = use_consistency
        
        self.confidence_history = []
        self.strategy_history = []
        self.step_count = 0
        self.switch_count = 0
        self.current_strategy = "linear"
        self.reasoning_trace = []
    
    def estimate_step_confidence(self, text: str, logprobs: List[float], prompt: str) -> float:
        """Estimate confidence for a reasoning step."""
        # Token-level confidence
        token_conf = compute_token_confidence(logprobs)
        
        # Self-consistency confidence (if enabled)
        if self.use_consistency and self.beta < 1.0:
            consistency_conf = compute_self_consistency_confidence(
                self.model, prompt, k=3, temperature=0.5
            )
        else:
            consistency_conf = token_conf
        
        # Composite confidence
        return compute_composite_confidence(token_conf, consistency_conf, self.beta)
    
    def compute_dynamics(self) -> Tuple[float, float, str]:
        """Compute confidence dynamics."""
        if len(self.confidence_history) < self.window_size:
            return 0.0, 0.0, "progressing"
        
        recent = self.confidence_history[-self.window_size:]
        
        # Velocity
        velocity = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
        
        # Variance
        variance = float(np.var(recent))
        
        # Classify trajectory
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
        strategy_map = {
            "progressing": "linear",
            "stagnant": "analogical",
            "oscillating": "decomposition",
            "declining": "verification",
        }
        return strategy_map.get(trajectory, "linear")
    
    def build_prompt(self, question: str, strategy: str = None) -> str:
        """Build prompt for current strategy."""
        strategy = strategy or self.current_strategy
        
        # Summarize reasoning context
        context = ""
        if self.reasoning_trace:
            recent = self.reasoning_trace[-2:]
            context = " ".join([f"Step: {step[:150]}..." for step in recent])
        
        prompts = {
            "linear": LINEAR_PROMPT.format(question=question, context=context),
            "analogical": ANALOGICAL_PROMPT.format(question=question, context=context),
            "decomposition": DECOMPOSITION_PROMPT.format(question=question, context=context),
            "verification": VERIFICATION_PROMPT.format(question=question, context=context),
        }
        
        return prompts.get(strategy, prompts["linear"])
    
    def solve(self, question: str) -> Dict:
        """Solve problem using CDHR."""
        self.confidence_history = []
        self.strategy_history = []
        self.step_count = 0
        self.switch_count = 0
        self.current_strategy = "linear"
        self.reasoning_trace = []
        
        full_reasoning = []
        total_tokens = 0
        
        for step in range(self.max_steps):
            self.step_count += 1
            
            # Build prompt
            prompt = self.build_prompt(question)
            
            # Generate with logprobs
            response, logprobs = self.model.generate_with_logprobs(
                prompt, temperature=0.0, max_tokens=512
            )
            
            # Estimate confidence
            confidence = self.estimate_step_confidence(response, logprobs, prompt)
            self.confidence_history.append(confidence)
            self.strategy_history.append(self.current_strategy)
            self.reasoning_trace.append(response)
            full_reasoning.append(f"[{self.current_strategy}] {response}")
            
            # Count tokens
            total_tokens += len(self.model.tokenizer.encode(response))
            
            # Compute dynamics and switch if needed
            velocity, variance, trajectory = self.compute_dynamics()
            new_strategy = self.select_strategy(trajectory)
            
            if new_strategy != self.current_strategy and self.switch_count < self.max_switches:
                self.current_strategy = new_strategy
                self.switch_count += 1
            
            # Check completion
            if self.is_complete(response):
                break
        
        # Extract answer
        all_text = " ".join(self.reasoning_trace)
        pred_answer = self._extract_answer(all_text)
        
        # Strategy distribution
        strategy_dist = Counter(self.strategy_history)
        
        return {
            'answer': pred_answer,
            'full_reasoning': '\n'.join(full_reasoning),
            'steps': self.step_count,
            'strategy_switches': self.switch_count,
            'strategy_distribution': dict(strategy_dist),
            'confidence_trajectory': self.confidence_history,
            'total_tokens': total_tokens,
        }
    
    def is_complete(self, text: str) -> bool:
        """Check if reasoning appears complete."""
        completion_markers = [
            "the answer is", "therefore", "in conclusion", "####",
            "final answer", "so the answer", "thus the answer"
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in completion_markers)
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer."""
        patterns = [
            r'####\s*(-?\d+(?:\.\d+)?)',
            r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)',
            r'(?:therefore|thus|so)\s*[^.]*?(-?\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else None


def run_cdhr_experiment(
    model: LLMWrapper,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    theta_v: float = 0.05,
    theta_sigma: float = 0.1,
    beta: float = 0.5,
    use_consistency: bool = True,
    limit: int = None,
):
    """Run CDHR experiment."""
    print(f"\n{'='*60}")
    print(f"CDHR Experiment - Seed: {seed}, θv={theta_v}, θσ={theta_sigma}, β={beta}")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    with open(dataset_path, 'r') as f:
        problems = json.load(f)
    
    if limit:
        problems = problems[:limit]
    
    print(f"Running on {len(problems)} problems...")
    
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        gen_start = time.time()
        
        # Run CDHR
        cdhr = CDHREngine(
            model,
            theta_v=theta_v,
            theta_sigma=theta_sigma,
            beta=beta,
            use_consistency=use_consistency,
        )
        output = cdhr.solve(question)
        
        gen_time = time.time() - gen_start
        
        # Evaluate
        pred_answer = output['answer']
        pred_normalized = normalize_answer(pred_answer)
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_normalized == gold_normalized
        
        if is_correct:
            correct += 1
        
        total_tokens += output['total_tokens']
        
        results.append({
            'id': problem['id'],
            'correct': is_correct,
            'tokens': output['total_tokens'],
            'latency': gen_time,
            'steps': output['steps'],
            'switches': output['strategy_switches'],
            'strategies': output['strategy_distribution'],
        })
        
        if (i + 1) % 5 == 0 or i == len(problems) - 1:
            acc = correct / (i + 1)
            print(f"  [{i+1}/{len(problems)}] Accuracy: {acc:.3f}")
    
    total_time = time.time() - start_time
    
    # Strategy distribution across all problems
    all_strategies = []
    for r in results:
        all_strategies.extend([s for s, count in r['strategies'].items() for _ in range(count)])
    strategy_counts = Counter(all_strategies)
    
    # Compute strategy entropy
    total_strategy_usage = sum(strategy_counts.values())
    if total_strategy_usage > 0:
        probabilities = [c / total_strategy_usage for c in strategy_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    else:
        entropy = 0.0
    
    metrics = {
        'accuracy': correct / len(results) if results else 0,
        'correct': correct,
        'total': len(results),
        'avg_tokens': total_tokens / len(results) if results else 0,
        'avg_latency': total_time / len(results) if results else 0,
        'total_time': total_time,
        'strategy_distribution': dict(strategy_counts),
        'strategy_entropy': entropy,
    }
    
    print(f"\nResults: Accuracy={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}, Entropy={entropy:.3f}")
    
    output = {
        'experiment': f'cdhr_seed{seed}',
        'seed': seed,
        'parameters': {'theta_v': theta_v, 'theta_sigma': theta_sigma, 'beta': beta},
        'metrics': metrics,
        'results': results,
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    return output


# ============== Main Experiment Runner ==============
def run_all_experiments(model_name: str = "llama-3.1-8b", backend: str = "transformers"):
    """Run complete experiment suite."""
    print("="*70)
    print(f"CDHR Complete Experiments - Model: {model_name}")
    print("="*70)
    
    # Load model once
    print(f"\nLoading model: {model_name}")
    backend_obj = load_model(model_name, backend=backend)
    model = LLMWrapper(backend_obj, model_name)
    
    results_summary = {}
    
    # Define experiments to run
    experiments = []
    
    # 1. Baseline CoT - 3 seeds
    for seed in SEEDS:
        experiments.append({
            'name': f'baseline_cot_seed{seed}',
            'func': run_cot_baseline,
            'kwargs': {'seed': seed, 'limit': 300},  # Use subset for speed
            'dataset': 'data/gsm8k.json',
        })
    
    # 2. CDHR Main - 3 seeds
    for seed in SEEDS:
        experiments.append({
            'name': f'cdhr_main_seed{seed}',
            'func': run_cdhr_experiment,
            'kwargs': {'seed': seed, 'theta_v': 0.05, 'theta_sigma': 0.1, 'beta': 0.5, 'limit': 300},
            'dataset': 'data/gsm8k.json',
        })
    
    # 3. Self-Consistency-16
    experiments.append({
        'name': 'baseline_sc16',
        'func': run_self_consistency_baseline,
        'kwargs': {'seed': 42, 'num_samples': 16, 'limit': 100},
        'dataset': 'data/gsm8k.json',
    })
    
    # 4. Refinement baseline
    experiments.append({
        'name': 'baseline_refinement',
        'func': run_refinement_baseline,
        'kwargs': {'seed': 42, 'limit': 100},
        'dataset': 'data/gsm8k.json',
    })
    
    # 5. Ablation: Token-only confidence
    experiments.append({
        'name': 'ablation_token_only',
        'func': run_cdhr_experiment,
        'kwargs': {'seed': 42, 'theta_v': 0.05, 'theta_sigma': 0.1, 'beta': 1.0, 'use_consistency': False, 'limit': 100},
        'dataset': 'data/gsm8k.json',
    })
    
    # 6. Ablation: Consistency-only
    experiments.append({
        'name': 'ablation_consistency_only',
        'func': run_cdhr_experiment,
        'kwargs': {'seed': 42, 'theta_v': 0.05, 'theta_sigma': 0.1, 'beta': 0.0, 'limit': 100},
        'dataset': 'data/gsm8k.json',
    })
    
    # 7. Ablation: Different beta values
    for beta in [0.25, 0.75]:
        experiments.append({
            'name': f'ablation_beta{beta}',
            'func': run_cdhr_experiment,
            'kwargs': {'seed': 42, 'theta_v': 0.05, 'theta_sigma': 0.1, 'beta': beta, 'limit': 100},
            'dataset': 'data/gsm8k.json',
        })
    
    # 8. Ablation: Different thresholds
    for theta_v in [0.03, 0.07]:
        for theta_sigma in [0.075, 0.125]:
            experiments.append({
                'name': f'ablation_th{theta_v}_{theta_sigma}',
                'func': run_cdhr_experiment,
                'kwargs': {'seed': 42, 'theta_v': theta_v, 'theta_sigma': theta_sigma, 'beta': 0.5, 'limit': 50},
                'dataset': 'data/gsm8k.json',
            })
    
    # Run all experiments
    total_start = time.time()
    for i, exp in enumerate(experiments):
        print(f"\n{'='*70}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp['name']}")
        print(f"{'='*70}")
        
        try:
            output_path = f"results/{exp['name']}.json"
            result = exp['func'](
                model=model,
                dataset_path=exp['dataset'],
                output_path=output_path,
                **exp['kwargs']
            )
            results_summary[exp['name']] = result['metrics']
        except Exception as e:
            print(f"ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[exp['name']] = {'error': str(e)}
    
    total_time = time.time() - total_start
    
    # Save summary
    print(f"\n{'='*70}")
    print(f"All Experiments Complete! Total time: {total_time/3600:.2f} hours")
    print(f"{'='*70}")
    
    with open('results/summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nResults Summary:")
    for name, metrics in results_summary.items():
        if 'error' not in metrics:
            print(f"  {name}: Acc={metrics['accuracy']:.4f}, Tokens={metrics['avg_tokens']:.1f}")
        else:
            print(f"  {name}: ERROR - {metrics['error']}")
    
    return results_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama-3.1-8b')
    parser.add_argument('--backend', type=str, default='transformers', choices=['transformers', 'vllm'])
    args = parser.parse_args()
    
    run_all_experiments(model_name=args.model, backend=args.backend)
