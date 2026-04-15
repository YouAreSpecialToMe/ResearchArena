#!/usr/bin/env python3
"""
CDHR Main Experiment: Confidence-Dynamic Heterogeneous Reasoning
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import re
import argparse
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
from shared.model_loader import load_model, LLMWrapper
from shared.data_loader import normalize_answer, extract_answer_from_text


class Strategy(Enum):
    LINEAR = "linear"
    ANALOGICAL = "analogical"
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"


STRATEGY_PROMPTS = {
    Strategy.LINEAR: "Let's solve this step by step.",
    Strategy.ANALOGICAL: "I'm stuck. Let me think of a similar problem and adapt its solution.",
    Strategy.DECOMPOSITION: "This is complex. Let me break it down into smaller parts.",
    Strategy.VERIFICATION: "My confidence is dropping. Let me verify my previous steps.",
}


class CDHREngine:
    """Simplified CDHR engine for experiments."""
    
    def __init__(
        self,
        model_wrapper: LLMWrapper,
        theta_v: float = 0.05,
        theta_sigma: float = 0.1,
        beta: float = 0.5,
        window_size: int = 3,
        max_switches: int = 5,
        max_steps: int = 10,
    ):
        self.model = model_wrapper
        self.theta_v = theta_v
        self.theta_sigma = theta_sigma
        self.beta = beta
        self.window_size = window_size
        self.max_switches = max_switches
        self.max_steps = max_steps
        
        self.confidence_history = []
        self.strategy_history = []
        self.step_count = 0
        self.switch_count = 0
        self.current_strategy = Strategy.LINEAR
        self.reasoning_trace = []
    
    def estimate_confidence(self, text: str) -> float:
        """Estimate confidence from response text."""
        # Use heuristics to estimate confidence
        confidence = 0.5  # Base confidence
        
        # Longer reasoning suggests more engagement
        words = len(text.split())
        if words > 20:
            confidence += 0.1
        if words > 50:
            confidence += 0.1
        
        # Presence of mathematical operations
        if re.search(r'[=+\-*/]', text):
            confidence += 0.1
        
        # Presence of definitive statements
        if any(word in text.lower() for word in ['therefore', 'thus', 'so', 'hence']):
            confidence += 0.1
        
        # Presence of uncertainty markers
        if any(word in text.lower() for word in ['maybe', 'perhaps', 'unclear', 'not sure']):
            confidence -= 0.15
        
        # Numerical results present
        if re.search(r'\d+', text):
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def compute_dynamics(self) -> Tuple[float, float, str]:
        """Compute confidence dynamics."""
        if len(self.confidence_history) < self.window_size:
            return 0.0, 0.0, "unknown"
        
        recent = self.confidence_history[-self.window_size:]
        
        # Velocity (rate of change)
        velocity = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
        
        # Variance (stability)
        variance = np.var(recent)
        
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
    
    def select_strategy(self, trajectory: str) -> Strategy:
        """Select strategy based on trajectory type."""
        strategy_map = {
            "progressing": Strategy.LINEAR,
            "stagnant": Strategy.ANALOGICAL,
            "oscillating": Strategy.DECOMPOSITION,
            "declining": Strategy.VERIFICATION,
        }
        return strategy_map.get(trajectory, Strategy.LINEAR)
    
    def build_prompt(self, question: str) -> str:
        """Build prompt for current strategy."""
        base_prompt = STRATEGY_PROMPTS[self.current_strategy]
        
        # Add reasoning history
        context = ""
        if self.reasoning_trace:
            # Summarize last few steps
            recent = self.reasoning_trace[-3:]
            context = "\n\nPrevious reasoning:\n" + "\n".join([f"Step {i+1}: {step[:100]}..." for i, step in enumerate(recent)])
        
        # Add dynamics info for verification strategy
        if self.current_strategy == Strategy.VERIFICATION:
            context += "\n\nPlease carefully check my previous calculations and assumptions."
        elif self.current_strategy == Strategy.DECOMPOSITION:
            context += "\n\nLet me identify the key components and solve each part separately."
        elif self.current_strategy == Strategy.ANALOGICAL:
            context += "\n\nLet me think of a simpler, related problem and how I solved it."
        
        full_prompt = f"{base_prompt}{context}\n\nProblem: {question}\n\n"
        return full_prompt
    
    def is_complete(self, text: str) -> bool:
        """Check if reasoning appears complete."""
        completion_markers = [
            "the answer is", "therefore", "in conclusion", "####",
            "final answer", "so the answer", "thus the answer"
        ]
        text_lower = text.lower()
        return any(marker in text_lower for marker in completion_markers)
    
    def solve(self, question: str) -> Dict:
        """Solve problem using CDHR."""
        self.confidence_history = []
        self.strategy_history = []
        self.step_count = 0
        self.switch_count = 0
        self.current_strategy = Strategy.LINEAR
        self.reasoning_trace = []
        
        full_reasoning = []
        
        for step in range(self.max_steps):
            self.step_count += 1
            
            # Build prompt
            prompt = self.build_prompt(question)
            
            # Generate
            response = self.model.generate(
                prompt,
                temperature=0.0,
                max_tokens=512
            )
            
            # Estimate confidence
            confidence = self.estimate_confidence(response)
            self.confidence_history.append(confidence)
            self.strategy_history.append(self.current_strategy.value)
            self.reasoning_trace.append(response)
            full_reasoning.append(f"[{self.current_strategy.value}] {response}")
            
            # Compute dynamics and potentially switch
            velocity, variance, trajectory = self.compute_dynamics()
            new_strategy = self.select_strategy(trajectory)
            
            if new_strategy != self.current_strategy and self.switch_count < self.max_switches:
                self.current_strategy = new_strategy
                self.switch_count += 1
            
            # Check completion
            if self.is_complete(response):
                break
        
        # Extract final answer
        all_text = " ".join(self.reasoning_trace)
        pred_answer = self._extract_answer(all_text)
        
        # Strategy distribution
        strategy_dist = {}
        for s in self.strategy_history:
            strategy_dist[s] = strategy_dist.get(s, 0) + 1
        
        return {
            'answer': pred_answer,
            'full_reasoning': '\n'.join(full_reasoning),
            'steps': self.step_count,
            'strategy_switches': self.switch_count,
            'strategy_distribution': strategy_dist,
            'confidence_trajectory': self.confidence_history,
            'final_strategy': self.current_strategy.value,
        }
    
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
    model_name: str,
    dataset_path: str,
    output_path: str,
    seed: int = 42,
    theta_v: float = 0.05,
    theta_sigma: float = 0.1,
    beta: float = 0.5,
    limit: int = None,
):
    """Run CDHR experiment."""
    print(f"=" * 60)
    print(f"CDHR Experiment: {model_name}, Seed: {seed}")
    print(f"Parameters: theta_v={theta_v}, theta_sigma={theta_sigma}, beta={beta}")
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
    
    # Run CDHR
    results = []
    correct = 0
    total_tokens = 0
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        question = problem['question']
        gold_answer = problem['answer']
        dataset_type = problem.get('dataset', 'generic')
        
        # Solve with CDHR
        gen_start = time.time()
        cdhr = CDHREngine(
            model_wrapper,
            theta_v=theta_v,
            theta_sigma=theta_sigma,
            beta=beta
        )
        output = cdhr.solve(question)
        gen_time = time.time() - gen_start
        
        # Extract and compare answer
        pred_answer = output['answer']
        pred_normalized = normalize_answer(pred_answer)
        gold_normalized = normalize_answer(gold_answer)
        is_correct = pred_normalized == gold_normalized
        
        if is_correct:
            correct += 1
        
        # Estimate tokens
        num_tokens = len(output['full_reasoning'].split())
        total_tokens += num_tokens
        
        results.append({
            'id': problem['id'],
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': pred_answer,
            'correct': is_correct,
            'tokens': num_tokens,
            'latency': gen_time,
            'cdhr_output': output,
        })
        
        if (i + 1) % 10 == 0:
            acc = correct / (i + 1)
            print(f"  Processed {i+1}/{len(problems)} | Accuracy: {acc:.3f}")
    
    total_time = time.time() - start_time
    
    # Compute metrics
    accuracy = correct / len(results) if results else 0
    avg_tokens = total_tokens / len(results) if results else 0
    avg_latency = total_time / len(results) if results else 0
    
    # Strategy statistics
    all_distributions = [r['cdhr_output']['strategy_distribution'] for r in results]
    strategy_counts = {}
    for dist in all_distributions:
        for strategy, count in dist.items():
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + count
    
    # Compute strategy entropy
    total_strategy_usage = sum(strategy_counts.values())
    if total_strategy_usage > 0:
        probabilities = [c / total_strategy_usage for c in strategy_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    else:
        entropy = 0.0
    
    print(f"\n{'=' * 60}")
    print(f"Results Summary")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})")
    print(f"Avg Tokens: {avg_tokens:.1f}")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Strategy Distribution: {strategy_counts}")
    print(f"Strategy Entropy: {entropy:.3f} bits")
    
    # Save results
    output = {
        'experiment': f'cdhr_{model_name}_seed{seed}',
        'model': model_name,
        'dataset': dataset_path,
        'seed': seed,
        'parameters': {
            'theta_v': theta_v,
            'theta_sigma': theta_sigma,
            'beta': beta,
        },
        'metrics': {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(results),
            'avg_tokens': avg_tokens,
            'avg_latency': avg_latency,
            'total_time': total_time,
            'strategy_distribution': strategy_counts,
            'strategy_entropy': entropy,
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
    parser.add_argument('--theta_v', type=float, default=0.05)
    parser.add_argument('--theta_sigma', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    if args.output is None:
        dataset_name = os.path.basename(args.dataset).replace('.json', '')
        args.output = f'results/cdhr_main/{args.model}_{dataset_name}_seed{args.seed}.json'
    
    run_cdhr_experiment(
        model_name=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        seed=args.seed,
        theta_v=args.theta_v,
        theta_sigma=args.theta_sigma,
        beta=args.beta,
        limit=args.limit,
    )


if __name__ == '__main__':
    main()
