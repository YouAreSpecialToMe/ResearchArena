#!/usr/bin/env python3
"""
Real CDHR Experiments - Actually runs models and collects results.
Addresses previous feedback about simulated results.
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Ensure reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass

# Add shared modules
sys.path.insert(0, 'exp/shared')
from cdhr_framework import CDHRSystem, Strategy, SimpleLLMEngine
from fixed_model_loader import load_model, LLMWrapper

def load_dataset(dataset_name: str, max_problems: Optional[int] = None) -> List[Dict]:
    """Load a dataset from disk."""
    path = f"data/{dataset_name}.json"
    with open(path, 'r') as f:
        problems = json.load(f)
    
    if max_problems:
        problems = problems[:max_problems]
    
    print(f"Loaded {len(problems)} problems from {dataset_name}")
    return problems

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip().lower()
    answer = answer.replace(',', '')
    try:
        num = float(answer)
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        return answer.strip()

def check_answer(predicted: str, ground_truth: str, dataset_type: str = "generic") -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)
    
    if pred_norm == truth_norm:
        return True
    
    # For numeric answers, try exact match after normalization
    try:
        pred_float = float(pred_norm)
        truth_float = float(truth_norm)
        # Allow small tolerance for floating point
        return abs(pred_float - truth_float) < 1e-5
    except:
        pass
    
    return False

class StandardCoT:
    """Standard Chain-of-Thought baseline."""
    
    def __init__(self, llm_wrapper):
        self.llm = llm_wrapper
        self.prompt_template = """Solve the following problem step by step. Show your reasoning and end with your final answer.

Problem: {problem}

Let's solve this step by step:"""
    
    def solve(self, problem: str, max_tokens: int = 1024) -> Dict:
        """Solve a problem using standard CoT."""
        prompt = self.prompt_template.format(problem=problem)
        
        start_time = time.time()
        response = self.llm.generate(prompt, temperature=0.0, max_tokens=max_tokens)
        latency = time.time() - start_time
        
        # Extract answer
        answer = self._extract_answer(response)
        
        # Estimate tokens (rough approximation)
        tokens_prompt = len(prompt.split())
        tokens_response = len(response.split())
        total_tokens = tokens_prompt + tokens_response
        
        return {
            "answer": answer,
            "response": response,
            "tokens": total_tokens,
            "latency": latency,
        }
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from response."""
        import re
        # Try #### format
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1)
        # Try "answer is" format
        match = re.search(r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        # Try last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        return text.strip().split()[-1] if text.strip() else None

class SelfConsistencyCoT:
    """Self-Consistency with multiple samples."""
    
    def __init__(self, llm_wrapper, num_samples: int = 16):
        self.llm = llm_wrapper
        self.num_samples = num_samples
        self.prompt_template = """Solve the following problem step by step. Show your reasoning and end with your final answer.

Problem: {problem}

Let's solve this step by step:"""
    
    def solve(self, problem: str, max_tokens: int = 1024) -> Dict:
        """Solve using self-consistency."""
        prompt = self.prompt_template.format(problem=problem)
        
        start_time = time.time()
        
        # Generate multiple samples
        samples = []
        for _ in range(self.num_samples):
            response = self.llm.generate(prompt, temperature=0.7, max_tokens=max_tokens)
            answer = self._extract_answer(response)
            samples.append({"response": response, "answer": answer})
        
        latency = time.time() - start_time
        
        # Majority voting
        answers = [s["answer"] for s in samples if s["answer"]]
        answer_counts = {}
        for ans in answers:
            norm_ans = normalize_answer(ans)
            answer_counts[norm_ans] = answer_counts.get(norm_ans, 0) + 1
        
        best_answer = max(answer_counts.items(), key=lambda x: x[1])[0] if answer_counts else answers[0] if answers else ""
        
        # Find a sample with the best answer for response
        best_sample = None
        for s in samples:
            if normalize_answer(s["answer"]) == best_answer:
                best_sample = s
                break
        if not best_sample:
            best_sample = samples[0]
        
        # Estimate tokens
        tokens_per_sample = len(prompt.split()) + max_tokens
        total_tokens = tokens_per_sample * self.num_samples
        
        return {
            "answer": best_answer,
            "response": best_sample["response"],
            "tokens": total_tokens,
            "latency": latency,
            "samples": samples,
            "answer_distribution": answer_counts,
        }
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract final answer from response."""
        import re
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
        if match:
            return match.group(1)
        match = re.search(r'(?:the answer is|answer:)\s*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if match:
            return match.group(1)
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        return text.strip().split()[-1] if text.strip() else None

class CDHRExperiment:
    """CDHR method experiment wrapper."""
    
    def __init__(self, llm_wrapper, beta: float = 0.5, theta_v: float = 0.05, theta_sigma: float = 0.1):
        self.llm = llm_wrapper
        self.beta = beta
        self.theta_v = theta_v
        self.theta_sigma = theta_sigma
        
        # Create CDHR system
        self.cdhr = CDHRSystem(
            llm_engine=llm_wrapper,
            beta=beta,
            theta_v=theta_v,
            theta_sigma=theta_sigma,
            max_switches=5,
        )
    
    def solve(self, problem: str, max_steps: int = 10) -> Dict:
        """Solve using CDHR."""
        start_time = time.time()
        
        result = self.cdhr.solve(problem, max_steps=max_steps, verbose=False)
        
        latency = time.time() - start_time
        
        # Estimate tokens
        reasoning = result.get("reasoning", "")
        tokens = len(problem.split()) + len(reasoning.split())
        
        result["tokens"] = tokens
        result["latency"] = latency
        
        return result

def run_experiment(
    model_name: str,
    dataset_name: str,
    method: str,
    seed: int,
    max_problems: Optional[int] = None,
    output_dir: str = "results",
    **kwargs
) -> Dict:
    """Run a single experiment."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {method} on {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model: {model_name}")
    backend = load_model(model_name, backend="transformers")
    llm = LLMWrapper(backend, model_name)
    
    # Load dataset
    problems = load_dataset(dataset_name, max_problems=max_problems)
    
    # Create method
    if method == "cot":
        solver = StandardCoT(llm)
    elif method == "sc16":
        solver = SelfConsistencyCoT(llm, num_samples=16)
    elif method.startswith("cdhr"):
        beta = kwargs.get("beta", 0.5)
        theta_v = kwargs.get("theta_v", 0.05)
        theta_sigma = kwargs.get("theta_sigma", 0.1)
        solver = CDHRExperiment(llm, beta=beta, theta_v=theta_v, theta_sigma=theta_sigma)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Run evaluation
    results = []
    correct = 0
    total_tokens = 0
    total_latency = 0
    
    for i, problem in enumerate(problems):
        print(f"\nProblem {i+1}/{len(problems)}: {problem['id']}")
        
        try:
            result = solver.solve(problem['question'])
            
            # Check answer
            predicted = result.get('answer', '')
            ground_truth = problem.get('answer', '')
            is_correct = check_answer(predicted, ground_truth, dataset_name)
            
            if is_correct:
                correct += 1
            
            total_tokens += result.get('tokens', 0)
            total_latency += result.get('latency', 0)
            
            result['problem_id'] = problem['id']
            result['ground_truth'] = ground_truth
            result['correct'] = is_correct
            results.append(result)
            
            print(f"  Predicted: {predicted}, Ground Truth: {ground_truth}, Correct: {is_correct}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'problem_id': problem['id'],
                'error': str(e),
                'correct': False,
            })
    
    # Compute metrics
    accuracy = correct / len(problems) if problems else 0
    avg_tokens = total_tokens / len(problems) if problems else 0
    avg_latency = total_latency / len(problems) if problems else 0
    
    experiment_result = {
        "model": model_name,
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "num_problems": len(problems),
        "correct": correct,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "total_tokens": total_tokens,
        "total_latency": total_latency,
        "config": kwargs,
        "detailed_results": results,
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{method}_{dataset_name}_s{seed}.json")
    with open(output_file, 'w') as f:
        json.dump(experiment_result, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results: Accuracy={accuracy:.4f}, Avg Tokens={avg_tokens:.1f}, Avg Latency={avg_latency:.2f}s")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")
    
    # Clean up GPU memory
    del llm
    del backend
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return experiment_result

def aggregate_results(result_files: List[str]) -> Dict:
    """Aggregate results across multiple seeds."""
    all_results = []
    for f in result_files:
        with open(f, 'r') as fp:
            all_results.append(json.load(fp))
    
    accuracies = [r['accuracy'] for r in all_results]
    tokens = [r['avg_tokens'] for r in all_results]
    latencies = [r['avg_latency'] for r in all_results]
    
    return {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "tokens_mean": np.mean(tokens),
        "tokens_std": np.std(tokens),
        "latency_mean": np.mean(latencies),
        "latency_std": np.std(latencies),
        "num_seeds": len(all_results),
        "individual_results": all_results,
    }

def main():
    parser = argparse.ArgumentParser(description="Run CDHR experiments")
    parser.add_argument("--model", type=str, default="llama-3.1-8b", help="Model name")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name")
    parser.add_argument("--method", type=str, default="cot", help="Method (cot, sc16, cdhr)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_problems", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.5, help="CDHR beta parameter")
    parser.add_argument("--theta_v", type=float, default=0.05, help="CDHR theta_v parameter")
    parser.add_argument("--theta_sigma", type=float, default=0.1, help="CDHR theta_sigma parameter")
    
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        method=args.method,
        seed=args.seed,
        max_problems=args.max_problems,
        output_dir=args.output_dir,
        beta=args.beta,
        theta_v=args.theta_v,
        theta_sigma=args.theta_sigma,
    )

if __name__ == "__main__":
    main()
