"""Fast experiment runner for ESR - optimized for time budget."""

import sys
import json
import time
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loader import load_json, create_cot_prompt
from shared.metrics import compare_answers, compute_statistics, extract_answer_from_text
from shared.utils import set_seed


class FastExperimentRunner:
    """Optimized experiment runner with model caching."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model once."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading {self.model_name}...")
        start = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )
        self.model.eval()
        
        print(f"Model loaded in {time.time()-start:.1f}s")
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> tuple:
        """Generate text."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            if temperature > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        new_text = generated_text[len(prompt_text):]
        
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        return new_text, num_tokens
    
    def compute_entropy_varentropy(self, logits: torch.Tensor) -> tuple:
        """Compute entropy and varentropy."""
        import torch.nn.functional as F
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        entropy = -(probs * log_probs).sum(dim=-1)
        varentropy = (probs * (log_probs + entropy.unsqueeze(-1)) ** 2).sum(dim=-1)
        
        return entropy.item(), varentropy.item()
    
    def run_vanilla(self, dataset: List[Dict], seed: int) -> Dict:
        """Run vanilla CoT baseline."""
        set_seed(seed)
        results = []
        
        for i, item in enumerate(dataset):
            prompt = create_cot_prompt(item["question"])
            output, tokens = self.generate(prompt, max_tokens=512)
            
            pred = extract_answer_from_text(output)
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": tokens,
                "output": output[:200]  # Truncate for storage
            })
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "results": results
        }
    
    def run_esr(self, dataset: List[Dict], seed: int, 
                tau_h: float, tau_v: float, r_max: int) -> Dict:
        """Run ESR with entropy-varentropy."""
        import torch.nn.functional as F
        
        set_seed(seed)
        results = []
        
        for i, item in enumerate(dataset):
            prompt = create_cot_prompt(item["question"])
            
            # First pass - track uncertainty
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            generated_ids = inputs["input_ids"]
            uncertainty_readings = []
            
            with torch.no_grad():
                for _ in range(512):
                    outputs = self.model(input_ids=generated_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    # Compute uncertainty
                    h, v = self.compute_entropy_varentropy(next_logits[0])
                    uncertainty_readings.append((h, v))
                    
                    # Greedy next token
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            output1 = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            output1 = output1[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            
            tokens1 = generated_ids.shape[1] - inputs["input_ids"].shape[1]
            
            # Check for high uncertainty
            high_uncertainty = any(h > tau_h and v < tau_v for h, v in uncertainty_readings)
            revision_triggered = high_uncertainty and tokens1 > 50
            
            if revision_triggered:
                # Revision pass
                revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully. I think there might be an issue with my reasoning. Let me work through this again step by step.\n"
                output2, tokens2 = self.generate(revision_prompt, max_tokens=512)
                
                pred = extract_answer_from_text(output2)
                total_tokens = tokens1 + tokens2
                final_output = output2
                revision_count = 1
            else:
                pred = extract_answer_from_text(output1)
                total_tokens = tokens1
                final_output = output1
                revision_count = 0
            
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens,
                "revision_triggered": revision_triggered,
                "revision_count": revision_count,
                "output": final_output[:200]
            })
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        revision_rate = sum(r["revision_triggered"] for r in results) / len(results)
        
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "revision_rate": revision_rate,
            "results": results
        }
    
    def run_entropy_only(self, dataset: List[Dict], seed: int, 
                         tau_h: float, r_max: int) -> Dict:
        """Run entropy-only baseline."""
        set_seed(seed)
        results = []
        
        for i, item in enumerate(dataset):
            prompt = create_cot_prompt(item["question"])
            
            # First pass
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            generated_ids = inputs["input_ids"]
            entropy_readings = []
            
            with torch.no_grad():
                for _ in range(512):
                    outputs = self.model(input_ids=generated_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    # Compute entropy only
                    probs = torch.nn.functional.softmax(next_logits, dim=-1)
                    log_probs = torch.nn.functional.log_softmax(next_logits, dim=-1)
                    h = -(probs * log_probs).sum(dim=-1).item()
                    entropy_readings.append(h)
                    
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            
            output1 = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            output1 = output1[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            
            tokens1 = generated_ids.shape[1] - inputs["input_ids"].shape[1]
            
            # Check for high entropy only
            high_entropy = any(h > tau_h for h in entropy_readings)
            revision_triggered = high_entropy and tokens1 > 50
            
            if revision_triggered:
                revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully.\n"
                output2, tokens2 = self.generate(revision_prompt, max_tokens=512)
                
                pred = extract_answer_from_text(output2)
                total_tokens = tokens1 + tokens2
                revision_count = 1
            else:
                pred = extract_answer_from_text(output1)
                total_tokens = tokens1
                revision_count = 0
            
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens,
                "revision_triggered": revision_triggered,
                "revision_count": revision_count
            })
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        revision_rate = sum(r["revision_triggered"] for r in results) / len(results)
        
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "revision_rate": revision_rate,
            "results": results
        }
    
    def run_egl(self, dataset: List[Dict], seed: int, tau_h: float) -> Dict:
        """Run EGL-style post-hoc refinement."""
        set_seed(seed)
        results = []
        
        for i, item in enumerate(dataset):
            prompt = create_cot_prompt(item["question"])
            
            # First pass
            output1, tokens1 = self.generate(prompt, max_tokens=512)
            
            # Check if refinement needed (simplified: check for uncertainty markers)
            uncertainty_markers = ["maybe", "perhaps", "uncertain", "not sure", "think", "might"]
            needs_refinement = any(m in output1.lower() for m in uncertainty_markers) or len(output1) > 300
            
            if needs_refinement:
                refinement_prompt = f"{prompt}{output1}\n\nLet me review and correct if needed:\n"
                output2, tokens2 = self.generate(refinement_prompt, max_tokens=512)
                
                pred = extract_answer_from_text(output2)
                total_tokens = tokens1 + tokens2
                refined = True
            else:
                pred = extract_answer_from_text(output1)
                total_tokens = tokens1
                refined = False
            
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens,
                "refined": refined
            })
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        refinement_rate = sum(r["refined"] for r in results) / len(results)
        
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "refinement_rate": refinement_rate,
            "results": results
        }
    
    def run_bestofn(self, dataset: List[Dict], seed: int, n: int = 4) -> Dict:
        """Run Best-of-N baseline."""
        set_seed(seed)
        results = []
        
        for i, item in enumerate(dataset):
            prompt = create_cot_prompt(item["question"])
            
            answers = []
            total_tokens = 0
            
            for _ in range(n):
                output, tokens = self.generate(prompt, max_tokens=512, temperature=0.7)
                total_tokens += tokens
                pred = extract_answer_from_text(output)
                answers.append(str(pred))
            
            # Majority voting
            from collections import Counter
            answer_counts = Counter(answers)
            majority_answer = answer_counts.most_common(1)[0][0]
            
            try:
                majority_answer = float(majority_answer)
            except:
                pass
            
            correct = compare_answers(majority_answer, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens
            })
        
        accuracy = sum(r["correct"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        
        return {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "results": results
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="exp/results")
    parser.add_argument("--tau_h", type=float, default=2.5, help="Entropy threshold")
    parser.add_argument("--tau_v", type=float, default=1.5, help="Varentropy threshold")
    parser.add_argument("--single_seed", type=int, default=None, help="Run with single seed (overrides --seeds)")
    args = parser.parse_args()
    
    # Load dataset
    data_path = f"exp/data/{args.dataset}_test.json" if args.dataset == "gsm8k" else f"exp/data/math_500.json"
    dataset = load_json(data_path)
    
    if args.limit:
        dataset = dataset[:args.limit]
    
    print(f"Running experiments on {len(dataset)} examples from {args.dataset}")
    
    # Use single seed if specified
    if args.single_seed is not None:
        args.seeds = [args.single_seed]
        print(f"Seed: {args.single_seed}")
    else:
        print(f"Seeds: {args.seeds}")
    
    print(f"Thresholds: tau_h={args.tau_h}, tau_v={args.tau_v}")
    
    # Initialize runner
    runner = FastExperimentRunner(args.model)
    
    # Thresholds
    tau_h, tau_v = args.tau_h, args.tau_v
    
    all_results = {}
    
    # Run Vanilla CoT
    print("\n" + "="*60)
    print("Running Vanilla CoT...")
    vanilla_results = []
    for seed in args.seeds:
        print(f"  Seed {seed}...")
        result = runner.run_vanilla(dataset, seed)
        vanilla_results.append(result)
        print(f"    Accuracy: {result['accuracy']:.3f}, Tokens: {result['avg_tokens']:.1f}")
    
    accs = [r["accuracy"] for r in vanilla_results]
    toks = [r["avg_tokens"] for r in vanilla_results]
    all_results["vanilla"] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "tokens_mean": float(np.mean(toks)),
        "tokens_std": float(np.std(toks)),
        "raw_results": vanilla_results
    }
    
    # Run ESR
    print("\n" + "="*60)
    print("Running ESR...")
    esr_results = []
    for seed in args.seeds:
        print(f"  Seed {seed}...")
        result = runner.run_esr(dataset, seed, tau_h, tau_v, r_max=3)
        esr_results.append(result)
        print(f"    Accuracy: {result['accuracy']:.3f}, Tokens: {result['avg_tokens']:.1f}, Revision Rate: {result['revision_rate']:.2%}")
    
    accs = [r["accuracy"] for r in esr_results]
    toks = [r["avg_tokens"] for r in esr_results]
    revs = [r["revision_rate"] for r in esr_results]
    all_results["esr"] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "tokens_mean": float(np.mean(toks)),
        "tokens_std": float(np.std(toks)),
        "revision_rate_mean": float(np.mean(revs)),
        "raw_results": esr_results
    }
    
    # Run Entropy-Only
    print("\n" + "="*60)
    print("Running Entropy-Only...")
    entropy_only_results = []
    for seed in args.seeds:
        print(f"  Seed {seed}...")
        result = runner.run_entropy_only(dataset, seed, tau_h, r_max=3)
        entropy_only_results.append(result)
        print(f"    Accuracy: {result['accuracy']:.3f}, Tokens: {result['avg_tokens']:.1f}, Revision Rate: {result['revision_rate']:.2%}")
    
    accs = [r["accuracy"] for r in entropy_only_results]
    toks = [r["avg_tokens"] for r in entropy_only_results]
    all_results["entropy_only"] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "tokens_mean": float(np.mean(toks)),
        "tokens_std": float(np.std(toks)),
        "raw_results": entropy_only_results
    }
    
    # Run EGL
    print("\n" + "="*60)
    print("Running EGL...")
    egl_results = []
    for seed in args.seeds:
        print(f"  Seed {seed}...")
        result = runner.run_egl(dataset, seed, tau_h)
        egl_results.append(result)
        print(f"    Accuracy: {result['accuracy']:.3f}, Tokens: {result['avg_tokens']:.1f}, Refinement Rate: {result['refinement_rate']:.2%}")
    
    accs = [r["accuracy"] for r in egl_results]
    toks = [r["avg_tokens"] for r in egl_results]
    all_results["egl"] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "tokens_mean": float(np.mean(toks)),
        "tokens_std": float(np.std(toks)),
        "raw_results": egl_results
    }
    
    # Run Best-of-N (only first seed due to cost)
    print("\n" + "="*60)
    print("Running Best-of-N...")
    result = runner.run_bestofn(dataset, args.seeds[0], n=4)
    print(f"    Accuracy: {result['accuracy']:.3f}, Tokens: {result['avg_tokens']:.1f}")
    all_results["bestofn"] = {
        "accuracy_mean": result["accuracy"],
        "accuracy_std": 0.0,
        "tokens_mean": result["avg_tokens"],
        "tokens_std": 0.0,
        "raw_results": [result]
    }
    
    # Save results
    if args.single_seed is not None:
        output_path = Path(args.output_dir) / f"all_methods_seed{args.single_seed}_tuned.json"
    else:
        output_path = Path(args.output_dir) / f"all_results_{args.dataset}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add config to results
    output_data = {
        "results": all_results,
        "config": {
            "dataset": args.dataset,
            "limit": args.limit,
            "seeds": args.seeds,
            "tau_h": tau_h,
            "tau_v": tau_v,
            "model": args.model
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Accuracy':<20} {'Tokens':<15} {'Revision %':<12}")
    print("-"*60)
    for method, res in all_results.items():
        acc_str = f"{res['accuracy_mean']:.3f} ± {res['accuracy_std']:.3f}"
        tok_str = f"{res['tokens_mean']:.1f}"
        rev_str = ""
        if "revision_rate_mean" in res:
            rev_str = f"{res['revision_rate_mean']:.1%}"
        print(f"{method:<20} {acc_str:<20} {tok_str:<15} {rev_str:<12}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
