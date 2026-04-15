"""Optimized experiment runner with tuned thresholds."""

import sys
import json
import time
import torch
import os
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from shared.data_loader import load_json, create_cot_prompt
from shared.metrics import compare_answers
from shared.utils import set_seed


def extract_answer(text: str):
    import re
    if not text:
        return None
    if "####" in text:
        match = re.search(r"####\s*([-\d.,]+)", text)
        if match:
            return match.group(1)
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    return text.strip()


def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum().item()


def compute_varentropy(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    return (probs * (log_probs + entropy) ** 2).sum().item()


def generate_with_uncertainty(model, tokenizer, prompt, max_tokens=128):
    """Generate and compute uncertainty metrics."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    generated_ids = inputs["input_ids"]
    uncertainties = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids=generated_ids, return_dict=True)
            next_logits = outputs.logits[:, -1, :]
            
            h = compute_entropy(next_logits[0])
            v = compute_varentropy(next_logits[0])
            uncertainties.append((h, v))
            
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output = output[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokens = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    
    return output, tokens, uncertainties


def generate_vanilla(model, tokenizer, prompt, max_tokens=128):
    """Vanilla CoT generation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return output_text, tokens


def run_experiments(dataset, seed=42, limit=None, tau_h=0.55, tau_v=0.005):
    set_seed(seed)
    
    if limit:
        dataset = dataset[:limit]
    
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Model loaded. Running experiments on {len(dataset)} problems...", flush=True)
    print(f"Thresholds: tau_h={tau_h}, tau_v={tau_v}", flush=True)
    
    results = {
        "vanilla": [],
        "esr": [],
        "entropy_only": [],
        "egl": [],
        "bestofn": []
    }
    
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {i}/{len(dataset)} ({elapsed:.1f}s)", flush=True)
        
        prompt = create_cot_prompt(item["question"])
        
        # Vanilla CoT (fast - use generate)
        output, tokens = generate_vanilla(model, tokenizer, prompt)
        pred = extract_answer(output)
        correct = compare_answers(pred, item["answer"])
        results["vanilla"].append({"correct": correct, "tokens": tokens})
        
        # ESR and Entropy-only (share the same generation with uncertainty)
        output1, tokens1, uncertainties = generate_with_uncertainty(model, tokenizer, prompt)
        
        # ESR check: high entropy + low varentropy
        confused_positions = [(h, v) for h, v in uncertainties if h > tau_h and v < tau_v]
        esr_triggered = len(confused_positions) >= 3 and tokens1 > 30
        
        if esr_triggered:
            revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully.\n"
            output2, tokens2 = generate_vanilla(model, tokenizer, revision_prompt)
            pred = extract_answer(output2)
            correct = compare_answers(pred, item["answer"])
            results["esr"].append({"correct": correct, "tokens": tokens1 + tokens2, "revised": True, "confused_count": len(confused_positions)})
        else:
            pred = extract_answer(output1)
            correct = compare_answers(pred, item["answer"])
            results["esr"].append({"correct": correct, "tokens": tokens1, "revised": False, "confused_count": len(confused_positions)})
        
        # Entropy-only check: high entropy only
        high_entropy_count = sum(1 for h, v in uncertainties if h > tau_h)
        entropy_triggered = high_entropy_count >= 5 and tokens1 > 30
        
        if entropy_triggered:
            revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider.\n"
            output2, tokens2 = generate_vanilla(model, tokenizer, revision_prompt)
            pred = extract_answer(output2)
            correct = compare_answers(pred, item["answer"])
            results["entropy_only"].append({"correct": correct, "tokens": tokens1 + tokens2, "revised": True})
        else:
            pred = extract_answer(output1)
            correct = compare_answers(pred, item["answer"])
            results["entropy_only"].append({"correct": correct, "tokens": tokens1, "revised": False})
        
        # EGL - simple heuristic-based
        output_egl, tokens_egl = generate_vanilla(model, tokenizer, prompt)
        markers = ["maybe", "perhaps", "uncertain", "think", "not sure"]
        needs_refine = any(m in output_egl.lower() for m in markers) or len(output_egl) > 200
        
        if needs_refine:
            refine_prompt = f"{prompt}{output_egl}\n\nLet me review and correct:\n"
            output2, tokens2 = generate_vanilla(model, tokenizer, refine_prompt)
            pred = extract_answer(output2)
            correct = compare_answers(pred, item["answer"])
            results["egl"].append({"correct": correct, "tokens": tokens_egl + tokens2, "refined": True})
        else:
            pred = extract_answer(output_egl)
            correct = compare_answers(pred, item["answer"])
            results["egl"].append({"correct": correct, "tokens": tokens_egl, "refined": False})
        
        # Best-of-N (only on subset to save time)
        if i < min(50, len(dataset)):
            from collections import Counter
            answers = []
            total_tokens = 0
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            
            for _ in range(4):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=256, 
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.pad_token_id
                    )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                output_text = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
                tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                
                pred = extract_answer(output_text)
                answers.append(pred)
                total_tokens += tokens
            
            answer_counts = Counter(answers)
            best_answer = answer_counts.most_common(1)[0][0]
            correct = compare_answers(best_answer, item["answer"])
            results["bestofn"].append({"correct": correct, "tokens": total_tokens})
    
    runtime = time.time() - start_time
    
    # Compute summary statistics
    summary = {}
    for method, res in results.items():
        if not res:
            continue
        accuracy = sum(r["correct"] for r in res) / len(res)
        avg_tokens = sum(r["tokens"] for r in res) / len(res)
        
        summary[method] = {
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "count": len(res)
        }
        
        if method in ["esr", "entropy_only"]:
            revision_rate = sum(r.get("revised", False) for r in res) / len(res)
            summary[method]["revision_rate"] = revision_rate
        
        if method == "egl":
            refine_rate = sum(r.get("refined", False) for r in res) / len(res)
            summary[method]["refine_rate"] = refine_rate
    
    summary["runtime"] = runtime
    summary["seed"] = seed
    
    return results, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", type=str, default="exp/results/all_methods_results.json")
    parser.add_argument("--tau_h", type=float, default=0.55)
    parser.add_argument("--tau_v", type=float, default=0.005)
    args = parser.parse_args()
    
    dataset = load_json("exp/data/gsm8k_test.json")
    results, summary = run_experiments(dataset, args.seed, args.limit, args.tau_h, args.tau_v)
    
    output_data = {
        "results": results,
        "summary": summary,
        "config": {
            "seed": args.seed,
            "limit": args.limit,
            "tau_h": args.tau_h,
            "tau_v": args.tau_v
        }
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n=== Results Summary (seed {args.seed}) ===")
    for method, stats in summary.items():
        if isinstance(stats, dict) and "accuracy" in stats:
            print(f"{method:15s}: Accuracy={stats['accuracy']:.3f}, Tokens={stats['avg_tokens']:.1f}")
            if "revision_rate" in stats:
                print(f"                 Revision Rate={stats['revision_rate']:.2%}")
    print(f"\nRuntime: {summary['runtime']:.1f}s")
    print(f"Results saved to {args.output}")
