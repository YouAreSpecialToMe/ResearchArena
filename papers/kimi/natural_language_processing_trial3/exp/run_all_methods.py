"""Run all methods efficiently - load model once, run all experiments."""

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
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


def generate_vanilla(model, tokenizer, prompt, max_tokens=256):
    """Vanilla CoT generation."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return output_text, tokens


def generate_esr(model, tokenizer, prompt, tau_h=2.5, tau_v=1.5, max_tokens=256):
    """ESR generation with entropy-varentropy triggering."""
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
    
    output1 = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokens1 = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    
    # Check for confusion state (high entropy, low varentropy)
    confused_positions = [(h, v) for h, v in uncertainties if h > tau_h and v < tau_v]
    revision_triggered = len(confused_positions) >= 3 and tokens1 > 30
    
    if revision_triggered:
        revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully.\n"
        rev_inputs = tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            rev_outputs = model.generate(**rev_inputs, max_new_tokens=max_tokens, do_sample=False)
        output2 = tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
        output2 = output2[len(tokenizer.decode(rev_inputs["input_ids"][0], skip_special_tokens=True)):]
        tokens2 = rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1]
        pred = extract_answer(output2)
        return pred, tokens1 + tokens2, True, len(confused_positions)
    
    pred = extract_answer(output1)
    return pred, tokens1, False, len(confused_positions)


def generate_entropy_only(model, tokenizer, prompt, tau_h=2.5, max_tokens=256):
    """Entropy-only baseline."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    generated_ids = inputs["input_ids"]
    entropies = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids=generated_ids, return_dict=True)
            next_logits = outputs.logits[:, -1, :]
            h = compute_entropy(next_logits[0])
            entropies.append(h)
            
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    output1 = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokens1 = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    
    high_entropy_count = sum(1 for h in entropies if h > tau_h)
    revision_triggered = high_entropy_count >= 5 and tokens1 > 30
    
    if revision_triggered:
        revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider.\n"
        rev_inputs = tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            rev_outputs = model.generate(**rev_inputs, max_new_tokens=max_tokens, do_sample=False)
        output2 = tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
        output2 = output2[len(tokenizer.decode(rev_inputs["input_ids"][0], skip_special_tokens=True)):]
        tokens2 = rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1]
        pred = extract_answer(output2)
        return pred, tokens1 + tokens2, True
    
    pred = extract_answer(output1)
    return pred, tokens1, False


def generate_egl(model, tokenizer, prompt, max_tokens=256):
    """EGL-style post-hoc refinement."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    output1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    tokens1 = outputs.shape[1] - inputs["input_ids"].shape[1]
    
    # Trigger refinement based on markers or length
    markers = ["maybe", "perhaps", "uncertain", "think", "not sure"]
    needs_refine = any(m in output1.lower() for m in markers) or len(output1) > 200
    
    if needs_refine:
        refine_prompt = f"{prompt}{output1}\n\nLet me review and correct:\n"
        ref_inputs = tokenizer(refine_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            ref_outputs = model.generate(**ref_inputs, max_new_tokens=max_tokens, do_sample=False)
        output2 = tokenizer.decode(ref_outputs[0], skip_special_tokens=True)
        output2 = output2[len(tokenizer.decode(ref_inputs["input_ids"][0], skip_special_tokens=True)):]
        tokens2 = ref_outputs.shape[1] - ref_inputs["input_ids"].shape[1]
        pred = extract_answer(output2)
        return pred, tokens1 + tokens2, True
    
    pred = extract_answer(output1)
    return pred, tokens1, False


def generate_bestofn(model, tokenizer, prompt, n=4, temperature=0.7, max_tokens=256):
    """Best-of-N sampling."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    answers = []
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(n):
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id
            )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            pred = extract_answer(output_text)
            answers.append(pred)
            total_tokens += tokens
    
    # Majority voting
    from collections import Counter
    answer_counts = Counter(answers)
    best_answer = answer_counts.most_common(1)[0][0]
    
    return best_answer, total_tokens


def run_experiments(dataset, seed=42, limit=None, tau_h=2.5, tau_v=1.5):
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
    print(f"Model loaded. Running experiments on {len(dataset)} problems...")
    
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
            print(f"  Progress: {i}/{len(dataset)} ({elapsed:.1f}s)")
        
        prompt = create_cot_prompt(item["question"])
        
        # Vanilla CoT
        output, tokens = generate_vanilla(model, tokenizer, prompt)
        pred = extract_answer(output)
        correct = compare_answers(pred, item["answer"])
        results["vanilla"].append({"correct": correct, "tokens": tokens})
        
        # ESR
        pred, tokens, revised, confused_count = generate_esr(model, tokenizer, prompt, tau_h, tau_v)
        correct = compare_answers(pred, item["answer"])
        results["esr"].append({"correct": correct, "tokens": tokens, "revised": revised, "confused_count": confused_count})
        
        # Entropy-Only
        pred, tokens, revised = generate_entropy_only(model, tokenizer, prompt, tau_h)
        correct = compare_answers(pred, item["answer"])
        results["entropy_only"].append({"correct": correct, "tokens": tokens, "revised": revised})
        
        # EGL
        pred, tokens, refined = generate_egl(model, tokenizer, prompt)
        correct = compare_answers(pred, item["answer"])
        results["egl"].append({"correct": correct, "tokens": tokens, "refined": refined})
        
        # Best-of-N (only on subset to save time)
        if i < min(50, len(dataset)):
            pred, tokens = generate_bestofn(model, tokenizer, prompt)
            correct = compare_answers(pred, item["answer"])
            results["bestofn"].append({"correct": correct, "tokens": tokens})
    
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
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=str, default="exp/results/all_methods_results.json")
    parser.add_argument("--tau_h", type=float, default=0.55)  # Based on analysis: ~80th percentile
    parser.add_argument("--tau_v", type=float, default=0.005)  # Based on analysis: ~25th percentile
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
