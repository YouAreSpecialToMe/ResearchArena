"""Mini experiment runner - runs on small subset for quick validation."""

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from shared.data_loader import load_json, create_cot_prompt
from shared.metrics import compare_answers
from shared.utils import set_seed


def extract_answer(text: str):
    """Extract answer from text."""
    import re
    if not text:
        return None
    
    # Look for #### pattern
    if "####" in text:
        match = re.search(r"####\s*([-\d.,]+)", text)
        if match:
            return match.group(1)
    
    # Look for boxed answer
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for last number
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    
    return text.strip()


def run_experiment(method: str, dataset: List[Dict], seed: int, 
                   tau_h: float = 2.5, tau_v: float = 1.5) -> Dict:
    """Run single experiment."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    set_seed(seed)
    
    print(f"  Loading model...")
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
    
    results = []
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(dataset)}")
        
        prompt = create_cot_prompt(item["question"])
        
        if method == "vanilla":
            # Simple generation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            pred = extract_answer(output_text)
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": tokens,
                "revision_triggered": False
            })
            
        elif method == "esr":
            # ESR with entropy-varentropy
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            generated_ids = inputs["input_ids"]
            uncertainties = []
            
            with torch.no_grad():
                for _ in range(256):
                    outputs = model(input_ids=generated_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    # Compute entropy and varentropy
                    probs = F.softmax(next_logits[0], dim=-1)
                    log_probs = F.log_softmax(next_logits[0], dim=-1)
                    h = -(probs * log_probs).sum().item()
                    v = (probs * (log_probs + h) ** 2).sum().item()
                    uncertainties.append((h, v))
                    
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            output1 = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens1 = generated_ids.shape[1] - inputs["input_ids"].shape[1]
            
            # Check for high uncertainty
            high_uncertainty = any(h > tau_h and v < tau_v for h, v in uncertainties)
            revision_triggered = high_uncertainty and tokens1 > 30
            
            if revision_triggered:
                # Revision
                revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully.\n"
                rev_inputs = tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    rev_outputs = model.generate(
                        **rev_inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                output2 = tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
                output2 = output2[len(tokenizer.decode(rev_inputs["input_ids"][0], skip_special_tokens=True)):]
                tokens2 = rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1]
                
                pred = extract_answer(output2)
                total_tokens = tokens1 + tokens2
            else:
                pred = extract_answer(output1)
                total_tokens = tokens1
            
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens,
                "revision_triggered": revision_triggered
            })
            
        elif method == "entropy_only":
            # Entropy only
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            generated_ids = inputs["input_ids"]
            entropies = []
            
            with torch.no_grad():
                for _ in range(256):
                    outputs = model(input_ids=generated_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    probs = F.softmax(next_logits[0], dim=-1)
                    log_probs = F.log_softmax(next_logits[0], dim=-1)
                    h = -(probs * log_probs).sum().item()
                    entropies.append(h)
                    
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            output1 = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens1 = generated_ids.shape[1] - inputs["input_ids"].shape[1]
            
            high_entropy = any(h > tau_h for h in entropies)
            revision_triggered = high_entropy and tokens1 > 30
            
            if revision_triggered:
                revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully.\n"
                rev_inputs = tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    rev_outputs = model.generate(
                        **rev_inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                output2 = tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
                output2 = output2[len(tokenizer.decode(rev_inputs["input_ids"][0], skip_special_tokens=True)):]
                tokens2 = rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1]
                
                pred = extract_answer(output2)
                total_tokens = tokens1 + tokens2
            else:
                pred = extract_answer(output1)
                total_tokens = tokens1
            
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens,
                "revision_triggered": revision_triggered
            })
            
        elif method == "egl":
            # Post-hoc refinement
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens1 = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            # Check for refinement need
            uncertainty_markers = ["maybe", "perhaps", "uncertain", "think", "might"]
            needs_refinement = any(m in output1.lower() for m in uncertainty_markers) or len(output1) > 200
            
            if needs_refinement:
                refinement_prompt = f"{prompt}{output1}\n\nLet me review and correct:\n"
                ref_inputs = tokenizer(refinement_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    ref_outputs = model.generate(
                        **ref_inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                output2 = tokenizer.decode(ref_outputs[0], skip_special_tokens=True)
                output2 = output2[len(tokenizer.decode(ref_inputs["input_ids"][0], skip_special_tokens=True)):]
                tokens2 = ref_outputs.shape[1] - ref_inputs["input_ids"].shape[1]
                
                pred = extract_answer(output2)
                total_tokens = tokens1 + tokens2
            else:
                pred = extract_answer(output1)
                total_tokens = tokens1
            
            correct = compare_answers(pred, item["answer"])
            
            results.append({
                "correct": correct,
                "tokens": total_tokens,
                "revision_triggered": needs_refinement
            })
    
    runtime = time.time() - start_time
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return {
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results),
        "revision_rate": sum(r["revision_triggered"] for r in results) / len(results),
        "runtime": runtime,
        "results": results
    }


def main():
    # Load dataset
    dataset = load_json("exp/data/gsm8k_test.json")
    
    # Use subset for faster experimentation
    subset_size = 50
    dataset = dataset[:subset_size]
    
    print(f"Running mini experiments on {subset_size} examples")
    
    seeds = [42, 123, 456]
    methods = ["vanilla", "entropy_only", "esr", "egl"]
    
    all_results = {}
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running {method}...")
        print(f"{'='*60}")
        
        method_results = []
        for seed in seeds:
            print(f"\n  Seed {seed}:")
            result = run_experiment(method, dataset, seed)
            method_results.append(result)
            print(f"    Accuracy: {result['accuracy']:.3f}")
            print(f"    Avg Tokens: {result['avg_tokens']:.1f}")
            print(f"    Revision Rate: {result.get('revision_rate', 0):.2%}")
            print(f"    Runtime: {result['runtime']:.1f}s")
        
        all_results[method] = {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in method_results])),
            "accuracy_std": float(np.std([r["accuracy"] for r in method_results])),
            "tokens_mean": float(np.mean([r["avg_tokens"] for r in method_results])),
            "tokens_std": float(np.std([r["avg_tokens"] for r in method_results])),
            "revision_rate_mean": float(np.mean([r.get("revision_rate", 0) for r in method_results])),
            "seeds": method_results
        }
    
    # Save results
    Path("exp/results").mkdir(parents=True, exist_ok=True)
    with open("exp/results/mini_experiments.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Accuracy':<20} {'Tokens':<15} {'Revision Rate':<15}")
    print("-"*60)
    for method, res in all_results.items():
        acc_str = f"{res['accuracy_mean']:.3f} ± {res['accuracy_std']:.3f}"
        tok_str = f"{res['tokens_mean']:.1f}"
        rev_str = f"{res['revision_rate_mean']:.1%}"
        print(f"{method:<20} {acc_str:<20} {tok_str:<15} {rev_str:<15}")
    print("="*60)


if __name__ == "__main__":
    main()
