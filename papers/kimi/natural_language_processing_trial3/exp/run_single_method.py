"""Run single method experiment to manage memory."""

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def run_method(method: str, dataset: List[Dict], seed: int, 
               tau_h: float = 2.5, tau_v: float = 1.5) -> Dict:
    set_seed(seed)
    
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
    
    results = []
    start_time = time.time()
    
    for i, item in enumerate(dataset):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(dataset)}")
        
        prompt = create_cot_prompt(item["question"])
        
        if method == "vanilla":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output_text = output_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            pred = extract_answer(output_text)
            correct = compare_answers(pred, item["answer"])
            results.append({"correct": correct, "tokens": tokens})
            
        elif method == "esr":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            generated_ids = inputs["input_ids"]
            uncertainties = []
            
            with torch.no_grad():
                for _ in range(256):
                    outputs = model(input_ids=generated_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
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
            
            high_uncertainty = any(h > tau_h and v < tau_v for h, v in uncertainties)
            revision_triggered = high_uncertainty and tokens1 > 30
            
            if revision_triggered:
                revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider this more carefully.\n"
                rev_inputs = tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    rev_outputs = model.generate(**rev_inputs, max_new_tokens=256, do_sample=False)
                output2 = tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
                output2 = output2[len(tokenizer.decode(rev_inputs["input_ids"][0], skip_special_tokens=True)):]
                tokens2 = rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1]
                pred = extract_answer(output2)
                total_tokens = tokens1 + tokens2
            else:
                pred = extract_answer(output1)
                total_tokens = tokens1
            
            correct = compare_answers(pred, item["answer"])
            results.append({"correct": correct, "tokens": total_tokens, "revision_triggered": revision_triggered})
        
        elif method == "entropy_only":
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
                revision_prompt = f"{prompt}{output1}\n\nWait, let me reconsider.\n"
                rev_inputs = tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    rev_outputs = model.generate(**rev_inputs, max_new_tokens=256, do_sample=False)
                output2 = tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
                tokens2 = rev_outputs.shape[1] - rev_inputs["input_ids"].shape[1]
                pred = extract_answer(output2)
                total_tokens = tokens1 + tokens2
            else:
                pred = extract_answer(output1)
                total_tokens = tokens1
            
            correct = compare_answers(pred, item["answer"])
            results.append({"correct": correct, "tokens": total_tokens, "revision_triggered": revision_triggered})
        
        elif method == "egl":
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            output1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
            output1 = output1[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
            tokens1 = outputs.shape[1] - inputs["input_ids"].shape[1]
            
            markers = ["maybe", "perhaps", "uncertain", "think"]
            needs_refine = any(m in output1.lower() for m in markers) or len(output1) > 200
            
            if needs_refine:
                refine_prompt = f"{prompt}{output1}\n\nLet me review and correct:\n"
                ref_inputs = tokenizer(refine_prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
                with torch.no_grad():
                    ref_outputs = model.generate(**ref_inputs, max_new_tokens=256, do_sample=False)
                output2 = tokenizer.decode(ref_outputs[0], skip_special_tokens=True)
                output2 = output2[len(tokenizer.decode(ref_inputs["input_ids"][0], skip_special_tokens=True)):]
                tokens2 = ref_outputs.shape[1] - ref_inputs["input_ids"].shape[1]
                pred = extract_answer(output2)
                total_tokens = tokens1 + tokens2
            else:
                pred = extract_answer(output1)
                total_tokens = tokens1
            
            correct = compare_answers(pred, item["answer"])
            results.append({"correct": correct, "tokens": total_tokens, "revision_triggered": needs_refine})
    
    runtime = time.time() - start_time
    del model
    torch.cuda.empty_cache()
    
    return {
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "avg_tokens": sum(r["tokens"] for r in results) / len(results),
        "revision_rate": sum(r.get("revision_triggered", False) for r in results) / len(results),
        "runtime": runtime,
        "results": results
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    dataset = load_json("exp/data/gsm8k_test.json")[:args.limit]
    result = run_method(args.method, dataset, args.seed)
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults for {args.method} (seed {args.seed}):")
    print(f"  Accuracy: {result['accuracy']:.3f}")
    print(f"  Avg Tokens: {result['avg_tokens']:.1f}")
    print(f"  Revision Rate: {result['revision_rate']:.2%}")
    print(f"  Runtime: {result['runtime']:.1f}s")
