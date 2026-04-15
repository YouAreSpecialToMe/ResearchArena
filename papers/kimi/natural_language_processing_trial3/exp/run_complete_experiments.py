"""
Complete experiment runner with inline class definitions.
This avoids import path issues.
"""

import torch
import torch.nn.functional as F
import json
import time
import random
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import Dict, Any, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from shared.models import load_model, generate_vanilla_cot, compute_entropy
from shared.data_loader import load_gsm8k, load_math500, create_cot_prompt, extract_numeric_answer


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# ESR Generator
# ============================================================================
class ESRGenerator:
    """Generator with entropy-guided stepwise revision."""
    
    def __init__(self, model, tokenizer, tau_h=2.5, tau_v=1.5, r_max=3, max_new_tokens=1024, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.tau_v = tau_v
        self.r_max = r_max
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.revision_count = 0
    
    def compute_uncertainty(self, logits):
        """Compute entropy and varentropy."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum()
        varentropy = (probs * (log_probs + entropy) ** 2).sum()
        return entropy.item(), varentropy.item()
    
    def should_revise(self, entropy, varentropy):
        """Check if revision should be triggered."""
        return entropy > self.tau_h and varentropy < self.tau_v
    
    def generate(self, prompt, track_uncertainty=True):
        """Generate with ESR."""
        self.revision_count = 0
        revision_history = []
        uncertainty_triggers = []
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        total_tokens = 0
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(input_ids=generated_ids, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]
                
                entropy, varentropy = self.compute_uncertainty(next_token_logits[0])
                
                # Check for revision trigger
                if (track_uncertainty and self.should_revise(entropy, varentropy) and 
                    self.revision_count < self.r_max and step > 10):
                    
                    current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reasoning_so_far = current_text[len(prompt):]
                    
                    # Only revise if we have substantial content
                    if len(reasoning_so_far) > 20:
                        uncertainty_triggers.append({
                            "position": step,
                            "entropy": entropy,
                            "varentropy": varentropy,
                        })
                        
                        # Generate revision
                        revision_prompt = (
                            f"{prompt}{reasoning_so_far}\n\n"
                            f"Wait, let me reconsider this step more carefully.\n"
                        )
                        
                        rev_inputs = self.tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048)
                        rev_inputs = {k: v.to(self.device) for k, v in rev_inputs.items()}
                        
                        rev_outputs = self.model.generate(
                            **rev_inputs,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        
                        rev_text = self.tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
                        revision_output = rev_text[len(revision_prompt):]
                        
                        # Continue from revision
                        new_prompt = f"{prompt}{reasoning_so_far}\n{revision_output}"
                        new_inputs = self.tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=2048)
                        new_inputs = {k: v.to(self.device) for k, v in new_inputs.items()}
                        
                        generated_ids = new_inputs["input_ids"]
                        self.revision_count += 1
                        revision_history.append({
                            "step": step,
                            "entropy": entropy,
                            "varentropy": varentropy,
                        })
                        continue
                
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                total_tokens += 1
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if generated_ids.shape[1] > 4096:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        return {
            "output": output_text,
            "total_tokens": total_tokens,
            "revision_count": self.revision_count,
            "revision_history": revision_history,
            "uncertainty_triggers": uncertainty_triggers,
        }


# ============================================================================
# Entropy-Only Generator
# ============================================================================
class EntropyOnlyGenerator:
    """Entropy-only trigger for revision (no varentropy)."""
    
    def __init__(self, model, tokenizer, tau_h=2.5, r_max=3, max_new_tokens=1024, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.r_max = r_max
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt):
        """Generate with entropy-only revision trigger."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        total_tokens = 0
        revision_count = 0
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(input_ids=generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :]
                
                probs = F.softmax(next_logits[0], dim=-1)
                log_probs = F.log_softmax(next_logits[0], dim=-1)
                entropy = -(probs * log_probs).sum().item()
                
                # Trigger revision on high entropy only
                if entropy > self.tau_h and revision_count < self.r_max and step > 10:
                    current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reasoning_so_far = current_text[len(prompt):]
                    
                    revision_prompt = f"{prompt}{reasoning_so_far}\n\nWait, let me reconsider this step more carefully.\n"
                    
                    rev_inputs = self.tokenizer(revision_prompt, return_tensors="pt", truncation=True, max_length=2048)
                    rev_inputs = {k: v.to(self.device) for k, v in rev_inputs.items()}
                    
                    rev_outputs = self.model.generate(
                        **rev_inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    rev_text = self.tokenizer.decode(rev_outputs[0], skip_special_tokens=True)
                    revision_output = rev_text[len(revision_prompt):]
                    
                    new_prompt = f"{prompt}{reasoning_so_far}\n{revision_output}"
                    new_inputs = self.tokenizer(new_prompt, return_tensors="pt", truncation=True, max_length=2048)
                    new_inputs = {k: v.to(self.device) for k, v in new_inputs.items()}
                    
                    generated_ids = new_inputs["input_ids"]
                    revision_count += 1
                    continue
                
                # Greedy decoding
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                total_tokens += 1
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if generated_ids.shape[1] > 4096:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        return {"output": output_text, "total_tokens": total_tokens, "revision_count": revision_count}


# ============================================================================
# EGL Post-Hoc
# ============================================================================
class EGLPostHoc:
    """EGL: Generate fully, then refine if average entropy is high."""
    
    def __init__(self, model, tokenizer, tau_h=2.5, max_new_tokens=1024, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt):
        """Generate with post-hoc refinement if needed."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        uncertainty_readings = []
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(input_ids=generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :]
                
                probs = F.softmax(next_logits[0], dim=-1)
                log_probs = F.log_softmax(next_logits[0], dim=-1)
                entropy = -(probs * log_probs).sum().item()
                uncertainty_readings.append(entropy)
                
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if generated_ids.shape[1] > 4096:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        avg_entropy = sum(uncertainty_readings) / len(uncertainty_readings)
        
        refined = False
        if avg_entropy > self.tau_h:
            refined = True
            refinement_prompt = f"{prompt}{output_text}\n\nThe previous reasoning may have errors. Please review and correct:\n"
            
            ref_inputs = self.tokenizer(refinement_prompt, return_tensors="pt", truncation=True, max_length=2048)
            ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}
            
            with torch.no_grad():
                ref_outputs = self.model.generate(
                    **ref_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            ref_text = self.tokenizer.decode(ref_outputs[0], skip_special_tokens=True)
            output_text = ref_text[len(refinement_prompt):]
        
        return {
            "output": output_text,
            "total_tokens": generated_ids.shape[1] - inputs["input_ids"].shape[1],
            "avg_entropy": avg_entropy,
            "refined": refined
        }


# ============================================================================
# EGB Beam Search
# ============================================================================
class EGBBeamSearch:
    """Entropy-Gated Beam Search with K=3 beams."""
    
    def __init__(self, model, tokenizer, tau_h=2.5, k_beams=3, max_new_tokens=1024, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.k_beams = k_beams
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt):
        """Generate with entropy-gated beam search."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        beams = [(inputs["input_ids"].clone(), 0.0, False)]
        total_tokens = 0
        branch_count = 0
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                new_beams = []
                
                for beam_ids, beam_score, branched in beams:
                    if branched:
                        new_beams.append((beam_ids, beam_score, True))
                        continue
                    
                    outputs = self.model(input_ids=beam_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    probs = F.softmax(next_logits[0], dim=-1)
                    log_probs = F.log_softmax(next_logits[0], dim=-1)
                    entropy = -(probs * log_probs).sum().item()
                    
                    log_probs_full = F.log_softmax(next_logits, dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs_full, self.k_beams, dim=-1)
                    
                    if entropy > self.tau_h and len(beams) < self.k_beams * 2:
                        branch_count += 1
                        for k in range(min(self.k_beams, 3)):
                            next_token = topk_indices[0, k:k+1].unsqueeze(0)
                            new_ids = torch.cat([beam_ids, next_token], dim=-1)
                            new_score = beam_score + topk_log_probs[0, k].item()
                            new_beams.append((new_ids, new_score, False))
                    else:
                        next_token = topk_indices[0, 0:1].unsqueeze(0)
                        new_ids = torch.cat([beam_ids, next_token], dim=-1)
                        new_score = beam_score + topk_log_probs[0, 0].item()
                        
                        if next_token.item() == self.tokenizer.eos_token_id:
                            new_beams.append((new_ids, new_score, True))
                        else:
                            new_beams.append((new_ids, new_score, False))
                    
                    total_tokens += 1
                
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:self.k_beams]
                
                if all(branched for _, _, branched in beams):
                    break
                
                if total_tokens > self.max_new_tokens * self.k_beams:
                    break
        
        best_beam = max(beams, key=lambda x: x[1])
        best_ids = best_beam[0]
        
        generated_text = self.tokenizer.decode(best_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        return {"output": output_text, "total_tokens": total_tokens, "branch_count": branch_count}


# ============================================================================
# Best-of-N
# ============================================================================
class BestOfN:
    """Best-of-N: Sample N=4 and select via majority voting."""
    
    def __init__(self, model, tokenizer, n_samples=4, temperature=0.7, max_new_tokens=1024, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate_single(self, prompt):
        """Generate a single sample."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        num_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        
        return {"output": output_text, "tokens": num_tokens}
    
    def generate(self, prompt):
        """Generate N samples and select via majority voting."""
        samples = []
        total_tokens = 0
        
        for i in range(self.n_samples):
            sample = self.generate_single(prompt)
            samples.append(sample)
            total_tokens += sample["tokens"]
        
        answers = []
        for sample in samples:
            answer = extract_numeric_answer(sample["output"])
            answers.append(answer)
        
        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            answer_counts = Counter(valid_answers)
            best_answer = answer_counts.most_common(1)[0][0]
        else:
            best_answer = None
        
        best_sample_idx = 0
        for i, answer in enumerate(answers):
            if answer == best_answer:
                best_sample_idx = i
                break
        
        return {
            "output": samples[best_sample_idx]["output"],
            "total_tokens": total_tokens,
            "best_answer": best_answer,
            "agreement_count": answer_counts.get(best_answer, 0) if valid_answers else 0
        }


# ============================================================================
# Experiment Runners
# ============================================================================
def run_vanilla(model, tokenizer, data, seed, max_tokens=1024):
    """Run vanilla CoT baseline."""
    set_seed(seed)
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        output_text, num_tokens = generate_vanilla_cot(model, tokenizer, prompt, max_new_tokens=max_tokens, temperature=0.0)
        
        predicted = extract_numeric_answer(output_text)
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += num_tokens
        
        results.append({"question_idx": i, "predicted": predicted, "actual": actual, "correct": is_correct, "tokens": num_tokens})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {"accuracy": correct / len(data), "avg_tokens": total_tokens / len(data), "correct_count": correct, "total_problems": len(data), "results": results}


def run_entropy_only(model, tokenizer, data, seed, tau_h=2.5, max_tokens=1024):
    """Run entropy-only baseline."""
    set_seed(seed)
    generator = EntropyOnlyGenerator(model, tokenizer, tau_h=tau_h, r_max=3, max_new_tokens=max_tokens)
    
    results = []
    correct = 0
    total_tokens = 0
    total_revisions = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = generator.generate(prompt)
        
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        total_revisions += result["revision_count"]
        
        results.append({"question_idx": i, "predicted": predicted, "actual": actual, "correct": is_correct, "tokens": result["total_tokens"], "revisions": result["revision_count"]})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}, Revisions: {total_revisions}")
    
    revision_rate = sum(1 for r in results if r["revisions"] > 0) / len(data)
    return {"accuracy": correct / len(data), "avg_tokens": total_tokens / len(data), "avg_revisions": total_revisions / len(data), "revision_rate": revision_rate, "correct_count": correct, "total_problems": len(data), "results": results}


def run_esr(model, tokenizer, data, seed, tau_h=2.5, tau_v=1.5, max_tokens=1024):
    """Run full ESR method."""
    set_seed(seed)
    generator = ESRGenerator(model, tokenizer, tau_h=tau_h, tau_v=tau_v, r_max=3, max_new_tokens=max_tokens)
    
    results = []
    correct = 0
    total_tokens = 0
    total_revisions = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = generator.generate(prompt, track_uncertainty=True)
        
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        total_revisions += result["revision_count"]
        
        results.append({"question_idx": i, "predicted": predicted, "actual": actual, "correct": is_correct, "tokens": result["total_tokens"], "revisions": result["revision_count"], "revision_history": result.get("revision_history", []), "uncertainty_triggers": result.get("uncertainty_triggers", [])})
        
        if (i + 1) % 10 == 0:
            revision_rate = sum(1 for r in results if r["revisions"] > 0) / len(results)
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}, RevRate: {revision_rate:.3f}")
    
    revision_rate = sum(1 for r in results if r["revisions"] > 0) / len(data)
    return {"accuracy": correct / len(data), "avg_tokens": total_tokens / len(data), "avg_revisions": total_revisions / len(data), "revision_rate": revision_rate, "correct_count": correct, "total_problems": len(data), "results": results}


def run_egl_posthoc(model, tokenizer, data, seed, tau_h=2.5, max_tokens=1024):
    """Run EGL post-hoc baseline."""
    set_seed(seed)
    egl = EGLPostHoc(model, tokenizer, tau_h=tau_h, max_new_tokens=max_tokens)
    
    results = []
    correct = 0
    total_tokens = 0
    refined_count = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = egl.generate(prompt)
        
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        if result["refined"]:
            refined_count += 1
        
        results.append({"question_idx": i, "predicted": predicted, "actual": actual, "correct": is_correct, "tokens": result["total_tokens"], "refined": result["refined"], "avg_entropy": result["avg_entropy"]})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {"accuracy": correct / len(data), "avg_tokens": total_tokens / len(data), "refinement_rate": refined_count / len(data), "correct_count": correct, "total_problems": len(data), "results": results}


def run_egb_beam(model, tokenizer, data, seed, tau_h=2.5, max_tokens=1024):
    """Run EGB beam search baseline."""
    set_seed(seed)
    egb = EGBBeamSearch(model, tokenizer, tau_h=tau_h, k_beams=3, max_new_tokens=max_tokens)
    
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = egb.generate(prompt)
        
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        
        results.append({"question_idx": i, "predicted": predicted, "actual": actual, "correct": is_correct, "tokens": result["total_tokens"], "branch_count": result["branch_count"]})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {"accuracy": correct / len(data), "avg_tokens": total_tokens / len(data), "correct_count": correct, "total_problems": len(data), "results": results}


def run_bestofn(model, tokenizer, data, seed, max_tokens=1024):
    """Run Best-of-N baseline."""
    set_seed(seed)
    bon = BestOfN(model, tokenizer, n_samples=4, temperature=0.7, max_new_tokens=max_tokens)
    
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        result = bon.generate(prompt)
        
        predicted = result["best_answer"]
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        
        results.append({"question_idx": i, "predicted": predicted, "actual": actual, "correct": is_correct, "tokens": result["total_tokens"], "agreement": result["agreement_count"]})
        
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} done, Acc: {correct/(i+1):.3f}")
    
    return {"accuracy": correct / len(data), "avg_tokens": total_tokens / len(data), "correct_count": correct, "total_problems": len(data), "results": results}


METHOD_RUNNERS = {
    "vanilla": run_vanilla,
    "entropy_only": run_entropy_only,
    "esr": run_esr,
    "egl_posthoc": run_egl_posthoc,
    "egb_beam": run_egb_beam,
    "bestofn": run_bestofn,
}


def main():
    parser = argparse.ArgumentParser(description="Run complete experiments")
    parser.add_argument("--method", required=True, choices=["vanilla", "entropy_only", "esr", "egl_posthoc", "egb_beam", "bestofn", "all"], help="Method to run")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"], help="Dataset to use")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_problems", type=int, default=None, help="Maximum number of problems to evaluate")
    parser.add_argument("--tau_h", type=float, default=1.5, help="Entropy threshold")
    parser.add_argument("--tau_v", type=float, default=0.8, help="Varentropy threshold")
    parser.add_argument("--output_dir", default="exp/results", help="Output directory")
    args = parser.parse_args()
    
    print("="*70)
    print(f"Complete Experiment Runner")
    print("="*70)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Tau_H: {args.tau_h}, Tau_V: {args.tau_v}")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    if args.dataset == "gsm8k":
        all_data = load_gsm8k("test")
    else:
        all_data = load_math500()
    
    set_seed(args.seed)
    if args.max_problems:
        data = random.sample(all_data, min(args.max_problems, len(all_data)))
    else:
        data = all_data
    print(f"Loaded {len(data)} problems")
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model)
    
    # Determine which methods to run
    if args.method == "all":
        methods_to_run = ["vanilla", "entropy_only", "esr", "egl_posthoc"]
    else:
        methods_to_run = [args.method]
    
    # Run each method
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method in methods_to_run:
        print(f"\n{'='*70}")
        print(f"Running {method}...")
        print('='*70)
        
        start_time = time.time()
        runner = METHOD_RUNNERS[method]
        
        if method in ["entropy_only", "egl_posthoc", "egb_beam"]:
            result = runner(model, tokenizer, data, args.seed, tau_h=args.tau_h)
        elif method == "esr":
            result = runner(model, tokenizer, data, args.seed, tau_h=args.tau_h, tau_v=args.tau_v)
        else:
            result = runner(model, tokenizer, data, args.seed)
        
        elapsed = time.time() - start_time
        
        result["method"] = method
        result["dataset"] = args.dataset
        result["model"] = args.model
        result["seed"] = args.seed
        result["runtime_seconds"] = elapsed
        
        if method == "esr":
            result["tau_h"] = args.tau_h
            result["tau_v"] = args.tau_v
        elif method in ["entropy_only", "egl_posthoc", "egb_beam"]:
            result["tau_h"] = args.tau_h
        
        output_file = output_dir / f"{method}_{args.dataset}_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{method} completed in {elapsed/60:.1f} minutes")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Avg Tokens: {result['avg_tokens']:.1f}")
        if "revision_rate" in result:
            print(f"  Revision Rate: {result['revision_rate']:.3f}")
        if "refinement_rate" in result:
            print(f"  Refinement Rate: {result['refinement_rate']:.3f}")
        print(f"  Saved to: {output_file}")
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)


if __name__ == "__main__":
    main()
