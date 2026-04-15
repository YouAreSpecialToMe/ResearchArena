"""EGL-style Post-Hoc Refinement baseline implementation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model, compute_entropy
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


class EGLPostHoc:
    """EGL: Generate fully, then refine if average entropy is high."""
    
    def __init__(
        self,
        model,
        tokenizer,
        tau_h: float = 2.5,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate with post-hoc refinement if needed."""
        # First pass: generate full response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        uncertainty_readings = []
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(input_ids=generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :]
                
                entropy = compute_entropy(next_logits[0])
                uncertainty_readings.append(entropy)
                
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if generated_ids.shape[1] > 4096:
                    break
        
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        # Compute average entropy
        avg_entropy = sum(uncertainty_readings) / len(uncertainty_readings)
        
        # Check if refinement is needed
        refined = False
        if avg_entropy > self.tau_h:
            refined = True
            # Post-hoc refinement
            refinement_prompt = (
                f"{prompt}{output_text}\n\n"
                f"The previous reasoning may have errors. Please review and correct:\n"
            )
            
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


def run_egl_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                       seed: int = 42, max_problems: int = None):
    """Run EGL post-hoc experiment."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Running EGL Post-Hoc on {dataset_name} with seed {seed}")
    
    # Load model
    model, tokenizer = load_model(model_name)
    
    # Load data
    if dataset_name == "gsm8k":
        data = load_gsm8k("test")
    else:
        from shared.data_loader import load_math500
        data = load_math500()
    
    if max_problems:
        data = random.sample(data, min(max_problems, len(data)))
    
    # Initialize EGL
    egl = EGLPostHoc(model, tokenizer, tau_h=2.5)
    
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
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "refined": result["refined"],
            "avg_entropy": result["avg_entropy"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    refinement_rate = refined_count / len(data)
    
    output = {
        "method": "egl_posthoc",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "refinement_rate": refinement_rate,
        "total_problems": len(data),
        "correct_count": correct,
        "results": results
    }
    
    # Save results
    output_path = Path(f"exp/results/egl_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"EGL Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}, Refinement Rate={refinement_rate:.3f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()
    
    run_egl_experiment(args.dataset, args.model, args.seed, args.max_problems)
