"""Entropix-style dynamic sampling baseline implementation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model, compute_entropy, compute_varentropy
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


class EntropixSampler:
    """Entropix-style dynamic temperature adjustment based on entropy-varentropy."""
    
    def __init__(
        self,
        model,
        tokenizer,
        tau_h: float = 2.5,
        tau_v: float = 1.5,
        base_temp: float = 0.7,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.tau_v = tau_v
        self.base_temp = base_temp
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def get_sampling_params(self, entropy: float, varentropy: float) -> Dict[str, float]:
        """Get dynamic sampling parameters based on uncertainty regime."""
        if entropy < self.tau_h:
            # Confident: low temperature, focused sampling
            return {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 20
            }
        elif varentropy > self.tau_v:
            # Fork: medium temperature to explore alternatives
            return {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 50
            }
        else:
            # Confused: higher temperature to break out of confusion
            return {
                "temperature": 1.0,
                "top_p": 0.98,
                "top_k": 100
            }
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate with dynamic sampling based on entropy-varentropy."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        total_tokens = 0
        regime_counts = {"confident": 0, "fork": 0, "confused": 0}
        uncertainty_trace = []
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(input_ids=generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :]
                
                # Compute uncertainty
                entropy = compute_entropy(next_logits[0])
                varentropy = compute_varentropy(next_logits[0])
                
                # Classify regime
                if entropy < self.tau_h:
                    regime = "confident"
                elif varentropy > self.tau_v:
                    regime = "fork"
                else:
                    regime = "confused"
                
                regime_counts[regime] += 1
                uncertainty_trace.append({
                    "entropy": entropy,
                    "varentropy": varentropy,
                    "regime": regime
                })
                
                # Get sampling params for this regime
                params = self.get_sampling_params(entropy, varentropy)
                
                # Apply temperature
                scaled_logits = next_logits / params["temperature"]
                probs = F.softmax(scaled_logits, dim=-1)
                
                # Apply top-k filtering
                if params["top_k"] > 0:
                    indices_to_remove = probs < torch.topk(probs, params["top_k"])[0][..., -1, None]
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum()
                
                # Apply top-p filtering
                if params["top_p"] < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > params["top_p"]
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    probs[indices_to_remove] = 0
                    probs = probs / probs.sum()
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
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
            "regime_counts": regime_counts,
            "uncertainty_trace": uncertainty_trace
        }


def run_entropix_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                            seed: int = 42, max_problems: int = None):
    """Run Entropix experiment."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"Running Entropix on {dataset_name} with seed {seed}")
    
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
    
    # Initialize sampler
    sampler = EntropixSampler(model, tokenizer, tau_h=2.5, tau_v=1.5)
    
    results = []
    correct = 0
    total_tokens = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        
        result = sampler.generate(prompt)
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "regime_counts": result["regime_counts"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    
    output = {
        "method": "entropix",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "total_problems": len(data),
        "correct_count": correct,
        "results": results
    }
    
    # Save results
    output_path = Path(f"exp/results/entropix_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Entropix Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()
    
    run_entropix_experiment(args.dataset, args.model, args.seed, args.max_problems)
