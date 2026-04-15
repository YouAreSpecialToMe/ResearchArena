"""HALT-CoT: Entropy-based Early Stopping baseline implementation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model, compute_entropy
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


class HALTCoT:
    """HALT-CoT: Stop early when entropy is consistently low."""
    
    def __init__(
        self,
        model,
        tokenizer,
        tau_low: float = 1.0,
        patience: int = 10,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_low = tau_low
        self.patience = patience
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate with early stopping based on low entropy."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = inputs["input_ids"]
        total_tokens = 0
        low_entropy_count = 0
        stopped_early = False
        uncertainty_trace = []
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                outputs = self.model(input_ids=generated_ids, return_dict=True)
                next_logits = outputs.logits[:, -1, :]
                
                # Compute entropy
                entropy = compute_entropy(next_logits[0])
                uncertainty_trace.append({"entropy": entropy})
                
                # Check for low entropy (confident)
                if entropy < self.tau_low:
                    low_entropy_count += 1
                    if low_entropy_count >= self.patience:
                        # Model is confident for consecutive steps, stop early
                        stopped_early = True
                        break
                else:
                    low_entropy_count = 0
                
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
        
        return {
            "output": output_text,
            "total_tokens": total_tokens,
            "stopped_early": stopped_early,
            "uncertainty_trace": uncertainty_trace
        }


def run_halt_cot_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                            seed: int = 42, max_problems: int = None):
    """Run HALT-CoT experiment."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Running HALT-CoT on {dataset_name} with seed {seed}")
    
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
    
    # Initialize HALT-CoT
    halt_cot = HALTCoT(model, tokenizer, tau_low=1.0, patience=10)
    
    results = []
    correct = 0
    total_tokens = 0
    early_stop_count = 0
    
    for i, item in enumerate(data):
        prompt = create_cot_prompt(item["question"])
        
        result = halt_cot.generate(prompt)
        predicted = extract_numeric_answer(result["output"])
        actual = item["answer"]
        
        is_correct = False
        if predicted is not None and actual is not None:
            is_correct = abs(predicted - actual) < 1e-3
        
        if is_correct:
            correct += 1
        total_tokens += result["total_tokens"]
        if result["stopped_early"]:
            early_stop_count += 1
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "stopped_early": result["stopped_early"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    early_stop_rate = early_stop_count / len(data)
    
    output = {
        "method": "halt_cot",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "early_stop_rate": early_stop_rate,
        "total_problems": len(data),
        "correct_count": correct,
        "results": results
    }
    
    # Save results
    output_path = Path(f"exp/results/halt_cot_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"HALT-CoT Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}, Early Stop Rate={early_stop_rate:.3f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()
    
    run_halt_cot_experiment(args.dataset, args.model, args.seed, args.max_problems)
