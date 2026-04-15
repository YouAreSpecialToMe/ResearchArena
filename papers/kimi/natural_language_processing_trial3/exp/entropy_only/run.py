"""Entropy-Only baseline (ESR variant without varentropy)."""

import torch
from typing import List, Dict, Any
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model, compute_entropy
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


class EntropyOnlyGenerator:
    """Entropy-only trigger for revision (no varentropy)."""
    
    def __init__(
        self,
        model,
        tokenizer,
        tau_h: float = 2.5,
        r_max: int = 3,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.r_max = r_max
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt: str) -> Dict[str, Any]:
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
                
                entropy = compute_entropy(next_logits[0])
                
                # Trigger revision on high entropy only (no varentropy check)
                if entropy > self.tau_h and revision_count < self.r_max and step > 10:
                    # Generate revision
                    current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    reasoning_so_far = current_text[len(prompt):]
                    
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
        
        return {
            "output": output_text,
            "total_tokens": total_tokens,
            "revision_count": revision_count
        }


def run_entropy_only_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                                seed: int = 42, max_problems: int = None):
    """Run entropy-only experiment."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Running Entropy-Only on {dataset_name} with seed {seed}")
    
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
    
    # Initialize generator
    generator = EntropyOnlyGenerator(model, tokenizer, tau_h=2.5, r_max=3)
    
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
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "revisions": result["revision_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    avg_revisions = total_revisions / len(data)
    
    output = {
        "method": "entropy_only",
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "avg_revisions": avg_revisions,
        "total_problems": len(data),
        "correct_count": correct,
        "results": results
    }
    
    # Save results
    output_path = Path(f"exp/results/entropy_only_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Entropy-Only Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}, Avg Revisions={avg_revisions:.2f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()
    
    run_entropy_only_experiment(args.dataset, args.model, args.seed, args.max_problems)
