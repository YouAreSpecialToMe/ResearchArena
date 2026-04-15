"""Best-of-N Sampling with Majority Voting baseline implementation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer


class BestOfN:
    """Best-of-N: Sample N=4 and select via majority voting."""
    
    def __init__(
        self,
        model,
        tokenizer,
        n_samples: int = 4,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate_single(self, prompt: str) -> Dict[str, Any]:
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
        
        return {
            "output": output_text,
            "tokens": num_tokens
        }
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate N samples and select via majority voting."""
        samples = []
        total_tokens = 0
        
        for i in range(self.n_samples):
            sample = self.generate_single(prompt)
            samples.append(sample)
            total_tokens += sample["tokens"]
        
        # Extract answers from all samples
        answers = []
        for sample in samples:
            answer = extract_numeric_answer(sample["output"])
            answers.append(answer)
        
        # Majority voting (most frequent non-None answer)
        valid_answers = [a for a in answers if a is not None]
        if valid_answers:
            answer_counts = Counter(valid_answers)
            best_answer = answer_counts.most_common(1)[0][0]
        else:
            best_answer = None
        
        # Find the sample with the best answer
        best_sample_idx = 0
        for i, answer in enumerate(answers):
            if answer == best_answer:
                best_sample_idx = i
                break
        
        return {
            "output": samples[best_sample_idx]["output"],
            "total_tokens": total_tokens,
            "samples": samples,
            "answers": answers,
            "best_answer": best_answer,
            "agreement_count": answer_counts.get(best_answer, 0) if valid_answers else 0
        }


def run_bestofn_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                           seed: int = 42, max_problems: int = None):
    """Run Best-of-N experiment."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Running Best-of-N on {dataset_name} with seed {seed}")
    
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
    
    # Initialize Best-of-N
    bon = BestOfN(model, tokenizer, n_samples=4, temperature=0.7)
    
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
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "agreement": result["agreement_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    
    output = {
        "method": "bestofn",
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
    output_path = Path(f"exp/results/bestofn_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Best-of-N Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()
    
    run_bestofn_experiment(args.dataset, args.model, args.seed, args.max_problems)
