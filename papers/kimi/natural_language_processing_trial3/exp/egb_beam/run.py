"""EGB-style Entropy-Gated Beam Search baseline implementation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.models import load_model, compute_entropy, extract_answer
from shared.data_loader import load_gsm8k, create_cot_prompt, extract_numeric_answer
from shared.metrics import compute_metrics


class EGBBeamSearch:
    """Entropy-Gated Beam Search with K=3 beams."""
    
    def __init__(
        self,
        model,
        tokenizer,
        tau_h: float = 2.5,
        k_beams: int = 3,
        max_new_tokens: int = 1024,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tau_h = tau_h
        self.k_beams = k_beams
        self.max_new_tokens = max_new_tokens
        self.device = device
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate with entropy-gated beam search."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Initialize beams: each beam is (token_ids, score, branched)
        beams = [(inputs["input_ids"].clone(), 0.0, False)]
        total_tokens = 0
        branch_count = 0
        
        with torch.no_grad():
            for step in range(self.max_new_tokens):
                new_beams = []
                
                for beam_ids, beam_score, branched in beams:
                    if branched:
                        # This beam already ended (reached answer)
                        new_beams.append((beam_ids, beam_score, True))
                        continue
                    
                    # Get next token distribution
                    outputs = self.model(input_ids=beam_ids, return_dict=True)
                    next_logits = outputs.logits[:, -1, :]
                    
                    # Compute entropy
                    entropy = compute_entropy(next_logits[0])
                    
                    # Get top-k tokens
                    log_probs = F.log_softmax(next_logits, dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, self.k_beams, dim=-1)
                    
                    # Branch if entropy is high
                    if entropy > self.tau_h and len(beams) < self.k_beams * 2:
                        branch_count += 1
                        # Create multiple branches with top-k tokens
                        for k in range(min(self.k_beams, 3)):  # Max 3 branches
                            next_token = topk_indices[0, k:k+1].unsqueeze(0)
                            new_ids = torch.cat([beam_ids, next_token], dim=-1)
                            new_score = beam_score + topk_log_probs[0, k].item()
                            new_beams.append((new_ids, new_score, False))
                    else:
                        # Greedy continuation
                        next_token = topk_indices[0, 0:1].unsqueeze(0)
                        new_ids = torch.cat([beam_ids, next_token], dim=-1)
                        new_score = beam_score + topk_log_probs[0, 0].item()
                        
                        # Check for EOS
                        if next_token.item() == self.tokenizer.eos_token_id:
                            new_beams.append((new_ids, new_score, True))
                        else:
                            new_beams.append((new_ids, new_score, False))
                    
                    total_tokens += 1
                
                # Keep top-k beams by score
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:self.k_beams]
                
                # Check if all beams ended
                if all(branched for _, _, branched in beams):
                    break
                
                if total_tokens > self.max_new_tokens * self.k_beams:
                    break
        
        # Select best beam
        best_beam = max(beams, key=lambda x: x[1])
        best_ids = best_beam[0]
        
        generated_text = self.tokenizer.decode(best_ids[0], skip_special_tokens=True)
        output_text = generated_text[len(prompt):]
        
        return {
            "output": output_text,
            "total_tokens": total_tokens,
            "branch_count": branch_count,
            "beam_scores": [b[1] for b in beams]
        }


def run_egb_experiment(dataset_name: str = "gsm8k", model_name: str = "Qwen/Qwen3-1.7B", 
                       seed: int = 42, max_problems: int = None):
    """Run EGB beam search experiment."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Running EGB Beam Search on {dataset_name} with seed {seed}")
    
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
    generator = EGBBeamSearch(model, tokenizer, tau_h=2.5, k_beams=3)
    
    results = []
    correct = 0
    total_tokens = 0
    
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
        
        results.append({
            "question_idx": i,
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "tokens": result["total_tokens"],
            "branch_count": result["branch_count"]
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(data)}, Accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / len(data)
    avg_tokens = total_tokens / len(data)
    
    output = {
        "method": "egb_beam_search",
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
    output_path = Path(f"exp/results/egb_{dataset_name}_seed{seed}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"EGB Results: Accuracy={accuracy:.3f}, Avg Tokens={avg_tokens:.1f}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_problems", type=int, default=None)
    args = parser.parse_args()
    
    run_egb_experiment(args.dataset, args.model, args.seed, args.max_problems)
