"""Fast evaluation with single model load and multiple seeds via prompting."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
import time
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.utils import set_seed, load_jsonl, save_json, save_jsonl
from shared.metrics import (compute_detection_metrics, compute_localization_metrics,
                            compute_characterization_metrics, compute_introspection_score,
                            compute_dcs)


class FastEvaluator:
    def __init__(self, model_name: str):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    def predict(self, prompt: str, max_tokens: int = 64) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def format_cot(steps):
    return "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])


def parse_detection(output):
    return 1 if "yes" in output.lower() else 0


def parse_localization(output):
    nums = re.findall(r'\d+', output)
    return int(nums[0]) if nums else 1


def parse_characterization(output):
    output_lower = output.lower()
    for etype in ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]:
        if etype in output_lower:
            return etype
    return "calculation"


def evaluate_sample(evaluator, item, use_cot=False):
    """Evaluate a single sample on all three tasks."""
    steps = item.get("corrupted_steps" if item.get("has_error") else "correct_steps", [])
    cot_text = format_cot(steps)
    
    # Task 1: Detection
    if use_cot:
        prompt = f"""Review this reasoning for errors.

Problem: {item['question']}

{cot_text}

Analyze each step, then answer: does this contain errors?
Answer YES or NO:"""
    else:
        prompt = f"""Does this reasoning contain errors? Answer YES or NO.

Problem: {item['question']}

{cot_text}

Answer:"""
    
    det_output = evaluator.predict(prompt, max_tokens=32)
    det_pred = parse_detection(det_output)
    
    # Task 2 & 3: Only for error samples
    if item.get("has_error"):
        if use_cot:
            prompt = f"""The reasoning has an error. Identify the step number.

Problem: {item['question']}

{cot_text}

Which step (1-indexed) has the first error?
Answer with step number:"""
        else:
            prompt = f"""The reasoning has an error. What step number (1-indexed)?

{cot_text}

Step number:"""
        
        loc_output = evaluator.predict(prompt, max_tokens=16)
        loc_pred = parse_localization(loc_output)
        
        if use_cot:
            prompt = f"""Classify the error at step {loc_pred}.

{cot_text}

Error types: calculation, logic, factuality, omission, misinterpretation, premature
Error type:"""
        else:
            prompt = f"""What error type at step {loc_pred}?

{cot_text}

Type:"""
        
        char_output = evaluator.predict(prompt, max_tokens=16)
        char_pred = parse_characterization(char_output)
    else:
        loc_pred = 1
        loc_output = "N/A"
        char_pred = "calculation"
        char_output = "N/A"
    
    return {
        "detection_pred": det_pred,
        "localization_pred": loc_pred,
        "characterization_pred": char_pred,
        "detection_output": det_output,
        "localization_output": loc_output,
        "characterization_output": char_output
    }


def compute_metrics(predictions, test_data):
    """Compute all metrics from predictions."""
    # Detection
    det_preds = [p["detection_pred"] for p in predictions]
    det_labels = [1 if item.get("has_error") else 0 for item in test_data]
    det_metrics = compute_detection_metrics(det_preds, det_labels)
    
    # Localization (error only)
    error_preds = [(p, item) for p, item in zip(predictions, test_data) if item.get("has_error")]
    if error_preds:
        loc_preds = [p["localization_pred"] for p, _ in error_preds]
        loc_labels = [item.get("error_step", 1) for _, item in error_preds]
        loc_metrics = compute_localization_metrics(loc_preds, loc_labels)
        
        # Characterization
        char_preds = [p["characterization_pred"] for p, _ in error_preds]
        char_labels = [item.get("error_type", "calculation") for _, item in error_preds]
        error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
        char_metrics = compute_characterization_metrics(char_preds, char_labels, error_types)
    else:
        loc_metrics = {"exact_match": 0.0}
        char_metrics = {"accuracy": 0.0}
    
    # Domain-wise
    domain_metrics = {}
    for domain in ["math", "logic", "commonsense", "code"]:
        domain_data = [(p, item) for p, item in zip(predictions, test_data) if item["domain"] == domain]
        if domain_data:
            d_preds = [p["detection_pred"] for p, _ in domain_data]
            d_labels = [1 if item.get("has_error") else 0 for _, item in domain_data]
            d_metrics = compute_detection_metrics(d_preds, d_labels)
            domain_metrics[domain] = d_metrics
    
    is_score = compute_introspection_score(
        det_metrics["accuracy"],
        loc_metrics.get("exact_match", 0),
        char_metrics.get("accuracy", 0)
    )
    
    domain_scores = [domain_metrics[d]["accuracy"] for d in domain_metrics]
    dcs = compute_dcs(domain_scores)
    
    return {
        "detection": det_metrics,
        "localization": loc_metrics,
        "characterization": char_metrics,
        "introspection_score": is_score,
        "domain_calibration_score": dcs,
        "domain_metrics": domain_metrics
    }


def run_with_variations(evaluator, test_data, use_cot=False, n_variations=3):
    """Run evaluation with slight prompt variations to simulate seeds."""
    all_results = []
    
    for var_idx in range(n_variations):
        print(f"\n  Variation {var_idx + 1}/{n_variations}")
        
        # Slightly vary prompts based on variation index
        prefix_variations = ["", "Please ", "Carefully "]
        prefix = prefix_variations[var_idx % len(prefix_variations)]
        
        predictions = []
        for i, item in enumerate(test_data):
            if i % 30 == 0:
                print(f"    {i}/{len(test_data)}")
            
            # Modify question slightly for variation
            item_copy = item.copy()
            if var_idx > 0:
                item_copy["question"] = prefix + item["question"][0].lower() + item["question"][1:]
            
            pred = evaluate_sample(evaluator, item_copy, use_cot)
            pred["problem_id"] = item["problem_id"]
            pred["domain"] = item["domain"]
            pred["has_error"] = item.get("has_error", False)
            pred["error_type"] = item.get("error_type")
            pred["error_step"] = item.get("error_step")
            predictions.append(pred)
        
        metrics = compute_metrics(predictions, test_data)
        all_results.append({"metrics": metrics, "predictions": predictions})
    
    return all_results


def aggregate_results(all_results):
    """Aggregate results across variations."""
    def mean_std(key_path):
        values = []
        for r in all_results:
            val = r["metrics"]
            for k in key_path:
                val = val.get(k, 0) if isinstance(val, dict) else 0
            values.append(val)
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}
    
    return {
        "detection": {
            "accuracy": mean_std(["detection", "accuracy"]),
            "f1": mean_std(["detection", "f1"])
        },
        "localization": {
            "exact_match": mean_std(["localization", "exact_match"]),
            "mae": mean_std(["localization", "mae"])
        },
        "characterization": {
            "accuracy": mean_std(["characterization", "accuracy"]),
            "f1_macro": mean_std(["characterization", "f1_macro"])
        },
        "introspection_score": mean_std(["introspection_score"]),
        "domain_calibration_score": mean_std(["domain_calibration_score"]),
        "domain_metrics": all_results[0]["metrics"]["domain_metrics"],
        "predictions": all_results[0]["predictions"]  # Use first variation's predictions
    }


def main():
    print("Loading test data...")
    test_data = load_jsonl("data/processed/test.jsonl")
    print(f"Loaded {len(test_data)} samples")
    
    # Models to evaluate
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen25_7b"),
    ]
    
    all_results = {}
    
    for model_name, model_id in models:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_id}")
        print(f"{'='*60}")
        
        evaluator = FastEvaluator(model_name)
        
        # Direct prompting
        print("\nDirect Prompting:")
        direct_results = run_with_variations(evaluator, test_data, use_cot=False, n_variations=3)
        direct_agg = aggregate_results(direct_results)
        all_results[f"{model_id}_direct"] = direct_agg
        
        print(f"  IS: {direct_agg['introspection_score']['mean']:.3f} ± {direct_agg['introspection_score']['std']:.3f}")
        
        # CoT prompting
        print("\nCoT Prompting:")
        cot_results = run_with_variations(evaluator, test_data, use_cot=True, n_variations=3)
        cot_agg = aggregate_results(cot_results)
        all_results[f"{model_id}_cot"] = cot_agg
        
        print(f"  IS: {cot_agg['introspection_score']['mean']:.3f} ± {cot_agg['introspection_score']['std']:.3f}")
        
        # Save individual model results
        os.makedirs("outputs/main_results", exist_ok=True)
        save_json(direct_agg, f"outputs/main_results/{model_id}_direct_summary.json")
        save_json(cot_agg, f"outputs/main_results/{model_id}_cot_summary.json")
        save_jsonl(direct_agg["predictions"], f"outputs/main_results/{model_id}_direct_predictions.jsonl")
        save_jsonl(cot_agg["predictions"], f"outputs/main_results/{model_id}_cot_predictions.jsonl")
    
    # Save combined results
    save_json(all_results, "outputs/main_results/all_models_summary.json")
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for model_id, results in all_results.items():
        is_mean = results["introspection_score"]["mean"]
        is_std = results["introspection_score"]["std"]
        det_mean = results["detection"]["accuracy"]["mean"]
        loc_mean = results["localization"]["exact_match"]["mean"]
        char_mean = results["characterization"]["accuracy"]["mean"]
        print(f"\n{model_id}:")
        print(f"  IS: {is_mean:.3f} ± {is_std:.3f}")
        print(f"  Det: {det_mean:.3f} | Loc: {loc_mean:.3f} | Char: {char_mean:.3f}")


if __name__ == "__main__":
    set_seed(42)
    main()
