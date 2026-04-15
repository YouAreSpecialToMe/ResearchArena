"""Run all IntrospectBench experiments efficiently."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import random
import time
import numpy as np
import torch
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.utils import set_seed, load_jsonl, save_json, save_jsonl
from shared.metrics import (compute_detection_metrics, compute_localization_metrics,
                            compute_characterization_metrics, compute_introspection_score,
                            compute_pwls, compute_dcs)


# Simple LLM evaluator that doesn't load full model for every sample
class SimpleEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
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


def format_cot(steps: List[str]) -> str:
    return "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])


def parse_detection(output: str) -> int:
    return 1 if "yes" in output.lower() else 0


def parse_localization(output: str) -> int:
    import re
    nums = re.findall(r'\d+', output)
    return int(nums[0]) if nums else 1


def parse_characterization(output: str) -> str:
    output_lower = output.lower()
    for etype in ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]:
        if etype in output_lower:
            return etype
    return "calculation"


def evaluate_sample(evaluator: SimpleEvaluator, item: Dict, use_cot: bool = False) -> Dict:
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
        # Localization
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
        
        # Characterization
        if use_cot:
            prompt = f"""Classify the error at step {loc_pred}.

{cot_text}

Error types: calculation, logic, factuality, omission, misinterpretation, premature
Error type:"""
        else:
            prompt = f"""What error type at step {loc_pred}?

{cot_text}

Type (calculation/logic/factuality/omission/misinterpretation/premature):"""
        
        char_output = evaluator.predict(prompt, max_tokens=16)
        char_pred = parse_characterization(char_output)
    else:
        loc_pred = 1
        loc_output = "N/A"
        char_pred = "calculation"
        char_output = "N/A"
    
    return {
        "problem_id": item["problem_id"],
        "domain": item["domain"],
        "has_error": item.get("has_error", False),
        "error_type": item.get("error_type"),
        "error_step": item.get("error_step"),
        "detection_pred": det_pred,
        "detection_output": det_output,
        "localization_pred": loc_pred,
        "localization_output": loc_output,
        "characterization_pred": char_pred,
        "characterization_output": char_output
    }


def evaluate_model_on_dataset(model_name: str, test_data: List[Dict], 
                               use_cot: bool = False, seed: int = 42) -> Dict:
    """Evaluate a model on the full test set."""
    set_seed(seed)
    
    print(f"\nLoading model: {model_name}")
    evaluator = SimpleEvaluator(model_name)
    
    print(f"Evaluating {len(test_data)} samples...")
    predictions = []
    start = time.time()
    
    for i, item in enumerate(test_data):
        if i % 20 == 0:
            print(f"  {i}/{len(test_data)}")
        pred = evaluate_sample(evaluator, item, use_cot)
        predictions.append(pred)
    
    runtime = time.time() - start
    
    # Compute metrics
    # Detection
    det_preds = [p["detection_pred"] for p in predictions]
    det_labels = [1 if item.get("has_error") else 0 for item in test_data]
    det_metrics = compute_detection_metrics(det_preds, det_labels)
    
    # Localization (error only)
    error_preds = [p for p in predictions if p["has_error"]]
    error_items = [item for item in test_data if item.get("has_error")]
    
    if error_preds:
        loc_preds = [p["localization_pred"] for p in error_preds]
        loc_labels = [item.get("error_step", 1) for item in error_items]
        loc_metrics = compute_localization_metrics(loc_preds, loc_labels)
        
        # Characterization
        char_preds = [p["characterization_pred"] for p in error_preds]
        char_labels = [item.get("error_type", "calculation") for item in error_items]
        error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
        char_metrics = compute_characterization_metrics(char_preds, char_labels, error_types)
    else:
        loc_metrics = {"exact_match": 0.0}
        char_metrics = {"accuracy": 0.0}
    
    # Domain-wise metrics
    domain_metrics = {}
    for domain in ["math", "logic", "commonsense", "code"]:
        domain_preds = [p for p, item in zip(predictions, test_data) if item["domain"] == domain]
        domain_items = [item for item in test_data if item["domain"] == domain]
        if domain_items:
            d_preds = [p["detection_pred"] for p in domain_preds]
            d_labels = [1 if item.get("has_error") else 0 for item in domain_items]
            d_metrics = compute_detection_metrics(d_preds, d_labels)
            domain_metrics[domain] = d_metrics
    
    # Novel metrics
    introspection_score = compute_introspection_score(
        det_metrics["accuracy"],
        loc_metrics.get("exact_match", 0),
        char_metrics.get("accuracy", 0)
    )
    
    domain_scores = [domain_metrics[d]["accuracy"] for d in domain_metrics]
    dcs = compute_dcs(domain_scores)
    
    return {
        "model": model_name,
        "use_cot": use_cot,
        "seed": seed,
        "runtime_seconds": runtime,
        "detection": det_metrics,
        "localization": loc_metrics,
        "characterization": char_metrics,
        "introspection_score": introspection_score,
        "domain_calibration_score": dcs,
        "domain_metrics": domain_metrics,
        "predictions": predictions
    }


def run_multiple_seeds(model_name: str, test_data: List[Dict], 
                       use_cot: bool = False, seeds: List[int] = [42, 43, 44]) -> Dict:
    """Run evaluation with multiple seeds and aggregate results."""
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
        result = evaluate_model_on_dataset(model_name, test_data, use_cot, seed)
        all_results.append(result)
        
        print(f"\nResults (seed {seed}):")
        print(f"  Detection Acc: {result['detection']['accuracy']:.3f}")
        print(f"  Localization EM: {result['localization'].get('exact_match', 0):.3f}")
        print(f"  Char Acc: {result['characterization'].get('accuracy', 0):.3f}")
        print(f"  IS: {result['introspection_score']:.3f}")
    
    # Aggregate across seeds
    def mean_std(values):
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}
    
    aggregated = {
        "model": model_name,
        "use_cot": use_cot,
        "seeds": seeds,
        "detection": {
            "accuracy": mean_std([r["detection"]["accuracy"] for r in all_results]),
            "f1": mean_std([r["detection"]["f1"] for r in all_results])
        },
        "localization": {
            "exact_match": mean_std([r["localization"].get("exact_match", 0) for r in all_results]),
            "mae": mean_std([r["localization"].get("mae", 0) for r in all_results])
        },
        "characterization": {
            "accuracy": mean_std([r["characterization"].get("accuracy", 0) for r in all_results]),
            "f1_macro": mean_std([r["characterization"].get("f1_macro", 0) for r in all_results])
        },
        "introspection_score": mean_std([r["introspection_score"] for r in all_results]),
        "domain_calibration_score": mean_std([r["domain_calibration_score"] for r in all_results]),
        "domain_metrics": all_results[0]["domain_metrics"],  # Domain metrics from first seed
        "individual_results": all_results
    }
    
    return aggregated


def main():
    # Load test data
    test_data = load_jsonl("data/processed/test.jsonl")
    print(f"Loaded {len(test_data)} test samples")
    
    # Models to evaluate
    models = [
        ("Qwen/Qwen2.5-7B-Instruct", "qwen25_7b"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama31_8b"),
    ]
    
    all_results = {}
    
    for model_name, model_id in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_id}")
        print(f"{'='*70}")
        
        # Run without CoT
        print("\n--- Direct Prompting ---")
        results_direct = run_multiple_seeds(model_name, test_data, use_cot=False)
        all_results[f"{model_id}_direct"] = results_direct
        
        # Run with CoT
        print("\n--- CoT Prompting ---")
        results_cot = run_multiple_seeds(model_name, test_data, use_cot=True)
        all_results[f"{model_id}_cot"] = results_cot
    
    # Save all results
    os.makedirs("outputs/main_results", exist_ok=True)
    
    # Remove predictions for summary
    summary = {k: {key: val for key, val in v.items() if key != "individual_results"} 
               for k, v in all_results.items()}
    save_json(summary, "outputs/main_results/all_models_summary.json")
    
    # Save full results
    save_json(all_results, "outputs/main_results/all_models_full.json")
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    for model_id, results in summary.items():
        is_mean = results["introspection_score"]["mean"]
        is_std = results["introspection_score"]["std"]
        det_mean = results["detection"]["accuracy"]["mean"]
        loc_mean = results["localization"]["exact_match"]["mean"]
        char_mean = results["characterization"]["accuracy"]["mean"]
        print(f"\n{model_id}:")
        print(f"  IS: {is_mean:.3f} ± {is_std:.3f}")
        print(f"  Detection: {det_mean:.3f}")
        print(f"  Localization: {loc_mean:.3f}")
        print(f"  Characterization: {char_mean:.3f}")
    
    print("\nResults saved to outputs/main_results/")


if __name__ == "__main__":
    main()
