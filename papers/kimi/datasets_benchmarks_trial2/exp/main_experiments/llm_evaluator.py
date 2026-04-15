"""LLM-based evaluation for IntrospectBench."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import re
import time
import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from shared.utils import set_seed, load_jsonl, save_json, save_jsonl
from shared.metrics import (compute_detection_metrics, compute_localization_metrics,
                            compute_characterization_metrics, compute_introspection_score,
                            compute_pwls)


class LLMEvaluator:
    """Evaluate LLMs on IntrospectBench tasks."""
    
    def __init__(self, model_name: str, device: str = "cuda", load_in_4bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.load_in_4bit = load_in_4bit
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self.model.eval()
        print(f"Model loaded. Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def generate(self, prompt: str, max_new_tokens: int = 256, 
                 temperature: float = 0.0) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                max_length=2048).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text.strip()
    
    def evaluate_detection(self, problem: Dict, cot_steps: List[str],
                          use_cot_prompt: bool = False) -> Tuple[int, str]:
        """Evaluate error detection. Returns (has_error_prediction, raw_output)."""
        cot_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(cot_steps)])
        
        if use_cot_prompt:
            prompt = f"""You are evaluating a reasoning chain for errors.

Problem: {problem['question']}

Reasoning:
{cot_text}

Review the reasoning step by step. First, analyze each step for correctness. Then conclude: does the reasoning contain errors?

Format your response as:
Analysis: [your step-by-step review]
Answer: [YES or NO]"""
        else:
            prompt = f"""Review the following reasoning. Does it contain any errors? Answer only YES or NO.

Problem: {problem['question']}

Reasoning:
{cot_text}

Answer (YES or NO):"""
        
        output = self.generate(prompt, max_new_tokens=128, temperature=0.0)
        
        # Parse output
        has_error = 1 if "yes" in output.lower() else 0
        
        return has_error, output
    
    def evaluate_localization(self, problem: Dict, cot_steps: List[str],
                             use_cot_prompt: bool = False) -> Tuple[int, str]:
        """Evaluate error localization. Returns (step_number, raw_output)."""
        cot_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(cot_steps)])
        
        if use_cot_prompt:
            prompt = f"""The reasoning below contains an error. Review the reasoning and identify where errors occur by checking each step.

Problem: {problem['question']}

Reasoning:
{cot_text}

Identify the step number (1-indexed) where the first error occurs.

Format your response as:
Analysis: [your review]
Answer: [step number]"""
        else:
            prompt = f"""The reasoning below contains an error. Identify the step number (1-indexed) where the first error occurs. Answer with only the step number.

Problem: {problem['question']}

Reasoning:
{cot_text}

Step number:"""
        
        output = self.generate(prompt, max_new_tokens=64, temperature=0.0)
        
        # Parse step number
        step_num = 1
        numbers = re.findall(r'\d+', output)
        if numbers:
            step_num = int(numbers[0])
        
        return step_num, output
    
    def evaluate_characterization(self, problem: Dict, cot_steps: List[str],
                                  error_step: int,
                                  use_cot_prompt: bool = False) -> Tuple[str, str]:
        """Evaluate error characterization. Returns (error_type, raw_output)."""
        cot_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(cot_steps)])
        
        error_types = "calculation, logic, factuality, omission, misinterpretation, premature"
        
        if use_cot_prompt:
            prompt = f"""Analyze the error in the reasoning below.

Problem: {problem['question']}

Reasoning:
{cot_text}

The error is at step {error_step}. Classify the error type from: {error_types}.

Format your response as:
Analysis: [your reasoning]
Answer: [type]"""
        else:
            prompt = f"""The reasoning below has an error at step {error_step}. Classify the error type: {error_types}. Answer with only the type.

Problem: {problem['question']}

Reasoning:
{cot_text}

Error type:"""
        
        output = self.generate(prompt, max_new_tokens=64, temperature=0.0)
        
        # Parse error type
        output_lower = output.lower()
        type_mapping = {
            "calculation": "calculation",
            "logic": "logic",
            "factuality": "factuality",
            "omission": "omission",
            "misinterpretation": "misinterpretation",
            "premature": "premature"
        }
        
        error_type = "calculation"  # default
        for key in type_mapping:
            if key in output_lower:
                error_type = type_mapping[key]
                break
        
        return error_type, output
    
    def evaluate_dataset(self, test_data: List[Dict], 
                        use_cot_prompt: bool = False,
                        max_samples: Optional[int] = None) -> Dict:
        """Evaluate on full dataset."""
        if max_samples:
            test_data = test_data[:max_samples]
        
        predictions = []
        
        for i, item in enumerate(test_data):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(test_data)}...")
            
            # Get the appropriate CoT (correct or corrupted)
            if item.get("has_error"):
                cot_steps = item.get("corrupted_steps", item.get("correct_steps", []))
            else:
                cot_steps = item.get("correct_steps", [])
            
            # Task 1: Detection
            detection_pred, detection_raw = self.evaluate_detection(
                item, cot_steps, use_cot_prompt
            )
            
            # Task 2: Localization (only for error samples)
            if item.get("has_error"):
                localization_pred, localization_raw = self.evaluate_localization(
                    item, cot_steps, use_cot_prompt
                )
                # Task 3: Characterization
                char_pred, char_raw = self.evaluate_characterization(
                    item, cot_steps, localization_pred, use_cot_prompt
                )
            else:
                localization_pred = 1
                localization_raw = "N/A"
                char_pred = "calculation"
                char_raw = "N/A"
            
            predictions.append({
                "problem_id": item["problem_id"],
                "domain": item["domain"],
                "has_error": item.get("has_error", False),
                "error_type": item.get("error_type"),
                "error_step": item.get("error_step"),
                "detection_pred": detection_pred,
                "detection_raw": detection_raw,
                "localization_pred": localization_pred,
                "localization_raw": localization_raw,
                "characterization_pred": char_pred,
                "characterization_raw": char_raw
            })
        
        return self.compute_all_metrics(predictions, test_data)
    
    def compute_all_metrics(self, predictions: List[Dict], 
                           test_data: List[Dict]) -> Dict:
        """Compute all evaluation metrics."""
        # Detection metrics (all samples)
        detection_preds = [p["detection_pred"] for p in predictions]
        detection_labels = [1 if item.get("has_error", False) else 0 for item in test_data]
        detection_metrics = compute_detection_metrics(detection_preds, detection_labels)
        
        # Localization metrics (error samples only)
        error_predictions = [p for p in predictions if p["has_error"]]
        error_test_data = [item for item in test_data if item.get("has_error")]
        
        if error_predictions:
            localization_preds = [p["localization_pred"] for p in error_predictions]
            localization_labels = [item.get("error_step", 1) for item in error_test_data]
            localization_metrics = compute_localization_metrics(localization_preds, localization_labels)
            
            # Characterization metrics
            char_preds = [p["characterization_pred"] for p in error_predictions]
            char_labels = [item.get("error_type", "calculation") for item in error_test_data]
            error_types = ["calculation", "logic", "factuality", "omission", "misinterpretation", "premature"]
            characterization_metrics = compute_characterization_metrics(char_preds, char_labels, error_types)
        else:
            localization_metrics = {"exact_match": 0.0, "mae": 0.0}
            characterization_metrics = {"accuracy": 0.0, "f1_macro": 0.0}
        
        # Compute novel metrics
        introspection_score = compute_introspection_score(
            detection_metrics["accuracy"],
            localization_metrics.get("exact_match", 0.0),
            characterization_metrics.get("accuracy", 0.0)
        )
        
        # Domain-wise analysis
        domain_results = {}
        for domain in ["math", "logic", "commonsense", "code"]:
            domain_preds = [p for p, item in zip(predictions, test_data) if item["domain"] == domain]
            domain_data = [item for item in test_data if item["domain"] == domain]
            
            if domain_data:
                domain_det_preds = [p["detection_pred"] for p in domain_preds]
                domain_det_labels = [1 if item.get("has_error", False) else 0 for item in domain_data]
                domain_det = compute_detection_metrics(domain_det_preds, domain_det_labels)
                domain_results[domain] = {"detection_accuracy": domain_det["accuracy"]}
        
        return {
            "detection": detection_metrics,
            "localization": localization_metrics,
            "characterization": characterization_metrics,
            "introspection_score": introspection_score,
            "domain_results": domain_results,
            "predictions": predictions
        }


def evaluate_model(model_name: str, model_id: str, use_cot: bool = False,
                   load_in_4bit: bool = False, max_samples: Optional[int] = None) -> Dict:
    """Evaluate a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id} (CoT={use_cot})")
    print(f"{'='*60}")
    
    # Load test data
    test_data = load_jsonl("data/processed/test.jsonl")
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"Test samples: {len(test_data)}")
    
    # Create evaluator
    evaluator = LLMEvaluator(model_name, load_in_4bit=load_in_4bit)
    evaluator.load_model()
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_dataset(test_data, use_cot_prompt=use_cot)
    runtime = time.time() - start_time
    
    results["runtime_seconds"] = runtime
    results["model"] = model_id
    results["use_cot"] = use_cot
    
    print(f"\nResults for {model_id}:")
    print(f"  Detection Accuracy: {results['detection']['accuracy']:.3f}")
    print(f"  Detection F1: {results['detection']['f1']:.3f}")
    print(f"  Localization Exact Match: {results['localization'].get('exact_match', 0):.3f}")
    print(f"  Characterization Accuracy: {results['characterization'].get('accuracy', 0):.3f}")
    print(f"  Introspection Score: {results['introspection_score']:.3f}")
    print(f"  Runtime: {runtime:.1f}s")
    
    # Save results
    output_dir = "outputs/main_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    save_jsonl(results["predictions"], 
               f"{output_dir}/{model_id}_predictions.jsonl")
    
    # Save metrics (without predictions to keep file small)
    metrics_only = {k: v for k, v in results.items() if k != "predictions"}
    save_json(metrics_only, f"{output_dir}/{model_id}_metrics.json")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    model_id = args.model_id or args.model.replace("/", "_")
    
    evaluate_model(
        model_name=args.model,
        model_id=f"{model_id}_cot" if args.cot else model_id,
        use_cot=args.cot,
        load_in_4bit=getattr(args, '4bit', False),
        max_samples=args.max_samples
    )
