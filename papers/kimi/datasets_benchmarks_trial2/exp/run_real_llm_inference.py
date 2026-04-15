#!/usr/bin/env python3
"""
Run ACTUAL LLM inference on MMLU and GSM8K datasets.
This generates real model responses for IRT calibration and evaluation.
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.data_loader import load_questions

# Model configurations - 12 models as specified in proposal
MODELS_CONFIG = [
    # Smaller models first (faster)
    {'name': 'Qwen2.5-0.5B', 'hf_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'size_gb': 1.0},
    {'name': 'Qwen2.5-1.8B', 'hf_id': 'Qwen/Qwen2.5-1.8B-Instruct', 'size_gb': 3.6},
    {'name': 'Gemma-2-2B', 'hf_id': 'google/gemma-2-2b-it', 'size_gb': 4.0},
    {'name': 'Gemma-2-9B', 'hf_id': 'google/gemma-2-9b-it', 'size_gb': 18.0, 'load_4bit': True},
    {'name': 'Llama-3.1-8B', 'hf_id': 'meta-llama/Llama-3.1-8B-Instruct', 'size_gb': 16.0, 'load_4bit': True},
    {'name': 'Mistral-7B', 'hf_id': 'mistralai/Mistral-7B-Instruct-v0.3', 'size_gb': 14.0, 'load_4bit': True},
    {'name': 'Qwen2.5-7B', 'hf_id': 'Qwen/Qwen2.5-7B-Instruct', 'size_gb': 14.0, 'load_4bit': True},
    {'name': 'Qwen2.5-14B', 'hf_id': 'Qwen/Qwen2.5-14B-Instruct', 'size_gb': 28.0, 'load_4bit': True},
    {'name': 'Phi-4', 'hf_id': 'microsoft/Phi-4', 'size_gb': 28.0, 'load_4bit': True},
    {'name': 'Gemma-2-27B', 'hf_id': 'google/gemma-2-27b-it', 'size_gb': 54.0, 'load_4bit': True},
    {'name': 'Qwen2.5-32B', 'hf_id': 'Qwen/Qwen2.5-32B-Instruct', 'size_gb': 64.0, 'load_4bit': True},
    {'name': 'Llama-3.1-70B', 'hf_id': 'meta-llama/Llama-3.1-70B-Instruct', 'size_gb': 140.0, 'load_4bit': True},
]


def format_mmlu_prompt(question, choices, few_shot_examples=None):
    """Format MMLU question with 5-shot examples."""
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for ex in few_shot_examples[:5]:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += f"Answer: {chr(65+ex['answer'])}\n\n"
    
    # Add the target question
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    
    return prompt


def format_gsm8k_prompt(question, few_shot_examples=None):
    """Format GSM8K question with 8-shot examples."""
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for ex in few_shot_examples[:8]:
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    
    # Add the target question
    prompt += f"Question: {question}\nAnswer:"
    
    return prompt


def extract_mmlu_answer(text):
    """Extract multiple choice answer (A, B, C, or D) from model output."""
    text = text.strip()
    
    # Look for answer patterns
    import re
    
    # Pattern 1: "Answer is X" or "Answer: X"
    match = re.search(r'[Aa]nswer[:\s]+([A-D])', text)
    if match:
        return ord(match.group(1).upper()) - ord('A')
    
    # Pattern 2: Just the letter at the start
    if text and text[0].upper() in 'ABCD':
        return ord(text[0].upper()) - ord('A')
    
    # Pattern 3: Look for "X." or "X)" patterns
    match = re.search(r'\b([A-D])[\.\)]', text)
    if match:
        return ord(match.group(1).upper()) - ord('A')
    
    # Default: check first occurrence of A, B, C, or D
    for char in text.upper():
        if char in 'ABCD':
            return ord(char) - ord('A')
    
    return -1  # Invalid


def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K output."""
    import re
    
    # Look for "####" pattern which GSM8K uses to mark final answer
    match = re.search(r'####\s*(-?\d[\d,]*)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for numbers in the text
    numbers = re.findall(r'-?\d[\d,]*', text)
    if numbers:
        # Return the last number found (usually the answer)
        return numbers[-1].replace(',', '')
    
    return None


def load_model_and_tokenizer(model_config, device='cuda'):
    """Load model and tokenizer with appropriate settings."""
    print(f"\nLoading {model_config['name']} ({model_config['hf_id']})...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['hf_id'], trust_remote_code=True)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    
    # Use 4-bit quantization for large models
    if model_config.get('load_4bit', False):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs['quantization_config'] = bnb_config
        print("  Using 4-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['hf_id'],
        **load_kwargs
    )
    
    print(f"  Model loaded on {model.device if hasattr(model, 'device') else 'multiple devices'}")
    return model, tokenizer


def run_inference_on_dataset(model, tokenizer, dataset_name, questions, max_new_tokens=10):
    """Run inference on a dataset."""
    responses = []
    
    print(f"\nRunning inference on {len(questions)} {dataset_name} questions...")
    
    for i, q in enumerate(tqdm(questions, desc=f"Processing {dataset_name}")):
        if dataset_name == 'mmlu':
            prompt = format_mmlu_prompt(q['question'], q['choices'])
            max_tokens = 10
        else:  # gsm8k
            prompt = format_gsm8k_prompt(q['question'])
            max_tokens = 256
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,  # Greedy decoding for deterministic results
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        if dataset_name == 'mmlu':
            predicted_answer = extract_mmlu_answer(generated_text)
            correct = (predicted_answer == q['answer'])
        else:  # gsm8k
            predicted_answer_text = extract_gsm8k_answer(generated_text)
            # For GSM8K, we need to compare with the ground truth
            # Ground truth format is typically "solution #### answer"
            ground_truth_answer = extract_gsm8k_answer(q.get('answer', ''))
            correct = (predicted_answer_text == ground_truth_answer) if predicted_answer_text else False
        
        responses.append({
            'question_id': q['id'],
            'predicted_answer': predicted_answer if dataset_name == 'mmlu' else predicted_answer_text,
            'correct_answer': q['answer'],
            'correct': correct,
            'generated_text': generated_text[:500]  # Truncate for storage
        })
        
        # Clear cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    return responses


def evaluate_model(model_config, mmlu_questions, gsm8k_questions, output_dir):
    """Evaluate a single model on both datasets."""
    model_name = model_config['name']
    output_file = os.path.join(output_dir, f'{model_name}_responses.json')
    
    # Check if already evaluated
    if os.path.exists(output_file):
        print(f"\nSkipping {model_name} - already evaluated")
        with open(output_file) as f:
            return json.load(f)
    
    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_config)
        
        # Run inference
        mmlu_responses = run_inference_on_dataset(model, tokenizer, 'mmlu', mmlu_questions)
        gsm8k_responses = run_inference_on_dataset(model, tokenizer, 'gsm8k', gsm8k_questions)
        
        # Calculate accuracies
        mmlu_acc = np.mean([r['correct'] for r in mmlu_responses])
        gsm8k_acc = np.mean([r['correct'] for r in gsm8k_responses])
        combined_acc = (mmlu_acc * len(mmlu_responses) + gsm8k_acc * len(gsm8k_responses)) / \
                       (len(mmlu_responses) + len(gsm8k_responses))
        
        results = {
            'model_name': model_name,
            'model_hf_id': model_config['hf_id'],
            'timestamp': datetime.now().isoformat(),
            'mmlu': {
                'n_questions': len(mmlu_responses),
                'accuracy': float(mmlu_acc),
                'responses': mmlu_responses
            },
            'gsm8k': {
                'n_questions': len(gsm8k_responses),
                'accuracy': float(gsm8k_acc),
                'responses': gsm8k_responses
            },
            'combined_accuracy': float(combined_acc)
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{model_name} Results:")
        print(f"  MMLU: {mmlu_acc:.3f} ({int(mmlu_acc*len(mmlu_responses))}/{len(mmlu_responses)})")
        print(f"  GSM8K: {gsm8k_acc:.3f} ({int(gsm8k_acc*len(gsm8k_responses))}/{len(gsm8k_responses)})")
        print(f"  Combined: {combined_acc:.3f}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"\nERROR evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None, help='Specific models to evaluate')
    parser.add_argument('--output_dir', default='data/model_responses', help='Output directory')
    parser.add_argument('--test', action='store_true', help='Test mode - run on subset')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    mmlu_questions = load_questions('data/mmlu_test.json')
    gsm8k_questions = load_questions('data/gsm8k_test.json')
    
    if args.test:
        print("TEST MODE: Using only 50 questions per dataset")
        mmlu_questions = mmlu_questions[:50]
        gsm8k_questions = gsm8k_questions[:50]
    
    print(f"MMLU: {len(mmlu_questions)} questions")
    print(f"GSM8K: {len(gsm8k_questions)} questions")
    
    # Filter models if specified
    models_to_eval = MODELS_CONFIG
    if args.models:
        models_to_eval = [m for m in MODELS_CONFIG if m['name'] in args.models]
    
    print(f"\nWill evaluate {len(models_to_eval)} models:")
    for m in models_to_eval:
        print(f"  - {m['name']}")
    
    # Evaluate each model
    all_results = {}
    for model_config in models_to_eval:
        result = evaluate_model(model_config, mmlu_questions, gsm8k_questions, args.output_dir)
        if result:
            all_results[model_config['name']] = result
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_evaluated': list(all_results.keys()),
        'n_models': len(all_results),
        'n_mmlu': len(mmlu_questions),
        'n_gsm8k': len(gsm8k_questions),
        'model_accuracies': {name: r['combined_accuracy'] for name, r in all_results.items()}
    }
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Models evaluated: {len(all_results)}/{len(models_to_eval)}")
    print("\nFinal Rankings (by combined accuracy):")
    rankings = sorted(summary['model_accuracies'].items(), key=lambda x: x[1], reverse=True)
    for i, (name, acc) in enumerate(rankings, 1):
        print(f"  {i}. {name}: {acc:.3f}")


if __name__ == '__main__':
    main()
