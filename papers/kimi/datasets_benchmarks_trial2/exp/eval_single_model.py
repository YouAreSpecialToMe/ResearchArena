#!/usr/bin/env python3
"""
Evaluate a single model - designed to be run in parallel for multiple models.
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

# Model configurations
MODELS_CONFIG = {
    'Qwen2.5-0.5B': {'hf_id': 'Qwen/Qwen2.5-0.5B-Instruct', 'load_4bit': False},
    'Qwen2.5-1.8B': {'hf_id': 'Qwen/Qwen2.5-1.8B-Instruct', 'load_4bit': False},
    'Gemma-2-2B': {'hf_id': 'google/gemma-2-2b-it', 'load_4bit': False},
    'Gemma-2-9B': {'hf_id': 'google/gemma-2-9b-it', 'load_4bit': True},
    'Llama-3.1-8B': {'hf_id': 'meta-llama/Llama-3.1-8B-Instruct', 'load_4bit': True},
    'Mistral-7B': {'hf_id': 'mistralai/Mistral-7B-Instruct-v0.3', 'load_4bit': True},
    'Qwen2.5-7B': {'hf_id': 'Qwen/Qwen2.5-7B-Instruct', 'load_4bit': True},
    'Qwen2.5-14B': {'hf_id': 'Qwen/Qwen2.5-14B-Instruct', 'load_4bit': True},
    'Phi-4': {'hf_id': 'microsoft/Phi-4', 'load_4bit': True},
    'Gemma-2-27B': {'hf_id': 'google/gemma-2-27b-it', 'load_4bit': True},
    'Qwen2.5-32B': {'hf_id': 'Qwen/Qwen2.5-32B-Instruct', 'load_4bit': True},
    'Llama-3.1-70B': {'hf_id': 'meta-llama/Llama-3.1-70B-Instruct', 'load_4bit': True},
}


def format_mmlu_prompt(question, choices):
    """Format MMLU question."""
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"
    return prompt


def format_gsm8k_prompt(question):
    """Format GSM8K question."""
    return f"Question: {question}\nAnswer:"


def extract_mmlu_answer(text):
    """Extract multiple choice answer from model output."""
    import re
    text = text.strip()
    
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
    
    # Default: check first occurrence
    for char in text.upper():
        if char in 'ABCD':
            return ord(char) - ord('A')
    
    return -1


def extract_gsm8k_answer(text):
    """Extract numeric answer from GSM8K output."""
    import re
    # Look for "####" pattern
    match = re.search(r'####\s*(-?\d[\d,]*)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for numbers
    numbers = re.findall(r'-?\d[\d,]*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def load_model(model_name, config):
    """Load model and tokenizer."""
    print(f"[{model_name}] Loading {config['hf_id']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['hf_id'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    
    if config.get('load_4bit', False):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs['quantization_config'] = bnb_config
        print(f"[{model_name}] Using 4-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(config['hf_id'], **load_kwargs)
    return model, tokenizer


def evaluate(model_name, mmlu_questions, gsm8k_questions, output_dir):
    """Evaluate a single model."""
    output_file = os.path.join(output_dir, f'{model_name}_responses.json')
    
    if os.path.exists(output_file):
        print(f"[{model_name}] Already evaluated, loading existing results...")
        with open(output_file) as f:
            return json.load(f)
    
    config = MODELS_CONFIG[model_name]
    
    try:
        model, tokenizer = load_model(model_name, config)
        
        # Evaluate MMLU
        print(f"[{model_name}] Evaluating on {len(mmlu_questions)} MMLU questions...")
        mmlu_results = []
        for q in tqdm(mmlu_questions, desc=f"{model_name} MMLU", disable=True):
            prompt = format_mmlu_prompt(q['question'], q['choices'])
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            
            text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_mmlu_answer(text)
            mmlu_results.append({
                'question_id': q['id'],
                'correct': pred == q['answer'],
                'predicted': pred,
                'actual': q['answer']
            })
            
            if len(mmlu_results) % 100 == 0:
                torch.cuda.empty_cache()
        
        # Evaluate GSM8K
        print(f"[{model_name}] Evaluating on {len(gsm8k_questions)} GSM8K questions...")
        gsm8k_results = []
        for q in tqdm(gsm8k_questions, desc=f"{model_name} GSM8K", disable=True):
            prompt = format_gsm8k_prompt(q['question'])
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            
            text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            pred = extract_gsm8k_answer(text)
            actual = extract_gsm8k_answer(q.get('answer', ''))
            gsm8k_results.append({
                'question_id': q['id'],
                'correct': pred == actual if pred else False,
                'predicted': pred,
                'actual': actual
            })
            
            if len(gsm8k_results) % 50 == 0:
                torch.cuda.empty_cache()
        
        # Calculate accuracies
        mmlu_acc = np.mean([r['correct'] for r in mmlu_results])
        gsm8k_acc = np.mean([r['correct'] for r in gsm8k_results])
        combined_acc = (mmlu_acc * len(mmlu_results) + gsm8k_acc * len(gsm8k_results)) / \
                       (len(mmlu_results) + len(gsm8k_results))
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'mmlu_accuracy': float(mmlu_acc),
            'gsm8k_accuracy': float(gsm8k_acc),
            'combined_accuracy': float(combined_acc),
            'mmlu_results': mmlu_results,
            'gsm8k_results': gsm8k_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"[{model_name}] MMLU: {mmlu_acc:.3f}, GSM8K: {gsm8k_acc:.3f}, Combined: {combined_acc:.3f}")
        
        del model
        torch.cuda.empty_cache()
        return results
        
    except Exception as e:
        print(f"[{model_name}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model name to evaluate')
    parser.add_argument('--output_dir', default='data/model_responses')
    parser.add_argument('--mmlu', default='data/mmlu_test.json')
    parser.add_argument('--gsm8k', default='data/gsm8k_test.json')
    parser.add_argument('--test', action='store_true', help='Test mode (50 questions)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    with open(args.mmlu) as f:
        mmlu = json.load(f)
    with open(args.gsm8k) as f:
        gsm8k = json.load(f)
    
    if args.test:
        mmlu = mmlu[:50]
        gsm8k = gsm8k[:50]
    
    print(f"[{args.model}] Starting evaluation on {len(mmlu)} MMLU + {len(gsm8k)} GSM8K questions")
    result = evaluate(args.model, mmlu, gsm8k, args.output_dir)
    
    if result:
        print(f"[{args.model}] Evaluation complete!")
    else:
        print(f"[{args.model}] Evaluation failed!")


if __name__ == '__main__':
    main()
