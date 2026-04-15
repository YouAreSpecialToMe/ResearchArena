"""Simplified IOI and RAVEL experiments for faster execution."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

from exp.shared.utils import set_seed, save_json
from exp.shared.metrics_fixed import compute_cgas_fixed


def extract_activations_simple(model, tokenizer, texts, layer_idx=9, device='cuda', batch_size=16):
    """Extract activations with simpler batching."""
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, 
                              max_length=30, padding=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_idx]
            
            for b in range(hidden.shape[0]):
                seq_len = inputs.attention_mask[b].sum().item()
                last_token_hidden = hidden[b, seq_len-1, :].cpu().numpy()
                all_activations.append(last_token_hidden)
    
    return np.array(all_activations)


def create_ioi_dataset_small():
    """Create smaller IOI dataset."""
    names = ["Alice", "Bob", "Carol", "David", "Eve", "Frank"]
    templates = [
        "When {A} and {B} went to the store, {A} gave a drink to {B}",
        "{A} and {B} had lunch. {A} gave a sandwich to {B}",
    ]
    
    texts = []
    name_pairs = []
    
    for template in templates:
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    text = template.format(A=names[i], B=names[j])
                    texts.append(text)
                    name_pairs.append((names[i], names[j]))
    
    return texts, name_pairs


def create_ravel_dataset_small():
    """Create smaller RAVEL dataset."""
    data = [
        ("France", "Paris", "country-capital"),
        ("Germany", "Berlin", "country-capital"),
        ("Einstein", "physicist", "name-occupation"),
        ("Shakespeare", "writer", "name-occupation"),
    ]
    
    dataset = []
    for entity, attribute, attr_type in data:
        if attr_type == "country-capital":
            base = f"The capital of {entity} is"
        else:
            base = f"{entity} worked as a"
        dataset.append({
            'entity': entity,
            'base_prompt': base,
            'attribute_type': attr_type
        })
    
    return dataset


def run_ioi_simple(device='cuda'):
    """Run simplified IOI experiment."""
    print("="*60)
    print("IOI Experiment (Simplified)")
    print("="*60)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    # Create dataset
    texts, name_pairs = create_ioi_dataset_small()
    print(f"Samples: {len(texts)}")
    
    # Split
    n_train = int(0.8 * len(texts))
    train_texts, val_texts = texts[:n_train], texts[n_train:]
    
    # Extract activations
    print("Extracting activations...")
    train_acts = extract_activations_simple(model, tokenizer, train_texts, 9, device)
    val_acts = extract_activations_simple(model, tokenizer, val_texts, 9, device)
    
    input_dim = train_acts.shape[1]
    print(f"Activations: train={train_acts.shape}, val={val_acts.shape}")
    
    # Use variance-based proxy for causal dims (simplified)
    dim_vars = np.var(val_acts, axis=0)
    top_dims = np.argsort(dim_vars)[-50:]
    causal_subspaces = val_acts[:, top_dims]
    
    # Simple baseline features
    SEEDS = [42, 123, 456]
    cgas_results = []
    
    for seed in SEEDS:
        set_seed(seed)
        
        # Random 1x
        projection = np.random.randn(input_dim, input_dim).astype(np.float32)
        projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
        features = val_acts @ projection
        
        cgas_result = compute_cgas_fixed(
            causal_subspaces=causal_subspaces,
            explanation_features=features,
            full_activations=val_acts,
            distance_metric='cosine',
            top_k=20,
            dictionary_size=input_dim,
            input_dim=input_dim
        )
        
        cgas_results.append({
            'method': 'random',
            'overcomplete': '1x',
            'seed': seed,
            'cgas': cgas_result['cgas']
        })
    
    # Summary
    cgas_vals = [r['cgas'] for r in cgas_results]
    summary = {
        'random': {
            '1x': {
                'mean': float(np.mean(cgas_vals)),
                'std': float(np.std(cgas_vals))
            }
        }
    }
    
    os.makedirs('exp/ioi/cgas', exist_ok=True)
    save_json({
        'cgas_all': cgas_results,
        'summary': summary,
        'note': 'Simplified experiment with variance-based proxy for causal subspaces'
    }, 'exp/ioi/cgas/results_fixed.json')
    
    print(f"\nResults: C-GAS = {np.mean(cgas_vals):.4f} ± {np.std(cgas_vals):.4f}")
    print("Saved to exp/ioi/cgas/results_fixed.json")


def run_ravel_simple(device='cuda'):
    """Run simplified RAVEL experiment."""
    print("\n" + "="*60)
    print("RAVEL Experiment (Simplified)")
    print("="*60)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    
    # Create dataset
    dataset = create_ravel_dataset_small()
    print(f"Samples: {len(dataset)}")
    
    # Extract activations
    texts = [item['base_prompt'] for item in dataset]
    acts = extract_activations_simple(model, tokenizer, texts, 9, device)
    
    input_dim = acts.shape[1]
    print(f"Activations: {acts.shape}")
    
    # Use variance-based proxy
    dim_vars = np.var(acts, axis=0)
    top_dims = np.argsort(dim_vars)[-30:]
    causal_subspaces = acts[:, top_dims]
    
    # Simple baseline
    SEEDS = [42, 123, 456]
    cgas_results = []
    
    for seed in SEEDS:
        set_seed(seed)
        
        projection = np.random.randn(input_dim, input_dim).astype(np.float32)
        projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
        features = acts @ projection
        
        cgas_result = compute_cgas_fixed(
            causal_subspaces=causal_subspaces,
            explanation_features=features,
            full_activations=acts,
            distance_metric='cosine',
            top_k=10,
            dictionary_size=input_dim,
            input_dim=input_dim
        )
        
        cgas_results.append({
            'method': 'random',
            'overcomplete': '1x',
            'seed': seed,
            'cgas': cgas_result['cgas']
        })
    
    cgas_vals = [r['cgas'] for r in cgas_results]
    summary = {
        'random': {
            '1x': {
                'mean': float(np.mean(cgas_vals)),
                'std': float(np.std(cgas_vals))
            }
        }
    }
    
    os.makedirs('exp/ravel/cgas', exist_ok=True)
    save_json({
        'cgas_all': cgas_results,
        'summary': summary,
        'note': 'Simplified experiment with variance-based proxy for causal subspaces'
    }, 'exp/ravel/cgas/results_fixed.json')
    
    print(f"\nResults: C-GAS = {np.mean(cgas_vals):.4f} ± {np.std(cgas_vals):.4f}")
    print("Saved to exp/ravel/cgas/results_fixed.json")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_ioi_simple(device)
    run_ravel_simple(device)
    
    print("\n" + "="*60)
    print("Simplified experiments complete!")
    print("="*60)


if __name__ == '__main__':
    main()
