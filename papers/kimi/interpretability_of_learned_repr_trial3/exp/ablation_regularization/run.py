#!/usr/bin/env python3
"""
Ablation Study: Sparsity Regularization Impact
Compare L1, L2, and no regularization on Pythia checkpoints.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Subset of checkpoints for ablation
ablation_checkpoints = [16000, 64000, 143000]

# Subset of concepts
concepts = ['question', 'number', 'person_name', 'positive_sentiment', 'technical']
layers = [3, 6]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_texts(n_samples=1000, seed=42):
    """Generate synthetic texts with labels."""
    np.random.seed(seed)
    
    templates = {
        'question': ["What is {}?", "How does {} work?", "Why is {} important?"],
        'statement': ["The {} is interesting.", "We studied {} today.", "{} is complex."],
        'person': ["Einstein developed {}.", "Newton discovered {}.", "Tesla invented {}."],
        'positive': ["I enjoy {}.", "{} is wonderful.", "Great progress in {}!"],
        'technical': ["The API handles {}.", "Configure {} parameter.", "Optimize {} algorithm."]
    }
    
    topics = ['science', 'history', 'mathematics', 'physics', 'technology']
    
    texts = []
    labels = {c: [] for c in concepts}
    
    samples_per = n_samples // 5
    
    for _ in range(samples_per):
        topic = np.random.choice(topics)
        
        # Question
        texts.append(np.random.choice(templates['question']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'question' else 0)
        
        # Number
        texts.append(f"The value is {np.random.randint(10, 999)}.")
        for c in concepts:
            labels[c].append(1 if c == 'number' else 0)
        
        # Person
        texts.append(np.random.choice(templates['person']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'person_name' else 0)
        
        # Positive
        texts.append(np.random.choice(templates['positive']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'positive_sentiment' else 0)
        
        # Technical
        texts.append(np.random.choice(templates['technical']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'technical' else 0)
    
    for c in labels:
        labels[c] = np.array(labels[c][:n_samples])
    
    return texts[:n_samples], labels


def extract_activations(model, tokenizer, texts, layer, device, batch_size=16):
    """Extract activations."""
    all_activations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]
            pooled = hidden.mean(dim=1).cpu().numpy()
            all_activations.append(pooled)
    
    return np.vstack(all_activations)


def train_probe(X, y, regularization='l1', C=0.01, seed=42):
    """Train probe with specified regularization."""
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    if len(np.unique(y_train)) < 2:
        return None
    
    start_time = time.time()
    
    if regularization == 'l1':
        probe = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=500, random_state=seed)
    elif regularization == 'l2':
        probe = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=500, random_state=seed)
    elif regularization == 'none':
        probe = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500, random_state=seed)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    
    probe.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    val_acc = probe.score(X_val, y_val)
    
    weights = probe.coef_[0]
    l1_norm = np.linalg.norm(weights, 1)
    l2_norm = np.linalg.norm(weights, 2)
    non_zero = np.sum(np.abs(weights) > 1e-6)
    
    # Concentration score
    k = max(1, int(0.1 * len(weights)))
    top_k_indices = np.argsort(np.abs(weights))[-k:]
    top_k_weights = weights[top_k_indices]
    concentration_score = np.linalg.norm(top_k_weights, 1) / (l1_norm + 1e-10)
    
    return {
        'val_accuracy': float(val_acc),
        'l1_norm': float(l1_norm),
        'l2_norm': float(l2_norm),
        'non_zero_weights': int(non_zero),
        'concentration_score': float(concentration_score),
        'train_time_seconds': float(train_time)
    }


def run_ablation(checkpoint_step, seed=42):
    """Run regularization ablation for one checkpoint."""
    print(f"\n{'='*60}")
    print(f"Regularization Ablation - Checkpoint {checkpoint_step}, Seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_name = "EleutherAI/pythia-160m"
    revision = f"step{checkpoint_step}"
    
    print(f"Loading {revision}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model = model.to(device)
    model.eval()
    
    # Generate data
    texts, labels_dict = generate_texts(n_samples=1000, seed=seed)
    
    results = {
        'checkpoint': checkpoint_step,
        'seed': seed,
        'l1': {},
        'l2': {},
        'none': {}
    }
    
    for concept in concepts:
        print(f"\nConcept: {concept}")
        y = labels_dict[concept]
        
        if len(np.unique(y)) < 2:
            continue
        
        for reg_type in ['l1', 'l2', 'none']:
            results[reg_type][concept] = {}
        
        for layer in layers:
            print(f"  Layer {layer}...")
            
            # Extract activations once
            X = extract_activations(model, tokenizer, texts, layer, device)
            
            # Train with different regularizations
            for reg_type in ['l1', 'l2', 'none']:
                result = train_probe(X, y, regularization=reg_type, C=0.01, seed=seed)
                if result:
                    results[reg_type][concept][f'layer_{layer}'] = result
                    print(f"    {reg_type.upper()}: Acc={result['val_accuracy']:.3f}, L0={result['non_zero_weights']}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def compute_agreement(results):
    """Compute agreement between regularization types."""
    agreements = {}
    
    for concept in concepts:
        agreements[concept] = {}
        
        for layer in layers:
            key = f'layer_{layer}'
            
            l1_acc = results['l1'].get(concept, {}).get(key, {}).get('val_accuracy', 0)
            l2_acc = results['l2'].get(concept, {}).get(key, {}).get('val_accuracy', 0)
            none_acc = results['none'].get(concept, {}).get(key, {}).get('val_accuracy', 0)
            
            # Agreement: within 5% accuracy
            l1_l2_agree = abs(l1_acc - l2_acc) < 0.05
            l1_none_agree = abs(l1_acc - none_acc) < 0.05
            l2_none_agree = abs(l2_acc - none_acc) < 0.05
            
            agreements[concept][key] = {
                'l1_l2_agree': l1_l2_agree,
                'l1_none_agree': l1_none_agree,
                'l2_none_agree': l2_none_agree,
                'l1_acc': l1_acc,
                'l2_acc': l2_acc,
                'none_acc': none_acc
            }
    
    return agreements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    all_results = []
    
    for checkpoint_step in ablation_checkpoints:
        result = run_ablation(checkpoint_step, seed=args.seed)
        result['agreements'] = compute_agreement(result)
        all_results.append(result)
        
        # Save intermediate
        output_file = os.path.join(args.output_dir, f'ablation_ckpt_{checkpoint_step}_seed_{args.seed}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_file}")
    
    # Save combined
    combined_file = os.path.join(args.output_dir, f'all_ablations_seed_{args.seed}.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Ablation study complete. Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
