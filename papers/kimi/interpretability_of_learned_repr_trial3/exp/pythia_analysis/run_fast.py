#!/usr/bin/env python3
"""
Fast Pythia Checkpoint Analysis
Analyzes 8 checkpoints with reduced concept set for faster completion.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# All 8 checkpoints
checkpoints = [1, 1000, 4000, 16000, 32000, 64000, 100000, 143000]

# Reduced concept set for speed
concepts = ['question', 'number', 'positive_sentiment', 'technical']
layers = [3, 6, 9]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_texts(n_samples=800, seed=42):
    """Generate synthetic texts."""
    np.random.seed(seed)
    
    templates = {
        'question': ["What is {}?", "How does {} work?"],
        'statement': ["The {} is interesting.", "We studied {} today."],
        'positive': ["I enjoy {}.", "{} is wonderful."],
        'technical': ["The API handles {}.", "Configure {} parameter."]
    }
    
    topics = ['science', 'history', 'mathematics', 'technology']
    
    texts = []
    labels = {c: [] for c in concepts}
    
    samples_per = n_samples // 4
    
    for _ in range(samples_per):
        topic = np.random.choice(topics)
        
        texts.append(np.random.choice(templates['question']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'question' else 0)
        
        texts.append(f"The value is {np.random.randint(10, 999)}.")
        for c in concepts:
            labels[c].append(1 if c == 'number' else 0)
        
        texts.append(np.random.choice(templates['positive']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'positive_sentiment' else 0)
        
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


def train_probe(X, y, seed=42):
    """Train L1-regularized probe."""
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    if len(np.unique(y_train)) < 2:
        return None
    
    probe = LogisticRegression(penalty='l1', C=0.01, solver='saga', max_iter=500, random_state=seed)
    probe.fit(X_train, y_train)
    
    val_acc = probe.score(X_val, y_val)
    weights = probe.coef_[0]
    l1_norm = np.linalg.norm(weights, 1)
    non_zero = np.sum(np.abs(weights) > 1e-6)
    
    k = max(1, int(0.1 * len(weights)))
    top_k_indices = np.argsort(np.abs(weights))[-k:]
    top_k_weights = weights[top_k_indices]
    concentration_score = np.linalg.norm(top_k_weights, 1) / (l1_norm + 1e-10)
    
    return {
        'val_accuracy': float(val_acc),
        'l1_norm': float(l1_norm),
        'non_zero_weights': int(non_zero),
        'concentration_score': float(concentration_score)
    }


def run_pythia_analysis(seed=42):
    """Run PhaseMine analysis on all Pythia checkpoints."""
    print(f"\n{'='*60}")
    print(f"Fast Pythia Analysis - Seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    texts, labels_dict = generate_texts(n_samples=800, seed=seed)
    
    checkpoint_results = {}
    
    for checkpoint_step in checkpoints:
        print(f"\nCheckpoint {checkpoint_step}...")
        
        model_name = "EleutherAI/pythia-160m"
        revision = f"step{checkpoint_step}"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        model = model.to(device)
        model.eval()
        
        checkpoint_results[checkpoint_step] = {}
        
        for concept in concepts:
            checkpoint_results[checkpoint_step][concept] = {}
            y = labels_dict[concept]
            
            if len(np.unique(y)) < 2:
                continue
            
            for layer in layers:
                X = extract_activations(model, tokenizer, texts, layer, device)
                result = train_probe(X, y, seed=seed)
                
                if result:
                    checkpoint_results[checkpoint_step][concept][f'layer_{layer}'] = result
        
        del model
        torch.cuda.empty_cache()
    
    # Detect transitions
    transitions = {}
    for concept in concepts:
        for layer in layers:
            key = f'{concept}_layer{layer}'
            accuracies = []
            concentration_scores = []
            valid_steps = []
            
            for step in checkpoints:
                if concept in checkpoint_results[step] and f'layer_{layer}' in checkpoint_results[step][concept]:
                    result = checkpoint_results[step][concept][f'layer_{layer}']
                    accuracies.append(result['val_accuracy'])
                    concentration_scores.append(result['concentration_score'])
                    valid_steps.append(step)
            
            if len(accuracies) >= 3:
                # Simple threshold-based detection
                for i in range(1, len(valid_steps)):
                    if accuracies[i] > 0.7 and concentration_scores[i] > 0.8:
                        if key not in transitions:
                            transitions[key] = []
                        transitions[key].append({'step': valid_steps[i]})
                        break
    
    return {
        'seed': seed,
        'model': 'pythia-160m',
        'checkpoints': checkpoints,
        'concepts': concepts,
        'layers': layers,
        'checkpoint_results': checkpoint_results,
        'detected_transitions': transitions,
        'summary': {
            'n_transitions': len(transitions),
            'n_concepts': len(concepts),
            'n_layers': len(layers),
            'n_checkpoints': len(checkpoints)
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = run_pythia_analysis(seed=args.seed)
    
    output_file = os.path.join(args.output_dir, f'pythia_analysis_seed_{args.seed}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
