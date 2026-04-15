#!/usr/bin/env python3
"""
Ablation Study: Checkpoint Frequency Analysis
Study how checkpoint frequency affects detection accuracy.
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

# All checkpoints
all_checkpoints = [1, 1000, 4000, 16000, 32000, 64000, 100000, 143000]

# Subsampled checkpoint sets
checkpoint_sets = {
    'full_8': all_checkpoints,
    'medium_4': [1, 16000, 64000, 143000],
    'sparse_4': [1, 4000, 64000, 143000],
}

concepts = ['question', 'number', 'positive_sentiment', 'technical']
layers = [3, 6]


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


def compute_emergence_sharpness(accuracies, steps):
    """Compute emergence sharpness."""
    if len(accuracies) < 3:
        return [0.0] * len(accuracies)
    
    es = []
    for i in range(1, len(accuracies) - 1):
        dt = steps[i+1] - steps[i-1]
        if dt == 0:
            es.append(0.0)
        else:
            second_deriv = (accuracies[i+1] - 2*accuracies[i] + accuracies[i-1]) / (dt/2)**2
            es.append(float(second_deriv))
    
    return [0.0] + es + [0.0]


def run_frequency_ablation(checkpoint_subset_name, checkpoint_subset, seed=42):
    """Run analysis with a specific checkpoint frequency."""
    print(f"\n{'='*60}")
    print(f"Checkpoint Frequency Ablation - {checkpoint_subset_name}")
    print(f"Checkpoints: {checkpoint_subset}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data once
    texts, labels_dict = generate_texts(n_samples=800, seed=seed)
    
    checkpoint_results = {}
    
    for checkpoint_step in checkpoint_subset:
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
            
            for step in checkpoint_subset:
                if concept in checkpoint_results[step] and f'layer_{layer}' in checkpoint_results[step][concept]:
                    result = checkpoint_results[step][concept][f'layer_{layer}']
                    accuracies.append(result['val_accuracy'])
                    concentration_scores.append(result['concentration_score'])
                    valid_steps.append(step)
            
            if len(accuracies) >= 3:
                es_values = compute_emergence_sharpness(accuracies, valid_steps)
                threshold_es = 0.0005
                
                detected = []
                for i in range(1, len(valid_steps)):
                    if es_values[i] > threshold_es and concentration_scores[i] > concentration_scores[i-1] * 1.05:
                        detected.append({
                            'step': valid_steps[i],
                            'emergence_sharpness': es_values[i]
                        })
                
                if detected:
                    transitions[key] = detected
    
    return {
        'subset_name': checkpoint_subset_name,
        'checkpoints': checkpoint_subset,
        'n_checkpoints': len(checkpoint_subset),
        'checkpoint_results': checkpoint_results,
        'transitions': transitions,
        'n_transitions': len(transitions)
    }


def compare_frequencies(all_results):
    """Compare transition detection across frequencies."""
    # Use full_8 as reference
    reference_transitions = all_results['full_8']['transitions']
    
    comparisons = {}
    for subset_name in ['medium_4', 'sparse_4']:
        subset_transitions = all_results[subset_name]['transitions']
        
        # Count agreements (same concept detected with transition)
        common_keys = set(reference_transitions.keys()) & set(subset_transitions.keys())
        agreement = len(common_keys)
        
        reference_only = len(set(reference_transitions.keys()) - set(subset_transitions.keys()))
        subset_only = len(set(subset_transitions.keys()) - set(reference_transitions.keys()))
        
        comparisons[subset_name] = {
            'agreement_count': agreement,
            'reference_only': reference_only,
            'subset_only': subset_only,
            'agreement_rate': agreement / len(reference_transitions) if reference_transitions else 0
        }
    
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    all_results = {}
    
    for subset_name, checkpoint_subset in checkpoint_sets.items():
        result = run_frequency_ablation(subset_name, checkpoint_subset, seed=args.seed)
        all_results[subset_name] = result
        
        output_file = os.path.join(args.output_dir, f'freq_{subset_name}_seed_{args.seed}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_file}")
    
    # Compare frequencies
    comparisons = compare_frequencies(all_results)
    
    summary = {
        'results_by_frequency': all_results,
        'comparisons': comparisons,
        'recommendation': '4 checkpoints sufficient for detection' if all(
            c['agreement_rate'] > 0.7 for c in comparisons.values()
        ) else '8 checkpoints recommended'
    }
    
    summary_file = os.path.join(args.output_dir, f'summary_seed_{args.seed}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Checkpoint frequency ablation complete.")
    print(f"Comparisons: {comparisons}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
