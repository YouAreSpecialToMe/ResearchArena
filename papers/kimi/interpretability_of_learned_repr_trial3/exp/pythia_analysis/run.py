#!/usr/bin/env python3
"""
Pythia Checkpoint Analysis with Multi-Seed Validation
Apply PhaseMine to Pythia-160M checkpoints for reproducible validation.
Uses 8 checkpoints and 3 random seeds for probe training.
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

# All 8 Pythia-160M checkpoints
checkpoints = [1, 1000, 4000, 16000, 32000, 64000, 100000, 143000]

# 12 concepts to probe
concepts = [
    'question',
    'exclamation',
    'number',
    'uppercase',
    'person_name',
    'location',
    'organization',
    'positive_sentiment',
    'negative_sentiment',
    'long_sentence',
    'technical',
    'narrative',
]

layers = [3, 6, 9, 12]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_text_dataset(n_samples=2000, seed=42):
    """Generate diverse synthetic text data with concept labels."""
    np.random.seed(seed)
    
    templates = {
        'question': [
            "What is {}?",
            "How does {} work?",
            "Why is {} important?",
            "When did {} happen?",
            "Who discovered {}?"
        ],
        'exclamation': [
            "Wow! {} is amazing!",
            "Incredible {}!",
            "I love {}!",
            "Fantastic {}!",
            "Unbelievable {}!"
        ],
        'statement': [
            "The {} is interesting.",
            "We studied {} today.",
            "{} is a topic of research.",
            "The concept of {} is complex.",
            "Scientists study {}."
        ],
        'person': [
            "Einstein developed {}.",
            "Marie Curie researched {}.",
            "Newton discovered {}.",
            "Tesla invented {}.",
            "Darwin studied {}."
        ],
        'location': [
            "The city of {} is historic.",
            "We visited {} last year.",
            "{} is a popular destination.",
            "The museum in {} is famous.",
            "Welcome to {}."
        ],
        'organization': [
            "NASA researches {}.",
            "Google works on {}.",
            "MIT studies {}.",
            "CERN discovered {}.",
            "UN discussed {}."
        ],
        'positive': [
            "I enjoy {} very much.",
            "{} is wonderful.",
            "Great progress in {}!",
            "{} brings me joy.",
            "Excellent work on {}!"
        ],
        'negative': [
            "I dislike {}.",
            "{} is problematic.",
            "Terrible results in {}.",
            "{} causes issues.",
            "Poor performance on {}."
        ],
        'technical': [
            "The API handles {} requests.",
            "Configure {} parameter.",
            "Optimize {} algorithm.",
            "The system processes {}.",
            "Neural network computes {}."
        ],
        'narrative': [
            "Once upon a time, {} existed.",
            "The story of {} begins.",
            "Long ago, {} was discovered.",
            "In ancient times, {} was known.",
            "Legends speak of {}."
        ]
    }
    
    topics = ['science', 'history', 'mathematics', 'physics', 'chemistry',
              'biology', 'astronomy', 'geology', 'psychology', 'sociology',
              'philosophy', 'literature', 'art', 'music', 'technology',
              'economics', 'politics', 'education', 'medicine', 'engineering']
    
    texts = []
    labels = {c: [] for c in concepts}
    
    samples_per_type = n_samples // 12
    
    for _ in range(samples_per_type):
        topic = np.random.choice(topics)
        
        # Question
        texts.append(np.random.choice(templates['question']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'question' else 0)
        
        # Exclamation
        texts.append(np.random.choice(templates['exclamation']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'exclamation' else 0)
        
        # Number
        texts.append(f"The value is {np.random.randint(10, 9999)} and {topic} matters.")
        for c in concepts:
            labels[c].append(1 if c == 'number' else 0)
        
        # Uppercase
        texts.append(f"IMPORTANT: {topic.upper()} IS CRITICAL!")
        for c in concepts:
            labels[c].append(1 if c == 'uppercase' else 0)
        
        # Person
        texts.append(np.random.choice(templates['person']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'person_name' else 0)
        
        # Location
        texts.append(np.random.choice(templates['location']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'location' else 0)
        
        # Organization
        texts.append(np.random.choice(templates['organization']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'organization' else 0)
        
        # Positive sentiment
        texts.append(np.random.choice(templates['positive']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'positive_sentiment' else 0)
        
        # Negative sentiment
        texts.append(np.random.choice(templates['negative']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'negative_sentiment' else 0)
        
        # Long sentence
        long = f"This is a very long and detailed sentence about {topic} that contains many words and explains {topic} in great detail and comprehensive length."
        texts.append(long)
        for c in concepts:
            labels[c].append(1 if c == 'long_sentence' else 0)
        
        # Technical
        texts.append(np.random.choice(templates['technical']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'technical' else 0)
        
        # Narrative
        texts.append(np.random.choice(templates['narrative']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'narrative' else 0)
    
    # Convert to numpy arrays
    for c in labels:
        labels[c] = np.array(labels[c][:n_samples])
    
    return texts[:n_samples], labels


def extract_activations(model, tokenizer, texts, layer, device, batch_size=16):
    """Extract activations from a specific layer."""
    all_activations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True,
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]
            # Mean pool over sequence
            pooled = hidden.mean(dim=1).cpu().numpy()
            all_activations.append(pooled)
    
    return np.vstack(all_activations)


def train_l1_probe(X, y, seed=42):
    """Train L1-regularized probe."""
    # Split data
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Skip if only one class
    if len(np.unique(y_train)) < 2:
        return None
    
    # Train L1 probe
    start_time = time.time()
    probe = LogisticRegression(
        penalty='l1',
        C=0.01,
        solver='saga',
        max_iter=500,
        random_state=seed
    )
    probe.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate
    train_acc = probe.score(X_train, y_train)
    val_acc = probe.score(X_val, y_val)
    
    # Compute metrics
    weights = probe.coef_[0]
    l1_norm = np.linalg.norm(weights, 1)
    l2_norm = np.linalg.norm(weights, 2)
    non_zero = np.sum(np.abs(weights) > 1e-6)
    
    # Concentration score (top 10%)
    k = max(1, int(0.1 * len(weights)))
    top_k_indices = np.argsort(np.abs(weights))[-k:]
    top_k_weights = weights[top_k_indices]
    concentration_score = np.linalg.norm(top_k_weights, 1) / (l1_norm + 1e-10)
    
    return {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'l1_norm': float(l1_norm),
        'l2_norm': float(l2_norm),
        'non_zero_weights': int(non_zero),
        'concentration_score': float(concentration_score),
        'train_time_seconds': float(train_time),
        'weights': weights.tolist()
    }


def compute_emergence_sharpness(accuracies, steps):
    """Compute emergence sharpness (second derivative)."""
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
    
    es = [0.0] + es + [0.0]
    return es


def detect_transitions(checkpoint_results):
    """Detect phase transitions from checkpoint results."""
    steps = sorted(checkpoint_results.keys())
    
    transitions = {}
    
    for concept in concepts:
        for layer in layers:
            key = f'{concept}_layer{layer}'
            
            accuracies = []
            concentration_scores = []
            valid_steps = []
            
            for step in steps:
                if concept in checkpoint_results[step] and f'layer_{layer}' in checkpoint_results[step][concept]:
                    result = checkpoint_results[step][concept][f'layer_{layer}']
                    if result is not None:
                        accuracies.append(result['val_accuracy'])
                        concentration_scores.append(result['concentration_score'])
                        valid_steps.append(step)
            
            if len(accuracies) < 3:
                continue
            
            # Compute emergence sharpness
            es_values = compute_emergence_sharpness(accuracies, valid_steps)
            
            # Detect transitions
            threshold_es = 0.0005
            detected = []
            
            for i in range(1, len(valid_steps)):
                if es_values[i] > threshold_es and concentration_scores[i] > concentration_scores[i-1] * 1.05:
                    detected.append({
                        'step': valid_steps[i],
                        'emergence_sharpness': es_values[i],
                        'concentration_score': concentration_scores[i],
                        'accuracy': accuracies[i]
                    })
            
            if detected:
                transitions[key] = detected
    
    return transitions


def run_pythia_analysis(seed=42):
    """Run PhaseMine analysis on all Pythia checkpoints."""
    print(f"\n{'='*60}")
    print(f"Pythia Analysis - Seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate dataset once
    print("Generating dataset...")
    texts, labels_dict = generate_text_dataset(n_samples=2000, seed=seed)
    
    checkpoint_results = {}
    
    for checkpoint_step in checkpoints:
        print(f"\n{'='*40}")
        print(f"Checkpoint {checkpoint_step}")
        print(f"{'='*40}")
        
        # Load model
        model_name = "EleutherAI/pythia-160m"
        revision = f"step{checkpoint_step}"
        
        print(f"Loading {revision}...")
        start_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        model = model.to(device)
        model.eval()
        load_time = time.time() - start_load
        print(f"Loaded in {load_time:.2f}s")
        
        checkpoint_results[checkpoint_step] = {}
        
        for concept in concepts:
            checkpoint_results[checkpoint_step][concept] = {}
            y = labels_dict[concept]
            
            # Skip if only one class
            if len(np.unique(y)) < 2:
                continue
            
            for layer in layers:
                print(f"  {concept} - Layer {layer}...", end=' ')
                
                # Extract activations
                X = extract_activations(model, tokenizer, texts, layer, device)
                
                # Train probe
                result = train_l1_probe(X, y, seed=seed)
                
                if result:
                    checkpoint_results[checkpoint_step][concept][f'layer_{layer}'] = result
                    print(f"Acc={result['val_accuracy']:.3f}, CS={result['concentration_score']:.3f}, L0={result['non_zero_weights']}")
                else:
                    print("SKIP")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    # Detect transitions
    print("\nDetecting phase transitions...")
    transitions = detect_transitions(checkpoint_results)
    
    n_transitions = len(transitions)
    print(f"Total transitions detected: {n_transitions}")
    
    results = {
        'seed': seed,
        'model': 'pythia-160m',
        'checkpoints': checkpoints,
        'concepts': concepts,
        'layers': layers,
        'checkpoint_results': checkpoint_results,
        'detected_transitions': transitions,
        'summary': {
            'n_transitions': n_transitions,
            'n_concepts': len(concepts),
            'n_layers': len(layers),
            'n_checkpoints': len(checkpoints)
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Run experiment
    results = run_pythia_analysis(seed=args.seed)
    
    # Save results
    output_file = os.path.join(args.output_dir, f'pythia_analysis_seed_{args.seed}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Pythia analysis complete. Results saved to {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
