#!/usr/bin/env python3
"""
Dense Linear Probing Baseline
Implements standard dense linear probing without sparsity regularization.
Runs on Pythia-160M checkpoints to establish comparison baseline.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Pythia-160M checkpoints
checkpoints = [1, 1000, 4000, 16000, 32000, 64000, 100000, 143000]

# 12 concepts to probe
concepts = [
    'question',      # Syntactic: contains '?'
    'exclamation',   # Syntactic: contains '!'
    'number',        # Syntactic: contains digits
    'uppercase',     # Syntactic: contains uppercase words
    'person_name',   # Semantic: person entity
    'location',      # Semantic: location entity
    'organization',  # Semantic: organization entity
    'positive_sentiment',  # Semantic: positive sentiment
    'negative_sentiment',  # Semantic: negative sentiment
    'long_sentence', # Task-specific: sentence length
    'technical',     # Task-specific: technical terms
    ' narrative',    # Task-specific: narrative/story
]

layers = [3, 6, 9, 12]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(n_samples=1000, seed=42):
    """Generate synthetic text data with concept labels."""
    np.random.seed(seed)
    
    templates = {
        'question': [
            "What is the capital of {}?",
            "How do I solve {}?",
            "Why does {} happen?",
            "When will {} arrive?",
            "Who wrote {}?"
        ],
        'statement': [
            "The capital of {} is important.",
            "Solving {} requires effort.",
            "{} happens regularly.",
            "{} will arrive soon.",
            "The author of {} is famous."
        ],
        'exclamation': [
            "What a wonderful {}!",
            "I can't believe {}!",
            "Amazing {}!",
            "Incredible {}!",
            "Fantastic {}!"
        ],
        'person': [
            "John Smith visited {}.",
            "Mary Johnson wrote about {}.",
            "David Lee studied {}.",
            "Sarah Brown discovered {}.",
            "Michael Chen explored {}."
        ],
        'location': [
            "The city of {} is beautiful.",
            "Paris and {} are popular.",
            "We traveled to {}.",
            "The road to {} is long.",
            "Welcome to {}."
        ],
        'organization': [
            "Microsoft developed {}.",
            "Google researched {}.",
            "Amazon sells {}.",
            "Apple created {}.",
            "OpenAI studied {}."
        ],
        'positive': [
            "I love {}!",
            "{} is wonderful.",
            "Great experience with {}.",
            "{} makes me happy.",
            "Best {} ever!"
        ],
        'negative': [
            "I hate {}.",
            "{} is terrible.",
            "Awful experience with {}.",
            "{} makes me sad.",
            "Worst {} ever."
        ],
        'technical': [
            "The API requires {} authentication.",
            "Configure the {} parameter.",
            "The algorithm optimizes {}.",
            "Neural networks process {}.",
            "The database stores {}."
        ],
        'narrative': [
            "Once upon a time in {}, there lived a dragon.",
            "The story of {} began long ago.",
            "In {}, a hero emerged.",
            "Long ago in {}, magic existed.",
            "The legend of {} continues."
        ]
    }
    
    topics = ['science', 'history', 'mathematics', 'literature', 'technology', 
              'philosophy', 'art', 'music', 'sports', 'politics']
    
    texts = []
    labels_dict = {concept: [] for concept in concepts}
    
    for _ in range(n_samples // 2):
        # Positive examples for each concept
        topic = np.random.choice(topics)
        
        # Question
        texts.append(np.random.choice(templates['question']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'question' else 0)
        
        # Number
        texts.append(f"The value is {np.random.randint(10, 999)}.")
        for c in concepts:
            labels_dict[c].append(1 if c == 'number' else 0)
        
        # Person
        texts.append(np.random.choice(templates['person']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'person_name' else 0)
        
        # Location
        texts.append(np.random.choice(templates['location']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'location' else 0)
        
        # Organization
        texts.append(np.random.choice(templates['organization']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'organization' else 0)
        
        # Positive sentiment
        texts.append(np.random.choice(templates['positive']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'positive_sentiment' else 0)
        
        # Technical
        texts.append(np.random.choice(templates['technical']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'technical' else 0)
        
        # Long sentence (>50 chars)
        long_text = f"This is a very long sentence about {topic} that contains many words and discusses {topic} in great detail and length."
        texts.append(long_text)
        for c in concepts:
            labels_dict[c].append(1 if c == 'long_sentence' else 0)
        
        # Exclamation
        texts.append(np.random.choice(templates['exclamation']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'exclamation' else 0)
        
        # Uppercase
        texts.append(f"IMPORTANT NEWS ABOUT {topic.upper()}!")
        for c in concepts:
            labels_dict[c].append(1 if c == 'uppercase' else 0)
        
        # Negative sentiment
        texts.append(np.random.choice(templates['negative']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'negative_sentiment' else 0)
        
        # Narrative
        texts.append(np.random.choice(templates['narrative']).format(topic))
        for c in concepts:
            labels_dict[c].append(1 if c == 'narrative' else 0)
    
    # Convert to numpy arrays
    for c in labels_dict:
        labels_dict[c] = np.array(labels_dict[c][:n_samples])
    
    return texts[:n_samples], labels_dict


def extract_activations(model, tokenizer, texts, layer, device, batch_size=8):
    """Extract activations from a specific layer."""
    activations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, 
                          truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get hidden states at the specified layer
            hidden = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]
            # Pool by averaging over sequence length
            pooled = hidden.mean(dim=1).cpu().numpy()  # [batch, hidden_dim]
            activations.append(pooled)
    
    return np.vstack(activations)


def train_dense_probe(X, y, seed=42):
    """Train dense L2-regularized probe."""
    start_time = time.time()
    
    # Split data
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train dense probe with L2 regularization
    probe = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=500,
        random_state=seed
    )
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_acc = probe.score(X_train, y_train)
    val_acc = probe.score(X_val, y_val)
    
    # Compute metrics
    weights = probe.coef_[0]
    l2_norm = np.linalg.norm(weights, 2)
    l1_norm = np.linalg.norm(weights, 1)
    
    # Non-zero weights (for comparison, though dense probes don't enforce sparsity)
    non_zero = np.sum(np.abs(weights) > 1e-6)
    
    train_time = time.time() - start_time
    
    return {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'l2_norm': float(l2_norm),
        'l1_norm': float(l1_norm),
        'non_zero_weights': int(non_zero),
        'train_time_seconds': float(train_time),
        'n_samples': n_samples
    }


def run_dense_baseline(checkpoint_step, seed=42):
    """Run dense baseline for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Dense Baseline - Checkpoint {checkpoint_step}, Seed {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_name = f"EleutherAI/pythia-160m"
    revision = f"step{checkpoint_step}"
    
    print(f"Loading {model_name} at {revision}...")
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model = model.to(device)
    model.eval()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    texts, labels_dict = generate_synthetic_data(n_samples=1000, seed=seed)
    
    results = {
        'checkpoint': checkpoint_step,
        'seed': seed,
        'model': 'pythia-160m',
        'load_time_seconds': load_time,
        'concepts': {}
    }
    
    total_probe_time = 0
    
    for concept in concepts:
        print(f"\n  Concept: {concept}")
        results['concepts'][concept] = {}
        
        for layer in layers:
            print(f"    Layer {layer}...", end=' ')
            
            # Extract activations
            X = extract_activations(model, tokenizer, texts, layer, device)
            y = labels_dict[concept]
            
            # Skip if only one class
            if len(np.unique(y)) < 2:
                print(f"SKIP (single class)")
                continue
            
            # Train probe
            probe_results = train_dense_probe(X, y, seed=seed)
            results['concepts'][concept][f'layer_{layer}'] = probe_results
            total_probe_time += probe_results['train_time_seconds']
            
            print(f"Acc={probe_results['val_accuracy']:.3f}, L2={probe_results['l2_norm']:.2f}")
    
    results['total_probe_time_seconds'] = total_probe_time
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=int, default=None, 
                       help='Single checkpoint to run (default: all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, 
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Determine which checkpoints to run
    if args.checkpoint is not None:
        checkpoint_list = [args.checkpoint]
    else:
        checkpoint_list = checkpoints
    
    all_results = []
    
    for checkpoint_step in checkpoint_list:
        result = run_dense_baseline(checkpoint_step, seed=args.seed)
        all_results.append(result)
        
        # Save intermediate result
        output_file = os.path.join(args.output_dir, f'checkpoint_{checkpoint_step}_seed_{args.seed}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {output_file}")
    
    # Save combined results
    combined_file = os.path.join(args.output_dir, f'all_checkpoints_seed_{args.seed}.json')
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Dense baseline complete. Results saved to {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
