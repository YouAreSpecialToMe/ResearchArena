#!/usr/bin/env python3
"""
Meta-Predictor for Feature Emergence Timing
Train lightweight MLP to predict feature emergence from early-training signals.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Use all 8 checkpoints for meta-predictor training
checkpoints = [1, 1000, 4000, 16000, 32000, 64000, 100000, 143000]
early_checkpoint_cutoff = 32000  # First 3 checkpoints are "early"

concepts = ['question', 'number', 'positive_sentiment', 'technical', 'person_name']
layers = [3, 6, 9]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_texts(n_samples=1500, seed=42):
    """Generate synthetic texts."""
    np.random.seed(seed)
    
    templates = {
        'question': ["What is {}?", "How does {} work?", "Why is {} important?"],
        'statement': ["The {} is interesting.", "We studied {} today."],
        'positive': ["I enjoy {}.", "{} is wonderful."],
        'technical': ["The API handles {}.", "Configure {} parameter."],
        'person': ["Einstein developed {}.", "Newton discovered {}."]
    }
    
    topics = ['science', 'history', 'mathematics', 'physics', 'technology']
    
    texts = []
    labels = {c: [] for c in concepts}
    
    samples_per = n_samples // 5
    
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
        
        texts.append(np.random.choice(templates['person']).format(topic))
        for c in concepts:
            labels[c].append(1 if c == 'person_name' else 0)
    
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


def train_probe_and_get_metrics(X, y, seed=42):
    """Train probe and return early metrics."""
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
    
    # Concentration score
    k = max(1, int(0.1 * len(weights)))
    top_k_indices = np.argsort(np.abs(weights))[-k:]
    top_k_weights = weights[top_k_indices]
    concentration_score = np.linalg.norm(top_k_weights, 1) / (l1_norm + 1e-10)
    
    return {
        'val_accuracy': val_acc,
        'l1_norm': l1_norm,
        'non_zero_weights': non_zero,
        'concentration_score': concentration_score
    }


class MetaPredictor(nn.Module):
    """Lightweight MLP meta-predictor."""
    
    def __init__(self, input_dim=16, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


def collect_probe_data(seed=42):
    """Collect probe characteristics across all checkpoints."""
    print("\nCollecting probe data across checkpoints...")
    
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    texts, labels_dict = generate_texts(n_samples=1500, seed=seed)
    
    # Data structure: {concept_layer: {checkpoint: metrics}}
    all_data = {}
    
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
        
        for concept in concepts:
            y = labels_dict[concept]
            
            if len(np.unique(y)) < 2:
                continue
            
            for layer in layers:
                key = f"{concept}_layer{layer}"
                if key not in all_data:
                    all_data[key] = {}
                
                X = extract_activations(model, tokenizer, texts, layer, device)
                metrics = train_probe_and_get_metrics(X, y, seed=seed)
                
                if metrics:
                    all_data[key][checkpoint_step] = metrics
        
        del model
        torch.cuda.empty_cache()
    
    return all_data


def detect_emergence_step(probe_data, threshold_acc=0.7):
    """Detect emergence step for each concept-layer."""
    emergence_steps = {}
    
    for key, checkpoint_metrics in probe_data.items():
        # Find first checkpoint where accuracy exceeds threshold
        steps = sorted(checkpoint_metrics.keys())
        for step in steps:
            if checkpoint_metrics[step]['val_accuracy'] >= threshold_acc:
                emergence_steps[key] = step
                break
        
        if key not in emergence_steps:
            # If never reaches threshold, use last checkpoint
            emergence_steps[key] = steps[-1]
    
    return emergence_steps


def build_meta_predictor_dataset(probe_data, emergence_steps):
    """Build dataset for meta-predictor."""
    features = []
    targets = []
    
    early_checkpoints = [c for c in checkpoints if c <= early_checkpoint_cutoff]
    
    for key, checkpoint_metrics in probe_data.items():
        if key not in emergence_steps:
            continue
        
        # Extract early-training features
        early_features = []
        
        for ckpt in early_checkpoints:
            if ckpt in checkpoint_metrics:
                m = checkpoint_metrics[ckpt]
                early_features.extend([
                    m['val_accuracy'],
                    m['l1_norm'],
                    m['non_zero_weights'],
                    m['concentration_score']
                ])
            else:
                early_features.extend([0, 0, 0, 0])
        
        # Target: emergence step
        target = emergence_steps[key]
        
        features.append(early_features)
        targets.append(target)
    
    return np.array(features), np.array(targets)


def train_meta_predictor(X, y, seed=42, n_epochs=300):
    """Train meta-predictor."""
    set_seed(seed)
    
    # Split data
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    
    # Create model
    model = MetaPredictor(input_dim=X.shape[1], hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t)
            print(f"  Epoch {epoch+1}/{n_epochs}: Train Loss={loss.item():.2f}, Val Loss={val_loss.item():.2f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t).numpy()
        y_pred_val = model(X_val_t).numpy()
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    
    return {
        'r2_train': float(r2_train),
        'r2_val': float(r2_val),
        'mae_val': float(mae_val),
        'n_train': n_train,
        'n_val': n_samples - n_train
    }


def run_meta_predictor(seed=42):
    """Run meta-predictor experiment."""
    print(f"\n{'='*60}")
    print(f"Meta-Predictor Experiment - Seed {seed}")
    print(f"{'='*60}")
    
    # Collect probe data across checkpoints
    probe_data = collect_probe_data(seed=seed)
    
    print(f"\nCollected data for {len(probe_data)} concept-layer pairs")
    
    # Detect emergence steps
    emergence_steps = detect_emergence_step(probe_data)
    print(f"Detected emergence steps for {len(emergence_steps)} pairs")
    
    # Build dataset
    X, y = build_meta_predictor_dataset(probe_data, emergence_steps)
    print(f"Meta-predictor dataset: {X.shape}")
    
    if len(X) < 10:
        print("Not enough data for meta-predictor")
        return {
            'seed': seed,
            'error': 'Not enough data',
            'n_samples': len(X)
        }
    
    # Train meta-predictor
    print("\nTraining meta-predictor...")
    results = train_meta_predictor(X, y, seed=seed)
    
    results['seed'] = seed
    results['n_concept_layers'] = len(probe_data)
    results['emergence_steps'] = emergence_steps
    
    print(f"\nMeta-Predictor Results:")
    print(f"  R² (train): {results['r2_train']:.4f}")
    print(f"  R² (val): {results['r2_val']:.4f}")
    print(f"  MAE (val): {results['mae_val']:.2f} steps")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str,
                       default=str(Path(__file__).parent / 'results'))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    results = run_meta_predictor(seed=args.seed)
    
    output_file = os.path.join(args.output_dir, f'meta_predictor_seed_{args.seed}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Meta-predictor experiment complete. Results saved to {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
