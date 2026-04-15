#!/usr/bin/env python3
"""
Synthetic Validation: Modular Addition with Fourier Features
Train 2-layer transformer on modular addition (p=97) and validate PhaseMine
detects known Fourier feature emergence.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Modular addition parameters
P = 97  # Prime modulus
EMBEDDING_DIM = 128
NUM_HEADS = 4
D_FF = 512
NUM_LAYERS = 2

# Fourier frequency pairs to detect
FOURIER_FREQUENCIES = [0, 1, 2, 3, 4, 5]


class ModularAdditionDataset(Dataset):
    """Dataset for modular addition a + b ≡ c (mod p)."""
    
    def __init__(self, n_samples, p=P, seed=42):
        np.random.seed(seed)
        self.p = p
        self.n_samples = n_samples
        
        # Generate all possible pairs
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        np.random.shuffle(all_pairs)
        
        selected = all_pairs[:n_samples]
        self.a = np.array([x[0] for x in selected])
        self.b = np.array([x[1] for x in selected])
        self.c = (self.a + self.b) % p
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'a': self.a[idx],
            'b': self.b[idx],
            'c': self.c[idx]
        }


class TwoLayerTransformer(nn.Module):
    """2-layer transformer for modular addition."""
    
    def __init__(self, p=P, d_model=EMBEDDING_DIM, n_heads=NUM_HEADS, 
                 d_ff=D_FF, n_layers=NUM_LAYERS):
        super().__init__()
        self.p = p
        self.d_model = d_model
        
        # Embeddings for a, b, and = token
        self.embedding = nn.Embedding(p + 1, d_model)  # 0..p-1 for numbers, p for =
        self.pos_embedding = nn.Embedding(3, d_model)  # positions 0, 1, 2
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output = nn.Linear(d_model, p)
    
    def forward(self, a, b, return_activations=False):
        batch_size = a.size(0)
        device = a.device
        
        # Create input sequence: [a, b, =]
        equals_token = torch.full((batch_size,), self.p, device=device, dtype=torch.long)
        
        tok_emb = torch.stack([
            self.embedding(a),
            self.embedding(b),
            self.embedding(equals_token)
        ], dim=1)  # [batch, 3, d_model]
        
        pos_emb = self.pos_embedding(torch.arange(3, device=device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = tok_emb + pos_emb
        
        # Transformer
        hidden = self.transformer(x)  # [batch, 3, d_model]
        
        # Output from last position
        logits = self.output(hidden[:, -1, :])  # [batch, p]
        
        if return_activations:
            return logits, hidden
        return logits
    
    def get_layer_activations(self, a, b, layer_idx):
        """Extract activations from a specific layer."""
        batch_size = a.size(0)
        device = a.device
        
        equals_token = torch.full((batch_size,), self.p, device=device, dtype=torch.long)
        
        tok_emb = torch.stack([
            self.embedding(a),
            self.embedding(b),
            self.embedding(equals_token)
        ], dim=1)
        
        pos_emb = self.pos_embedding(torch.arange(3, device=device))
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        
        x = tok_emb + pos_emb
        
        # Manually get activations from specific layer
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            if i == layer_idx:
                return x
        
        return x


def train_model(train_loader, val_loader, device, n_steps=5000, seed=42):
    """Train 2-layer transformer on modular addition."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = TwoLayerTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    checkpoints = []
    checkpoint_steps = list(range(0, n_steps + 1, 500))
    
    step = 0
    train_iter = iter(train_loader)
    
    start_time = time.time()
    
    while step <= n_steps:
        # Save checkpoint
        if step in checkpoint_steps:
            ckpt_path = f'/tmp/modular_addition_ckpt_{step}.pt'
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            checkpoints.append({'step': step, 'path': ckpt_path})
            
            # Evaluate
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    a, b, c = batch['a'].to(device), batch['b'].to(device), batch['c'].to(device)
                    logits = model(a, b)
                    pred = logits.argmax(dim=-1)
                    val_correct += (pred == c).sum().item()
                    val_total += c.size(0)
            val_acc = val_correct / val_total
            print(f"Step {step}: Val Acc = {val_acc:.4f}")
            model.train()
        
        if step >= n_steps:
            break
        
        # Training step
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        a, b, c = batch['a'].to(device), batch['b'].to(device), batch['c'].to(device)
        
        optimizer.zero_grad()
        logits = model(a, b)
        loss = criterion(logits, c)
        loss.backward()
        optimizer.step()
        
        step += 1
        
        if step % 500 == 0:
            print(f"  Step {step}/{n_steps}, Loss: {loss.item():.4f}")
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f}s")
    
    return model, checkpoints


def compute_fourier_features(a, b, freq):
    """Compute Fourier features for given frequency."""
    # cos(2πk a/p), sin(2πk a/p), cos(2πk b/p), sin(2πk b/p)
    p = P
    cos_a = np.cos(2 * np.pi * freq * a / p)
    sin_a = np.sin(2 * np.pi * freq * a / p)
    cos_b = np.cos(2 * np.pi * freq * b / p)
    sin_b = np.sin(2 * np.pi * freq * b / p)
    
    return cos_a, sin_a, cos_b, sin_b


def create_fourier_labels(dataset, freq, component='cos_a'):
    """Create binary labels for Fourier component."""
    a, b = dataset.a, dataset.b
    cos_a, sin_a, cos_b, sin_b = compute_fourier_features(a, b, freq)
    
    # Binarize by median
    if component == 'cos_a':
        values = cos_a
    elif component == 'sin_a':
        values = sin_a
    elif component == 'cos_b':
        values = cos_b
    elif component == 'sin_b':
        values = sin_b
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # For freq=0, cos is always 1, so handle this specially
    if freq == 0 and 'cos' in component:
        # All same value, can't classify
        return np.ones(len(values), dtype=int)
    
    median = np.median(values)
    labels = (values > median).astype(int)
    
    # Ensure we have both classes
    if len(np.unique(labels)) < 2:
        # Use threshold at 0 instead
        labels = (values > 0).astype(int)
    
    return labels


def extract_activations_from_checkpoint(ckpt_path, dataset, device, layer=0):
    """Extract activations from a saved checkpoint."""
    model = TwoLayerTransformer().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_activations = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            a = torch.tensor(batch['a'], device=device)
            b = torch.tensor(batch['b'], device=device)
            
            activations = model.get_layer_activations(a, b, layer)
            # Pool over sequence dimension
            pooled = activations.mean(dim=1).cpu().numpy()
            all_activations.append(pooled)
    
    return np.vstack(all_activations)


def train_sparse_probe(X, y, seed=42):
    """Train L1-regularized sparse probe."""
    # Check if we have at least 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return None
    
    # Split data
    n_samples = len(y)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Check if both splits have at least 2 classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None
    
    # Train L1-regularized probe
    probe = LogisticRegression(
        penalty='l1',
        C=0.1,
        solver='saga',
        max_iter=500,
        random_state=seed
    )
    probe.fit(X_train, y_train)
    
    # Evaluate
    train_acc = probe.score(X_train, y_train)
    val_acc = probe.score(X_val, y_val)
    
    # Compute metrics
    weights = probe.coef_[0]
    l1_norm = np.linalg.norm(weights, 1)
    l2_norm = np.linalg.norm(weights, 2)
    
    # Sparsity metrics
    non_zero = np.sum(np.abs(weights) > 1e-6)
    
    # Concentration score (top 5% of weights)
    k = max(1, int(0.05 * len(weights)))
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
        'weights': weights.tolist()
    }


def compute_emergence_sharpness(accuracies, steps):
    """Compute second derivative of accuracy (emergence sharpness)."""
    if len(accuracies) < 3:
        return [0.0] * len(accuracies)
    
    # Compute second derivative
    es = []
    for i in range(1, len(accuracies) - 1):
        dt = steps[i+1] - steps[i-1]
        if dt == 0:
            es.append(0.0)
        else:
            second_deriv = (accuracies[i+1] - 2*accuracies[i] + accuracies[i-1]) / (dt/2)**2
            es.append(float(second_deriv))
    
    # Pad ends
    es = [0.0] + es + [0.0]
    return es


def detect_transitions(results_per_component):
    """Detect phase transitions for each Fourier component."""
    transitions = {}
    
    for component, results in results_per_component.items():
        steps = results['steps']
        accuracies = results['accuracies']
        concentration_scores = results['concentration_scores']
        
        # Compute emergence sharpness
        es_values = compute_emergence_sharpness(accuracies, steps)
        
        # Find transitions: ES > threshold AND CS increases
        threshold_es = 0.001
        detected = []
        
        for i in range(1, len(steps)):
            if es_values[i] > threshold_es:
                # Check if concentration score increases
                if concentration_scores[i] > concentration_scores[i-1]:
                    detected.append({
                        'step': steps[i],
                        'emergence_sharpness': es_values[i],
                        'concentration_score': concentration_scores[i],
                        'accuracy': accuracies[i]
                    })
        
        transitions[component] = detected
    
    return transitions


def run_synthetic_validation(seed=42):
    """Run complete synthetic validation experiment."""
    print(f"\n{'='*60}")
    print(f"Synthetic Validation - Modular Addition (p={P})")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nGenerating datasets...")
    train_dataset = ModularAdditionDataset(n_samples=5000, seed=seed)
    val_dataset = ModularAdditionDataset(n_samples=1000, seed=seed+1000)
    test_dataset = ModularAdditionDataset(n_samples=1000, seed=seed+2000)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)
    
    # Train model
    print("\nTraining 2-layer transformer...")
    model, checkpoints = train_model(train_loader, val_loader, device, 
                                     n_steps=5000, seed=seed)
    
    # Test final accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for i in range(0, len(test_dataset), 256):
            batch = test_dataset[i:i+256]
            a = torch.tensor(batch['a'], device=device)
            b = torch.tensor(batch['b'], device=device)
            c = batch['c']
            
            logits = model(a, b)
            pred = logits.argmax(dim=-1).cpu().numpy()
            test_correct += (pred == c).sum()
            test_total += len(c)
    
    test_accuracy = test_correct / test_total
    print(f"\nFinal test accuracy: {test_accuracy:.4f}")
    
    # Train sparse probes for Fourier components at each checkpoint
    print("\nTraining sparse probes for Fourier components...")
    
    # Define Fourier components to probe
    # Skip freq=0 cosine components (always constant = 1)
    fourier_components = []
    for freq in FOURIER_FREQUENCIES:
        if freq > 0:  # Skip cos_0 since it's always 1
            fourier_components.append(f'cos_a_freq{freq}')
        fourier_components.append(f'sin_a_freq{freq}')
        if freq > 0:
            fourier_components.append(f'cos_b_freq{freq}')
        fourier_components.append(f'sin_b_freq{freq}')
    
    results_per_component = {comp: {
        'steps': [],
        'accuracies': [],
        'l1_norms': [],
        'concentration_scores': [],
        'non_zero_weights': []
    } for comp in fourier_components}
    
    for ckpt_info in checkpoints:
        step = ckpt_info['step']
        path = ckpt_info['path']
        
        print(f"\nCheckpoint {step}...")
        
        # Extract activations
        activations = extract_activations_from_checkpoint(path, train_dataset, device, layer=0)
        
        for comp in fourier_components:
            # Parse component name
            parts = comp.split('_')
            component_type = parts[0]  # cos or sin
            var = parts[1]  # a or b
            freq = int(parts[2].replace('freq', ''))
            
            # Create labels
            labels = create_fourier_labels(train_dataset, freq, component=f'{component_type}_{var}')
            
            # Train probe
            probe_results = train_sparse_probe(activations, labels, seed=seed)
            
            # Skip if probe training failed
            if probe_results is None:
                continue
            
            # Store results
            results_per_component[comp]['steps'].append(step)
            results_per_component[comp]['accuracies'].append(probe_results['val_accuracy'])
            results_per_component[comp]['l1_norms'].append(probe_results['l1_norm'])
            results_per_component[comp]['concentration_scores'].append(probe_results['concentration_score'])
            results_per_component[comp]['non_zero_weights'].append(probe_results['non_zero_weights'])
    
    # Detect phase transitions
    print("\nDetecting phase transitions...")
    transitions = detect_transitions(results_per_component)
    
    # Count transitions per component
    n_transitions = sum(len(t) for t in transitions.values())
    n_components_with_transitions = sum(1 for t in transitions.values() if len(t) > 0)
    
    print(f"\nTransition Detection Summary:")
    print(f"  Components with transitions: {n_components_with_transitions}/{len(fourier_components)}")
    print(f"  Total transitions detected: {n_transitions}")
    
    results = {
        'seed': seed,
        'p': P,
        'test_accuracy': float(test_accuracy),
        'n_checkpoints': len(checkpoints),
        'fourier_components': fourier_components,
        'probe_trajectories': results_per_component,
        'detected_transitions': transitions,
        'summary': {
            'n_components_with_transitions': n_components_with_transitions,
            'total_transitions': n_transitions,
            'n_fourier_components': len(fourier_components)
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
    results = run_synthetic_validation(seed=args.seed)
    
    # Save results
    output_file = os.path.join(args.output_dir, f'synthetic_validation_seed_{args.seed}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Synthetic validation complete. Results saved to {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
