"""
Fast Synthetic Validation for PhaseMine
Minimal version for time constraints.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import json
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleModel(nn.Module):
    """Simple 2-layer MLP for modular addition."""
    def __init__(self, p=97, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(p, 32)
        self.net = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p)
        )
    
    def forward(self, x):
        e1 = self.embed(x[:, 0])
        e2 = self.embed(x[:, 1])
        h = torch.cat([e1, e2], dim=-1)
        return self.net(h), h


def train_and_probe(seed=42):
    """Train model and run PhaseMine."""
    set_seed(seed)
    
    p = 97
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate modular addition data
    data = []
    for a in range(p):
        for b in range(p):
            c = (a + b) % p
            data.append((a, b, c))
    
    np.random.shuffle(data)
    train_data = data[:7000]
    val_data = data[7000:]
    
    # Train model
    model = SimpleModel(p=p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    checkpoints = [0, 250, 500, 750, 1000]
    checkpoint_data = {}
    
    for step in range(1001):
        model.train()
        batch = np.random.choice(len(train_data), 256, replace=False)
        inputs = torch.tensor([train_data[i][:2] for i in batch]).to(device)
        targets = torch.tensor([train_data[i][2] for i in batch]).to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if step in checkpoints:
            # Extract activations and train probes
            model.eval()
            with torch.no_grad():
                val_inputs = torch.tensor([d[:2] for d in val_data]).to(device)
                val_targets = torch.tensor([d[2] for d in val_data]).to(device)
                _, activations = model(val_inputs)
                
                # Predict even/odd
                labels = (val_targets.cpu().numpy() % 2).astype(int)
                acts = activations.cpu().numpy()
                
                # Train L1 probe
                probe = LogisticRegression(penalty='l1', C=0.1, solver='saga', max_iter=200, random_state=seed)
                probe.fit(acts, labels)
                acc = accuracy_score(labels, probe.predict(acts))
                l1_norm = np.sum(np.abs(probe.coef_[0]))
                
                # Compute concentration score
                abs_w = np.abs(probe.coef_[0])
                k = max(1, len(abs_w) // 20)
                top_k = np.sort(abs_w)[-k:]
                cs = np.sum(top_k) / (l1_norm + 1e-10)
                
                checkpoint_data[step] = {
                    'accuracy': float(acc),
                    'l1_norm': float(l1_norm),
                    'concentration_score': float(cs)
                }
    
    # Detect transitions
    steps = sorted(checkpoint_data.keys())
    accs = [checkpoint_data[s]['accuracy'] for s in steps]
    css = [checkpoint_data[s]['concentration_score'] for s in steps]
    
    # Compute emergence sharpness (second derivative)
    transitions = []
    for i in range(1, len(steps) - 1):
        es = accs[i+1] - 2*accs[i] + accs[i-1]
        cs_increasing = css[i] > css[i-1] if i > 0 else False
        
        if es > 0.001 and cs_increasing:
            transitions.append({
                'step': steps[i],
                'emergence_sharpness': float(es),
                'concentration_score': float(css[i]),
                'accuracy': float(accs[i])
            })
    
    return {
        'seed': seed,
        'final_accuracy': float(accs[-1]),
        'transitions': transitions,
        'checkpoint_data': checkpoint_data,
        'n_transitions': len(transitions)
    }


def main():
    start_time = time.time()
    
    seeds = [42, 43]
    results = []
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        result = train_and_probe(seed)
        results.append(result)
        print(f"  Final acc: {result['final_accuracy']:.4f}, Transitions: {result['n_transitions']}")
    
    # Aggregate
    aggregated = {
        'seeds': seeds,
        'n_seeds': len(seeds),
        'final_accuracy': {
            'mean': float(np.mean([r['final_accuracy'] for r in results])),
            'std': float(np.std([r['final_accuracy'] for r in results])),
            'values': [r['final_accuracy'] for r in results]
        },
        'n_transitions': {
            'mean': float(np.mean([r['n_transitions'] for r in results])),
            'std': float(np.std([r['n_transitions'] for r in results]))
        },
        'individual_results': results
    }
    
    os.makedirs('results/synthetic', exist_ok=True)
    with open('results/synthetic/validation_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\nComplete in {elapsed:.1f}s")
    print(f"Final accuracy: {aggregated['final_accuracy']['mean']:.4f} ± {aggregated['final_accuracy']['std']:.4f}")


if __name__ == '__main__':
    main()
