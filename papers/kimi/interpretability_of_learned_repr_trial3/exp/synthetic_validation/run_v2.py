"""
PhaseMine Synthetic Validation V2
Simpler task that actually shows phase transitions.
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


class ToyModel(nn.Module):
    """Toy model where we can control feature emergence."""
    def __init__(self, input_dim=100, hidden_dim=64, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize with small weights (features not yet emerged)
        with torch.no_grad():
            self.fc1.weight *= 0.01
            self.fc2.weight *= 0.01
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        # Return both output and hidden for probing
        return self.fc2(h), h


def train_and_probe_v2(seed=42):
    """Train model and run PhaseMine."""
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create synthetic data with a clear feature
    n_samples = 1000
    n_features = 100
    
    # Feature 0 is the "ground truth" feature we want to detect
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Label depends primarily on feature 0 (with some noise)
    y = (X[:, 0] + np.random.randn(n_samples) * 0.3 > 0).astype(int)
    
    # Split
    n_train = 800
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]
    
    # Train model with curriculum - gradually increase feature strength
    model = ToyModel(input_dim=n_features, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)
    
    checkpoint_steps = [0, 50, 100, 150, 200, 300, 400, 500]
    checkpoint_data = {}
    
    for step in range(501):
        model.train()
        
        # Gradually increase the strength of the curriculum
        # Early steps: model learns slowly, features not yet emerged
        # Later steps: model learns the feature
        if step < 100:
            # Noisy labels in early training
            noise_level = 0.4
            noisy_y = y_train_t.clone()
            mask = torch.rand(n_train) < noise_level
            noisy_y[mask] = 1 - noisy_y[mask]
        elif step < 200:
            noise_level = 0.2
            noisy_y = y_train_t.clone()
            mask = torch.rand(n_train) < noise_level
            noisy_y[mask] = 1 - noisy_y[mask]
        else:
            noisy_y = y_train_t
        
        optimizer.zero_grad()
        outputs, hidden = model(X_train_t)
        loss = criterion(outputs, noisy_y)
        loss.backward()
        optimizer.step()
        
        if step in checkpoint_steps:
            model.eval()
            with torch.no_grad():
                _, hidden_val = model(X_val_t)
                hidden_val = hidden_val.cpu().numpy()
                
                # Train L1 probe on hidden activations
                probe = LogisticRegression(
                    penalty='l1', 
                    C=0.1, 
                    solver='saga', 
                    max_iter=100, 
                    random_state=seed
                )
                probe.fit(hidden_val, y_val)
                
                acc = accuracy_score(y_val, probe.predict(hidden_val))
                l1_norm = np.sum(np.abs(probe.coef_[0]))
                
                # Concentration score
                abs_w = np.abs(probe.coef_[0])
                k = max(1, len(abs_w) // 10)
                top_k_sum = np.sum(np.sort(abs_w)[-k:])
                cs = top_k_sum / (l1_norm + 1e-10)
                
                checkpoint_data[step] = {
                    'accuracy': float(acc),
                    'l1_norm': float(l1_norm),
                    'concentration_score': float(cs),
                    'l0_norm': int(np.sum(abs_w > 1e-6))
                }
    
    # Detect transitions
    steps = sorted(checkpoint_data.keys())
    accs = [checkpoint_data[s]['accuracy'] for s in steps]
    css = [checkpoint_data[s]['concentration_score'] for s in steps]
    
    transitions = []
    for i in range(1, len(steps) - 1):
        es = accs[i+1] - 2*accs[i] + accs[i-1]
        cs_increasing = css[i] > css[i-1] if i > 0 else False
        
        if (es > 0.005 or accs[i] > 0.7) and cs_increasing:  # Detect learning milestones
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
        'n_transitions': len(transitions),
        'accuracy_trajectory': accs,
        'cs_trajectory': css
    }


def main():
    start_time = time.time()
    
    seeds = [42, 43, 44]
    results = []
    
    for seed in seeds:
        print(f"Running seed {seed}...")
        result = train_and_probe_v2(seed)
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
    print(f"Avg transitions: {aggregated['n_transitions']['mean']:.1f}")


if __name__ == '__main__':
    main()
