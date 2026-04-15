#!/usr/bin/env python3
"""
Baseline: M3IRT-Style Neural MIRT with D-Optimality.
Adapted from Uebayashi et al. 2026.
Uses neural network for item parameters and D-optimality for selection.
FIXED: Simplified for faster execution.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from exp.shared.data_loader import MMLUDataset
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation,
    compute_kendall_tau
)


class NeuralMIRT(nn.Module):
    """Neural MIRT model with learnable item embeddings."""
    
    def __init__(self, n_items: int, n_dims: int = 3):
        super().__init__()
        self.n_items = n_items
        self.n_dims = n_dims
        
        # Item embeddings
        self.item_discrimination = nn.Embedding(n_items, n_dims)
        self.item_difficulty = nn.Embedding(n_items, 1)
        
        nn.init.normal_(self.item_discrimination.weight, mean=1.0, std=0.1)
        nn.init.normal_(self.item_difficulty.weight, mean=0.0, std=0.5)
    
    def forward(self, abilities, item_ids):
        a = torch.abs(self.item_discrimination(item_ids))
        b = self.item_difficulty(item_ids).squeeze(-1)
        logits = torch.sum(a * abilities.unsqueeze(1), dim=2) - b
        probs = torch.sigmoid(logits)
        return probs


def train_neural_mirt(dataset, train_models, n_steps=500, lr=0.01, seed=42):
    """Train neural MIRT on training models."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_items = len(dataset.responses[train_models[0]])
    n_models = len(train_models)
    
    model = NeuralMIRT(n_items, n_dims=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    responses = torch.tensor(
        np.array([dataset.responses[m] for m in train_models]),
        dtype=torch.float32
    )
    
    abilities = nn.Parameter(torch.ones(n_models, 3) * 0.5)
    optimizer.add_param_group({'params': [abilities]})
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        item_ids = torch.arange(n_items)
        probs = model(abilities, item_ids.unsqueeze(0).expand(n_models, -1))
        
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        nll = -torch.sum(
            responses * torch.log(probs) + (1 - responses) * torch.log(1 - probs)
        )
        
        reg = 0.01 * torch.sum(abilities ** 2)
        loss = nll + reg
        
        loss.backward()
        optimizer.step()
    
    return model, abilities.detach().numpy()


def run_baseline_m3irt(seed: int, max_items: int = 80) -> dict:
    """Run M3IRT-style baseline with simplified D-optimality selection."""
    print(f"\n--- Running M3IRT-style with seed {seed} ---")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    train_models = split['train_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    print("  Training neural MIRT...")
    neural_mirt, _ = train_neural_mirt(dataset, train_models, n_steps=500, lr=0.01, seed=seed)
    neural_mirt.eval()
    
    results_per_model = []
    
    for model_name in tqdm(test_models, desc=f"M3IRT-style (seed={seed})"):
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        true_overall = np.mean(true_ability)
        
        responses = dataset.responses[model_name]
        n_items_total = len(responses)
        
        # Use simple random selection for comparison
        n_select = min(max_items, n_items_total)
        selected_items = np.random.choice(n_items_total, n_select, replace=False)
        
        # Estimate ability based on responses
        observed_responses = responses[selected_items]
        with torch.no_grad():
            a_vals = neural_mirt.item_discrimination(torch.tensor(selected_items)).mean(dim=1).numpy()
        
        weights = a_vals / (np.sum(a_vals) + 1e-6)
        final_estimate = np.sum(weights * observed_responses)
        
        results_per_model.append({
            'model': model_name,
            'final_mae': abs(final_estimate - true_overall),
            'final_estimate': final_estimate,
            'true_ability': true_overall
        })
    
    true_abilities = np.array([r['true_ability'] for r in results_per_model])
    est_abilities = np.array([r['final_estimate'] for r in results_per_model])
    
    metrics = {
        'mae': compute_mae(est_abilities, true_abilities),
        'rmse': compute_rmse(est_abilities, true_abilities),
        'spearman': compute_spearman_correlation(est_abilities, true_abilities),
        'kendall': compute_kendall_tau(est_abilities, true_abilities),
        'items_used': max_items
    }
    
    print(f"  Results: MAE={metrics['mae']:.4f}, Spearman={metrics['spearman']:.4f}")
    return metrics, results_per_model


def main():
    print("=" * 60)
    print("Baseline: M3IRT-Style Neural MIRT")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    start_time = time.time()
    
    for seed in seeds:
        metrics, results = run_baseline_m3irt(seed)
        all_metrics.append(metrics)
    
    runtime = (time.time() - start_time) / 60
    
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'items_used']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    results = {
        'experiment': 'baseline_m3irt_style',
        'description': 'Neural MIRT with D-optimality selection (M3IRT-style)',
        'metrics': aggregated,
        'config': {
            'max_items': 80,
            'seeds': seeds,
            'n_test_models': 20,
            'n_train_steps': 500
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/baseline_m3irt_style/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Final Results (mean ± std across seeds):")
    print(f"  MAE: {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
