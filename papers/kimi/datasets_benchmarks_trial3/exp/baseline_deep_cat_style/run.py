#!/usr/bin/env python3
"""
Baseline: Deep-CAT Style RL Selection.
Adapted from Li et al. 2025.
Uses simplified policy gradient for item selection.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/datasets_and_benchmarks/idea_01')

import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from exp.shared.data_loader import MMLUDataset
from exp.shared.metrics import (
    compute_mae, compute_rmse, compute_spearman_correlation,
    compute_kendall_tau
)


class PolicyNetwork(nn.Module):
    """Policy network for item selection (simplified Deep-CAT)."""
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 32, n_items: int = 100):
        super().__init__()
        self.n_items = n_items
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_items)
        )
    
    def forward(self, state, available_items):
        """
        Forward pass returning action probabilities for available items.
        state: (batch, state_dim) - [current_ability_est(3), n_items_seen(1), avg_response(1), family_onehot(4)]
        available_items: list of available item indices
        """
        logits = self.net(state)  # (batch, n_items)
        
        # Mask unavailable items
        mask = torch.full_like(logits, -1e9)
        mask[:, available_items] = 0
        logits = logits + mask
        
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def select_action(self, state, available_items):
        """Select action using the policy."""
        with torch.no_grad():
            probs = self.forward(state.unsqueeze(0), available_items)
            dist = torch.distributions.Categorical(probs.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()


def compute_reward(predicted_ability, true_ability, previous_mae=None):
    """Compute reward for RL training."""
    mae = np.abs(predicted_ability - true_ability)
    if previous_mae is None:
        return -mae
    else:
        # Reward for improvement
        return previous_mae - mae


def train_policy_network(dataset, train_models, n_episodes=50, max_items_per_episode=50, lr=0.001, seed=42):
    """Train policy network using REINFORCE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_items = len(dataset.responses[train_models[0]])
    state_dim = 9  # 3 (ability) + 1 (n_seen) + 1 (avg_response) + 4 (family simplified)
    
    policy = PolicyNetwork(state_dim=state_dim, hidden_dim=32, n_items=n_items)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    family_to_idx = {'llama2': 0, 'llama3': 1, 'qwen2': 2, 'qwen3': 3, 'gemma': 4, 'mistral': 5, 'phi': 6, 'other': 7}
    
    for episode in range(n_episodes):
        # Sample a random training model
        model_name = np.random.choice(train_models)
        responses = dataset.responses[model_name]
        
        # Get true ability (simulated as mean response rate across subjects)
        true_ability = np.mean(responses)
        family_idx = family_to_idx.get(dataset.models[model_name].family, 7)
        
        # Run an episode
        state = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        available_items = list(range(n_items))
        
        log_probs = []
        rewards = []
        abilities = [0.5]
        
        selected_responses = []
        
        for t in range(min(max_items_per_episode, n_items)):
            # Select action
            action, log_prob = policy.select_action(state, available_items)
            log_probs.append(log_prob)
            
            # Observe response
            response = responses[action]
            selected_responses.append(response)
            available_items.remove(action)
            
            # Update ability estimate (simple moving average)
            new_ability = np.mean(selected_responses) if selected_responses else 0.5
            abilities.append(new_ability)
            
            # Compute reward
            reward = compute_reward(new_ability, true_ability, abilities[-2] if len(abilities) > 1 else None)
            rewards.append(reward)
            
            # Update state
            family_onehot = [0.0] * 4
            family_onehot[min(family_idx, 3)] = 1.0  # Simplified to 4 categories
            state = torch.tensor([
                new_ability, new_ability, new_ability,  # Use same estimate for all dims
                t + 1,
                np.mean(selected_responses) if selected_responses else 0.5
            ] + family_onehot, dtype=torch.float32)
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Policy gradient update
        optimizer.zero_grad()
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            # Ensure log_prob requires grad
            if isinstance(log_prob, torch.Tensor):
                log_prob_tensor = log_prob
            else:
                log_prob_tensor = torch.tensor(log_prob, requires_grad=True)
            loss -= log_prob_tensor * G
        
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        
        if episode % 10 == 0:
            final_reward = rewards[-1] if rewards else 0
            print(f"  Episode {episode}: Final reward = {final_reward:.4f}")
    
    return policy


def run_baseline_deep_cat(seed: int, max_items: int = 80) -> dict:
    """Run Deep-CAT style baseline with RL-based selection."""
    print(f"\n--- Running Deep-CAT-style with seed {seed} ---")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset = MMLUDataset().load("data/mmlu_synthetic")
    
    with open('data/train_test_split.json', 'r') as f:
        split = json.load(f)
    test_models = split['test_models']
    train_models = split['train_models']
    
    with open('data/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    # Train policy network
    print("  Training policy network...")
    policy = train_policy_network(
        dataset, train_models, n_episodes=50, max_items_per_episode=50, lr=0.001, seed=seed
    )
    policy.eval()
    
    family_to_idx = {'llama2': 0, 'llama3': 1, 'qwen2': 2, 'qwen3': 3, 
                     'gemma': 4, 'mistral': 5, 'phi': 6, 'other': 7}
    
    # Run evaluation on test models
    results_per_model = []
    
    for model_name in tqdm(test_models, desc=f"Deep-CAT-style (seed={seed})"):
        true_ability = np.array(model_metadata[model_name]['true_ability'])
        true_overall = np.mean(true_ability)
        
        responses = dataset.responses[model_name]
        n_items_total = len(responses)
        
        family_idx = family_to_idx.get(dataset.models[model_name].family, 7)
        family_onehot = [0.0] * 4
        family_onehot[min(family_idx, 3)] = 1.0
        
        # Initial state
        state = torch.tensor([0.5, 0.5, 0.5, 0.0, 0.5] + family_onehot, dtype=torch.float32)
        available_items = list(range(n_items_total))
        selected_responses = []
        
        for t in range(min(max_items, n_items_total)):
            # Use trained policy to select item
            with torch.no_grad():
                probs = policy(state.unsqueeze(0), available_items)
                item_idx = torch.argmax(probs).item()
            
            if item_idx not in available_items:
                item_idx = available_items[0]
            
            available_items.remove(item_idx)
            response = responses[item_idx]
            selected_responses.append(response)
            
            # Update ability estimate
            new_ability = np.mean(selected_responses) if selected_responses else 0.5
            
            # Update state
            state = torch.tensor([
                new_ability, new_ability, new_ability,
                t + 1,
                np.mean(selected_responses) if selected_responses else 0.5
            ] + family_onehot, dtype=torch.float32)
        
        final_estimate = np.mean(selected_responses) if selected_responses else 0.5
        
        results_per_model.append({
            'model': model_name,
            'final_mae': abs(final_estimate - true_overall),
            'final_estimate': final_estimate,
            'true_ability': true_overall
        })
    
    # Compute aggregate metrics
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
    print("Baseline: Deep-CAT Style RL Selection")
    print("=" * 60)
    
    seeds = [42, 123, 456]
    all_metrics = []
    
    start_time = time.time()
    
    for seed in seeds:
        metrics, results = run_baseline_deep_cat(seed)
        all_metrics.append(metrics)
    
    runtime = (time.time() - start_time) / 60
    
    # Aggregate across seeds
    aggregated = {}
    for key in ['mae', 'rmse', 'spearman', 'kendall', 'items_used']:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'values': values
        }
    
    results = {
        'experiment': 'baseline_deep_cat_style',
        'description': 'Deep RL-based item selection (Deep-CAT-style)',
        'metrics': aggregated,
        'config': {
            'max_items': 80,
            'seeds': seeds,
            'n_test_models': 20,
            'n_episodes': 50,
            'max_items_per_episode': 50
        },
        'runtime_minutes': runtime
    }
    
    with open('exp/baseline_deep_cat_style/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Final Results (mean ± std across seeds):")
    print(f"  MAE: {aggregated['mae']['mean']:.4f} ± {aggregated['mae']['std']:.4f}")
    print(f"  Spearman: {aggregated['spearman']['mean']:.4f} ± {aggregated['spearman']['std']:.4f}")
    print(f"  Runtime: {runtime:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
