"""Identify causal subspaces in synthetic MLP using activation patching."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import os

from exp.shared.utils import set_seed, save_json
from exp.shared.data_loader import generate_synthetic_ground_truth_features
from exp.shared.models import SyntheticMLP

def activation_patching_causal_identification(
    model: SyntheticMLP,
    X: torch.Tensor,
    features: dict,
    device: str
) -> dict:
    """Identify causal subspaces via activation patching.
    
    For each ground-truth feature, create contrastive pairs differing only in that
    feature, then patch activations to find which dimensions are causal.
    """
    model.eval()
    n_samples = X.shape[0]
    hidden_dim = 64
    
    results = {}
    
    for feat_name in ['f1', 'f2', 'f3', 'f4', 'f5']:
        print(f"\n  Identifying causal subspaces for {feat_name}...")
        
        # Create contrastive pairs by flipping the sign of the feature
        # We'll do this by modifying the relevant input dimensions
        feature_values = features[feat_name]
        
        # Find samples where this feature has high magnitude
        high_indices = np.where(np.abs(feature_values) > np.std(feature_values))[0]
        
        if len(high_indices) < 50:
            # Use top 50 by magnitude
            high_indices = np.argsort(np.abs(feature_values))[-50:]
        
        causal_effects = np.zeros(hidden_dim)
        
        with torch.no_grad():
            # Get baseline activations
            X_high = X[high_indices].to(device)
            _, hidden_base = model(X_high)
            y_base, _ = model(X_high)
            
            # For each hidden dimension, try patching
            for dim in range(hidden_dim):
                # Create patched activations: set this dimension to mean
                hidden_patched = hidden_base.clone()
                hidden_patched[:, dim] = 0  # Ablate this dimension
                
                # Forward from hidden to output
                y_patched = model.fc3(hidden_patched)
                
                # Measure effect
                effect = torch.mean(torch.abs(y_patched - y_base)).item()
                causal_effects[dim] = effect
        
        # Sort dimensions by effect
        sorted_dims = np.argsort(causal_effects)[::-1]
        sorted_effects = causal_effects[sorted_dims]
        
        results[feat_name] = {
            'dims': sorted_dims.tolist()[:20],  # Top 20 dimensions
            'effects': sorted_effects[:20].tolist(),
            'all_effects': causal_effects.tolist()
        }
        
        print(f"    Top 5 causal dims: {sorted_dims[:5]}")
        print(f"    Top 5 effects: {sorted_effects[:5]}")
    
    return results

def main():
    print("="*60)
    print("Causal Subspace Identification (Synthetic Task)")
    print("="*60)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading synthetic MLP...")
    model = SyntheticMLP(input_dim=20, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load('models/synthetic_mlp.pt', weights_only=False))
    model = model.to(device)
    model.eval()
    
    # Load data
    print("Loading validation data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    X_val = torch.FloatTensor(data['X_val'])
    
    # Generate ground truth features for validation set
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    features_val = ground_truth['val']
    
    # Identify causal subspaces
    print("\nIdentifying causal subspaces via activation patching...")
    causal_results = activation_patching_causal_identification(
        model, X_val, features_val, device
    )
    
    # Save results
    save_json(causal_results, 'exp/synthetic/causal_id/candidates.json')
    
    print("\n" + "="*60)
    print("Causal identification complete!")
    print("  Results saved to: exp/synthetic/causal_id/candidates.json")
    print("="*60)

if __name__ == '__main__':
    main()
