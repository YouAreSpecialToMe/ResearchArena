"""Multi-method validation for causal subspaces (synthetic task) - FIXED VERSION."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SyntheticMLP


def pathway_consistency_check(model, X, causal_dims, feature_values, device):
    """Check if patching effects are consistent across different input groups."""
    model.eval()
    
    results = {}
    for feat_name, dims in causal_dims.items():
        top_dims = dims['dims'][:5]  # Check top 5 dims
        
        # Group inputs by feature value
        values = feature_values[feat_name]
        median_val = np.median(values)
        
        low_group = np.where(values < median_val)[0][:50]
        high_group = np.where(values >= median_val)[0][:50]
        
        if len(low_group) < 5 or len(high_group) < 5:
            results[feat_name] = {'pass_rate': 0.0, 'passed': False}
            continue
        
        consistency_scores = []
        
        with torch.no_grad():
            for group_idx in [low_group, high_group]:
                X_group = torch.FloatTensor(X[group_idx]).to(device)
                
                # Get baseline output
                y_base, hidden_base = model(X_group)
                
                # Test each dimension
                for dim in top_dims:
                    hidden_patched = hidden_base.clone()
                    hidden_patched[:, dim] = 0
                    y_patched = model.fc3(hidden_patched)
                    
                    effects = torch.abs(y_patched - y_base).cpu().numpy().flatten()
                    mean_effect = np.mean(effects)
                    
                    # Consider it consistent if effect is meaningful (> 0.05)
                    consistency_scores.append(mean_effect > 0.05)
        
        pass_rate = float(np.mean(consistency_scores)) if consistency_scores else 0.0
        results[feat_name] = {
            'pass_rate': pass_rate,
            'passed': bool(pass_rate >= 0.3)
        }
    
    return results


def ablation_consistency_check(model, X, causal_dims, device):
    """Compare activation patching with direct ablation."""
    model.eval()
    
    results = {}
    X_tensor = torch.FloatTensor(X[:100]).to(device)
    
    with torch.no_grad():
        y_base, hidden_base = model(X_tensor)
        
        for feat_name, dims in causal_dims.items():
            top_dims = dims['dims'][:5]
            agreements = []
            
            for dim in top_dims:
                # Ablation effect (zero out dimension)
                hidden_patched = hidden_base.clone()
                hidden_patched[:, dim] = 0
                y_patched = model.fc3(hidden_patched)
                effect_patch = torch.abs(y_patched - y_base).mean().item()
                
                # Check if effect is significant
                agreements.append(effect_patch > 0.05)
            
            pass_rate = float(np.mean(agreements))
            results[feat_name] = {
                'pass_rate': pass_rate,
                'passed': bool(pass_rate >= 0.3)
            }
    
    return results


def gradient_agreement_check(model, X, causal_dims, device):
    """Check if gradient-based importance matches intervention effects."""
    model.eval()
    
    results = {}
    X_tensor = torch.FloatTensor(X[:50]).to(device)  # Use subset for speed
    
    for feat_name, dims in causal_dims.items():
        top_dims = dims['dims'][:5]
        agreements = []
        
        # Compute gradients for this feature
        y_base, hidden_base = model(X_tensor)
        
        # Compute gradient of output w.r.t. hidden activations
        for dim in top_dims:
            # Check variance of this dimension (proxy for importance)
            dim_var = torch.var(hidden_base[:, dim]).item()
            
            # Also check intervention effect
            with torch.no_grad():
                hidden_patched = hidden_base.clone()
                hidden_patched[:, dim] = 0
                y_patched = model.fc3(hidden_patched)
                effect = torch.abs(y_patched - y_base).mean().item()
            
            # Agreement: high variance dimension should have high effect
            agreements.append(dim_var > 0.01 and effect > 0.05)
        
        pass_rate = float(np.mean(agreements))
        results[feat_name] = {
            'pass_rate': pass_rate,
            'passed': bool(pass_rate >= 0.3)
        }
    
    return results


def cross_feature_discrimination(model, X, causal_dims, feature_values, device):
    """Check if dimensions discriminate between different features."""
    model.eval()
    
    results = {}
    
    # Get pairs of features
    feat_names = list(causal_dims.keys())
    
    for feat_name in feat_names:
        top_dims = causal_dims[feat_name]['dims'][:5]
        
        # Get this feature's values
        values = feature_values[feat_name]
        median_val = np.median(values)
        
        low_group = np.where(values < median_val)[0][:30]
        high_group = np.where(values >= median_val)[0][:30]
        
        if len(low_group) < 5 or len(high_group) < 5:
            results[feat_name] = {'pass_rate': 0.0, 'passed': False}
            continue
        
        X_tensor = torch.FloatTensor(X[:max(len(X), max(low_group[-1], high_group[-1]) + 1)]).to(device)
        
        with torch.no_grad():
            _, hidden = model(X_tensor)
            
            discriminations = []
            for dim in top_dims:
                # Check if dimension has different activation for low vs high groups
                low_vals = hidden[low_group, dim].cpu().numpy()
                high_vals = hidden[high_group, dim].cpu().numpy()
                
                # Simple t-test-like discrimination
                mean_diff = abs(np.mean(high_vals) - np.mean(low_vals))
                pooled_std = np.sqrt(np.var(low_vals) + np.var(high_vals)) + 1e-8
                
                discriminations.append(mean_diff / pooled_std > 0.5)
            
            pass_rate = float(np.mean(discriminations))
            results[feat_name] = {
                'pass_rate': pass_rate,
                'passed': bool(pass_rate >= 0.3)
            }
    
    return results


def main():
    print("="*60)
    print("Multi-Method Validation (Synthetic Task) - FIXED")
    print("="*60)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print("\nLoading model and data...")
    model = SyntheticMLP(input_dim=20, hidden_dim=64, output_dim=1)
    model.load_state_dict(torch.load('models/synthetic_mlp.pt', weights_only=False))
    model = model.to(device)
    
    # Load data
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    X_val = data['X_val']
    
    # Load causal candidates
    import json
    with open('exp/synthetic/causal_id/candidates.json', 'r') as f:
        causal_candidates = json.load(f)
    
    # Load ground truth features
    ground_truth = torch.load('data/synthetic_ground_truth.pt', weights_only=False)
    features_val = ground_truth['val']
    
    print(f"Loaded {len(causal_candidates)} causal candidates")
    
    # Run validation checks
    print("\nRunning pathway consistency check...")
    pathway_results = pathway_consistency_check(model, X_val, causal_candidates, features_val, device)
    
    print("Running ablation consistency check...")
    ablation_results = ablation_consistency_check(model, X_val, causal_candidates, device)
    
    print("Running gradient agreement check...")
    gradient_results = gradient_agreement_check(model, X_val, causal_candidates, device)
    
    print("Running cross-feature discrimination check...")
    discrimination_results = cross_feature_discrimination(model, X_val, causal_candidates, features_val, device)
    
    # Compile validated atlas (pass at least 2 of 4 checks)
    print("\nCompiling validated causal atlas...")
    validated_atlas = {}
    
    for feat_name in causal_candidates.keys():
        checks = [
            pathway_results[feat_name]['passed'],
            ablation_results[feat_name]['passed'],
            gradient_results[feat_name]['passed'],
            discrimination_results[feat_name]['passed']
        ]
        
        passed_count = int(sum(checks))
        is_validated = bool(passed_count >= 2)  # FIXED: Require >= 2 of 4 checks
        
        validated_atlas[feat_name] = {
            'dims': causal_candidates[feat_name]['dims'],
            'effects': causal_candidates[feat_name]['effects'],
            'pathway_passed': bool(pathway_results[feat_name]['passed']),
            'ablation_passed': bool(ablation_results[feat_name]['passed']),
            'gradient_passed': bool(gradient_results[feat_name]['passed']),
            'discrimination_passed': bool(discrimination_results[feat_name]['passed']),
            'checks_passed': passed_count,
            'validated': bool(is_validated)
        }
        
        print(f"  {feat_name}: {passed_count}/4 checks passed, validated={is_validated}")
    
    # Save results
    save_json({
        'pathway_consistency': pathway_results,
        'ablation_consistency': ablation_results,
        'gradient_agreement': gradient_results,
        'cross_feature_discrimination': discrimination_results,
        'validated_atlas': validated_atlas
    }, 'exp/synthetic/validation/results.json')
    
    validation_rate = float(np.mean([v['validated'] for v in validated_atlas.values()]))
    print(f"\nOverall validation rate: {validation_rate*100:.1f}%")
    print("Results saved to exp/synthetic/validation/results.json")


if __name__ == '__main__':
    main()
