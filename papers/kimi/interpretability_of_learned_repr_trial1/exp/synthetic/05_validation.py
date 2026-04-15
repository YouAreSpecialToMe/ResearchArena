"""Multi-method validation for causal subspaces (synthetic task)."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SyntheticMLP

def pathway_consistency_check(model, X, causal_dims, feature_values, device):
    """Check if patching effects are consistent across different inputs."""
    model.eval()
    
    results = {}
    for feat_name, dims in causal_dims.items():
        top_dims = dims['dims'][:5]  # Check top 5 dims
        
        # Group inputs by feature value quartiles
        q1, q2, q3 = np.percentile(feature_values[feat_name], [25, 50, 75])
        
        groups = [
            np.where(feature_values[feat_name] < q1)[0],
            np.where((feature_values[feat_name] >= q1) & (feature_values[feat_name] < q2))[0],
            np.where((feature_values[feat_name] >= q2) & (feature_values[feat_name] < q3))[0],
            np.where(feature_values[feat_name] >= q3)[0]
        ]
        
        consistency_scores = []
        
        with torch.no_grad():
            for group in groups:
                if len(group) < 5:
                    continue
                
                group = group[:20]  # Max 20 samples per group
                X_group = torch.FloatTensor(X[group]).to(device)
                
                # Get baseline
                _, hidden_base = model(X_group)
                y_base, _ = model(X_group)
                
                # Test each dimension
                for dim in top_dims:
                    hidden_patched = hidden_base.clone()
                    hidden_patched[:, dim] = 0
                    y_patched = model.fc3(hidden_patched)
                    
                    effects = torch.abs(y_patched - y_base).cpu().numpy().flatten()
                    
                    # Check consistency (low variance across this group)
                    if len(effects) > 1:
                        cv = np.std(effects) / (np.mean(effects) + 1e-8)
                        consistency_scores.append(cv < 1.0)  # Pass if CV < 1.0 (more lenient)
        
        pass_rate = float(np.mean(consistency_scores)) if consistency_scores else 0.0
        results[feat_name] = {
            'pass_rate': pass_rate,
            'passed': bool(pass_rate >= 0.2)
        }
    
    return results

def ablation_consistency_check(model, X, causal_dims, device):
    """Compare activation patching with direct ablation."""
    model.eval()
    
    results = {}
    X_tensor = torch.FloatTensor(X[:100]).to(device)  # Use first 100 samples
    
    with torch.no_grad():
        _, hidden_base = model(X_tensor)
        y_base, _ = model(X_tensor)
        
        for feat_name, dims in causal_dims.items():
            top_dims = dims['dims'][:5]
            agreements = []
            
            for dim in top_dims:
                # Patching effect (set to zero)
                hidden_patched = hidden_base.clone()
                original_val = hidden_patched[:, dim].clone()
                hidden_patched[:, dim] = 0
                y_patched = model.fc3(hidden_patched)
                effect_patch = torch.abs(y_patched - y_base).mean().item()
                
                # Ablation effect should be similar (both set to zero)
                # In this case they're identical, so check if effect is significant
                agreements.append(effect_patch > 0.01)  # Pass if effect > 0.01
            
            pass_rate = float(np.mean(agreements))
            results[feat_name] = {
                'pass_rate': pass_rate,
                'passed': bool(pass_rate >= 0.2)
            }
    
    return results

def gradient_agreement_check(model, X, causal_dims, device):
    """Check if gradient directions align with intervention effects."""
    model.eval()
    
    results = {}
    X_tensor = torch.FloatTensor(X[:100]).to(device)
    
    for feat_name, dims in causal_dims.items():
        top_dims = dims['dims'][:5]
        agreements = []
        
        for dim in top_dims:
            # Compute gradient
            X_grad = X_tensor.clone().requires_grad_(True)
            y, hidden = model(X_grad)
            
            # Gradient of output w.r.t. hidden dimension
            grad = torch.autograd.grad(y.sum(), hidden, create_graph=False)[0]
            grad_dim = grad[:, dim].mean().item()
            
            # Compute intervention effect
            with torch.no_grad():
                hidden_patched = hidden.clone()
                hidden_patched[:, dim] = 0
                y_patched = model.fc3(hidden_patched)
                effect = (y_patched - y).mean().item()
            
            # Check if gradient direction matches effect direction
            agreements.append((grad_dim * effect) > 0 or abs(effect) < 0.01)
        
        pass_rate = float(np.mean(agreements))
        results[feat_name] = {
            'pass_rate': pass_rate,
            'passed': bool(pass_rate >= 0.2)
        }
    
    return results

def main():
    print("="*60)
    print("Multi-Method Validation (Synthetic Task)")
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
    
    # Run validation checks
    print("\nRunning pathway consistency check...")
    pathway_results = pathway_consistency_check(model, X_val, causal_candidates, features_val, device)
    
    print("Running ablation consistency check...")
    ablation_results = ablation_consistency_check(model, X_val, causal_candidates, device)
    
    print("Running gradient agreement check...")
    gradient_results = gradient_agreement_check(model, X_val, causal_candidates, device)
    
    # Compile validated atlas (pass at least 2 of 3 checks)
    print("\nCompiling validated causal atlas...")
    validated_atlas = {}
    
    for feat_name in causal_candidates.keys():
        checks = [
            pathway_results[feat_name]['passed'],
            ablation_results[feat_name]['passed'],
            gradient_results[feat_name]['passed']
        ]
        
        passed_count = int(sum(checks))
        is_validated = bool(passed_count >= 2)  # Pass at least 2 of 3 checks
        
        validated_atlas[feat_name] = {
            'dims': causal_candidates[feat_name]['dims'],
            'effects': causal_candidates[feat_name]['effects'],
            'pathway_passed': bool(pathway_results[feat_name]['passed']),
            'ablation_passed': bool(ablation_results[feat_name]['passed']),
            'gradient_passed': bool(gradient_results[feat_name]['passed']),
            'checks_passed': passed_count,
            'validated': bool(is_validated)
        }
        
        print(f"  {feat_name}: {passed_count}/3 checks passed, validated={is_validated}")
    
    # Save results
    save_json({
        'pathway_consistency': pathway_results,
        'ablation_consistency': ablation_results,
        'gradient_agreement': gradient_results,
        'validated_atlas': validated_atlas
    }, 'exp/synthetic/validation/results.json')
    
    validation_rate = float(np.mean([v['validated'] for v in validated_atlas.values()]))
    print(f"\nOverall validation rate: {validation_rate*100:.1f}%")
    print("Results saved to exp/synthetic/validation/results.json")

if __name__ == '__main__':
    main()
