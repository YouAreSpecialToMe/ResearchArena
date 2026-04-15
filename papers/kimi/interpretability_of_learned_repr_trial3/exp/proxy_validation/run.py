"""
Validate unsupervised robustness proxy.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
import numpy as np
from scipy import stats

from exp.shared.models import RobustSAE
from exp.shared.utils import save_results

def load_robust_sae(checkpoint_path, device='cuda'):
    """Load RobustSAE model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    d_model = checkpoint['model_state_dict']['W_enc'].shape[0]
    d_sae = checkpoint['model_state_dict']['W_enc'].shape[1]
    
    model = RobustSAE(d_model, d_sae, topk=32)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def compute_proxy_validation(device='cuda'):
    # Load data
    data = torch.load('data/activations_pythia70m_layer3.pt', weights_only=False)
    val_acts = data['val'][:1000].to(device)  # Use subset
    
    # Load RobustSAE model
    model = load_robust_sae('models/robustsae_full_seed42_best.pt', device)
    
    # Compute proxy scores
    print("Computing proxy scores...")
    proxy_scores = model.compute_proxy_scores(val_acts)
    
    # Simulate empirical robustness: measure activation variance under noise
    print("Computing empirical robustness...")
    empirical_robustness = []
    
    with torch.no_grad():
        for feat_idx in range(min(100, model.d_sae)):  # Sample features
            # Get baseline activation
            z_orig, _ = model.encode(val_acts)
            baseline_act = z_orig[:, feat_idx].mean().item()
            
            # Add noise and measure change
            noise_levels = [0.1, 0.2, 0.3]
            max_changes = []
            
            for noise in noise_levels:
                noise_mask = torch.rand_like(val_acts) > noise
                val_noisy = val_acts * noise_mask
                z_noisy, _ = model.encode(val_noisy)
                
                act_change = torch.abs(z_noisy[:, feat_idx] - z_orig[:, feat_idx]).mean().item()
                max_changes.append(act_change)
            
            # Feature with higher changes under noise = less robust
            avg_change = np.mean(max_changes)
            empirical_robustness.append(avg_change)
    
    empirical_robustness = torch.tensor(empirical_robustness, device=device)
    proxy_scores_subset = proxy_scores[:len(empirical_robustness)]
    
    # Compute correlations
    proxy_np = proxy_scores_subset.cpu().numpy()
    empirical_np = empirical_robustness.cpu().numpy()
    
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(proxy_np, empirical_np)
    
    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(proxy_np, empirical_np)
    
    # Kendall tau
    kendall_tau, kendall_p = stats.kendalltau(proxy_np, empirical_np)
    
    results = {
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'kendall_tau': float(kendall_tau),
        'kendall_p': float(kendall_p),
        'proxy_scores_stats': {
            'mean': float(proxy_scores.mean()),
            'std': float(proxy_scores.std()),
            'min': float(proxy_scores.min()),
            'max': float(proxy_scores.max())
        },
        'n_features_tested': len(empirical_robustness)
    }
    
    print("\nProxy Validation Results:")
    print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"  Kendall tau: {kendall_tau:.4f} (p={kendall_p:.4f})")
    
    save_results(results, 'results/proxy_validation.json')
    print("\nResults saved to results/proxy_validation.json")
    
    return results

if __name__ == '__main__':
    compute_proxy_validation()
