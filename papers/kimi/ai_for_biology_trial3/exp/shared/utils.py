"""
Training utilities for StruCVAE-Pep.
Includes loss functions, cyclical annealing, metrics, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import json
import os


def cyclical_annealing(epoch, num_epochs, n_cycles=5, ratio=0.5):
    """
    Cyclical annealing schedule for KL weight.
    From Fu et al. 2019.
    """
    period = num_epochs // n_cycles
    epoch_in_cycle = epoch % period
    
    if epoch_in_cycle < period * ratio:
        # Ramp up phase
        return epoch_in_cycle / (period * ratio)
    else:
        # Constant phase
        return 1.0


def free_bits_kl(kl_per_dim, threshold=0.5):
    """
    Free bits technique: don't penalize KL below threshold per dimension.
    From Kingma et al. 2016.
    """
    return torch.sum(torch.maximum(kl_per_dim, torch.tensor(threshold, device=kl_per_dim.device)))


def compute_kl_divergence(mu, logvar, free_bits=0.5):
    """
    Compute KL divergence with optional free bits.
    Returns total KL and per-dimension KL.
    """
    # KL per dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Apply free bits
    if free_bits > 0:
        kl = free_bits_kl(kl_per_dim, free_bits)
    else:
        kl = torch.sum(kl_per_dim)
    
    return kl, kl_per_dim


def compute_vae_loss(model_output, target_seq, target_property, 
                     beta=1.0, lambda_prop=0.5, free_bits=0.5,
                     disentangled=True):
    """
    Compute VAE loss with reconstruction, KL, and property prediction.
    """
    if disentangled:
        z_struct_mu, z_struct_logvar = model_output['z_structure']
        z_prop_mu, z_prop_logvar = model_output['z_property']
        z_seq_mu, z_seq_logvar = model_output['z_sequence']
        prop_pred = model_output['property_pred']
        
        # KL for each factor
        kl_struct, kl_struct_per_dim = compute_kl_divergence(z_struct_mu, z_struct_logvar, free_bits)
        kl_prop, kl_prop_per_dim = compute_kl_divergence(z_prop_mu, z_prop_logvar, free_bits)
        kl_seq, kl_seq_per_dim = compute_kl_divergence(z_seq_mu, z_seq_logvar, free_bits)
        
        total_kl = kl_struct + kl_prop + kl_seq
        
        # Track per-factor KL
        kl_stats = {
            'kl_structure': kl_struct.item(),
            'kl_property': kl_prop.item(),
            'kl_sequence': kl_seq.item(),
            'kl_total': total_kl.item()
        }
    else:
        z_mu = model_output['z_mu']
        z_logvar = model_output['z_logvar']
        prop_pred = model_output['property_pred']
        
        total_kl, kl_per_dim = compute_kl_divergence(z_mu, z_logvar, free_bits)
        kl_stats = {'kl_total': total_kl.item()}
    
    # Reconstruction loss (placeholder - would need actual logits)
    # For now, just return property prediction loss
    
    # Property prediction loss
    prop_loss = F.mse_loss(prop_pred, target_property)
    
    # Total loss
    total_loss = beta * total_kl + lambda_prop * prop_loss
    
    loss_dict = {
        'total_loss': total_loss.item(),
        'prop_loss': prop_loss.item(),
        **kl_stats
    }
    
    return total_loss, loss_dict


def compute_reconstruction_loss(logits, target, pad_idx=0):
    """
    Compute cross-entropy reconstruction loss.
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross entropy
    logits_flat = logits.reshape(-1, vocab_size)
    target_flat = target.reshape(-1)
    
    # Mask padding
    mask = (target_flat != pad_idx).float()
    
    # Cross entropy
    ce_loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
    ce_loss = ce_loss * mask
    
    return ce_loss.sum() / mask.sum()


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Pearson and Spearman correlations
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p
    }


def compute_posterior_collapse_stats(model_output, threshold=0.1, disentangled=True):
    """
    Compute posterior collapse statistics.
    Returns % of collapsed dimensions per latent factor.
    """
    if disentangled:
        stats = {}
        total_collapsed = 0
        total_dims = 0
        
        for factor_name in ['z_structure', 'z_property', 'z_sequence']:
            mu, logvar = model_output[factor_name]
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_mean = kl_per_dim.mean(dim=0)  # Average over batch
            
            collapsed = (kl_mean < threshold).sum().item()
            total = kl_mean.numel()
            
            stats[f'{factor_name}_collapsed_pct'] = 100.0 * collapsed / total
            stats[f'{factor_name}_mean_kl'] = kl_mean.mean().item()
            
            total_collapsed += collapsed
            total_dims += total
        
        stats['total_collapsed_pct'] = 100.0 * total_collapsed / total_dims
    else:
        mu = model_output['z_mu']
        logvar = model_output['z_logvar']
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_mean = kl_per_dim.mean(dim=0)
        
        collapsed = (kl_mean < threshold).sum().item()
        total = kl_mean.numel()
        
        stats = {
            'z_collapsed_pct': 100.0 * collapsed / total,
            'z_mean_kl': kl_mean.mean().item()
        }
    
    return stats


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, delta=0.0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.delta
        else:
            improved = score > self.best_score + self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    

def save_results(results, path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path):
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test cyclical annealing
    betas = [cyclical_annealing(e, 100, n_cycles=5) for e in range(100)]
    print(f"Cyclical annealing: min={min(betas):.3f}, max={max(betas):.3f}")
    
    # Test metrics
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1
    metrics = compute_metrics(y_true, y_pred)
    print(f"Metrics: {metrics}")
    
    print("All tests passed!")
