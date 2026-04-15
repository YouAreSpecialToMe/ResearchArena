"""
Train RobustSAE with consistency regularization.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import argparse

from exp.shared.models import RobustSAE
from exp.shared.metrics import evaluate_sae_model
from exp.shared.utils import set_seed, save_results, save_checkpoint

def add_activation_noise(x, noise_ratio=0.3):
    """Add random dropout noise to activations."""
    mask = torch.rand_like(x) > noise_ratio
    return x * mask

def train_robust_sae(seed=42, device='cuda', lambda_consist=0.1, use_proxy=True):
    set_seed(seed)
    
    # Load data
    data = torch.load('data/activations_pythia70m_layer3.pt', weights_only=False)
    train_acts = data['train']
    val_acts = data['val']
    d_model = data['d_model']
    
    print(f"Data loaded: train={train_acts.shape}, val={val_acts.shape}")
    
    # Hyperparameters
    d_sae = d_model * 4
    topk = 32
    lr = 3e-4
    batch_size = 2048  # Reduced due to double forward pass
    num_epochs = 10
    consistency_gamma = 0.5
    noise_ratio = 0.3
    
    # Create model
    model = RobustSAE(d_model, d_sae, topk=topk, 
                     lambda_consist=lambda_consist,
                     consistency_gamma=consistency_gamma,
                     use_proxy=use_proxy).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Create dataloaders
    train_loader = DataLoader(TensorDataset(train_acts), batch_size=batch_size, shuffle=True)
    
    # Training loop
    best_fvu = float('inf')
    train_losses = []
    consistency_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_consistency = 0
        
        for (batch,) in train_loader:
            batch = batch.to(device)
            
            # Create perturbed version (activation-level noise)
            batch_perturbed = add_activation_noise(batch, noise_ratio)
            
            # Forward pass with consistency
            (x_recon_orig, x_recon_pert), (z_orig, z_pert), consistency_components = \
                model.forward_with_consistency(batch, batch_perturbed)
            
            # Compute loss
            loss, loss_dict = model.get_loss(
                batch, batch_perturbed, x_recon_orig, x_recon_pert,
                z_orig, z_pert, consistency_components
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize decoder weights
            with torch.no_grad():
                model.W_dec.data = model.W_dec.data / model.W_dec.data.norm(dim=1, keepdim=True)
            
            epoch_loss += loss_dict['total_loss']
            epoch_consistency += loss_dict['consistency_loss']
        
        avg_loss = epoch_loss / len(train_loader)
        avg_consistency = epoch_consistency / len(train_loader)
        train_losses.append(avg_loss)
        consistency_losses.append(avg_consistency)
        
        # Evaluate
        metrics = evaluate_sae_model(model, val_acts, batch_size=batch_size, device=device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}, "
              f"Consist: {avg_consistency:.6f}, FVU: {metrics['fvu']:.6f}, "
              f"L0: {metrics['l0_sparsity']:.2f}")
        
        # Save best model
        if metrics['fvu'] < best_fvu:
            best_fvu = metrics['fvu']
            save_checkpoint(model, optimizer, epoch, best_fvu,
                          f'models/robustsae_full_seed{seed}_best.pt')
    
    # Compute proxy scores if enabled
    proxy_scores = None
    if use_proxy:
        model.eval()
        with torch.no_grad():
            # Sample subset for proxy computation
            sample_size = min(1000, len(val_acts))
            sample = val_acts[:sample_size].to(device)
            proxy_scores = model.compute_proxy_scores(sample)
    
    # Final evaluation
    final_metrics = evaluate_sae_model(model, val_acts, batch_size=batch_size, device=device)
    final_metrics['train_losses'] = train_losses
    final_metrics['consistency_losses'] = consistency_losses
    final_metrics['seed'] = seed
    final_metrics['config'] = {
        'd_model': d_model,
        'd_sae': d_sae,
        'topk': topk,
        'lambda_consist': lambda_consist,
        'consistency_gamma': consistency_gamma,
        'use_proxy': use_proxy,
        'lr': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs
    }
    
    if proxy_scores is not None:
        final_metrics['proxy_scores'] = {
            'mean': proxy_scores.mean().item(),
            'std': proxy_scores.std().item(),
            'max': proxy_scores.max().item(),
            'min': proxy_scores.min().item(),
            'values': proxy_scores.cpu().numpy().tolist()
        }
    
    # Save results
    os.makedirs('exp/robustsae_full', exist_ok=True)
    save_results(final_metrics, f'exp/robustsae_full/results_seed{seed}.json')
    
    # Save final model
    save_checkpoint(model, optimizer, num_epochs, best_fvu,
                   f'models/robustsae_full_seed{seed}_final.pt')
    
    print(f"\nFinal metrics (seed {seed}):")
    print(f"  FVU: {final_metrics['fvu']:.6f}")
    print(f"  L0 Sparsity: {final_metrics['l0_sparsity']:.2f}")
    print(f"  Dead Features: {final_metrics['dead_features_pct']:.2f}%")
    
    return final_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lambda_consist', type=float, default=0.1)
    parser.add_argument('--use_proxy', type=bool, default=True)
    args = parser.parse_args()
    
    train_robust_sae(seed=args.seed, device=args.device, 
                    lambda_consist=args.lambda_consist, use_proxy=args.use_proxy)
