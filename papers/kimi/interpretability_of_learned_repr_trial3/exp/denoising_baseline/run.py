"""
Train Denoising SAE baseline (Pan et al. 2025 replication).
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import os
import argparse

from exp.shared.models import DenoisingSAE
from exp.shared.metrics import evaluate_sae_model
from exp.shared.utils import set_seed, save_results, save_checkpoint

def train_denoising_sae(seed=42, device='cuda'):
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
    dropout_rate = 0.3
    lr = 3e-4
    batch_size = 4096
    num_epochs = 10
    
    # Create model
    model = DenoisingSAE(d_model, d_sae, topk=topk, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Create dataloaders
    train_loader = DataLoader(TensorDataset(train_acts), batch_size=batch_size, shuffle=True)
    
    # Training loop
    best_fvu = float('inf')
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for (batch,) in train_loader:
            batch = batch.to(device)
            
            # Forward (includes denoising)
            x_recon, z, z_pre = model(batch)
            
            # Loss - reconstruct clean input
            loss, loss_dict = model.get_loss(batch, x_recon, z, z_pre)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize decoder weights
            with torch.no_grad():
                model.W_dec.data = model.W_dec.data / model.W_dec.data.norm(dim=1, keepdim=True)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate (without denoising)
        model.eval()
        metrics = evaluate_sae_model(model, val_acts, batch_size=batch_size, device=device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}, "
              f"FVU: {metrics['fvu']:.6f}, L0: {metrics['l0_sparsity']:.2f}")
        
        # Save best model
        if metrics['fvu'] < best_fvu:
            best_fvu = metrics['fvu']
            save_checkpoint(model, optimizer, epoch, best_fvu,
                          f'models/denoising_baseline_seed{seed}_best.pt')
    
    # Final evaluation
    final_metrics = evaluate_sae_model(model, val_acts, batch_size=batch_size, device=device)
    final_metrics['train_losses'] = train_losses
    final_metrics['seed'] = seed
    final_metrics['config'] = {
        'd_model': d_model,
        'd_sae': d_sae,
        'topk': topk,
        'dropout_rate': dropout_rate,
        'lr': lr,
        'batch_size': batch_size,
        'num_epochs': num_epochs
    }
    
    # Save results
    os.makedirs('exp/denoising_baseline', exist_ok=True)
    save_results(final_metrics, f'exp/denoising_baseline/results_seed{seed}.json')
    
    # Save final model
    save_checkpoint(model, optimizer, num_epochs, best_fvu,
                   f'models/denoising_baseline_seed{seed}_final.pt')
    
    print(f"\nFinal metrics (seed {seed}):")
    print(f"  FVU: {final_metrics['fvu']:.6f}")
    print(f"  L0 Sparsity: {final_metrics['l0_sparsity']:.2f}")
    print(f"  Dead Features: {final_metrics['dead_features_pct']:.2f}%")
    
    return final_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train_denoising_sae(seed=args.seed, device=args.device)
