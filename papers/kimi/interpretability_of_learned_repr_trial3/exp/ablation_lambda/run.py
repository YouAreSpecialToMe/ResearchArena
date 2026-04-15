"""
Ablation study: Varying consistency loss weight lambda_consist.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/interpretability_of_learned_representations/idea_01')

import torch
import json
import os
import argparse

from exp.shared.models import RobustSAE
from exp.shared.metrics import evaluate_sae_model
from exp.shared.utils import set_seed, save_results
from torch.utils.data import DataLoader, TensorDataset

def add_activation_noise(x, noise_ratio=0.3):
    mask = torch.rand_like(x) > noise_ratio
    return x * mask

def train_with_lambda(lambda_consist, seed=42, device='cuda'):
    set_seed(seed)
    
    # Load data
    data = torch.load('data/activations_pythia70m_layer3.pt', weights_only=False)
    train_acts = data['train']
    val_acts = data['val']
    d_model = data['d_model']
    
    # Hyperparameters
    d_sae = d_model * 4
    topk = 32
    lr = 3e-4
    batch_size = 2048
    num_epochs = 10
    consistency_gamma = 0.5
    noise_ratio = 0.3
    
    model = RobustSAE(d_model, d_sae, topk=topk, 
                     lambda_consist=lambda_consist,
                     consistency_gamma=consistency_gamma,
                     use_proxy=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    train_loader = DataLoader(TensorDataset(train_acts), batch_size=batch_size, shuffle=True)
    
    best_fvu = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            batch_perturbed = add_activation_noise(batch, noise_ratio)
            
            (x_recon_orig, x_recon_pert), (z_orig, z_pert), consistency_components = \
                model.forward_with_consistency(batch, batch_perturbed)
            
            loss, loss_dict = model.get_loss(
                batch, batch_perturbed, x_recon_orig, x_recon_pert,
                z_orig, z_pert, consistency_components
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                model.W_dec.data = model.W_dec.data / model.W_dec.data.norm(dim=1, keepdim=True)
        
        # Evaluate
        metrics = evaluate_sae_model(model, val_acts, batch_size=batch_size, device=device)
        if metrics['fvu'] < best_fvu:
            best_fvu = metrics['fvu']
    
    final_metrics = evaluate_sae_model(model, val_acts, batch_size=batch_size, device=device)
    final_metrics['lambda_consist'] = lambda_consist
    final_metrics['seed'] = seed
    
    return final_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_value', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    results = train_with_lambda(args.lambda_value, seed=args.seed, device=args.device)
    
    os.makedirs('exp/ablation_lambda', exist_ok=True)
    save_results(results, f'exp/ablation_lambda/lambda_{args.lambda_value}.json')
    
    print(f"\nResults for lambda={args.lambda_value}:")
    print(f"  FVU: {results['fvu']:.6f}")
    print(f"  L0 Sparsity: {results['l0_sparsity']:.2f}")
