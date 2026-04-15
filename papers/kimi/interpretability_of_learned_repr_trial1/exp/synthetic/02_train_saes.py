"""Train Sparse Autoencoders on synthetic MLP activations."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import numpy as np
import os
import json

from exp.shared.utils import set_seed, save_json
from exp.shared.models import SparseAutoencoder, train_sae

def train_sae_for_seed(seed: int, dict_size: int, hidden_train: torch.Tensor, 
                       hidden_val: torch.Tensor, device: str) -> dict:
    """Train SAE for a specific seed and dictionary size."""
    set_seed(seed)
    
    input_dim = hidden_train.shape[1]
    overcomplete = dict_size // input_dim
    
    print(f"\n  Training SAE {overcomplete}x (dict_size={dict_size}) with seed={seed}...")
    
    # Create model
    model = SparseAutoencoder(
        input_dim=input_dim,
        dict_size=dict_size,
        sparsity_penalty=1e-4,
        tied_weights=False
    )
    
    # Train
    history = train_sae(
        model=model,
        activations=hidden_train,
        val_activations=hidden_val,
        epochs=500,
        batch_size=256,
        lr=1e-3,
        early_stopping_patience=20,
        device=device
    )
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        hidden_val_device = hidden_val.to(device)
        features_val = model.encode(hidden_val_device).cpu().numpy()
        recon_val, _ = model(hidden_val_device)
        recon_error = torch.mean((recon_val - hidden_val_device) ** 2).item()
    
    # Compute sparsity metrics
    l0_sparsity = np.mean(features_val < 1e-6)
    l1_sparsity = np.mean(np.sum(np.abs(features_val), axis=1))
    dead_neurons = np.mean(np.all(np.abs(features_val) < 1e-6, axis=0))
    
    result = {
        'seed': seed,
        'dict_size': dict_size,
        'overcomplete': overcomplete,
        'recon_error': recon_error,
        'l0_sparsity': float(l0_sparsity),
        'l1_sparsity': float(l1_sparsity),
        'dead_neurons_pct': float(dead_neurons) * 100,
        'final_train_loss': history['train_loss'][-1],
        'history': history
    }
    
    # Save model
    save_path = f'models/sae_synthetic_{overcomplete}x_seed{seed}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'features_val': features_val,
        'result': result
    }, save_path)
    
    print(f"    Recon error: {recon_error:.6f}, L0 sparsity: {l0_sparsity:.4f}, Dead neurons: {dead_neurons*100:.2f}%")
    
    return result

def main():
    print("="*60)
    print("Training Sparse Autoencoders on Synthetic Data")
    print("="*60)
    
    # Configuration
    SEEDS = [42, 123, 456]
    DICT_SIZES = [64, 256, 1024]  # 1x, 4x, 16x overcomplete
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading synthetic data...")
    data = torch.load('data/synthetic_data.pt', weights_only=False)
    hidden_train = torch.FloatTensor(data['hidden_train'])
    hidden_val = torch.FloatTensor(data['hidden_val'])
    
    print(f"  Train activations: {hidden_train.shape}")
    print(f"  Val activations: {hidden_val.shape}")
    
    # Train SAEs for each configuration
    all_results = {}
    
    for dict_size in DICT_SIZES:
        overcomplete = dict_size // 64
        all_results[f'{overcomplete}x'] = []
        
        print(f"\n{'='*60}")
        print(f"Training SAE {overcomplete}x overcomplete (dict_size={dict_size})")
        print('='*60)
        
        for seed in SEEDS:
            result = train_sae_for_seed(seed, dict_size, hidden_train, hidden_val, device)
            all_results[f'{overcomplete}x'].append(result)
    
    # Compute statistics across seeds
    summary = {}
    for key, results in all_results.items():
        summary[key] = {
            'recon_error_mean': np.mean([r['recon_error'] for r in results]),
            'recon_error_std': np.std([r['recon_error'] for r in results]),
            'l0_sparsity_mean': np.mean([r['l0_sparsity'] for r in results]),
            'l1_sparsity_mean': np.mean([r['l1_sparsity'] for r in results]),
            'dead_neurons_mean': np.mean([r['dead_neurons_pct'] for r in results]),
        }
    
    # Save summary
    save_json({'all_results': all_results, 'summary': summary}, 
              'exp/synthetic/sae/results.json')
    
    print("\n" + "="*60)
    print("SAE Training Summary:")
    print("="*60)
    for key, stats in summary.items():
        print(f"\nSAE {key}:")
        print(f"  Recon error: {stats['recon_error_mean']:.6f} ± {stats['recon_error_std']:.6f}")
        print(f"  L0 sparsity: {stats['l0_sparsity_mean']:.4f}")
        print(f"  Dead neurons: {stats['dead_neurons_mean']:.2f}%")
    
    print("\nResults saved to exp/synthetic/sae/results.json")

if __name__ == '__main__':
    main()
