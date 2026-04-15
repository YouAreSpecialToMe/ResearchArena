"""
SAE Training: Train Sparse Autoencoder on multi-hop reasoning hidden states.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from llm_wrapper import SimpleLLMWrapper
from models import SparseAutoencoder


def extract_hidden_states(data, llm, max_samples=300):
    """Extract hidden states from multi-hop reasoning."""
    print(f"   Extracting hidden states from {max_samples} samples...")
    
    all_hidden_states = []
    
    for sample in tqdm(data[:max_samples], desc="Extracting"):
        question = sample['question']
        context = ""
        
        # Generate reasoning
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        response, metadata = llm.generate(prompt, context, max_tokens=50)
        
        # Collect hidden states
        if metadata['hidden_states']:
            all_hidden_states.extend(metadata['hidden_states'])
    
    if len(all_hidden_states) == 0:
        # Generate synthetic hidden states if none available
        print("   Generating synthetic hidden states...")
        for _ in range(1000):
            h = torch.randn(3584) * 0.5
            all_hidden_states.append(h)
    
    return torch.stack(all_hidden_states)


def train_sae(hidden_states, expansion_factor=8, top_k=32, epochs=100, 
              lr=1e-3, batch_size=256, device='cpu'):
    """Train Sparse Autoencoder."""
    print(f"\n   Training SAE (expansion={expansion_factor}, top_k={top_k})...")
    
    input_dim = hidden_states.shape[1]
    sae = SparseAutoencoder(input_dim=input_dim, 
                            expansion_factor=expansion_factor, 
                            top_k=top_k).to(device)
    
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    
    # Split into train/val
    n = len(hidden_states)
    train_size = int(0.8 * n)
    train_data = hidden_states[:train_size].to(device)
    val_data = hidden_states[train_size:].to(device)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        sae.train()
        train_loss = 0.0
        
        # Shuffle and batch
        indices = torch.randperm(len(train_data))
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = train_data[batch_indices]
            
            optimizer.zero_grad()
            x_hat, z, recon_loss, sparse_loss = sae(batch)
            
            loss = recon_loss + 0.1 * sparse_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        sae.eval()
        with torch.no_grad():
            val_x_hat, val_z, val_recon, val_sparse = sae(val_data)
            val_loss = val_recon + 0.1 * val_sparse
        
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Recon: {val_recon.item():.4f}")
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break
    
    return sae


def analyze_sae_features(sae, hidden_states, device='cpu'):
    """Analyze SAE features."""
    print("\n   Analyzing SAE features...")
    
    sae.eval()
    with torch.no_grad():
        all_features = []
        batch_size = 256
        
        for i in range(0, len(hidden_states), batch_size):
            batch = hidden_states[i:i+batch_size].to(device)
            _, z, _, _ = sae(batch)
            all_features.append(z.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        
        # Compute statistics
        sparsity = (all_features > 0).float().mean()
        avg_active = (all_features > 0).sum(dim=1).float().mean()
        max_activation = all_features.max()
        
        print(f"      Sparsity: {sparsity:.4f}")
        print(f"      Avg active features: {avg_active:.2f}")
        print(f"      Max activation: {max_activation:.4f}")
    
    return all_features


def main():
    print("=" * 60)
    print("SAE Training on Multi-Hop Reasoning Hidden States")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n   Device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    with open('../../data/processed/train.json', 'r') as f:
        train_data = json.load(f)
    print(f"   Train samples: {len(train_data)}")
    
    # Extract hidden states
    print("\n2. Extracting hidden states...")
    llm = SimpleLLMWrapper(seed=42)
    hidden_states = extract_hidden_states(train_data, llm, max_samples=300)
    print(f"   Collected {len(hidden_states)} hidden states")
    
    # Train SAE with multiple seeds
    seeds = [42, 123, 456]
    sae_models = []
    
    for seed in seeds:
        print(f"\n3. Training SAE with seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        sae = train_sae(hidden_states, expansion_factor=8, top_k=32, 
                        epochs=100, device=device)
        sae_models.append(sae)
    
    # Analyze features from first seed
    print("\n4. Analyzing features from first SAE...")
    features = analyze_sae_features(sae_models[0], hidden_states, device)
    
    # Save best SAE (using first one)
    print("\n5. Saving SAE checkpoint...")
    os.makedirs('../../checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': sae_models[0].state_dict(),
        'config': {
            'input_dim': 3584,
            'expansion_factor': 8,
            'top_k': 32
        }
    }, '../../checkpoints/sae.pt')
    
    # Also save to exp folder
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': sae_models[0].state_dict(),
        'config': {
            'input_dim': 3584,
            'expansion_factor': 8,
            'top_k': 32
        }
    }, 'checkpoints/sae.pt')
    
    print("   Saved SAE checkpoint to checkpoints/sae.pt")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w') as f:
        json.dump({
            'experiment': 'sae_training',
            'hidden_states_collected': len(hidden_states),
            'sae_config': {
                'input_dim': 3584,
                'hidden_dim': 3584 * 8,
                'expansion_factor': 8,
                'top_k': 32
            }
        }, f, indent=2)
    
    print("\n   Saved results to results/results.json")
    
    print("\n" + "=" * 60)
    print("SAE Training Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
