"""Generate synthetic ground-truth dataset for validating C-GAS metric."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from exp.shared.utils import set_seed, save_json
from exp.shared.data_loader import generate_synthetic_dataset
from exp.shared.models import SyntheticMLP

def main():
    print("="*60)
    print("Generating Synthetic Ground-Truth Dataset")
    print("="*60)
    
    # Configuration
    SEED = 42
    N_TRAIN = 10000
    N_VAL = 1000
    HIDDEN_DIM = 64
    
    set_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate training data
    print(f"\nGenerating {N_TRAIN} training samples...")
    X_train, y_train, features_train = generate_synthetic_dataset(N_TRAIN, seed=SEED)
    
    # Generate validation data
    print(f"Generating {N_VAL} validation samples...")
    X_val, y_val, features_val = generate_synthetic_dataset(N_VAL, seed=SEED + 1)
    
    # Create model
    print(f"\nCreating MLP with hidden_dim={HIDDEN_DIM}...")
    model = SyntheticMLP(input_dim=20, hidden_dim=HIDDEN_DIM, output_dim=1)
    model = model.to(device)
    
    # Train model
    print("\nTraining MLP...")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    n_epochs = 100
    batch_size = 128
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        
        # Shuffle
        indices = torch.randperm(N_TRAIN)
        
        for i in range(0, N_TRAIN, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred, _ = model(X_val_t)
            val_loss = criterion(y_val_pred, y_val_t).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={np.mean(train_losses):.4f}, val_loss={val_loss:.4f}")
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    # Extract hidden activations
    print("\nExtracting hidden activations...")
    model.eval()
    with torch.no_grad():
        _, hidden_train = model(X_train_t)
        _, hidden_val = model(X_val_t)
    
    hidden_train_np = hidden_train.cpu().numpy()
    hidden_val_np = hidden_val.cpu().numpy()
    
    # Save ground truth features
    print("\nSaving ground truth features...")
    ground_truth = {
        'train': {
            'f1': features_train['f1'],
            'f2': features_train['f2'],
            'f3': features_train['f3'],
            'f4': features_train['f4'],
            'f5': features_train['f5'],
        },
        'val': {
            'f1': features_val['f1'],
            'f2': features_val['f2'],
            'f3': features_val['f3'],
            'f4': features_val['f4'],
            'f5': features_val['f5'],
        }
    }
    
    # Save everything
    torch.save({
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'hidden_train': hidden_train_np,
        'hidden_val': hidden_val_np,
    }, 'data/synthetic_data.pt')
    
    torch.save(ground_truth, 'data/synthetic_ground_truth.pt')
    
    # Save model
    torch.save(model.state_dict(), 'models/synthetic_mlp.pt')
    
    # Save metadata
    metadata = {
        'n_train': N_TRAIN,
        'n_val': N_VAL,
        'input_dim': 20,
        'hidden_dim': HIDDEN_DIM,
        'output_dim': 1,
        'seed': SEED,
        'final_val_loss': best_val_loss,
        'causal_features': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'noise_features': 15,
    }
    save_json(metadata, 'data/synthetic_metadata.json')
    
    print("\n" + "="*60)
    print("Synthetic dataset generation complete!")
    print("  - Data: data/synthetic_data.pt")
    print("  - Ground truth: data/synthetic_ground_truth.pt")
    print("  - Model: models/synthetic_mlp.pt")
    print("="*60)

if __name__ == '__main__':
    main()
