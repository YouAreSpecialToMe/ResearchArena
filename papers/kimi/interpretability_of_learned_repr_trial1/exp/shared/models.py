"""Model implementations for CAGER experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for learning interpretable features."""
    
    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        sparsity_penalty: float = 1e-4,
        tied_weights: bool = False
    ):
        """
        Args:
            input_dim: Dimensionality of input activations
            dict_size: Size of dictionary (number of features)
            sparsity_penalty: L1 sparsity penalty coefficient
            tied_weights: Whether to tie encoder/decoder weights
        """
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_penalty = sparsity_penalty
        self.tied_weights = tied_weights
        
        # Encoder: input -> dictionary
        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        
        # Decoder: dictionary -> input
        if not tied_weights:
            self.decoder = nn.Linear(dict_size, input_dim, bias=True)
        else:
            # Will use encoder weight transpose
            pass
        
        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        if not tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
        nn.init.zeros_(self.encoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features."""
        return F.relu(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode features to input reconstruction."""
        if self.tied_weights:
            return F.linear(z, self.encoder.weight.t())
        else:
            return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (reconstruction, features)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss with reconstruction and sparsity.
        
        Returns:
            (total_loss, loss_dict)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Sparsity loss (L1)
        sparsity_loss = torch.mean(torch.sum(torch.abs(z), dim=1))
        
        # Total loss
        total_loss = recon_loss + self.sparsity_penalty * sparsity_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'sparsity': sparsity_loss.item()
        }
        
        return total_loss, loss_dict
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'dict_size': self.dict_size,
            'sparsity_penalty': self.sparsity_penalty,
            'tied_weights': self.tied_weights
        }


class SyntheticMLP(nn.Module):
    """MLP for synthetic ground-truth experiments."""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(x))  # This is the layer we analyze
        output = self.fc3(hidden)
        return output, hidden
    
    def get_hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Get activations from the hidden layer."""
        x = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(x))
        return hidden


def train_sae(
    model: SparseAutoencoder,
    activations: torch.Tensor,
    val_activations: Optional[torch.Tensor] = None,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 5e-4,
    early_stopping_patience: int = 10,
    device: str = 'cuda'
) -> dict:
    """Train a Sparse Autoencoder.
    
    Args:
        model: SAE model
        activations: Training activations (n_samples, input_dim)
        val_activations: Validation activations (optional)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        early_stopping_patience: Patience for early stopping
        device: Device to train on
    
    Returns:
        Training history dict
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_samples = activations.shape[0]
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_sparsity': [],
        'val_loss': [],
        'epochs': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_recon = []
        epoch_sparsity = []
        
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = activations[batch_indices].to(device)
            
            # Forward pass
            x_recon, z = model(batch)
            loss, loss_dict = model.compute_loss(batch, x_recon, z)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_dict['total'])
            epoch_recon.append(loss_dict['recon'])
            epoch_sparsity.append(loss_dict['sparsity'])
        
        # Record training metrics
        history['train_loss'].append(np.mean(epoch_losses))
        history['train_recon'].append(np.mean(epoch_recon))
        history['train_sparsity'].append(np.mean(epoch_sparsity))
        history['epochs'].append(epoch)
        
        # Validation
        if val_activations is not None:
            model.eval()
            with torch.no_grad():
                x_recon, z = model(val_activations.to(device))
                val_loss, val_dict = model.compute_loss(val_activations.to(device), x_recon, z)
                history['val_loss'].append(val_dict['total'])
                
                # Early stopping check
                if val_dict['total'] < best_val_loss:
                    best_val_loss = val_dict['total']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: train_loss={history['train_loss'][-1]:.4f}, "
                  f"recon={history['train_recon'][-1]:.4f}, "
                  f"sparsity={history['train_sparsity'][-1]:.4f}")
    
    return history


import numpy as np
