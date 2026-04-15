"""
Model definitions for SAE-GUIDE and baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for decomposing hidden states."""
    
    def __init__(self, input_dim: int = 3584, expansion_factor: int = 8, top_k: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion_factor
        self.top_k = top_k
        
        self.encoder = nn.Linear(input_dim, self.hidden_dim, bias=True)
        self.decoder = nn.Linear(self.hidden_dim, input_dim, bias=True)
        
        # Initialize
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to sparse representation with top-k activation."""
        z = F.relu(self.encoder(x))
        
        # Top-k sparsity
        if self.top_k > 0 and z.shape[-1] >= self.top_k:
            top_k_values, top_k_indices = torch.topk(z, self.top_k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, top_k_indices, top_k_values)
        else:
            z_sparse = z
        
        return z_sparse
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from sparse representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent representation."""
        z = self.encode(x)
        x_hat = self.decode(z)
        
        # Compute losses
        reconstruction_loss = F.mse_loss(x_hat, x)
        sparsity_loss = torch.mean(torch.abs(z))
        
        return x_hat, z, reconstruction_loss, sparsity_loss


class InformationNeedProbe(nn.Module):
    """Probe for detecting information needs from SAE features."""
    
    def __init__(self, input_dim: int = 28672, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability of information need."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc2(h)
        return torch.sigmoid(logits).squeeze(-1)


class BinaryProbe(nn.Module):
    """Simple binary probe on hidden states (Probing-RAG baseline)."""
    
    def __init__(self, input_dim: int = 3584, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability of retrieval need."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        logits = self.fc2(h)
        return torch.sigmoid(logits).squeeze(-1)


class CumulativeFeatureTracker:
    """Track cumulative feature activations across reasoning steps."""
    
    def __init__(self, feature_dim: int = 28672, decay: float = 0.9):
        self.feature_dim = feature_dim
        self.decay = decay
        self.cumulative = None
        self.history = []
    
    def reset(self):
        """Reset cumulative activation."""
        self.cumulative = None
        self.history = []
    
    def update(self, features: torch.Tensor) -> torch.Tensor:
        """Update cumulative activation with new features."""
        if self.cumulative is None:
            self.cumulative = features.clone()
        else:
            self.cumulative = self.decay * self.cumulative + features
        
        self.history.append(self.cumulative.clone())
        return self.cumulative
    
    def get_top_features(self, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k activated features."""
        if self.cumulative is None:
            return torch.tensor([]), torch.tensor([])
        
        values, indices = torch.topk(self.cumulative, k)
        return values, indices


class FeatureToQueryMapper:
    """Map activated features to query augmentations."""
    
    def __init__(self):
        # Define feature types (these would be learned in practice)
        self.feature_templates = {
            'entity': ["{entity} information", "{entity} details"],
            'temporal': ["when date timeline", "year time period"],
            'relation': ["relationship between", "connection to"],
            'location': ["where location place", "geographic"],
            'default': ["additional information", "more details"]
        }
    
    def classify_features(self, feature_indices: torch.Tensor) -> Dict[str, List[int]]:
        """
        Classify features into semantic types.
        In practice, this would be learned from data.
        Here we use heuristics for demonstration.
        """
        # Simple heuristic: bucket features by index ranges
        types = {
            'entity': [],
            'temporal': [],
            'relation': [],
            'location': [],
            'other': []
        }
        
        for idx in feature_indices.tolist():
            idx_int = int(idx)
            if idx_int % 5 == 0:
                types['entity'].append(idx_int)
            elif idx_int % 5 == 1:
                types['temporal'].append(idx_int)
            elif idx_int % 5 == 2:
                types['relation'].append(idx_int)
            elif idx_int % 5 == 3:
                types['location'].append(idx_int)
            else:
                types['other'].append(idx_int)
        
        return types
    
    def map_to_query(self, feature_indices: torch.Tensor, 
                     feature_values: torch.Tensor,
                     question: str, 
                     context: str) -> str:
        """Map features to query augmentation."""
        types = self.classify_features(feature_indices)
        
        augmentations = []
        
        # Extract entities from context if entity features are active
        if types['entity']:
            # Simple entity extraction from context
            words = context.split()
            entities = [w for w in words if w and w[0].isupper()]
            if entities:
                entity = entities[-1]  # Use most recent entity
                augmentations.append(f"{entity} information")
        
        if types['temporal']:
            augmentations.append("date timeline when")
        
        if types['relation']:
            augmentations.append("relationship connection")
        
        if not augmentations:
            augmentations.append("more information")
        
        return question + " " + " ".join(augmentations)


if __name__ == '__main__':
    # Test models
    sae = SparseAutoencoder(input_dim=3584, expansion_factor=8, top_k=32)
    probe = InformationNeedProbe(input_dim=28672)
    
    # Test forward pass
    x = torch.randn(4, 3584)
    x_hat, z, recon_loss, sparse_loss = sae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    
    # Test probe
    uncertainty = probe(z)
    print(f"Uncertainty: {uncertainty}")
    
    # Test cumulative tracker
    tracker = CumulativeFeatureTracker(feature_dim=28672)
    for i in range(5):
        features = torch.randn(28672)
        cum = tracker.update(features)
        print(f"Step {i}: cumulative norm = {cum.norm().item():.4f}")
