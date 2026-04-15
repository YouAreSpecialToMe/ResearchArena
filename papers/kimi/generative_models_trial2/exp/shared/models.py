"""
Point Transformer model for flow matching on point clouds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PointTransformerEncoder(nn.Module):
    """Point Transformer encoder for point cloud processing."""
    
    def __init__(self, in_dim=3, hidden_dim=256, num_layers=4, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point coordinates
        Returns:
            (B, N, hidden_dim) features
        """
        # Embed input
        h = self.input_embed(x)  # (B, N, hidden_dim)
        
        # Apply transformer layers
        for layer, norm in zip(self.transformer_layers, self.norms):
            h = h + layer(norm(h))
        
        return h


class TransformerLayer(nn.Module):
    """Single transformer layer with memory-efficient attention."""
    
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Enable Flash Attention if available (PyTorch 2.0+)
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(self, x):
        """
        Args:
            x: (B, N, hidden_dim)
        Returns:
            (B, N, hidden_dim)
        """
        B, N, C = x.shape
        
        # Multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use Flash Attention if available (much more memory efficient)
        if self.use_flash and x.is_cuda:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        else:
            # Manual attention with chunking to save memory
            chunk_size = 1024  # Process in chunks
            if N > chunk_size:
                out_chunks = []
                for i in range(0, N, chunk_size):
                    q_chunk = q[:, :, i:i+chunk_size, :]
                    attn = (q_chunk @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    attn = F.softmax(attn, dim=-1)
                    out_chunk = (attn @ v)
                    out_chunks.append(out_chunk)
                out = torch.cat(out_chunks, dim=2)
            else:
                attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn = F.softmax(attn, dim=-1)
                out = (attn @ v)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        # MLP
        out = self.mlp(out)
        
        return out


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""
    
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.gamma_proj = nn.Linear(condition_dim, feature_dim)
        self.beta_proj = nn.Linear(condition_dim, feature_dim)
    
    def forward(self, x, condition):
        """
        Args:
            x: (B, N, feature_dim)
            condition: (B, condition_dim)
        Returns:
            (B, N, feature_dim)
        """
        gamma = self.gamma_proj(condition).unsqueeze(1)  # (B, 1, feature_dim)
        beta = self.beta_proj(condition).unsqueeze(1)    # (B, 1, feature_dim)
        return gamma * x + beta


class VelocityNetwork(nn.Module):
    """
    Velocity network for flow matching.
    Predicts the velocity field v_theta(x_t, t).
    """
    
    def __init__(
        self,
        point_dim=3,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        time_embed_dim=128,
        use_distance_conditioning=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_distance_conditioning = use_distance_conditioning
        
        # Time embedding
        self.time_embed_dim = time_embed_dim
        
        # Distance conditioning
        if use_distance_conditioning:
            # Condition on mean and max radial distance
            self.distance_embed = nn.Sequential(
                nn.Linear(2, time_embed_dim // 2),
                nn.ReLU(),
                nn.Linear(time_embed_dim // 2, time_embed_dim // 2),
            )
            total_condition_dim = time_embed_dim + time_embed_dim // 2
        else:
            total_condition_dim = time_embed_dim
        
        # Point encoder
        self.encoder = PointTransformerEncoder(
            in_dim=point_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # FiLM conditioning
        self.film = FiLM(total_condition_dim, hidden_dim)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, point_dim),
        )
        
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def get_timestep_embedding(self, t, embedding_dim):
        """
        Sinusoidal timestep embedding.
        Args:
            t: (B,) timestep in [0, 1]
            embedding_dim: dimension of embedding
        Returns:
            (B, embedding_dim)
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x_t, t, radial_dist=None):
        """
        Args:
            x_t: (B, N, 3) noisy point cloud
            t: (B,) timestep
            radial_dist: (B, N) radial distances [0, 1]
        Returns:
            v: (B, N, 3) predicted velocity
        """
        B, N, _ = x_t.shape
        
        # Get timestep embedding
        t_emb = self.get_timestep_embedding(t, self.time_embed_dim)
        
        # Distance conditioning
        if self.use_distance_conditioning and radial_dist is not None:
            dist_stats = torch.stack([
                radial_dist.mean(dim=1),
                radial_dist.max(dim=1)[0],
            ], dim=-1)  # (B, 2)
            dist_emb = self.distance_embed(dist_stats)
            condition = torch.cat([t_emb, dist_emb], dim=-1)
        else:
            condition = t_emb
        
        # Encode points
        h = self.encoder(x_t)
        
        # Apply FiLM conditioning
        h = self.film(h, condition)
        
        # Predict velocity
        v = self.output_head(h)
        
        return v


class WeightPredictorMLP(nn.Module):
    """
    Learned Adaptive Weighting (LAW) MLP.
    Predicts per-point weights for flow matching loss.
    """
    
    def __init__(self, in_dim=4, hidden_dims=[32, 16], out_dim=1):
        super().__init__()
        
        layers = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]
        
        self.mlp = nn.Sequential(*layers)
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, features):
        """
        Args:
            features: (B, N, in_dim) - [mean_knn, std_knn, r_norm, z_norm]
        Returns:
            weights: (B, N, 1) in [0.5, 2.0]
        """
        w = self.mlp(features)  # (B, N, 1) in [0, 1]
        # Scale to [0.5, 2.0]
        w = 0.5 + w * 1.5
        return w


def compute_knn_features(points, radial_dist, k=8):
    """
    Compute k-NN based features for each point.
    
    Args:
        points: (B, N, 3) point coordinates
        radial_dist: (B, N) radial distances
        k: number of neighbors
    Returns:
        features: (B, N, 4) - [mean_knn_dist, std_knn_dist, r_norm, z_norm]
    """
    B, N, _ = points.shape
    device = points.device
    
    # Compute pairwise distances
    # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N)
    dist_matrix = torch.cdist(points, points)  # (B, N, N)
    
    # Get k nearest neighbors (excluding self)
    knn_dists, _ = torch.topk(dist_matrix, k + 1, largest=False, dim=-1)
    knn_dists = knn_dists[:, :, 1:]  # Remove self (distance 0)
    
    # Statistics
    mean_knn = knn_dists.mean(dim=-1)  # (B, N)
    std_knn = knn_dists.std(dim=-1)    # (B, N)
    
    # Height (z-coordinate)
    z_norm = points[:, :, 2] / 5.0  # Normalize by typical height
    
    # Combine features
    features = torch.stack([mean_knn, std_knn, radial_dist, z_norm], dim=-1)
    
    return features


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VelocityNetwork(
        point_dim=3,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        use_distance_conditioning=True,
    ).to(device)
    
    print(f"Model parameters: {model.num_params / 1e6:.2f}M")
    
    # Test forward pass
    x_t = torch.randn(2, 8192, 3).to(device)
    t = torch.rand(2).to(device)
    radial_dist = torch.rand(2, 8192).to(device)
    
    v = model(x_t, t, radial_dist)
    print(f"Output shape: {v.shape}")
    
    # Test weight predictor
    weight_mlp = WeightPredictorMLP().to(device)
    print(f"Weight MLP parameters: {weight_mlp.num_params}")
    
    features = compute_knn_features(x_t, radial_dist)
    weights = weight_mlp(features)
    print(f"Weights shape: {weights.shape}, range: [{weights.min():.3f}, {weights.max():.3f}]")
