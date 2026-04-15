"""Model definitions: Projection head for contrastive learning."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """Two-layer MLP projection head: 1280 -> 1024 -> 512, L2-normalized output."""
    def __init__(self, input_dim=1280, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)
