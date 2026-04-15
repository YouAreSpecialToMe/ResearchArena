"""Model definitions: EpiGNN, MLP baseline."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.data import Data, Batch


class EpiGNN(nn.Module):
    """
    Epistasis-Aware GNN for multi-mutation fitness prediction.

    Operates on mutation-centric subgraphs where:
    - Nodes = mutated positions with PLM-derived embedding deltas
    - Edges = PLM attention-based residue coupling

    Predicts an epistatic correction added to the additive PLM score.
    """
    def __init__(self, input_dim=1280, hidden_dim=128, num_heads=4,
                 num_layers=2, edge_dim=3, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            self.gat_layers.append(
                GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads,
                          edge_dim=hidden_dim, concat=True, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)

        for gat, norm in zip(self.gat_layers, self.norms):
            x_new = gat(x, data.edge_index, edge_attr=edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # residual

        # Global sum pooling
        out = global_add_pool(x, data.batch)
        epsilon = self.readout(out).squeeze(-1)
        return epsilon


class MLPBaseline(nn.Module):
    """
    MLP baseline: same features as EpiGNN but no graph structure.
    Pools mutation features by summing, then passes through MLP.
    """
    def __init__(self, input_dim=1280, hidden_dims=[256, 128],
                 extra_features=4, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(input_dim, hidden_dims[0])

        # extra_features: additive_score, num_mutations, mean_coupling, max_coupling
        layers = []
        in_dim = hidden_dims[0] + extra_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, node_features, extra_features, batch_indices):
        """
        node_features: [total_nodes, input_dim]
        extra_features: [batch_size, extra_features]
        batch_indices: [total_nodes] mapping nodes to graphs
        """
        x = self.node_proj(node_features)
        x = F.relu(x)

        # Sum pool by graph
        batch_size = extra_features.shape[0]
        pooled = torch.zeros(batch_size, x.shape[1], device=x.device)
        pooled.scatter_add_(0, batch_indices.unsqueeze(1).expand_as(x), x)

        combined = torch.cat([pooled, extra_features], dim=1)
        epsilon = self.mlp(combined).squeeze(-1)
        return epsilon
