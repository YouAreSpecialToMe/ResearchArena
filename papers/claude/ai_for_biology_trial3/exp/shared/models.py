"""
Model definitions for REN (Residual Epistasis Networks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv


class REN(nn.Module):
    """
    Residual Epistasis Network.
    A GAT operating on protein structure contact graphs to predict
    the epistatic residual (observed fitness - PLM additive prediction).
    """
    def __init__(
        self,
        node_feat_dim=1301,  # 1280 ESM-2 + 21 mutation identity
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        edge_feat_dim=0,
        conv_type='gat',
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim
            if conv_type == 'gat':
                conv = GATConv(
                    in_dim, hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            elif conv_type == 'gcn':
                conv = GCNConv(in_dim, hidden_dim)
            elif conv_type == 'transformer':
                conv = TransformerConv(
                    in_dim, hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=edge_feat_dim if edge_feat_dim > 0 else None,
                    dropout=dropout,
                    concat=True,
                )
            else:
                raise ValueError(f"Unknown conv type: {conv_type}")

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Attention-weighted pooling over mutation sites
        self.pool_query = nn.Parameter(torch.randn(hidden_dim))
        self.pool_key = nn.Linear(hidden_dim, hidden_dim)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, mutation_site_masks, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, node_feat_dim]
            edge_index: Graph edges [2, num_edges]
            mutation_site_masks: List of tensors, each containing indices of mutation sites
                                 for variants in the batch [batch_size, variable]
            edge_attr: Optional edge features [num_edges, edge_feat_dim]
        Returns:
            predictions: [batch_size, 1] epistatic residual predictions
        """
        # Input projection
        h = self.input_proj(x)

        # GNN message passing
        for i in range(self.num_layers):
            if self.conv_type == 'transformer' and edge_attr is not None:
                h_new = self.convs[i](h, edge_index, edge_attr=edge_attr)
            else:
                h_new = self.convs[i](h, edge_index)
            h_new = F.elu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = self.norms[i](h_new + h)  # Residual connection

        # Attention-weighted pooling over mutation sites
        predictions = []
        query = self.pool_query.unsqueeze(0)  # [1, hidden_dim]

        for sites in mutation_site_masks:
            if len(sites) == 0:
                predictions.append(torch.zeros(1, device=x.device))
                continue

            site_embeddings = h[sites]  # [n_sites, hidden_dim]
            keys = self.pool_key(site_embeddings)  # [n_sites, hidden_dim]
            attn_scores = (query * keys).sum(dim=-1) / (h.shape[-1] ** 0.5)  # [n_sites]
            attn_weights = F.softmax(attn_scores, dim=0)
            pooled = (attn_weights.unsqueeze(-1) * site_embeddings).sum(dim=0)  # [hidden_dim]
            predictions.append(pooled.unsqueeze(0))

        pooled_batch = torch.cat(predictions, dim=0)  # [batch_size, hidden_dim]
        output = self.mlp(pooled_batch)  # [batch_size, 1]
        return output.squeeze(-1)

    def get_attention_weights(self, x, edge_index, edge_attr=None):
        """Extract GAT attention weights for interpretability."""
        h = self.input_proj(x)
        all_attention = []

        for i in range(self.num_layers):
            if self.conv_type == 'gat':
                h_new, (edge_index_out, attn_weights) = self.convs[i](
                    h, edge_index, return_attention_weights=True
                )
                all_attention.append(attn_weights)
            else:
                h_new = self.convs[i](h, edge_index)

            h_new = F.elu(h_new)
            h = self.norms[i](h_new + h)

        return all_attention


class MLPBaseline(nn.Module):
    """MLP-only baseline (no GNN message passing)."""
    def __init__(self, node_feat_dim=1301, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.pool_query = nn.Parameter(torch.randn(hidden_dim))
        self.pool_key = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index, mutation_site_masks, edge_attr=None):
        h = self.input_proj(x)
        h = F.elu(h)

        predictions = []
        query = self.pool_query.unsqueeze(0)
        for sites in mutation_site_masks:
            if len(sites) == 0:
                predictions.append(torch.zeros(1, device=x.device))
                continue
            site_embeddings = h[sites]
            keys = self.pool_key(site_embeddings)
            attn_scores = (query * keys).sum(dim=-1) / (h.shape[-1] ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=0)
            pooled = (attn_weights.unsqueeze(-1) * site_embeddings).sum(dim=0)
            predictions.append(pooled.unsqueeze(0))

        pooled_batch = torch.cat(predictions, dim=0)
        return self.mlp(pooled_batch).squeeze(-1)
