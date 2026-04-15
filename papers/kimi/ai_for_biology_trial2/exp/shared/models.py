"""Model architectures for CROSS-GRN and baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerEncoder(nn.Module):
    """Transformer encoder for gene expression or chromatin accessibility."""
    
    def __init__(self, input_dim, hidden_dim=384, num_layers=4, num_heads=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.norm(x)
        return x


class AsymmetricCrossAttention(nn.Module):
    """Dual asymmetric cross-attention between expression and accessibility."""
    
    def __init__(self, hidden_dim=384, num_heads=6, dropout=0.1):
        super().__init__()
        # Forward: TFs -> targets
        self.q_forward = nn.Linear(hidden_dim, hidden_dim)
        self.k_forward = nn.Linear(hidden_dim, hidden_dim)
        self.v_forward = nn.Linear(hidden_dim, hidden_dim)
        
        # Reverse: targets -> TFs
        self.q_reverse = nn.Linear(hidden_dim, hidden_dim)
        self.k_reverse = nn.Linear(hidden_dim, hidden_dim)
        self.v_reverse = nn.Linear(hidden_dim, hidden_dim)
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, expr, atac, return_attention=True):
        batch_size = expr.size(0)
        
        # Forward attention (TF-like -> target-like)
        Q_f = self.q_forward(expr).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_f = self.k_forward(atac).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_f = self.v_forward(atac).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_f = torch.matmul(Q_f, K_f.transpose(-2, -1)) * self.scale
        attn_f = F.softmax(attn_f, dim=-1)
        attn_f = self.dropout(attn_f)
        out_f = torch.matmul(attn_f, V_f).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Reverse attention
        Q_r = self.q_reverse(atac).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K_r = self.k_reverse(expr).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_r = self.v_reverse(expr).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_r = torch.matmul(Q_r, K_r.transpose(-2, -1)) * self.scale
        attn_r = F.softmax(attn_r, dim=-1)
        attn_r = self.dropout(attn_r)
        out_r = torch.matmul(attn_r, V_r).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        if return_attention:
            return out_f, out_r, attn_f.mean(1), attn_r.mean(1)  # Average over heads
        return out_f, out_r


class CellTypeConditioning(nn.Module):
    """Cell-type-conditioned attention modulation."""
    
    def __init__(self, num_cell_types, hidden_dim=384, cond_dim=128):
        super().__init__()
        self.cell_type_emb = nn.Embedding(num_cell_types, cond_dim)
        self.modulation = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, cell_type_idx):
        # x: (batch, seq_len, hidden_dim)
        # cell_type_idx: (batch,)
        cond = self.cell_type_emb(cell_type_idx)
        mod = self.modulation(cond).unsqueeze(1)  # (batch, 1, hidden_dim)
        return x * mod


class CROSSGRN(nn.Module):
    """CROSS-GRN: Directionality-Aware Cross-Modal Attention for Signed GRN."""
    
    def __init__(self, n_genes, n_peaks, n_cell_types, hidden_dim=384, 
                 num_layers=4, num_heads=6, use_cell_type_cond=True, 
                 use_asymmetric=True, predict_sign=True):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.use_cell_type_cond = use_cell_type_cond
        self.use_asymmetric = use_asymmetric
        self.predict_sign = predict_sign
        
        # Expression encoder
        self.expr_encoder = TransformerEncoder(n_genes, hidden_dim, num_layers, num_heads)
        
        # Accessibility encoder  
        self.atac_encoder = TransformerEncoder(n_peaks, hidden_dim, num_layers, num_heads)
        
        # Cross-attention
        if use_asymmetric:
            self.cross_attn = AsymmetricCrossAttention(hidden_dim, num_heads)
        else:
            # Symmetric cross-attention
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Cell-type conditioning
        if use_cell_type_cond:
            self.cell_type_cond = CellTypeConditioning(n_cell_types, hidden_dim)
        
        # Prediction heads
        self.expr_pred = nn.Linear(hidden_dim, n_genes)
        self.atac_pred = nn.Linear(hidden_dim, n_peaks)
        self.edge_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        if predict_sign:
            self.sign_pred = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
    
    def forward(self, expr, atac, cell_type_idx, return_attention=False):
        batch_size = expr.size(0)
        
        # Encode modalities
        expr_enc = self.expr_encoder(expr.unsqueeze(1))  # (batch, 1, hidden)
        atac_enc = self.atac_encoder(atac.unsqueeze(1))  # (batch, 1, hidden)
        
        # Cross-attention
        if self.use_asymmetric:
            expr_out, atac_out, attn_f, attn_r = self.cross_attn(expr_enc, atac_enc, return_attention=True)
        else:
            expr_out, _ = self.cross_attn(expr_enc, atac_enc, atac_enc)
            atac_out, _ = self.cross_attn(atac_enc, expr_enc, expr_enc)
            attn_f, attn_r = None, None
        
        # Cell-type conditioning
        if self.use_cell_type_cond:
            expr_out = self.cell_type_cond(expr_out, cell_type_idx)
            atac_out = self.cell_type_cond(atac_out, cell_type_idx)
        
        # Predictions
        expr_pred = self.expr_pred(expr_out.squeeze(1))
        atac_pred = self.atac_pred(atac_out.squeeze(1))
        
        outputs = {
            'expr_pred': expr_pred,
            'atac_pred': atac_pred,
            'expr_enc': expr_enc,
            'atac_enc': atac_enc,
            'attn_forward': attn_f,
            'attn_reverse': attn_r
        }
        
        return outputs
    
    def predict_grn_edges(self, expr, atac, cell_type_idx, tf_indices, target_indices):
        """Predict GRN edges between TFs and targets."""
        batch_size = expr.size(0)
        
        with torch.no_grad():
            outputs = self.forward(expr, atac, cell_type_idx)
            
            # Get encoded representations
            expr_enc = outputs['expr_enc'].squeeze(1)  # (batch, hidden)
            
            edges = []
            for tf_idx in tf_indices:
                for target_idx in target_indices:
                    tf_repr = expr_enc[:, tf_idx, :]
                    target_repr = expr_enc[:, target_idx, :]
                    
                    # Concatenate representations
                    pair_repr = torch.cat([tf_repr, target_repr], dim=-1)
                    
                    # Predict edge existence
                    edge_prob = self.edge_pred(pair_repr).mean().item()
                    
                    # Predict sign
                    if self.predict_sign:
                        edge_sign = self.sign_pred(pair_repr).mean().item()
                    else:
                        edge_sign = 0
                    
                    edges.append({
                        'tf_idx': tf_idx,
                        'target_idx': target_idx,
                        'prob': edge_prob,
                        'sign': edge_sign
                    })
        
        return edges


class SimpleGCN(nn.Module):
    """Simple GCN baseline for scMultiomeGRN."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.grn_head = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
    
    def predict_edges(self, x, tf_indices, target_indices):
        """Predict edges between TFs and targets."""
        h = self.forward(x)
        edges = []
        for tf_idx in tf_indices:
            for target_idx in target_indices:
                pair = torch.cat([h[tf_idx], h[target_idx]])
                prob = torch.sigmoid(self.grn_head(pair)).item()
                edges.append({
                    'tf_idx': tf_idx,
                    'target_idx': target_idx,
                    'prob': prob
                })
        return edges


class XATGRNAdapter(nn.Module):
    """Adapted XATGRN architecture for single-cell multi-omics."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=256, num_layers=2):
        super().__init__()
        self.gene_emb = nn.Linear(n_genes, hidden_dim)
        self.peak_emb = nn.Linear(n_peaks, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Prediction heads
        self.edge_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.sign_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(self, expr, atac):
        h_gene = self.gene_emb(expr).unsqueeze(1)
        h_peak = self.peak_emb(atac).unsqueeze(1)
        
        # Cross-attention
        h_combined, _ = self.cross_attn(h_gene, h_peak, h_peak)
        
        return h_combined.squeeze(1)
    
    def predict_edges(self, expr, atac, tf_indices, target_indices):
        h = self.forward(expr, atac)
        edges = []
        for tf_idx in tf_indices:
            for target_idx in target_indices:
                pair = torch.cat([h[tf_idx], h[target_idx]])
                prob = torch.sigmoid(self.edge_pred(pair)).item()
                sign = torch.tanh(self.sign_pred(pair)).item()
                edges.append({
                    'tf_idx': tf_idx,
                    'target_idx': target_idx,
                    'prob': prob,
                    'sign': sign
                })
        return edges
