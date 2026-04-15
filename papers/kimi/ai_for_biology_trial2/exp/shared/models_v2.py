"""Fixed model architectures for CROSS-GRN and baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CROSSGRNv2(nn.Module):
    """
    Fixed CROSS-GRN with proper architecture for GRN inference.
    Memory-efficient version for 20k+ peaks.
    """
    
    def __init__(self, n_genes, n_peaks, n_cell_types, hidden_dim=128, 
                 num_layers=2, num_heads=4, dropout=0.1,
                 use_cell_type_cond=True, use_asymmetric=True, predict_sign=True):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        self.hidden_dim = hidden_dim
        self.use_cell_type_cond = use_cell_type_cond
        self.use_asymmetric = use_asymmetric
        self.predict_sign = predict_sign
        
        # Gene and peak embeddings
        self.gene_emb = nn.Linear(1, hidden_dim)
        
        # For ATAC: use a projection to reduce dimensionality first
        self.peak_proj = nn.Sequential(
            nn.Linear(n_peaks, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        
        # Expression encoder (transformer over genes)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2,
            dropout=dropout, batch_first=True
        )
        self.expr_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Cell-type conditioning
        if use_cell_type_cond:
            self.cell_type_emb = nn.Embedding(n_cell_types, hidden_dim)
            self.cond_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
        
        # Prediction heads
        # Expression prediction
        self.expr_pred_head = nn.Linear(hidden_dim, 1)
        
        # GRN edge prediction head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Sign prediction head
        if predict_sign:
            self.sign_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, expr, atac, cell_type_idx, return_embeddings=False):
        """
        Forward pass.
        
        Args:
            expr: (batch, n_genes) gene expression
            atac: (batch, n_peaks) chromatin accessibility
            cell_type_idx: (batch,) cell type indices
            
        Returns:
            dict with predictions and optionally embeddings
        """
        batch_size = expr.size(0)
        
        # Embed genes: (batch, n_genes, hidden_dim)
        expr_input = expr.unsqueeze(-1)  # (batch, n_genes, 1)
        gene_features = self.gene_emb(expr_input)
        
        # Project ATAC to hidden dim (batch, hidden_dim)
        atac_features = self.peak_proj(atac)  # (batch, hidden_dim)
        
        # Encode expression
        gene_encoded = self.expr_encoder(gene_features)  # (batch, n_genes, hidden)
        
        # Add ATAC info to gene representations (broadcasted)
        fused = gene_encoded + atac_features.unsqueeze(1)  # (batch, n_genes, hidden)
        
        # Cell-type conditioning
        if self.use_cell_type_cond:
            cell_emb = self.cell_type_emb(cell_type_idx)  # (batch, hidden)
            cond = self.cond_proj(cell_emb).unsqueeze(1)  # (batch, 1, hidden)
            fused = fused + cond  # Broadcast conditioning
        
        # Expression prediction
        expr_pred = self.expr_pred_head(fused).squeeze(-1)  # (batch, n_genes)
        
        outputs = {
            'expr_pred': expr_pred,
            'gene_embeddings': fused,  # (batch, n_genes, hidden)
        }
        
        return outputs
    
    def predict_edge_batch(self, gene_embeddings, tf_indices, target_indices):
        """
        Predict edges for batches of TF-target pairs.
        
        Args:
            gene_embeddings: (batch, n_genes, hidden)
            tf_indices: list of TF gene indices
            target_indices: list of target gene indices
            
        Returns:
            edge_probs: (len(tf_indices), len(target_indices))
            edge_signs: (len(tf_indices), len(target_indices))
        """
        batch_size = gene_embeddings.size(0)
        n_tfs = len(tf_indices)
        n_targets = len(target_indices)
        
        # Get TF embeddings: (batch, n_tfs, hidden)
        tf_emb = gene_embeddings[:, tf_indices, :]
        
        # Get target embeddings: (batch, n_targets, hidden)
        target_emb = gene_embeddings[:, target_indices, :]
        
        # Compute all pairwise combinations
        # Expand to (batch, n_tfs, n_targets, hidden)
        tf_expanded = tf_emb.unsqueeze(2).expand(-1, -1, n_targets, -1)
        target_expanded = target_emb.unsqueeze(1).expand(-1, n_tfs, -1, -1)
        
        # Concatenate: (batch, n_tfs, n_targets, hidden*2)
        pair_repr = torch.cat([tf_expanded, target_expanded], dim=-1)
        
        # Predict edges
        edge_logits = self.edge_head(pair_repr).squeeze(-1)  # (batch, n_tfs, n_targets)
        edge_probs = torch.sigmoid(edge_logits)
        
        if self.predict_sign:
            edge_signs = self.sign_head(pair_repr).squeeze(-1)  # (batch, n_tfs, n_targets)
        else:
            edge_signs = torch.zeros_like(edge_probs)
        
        # Average over batch
        edge_probs = edge_probs.mean(dim=0)  # (n_tfs, n_targets)
        edge_signs = edge_signs.mean(dim=0)  # (n_tfs, n_targets)
        
        return edge_probs, edge_signs


class scMultiomeGRN(nn.Module):
    """GNN-based baseline for multi-omics GRN inference."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=128, num_layers=2):
        super().__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        
        # Input projections (memory efficient)
        self.gene_proj = nn.Linear(n_genes, hidden_dim)
        self.peak_proj = nn.Sequential(
            nn.Linear(n_peaks, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layers
        self.output_proj = nn.Linear(hidden_dim, n_genes)
        
        # Edge prediction heads
        self.edge_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sign_pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(self, expr, atac):
        """
        Args:
            expr: (batch, n_genes)
            atac: (batch, n_peaks)
        """
        # Project to hidden space
        h_gene = self.gene_proj(expr)  # (batch, hidden)
        h_peak = self.peak_proj(atac)  # (batch, hidden)
        
        # Combine modalities
        h = h_gene + h_peak
        
        # Apply conv layers with residual
        for gnn_layer in self.gnn_layers:
            h_new = F.relu(gnn_layer(h))
            h = h + h_new
        
        # Output
        output = self.output_proj(h)
        
        return {
            'output': output,
            'h': h
        }
    
    def predict_grn(self, expr, atac, tf_indices, target_indices):
        """Predict GRN edges."""
        with torch.no_grad():
            outputs = self.forward(expr, atac)
            h = outputs['h']  # (batch, hidden)
            
            # Average over batch for cell representation
            h_mean = h.mean(dim=0)  # (hidden,)
            
            # Use expression correlation as base
            tf_expr = expr[:, tf_indices].mean(dim=0)  # (n_tfs,)
            target_expr = expr[:, target_indices].mean(dim=0)  # (n_targets,)
            
            n_tfs = len(tf_indices)
            n_targets = len(target_indices)
            
            edge_probs = torch.zeros(n_tfs, n_targets)
            edge_signs = torch.zeros(n_tfs, n_targets)
            
            for i, tf_idx in enumerate(tf_indices):
                for j, target_idx in enumerate(target_indices):
                    # Use expression correlation
                    tf_exp = expr[:, tf_idx]
                    target_exp = expr[:, target_idx]
                    
                    corr = torch.corrcoef(torch.stack([tf_exp, target_exp]))[0, 1].item()
                    corr = abs(corr) if not np.isnan(corr) else 0.1
                    
                    score = 0.5 * corr + 0.5 * torch.sigmoid(h_mean[:10].mean()).item()
                    edge_probs[i, j] = min(1.0, max(0.0, score))
                    
                    sign = 1 if corr > 0 else -1
                    edge_signs[i, j] = sign
            
            return edge_probs, edge_signs


class XATGRNBaseline(nn.Module):
    """XATGRN-style cross-attention baseline."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=128, num_layers=2):
        super().__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        
        # Input projections
        self.gene_proj = nn.Linear(n_genes, hidden_dim)
        self.peak_proj = nn.Sequential(
            nn.Linear(n_peaks, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_genes)
        
        # Edge prediction heads
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
        """
        Args:
            expr: (batch, n_genes)
            atac: (batch, n_peaks)
        """
        batch_size = expr.size(0)
        
        # Project to hidden space
        h_gene = self.gene_proj(expr).unsqueeze(1)  # (batch, 1, hidden)
        h_peak = self.peak_proj(atac).unsqueeze(1)  # (batch, 1, hidden)
        
        # Expand for attention
        h_gene = h_gene.expand(-1, self.n_genes, -1)  # (batch, n_genes, hidden)
        
        # Cross-attention layers
        for attn, norm in zip(self.cross_attn_layers, self.layer_norms):
            attn_out, _ = attn(h_gene, h_peak, h_peak)
            h_gene = norm(h_gene + attn_out)
        
        # Output prediction
        output = self.output_proj(h_gene.mean(dim=1))
        
        return {
            'output': output,
            'gene_repr': h_gene  # (batch, n_genes, hidden)
        }
    
    def predict_grn(self, expr, atac, tf_indices, target_indices):
        """Predict GRN edges using learned representations."""
        with torch.no_grad():
            outputs = self.forward(expr, atac)
            gene_repr = outputs['gene_repr']  # (batch, n_genes, hidden)
            
            # Average over batch
            gene_repr_mean = gene_repr.mean(dim=0)  # (n_genes, hidden)
            
            # Get TF and target representations
            tf_repr = gene_repr_mean[tf_indices]  # (n_tfs, hidden)
            target_repr = gene_repr_mean[target_indices]  # (n_targets, hidden)
            
            n_tfs = len(tf_indices)
            n_targets = len(target_indices)
            
            edge_probs = torch.zeros(n_tfs, n_targets)
            edge_signs = torch.zeros(n_tfs, n_targets)
            
            # Predict edges
            for i in range(n_tfs):
                for j in range(n_targets):
                    pair = torch.cat([tf_repr[i], target_repr[j]])
                    edge_probs[i, j] = self.edge_pred(pair).item()
                    edge_signs[i, j] = self.sign_pred(pair).item()
            
            return edge_probs, edge_signs
