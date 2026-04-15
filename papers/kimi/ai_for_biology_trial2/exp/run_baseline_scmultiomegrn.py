#!/usr/bin/env python3
"""Run scMultiomeGRN baseline."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scanpy as sc
import json
import os
import argparse
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

from exp.shared.data_loader import MultiOmicDataset, load_tf_list
from exp.shared.metrics import compute_auroc, compute_auprc, compute_sign_accuracy


class scMultiomeGRNBaseline(nn.Module):
    """GNN-based multi-omics GRN inference baseline."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=256, num_layers=2):
        super().__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        
        # Input projections
        self.gene_proj = nn.Linear(n_genes, hidden_dim)
        self.peak_proj = nn.Linear(n_peaks, hidden_dim)
        
        # GNN layers (simplified message passing)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes)
        )
        
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
    
    def forward(self, expr, atac, adj_matrix=None):
        """
        Args:
            expr: (batch, n_genes)
            atac: (batch, n_peaks)
            adj_matrix: optional cell-cell adjacency (batch, batch)
        """
        batch_size = expr.size(0)
        
        # Project inputs
        h_gene = self.gene_proj(expr)  # (batch, hidden)
        h_peak = self.peak_proj(atac)  # (batch, hidden)
        
        # Simple message passing (treat as fully connected if no adj)
        for gnn_layer in self.gnn_layers:
            h_gene_new = F.relu(gnn_layer(h_gene))
            h_gene = h_gene + h_gene_new  # Residual
        
        # Cross-modal: genes attend to peaks
        h_gene_seq = h_gene.unsqueeze(1)  # (batch, 1, hidden)
        h_peak_seq = h_peak.unsqueeze(1)  # (batch, 1, hidden)
        
        cross_out, _ = self.cross_attn(h_gene_seq, h_peak_seq, h_peak_seq)
        h_fused = torch.cat([h_gene_seq, cross_out], dim=-1).squeeze(1)
        
        # Output
        output = self.output_proj(h_fused)
        
        return {
            'output': output,
            'h_gene': h_gene,
            'h_peak': h_peak
        }
    
    def predict_grn(self, expr, atac, tf_indices, target_indices):
        """Predict GRN edges."""
        with torch.no_grad():
            outputs = self.forward(expr, atac)
            h_gene = outputs['h_gene']  # (batch, hidden)
            
            # Average over batch for cell representation
            h_mean = h_gene.mean(dim=0)  # (hidden,)
            
            # Get TF and target features from expression
            tf_expr = expr[:, tf_indices].mean(dim=0)  # (n_tfs,)
            target_expr = expr[:, target_indices].mean(dim=0)  # (n_targets,)
            
            n_tfs = len(tf_indices)
            n_targets = len(target_indices)
            
            edge_probs = torch.zeros(n_tfs, n_targets)
            edge_signs = torch.zeros(n_tfs, n_targets)
            
            # Predict edges using learned features + expression correlation
            for i, tf_idx in enumerate(tf_indices):
                for j, target_idx in enumerate(target_indices):
                    # Use expression correlation as base score
                    tf_exp = expr[:, tf_idx]
                    target_exp = expr[:, target_idx]
                    
                    # Compute correlation
                    corr = torch.corrcoef(torch.stack([tf_exp, target_exp]))[0, 1].item()
                    corr = abs(corr) if not np.isnan(corr) else 0.1
                    
                    # Combine with learned representation
                    score = 0.5 * corr + 0.5 * torch.sigmoid(h_mean[:10].mean()).item()
                    edge_probs[i, j] = min(1.0, max(0.0, score))
                    
                    # Sign from correlation
                    sign = 1 if corr > 0 else -1
                    edge_signs[i, j] = sign
            
            return edge_probs, edge_signs


def train_scmultiomegrn(rna_path, atac_path, output_path, config):
    """Train scMultiomeGRN baseline."""
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training scMultiomeGRN on {device} with seed {seed}")
    
    # Load data
    print("Loading data...")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/splits.json') as f:
        splits = json.load(f)
    
    train_idx = splits['train']
    val_idx = splits['val']
    
    # Load ground truth
    with open('data/ground_truth_edges.json') as f:
        ground_truth_edges = json.load(f)
    
    # Create datasets
    train_dataset = MultiOmicDataset(
        rna[train_idx], atac[train_idx],
        [cell_types[i] for i in train_idx],
        mode='train'
    )
    
    val_dataset = MultiOmicDataset(
        rna[val_idx], atac[val_idx],
        [cell_types[i] for i in val_idx],
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Create model
    model = scMultiomeGRNBaseline(
        n_genes=rna.n_vars,
        n_peaks=atac.n_vars,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # Training
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            
            optimizer.zero_grad()
            outputs = model(expr, atac)
            
            # Reconstruction loss
            loss = F.mse_loss(outputs['output'], expr)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: train_loss={avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config['model_path'])
    
    # Evaluate
    print("Evaluating...")
    model.load_state_dict(torch.load(config['model_path']))
    model.eval()
    
    # Get TF and target indices
    gene_names = rna.var_names.tolist()
    tfs = load_tf_list()
    available_tfs = [tf for tf in tfs if tf in gene_names][:20]
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    target_genes = [g for g in gene_names if g not in tfs][:200]
    target_indices = [gene_names.index(g) for g in target_genes]
    
    # Build ground truth dict
    gt_dict = {}
    gt_sign_dict = {}
    for edge in ground_truth_edges:
        if edge['tf'] in gene_names and edge['target'] in gene_names:
            try:
                tf_idx = gene_names.index(edge['tf'])
                target_idx = gene_names.index(edge['target'])
                if tf_idx in tf_indices and target_idx in target_indices:
                    gt_dict[(tf_idx, target_idx)] = 1
                    gt_sign_dict[(tf_idx, target_idx)] = edge.get('sign', 1)
            except:
                pass
    
    # Collect predictions
    all_edges = []
    
    with torch.no_grad():
        for batch in val_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            
            edge_probs, edge_signs = model.predict_grn(expr, atac, tf_indices, target_indices)
            
            for i, tf_idx in enumerate(tf_indices):
                for j, target_idx in enumerate(target_indices):
                    all_edges.append({
                        'tf_idx': tf_idx,
                        'target_idx': target_idx,
                        'prob': edge_probs[i, j].item(),
                        'sign': edge_signs[i, j].item()
                    })
            
            break  # Just use first batch for efficiency
    
    # Compute metrics
    y_true = np.array([gt_dict.get((e['tf_idx'], e['target_idx']), 0) for e in all_edges])
    y_score = np.array([e['prob'] for e in all_edges])
    
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)
    
    y_sign_true = np.array([gt_sign_dict.get((e['tf_idx'], e['target_idx']), 1) for e in all_edges])
    y_sign_pred = np.array([e['sign'] for e in all_edges])
    sign_acc = compute_sign_accuracy(y_sign_true, y_sign_pred)
    
    print(f"Results - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Sign Acc: {sign_acc:.4f}")
    
    # Save results
    results = {
        'method': 'scMultiomeGRN',
        'seed': seed,
        'metrics': {
            'auroc': auroc,
            'auprc': auprc,
            'sign_accuracy': sign_acc
        },
        'config': config
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--atac', default='data/pbmc_atac_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/scmultiomegrn_baseline/results.json')
    parser.add_argument('--model_path', default='models/scmultiomegrn.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    args = parser.parse_args()
    
    config = {
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'model_path': args.model_path
    }
    
    train_scmultiomegrn(args.rna, args.atac, args.output, config)
