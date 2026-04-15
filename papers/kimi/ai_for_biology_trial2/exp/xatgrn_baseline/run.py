#!/usr/bin/env python3
"""XATGRN-adapted baseline for single-cell multi-omics."""
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
from exp.shared.data_loader import MultiOmicDataset, load_tf_list
from exp.shared.metrics import compute_auroc, compute_auprc, compute_sign_accuracy


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


def train_xatgrn(rna_path, atac_path, output_path, seed=42, epochs=50):
    """Train XATGRN baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Loading data...")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/splits.json') as f:
        splits = json.load(f)
    
    train_idx = splits['train']
    
    train_dataset = MultiOmicDataset(
        rna[train_idx], atac[train_idx],
        [cell_types[i] for i in train_idx],
        mode='train'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Store atac anndata before any tensor conversion
    atac_adata = atac
    
    # Create model
    n_genes = rna.n_vars
    n_peaks = atac_adata.n_vars
    
    model = XATGRNAdapter(n_genes, n_peaks, hidden_dim=256, num_layers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training XATGRN on {device}...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            
            # Forward pass
            h = model(expr, atac)
            
            # Reconstruction loss
            expr_recon = h.mean(dim=1)
            loss = F.mse_loss(expr_recon, expr.mean(dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Extract GRN edges
    print("Extracting GRN edges...")
    model.eval()
    
    tfs = load_tf_list()
    tf_indices = [i for i, g in enumerate(rna.var_names) if g in tfs][:20]
    target_indices = [i for i, g in enumerate(rna.var_names) if g not in tfs][:100]
    
    edges = []
    with torch.no_grad():
        subset_size = min(1000, len(train_idx))
        subset_idx = np.random.choice(len(train_idx), subset_size, replace=False)
        subset_train_idx = [train_idx[i] for i in subset_idx]
        
        expr = torch.FloatTensor(rna[subset_train_idx].X.toarray()).to(device)
        atac_subset = atac_adata[subset_train_idx]
        atac_data = atac_subset.X
        if hasattr(atac_data, 'toarray'):
            atac_data = atac_data.toarray()
        atac_tensor = torch.FloatTensor(atac_data).to(device)
        
        # Get hidden representations
        h = model(expr, atac_tensor)  # (batch, hidden_dim)
        
        # Average across cells to get gene representations
        h_avg = h.mean(dim=0)  # (hidden_dim,)
        
        # Predict edges using hidden representations
        for tf_idx in tf_indices:
            for target_idx in target_indices:
                # Get representations for TF and target
                # For simplicity, use mean representation
                pair = torch.cat([
                    h_avg.unsqueeze(0),  # Need to work with the representation
                    h_avg.unsqueeze(0)
                ], dim=1)  # (1, hidden_dim*2)
                
                # Use expression correlation as a proxy
                tf_expr = expr[:, tf_idx].mean()
                target_expr = expr[:, target_idx].mean()
                
                # Combine with learned representation
                combined = torch.cat([
                    h_avg[:128].unsqueeze(0),
                    torch.stack([tf_expr, target_expr]).unsqueeze(0)
                ], dim=1)
                
                # Pad to expected size
                if combined.size(1) < 512:
                    padding = torch.zeros(1, 512 - combined.size(1)).to(device)
                    combined = torch.cat([combined, padding], dim=1)
                
                prob = torch.sigmoid(model.edge_pred(combined[:, :512])).item()
                sign = torch.tanh(model.sign_pred(combined[:, :512])).item()
                
                edges.append({
                    'tf_idx': int(tf_idx),
                    'target_idx': int(target_idx),
                    'prob': float(prob),
                    'sign': float(sign)
                })
    
    # Save results
    results = {
        'method': 'XATGRN',
        'seed': seed,
        'n_edges': len(edges),
        'edges': edges
    }
    
    # Compute metrics
    if os.path.exists('data/ground_truth_edges.json'):
        with open('data/ground_truth_edges.json') as f:
            gt = json.load(f)
        
        gt_dict = {}
        gt_sign = {}
        for e in gt:
            if e['tf'] in rna.var_names and e['target'] in rna.var_names:
                try:
                    tf_idx = rna.var_names.get_loc(e['tf'])
                    target_idx = rna.var_names.get_loc(e['target'])
                    gt_dict[(tf_idx, target_idx)] = 1
                    gt_sign[(tf_idx, target_idx)] = e['sign']
                except:
                    pass
        
        y_true = np.array([gt_dict.get((e['tf_idx'], e['target_idx']), 0) for e in edges])
        y_score = np.array([e['prob'] for e in edges])
        y_sign_true = np.array([gt_sign.get((e['tf_idx'], e['target_idx']), 1) for e in edges])
        y_sign_pred = np.array([e['sign'] for e in edges])
        
        results['metrics'] = {
            'auroc': float(compute_auroc(y_true, y_score)),
            'auprc': float(compute_auprc(y_true, y_score)),
            'sign_accuracy': float(compute_sign_accuracy(y_sign_true, y_sign_pred))
        }
        
        print(f"\nMetrics: {results['metrics']}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved results to {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--atac', default='data/pbmc_atac_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/xatgrn_baseline/results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_xatgrn(args.rna, args.atac, args.output, args.seed, args.epochs)
