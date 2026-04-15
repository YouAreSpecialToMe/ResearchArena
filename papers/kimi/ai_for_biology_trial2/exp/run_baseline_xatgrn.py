#!/usr/bin/env python3
"""Run XATGRN baseline."""
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


class XATGRNBaseline(nn.Module):
    """XATGRN-style cross-attention baseline."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=256, num_layers=2):
        super().__init__()
        self.n_genes = n_genes
        self.n_peaks = n_peaks
        
        # Initial embeddings
        self.gene_emb = nn.Linear(1, hidden_dim)
        self.peak_emb = nn.Linear(1, hidden_dim)
        
        # Cross-attention layers (XATGRN style: dual complex embedding)
        self.cross_attn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Self-attention for gene-gene interactions
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
    
    def forward(self, expr, atac):
        """
        Args:
            expr: (batch, n_genes)
            atac: (batch, n_peaks)
        """
        batch_size = expr.size(0)
        
        # Embed genes and peaks
        expr_input = expr.unsqueeze(-1)  # (batch, n_genes, 1)
        atac_input = atac.unsqueeze(-1)  # (batch, n_peaks, 1)
        
        h_gene = self.gene_emb(expr_input)  # (batch, n_genes, hidden)
        h_peak = self.peak_emb(atac_input)  # (batch, n_peaks, hidden)
        
        # Cross-attention: genes attend to peaks (XATGRN key component)
        for attn, norm in zip(self.cross_attn_layers, self.layer_norms):
            attn_out, _ = attn(h_gene, h_peak, h_peak)
            h_gene = norm(h_gene + attn_out)
        
        # Self-attention for gene-gene interactions
        self_attn_out, _ = self.self_attn(h_gene, h_gene, h_gene)
        h_gene = self.self_norm(h_gene + self_attn_out)
        
        # Output prediction
        output = self.output_proj(h_gene.mean(dim=1))  # Average pool then predict
        
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


def train_xatgrn(rna_path, atac_path, output_path, config):
    """Train XATGRN baseline."""
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training XATGRN on {device} with seed {seed}")
    
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
    model = XATGRNBaseline(
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
            loss = F.mse_loss(outputs['output'], expr.mean(dim=1, keepdim=True).expand(-1, expr.size(1)))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        'method': 'XATGRN',
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
    parser.add_argument('--output', default='exp/xatgrn_baseline/results.json')
    parser.add_argument('--model_path', default='models/xatgrn.pt')
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
    
    train_xatgrn(args.rna, args.atac, args.output, config)
