#!/usr/bin/env python3
"""Ablation: RNA-only (no ATAC data)."""
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
from exp.shared.data_loader import MultiOmicDataset, load_tf_list
from exp.shared.models import TransformerEncoder
from exp.shared.metrics import compute_auroc, compute_auprc, compute_epr


class RNAOnlyModel(nn.Module):
    """CROSS-GRN variant using only RNA data."""
    
    def __init__(self, n_genes, n_cell_types, hidden_dim=384, num_layers=4, num_heads=6):
        super().__init__()
        self.encoder = TransformerEncoder(n_genes, hidden_dim, num_layers, num_heads)
        self.cell_type_emb = nn.Embedding(n_cell_types, 128)
        self.expr_pred = nn.Linear(hidden_dim, n_genes)
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
    
    def forward(self, expr, cell_type_idx):
        h = self.encoder(expr.unsqueeze(1))
        expr_pred = self.expr_pred(h.squeeze(1))
        return {'expr_pred': expr_pred, 'h': h.squeeze(1)}


def train_rna_only(rna_path, output_path, seed=42, epochs=50):
    """Train RNA-only model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training RNA-only model on {device} with seed {seed}")
    
    # Load data
    print("Loading RNA data...")
    rna = sc.read_h5ad(rna_path)
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/splits.json') as f:
        splits = json.load(f)
    
    train_idx = splits['train']
    val_idx = splits['val']
    
    # Create dummy ATAC for dataset compatibility
    atac = rna.copy()
    
    train_dataset = MultiOmicDataset(
        rna[train_idx], atac[train_idx],
        [cell_types[i] for i in train_idx],
        mode='train', mask_ratio=0.15
    )
    
    val_dataset = MultiOmicDataset(
        rna[val_idx], atac[val_idx],
        [cell_types[i] for i in val_idx],
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    unique_cell_types = sorted(set(cell_types))
    n_cell_types = len(unique_cell_types)
    
    model = RNAOnlyModel(
        n_genes=rna.n_vars,
        n_cell_types=n_cell_types,
        hidden_dim=384,
        num_layers=4,
        num_heads=6
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            expr = batch['expr'].to(device)
            expr_mask = batch['expr_mask'].to(device)
            cell_type = batch['cell_type'].to(device)
            
            outputs = model(expr, cell_type)
            
            expr_pred = outputs['expr_pred']
            expr_pred_masked = expr_pred[expr_mask]
            expr_target_masked = expr[expr_mask]
            
            if len(expr_pred_masked) > 0:
                loss = F.mse_loss(expr_pred_masked, expr_target_masked)
            else:
                loss = torch.tensor(0.0, device=device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                expr = batch['expr'].to(device)
                cell_type = batch['cell_type'].to(device)
                
                outputs = model(expr, cell_type)
                loss = F.mse_loss(outputs['expr_pred'], expr)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    print("Training completed!")
    
    # Extract GRN edges
    print("Extracting GRN edges...")
    model.eval()
    
    tfs = load_tf_list()
    tf_indices = [i for i, g in enumerate(rna.var_names) if g in tfs][:20]
    target_indices = [i for i, g in enumerate(rna.var_names) if g not in tfs][:100]
    
    edges = []
    
    with torch.no_grad():
        for batch in val_loader:
            expr = batch['expr'].to(device)
            cell_type = batch['cell_type'].to(device)
            
            outputs = model(expr, cell_type)
            h = outputs['h']
            
            for b in range(expr.size(0)):
                for tf_idx in tf_indices:
                    for target_idx in target_indices:
                        tf_repr = h[b, tf_idx]
                        target_repr = h[b, target_idx]
                        pair = torch.cat([tf_repr.unsqueeze(0), target_repr.unsqueeze(0)])
                        
                        prob = model.edge_pred(pair).item()
                        sign = model.sign_pred(pair).item()
                        
                        edges.append({
                            'tf_idx': int(tf_idx),
                            'target_idx': int(target_idx),
                            'prob': float(prob),
                            'sign': float(sign)
                        })
    
    results = {
        'method': 'CROSS-GRN-RNA-only',
        'seed': seed,
        'n_edges': len(edges),
        'edges': edges,
        'history': history
    }
    
    # Compute metrics
    if os.path.exists('data/ground_truth_edges.json'):
        with open('data/ground_truth_edges.json') as f:
            gt = json.load(f)
        
        gt_dict = {}
        for e in gt:
            if e['tf'] in rna.var_names and e['target'] in rna.var_names:
                try:
                    tf_idx = rna.var_names.get_loc(e['tf'])
                    target_idx = rna.var_names.get_loc(e['target'])
                    gt_dict[(tf_idx, target_idx)] = 1
                except:
                    pass
        
        y_true = np.array([gt_dict.get((e['tf_idx'], e['target_idx']), 0) for e in edges])
        y_score = np.array([e['prob'] for e in edges])
        
        results['metrics'] = {
            'auroc': float(compute_auroc(y_true, y_score)),
            'auprc': float(compute_auprc(y_true, y_score)),
            'epr_100': float(compute_epr(y_true, y_score, k=100))
        }
        
        print(f"\nMetrics: {results['metrics']}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved results to {output_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/ablation_rna_only/results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_rna_only(args.rna, args.output, args.seed, args.epochs)
