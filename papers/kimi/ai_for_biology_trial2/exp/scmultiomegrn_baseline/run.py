#!/usr/bin/env python3
"""scMultiomeGRN baseline - GCN-based approach."""
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
from exp.shared.metrics import compute_auroc, compute_auprc, compute_epr


class SimpleGCN(nn.Module):
    """Simple GCN for scMultiomeGRN."""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.grn_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


def train_scmultiomegrn(rna_path, atac_path, output_path, seed=42, epochs=50):
    """Train scMultiomeGRN baseline."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("Loading data...")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    
    cell_types = rna.obs['cell_type'].tolist()
    
    # Get splits
    with open('data/splits.json') as f:
        splits = json.load(f)
    
    train_idx = splits['train']
    
    # Create datasets
    train_dataset = MultiOmicDataset(
        rna[train_idx], atac[train_idx], 
        [cell_types[i] for i in train_idx],
        mode='train'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # Store atac anndata before any tensor conversion
    atac_adata = atac
    
    # Create model
    n_genes = rna.n_vars
    n_peaks = atac_adata.n_vars
    
    # Concatenate RNA + ATAC as input
    input_dim = n_genes + n_peaks
    model = SimpleGCN(input_dim, hidden_dim=256, num_layers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            
            # Concatenate features
            x = torch.cat([expr, atac], dim=1)
            
            # Forward pass
            h = model(x)
            
            # Reconstruction loss
            loss = F.mse_loss(h.mean(dim=1), x.mean(dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Extract GRN edges
    print("Extracting GRN edges...")
    model.eval()
    
    # Get TF indices
    tfs = load_tf_list()
    tf_indices = [i for i, g in enumerate(rna.var_names) if g in tfs][:20]
    target_indices = [i for i, g in enumerate(rna.var_names) if g not in tfs][:100]
    
    edges = []
    with torch.no_grad():
        # Use a subset of cells
        subset_size = min(1000, len(train_idx))
        subset_idx = np.random.choice(len(train_idx), subset_size, replace=False)
        subset_train_idx = [train_idx[i] for i in subset_idx]
        
        expr = torch.FloatTensor(rna[subset_train_idx].X.toarray()).to(device)
        atac_subset = atac_adata[subset_train_idx]
        atac_data = atac_subset.X
        if hasattr(atac_data, 'toarray'):
            atac_data = atac_data.toarray()
        atac_tensor = torch.FloatTensor(atac_data).to(device)
        x = torch.cat([expr, atac_tensor], dim=1)
        
        h = model(x)
        
        # Predict edges using representation similarity
        for tf_idx in tf_indices:
            for target_idx in target_indices:
                # Get representations for TF and target
                tf_repr = h[:, tf_idx].mean(dim=0)  # Average across cells
                target_repr = h[:, target_idx].mean(dim=0)
                
                # Concatenate and predict
                pair = torch.cat([tf_repr, target_repr]).unsqueeze(0)
                prob = model.grn_head(pair).item()
                
                edges.append({
                    'tf_idx': int(tf_idx),
                    'target_idx': int(target_idx),
                    'prob': float(prob),
                    'sign': 0  # No sign prediction in baseline
                })
    
    # Save results
    results = {
        'method': 'scMultiomeGRN',
        'seed': seed,
        'n_edges': len(edges),
        'edges': edges
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--atac', default='data/pbmc_atac_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/scmultiomegrn_baseline/results.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_scmultiomegrn(args.rna, args.atac, args.output, args.seed, args.epochs)
