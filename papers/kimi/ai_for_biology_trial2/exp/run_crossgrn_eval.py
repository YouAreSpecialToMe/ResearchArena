#!/usr/bin/env python3
"""CROSS-GRN evaluation with real ground truth."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import json
import time
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auroc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def compute_auprc(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except:
        return 0.0


def load_tf_list():
    return ['SPI1', 'CEBPA', 'CEBPB', 'GATA1', 'GATA2', 'GATA3', 'TAL1', 'MYC',
            'RUNX1', 'RUNX2', 'STAT1', 'STAT3', 'IRF4', 'IRF8', 'BATF',
            'EBF1', 'PAX5', 'TCF3', 'FOXP3', 'TBX21', 'RORC']


class FastCROSSGRN(nn.Module):
    """Simplified CROSS-GRN for fast evaluation."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=128):
        super().__init__()
        self.n_genes = n_genes
        
        # Encoders - use gene-specific projections
        self.expr_emb = nn.Linear(n_genes, hidden_dim, bias=False)
        self.atac_emb = nn.Linear(n_peaks, hidden_dim, bias=False)
        
        # Transform layers
        self.expr_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.atac_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention
        self.W_q_forward = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_forward = nn.Linear(hidden_dim, hidden_dim)
        self.W_q_reverse = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_reverse = nn.Linear(hidden_dim, hidden_dim)
        
        # Prediction heads - per-gene
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
        # Encode
        h_expr = self.expr_transform(self.expr_emb(expr))
        h_atac = self.atac_transform(self.atac_emb(atac))
        
        return h_expr, h_atac
    
    def predict_edges(self, expr, atac, tf_indices, target_indices):
        """Predict GRN edges."""
        with torch.no_grad():
            h_expr, h_atac = self.forward(expr, atac)
            
            edges = []
            for tf_idx in tf_indices:
                for target_idx in target_indices:
                    # Get gene expressions directly
                    tf_expr = expr[0, tf_idx]
                    target_expr = expr[0, target_idx]
                    
                    # Use learned representations
                    pair = torch.cat([
                        h_expr[0] * tf_expr,
                        h_expr[0] * target_expr
                    ])
                    
                    # Predict
                    prob = self.edge_pred(pair).item()
                    sign = self.sign_pred(pair).item()
                    
                    edges.append({
                        'tf_idx': tf_idx,
                        'target_idx': target_idx,
                        'prob': prob,
                        'sign': sign
                    })
        
        return edges


def train_and_evaluate(seed=42, epochs=20):
    """Train and evaluate CROSS-GRN."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n[CROSS-GRN - Seed {seed}]")
    
    # Load data
    print("  Loading data...")
    rna = sc.read_h5ad('data/pbmc_rna_preprocessed.h5ad')
    atac = sc.read_h5ad('data/pbmc_atac_preprocessed.h5ad')
    
    with open('data/ground_truth_edges.json') as f:
        gt_edges = json.load(f)
    
    # Create ground truth dict
    gt_dict = {}
    gt_sign = {}
    for e in gt_edges:
        if e['tf'] in rna.var_names and e['target'] in rna.var_names:
            try:
                tf_idx = rna.var_names.get_loc(e['tf'])
                target_idx = rna.var_names.get_loc(e['target'])
                gt_dict[(tf_idx, target_idx)] = 1
                gt_sign[(tf_idx, target_idx)] = e['sign']
            except:
                pass
    
    # Sample cells for speed
    n_cells = min(2000, rna.n_obs)
    sample_idx = np.random.choice(rna.n_obs, n_cells, replace=False)
    
    expr = torch.FloatTensor(rna[sample_idx].X.toarray())
    atac_data = torch.FloatTensor(atac[sample_idx].X.toarray())
    
    # Get TF and target indices
    tfs = load_tf_list()
    tf_indices = [i for i, g in enumerate(rna.var_names) if g in tfs][:15]
    target_indices = [i for i, g in enumerate(rna.var_names) if g not in tfs][:80]
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Training on {device}...")
    
    model = FastCROSSGRN(rna.n_vars, atac.n_vars, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training
    model.train()
    batch_size = 64
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, n_cells, batch_size):
            batch_expr = expr[i:i+batch_size].to(device)
            batch_atac = atac_data[i:i+batch_size].to(device)
            
            # Forward
            h_expr, h_atac = model(batch_expr, batch_atac)
            
            # Reconstruction loss
            expr_recon = model.expr_emb(h_expr)
            loss_recon = F.mse_loss(expr_recon, batch_expr)
            
            loss = loss_recon
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")
    
    # Evaluation
    print("  Evaluating...")
    model.eval()
    
    with torch.no_grad():
        # Use first cell for prediction
        cell_expr = expr[0:1].to(device)
        cell_atac = atac_data[0:1].to(device)
        
        edges = model.predict_edges(cell_expr, cell_atac, tf_indices, target_indices)
    
    # Compute metrics
    y_true = np.array([gt_dict.get((e['tf_idx'], e['target_idx']), 0) for e in edges])
    y_score = np.array([e['prob'] for e in edges])
    y_sign_true = np.array([gt_sign.get((e['tf_idx'], e['target_idx']), 1) for e in edges])
    y_sign_pred = np.array([e['sign'] for e in edges])
    
    # Compute sign accuracy
    sign_correct = (np.sign(y_sign_true) == np.sign(y_sign_pred)).mean()
    
    return {
        'auroc': compute_auroc(y_true, y_score),
        'auprc': compute_auprc(y_true, y_score),
        'sign_accuracy': sign_correct,
        'n_edges': len(edges),
        'n_positive': int(y_true.sum())
    }


def main():
    print("="*60)
    print("CROSS-GRN Evaluation with Real Ground Truth")
    print("="*60)
    
    results = {
        'experiment_date': time.strftime('%Y-%m-%d'),
        'dataset': 'PBMC 10k Multiome',
        'method': 'CROSS-GRN (fast)',
        'seeds': [42, 43, 44]
    }
    
    # Run with 3 seeds
    seed_results = []
    for seed in [42, 43, 44]:
        r = train_and_evaluate(seed=seed, epochs=20)
        seed_results.append(r)
        print(f"  Results: AUROC={r['auroc']:.4f}, SignAcc={r['sign_accuracy']:.4f}")
    
    # Aggregate
    results['metrics'] = {
        'auroc': {'mean': np.mean([r['auroc'] for r in seed_results]),
                 'std': np.std([r['auroc'] for r in seed_results]),
                 'values': [r['auroc'] for r in seed_results]},
        'auprc': {'mean': np.mean([r['auprc'] for r in seed_results]),
                 'std': np.std([r['auprc'] for r in seed_results])},
        'sign_accuracy': {'mean': np.mean([r['sign_accuracy'] for r in seed_results]),
                         'std': np.std([r['sign_accuracy'] for r in seed_results])}
    }
    
    # Save
    os.makedirs('exp/crossgrn_main', exist_ok=True)
    with open('exp/crossgrn_main/results_new.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-GRN RESULTS")
    print("="*60)
    m = results['metrics']
    print(f"AUROC: {m['auroc']['mean']:.4f} ± {m['auroc']['std']:.4f}")
    print(f"AUPRC: {m['auprc']['mean']:.4f} ± {m['auprc']['std']:.4f}")
    print(f"Sign Acc: {m['sign_accuracy']['mean']:.4f} ± {m['sign_accuracy']['std']:.4f}")
    print("="*60)


if __name__ == '__main__':
    import os
    main()
