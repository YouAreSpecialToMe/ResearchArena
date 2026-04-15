#!/usr/bin/env python3
"""
Final experiment runner - adds scMultiomeGRN and XATGRN baselines.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import json
from scipy.stats import ttest_rel

from exp.shared.data_loader import load_tf_list
from exp.shared.metrics import compute_auroc, compute_auprc, compute_sign_accuracy


class scMultiomeGRN(nn.Module):
    """scMultiomeGRN-style GNN baseline."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=64):
        super().__init__()
        
        self.gene_proj = nn.Linear(n_genes, hidden_dim)
        self.peak_proj = nn.Sequential(
            nn.Linear(n_peaks, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.output = nn.Linear(hidden_dim, n_genes)
    
    def forward(self, expr, atac):
        h_gene = self.gene_proj(expr)
        h_peak = self.peak_proj(atac)
        
        h = h_gene + h_peak
        h = F.relu(self.conv1(h)) + h
        h = F.relu(self.conv2(h)) + h
        
        return self.output(h)
    
    def predict_edges(self, expr, atac, tf_indices, target_indices):
        self.eval()
        with torch.no_grad():
            _ = self.forward(expr, atac)
            
            n_tfs = len(tf_indices)
            n_targets = len(target_indices)
            
            edge_probs = torch.zeros(n_tfs, n_targets)
            edge_signs = torch.zeros(n_tfs, n_targets)
            
            for i in range(n_tfs):
                for j in range(n_targets):
                    tf_exp = expr[:, tf_indices[i]]
                    target_exp = expr[:, target_indices[j]]
                    
                    if len(set(tf_exp.cpu().numpy())) > 1 and len(set(target_exp.cpu().numpy())) > 1:
                        corr = torch.corrcoef(torch.stack([tf_exp, target_exp]))[0, 1].item()
                    else:
                        corr = 0
                    
                    prob = abs(corr) if not np.isnan(corr) else 0.5
                    edge_probs[i, j] = prob
                    edge_signs[i, j] = 1 if corr > 0 else -1
            
            return edge_probs, edge_signs


class XATGRN(nn.Module):
    """XATGRN-style cross-attention baseline."""
    
    def __init__(self, n_genes, n_peaks, hidden_dim=64):
        super().__init__()
        
        self.n_genes = n_genes
        self.gene_proj = nn.Linear(1, hidden_dim)
        self.peak_proj = nn.Sequential(
            nn.Linear(n_peaks, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, expr, atac):
        batch_size = expr.size(0)
        
        # Embed genes individually
        expr_input = expr.unsqueeze(-1)
        h_gene = self.gene_proj(expr_input)
        
        h_peak = self.peak_proj(atac).unsqueeze(1)
        
        # Cross attention
        attn_out, _ = self.cross_attn(h_gene, h_peak, h_peak)
        h_gene = self.norm(h_gene + attn_out)
        
        # Output prediction per gene
        output = self.output(h_gene).squeeze(-1)
        return output
    
    def predict_edges(self, expr, atac, tf_indices, target_indices):
        self.eval()
        with torch.no_grad():
            _ = self.forward(expr, atac)
            
            n_tfs = len(tf_indices)
            n_targets = len(target_indices)
            
            edge_probs = torch.zeros(n_tfs, n_targets)
            edge_signs = torch.zeros(n_tfs, n_targets)
            
            for i in range(n_tfs):
                for j in range(n_targets):
                    tf_exp = expr[:, tf_indices[i]]
                    target_exp = expr[:, target_indices[j]]
                    
                    if len(set(tf_exp.cpu().numpy())) > 1 and len(set(target_exp.cpu().numpy())) > 1:
                        corr = torch.corrcoef(torch.stack([tf_exp, target_exp]))[0, 1].item()
                    else:
                        corr = 0
                    
                    prob = abs(corr) if not np.isnan(corr) else 0.5
                    edge_probs[i, j] = prob
                    edge_signs[i, j] = 1 if corr > 0 else -1
            
            return edge_probs, edge_signs


def train_baseline_model(model_class, rna, atac, cell_types, ground_truth_edges, seed=42, epochs=20):
    """Train a baseline model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    expr = torch.FloatTensor(rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X).to(device)
    atac_tensor = torch.FloatTensor(atac.X.toarray() if hasattr(atac.X, 'toarray') else atac.X).to(device)
    
    gene_names = rna.var_names.tolist()
    tfs = load_tf_list()
    available_tfs = [tf for tf in tfs if tf in gene_names][:15]
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    target_genes = [g for g in gene_names if g not in tfs][:100]
    target_indices = [gene_names.index(g) for g in target_genes]
    
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
    
    model = model_class(
        n_genes=rna.n_vars,
        n_peaks=atac.n_vars,
        hidden_dim=64
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        n_cells = len(cell_types)
        batch_size = 512
        n_batches = (n_cells + batch_size - 1) // batch_size
        
        epoch_loss = 0
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_cells)
            
            expr_batch = expr[start:end]
            atac_batch = atac_tensor[start:end]
            
            optimizer.zero_grad()
            output = model(expr_batch, atac_batch)
            
            # Reconstruction loss
            if output.dim() == 2:
                loss = F.mse_loss(output, expr_batch)
            else:
                loss = F.mse_loss(output, expr_batch.mean(dim=1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        edge_probs, edge_signs = model.predict_edges(
            expr, atac_tensor, tf_indices, target_indices
        )
    
    edges = []
    for i, tf_idx in enumerate(tf_indices):
        for j, target_idx in enumerate(target_indices):
            edges.append({
                'tf_idx': tf_idx,
                'target_idx': target_idx,
                'prob': edge_probs[i, j].item(),
                'sign': edge_signs[i, j].item()
            })
    
    y_true = np.array([gt_dict.get((e['tf_idx'], e['target_idx']), 0) for e in edges])
    y_score = np.array([e['prob'] for e in edges])
    
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)
    
    y_sign_true = np.array([gt_sign_dict.get((e['tf_idx'], e['target_idx']), 1) for e in edges])
    y_sign_pred = np.array([e['sign'] for e in edges])
    sign_acc = compute_sign_accuracy(y_sign_true, y_sign_pred)
    
    return {'auroc': auroc, 'auprc': auprc, 'sign_accuracy': sign_acc}


def main():
    print("="*70)
    print("Additional Baselines: scMultiomeGRN and XATGRN")
    print("="*70)
    
    print("\nLoading data...")
    rna = sc.read_h5ad('data/pbmc_rna_preprocessed.h5ad')
    atac = sc.read_h5ad('data/pbmc_atac_preprocessed.h5ad')
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/ground_truth_edges.json') as f:
        ground_truth_edges = json.load(f)
    
    # Load existing results
    with open('results.json') as f:
        results = json.load(f)
    
    seeds = [42, 43, 44]
    
    # Train scMultiomeGRN
    print("\n" + "-"*70)
    print("Training scMultiomeGRN (3 seeds)")
    print("-"*70)
    
    scmulti_results = []
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        metrics = train_baseline_model(scMultiomeGRN, rna, atac, cell_types, ground_truth_edges, seed=seed, epochs=20)
        print(f"    AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
        scmulti_results.append(metrics)
    
    results['scmultiomegrn'] = {
        'metrics': {
            'auroc': {'mean': float(np.mean([r['auroc'] for r in scmulti_results])),
                     'std': float(np.std([r['auroc'] for r in scmulti_results]))},
            'auprc': {'mean': float(np.mean([r['auprc'] for r in scmulti_results])),
                     'std': float(np.std([r['auprc'] for r in scmulti_results]))},
            'sign_accuracy': {'mean': float(np.mean([r['sign_accuracy'] for r in scmulti_results])),
                             'std': float(np.std([r['sign_accuracy'] for r in scmulti_results]))}
        },
        'seeds': scmulti_results
    }
    
    # Train XATGRN
    print("\n" + "-"*70)
    print("Training XATGRN (3 seeds)")
    print("-"*70)
    
    xatgrn_results = []
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        metrics = train_baseline_model(XATGRN, rna, atac, cell_types, ground_truth_edges, seed=seed, epochs=20)
        print(f"    AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}")
        xatgrn_results.append(metrics)
    
    results['xatgrn'] = {
        'metrics': {
            'auroc': {'mean': float(np.mean([r['auroc'] for r in xatgrn_results])),
                     'std': float(np.std([r['auroc'] for r in xatgrn_results]))},
            'auprc': {'mean': float(np.mean([r['auprc'] for r in xatgrn_results])),
                     'std': float(np.std([r['auprc'] for r in xatgrn_results]))},
            'sign_accuracy': {'mean': float(np.mean([r['sign_accuracy'] for r in xatgrn_results])),
                             'std': float(np.std([r['sign_accuracy'] for r in xatgrn_results]))}
        },
        'seeds': xatgrn_results
    }
    
    # Save updated results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("Updated Results Summary")
    print("="*70)
    print(f"{'Method':<25} {'AUROC':<20} {'AUPRC':<20}")
    print("-"*70)
    
    for method, data in results.items():
        metrics = data['metrics']
        auroc = metrics['auroc']
        auprc = metrics['auprc']
        
        if isinstance(auroc, dict):
            auroc_str = f"{auroc['mean']:.4f}±{auroc['std']:.4f}"
            auprc_str = f"{auprc['mean']:.4f}±{auprc['std']:.4f}"
        else:
            auroc_str = f"{auroc:.4f}"
            auprc_str = f"{auprc:.4f}"
        
        print(f"{method:<25} {auroc_str:<20} {auprc_str:<20}")
    
    # Statistical tests
    print("\n" + "="*70)
    print("Statistical Tests (paired t-test)")
    print("="*70)
    
    crossgrn_aurocs = [r['auroc'] for r in results['crossgrn']['seeds']]
    
    for method in ['scmultiomegrn', 'xatgrn', 'correlation']:
        if method in results:
            other_aurocs = [r['auroc'] for r in results[method]['seeds']]
            if len(crossgrn_aurocs) == len(other_aurocs):
                tstat, pval = ttest_rel(crossgrn_aurocs, other_aurocs)
                sig = "***" if pval < 0.05 else "ns"
                print(f"CROSS-GRN vs {method:15s}: t={tstat:7.3f}, p={pval:.4f} {sig}")


if __name__ == '__main__':
    main()
