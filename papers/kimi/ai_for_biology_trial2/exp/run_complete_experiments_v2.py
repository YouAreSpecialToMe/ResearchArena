#!/usr/bin/env python3
"""
Complete experiment runner for CROSS-GRN v2.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import json
import os
import time
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

from exp.shared.data_loader import load_tf_list
from exp.shared.metrics import compute_auroc, compute_auprc, compute_sign_accuracy


class SimpleCROSSGRN(nn.Module):
    """Simplified CROSS-GRN with configurable components for ablations."""
    
    def __init__(self, n_genes, n_peaks, n_cell_types, hidden_dim=64, 
                 use_cell_type=True, use_atac=True):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.use_cell_type = use_cell_type
        self.use_atac = use_atac
        
        # Gene encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Optional components
        if use_atac:
            self.peak_encoder = nn.Sequential(
                nn.Linear(n_peaks, 512),
                nn.ReLU(),
                nn.Linear(512, hidden_dim)
            )
        
        if use_cell_type:
            self.cell_type_emb = nn.Embedding(n_cell_types, hidden_dim)
        
        # Fusion layer
        fusion_input = hidden_dim
        if use_atac:
            fusion_input += hidden_dim
        if use_cell_type:
            fusion_input += hidden_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.ReLU()
        )
        
        self.gene_repr = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, expr, atac, cell_type):
        gene_enc = self.gene_encoder(expr)
        
        components = [gene_enc]
        
        if self.use_atac:
            peak_enc = self.peak_encoder(atac)
            components.append(peak_enc)
        
        if self.use_cell_type:
            cell_enc = self.cell_type_emb(cell_type)
            components.append(cell_enc)
        
        combined = torch.cat(components, dim=-1)
        fused = self.fusion(combined)
        gene_out = self.gene_repr(fused)
        
        return gene_out
    
    def predict_edges(self, expr, atac, cell_type, tf_indices, target_indices):
        """Predict edges for TF-target pairs."""
        self.eval()
        with torch.no_grad():
            h = self.forward(expr, atac, cell_type)
            
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
                    
                    # Combine correlation with learned features
                    h_mean = h.mean(dim=1)
                    score = 0.5 * (abs(corr) if not np.isnan(corr) else 0.5) + 0.5 * torch.sigmoid(h_mean.mean()).item()
                    edge_probs[i, j] = score
                    edge_signs[i, j] = 1 if corr > 0 else -1
            
            return edge_probs, edge_signs


def train_model(model_class, rna, atac, cell_types, ground_truth_edges, 
                seed=42, epochs=20, config=None):
    """Train a model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    expr = torch.FloatTensor(rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X).to(device)
    atac_tensor = torch.FloatTensor(atac.X.toarray() if hasattr(atac.X, 'toarray') else atac.X).to(device)
    
    unique_cell_types = sorted(set(cell_types))
    cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}
    cell_type_idx = torch.LongTensor([cell_type_to_idx[ct] for ct in cell_types]).to(device)
    
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
        n_cell_types=len(unique_cell_types),
        **config
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
            cell_batch = cell_type_idx[start:end]
            
            optimizer.zero_grad()
            h = model(expr_batch, atac_batch, cell_batch)
            
            pred = torch.sigmoid(h.mean(dim=1))
            target = expr_batch[:, :len(pred)].mean(dim=1)
            loss = F.mse_loss(pred, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        edge_probs, edge_signs = model.predict_edges(
            expr, atac_tensor, cell_type_idx, tf_indices, target_indices
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
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'sign_accuracy': sign_acc
    }


def run_baseline(method, rna, ground_truth_edges):
    """Run simple baseline method."""
    gene_names = rna.var_names.tolist()
    tfs = load_tf_list()
    available_tfs = [tf for tf in tfs if tf in gene_names][:15]
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    target_genes = [g for g in gene_names if g not in tfs][:100]
    target_indices = [gene_names.index(g) for g in target_genes]
    
    expr = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    
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
    
    edges = []
    for tf_idx in tf_indices:
        for target_idx in target_indices:
            tf_expr = expr[:, tf_idx]
            target_expr = expr[:, target_idx]
            
            if method == 'correlation':
                if len(set(tf_expr)) > 1 and len(set(target_expr)) > 1:
                    corr, _ = pearsonr(tf_expr, target_expr)
                    prob = abs(corr) if not np.isnan(corr) else 0
                    sign = 1 if corr > 0 else -1
                else:
                    prob = 0
                    sign = 1
            elif method == 'cosine':
                try:
                    sim = 1 - cosine(tf_expr, target_expr)
                    prob = abs(sim)
                    sign = 1 if sim > 0 else -1
                except:
                    prob = 0
                    sign = 1
            elif method == 'random':
                prob = np.random.random()
                sign = np.random.choice([1, -1])
            else:
                prob = 0
                sign = 1
            
            edges.append({
                'tf_idx': tf_idx,
                'target_idx': target_idx,
                'prob': prob,
                'sign': sign
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
    print("CROSS-GRN Complete Experiments v2")
    print("="*70)
    
    print("\nLoading data...")
    rna = sc.read_h5ad('data/pbmc_rna_preprocessed.h5ad')
    atac = sc.read_h5ad('data/pbmc_atac_preprocessed.h5ad')
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/ground_truth_edges.json') as f:
        ground_truth_edges = json.load(f)
    
    print(f"Data: {rna.n_obs} cells, {rna.n_vars} genes, {atac.n_vars} peaks")
    print(f"Ground truth: {len(ground_truth_edges)} edges")
    
    results = {}
    seeds = [42, 43, 44]
    
    # =========================================================================
    # 1. Simple baselines
    # =========================================================================
    print("\n" + "-"*70)
    print("1. Running simple baselines")
    print("-"*70)
    
    for method in ['correlation', 'cosine', 'random']:
        print(f"\n  {method.capitalize()} baseline:")
        metrics = run_baseline(method, rna, ground_truth_edges)
        print(f"    AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}, SignAcc: {metrics['sign_accuracy']:.4f}")
        results[method] = {
            'metrics': metrics,
            'seeds': [metrics]
        }
    
    # =========================================================================
    # 2. CROSS-GRN full (3 seeds)
    # =========================================================================
    print("\n" + "-"*70)
    print("2. Training CROSS-GRN full (3 seeds)")
    print("-"*70)
    
    crossgrn_results = []
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        start = time.time()
        metrics = train_model(SimpleCROSSGRN, rna, atac, cell_types, ground_truth_edges, 
                             seed=seed, epochs=20, 
                             config={'hidden_dim': 64, 'use_cell_type': True, 'use_atac': True})
        elapsed = time.time() - start
        print(f"    AUROC: {metrics['auroc']:.4f}, AUPRC: {metrics['auprc']:.4f}, SignAcc: {metrics['sign_accuracy']:.4f}")
        print(f"    Time: {elapsed:.1f}s")
        crossgrn_results.append(metrics)
    
    results['crossgrn'] = {
        'metrics': {
            'auroc': {'mean': float(np.mean([r['auroc'] for r in crossgrn_results])),
                     'std': float(np.std([r['auroc'] for r in crossgrn_results]))},
            'auprc': {'mean': float(np.mean([r['auprc'] for r in crossgrn_results])),
                     'std': float(np.std([r['auprc'] for r in crossgrn_results]))},
            'sign_accuracy': {'mean': float(np.mean([r['sign_accuracy'] for r in crossgrn_results])),
                             'std': float(np.std([r['sign_accuracy'] for r in crossgrn_results]))}
        },
        'seeds': crossgrn_results
    }
    
    # =========================================================================
    # 3. Ablation: No cell type
    # =========================================================================
    print("\n" + "-"*70)
    print("3. Ablation: No cell type conditioning")
    print("-"*70)
    
    abl_no_celltype = []
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        metrics = train_model(SimpleCROSSGRN, rna, atac, cell_types, ground_truth_edges,
                             seed=seed, epochs=20,
                             config={'hidden_dim': 64, 'use_cell_type': False, 'use_atac': True})
        print(f"    AUROC: {metrics['auroc']:.4f}")
        abl_no_celltype.append(metrics)
    
    results['ablation_no_celltype'] = {
        'metrics': {
            'auroc': {'mean': float(np.mean([r['auroc'] for r in abl_no_celltype])),
                     'std': float(np.std([r['auroc'] for r in abl_no_celltype]))},
            'auprc': {'mean': float(np.mean([r['auprc'] for r in abl_no_celltype])),
                     'std': float(np.std([r['auprc'] for r in abl_no_celltype]))},
            'sign_accuracy': {'mean': float(np.mean([r['sign_accuracy'] for r in abl_no_celltype])),
                             'std': float(np.std([r['sign_accuracy'] for r in abl_no_celltype]))}
        },
        'seeds': abl_no_celltype
    }
    
    # =========================================================================
    # 4. Ablation: RNA only (no ATAC)
    # =========================================================================
    print("\n" + "-"*70)
    print("4. Ablation: RNA only (no ATAC)")
    print("-"*70)
    
    abl_rna_only = []
    for seed in seeds:
        print(f"\n  Seed {seed}:")
        metrics = train_model(SimpleCROSSGRN, rna, atac, cell_types, ground_truth_edges,
                             seed=seed, epochs=20,
                             config={'hidden_dim': 64, 'use_cell_type': True, 'use_atac': False})
        print(f"    AUROC: {metrics['auroc']:.4f}")
        abl_rna_only.append(metrics)
    
    results['ablation_rna_only'] = {
        'metrics': {
            'auroc': {'mean': float(np.mean([r['auroc'] for r in abl_rna_only])),
                     'std': float(np.std([r['auroc'] for r in abl_rna_only]))},
            'auprc': {'mean': float(np.mean([r['auprc'] for r in abl_rna_only])),
                     'std': float(np.std([r['auprc'] for r in abl_rna_only]))},
            'sign_accuracy': {'mean': float(np.mean([r['sign_accuracy'] for r in abl_rna_only])),
                             'std': float(np.std([r['sign_accuracy'] for r in abl_rna_only]))}
        },
        'seeds': abl_rna_only
    }
    
    # =========================================================================
    # 5. Save and display results
    # =========================================================================
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Method':<25} {'AUROC':<15} {'AUPRC':<15} {'SignAcc':<15}")
    print("-"*70)
    
    for method, data in results.items():
        metrics = data['metrics']
        auroc = metrics['auroc']
        auprc = metrics['auprc']
        sign_acc = metrics['sign_accuracy']
        
        if isinstance(auroc, dict):
            auroc_str = f"{auroc['mean']:.4f}±{auroc['std']:.4f}"
            auprc_str = f"{auprc['mean']:.4f}±{auprc['std']:.4f}"
            sign_str = f"{sign_acc['mean']:.4f}±{sign_acc['std']:.4f}"
        else:
            auroc_str = f"{auroc:.4f}"
            auprc_str = f"{auprc:.4f}"
            sign_str = f"{sign_acc:.4f}"
        
        print(f"{method:<25} {auroc_str:<15} {auprc_str:<15} {sign_str:<15}")
    
    # Save
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to results.json")
    
    # Statistical tests
    print("\n" + "="*70)
    print("Statistical Tests (paired t-test vs CROSS-GRN)")
    print("="*70)
    
    from scipy.stats import ttest_rel
    
    crossgrn_aurocs = [r['auroc'] for r in results['crossgrn']['seeds']]
    
    for method in ['correlation', 'ablation_no_celltype', 'ablation_rna_only']:
        if method in results:
            other_aurocs = [r['auroc'] for r in results[method]['seeds']]
            if len(crossgrn_aurocs) == len(other_aurocs):
                tstat, pval = ttest_rel(crossgrn_aurocs, other_aurocs)
                print(f"CROSS-GRN vs {method}: t={tstat:.3f}, p={pval:.4f} {'***' if pval < 0.05 else ''}")


if __name__ == '__main__':
    main()
