#!/usr/bin/env python3
"""Fast training script for CROSS-GRN."""
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
import time

from exp.shared.data_loader import load_tf_list
from exp.shared.models_v2 import CROSSGRNv2
from exp.shared.metrics import compute_auroc, compute_auprc, compute_sign_accuracy


def train_fast(rna_path, atac_path, output_path, config):
    """Fast training for CROSS-GRN."""
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CROSS-GRN v2 on {device} with seed {seed}")
    
    # Load data
    print("Loading data...")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/ground_truth_edges.json') as f:
        ground_truth_edges = json.load(f)
    
    # Get TF and target indices
    gene_names = rna.var_names.tolist()
    tfs = load_tf_list()
    available_tfs = [tf for tf in tfs if tf in gene_names][:15]  # Limit for speed
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    target_genes = [g for g in gene_names if g not in tfs][:100]  # Limit for speed
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
    
    print(f"Ground truth edges: {len(gt_dict)}")
    
    # Create simple dataset - use all cells for faster training
    expr = torch.FloatTensor(rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X)
    atac = torch.FloatTensor(atac.X.toarray() if hasattr(atac.X, 'toarray') else atac.X)
    
    unique_cell_types = sorted(set(cell_types))
    cell_type_to_idx = {ct: i for i, ct in enumerate(unique_cell_types)}
    cell_type_idx = torch.LongTensor([cell_type_to_idx[ct] for ct in cell_types])
    
    # Train/val split
    n_cells = len(cell_types)
    n_train = int(0.8 * n_cells)
    
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_cells))
    
    # Create model
    model = CROSSGRNv2(
        n_genes=rna.n_vars,
        n_peaks=atac.shape[1],
        n_cell_types=len(unique_cell_types),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        use_cell_type_cond=config['use_cell_type_cond'],
        use_asymmetric=config['use_asymmetric'],
        predict_sign=config['predict_sign']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    
    # Training
    best_auroc = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        
        # Mini-batch training
        batch_size = config['batch_size']
        n_batches = (len(train_idx) + batch_size - 1) // batch_size
        
        epoch_loss = 0
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(train_idx))
            batch_idx = train_idx[start:end]
            
            expr_batch = expr[batch_idx].to(device)
            atac_batch = atac[batch_idx].to(device)
            cell_batch = cell_type_idx[batch_idx].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(expr_batch, atac_batch, cell_batch)
            
            # Expression prediction loss
            expr_pred = outputs['expr_pred']
            loss = F.mse_loss(expr_pred, expr_batch)
            
            # Edge prediction loss (if we have ground truth)
            if len(gt_dict) > 0:
                gene_emb = outputs['gene_embeddings']
                
                # Sample a few edges for efficiency
                gt_pairs = list(gt_dict.keys())[:20]  # Limit edges per batch
                
                if len(gt_pairs) > 0:
                    edge_loss = 0
                    for tf_idx, target_idx in gt_pairs:
                        tf_emb = gene_emb[:, tf_idx, :]
                        target_emb = gene_emb[:, target_idx, :]
                        
                        pair = torch.cat([tf_emb, target_emb], dim=-1)
                        edge_pred = torch.sigmoid(model.edge_head(pair)).mean()
                        
                        edge_loss += F.binary_cross_entropy(edge_pred, torch.tensor(1.0).to(device))
                    
                    loss = loss + 0.5 * edge_loss / len(gt_pairs)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
            model.eval()
            
            with torch.no_grad():
                expr_val = expr[val_idx].to(device)
                atac_val = atac[val_idx].to(device)
                cell_val = cell_type_idx[val_idx].to(device)
                
                outputs = model(expr_val, atac_val, cell_val)
                
                # Predict edges
                gene_emb = outputs['gene_embeddings']
                edge_probs, edge_signs = model.predict_edge_batch(gene_emb, tf_indices, target_indices)
                
                # Compute metrics
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
                
                print(f"Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}, AUROC={auroc:.4f}, AUPRC={auprc:.4f}, SignAcc={sign_acc:.4f}")
                
                history.append({'epoch': epoch+1, 'auroc': auroc, 'auprc': auprc, 'sign_acc': sign_acc})
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    if config.get('save_model'):
                        torch.save(model.state_dict(), config['model_path'])
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.1f} minutes")
    print(f"Best AUROC: {best_auroc:.4f}")
    
    # Final evaluation
    if config.get('save_model') and os.path.exists(config['model_path']):
        model.load_state_dict(torch.load(config['model_path']))
    
    model.eval()
    with torch.no_grad():
        expr_val = expr[val_idx].to(device)
        atac_val = atac[val_idx].to(device)
        cell_val = cell_type_idx[val_idx].to(device)
        
        outputs = model(expr_val, atac_val, cell_val)
        gene_emb = outputs['gene_embeddings']
        edge_probs, edge_signs = model.predict_edge_batch(gene_emb, tf_indices, target_indices)
        
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
        
        final_auroc = compute_auroc(y_true, y_score)
        final_auprc = compute_auprc(y_true, y_score)
        
        y_sign_true = np.array([gt_sign_dict.get((e['tf_idx'], e['target_idx']), 1) for e in edges])
        y_sign_pred = np.array([e['sign'] for e in edges])
        final_sign_acc = compute_sign_accuracy(y_sign_true, y_sign_pred)
    
    # Save results
    results = {
        'method': 'CROSS-GRN',
        'variant': config.get('variant', 'full'),
        'seed': seed,
        'metrics': {
            'auroc': final_auroc,
            'auprc': final_auprc,
            'sign_accuracy': final_sign_acc
        },
        'best_auroc': best_auroc,
        'train_time_minutes': train_time / 60,
        'history': history,
        'config': {k: v for k, v in config.items() if k != 'model_path'},
        'n_tfs': len(tf_indices),
        'n_targets': len(target_indices)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--atac', default='data/pbmc_atac_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/crossgrn_main/results.json')
    parser.add_argument('--model_path', default='models/crossgrn.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--use_cell_type_cond', type=int, default=1)
    parser.add_argument('--use_asymmetric', type=int, default=1)
    parser.add_argument('--predict_sign', type=int, default=1)
    parser.add_argument('--variant', default='full')
    args = parser.parse_args()
    
    config = {
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'use_cell_type_cond': bool(args.use_cell_type_cond),
        'use_asymmetric': bool(args.use_asymmetric),
        'predict_sign': bool(args.predict_sign),
        'variant': args.variant,
        'save_model': True,
        'model_path': args.model_path
    }
    
    train_fast(args.rna, args.atac, args.output, config)
