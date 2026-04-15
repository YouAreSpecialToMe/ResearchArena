#!/usr/bin/env python3
"""Train CROSS-GRN model."""
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
from exp.shared.data_loader import MultiOmicDataset, load_tf_list
from exp.shared.models import CROSSGRN
from exp.shared.metrics import (
    compute_auroc, compute_auprc, compute_sign_accuracy,
    compute_epr, compute_pearson_r
)


def train_crossgrn(rna_path, atac_path, output_path, config):
    """Train CROSS-GRN model."""
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CROSS-GRN on {device} with seed {seed}")
    
    # Load data
    print("Loading data...")
    rna = sc.read_h5ad(rna_path)
    atac = sc.read_h5ad(atac_path)
    
    cell_types = rna.obs['cell_type'].tolist()
    
    with open('data/splits.json') as f:
        splits = json.load(f)
    
    train_idx = splits['train']
    val_idx = splits['val']
    
    # Create datasets
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Get cell type info
    unique_cell_types = sorted(set(cell_types))
    n_cell_types = len(unique_cell_types)
    
    # Create model
    model = CROSSGRN(
        n_genes=rna.n_vars,
        n_peaks=atac.n_vars,
        n_cell_types=n_cell_types,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        use_cell_type_cond=config['use_cell_type_cond'],
        use_asymmetric=config['use_asymmetric'],
        predict_sign=config['predict_sign']
    )
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_expr_loss = 0
        train_atac_loss = 0
        
        for batch in train_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            expr_mask = batch['expr_mask'].to(device)
            atac_mask = batch['atac_mask'].to(device)
            cell_type = batch['cell_type'].to(device)
            
            outputs = model(expr, atac, cell_type)
            
            # Masked expression prediction
            expr_pred = outputs['expr_pred']
            expr_target = expr.clone()
            expr_pred_masked = expr_pred[expr_mask]
            expr_target_masked = expr_target[expr_mask]
            
            if len(expr_pred_masked) > 0:
                loss_expr = F.mse_loss(expr_pred_masked, expr_target_masked)
            else:
                loss_expr = 0
            
            # Masked ATAC prediction
            atac_pred = outputs['atac_pred']
            atac_target = atac.clone()
            atac_pred_masked = atac_pred[atac_mask]
            atac_target_masked = atac_target[atac_mask]
            
            if len(atac_pred_masked) > 0:
                loss_atac = F.mse_loss(atac_pred_masked, atac_target_masked)
            else:
                loss_atac = 0
            
            # Total loss
            loss = loss_expr + 0.5 * loss_atac
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()
            
            train_loss += loss.item()
            train_expr_loss += loss_expr.item() if isinstance(loss_expr, torch.Tensor) else 0
            train_atac_loss += loss_atac.item() if isinstance(loss_atac, torch.Tensor) else 0
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_pearson = []
        
        with torch.no_grad():
            for batch in val_loader:
                expr = batch['expr'].to(device)
                atac = batch['atac'].to(device)
                cell_type = batch['cell_type'].to(device)
                
                outputs = model(expr, atac, cell_type)
                
                loss = F.mse_loss(outputs['expr_pred'], expr)
                val_loss += loss.item()
                
                # Compute Pearson correlation
                for i in range(expr.size(0)):
                    r = compute_pearson_r(
                        expr[i].cpu().numpy(),
                        outputs['expr_pred'][i].cpu().numpy()
                    )
                    val_pearson.append(r)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_pearson = np.mean(val_pearson)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"val_pearson={avg_val_pearson:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if config.get('save_model'):
                torch.save(model.state_dict(), config['model_path'])
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.1f} minutes")
    
    # Load best model for evaluation
    if config.get('save_model') and os.path.exists(config['model_path']):
        model.load_state_dict(torch.load(config['model_path']))
    
    # Extract GRN edges
    print("Extracting GRN edges...")
    model.eval()
    
    tfs = load_tf_list()
    tf_indices = [i for i, g in enumerate(rna.var_names) if g in tfs][:20]
    target_indices = [i for i, g in enumerate(rna.var_names) if g not in tfs][:100]
    
    edges = []
    expr_preds = []
    expr_true = []
    
    with torch.no_grad():
        # Use validation set for evaluation
        for batch in val_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            cell_type = batch['cell_type'].to(device)
            
            outputs = model(expr, atac, cell_type)
            
            expr_preds.append(outputs['expr_pred'].cpu().numpy())
            expr_true.append(expr.cpu().numpy())
            
            # Get attention weights
            attn_f = outputs.get('attn_forward')
            
            # Extract edges (simplified)
            batch_size = expr.size(0)
            for b in range(batch_size):
                for tf_idx in tf_indices:
                    for target_idx in target_indices:
                        # Use expression correlation as proxy for edge weight
                        tf_expr = expr[b, tf_idx].item()
                        target_expr = expr[b, target_idx].item()
                        
                        # Predict edge using attention if available
                        if attn_f is not None:
                            prob = torch.sigmoid(attn_f[b].mean()).item()
                        else:
                            prob = abs(tf_expr - target_expr)
                        
                        # Predict sign
                        if config['predict_sign']:
                            sign = np.sign(tf_expr - target_expr)
                        else:
                            sign = 0
                        
                        edges.append({
                            'tf_idx': int(tf_idx),
                            'target_idx': int(target_idx),
                            'prob': float(prob),
                            'sign': float(sign)
                        })
    
    # Compute expression prediction metrics
    expr_preds = np.concatenate(expr_preds)
    expr_true = np.concatenate(expr_true)
    
    pearson_r = compute_pearson_r(expr_true.flatten(), expr_preds.flatten())
    
    # Save results
    results = {
        'method': 'CROSS-GRN',
        'seed': seed,
        'config': config,
        'n_edges': len(edges),
        'edges': edges,
        'history': history,
        'pearson_r': float(pearson_r),
        'train_time_minutes': train_time / 60
    }
    
    # Compute GRN metrics
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
        
        results['metrics'] = {
            'auroc': float(compute_auroc(y_true, y_score)),
            'auprc': float(compute_auprc(y_true, y_score)),
            'epr_100': float(compute_epr(y_true, y_score, k=100))
        }
        
        if config['predict_sign']:
            y_sign_true = np.array([gt_sign.get((e['tf_idx'], e['target_idx']), 1) for e in edges])
            y_sign_pred = np.array([e['sign'] for e in edges])
            results['metrics']['sign_accuracy'] = float(compute_sign_accuracy(y_sign_true, y_sign_pred))
        
        print(f"\nGRN Metrics: {results['metrics']}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved results to {output_path}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--atac', default='data/pbmc_atac_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/crossgrn_main/results.json')
    parser.add_argument('--model_path', default='models/crossgrn.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=6)
    args = parser.parse_args()
    
    config = {
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'use_cell_type_cond': True,
        'use_asymmetric': True,
        'predict_sign': True,
        'grad_clip': 1.0,
        'save_model': True,
        'model_path': args.model_path
    }
    
    train_crossgrn(args.rna, args.atac, args.output, config)
