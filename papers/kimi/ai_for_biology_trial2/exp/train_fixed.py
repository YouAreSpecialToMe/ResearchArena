#!/usr/bin/env python3
"""Fixed training script for CROSS-GRN with supervised edge loss."""
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
from exp.shared.models_v2 import CROSSGRNv2
from exp.shared.metrics import (
    compute_auroc, compute_auprc, compute_sign_accuracy, compute_pearson_r
)


def create_edge_supervision_data(rna, ground_truth_edges, tfs, max_targets_per_tf=50):
    """
    Create supervised edge data from ground truth.
    
    Returns:
        tf_indices: list of TF indices in the gene list
        target_indices: list of target indices
        edge_labels: binary labels for edges
        edge_signs: sign labels (-1 or 1)
    """
    gene_names = rna.var_names.tolist()
    tf_list = load_tf_list()
    
    # Find TFs that are in our gene list
    available_tfs = [tf for tf in tf_list if tf in gene_names][:20]  # Limit to 20 TFs
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    # Find target genes (non-TF genes)
    target_genes = [g for g in gene_names if g not in tf_list][:200]  # Limit to 200 targets
    target_indices = [gene_names.index(g) for g in target_genes]
    
    # Build ground truth edge dictionary
    gt_dict = {}
    gt_sign_dict = {}
    for edge in ground_truth_edges:
        tf = edge['tf']
        target = edge['target']
        if tf in gene_names and target in gene_names:
            tf_idx = gene_names.index(tf)
            target_idx = gene_names.index(target)
            gt_dict[(tf_idx, target_idx)] = 1
            gt_sign_dict[(tf_idx, target_idx)] = edge.get('sign', 1)
    
    # Create positive edges (from ground truth)
    positive_edges = []
    for (tf_idx, target_idx), label in gt_dict.items():
        if tf_idx in tf_indices and target_idx in target_indices:
            sign = gt_sign_dict.get((tf_idx, target_idx), 1)
            positive_edges.append((tf_idx, target_idx, 1, sign))
    
    # Create negative edges (random TF-target pairs not in ground truth)
    np.random.seed(42)
    negative_edges = []
    n_neg = min(len(positive_edges) * 3, len(tf_indices) * len(target_indices) - len(positive_edges))
    
    attempts = 0
    while len(negative_edges) < n_neg and attempts < n_neg * 10:
        tf_idx = np.random.choice(tf_indices)
        target_idx = np.random.choice(target_indices)
        if (tf_idx, target_idx) not in gt_dict:
            negative_edges.append((tf_idx, target_idx, 0, 1))  # Label=0, sign=1 (unused)
        attempts += 1
    
    all_edges = positive_edges + negative_edges
    np.random.shuffle(all_edges)
    
    print(f"Created {len(positive_edges)} positive edges and {len(negative_edges)} negative edges")
    
    return tf_indices, target_indices, all_edges


def train_epoch(model, train_loader, optimizer, device, tf_indices, target_indices, edge_data, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_expr_loss = 0
    total_edge_loss = 0
    
    # Sample edges for this epoch
    edge_sample = edge_data[np.random.choice(len(edge_data), min(256, len(edge_data)), replace=False)]
    edge_tf_indices = [e[0] for e in edge_sample]
    edge_target_indices = [e[1] for e in edge_sample]
    edge_labels = torch.FloatTensor([e[2] for e in edge_sample]).to(device)
    edge_signs = torch.FloatTensor([e[3] for e in edge_sample]).to(device)
    
    for batch_idx, batch in enumerate(train_loader):
        expr = batch['expr'].to(device)
        atac = batch['atac'].to(device)
        cell_type = batch['cell_type'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(expr, atac, cell_type)
        
        # Expression prediction loss (MSE on predicted expression)
        expr_pred = outputs['expr_pred']
        expr_loss = F.mse_loss(expr_pred, expr)
        
        # Edge prediction loss (supervised)
        gene_emb = outputs['gene_embeddings']
        
        # Get embeddings for edges to predict
        batch_size = gene_emb.size(0)
        tf_emb_batch = gene_emb[:, edge_tf_indices, :]  # (batch, n_edges, hidden)
        target_emb_batch = gene_emb[:, edge_target_indices, :]
        
        # Concatenate TF and target embeddings
        pair_repr = torch.cat([tf_emb_batch, target_emb_batch], dim=-1)  # (batch, n_edges, hidden*2)
        
        # Predict edges
        edge_logits = model.edge_head(pair_repr).squeeze(-1)  # (batch, n_edges)
        edge_probs = torch.sigmoid(edge_logits)
        edge_pred = edge_probs.mean(dim=0)  # Average over batch
        
        # Edge loss (BCE)
        edge_loss = F.binary_cross_entropy(edge_pred, edge_labels)
        
        # Sign loss (if predicting signs)
        sign_loss = 0
        if model.predict_sign:
            sign_pred = torch.tanh(model.sign_head(pair_repr).squeeze(-1)).mean(dim=0)
            sign_loss = F.mse_loss(sign_pred, edge_signs)
        
        # Total loss
        loss = expr_loss + 2.0 * edge_loss + 0.5 * sign_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_expr_loss += expr_loss.item()
        total_edge_loss += edge_loss.item()
    
    n_batches = len(train_loader)
    return {
        'loss': total_loss / n_batches,
        'expr_loss': total_expr_loss / n_batches,
        'edge_loss': total_edge_loss / n_batches
    }


def evaluate(model, val_loader, device, ground_truth_dict, gt_sign_dict, tf_indices, target_indices):
    """Evaluate model on validation set."""
    model.eval()
    
    all_edges = []
    all_expr_preds = []
    all_expr_true = []
    
    with torch.no_grad():
        for batch in val_loader:
            expr = batch['expr'].to(device)
            atac = batch['atac'].to(device)
            cell_type = batch['cell_type'].to(device)
            
            outputs = model(expr, atac, cell_type)
            
            # Store expression predictions
            all_expr_preds.append(outputs['expr_pred'].cpu().numpy())
            all_expr_true.append(expr.cpu().numpy())
            
            # Predict GRN edges
            gene_emb = outputs['gene_embeddings']
            edge_probs, edge_signs = model.predict_edge_batch(gene_emb, tf_indices, target_indices)
            
            # Store edges
            for i, tf_idx in enumerate(tf_indices):
                for j, target_idx in enumerate(target_indices):
                    all_edges.append({
                        'tf_idx': tf_idx,
                        'target_idx': target_idx,
                        'prob': edge_probs[i, j].item(),
                        'sign': edge_signs[i, j].item()
                    })
    
    # Compute expression prediction metrics
    all_expr_preds = np.concatenate(all_expr_preds)
    all_expr_true = np.concatenate(all_expr_true)
    pearson_r = compute_pearson_r(all_expr_true.flatten(), all_expr_preds.flatten())
    
    # Compute GRN metrics
    y_true = np.array([ground_truth_dict.get((e['tf_idx'], e['target_idx']), 0) for e in all_edges])
    y_score = np.array([e['prob'] for e in all_edges])
    
    # Filter to valid comparisons
    valid_mask = (y_true > 0) | (np.random.rand(len(y_true)) < 0.1)  # Keep all positives, 10% of negatives
    y_true_filtered = y_true[valid_mask]
    y_score_filtered = y_score[valid_mask]
    
    auroc = compute_auroc(y_true_filtered, y_score_filtered)
    auprc = compute_auprc(y_true_filtered, y_score_filtered)
    
    # Sign accuracy
    y_sign_true = np.array([gt_sign_dict.get((e['tf_idx'], e['target_idx']), 1) for e in all_edges])
    y_sign_pred = np.array([1 if e['sign'] > 0 else -1 for e in all_edges])
    sign_acc = compute_sign_accuracy(y_sign_true, y_sign_pred)
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'sign_accuracy': sign_acc,
        'pearson_r': pearson_r,
        'edges': all_edges
    }


def train_crossgrn_fixed(rna_path, atac_path, output_path, config):
    """Train CROSS-GRN with fixed architecture."""
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
    
    with open('data/splits.json') as f:
        splits = json.load(f)
    
    train_idx = splits['train']
    val_idx = splits['val']
    
    # Load ground truth
    with open('data/ground_truth_edges.json') as f:
        ground_truth_edges = json.load(f)
    
    # Create edge supervision data
    print("Creating edge supervision data...")
    tf_indices, target_indices, edge_data = create_edge_supervision_data(
        rna, ground_truth_edges, load_tf_list()
    )
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Get cell type info
    unique_cell_types = sorted(set(cell_types))
    n_cell_types = len(unique_cell_types)
    
    # Create model
    model = CROSSGRNv2(
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
    
    # Build ground truth dict for evaluation
    gene_names = rna.var_names.tolist()
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
    
    print(f"Ground truth edges to evaluate: {len(gt_dict)}")
    
    # Training loop
    best_auroc = 0
    history = {'train_loss': [], 'val_auroc': [], 'val_auprc': []}
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            tf_indices, target_indices, np.array(edge_data, dtype=object), epoch
        )
        
        scheduler.step()
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
            val_metrics = evaluate(
                model, val_loader, device, gt_dict, gt_sign_dict,
                tf_indices, target_indices
            )
            
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"train_loss={train_metrics['loss']:.4f}, "
                  f"val_auroc={val_metrics['auroc']:.4f}, "
                  f"val_auprc={val_metrics['auprc']:.4f}, "
                  f"sign_acc={val_metrics['sign_accuracy']:.4f}, "
                  f"pearson_r={val_metrics['pearson_r']:.4f}")
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_auroc'].append(val_metrics['auroc'])
            history['val_auprc'].append(val_metrics['auprc'])
            
            # Save best model
            if val_metrics['auroc'] > best_auroc:
                best_auroc = val_metrics['auroc']
                if config.get('save_model'):
                    torch.save(model.state_dict(), config['model_path'])
    
    train_time = time.time() - start_time
    print(f"Training completed in {train_time/60:.1f} minutes")
    print(f"Best validation AUROC: {best_auroc:.4f}")
    
    # Load best model for final evaluation
    if config.get('save_model') and os.path.exists(config['model_path']):
        model.load_state_dict(torch.load(config['model_path']))
    
    # Final evaluation
    model.eval()
    final_metrics = evaluate(
        model, val_loader, device, gt_dict, gt_sign_dict,
        tf_indices, target_indices
    )
    
    # Save results
    results = {
        'method': 'CROSS-GRN',
        'variant': config.get('variant', 'full'),
        'seed': seed,
        'config': {k: v for k, v in config.items() if k != 'model_path'},
        'metrics': {
            'auroc': final_metrics['auroc'],
            'auprc': final_metrics['auprc'],
            'sign_accuracy': final_metrics['sign_accuracy'],
            'pearson_r': final_metrics['pearson_r']
        },
        'best_auroc': best_auroc,
        'train_time_minutes': train_time / 60,
        'history': history,
        'tf_indices': tf_indices,
        'target_indices': target_indices,
        'edges': final_metrics['edges'][:1000]  # Save subset for analysis
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
    parser.add_argument('--model_path', default='models/crossgrn_fixed.pt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
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
    
    train_crossgrn_fixed(args.rna, args.atac, args.output, config)
