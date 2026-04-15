#!/usr/bin/env python3
"""Simple baseline methods (correlation, cosine similarity)."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import numpy as np
import scanpy as sc
import json
import argparse
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from exp.shared.data_loader import load_tf_list
from exp.shared.metrics import compute_auroc, compute_auprc, compute_sign_accuracy


def compute_correlation_baseline(rna, ground_truth_edges, tfs):
    """Compute correlation-based baseline."""
    gene_names = rna.var_names.tolist()
    
    available_tfs = [tf for tf in tfs if tf in gene_names][:20]
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    target_genes = [g for g in gene_names if g not in tfs][:200]
    target_indices = [gene_names.index(g) for g in target_genes]
    
    # Get expression matrix
    expr = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    
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
    
    # Compute edges
    edges = []
    for tf_idx in tf_indices:
        tf_expr = expr[:, tf_idx]
        
        for target_idx in target_indices:
            target_expr = expr[:, target_idx]
            
            # Pearson correlation
            if len(set(tf_expr)) > 1 and len(set(target_expr)) > 1:
                corr, _ = pearsonr(tf_expr, target_expr)
                prob = abs(corr)
                sign = 1 if corr > 0 else -1
            else:
                prob = 0
                sign = 1
            
            edges.append({
                'tf_idx': tf_idx,
                'target_idx': target_idx,
                'prob': prob,
                'sign': sign
            })
    
    # Compute metrics
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
        'sign_accuracy': sign_acc,
        'edges': edges
    }


def compute_cosine_baseline(rna, ground_truth_edges, tfs):
    """Compute cosine similarity baseline."""
    gene_names = rna.var_names.tolist()
    
    available_tfs = [tf for tf in tfs if tf in gene_names][:20]
    tf_indices = [gene_names.index(tf) for tf in available_tfs]
    
    target_genes = [g for g in gene_names if g not in tfs][:200]
    target_indices = [gene_names.index(g) for g in target_genes]
    
    # Get expression matrix
    expr = rna.X.toarray() if hasattr(rna.X, 'toarray') else rna.X
    
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
    
    # Compute edges
    edges = []
    for tf_idx in tf_indices:
        tf_expr = expr[:, tf_idx]
        
        for target_idx in target_indices:
            target_expr = expr[:, target_idx]
            
            # Cosine similarity
            try:
                sim = 1 - cosine(tf_expr, target_expr)
                prob = abs(sim)
                sign = 1 if sim > 0 else -1
            except:
                prob = 0
                sign = 1
            
            edges.append({
                'tf_idx': tf_idx,
                'target_idx': target_idx,
                'prob': prob,
                'sign': sign
            })
    
    # Compute metrics
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
        'sign_accuracy': sign_acc,
        'edges': edges
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--method', choices=['correlation', 'cosine'], required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    print(f"Running {args.method} baseline...")
    
    # Load data
    rna = sc.read_h5ad(args.rna)
    
    with open('data/ground_truth_edges.json') as f:
        ground_truth_edges = json.load(f)
    
    tfs = load_tf_list()
    
    # Compute baseline
    if args.method == 'correlation':
        results = compute_correlation_baseline(rna, ground_truth_edges, tfs)
    elif args.method == 'cosine':
        results = compute_cosine_baseline(rna, ground_truth_edges, tfs)
    
    print(f"Results - AUROC: {results['auroc']:.4f}, AUPRC: {results['auprc']:.4f}, "
          f"Sign Acc: {results['sign_accuracy']:.4f}")
    
    # Save results
    output = {
        'method': args.method,
        'metrics': {
            'auroc': results['auroc'],
            'auprc': results['auprc'],
            'sign_accuracy': results['sign_accuracy']
        }
    }
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved results to {args.output}")


if __name__ == '__main__':
    main()
