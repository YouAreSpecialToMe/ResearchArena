#!/usr/bin/env python3
"""GENIE3 baseline for GRN inference."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

import numpy as np
import pandas as pd
import scanpy as sc
import json
import os
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import argparse
from exp.shared.data_loader import load_tf_list
from exp.shared.metrics import compute_auroc, compute_auprc, compute_epr


def run_genie3(rna_path, output_path, n_trees=100, max_depth=10):
    """Run GENIE3 on RNA data."""
    print("Loading RNA data...")
    rna = sc.read_h5ad(rna_path)
    
    # Get expression matrix
    expr = rna.X.toarray()
    gene_names = rna.var_names.tolist()
    
    # Get TFs
    tfs = load_tf_list()
    tf_indices = [i for i, g in enumerate(gene_names) if g in tfs]
    target_indices = [i for i, g in enumerate(gene_names) if g not in tfs]
    
    print(f"TFs: {len(tf_indices)}, Targets: {len(target_indices)}")
    
    # Run GENIE3 for each target gene
    edges = []
    n_targets = min(len(target_indices), 500)  # Limit for speed
    
    print(f"Running GENIE3 for {n_targets} target genes...")
    
    for idx, target_idx in enumerate(target_indices[:n_targets]):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{n_targets}")
        
        target_expr = expr[:, target_idx]
        tf_expr = expr[:, tf_indices]
        
        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            n_jobs=4,
            random_state=42
        )
        rf.fit(tf_expr, target_expr)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Compute correlations for sign
        for tf_idx, imp in zip(tf_indices, importances):
            corr, _ = pearsonr(expr[:, tf_idx], target_expr)
            sign = 1 if corr > 0 else -1
            
            edges.append({
                'tf_idx': int(tf_idx),
                'target_idx': int(target_idx),
                'prob': float(imp),
                'sign': int(sign),
                'importance': float(imp)
            })
    
    # Save results
    results = {
        'method': 'GENIE3',
        'n_edges': len(edges),
        'edges': edges
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f)
    
    print(f"Saved {len(edges)} edges to {output_path}")
    
    # Compute metrics if ground truth exists
    if os.path.exists('data/ground_truth_edges.json'):
        with open('data/ground_truth_edges.json') as f:
            gt = json.load(f)
        
        # Create ground truth dict
        gt_dict = {}
        for e in gt:
            key = (gene_names.index(e['tf']), gene_names.index(e['target']))
            gt_dict[key] = 1
        
        y_true = np.array([gt_dict.get((e['tf_idx'], e['target_idx']), 0) for e in edges])
        y_score = np.array([e['prob'] for e in edges])
        
        auroc = compute_auroc(y_true, y_score)
        auprc = compute_auprc(y_true, y_score)
        epr = compute_epr(y_true, y_score, k=100)
        
        print(f"\nMetrics: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, EPR@100={epr:.4f}")
        
        results['metrics'] = {
            'auroc': float(auroc),
            'auprc': float(auprc),
            'epr_100': float(epr)
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna', default='data/pbmc_rna_preprocessed.h5ad')
    parser.add_argument('--output', default='exp/genie3_baseline/results.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    results = run_genie3(args.rna, args.output)
    print("GENIE3 baseline complete!")
