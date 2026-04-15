"""
Data preparation for CellStratCP experiments.
Uses scanpy built-in datasets and generates synthetic data.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import scanpy as sc
import pandas as pd
import torch
from scrna_utils import set_seed, save_results
import tarfile
import anndata as ad


def prepare_pbmc_dataset(data_dir='data', seed=42):
    """Prepare PBMC dataset using scanpy's built-in dataset."""
    print("Loading PBMC 3k dataset from scanpy...")
    
    # Load from scanpy
    adata = sc.datasets.pbmc3k()
    
    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
    # Filter cells with high mitochondrial content
    adata = adata[adata.obs.pct_counts_mt < 20, :]
    
    # Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Select HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    
    # Annotate cell types using marker genes
    markers = {
        'CD4 T': ['IL7R'],
        'CD8 T': ['CD8A'],
        'B cell': ['CD79A', 'MS4A1'],
        'NK': ['GNLY', 'NKG7'],
        'Monocyte': ['LYZ', 'CD14'],
    }
    
    # Simple annotation based on marker expression
    cell_types = []
    for idx in range(adata.n_obs):
        scores = {}
        for ct, genes in markers.items():
            available_genes = [g for g in genes if g in adata.var_names]
            if len(available_genes) > 0:
                expr = adata[idx, available_genes].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray()
                scores[ct] = np.mean(expr)
        
        if scores and max(scores.values()) > 0:
            cell_type = max(scores, key=scores.get)
        else:
            cell_type = 'Unknown'
        cell_types.append(cell_type)
    
    adata.obs['cell_type'] = cell_types
    
    # Remove Unknown cells
    adata = adata[adata.obs['cell_type'] != 'Unknown']
    
    # Create splits
    np.random.seed(seed)
    cell_types_arr = adata.obs['cell_type'].values
    unique_types = np.unique(cell_types_arr)
    
    train_idx, cal_idx, test_idx = [], [], []
    for ct in unique_types:
        ct_mask = cell_types_arr == ct
        ct_indices = np.where(ct_mask)[0]
        np.random.shuffle(ct_indices)
        
        n = len(ct_indices)
        n_train = int(n * 0.6)
        n_cal = int(n * 0.2)
        
        train_idx.extend(ct_indices[:n_train])
        cal_idx.extend(ct_indices[n_train:n_train+n_cal])
        test_idx.extend(ct_indices[n_train+n_cal:])
    
    adata.obs['split'] = 'train'
    adata.obs.iloc[cal_idx, adata.obs.columns.get_loc('split')] = 'calibration'
    adata.obs.iloc[test_idx, adata.obs.columns.get_loc('split')] = 'test'
    
    return adata


def generate_synthetic_data(n_cells=3000, n_genes=1000, n_cell_types=6, 
                            dropout_rate=0.5, seed=42):
    """Generate synthetic scRNA-seq data with known ground truth."""
    set_seed(seed)
    
    # Generate cell type assignments
    cell_type_probs = np.random.dirichlet(np.ones(n_cell_types) * 0.5)
    cell_types = np.random.choice(n_cell_types, size=n_cells, p=cell_type_probs)
    
    # Generate gene expression parameters per cell type
    gene_means = np.exp(np.random.randn(n_cell_types, n_genes) * 0.5 + 0.5)
    
    # Generate true expression (before dropout)
    true_expression = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        ct = cell_types[i]
        mu = gene_means[ct]
        theta = np.random.gamma(2, 2, n_genes)
        
        # Sample from NB
        p = theta / (theta + mu)
        counts = np.random.negative_binomial(theta, p)
        true_expression[i] = counts
    
    # Apply dropout
    dropout_mask = np.random.random((n_cells, n_genes)) < dropout_rate
    observed_expression = true_expression.copy()
    observed_expression[dropout_mask] = 0
    
    # Create AnnData
    adata = ad.AnnData(X=observed_expression)
    adata.obs['cell_type'] = [f'CT_{ct}' for ct in cell_types]
    adata.layers['true'] = true_expression
    
    # Create splits
    np.random.seed(seed)
    cell_types_arr = adata.obs['cell_type'].values
    unique_types = np.unique(cell_types_arr)
    
    train_idx, cal_idx, test_idx = [], [], []
    for ct in unique_types:
        ct_mask = cell_types_arr == ct
        ct_indices = np.where(ct_mask)[0]
        np.random.shuffle(ct_indices)
        
        n = len(ct_indices)
        n_train = int(n * 0.6)
        n_cal = int(n * 0.2)
        
        train_idx.extend(ct_indices[:n_train])
        cal_idx.extend(ct_indices[n_train:n_train+n_cal])
        test_idx.extend(ct_indices[n_train+n_cal:])
    
    split_array = np.array(['train'] * n_cells)
    split_array[cal_idx] = 'calibration'
    split_array[test_idx] = 'test'
    adata.obs['split'] = split_array
    
    return adata


def prepare_all_datasets(seed=42):
    """Prepare all datasets for experiments."""
    print("="*60)
    print("PREPARING DATASETS FOR CELLSTRATCP")
    print("="*60)
    
    os.makedirs('data', exist_ok=True)
    results = {'status': 'success', 'datasets': {}}
    
    # 1. PBMC dataset
    print("\n--- Preparing PBMC dataset ---")
    try:
        adata_pbmc = prepare_pbmc_dataset(seed=seed)
        output_path = 'data/pbmc_processed.h5ad'
        adata_pbmc.write(output_path)
        print(f"Saved to {output_path}")
        print(f"  Cells: {adata_pbmc.n_obs}, Genes: {adata_pbmc.n_var}")
        print(f"  Cell types: {adata_pbmc.obs['cell_type'].value_counts().to_dict()}")
        
        results['datasets']['pbmc'] = {
            'n_cells': int(adata_pbmc.n_obs),
            'n_genes': int(adata_pbmc.n_var),
            'n_cell_types': int(adata_pbmc.obs['cell_type'].nunique()),
            'path': output_path
        }
    except Exception as e:
        print(f"PBMC preparation failed: {e}")
        import traceback
        traceback.print_exc()
        results['datasets']['pbmc'] = {'error': str(e)}
    
    # 2. Synthetic datasets
    print("\n--- Generating synthetic datasets ---")
    for dropout_rate in [0.3, 0.5, 0.7]:
        for seed_val in [42, 123, 456]:
            try:
                adata_syn = generate_synthetic_data(
                    n_cells=2000, n_genes=800, n_cell_types=6,
                    dropout_rate=dropout_rate, seed=seed_val
                )
                
                output_path = f'data/synthetic_d{int(dropout_rate*100)}_s{seed_val}.h5ad'
                adata_syn.write(output_path)
                
                key = f'synthetic_d{int(dropout_rate*100)}_s{seed_val}'
                results['datasets'][key] = {
                    'n_cells': int(adata_syn.n_obs),
                    'n_genes': int(adata_syn.n_vars),
                    'n_cell_types': int(adata_syn.obs['cell_type'].nunique()),
                    'path': output_path
                }
                
                if seed_val == 42:
                    print(f"Dropout {dropout_rate}: {adata_syn.n_obs} cells, "
                          f"{adata_syn.obs['cell_type'].nunique()} types")
            except Exception as e:
                print(f"Synthetic data generation failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    save_results(results, 'exp/data_prep/results.json')
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    prepare_all_datasets(seed=args.seed)
