"""Data loading and preprocessing utilities for CROSS-GRN."""
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os


def load_and_preprocess_pbmc(data_path='data/pbmc_10k_filtered.h5', output_dir='data'):
    """Load and preprocess 10x PBMC Multiome data."""
    print("Loading PBMC data...")
    adata = sc.read_10x_h5(data_path, gex_only=False)
    
    # Split RNA and ATAC
    rna = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
    atac = adata[:, adata.var['feature_types'] == 'Peaks'].copy()
    
    print(f"Original shapes - RNA: {rna.shape}, ATAC: {atac.shape}")
    
    # Filter genes and peaks
    sc.pp.filter_genes(rna, min_cells=50)
    sc.pp.filter_genes(atac, min_cells=20)
    
    # Normalize RNA
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    
    # Simple normalization for ATAC
    sc.pp.normalize_total(atac, target_sum=1e4)
    
    # Select highly variable genes
    sc.pp.highly_variable_genes(rna, n_top_genes=3000, subset=True)
    
    # Subset ATAC to manageable size
    peak_counts = np.array(atac.X.sum(axis=0)).flatten()
    top_peaks_idx = np.argsort(peak_counts)[-20000:]
    atac = atac[:, top_peaks_idx].copy()
    
    print(f"After filtering - RNA: {rna.shape}, ATAC: {atac.shape}")
    
    # Cell type annotation
    cell_types = annotate_cell_types(rna)
    rna.obs['cell_type'] = cell_types
    atac.obs['cell_type'] = cell_types
    
    # Save preprocessed data
    rna.write_h5ad(os.path.join(output_dir, 'pbmc_rna_preprocessed.h5ad'))
    atac.write_h5ad(os.path.join(output_dir, 'pbmc_atac_preprocessed.h5ad'))
    
    # Save cell types
    cell_type_df = pd.DataFrame({
        'cell_barcode': rna.obs_names,
        'cell_type': cell_types
    })
    cell_type_df.to_csv(os.path.join(output_dir, 'pbmc_cell_types.csv'), index=False)
    
    return rna, atac


def annotate_cell_types(rna):
    """Annotate cell types using marker genes."""
    markers = {
        'CD4_T': ['IL7R', 'CD3D', 'CD3E'],
        'CD8_T': ['CD8A', 'CD8B', 'CD3D'],
        'B_cell': ['CD79A', 'CD79B', 'MS4A1'],
        'Monocyte': ['CD14', 'LYZ', 'S100A9'],
        'NK': ['NKG7', 'GNLY', 'KLRD1'],
        'DC': ['FCER1A', 'CST3', 'CLEC10A'],
    }
    
    scores = np.zeros((rna.n_obs, len(markers)))
    marker_names = list(markers.keys())
    
    for i, (cell_type, genes) in enumerate(markers.items()):
        available_genes = [g for g in genes if g in rna.var_names]
        if available_genes:
            expr = rna[:, available_genes].X.toarray().mean(axis=1)
            scores[:, i] = expr.flatten()
    
    cell_types = [marker_names[i] if scores[j, i] > 0.1 else 'Other' 
                  for j, i in enumerate(np.argmax(scores, axis=1))]
    
    return cell_types


def load_tf_list():
    """Load list of transcription factors."""
    tfs = [
        'SPI1', 'CEBPA', 'CEBPB', 'GATA1', 'GATA2', 'GATA3', 'TAL1', 'MYC',
        'RUNX1', 'RUNX2', 'STAT1', 'STAT3', 'IRF4', 'IRF8', 'BATF',
        'EBF1', 'PAX5', 'TCF3', 'FOXP3', 'TBX21', 'RORC', 'GATA6', 'NFKB1',
        'RELA', 'JUN', 'FOS', 'ETS1', 'FLI1', 'ERG', 'MYB', 'IKZF1',
        'TCF7', 'LEF1', 'BCL11B', 'GATA4', 'GATA5', 'MEF2C', 'SRF', 'ELK1',
        'ATF4', 'CREB1', 'ATF2', 'JUNB', 'FOSB', 'MAF', 'NFE2', 'YY1'
    ]
    return tfs


def create_train_val_test_split(n_cells, train_frac=0.7, val_frac=0.15, seed=42):
    """Create train/val/test split indices."""
    np.random.seed(seed)
    indices = np.arange(n_cells)
    np.random.shuffle(indices)
    
    n_train = int(n_cells * train_frac)
    n_val = int(n_cells * val_frac)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    return train_idx, val_idx, test_idx


class MultiOmicDataset(Dataset):
    """Dataset for single-cell multi-omics."""
    
    def __init__(self, rna, atac, cell_types, mode='train', mask_ratio=0.15):
        self.rna = rna
        self.atac = atac
        self.cell_types = cell_types
        self.mode = mode
        self.mask_ratio = mask_ratio
        
        self.cell_type_to_idx = {ct: i for i, ct in enumerate(sorted(set(cell_types)))}
        
    def __len__(self):
        return self.rna.n_obs
    
    def __getitem__(self, idx):
        expr = torch.FloatTensor(self.rna[idx].X.toarray().flatten())
        atac = torch.FloatTensor(self.atac[idx].X.toarray().flatten())
        
        cell_type = self.cell_types[idx]
        cell_type_idx = self.cell_type_to_idx.get(cell_type, 0)
        
        if self.mode == 'train':
            expr_mask = torch.rand(len(expr)) < self.mask_ratio
            atac_mask = torch.rand(len(atac)) < self.mask_ratio
        else:
            expr_mask = torch.zeros(len(expr), dtype=torch.bool)
            atac_mask = torch.zeros(len(atac), dtype=torch.bool)
        
        return {
            'expr': expr,
            'atac': atac,
            'expr_mask': expr_mask,
            'atac_mask': atac_mask,
            'cell_type': cell_type_idx,
            'cell_barcode': self.rna.obs_names[idx]
        }


def create_synthetic_ground_truth(tfs, genes, output_dir='data'):
    """Create synthetic ground truth for evaluation."""
    np.random.seed(42)
    
    edges = []
    for tf in tfs:
        n_targets = np.random.randint(20, 80)
        for _ in range(n_targets):
            target = np.random.choice(genes)
            sign = np.random.choice([1, -1], p=[0.6, 0.4])
            edges.append({
                'tf': tf,
                'target': target,
                'sign': int(sign),
                'confidence': float(np.random.uniform(0.5, 1.0))
            })
    
    with open(os.path.join(output_dir, 'ground_truth_edges.json'), 'w') as f:
        json.dump(edges, f)
    
    return edges


def create_chipseq_labels(tfs, genes, output_dir='data'):
    """Create synthetic ChIP-seq labels."""
    np.random.seed(42)
    
    labels = {}
    for tf in tfs:
        n_pos = np.random.randint(50, 150)
        pos_targets = np.random.choice(genes, n_pos, replace=False)
        
        tf_labels = {gene: int(gene in pos_targets) for gene in genes}
        labels[tf] = tf_labels
    
    with open(os.path.join(output_dir, 'chipseq_labels.json'), 'w') as f:
        json.dump(labels, f)
    
    return labels
