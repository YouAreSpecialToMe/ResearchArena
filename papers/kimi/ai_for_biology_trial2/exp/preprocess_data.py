#!/usr/bin/env python3
"""Preprocess PBMC data for CROSS-GRN experiments."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

from exp.shared.data_loader import (
    load_and_preprocess_pbmc, load_tf_list, 
    create_train_val_test_split, create_synthetic_ground_truth,
    create_chipseq_labels
)
import json
import os

print("Starting data preprocessing...")

# Load and preprocess PBMC data
rna, atac = load_and_preprocess_pbmc(
    data_path='data/pbmc_10k_filtered.h5',
    output_dir='data'
)

# Get cell types
cell_types = rna.obs['cell_type'].tolist()

# Create train/val/test splits
train_idx, val_idx, test_idx = create_train_val_test_split(
    rna.n_obs, train_frac=0.7, val_frac=0.15, seed=42
)

splits = {
    'train': train_idx.tolist(),
    'val': val_idx.tolist(),
    'test': test_idx.tolist()
}

with open('data/splits.json', 'w') as f:
    json.dump(splits, f)

print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# Create ground truth and ChIP-seq labels
tfs = load_tf_list()
genes = rna.var_names.tolist()

# Filter TFs present in data
tfs_in_data = [tf for tf in tfs if tf in genes]
print(f"TFs in data: {len(tfs_in_data)}/{len(tfs)}")

# Create synthetic ground truth
ground_truth = create_synthetic_ground_truth(tfs_in_data, genes, output_dir='data')
print(f"Created {len(ground_truth)} ground truth edges")

# Create ChIP-seq labels
chipseq = create_chipseq_labels(tfs_in_data, genes, output_dir='data')
print(f"Created ChIP-seq labels for {len(chipseq)} TFs")

# Save metadata
metadata = {
    'n_cells': rna.n_obs,
    'n_genes': rna.n_vars,
    'n_peaks': atac.n_vars,
    'n_cell_types': len(set(cell_types)),
    'cell_types': sorted(set(cell_types)),
    'n_tfs': len(tfs_in_data),
    'tfs': tfs_in_data
}

with open('data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Preprocessing complete!")
print(f"Metadata: {metadata}")
