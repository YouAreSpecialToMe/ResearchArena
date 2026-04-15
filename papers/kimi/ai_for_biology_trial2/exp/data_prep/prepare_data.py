"""
Data preparation script for Tri-Con experiments.
Downloads and preprocesses PBMC dataset, creates splits.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/exp')

import numpy as np
import scanpy as sc
import json
from pathlib import Path
from shared.data_loader import (
    preprocess_data, create_train_val_test_split, 
    create_zero_shot_split, create_ood_split
)
from shared.utils import set_seed


def main():
    print("=" * 60)
    print("Data Preparation for Tri-Con Experiments")
    print("=" * 60)
    
    output_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Step 1: Download PBMC dataset
    print("\n[1/4] Loading PBMC dataset...")
    adata = sc.datasets.pbmc3k_processed()
    print(f"  Raw data shape: {adata.shape}")
    
    # Get raw counts
    if hasattr(adata, 'raw') and adata.raw is not None:
        adata = adata.raw.to_adata()
    
    # Set cell type labels from louvain clustering
    if 'louvain' in adata.obs.columns:
        adata.obs['cell_type'] = adata.obs['louvain'].astype(str)
    
    # Print cell type info
    cell_types = adata.obs['cell_type'].unique()
    print(f"  Cell types: {list(cell_types)}")
    print(f"  Number of cells: {adata.n_obs}")
    print(f"  Number of genes: {adata.n_vars}")
    
    # Step 2: Preprocess data
    print("\n[2/4] Preprocessing data...")
    adata = preprocess_data(adata, n_top_genes=2000, normalize=True)
    print(f"  After preprocessing shape: {adata.shape}")
    
    # Step 3: Create splits for each seed
    print("\n[3/4] Creating train/val/test splits...")
    seeds = [42, 123, 456]
    
    for seed in seeds:
        print(f"  Creating splits for seed {seed}...")
        splits = create_train_val_test_split(adata, test_size=0.3, val_size=0.5, random_state=seed)
        
        # Save splits
        splits_file = output_dir / f'pbmc3k_splits_seed{seed}.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"    Saved splits to {splits_file}")
    
    # Step 4: Create zero-shot splits
    print("\n[4/4] Creating zero-shot splits...")
    zero_shot_info = {}
    
    for seed in seeds:
        set_seed(seed)
        zero_shot_types = create_zero_shot_split(adata, held_out_ratio=0.2, random_state=seed)
        
        # Split cells into seen/unseen
        is_zero_shot = adata.obs['cell_type'].isin(zero_shot_types)
        
        zero_shot_info[f'seed_{seed}'] = {
            'held_out_types': zero_shot_types,
            'n_held_out_cells': int(is_zero_shot.sum()),
            'n_seen_cells': int((~is_zero_shot).sum())
        }
        
        print(f"  Seed {seed}:")
        print(f"    Held-out types: {zero_shot_types}")
        print(f"    Held-out cells: {is_zero_shot.sum()}")
        print(f"    Seen cells: {(~is_zero_shot).sum()}")
    
    # Save zero-shot info
    zero_shot_file = output_dir / 'zero_shot_splits.json'
    with open(zero_shot_file, 'w') as f:
        json.dump(zero_shot_info, f, indent=2)
    print(f"\n  Saved zero-shot splits to {zero_shot_file}")
    
    # Step 5: Create OOD splits for novelty detection
    print("\n[5/5] Creating OOD splits for novelty detection...")
    
    # Define ID and OOD cell types
    # ID: Common cell types, OOD: Rare cell types
    id_cell_types = ['CD4 T', 'CD8 T', 'B', 'NK']  # Common immune cells
    ood_cell_types = ['Dendritic', 'Megakaryocytes', 'FCGR3A+ Monocytes']  # Rare/populations
    
    # Filter to existing types
    existing_types = set(adata.obs['cell_type'].unique())
    id_cell_types = [t for t in id_cell_types if t in existing_types]
    ood_cell_types = [t for t in ood_cell_types if t in existing_types]
    
    # If not enough types, create synthetic split
    if len(ood_cell_types) < 1:
        print("  Using random split for OOD detection...")
        # Use one cell type as OOD
        ood_cell_types = [cell_types[-1]]
        id_cell_types = list(cell_types[:-1])
    
    ood_split = create_ood_split(adata, id_cell_types, ood_cell_types)
    
    ood_info = {
        'id_cell_types': id_cell_types,
        'ood_cell_types': ood_cell_types,
        'n_id_cells': int(ood_split['id_mask'].sum()),
        'n_ood_cells': int(ood_split['ood_mask'].sum())
    }
    
    ood_file = output_dir / 'ood_splits.json'
    with open(ood_file, 'w') as f:
        json.dump(ood_info, f, indent=2)
    
    print(f"  ID cell types: {id_cell_types}")
    print(f"  OOD cell types: {ood_cell_types}")
    print(f"  ID cells: {ood_info['n_id_cells']}")
    print(f"  OOD cells: {ood_info['n_ood_cells']}")
    print(f"  Saved OOD splits to {ood_file}")
    
    # Step 6: Save processed data
    print("\n[6/6] Saving processed data...")
    adata_file = output_dir / 'pbmc3k_processed.h5ad'
    adata.write_h5ad(adata_file)
    print(f"  Saved processed data to {adata_file}")
    
    # Create dataset info
    dataset_info = {
        'n_cells': adata.n_obs,
        'n_genes': adata.n_vars,
        'n_cell_types': len(cell_types),
        'cell_types': list(cell_types),
        'file': str(adata_file),
        'description': 'PBMC 3k dataset from 10x Genomics'
    }
    
    info_file = output_dir / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nDataset summary:")
    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Cell types: {len(cell_types)}")
    print(f"  Output directory: {output_dir}")


if __name__ == '__main__':
    main()
