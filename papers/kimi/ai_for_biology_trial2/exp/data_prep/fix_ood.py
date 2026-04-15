"""
Fix OOD split to use correct cell type names.
"""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/exp')

import numpy as np
import scanpy as sc
import json
from pathlib import Path
from shared.data_loader import create_ood_split


def main():
    output_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/data')
    
    # Load processed data
    adata = sc.read_h5ad(output_dir / 'pbmc3k_processed.h5ad')
    cell_types = adata.obs['cell_type'].unique().tolist()
    print(f"Available cell types: {cell_types}")
    
    # Define ID and OOD cell types using exact names
    # ID: Common cell types, OOD: Rare cell types
    id_cell_types = ['CD4 T cells', 'CD8 T cells', 'B cells', 'NK cells']
    ood_cell_types = ['Dendritic cells', 'Megakaryocytes']
    
    # Filter to existing types
    existing_types = set(cell_types)
    id_cell_types = [t for t in id_cell_types if t in existing_types]
    ood_cell_types = [t for t in ood_cell_types if t in existing_types]
    
    print(f"\nID cell types: {id_cell_types}")
    print(f"OOD cell types: {ood_cell_types}")
    
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
    
    print(f"\nID cells: {ood_info['n_id_cells']}")
    print(f"OOD cells: {ood_info['n_ood_cells']}")
    print(f"Saved OOD splits to {ood_file}")


if __name__ == '__main__':
    main()
