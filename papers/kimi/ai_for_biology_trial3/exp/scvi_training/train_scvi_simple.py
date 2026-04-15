"""
Simplified scVI training for CellStratCP experiments.
Trains models with fewer epochs for faster execution.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import scanpy as sc
import scvi
import torch
import pandas as pd
from scrna_utils import set_seed, save_results
import time
import glob


def train_and_extract(adata_path, seed=42, max_epochs=30):
    """Train scVI and extract parameters."""
    set_seed(seed)
    
    dataset_name = os.path.basename(adata_path).replace('.h5ad', '')
    print(f"\nProcessing {dataset_name}")
    
    # Load data
    adata = sc.read_h5ad(adata_path)
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    
    # Setup and train
    scvi.model.SCVI.setup_anndata(adata)
    
    model = scvi.model.SCVI(
        adata,
        n_layers=2,
        n_latent=10,
        dropout_rate=0.1,
        gene_likelihood="zinb"
    )
    
    start_time = time.time()
    model.train(max_epochs=max_epochs, early_stopping=True)
    train_time = time.time() - start_time
    
    print(f"  Training time: {train_time:.1f}s")
    
    # Extract parameters
    params = model.get_likelihood_parameters(adata=adata, give_mean=True)
    predictions = model.get_normalized_expression(adata, return_mean=True, n_samples=1)
    
    # Get cell info
    cell_types = adata.obs['cell_type'].values
    splits = adata.obs['split'].values
    
    # Create DataFrame - scVI uses 'theta' for dispersion in ZINB
    df = pd.DataFrame({
        'cell_idx': np.arange(adata.n_obs),
        'cell_type': cell_types,
        'split': splits,
        'mu': params['mean'].flatten(),
        'theta': params.get('theta', params.get('dispersion', np.ones(adata.n_obs))).flatten(),
        'pi': params['dropout'].flatten(),
        'predicted_expression': predictions.flatten()
    })
    
    # Save
    params_path = f"exp/scvi_training/zinb_params_{dataset_name}_seed{seed}.csv"
    df.to_csv(params_path, index=False)
    
    print(f"  Saved: {params_path}")
    
    return {
        'dataset': dataset_name,
        'seed': seed,
        'training_time': train_time,
        'params_path': params_path
    }


def main(seed=42):
    """Train on all datasets."""
    print("="*60)
    print("TRAINING SCVI MODELS")
    print("="*60)
    
    os.makedirs('exp/scvi_training/models', exist_ok=True)
    
    # Get datasets - limit to 3 for speed
    datasets = (glob.glob('data/pbmc_processed.h5ad') + 
                glob.glob('data/synthetic_d*_s*.h5ad'))[:4]
    
    print(f"Found {len(datasets)} datasets")
    
    results = []
    for dataset_path in datasets:
        try:
            result = train_and_extract(dataset_path, seed=seed, max_epochs=30)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    save_results({'models': results, 'status': 'success'}, 
                 'exp/scvi_training/results.json')
    
    print("\n" + "="*60)
    print("SCVI TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(seed=args.seed)
