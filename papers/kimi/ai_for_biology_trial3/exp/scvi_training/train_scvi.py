"""
Train scVI models as base imputation predictors for CellStratCP.
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


def train_scvi_model(adata_path, output_dir, seed=42, max_epochs=100):
    """Train scVI model on a dataset."""
    set_seed(seed)
    
    # Load data
    adata = sc.read_h5ad(adata_path)
    
    # Get dataset name
    dataset_name = os.path.basename(adata_path).replace('.h5ad', '')
    
    print(f"\nTraining scVI on {dataset_name}")
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")
    
    # Setup scVI
    scvi.model.SCVI.setup_anndata(adata)
    
    # Create model
    model = scvi.model.SCVI(
        adata,
        n_layers=2,
        n_latent=10,
        dropout_rate=0.1,
        gene_likelihood="zinb"
    )
    
    # Train
    start_time = time.time()
    model.train(
        max_epochs=max_epochs,
        early_stopping=True,
        batch_size=128,
        plan_kwargs={"lr": 1e-3}
    )
    training_time = time.time() - start_time
    
    # Save model
    model_dir = os.path.join(output_dir, f"scvi_{dataset_name}_seed{seed}")
    model.save(model_dir, overwrite=True, save_anndata=False)
    
    # Get training history
    try:
        history = model.history
        final_loss = float(history['train_loss_epoch'].iloc[-1]) if 'train_loss_epoch' in history else None
    except:
        final_loss = None
    
    results = {
        'dataset': dataset_name,
        'seed': seed,
        'training_time': training_time,
        'n_epochs': len(model.history['train_loss_epoch']) if hasattr(model, 'history') else max_epochs,
        'final_train_loss': final_loss,
        'model_path': model_dir,
        'adata_path': adata_path
    }
    
    print(f"  Training time: {training_time:.2f}s")
    if final_loss:
        print(f"  Final loss: {final_loss:.4f}")
    
    return model, results


def extract_zinb_parameters(model, adata, output_path):
    """Extract ZINB parameters (mu, theta, pi) from trained scVI model."""
    print("  Extracting ZINB parameters...")
    
    # Get likelihood parameters
    params = model.get_likelihood_parameters(
        adata=adata,
        batch_size=256,
        give_mean=True
    )
    
    # Get predictions (posterior means)
    posterior_means = model.get_normalized_expression(adata, return_mean=True, n_samples=1)
    
    # Get cell types and splits
    cell_types = adata.obs['cell_type'].values
    splits = adata.obs['split'].values if 'split' in adata.obs.columns else ['unknown'] * len(adata)
    
    # Create DataFrame
    df = pd.DataFrame({
        'cell_idx': np.arange(adata.n_obs),
        'cell_type': cell_types,
        'split': splits,
        'mu': params['mean'].flatten(),
        'theta': params['dispersion'].flatten(),
        'pi': params['dropout'].flatten(),
        'predicted_expression': posterior_means.flatten()
    })
    
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    
    return df


def train_all_models(seed=42):
    """Train scVI models on all prepared datasets."""
    print("="*60)
    print("TRAINING SCVI MODELS")
    print("="*60)
    
    os.makedirs('exp/scvi_training/models', exist_ok=True)
    
    results = {'models': [], 'status': 'success'}
    
    # Find all prepared datasets
    datasets = glob.glob('data/pbmc_processed.h5ad') + glob.glob('data/synthetic_d*_s*.h5ad')
    
    print(f"Found {len(datasets)} datasets")
    
    for dataset_path in datasets[:5]:  # Limit to 5 datasets for time
        try:
            # Train model
            model, model_results = train_scvi_model(
                dataset_path,
                'exp/scvi_training/models',
                seed=seed,
                max_epochs=50  # Reduced for speed
            )
            
            # Extract parameters
            dataset_name = model_results['dataset']
            params_path = f"exp/scvi_training/zinb_params_{dataset_name}_seed{seed}.csv"
            
            adata = sc.read_h5ad(dataset_path)
            extract_zinb_parameters(model, adata, params_path)
            
            model_results['params_path'] = params_path
            results['models'].append(model_results)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results['models'].append({
                'dataset': os.path.basename(dataset_path),
                'error': str(e)
            })
    
    # Save results
    save_results(results, 'exp/scvi_training/results.json')
    
    print("\n" + "="*60)
    print("SCVI TRAINING COMPLETE")
    print("="*60)
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_all_models(seed=args.seed)
