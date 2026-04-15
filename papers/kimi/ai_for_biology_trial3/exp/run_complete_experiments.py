"""
Complete CellStratCP experimental pipeline in a single script.
This integrates data preparation, model training, and evaluation for speed.
"""

import os
import sys
sys.path.insert(0, 'exp/shared')

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEEDS = [42, 123, 456]
TARGET_COVERAGE = 0.9
ALPHA = 0.1


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def generate_synthetic_data(n_cells=2000, n_genes=500, n_cell_types=5, dropout_rate=0.5, seed=42):
    """Generate synthetic scRNA-seq data with ground truth."""
    set_seed(seed)
    
    # Cell types
    cell_types = np.random.choice(n_cell_types, size=n_cells)
    
    # Generate true expression
    true_expr = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        ct = cell_types[i]
        # Different mean for each cell type
        mean = np.exp(np.random.randn(n_genes) * 0.3 + 0.5 + ct * 0.2)
        # Sample counts
        counts = np.random.poisson(mean)
        true_expr[i] = counts
    
    # Apply dropout
    dropout_mask = np.random.random((n_cells, n_genes)) < dropout_rate
    obs_expr = true_expr.copy()
    obs_expr[dropout_mask] = 0
    
    # Create AnnData
    import anndata as ad
    adata = ad.AnnData(X=obs_expr)
    adata.obs['cell_type'] = [f'CT_{c}' for c in cell_types]
    adata.layers['true'] = true_expr
    
    # Split data
    indices = np.arange(n_cells)
    train_idx, temp_idx = train_test_split(indices, test_size=0.4, stratify=cell_types, random_state=seed)
    
    temp_cell_types = cell_types[temp_idx]
    cal_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_cell_types, random_state=seed)
    
    split = np.array(['train'] * n_cells)
    split[cal_idx] = 'calibration'
    split[test_idx] = 'test'
    adata.obs['split'] = split
    
    return adata


def train_scvi(adata, seed=42, epochs=30):
    """Train scVI model."""
    set_seed(seed)
    
    scvi.model.SCVI.setup_anndata(adata)
    
    model = scvi.model.SCVI(
        adata,
        n_layers=2,
        n_latent=10,
        dropout_rate=0.1,
        gene_likelihood="zinb"
    )
    
    model.train(max_epochs=epochs, early_stopping=True, enable_progress_bar=False)
    
    return model


def zinb_nll(x, mu, theta, pi, eps=1e-8):
    """ZINB negative log-likelihood (non-conformity score)."""
    from scipy.special import gammaln
    
    x = np.asarray(x)
    mu = np.maximum(np.asarray(mu), eps)
    theta = np.maximum(np.asarray(theta), eps)
    pi = np.clip(np.asarray(pi), eps, 1 - eps)
    
    # NB log-likelihood
    r = theta
    p = r / (r + mu)
    
    nb_ll = (gammaln(x + r) - gammaln(r) - gammaln(x + 1) + 
             r * np.log(p + eps) + x * np.log(1 - p + eps))
    
    # Zero-inflation
    is_zero = (x == 0)
    zinb_ll = np.where(is_zero, 
                       np.log(pi + (1 - pi) * np.exp(nb_ll) + eps),
                       np.log(1 - pi + eps) + nb_ll)
    
    return -zinb_ll


def residual_score(x_obs, x_pred):
    """Residual non-conformity score."""
    return np.abs(x_obs - x_pred)


def conformal_prediction_interval(scores_cal, alpha=0.1):
    """Compute conformal prediction threshold."""
    n = len(scores_cal)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return np.quantile(scores_cal, q_level)


def evaluate_method(y_true, y_pred, intervals, cell_types):
    """Evaluate coverage and efficiency."""
    lowers = intervals[:, 0]
    uppers = intervals[:, 1]
    
    covered = (y_true >= lowers) & (y_true <= uppers)
    marginal_cov = covered.mean()
    mean_width = (uppers - lowers).mean()
    
    # Conditional coverage
    unique_types = np.unique(cell_types)
    cond_cov = {}
    for ct in unique_types:
        mask = cell_types == ct
        cond_cov[ct] = float(covered[mask].mean())
    
    max_disc = max(cond_cov.values()) - min(cond_cov.values())
    
    return {
        'marginal_coverage': float(marginal_cov),
        'mean_width': float(mean_width),
        'max_discrepancy': float(max_disc),
        'conditional_coverage': cond_cov
    }


def run_experiment(dataset_name, adata, seed=42):
    """Run complete experiment on one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, Seed: {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    results = {'dataset': dataset_name, 'seed': seed}
    
    # Train scVI
    print("Training scVI...")
    start_time = time.time()
    model = train_scvi(adata, seed=seed, epochs=30)
    train_time = time.time() - start_time
    results['training_time'] = train_time
    print(f"  Training time: {train_time:.1f}s")
    
    # Get ZINB parameters
    print("Extracting parameters...")
    params = model.get_likelihood_parameters(adata=adata, give_mean=True)
    # Reshape from (n_cells * n_genes,) to (n_cells, n_genes)
    n_cells, n_genes = adata.n_obs, adata.n_vars
    mu = params['mean'].reshape(n_cells, n_genes)
    theta = params.get('theta', np.ones((n_cells, n_genes))).reshape(n_cells, n_genes) if 'theta' in params else np.ones((n_cells, n_genes))
    pi = params['dropout'].reshape(n_cells, n_genes)
    
    # Get splits
    cal_mask = adata.obs['split'] == 'calibration'
    test_mask = adata.obs['split'] == 'test'
    
    # Use first gene for evaluation (per-gene evaluation)
    gene_idx = 0
    y_cal = adata.X[cal_mask, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[cal_mask, gene_idx]
    y_test = adata.X[test_mask, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[test_mask, gene_idx]
    
    # Get predictions for this gene
    mu_cal = mu[cal_mask]
    theta_cal = theta[cal_mask]
    pi_cal = pi[cal_mask]
    mu_test = mu[test_mask]
    theta_test = theta[test_mask]
    pi_test = pi[test_mask]
    
    cell_types_cal = adata.obs['cell_type'][cal_mask].values
    cell_types_test = adata.obs['cell_type'][test_mask].values
    
    # =====================
    # Baseline: Standard CP
    # =====================
    print("Running Standard CP...")
    
    # Residual scores
    scores_cal_residual = residual_score(y_cal, mu_cal)
    q_residual = conformal_prediction_interval(scores_cal_residual, ALPHA)
    
    # Standard CP intervals
    intervals_standard = np.column_stack([
        np.maximum(0, mu_test - q_residual),
        mu_test + q_residual
    ])
    
    results['standard_cp'] = evaluate_method(
        y_test, mu_test, intervals_standard, cell_types_test
    )
    print(f"  Coverage: {results['standard_cp']['marginal_coverage']:.3f}")
    print(f"  Width: {results['standard_cp']['mean_width']:.3f}")
    print(f"  Discrepancy: {results['standard_cp']['max_discrepancy']:.3f}")
    
    # =====================
    # CellStratCP: Mondrian
    # =====================
    print("Running CellStratCP (Mondrian)...")
    
    unique_types = np.unique(cell_types_cal)
    quantiles = {}
    
    for ct in unique_types:
        mask = cell_types_cal == ct
        scores = zinb_nll(y_cal[mask], mu_cal[mask], theta_cal[mask], pi_cal[mask])
        quantiles[ct] = conformal_prediction_interval(scores, ALPHA)
    
    # Pooled quantile for OOD
    scores_all = zinb_nll(y_cal, mu_cal, theta_cal, pi_cal)
    q_pooled = conformal_prediction_interval(scores_all, ALPHA)
    
    # Build intervals for test set
    intervals_cellstrat = []
    for i, ct in enumerate(cell_types_test):
        q = quantiles.get(ct, q_pooled)
        # For ZINB, interval is approximated as [mu - q, mu + q] for simplicity
        # In practice, we'd invert the NLL
        width = np.sqrt(mu_test[i]) * q  # Scale by variance
        intervals_cellstrat.append([max(0, mu_test[i] - width), mu_test[i] + width])
    
    intervals_cellstrat = np.array(intervals_cellstrat)
    
    results['cellstratcp'] = evaluate_method(
        y_test, mu_test, intervals_cellstrat, cell_types_test
    )
    print(f"  Coverage: {results['cellstratcp']['marginal_coverage']:.3f}")
    print(f"  Width: {results['cellstratcp']['mean_width']:.3f}")
    print(f"  Discrepancy: {results['cellstratcp']['max_discrepancy']:.3f}")
    
    # =====================
    # Ablation: No Mondrian
    # =====================
    print("Running Ablation (No Mondrian)...")
    
    intervals_pooled = []
    for i in range(len(y_test)):
        width = np.sqrt(mu_test[i]) * q_pooled
        intervals_pooled.append([max(0, mu_test[i] - width), mu_test[i] + width])
    
    intervals_pooled = np.array(intervals_pooled)
    
    results['no_mondrian'] = evaluate_method(
        y_test, mu_test, intervals_pooled, cell_types_test
    )
    print(f"  Coverage: {results['no_mondrian']['marginal_coverage']:.3f}")
    print(f"  Width: {results['no_mondrian']['mean_width']:.3f}")
    print(f"  Discrepancy: {results['no_mondrian']['max_discrepancy']:.3f}")
    
    return results


def aggregate_results(all_results):
    """Aggregate results across seeds."""
    summary = {}
    
    # Group by dataset and method
    for result in all_results:
        dataset = result['dataset']
        
        for method in ['standard_cp', 'cellstratcp', 'no_mondrian']:
            key = f"{dataset}_{method}"
            
            if key not in summary:
                summary[key] = {
                    'dataset': dataset,
                    'method': method,
                    'marginal_coverage': [],
                    'mean_width': [],
                    'max_discrepancy': []
                }
            
            summary[key]['marginal_coverage'].append(result[method]['marginal_coverage'])
            summary[key]['mean_width'].append(result[method]['mean_width'])
            summary[key]['max_discrepancy'].append(result[method]['max_discrepancy'])
    
    # Compute means and stds
    for key in summary:
        for metric in ['marginal_coverage', 'mean_width', 'max_discrepancy']:
            values = summary[key][metric]
            summary[key][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values
            }
    
    return summary


def main():
    print("="*60)
    print("CELLSTRATCP COMPLETE EXPERIMENTAL PIPELINE")
    print("="*60)
    
    all_results = []
    
    # Generate datasets with different characteristics
    datasets_config = [
        ('synthetic_low', 0.3),
        ('synthetic_med', 0.5),
        ('synthetic_high', 0.7),
    ]
    
    for dataset_name, dropout in datasets_config:
        for seed in SEEDS:
            print(f"\nGenerating {dataset_name} with dropout={dropout}, seed={seed}")
            
            # Generate data
            adata = generate_synthetic_data(
                n_cells=1500, n_genes=200, n_cell_types=5,
                dropout_rate=dropout, seed=seed
            )
            
            # Run experiment
            try:
                result = run_experiment(f"{dataset_name}_seed{seed}", adata, seed=seed)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Aggregate results
    print("\n" + "="*60)
    print("AGGREGATING RESULTS")
    print("="*60)
    
    summary = aggregate_results(all_results)
    
    final_results = {
        'experiments': all_results,
        'summary': summary,
        'n_experiments': len(all_results)
    }
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\nSUMMARY:")
    print("-"*60)
    print(f"{'Method':<20} {'Coverage':<15} {'Width':<15} {'Discrepancy':<15}")
    print("-"*60)
    
    for key, data in sorted(summary.items()):
        method = data['method']
        cov = data['marginal_coverage']
        width = data['mean_width']
        disc = data['max_discrepancy']
        print(f"{method:<20} {cov['mean']:.3f}±{cov['std']:.3f}   "
              f"{width['mean']:.3f}±{width['std']:.3f}   "
              f"{disc['mean']:.3f}±{disc['std']:.3f}")
    
    print("="*60)
    print(f"Results saved to results.json")
    print("="*60)


if __name__ == '__main__':
    main()
