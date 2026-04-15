"""
CellStratCP: Cell-Type-Stratified Adaptive Conformal Prediction.
Main implementation with ZINB-based non-conformity scores.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))

import numpy as np
import pandas as pd
from scrna_utils import (
    set_seed, zinb_nonconformity_score, compute_prediction_interval,
    conformal_prediction_intervals, evaluate_coverage, save_results,
    adaptive_conformal_inference
)
import glob
import time


def cellstratcp(cal_df, test_df, alpha=0.1, use_mondrian=True, use_aci=False, 
                gamma=0.01, cell_type_col='cell_type'):
    """
    CellStratCP with optional Mondrian stratification and ACI.
    
    Args:
        cal_df: DataFrame with calibration data
        test_df: DataFrame with test data
        alpha: target miscoverage level
        use_mondrian: if True, use cell-type stratification
        use_aci: if True, use adaptive conformal inference
        gamma: ACI learning rate
        cell_type_col: column name for cell types
    
    Returns:
        dict with prediction intervals and metrics
    """
    # Compute ZINB non-conformity scores for calibration set
    cal_df = cal_df.copy()
    cal_df['score'] = cal_df.apply(
        lambda row: zinb_nonconformity_score(
            row['x_obs'], row['x_pred'], 
            row['mu'], row['theta'], row['pi']
        ), axis=1
    )
    
    # Get unique cell types
    cell_types = cal_df[cell_type_col].unique()
    
    # Compute quantiles
    if use_mondrian:
        # Mondrian: separate quantile per cell type
        quantiles = {}
        for ct in cell_types:
            ct_scores = cal_df[cal_df[cell_type_col] == ct]['score'].values
            quantiles[ct] = conformal_prediction_intervals(ct_scores, None, alpha)
    else:
        # Pooled: single quantile for all
        pooled_quantile = conformal_prediction_intervals(cal_df['score'].values, None, alpha)
        quantiles = {ct: pooled_quantile for ct in cell_types}
    
    # For ACI, track alpha over time
    if use_aci:
        alpha_history = [alpha]
    
    # Compute prediction intervals for test set
    test_df = test_df.copy()
    pred_intervals = []
    
    # Sort test by cell type for ACI (to apply adaptation within each type)
    if use_aci:
        test_df = test_df.sort_values(by=cell_type_col).reset_index(drop=True)
    
    coverage_history = []
    
    for idx, row in test_df.iterrows():
        ct = row[cell_type_col]
        
        # Get quantile for this cell type
        if ct in quantiles:
            q = quantiles[ct]
        else:
            # Cell type not seen in calibration, use pooled
            q = conformal_prediction_intervals(cal_df['score'].values, None, alpha)
        
        # ACI adaptation
        if use_aci and len(coverage_history) > 0:
            # Update alpha based on recent coverage
            current_alpha = adaptive_conformal_inference(
                coverage_history, alpha, gamma, alpha_history[-1]
            )
            alpha_history.append(current_alpha)
            
            # Recompute quantile with new alpha
            if use_mondrian and ct in quantiles:
                ct_scores = cal_df[cal_df[cell_type_col] == ct]['score'].values
                q = conformal_prediction_intervals(ct_scores, None, current_alpha)
            else:
                q = conformal_prediction_intervals(cal_df['score'].values, None, current_alpha)
        
        # Compute prediction interval from ZINB parameters
        zinb_params = {
            'mu': row['mu'],
            'theta': row['theta'],
            'pi': row['pi']
        }
        lower, upper = compute_prediction_interval(zinb_params, q)
        pred_intervals.append([lower, upper])
        
        # Track coverage for ACI
        if use_aci:
            covered = (row['x_obs'] >= lower) and (row['x_obs'] <= upper)
            coverage_history.append(1 if covered else 0)
    
    pred_intervals = np.array(pred_intervals)
    
    return {
        'prediction_intervals': pred_intervals,
        'quantiles': quantiles,
        'test_df': test_df
    }


def run_cellstratcp(dataset_name, seed=42, configs=None):
    """Run CellStratCP on a dataset with various configurations."""
    set_seed(seed)
    
    if configs is None:
        configs = [
            {'name': 'CellStratCP (Mondrian)', 'use_mondrian': True, 'use_aci': False},
            {'name': 'CellStratCP (Mondrian+ACI)', 'use_mondrian': True, 'use_aci': True, 'gamma': 0.01},
        ]
    
    # Load ZINB parameters
    params_path = f"exp/scvi_training/zinb_params_{dataset_name}_seed{seed}.csv"
    
    if not os.path.exists(params_path):
        print(f"  Parameters not found: {params_path}")
        return None
    
    df = pd.read_csv(params_path)
    
    # Split into calibration and test
    cal_df = df[df['split'] == 'calibration'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Load ground truth
    adata_path = f"data/{dataset_name.replace('_seed' + str(seed), '')}.h5ad"
    if not os.path.exists(adata_path):
        alt_paths = [
            f"data/{dataset_name}.h5ad",
            f"data/{dataset_name.replace(f'_seed{seed}', '')}_processed.h5ad"
        ]
        for path in alt_paths:
            if os.path.exists(path):
                adata_path = path
                break
    
    try:
        import scanpy as sc
        adata = sc.read_h5ad(adata_path)
        
        cal_indices = cal_df['cell_idx'].values
        test_indices = test_df['cell_idx'].values
        
        # Use first gene for evaluation
        gene_idx = 0
        cal_df['x_obs'] = adata.X[cal_indices, gene_idx].toarray().flatten()
        test_df['x_obs'] = adata.X[test_indices, gene_idx].toarray().flatten()
        cal_df['x_pred'] = cal_df['mu'].values
        test_df['x_pred'] = test_df['mu'].values
        
    except Exception as e:
        print(f"  Using synthetic ground truth: {e}")
        cal_df['x_obs'] = cal_df['mu'].values + np.random.randn(len(cal_df)) * np.sqrt(cal_df['mu'].values)
        cal_df['x_obs'] = np.maximum(0, cal_df['x_obs'])
        test_df['x_obs'] = test_df['mu'].values + np.random.randn(len(test_df)) * np.sqrt(test_df['mu'].values)
        test_df['x_obs'] = np.maximum(0, test_df['x_obs'])
        cal_df['x_pred'] = cal_df['mu'].values
        test_df['x_pred'] = test_df['mu'].values
    
    all_results = []
    
    for config in configs:
        print(f"  Running: {config['name']}")
        
        start_time = time.time()
        
        # Run CellStratCP
        cp_results = cellstratcp(
            cal_df, test_df,
            alpha=0.1,
            use_mondrian=config.get('use_mondrian', True),
            use_aci=config.get('use_aci', False),
            gamma=config.get('gamma', 0.01)
        )
        
        runtime = time.time() - start_time
        
        # Evaluate coverage
        cell_types = cp_results['test_df']['cell_type'].values
        coverage_results = evaluate_coverage(
            cp_results['test_df']['x_obs'].values,
            cp_results['prediction_intervals'],
            cell_types=cell_types
        )
        
        result = {
            'dataset': dataset_name,
            'seed': seed,
            'method': config['name'],
            'config': config,
            'n_calibration': len(cal_df),
            'n_test': len(test_df),
            'runtime': runtime,
            'marginal_coverage': coverage_results['marginal_coverage'],
            'mean_interval_width': coverage_results['mean_interval_width'],
            'max_coverage_discrepancy': coverage_results['max_coverage_discrepancy'],
            'conditional_coverage': coverage_results.get('conditional_coverage', {})
        }
        
        all_results.append(result)
        
        print(f"    Coverage: {coverage_results['marginal_coverage']:.3f}")
        print(f"    Width: {coverage_results['mean_interval_width']:.3f}")
        print(f"    Discrepancy: {coverage_results['max_coverage_discrepancy']:.3f}")
    
    return all_results


def run_all(seed=42):
    """Run CellStratCP on all datasets."""
    print("="*60)
    print("CELLSTRATCP: CELL-TYPE-STRATIFIED CONFORMAL PREDICTION")
    print("="*60)
    
    all_results = []
    
    # Find all parameter files
    param_files = glob.glob('exp/scvi_training/zinb_params_*.csv')
    
    for params_path in param_files:
        basename = os.path.basename(params_path)
        parts = basename.replace('zinb_params_', '').replace('.csv', '').split('_seed')
        dataset_name = parts[0]
        
        print(f"\nProcessing {dataset_name}...")
        try:
            results = run_cellstratcp(dataset_name, seed=seed)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    save_results({'experiments': all_results}, 'exp/cellstratcp/results.json')
    
    print("\n" + "="*60)
    print("CELLSTRATCP COMPLETE")
    print("="*60)
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_all(seed=args.seed)
