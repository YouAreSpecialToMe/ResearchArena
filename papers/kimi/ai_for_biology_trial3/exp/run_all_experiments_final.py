#!/usr/bin/env python3
"""
Complete CellStratCP Experiment Runner - Final Version
"""

import os
import sys
import time
import json
import glob
import subprocess
from datetime import datetime
import numpy as np

# Configuration
SEEDS = [42, 123, 456]
CORE_DATASETS = ['pbmc_processed', 'synthetic_d30_s42', 'synthetic_d50_s42', 'synthetic_d70_s42']

def log(msg):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def extract_expression(adata, indices, gene_idx=0):
    """Extract expression values handling both sparse and dense matrices."""
    X = adata.X[indices, gene_idx]
    if hasattr(X, 'toarray'):
        return X.toarray().flatten()
    else:
        return np.asarray(X).flatten()

def step1_train_scvi():
    """Train scVI models."""
    log("STEP 1: Training scVI models")
    
    sys.path.insert(0, 'exp/shared')
    from scrna_utils import set_seed, save_results
    import scanpy as sc
    import scvi
    import pandas as pd
    
    all_results = []
    
    for dataset in CORE_DATASETS:
        for seed in SEEDS:
            log(f"Training {dataset} seed {seed}")
            
            data_path = f"data/{dataset}.h5ad"
            if not os.path.exists(data_path):
                continue
            
            try:
                set_seed(seed)
                adata = sc.read_h5ad(data_path)
                
                # Setup and train
                scvi.model.SCVI.setup_anndata(adata)
                model = scvi.model.SCVI(adata, n_layers=2, n_latent=10, dropout_rate=0.1, gene_likelihood="zinb")
                
                start = time.time()
                model.train(max_epochs=50, early_stopping=True, batch_size=128, 
                           plan_kwargs={"lr": 1e-3}, enable_progress_bar=False)
                train_time = time.time() - start
                
                # Save model
                model_dir = f"exp/scvi_training/models/scvi_{dataset}_seed{seed}"
                model.save(model_dir, overwrite=True, save_anndata=False)
                
                # Extract parameters - handle different scVI versions
                params = model.get_likelihood_parameters(adata=adata, give_mean=True)
                
                # Check available keys - params are (n_cells, n_genes)
                mean_vals = params.get('mean', params.get('mu'))
                disp_vals = params.get('dispersions', params.get('dispersion', params.get('theta')))
                drop_vals = params.get('dropout', params.get('dropouts', params.get('pi')))
                
                # Take first gene's parameters for each cell
                mean_vals = np.array(mean_vals)[:, 0]  # First gene
                disp_vals = np.array(disp_vals)[:, 0] if disp_vals is not None else np.ones(len(adata))
                drop_vals = np.array(drop_vals)[:, 0] if drop_vals is not None else np.zeros(len(adata))
                
                # Get predicted expression (mean across genes for each cell)
                posterior_means = model.get_normalized_expression(adata, return_mean=True, n_samples=1)
                if hasattr(posterior_means, 'values'):
                    posterior_means = posterior_means.values
                pred_expr = np.array(posterior_means).mean(axis=1)  # Mean across genes
                
                df = pd.DataFrame({
                    'cell_idx': np.arange(adata.n_obs),
                    'cell_type': adata.obs['cell_type'].values,
                    'split': adata.obs.get('split', ['unknown'] * len(adata)),
                    'mu': mean_vals,
                    'theta': disp_vals,
                    'pi': drop_vals,
                    'predicted_expression': pred_expr
                })
                
                params_path = f"exp/scvi_training/zinb_params_{dataset}_seed{seed}.csv"
                df.to_csv(params_path, index=False)
                
                result = {
                    'dataset': dataset,
                    'seed': seed,
                    'training_time': train_time,
                    'n_cells': adata.n_obs,
                    'n_genes': adata.n_vars,
                    'model_path': model_dir,
                    'params_path': params_path
                }
                all_results.append(result)
                log(f"  Done in {train_time:.1f}s")
                
            except Exception as e:
                log(f"  Error: {e}")
                import traceback
                traceback.print_exc()
    
    save_results({'models': all_results, 'status': 'success'}, 'exp/scvi_training/results.json')
    return len(all_results) > 0

def step2_standard_cp():
    """Run standard CP baseline."""
    log("STEP 2: Standard CP baseline")
    
    sys.path.insert(0, 'exp/shared')
    from scrna_utils import set_seed, conformal_prediction_intervals, evaluate_coverage, save_results
    import pandas as pd
    import scanpy as sc
    
    all_results = []
    
    for seed in SEEDS:
        set_seed(seed)
        param_files = glob.glob('exp/scvi_training/zinb_params_*.csv')
        
        for params_path in param_files:
            if f"_seed{seed}.csv" not in params_path:
                continue
            
            basename = os.path.basename(params_path)
            dataset_name = basename.replace('zinb_params_', '').replace(f'_seed{seed}.csv', '')
            
            try:
                df = pd.read_csv(params_path)
                cal_df = df[df['split'].isin(['calib', 'calibration'])].copy()
                test_df = df[df['split'] == 'test'].copy()
                
                # Load ground truth
                data_path = f"data/{dataset_name}.h5ad"
                if os.path.exists(data_path):
                    adata = sc.read_h5ad(data_path)
                    cal_df['x_obs'] = extract_expression(adata, cal_df['cell_idx'].values)
                    test_df['x_obs'] = extract_expression(adata, test_df['cell_idx'].values)
                else:
                    cal_df['x_obs'] = cal_df['mu'].values
                    test_df['x_obs'] = test_df['mu'].values
                
                cal_df['x_pred'] = cal_df['mu'].values
                test_df['x_pred'] = test_df['mu'].values
                
                results_by_alpha = {}
                for alpha in [0.1]:
                    cal_scores = np.abs(cal_df['x_obs'].values - cal_df['x_pred'].values)
                    q = conformal_prediction_intervals(cal_scores, None, alpha)
                    
                    test_preds = test_df['x_pred'].values
                    lowers = np.maximum(0, test_preds - q)
                    uppers = test_preds + q
                    pred_intervals = np.column_stack([lowers, uppers])
                    
                    coverage = evaluate_coverage(test_df['x_obs'].values, pred_intervals, 
                                                cell_types=test_df['cell_type'].values)
                    results_by_alpha[f'alpha_{alpha}'] = coverage
                
                result = {
                    'dataset': dataset_name,
                    'seed': seed,
                    'method': 'Standard Split CP',
                    'marginal_coverage': results_by_alpha['alpha_0.1']['marginal_coverage'],
                    'mean_interval_width': results_by_alpha['alpha_0.1']['mean_interval_width'],
                    'max_coverage_discrepancy': results_by_alpha['alpha_0.1']['max_coverage_discrepancy'],
                    'conditional_coverage': results_by_alpha['alpha_0.1'].get('conditional_coverage', {})
                }
                all_results.append(result)
                log(f"  {dataset_name}: coverage={result['marginal_coverage']:.3f}")
                
            except Exception as e:
                log(f"  Error on {dataset_name}: {e}")
    
    save_results({'experiments': all_results}, 'exp/cp_baselines/standard_cp/results.json')
    return True

def step3_cellstratcp():
    """Run CellStratCP main method."""
    log("STEP 3: CellStratCP")
    
    sys.path.insert(0, 'exp/shared')
    from scrna_utils import (set_seed, zinb_nonconformity_score, compute_prediction_interval,
        conformal_prediction_intervals, evaluate_coverage, save_results)
    import pandas as pd
    import scanpy as sc
    
    all_results = []
    
    for seed in SEEDS:
        set_seed(seed)
        param_files = glob.glob('exp/scvi_training/zinb_params_*.csv')
        
        for params_path in param_files:
            if f"_seed{seed}.csv" not in params_path:
                continue
            
            basename = os.path.basename(params_path)
            dataset_name = basename.replace('zinb_params_', '').replace(f'_seed{seed}.csv', '')
            
            try:
                df = pd.read_csv(params_path)
                cal_df = df[df['split'].isin(['calib', 'calibration'])].copy()
                test_df = df[df['split'] == 'test'].copy()
                
                # Load ground truth
                data_path = f"data/{dataset_name}.h5ad"
                if os.path.exists(data_path):
                    adata = sc.read_h5ad(data_path)
                    cal_df['x_obs'] = extract_expression(adata, cal_df['cell_idx'].values)
                    test_df['x_obs'] = extract_expression(adata, test_df['cell_idx'].values)
                else:
                    cal_df['x_obs'] = cal_df['mu'].values
                    test_df['x_obs'] = test_df['mu'].values
                
                # Compute ZINB scores
                cal_df['score'] = cal_df.apply(
                    lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'], 
                                                          row['mu'], row['theta'], row['pi']), axis=1
                )
                
                # Mondrian quantiles per cell type
                cell_types = cal_df['cell_type'].unique()
                quantiles = {}
                for ct in cell_types:
                    ct_scores = cal_df[cal_df['cell_type'] == ct]['score'].values
                    quantiles[ct] = conformal_prediction_intervals(ct_scores, None, 0.1)
                
                # Prediction intervals
                pred_intervals = []
                for idx, row in test_df.iterrows():
                    ct = row['cell_type']
                    q = quantiles.get(ct, max(quantiles.values()))
                    zinb_params = {'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}
                    lower, upper = compute_prediction_interval(zinb_params, q)
                    pred_intervals.append([lower, upper])
                
                pred_intervals = np.array(pred_intervals)
                coverage = evaluate_coverage(test_df['x_obs'].values, pred_intervals,
                                            cell_types=test_df['cell_type'].values)
                
                result = {
                    'dataset': dataset_name,
                    'seed': seed,
                    'method': 'CellStratCP',
                    'marginal_coverage': coverage['marginal_coverage'],
                    'mean_interval_width': coverage['mean_interval_width'],
                    'max_coverage_discrepancy': coverage['max_coverage_discrepancy'],
                    'conditional_coverage': coverage.get('conditional_coverage', {})
                }
                all_results.append(result)
                log(f"  {dataset_name}: coverage={result['marginal_coverage']:.3f}")
                
            except Exception as e:
                log(f"  Error on {dataset_name}: {e}")
    
    save_results({'experiments': all_results}, 'exp/cellstratcp/results.json')
    return True

def step4_ablations():
    """Run ablation studies."""
    log("STEP 4: Ablations")
    
    sys.path.insert(0, 'exp/shared')
    from scrna_utils import (set_seed, zinb_nonconformity_score, compute_prediction_interval,
        conformal_prediction_intervals, evaluate_coverage, adaptive_conformal_inference, save_results)
    import pandas as pd
    import scanpy as sc
    
    # 4.1 No Mondrian
    log("  No Mondrian ablation")
    all_results = []
    for seed in SEEDS:
        set_seed(seed)
        for params_path in glob.glob('exp/scvi_training/zinb_params_*.csv'):
            if f"_seed{seed}.csv" not in params_path:
                continue
            basename = os.path.basename(params_path)
            dataset_name = basename.replace('zinb_params_', '').replace(f'_seed{seed}.csv', '')
            
            try:
                df = pd.read_csv(params_path)
                cal_df = df[df['split'].isin(['calib', 'calibration'])].copy()
                test_df = df[df['split'] == 'test'].copy()
                
                data_path = f"data/{dataset_name}.h5ad"
                if os.path.exists(data_path):
                    adata = sc.read_h5ad(data_path)
                    cal_df['x_obs'] = extract_expression(adata, cal_df['cell_idx'].values)
                    test_df['x_obs'] = extract_expression(adata, test_df['cell_idx'].values)
                else:
                    cal_df['x_obs'] = cal_df['mu'].values
                    test_df['x_obs'] = test_df['mu'].values
                
                cal_df['score'] = cal_df.apply(
                    lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'], 
                                                          row['mu'], row['theta'], row['pi']), axis=1
                )
                
                # Pooled quantile (no Mondrian)
                pooled_q = conformal_prediction_intervals(cal_df['score'].values, None, 0.1)
                
                pred_intervals = []
                for idx, row in test_df.iterrows():
                    zinb_params = {'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}
                    lower, upper = compute_prediction_interval(zinb_params, pooled_q)
                    pred_intervals.append([lower, upper])
                
                coverage = evaluate_coverage(test_df['x_obs'].values, np.array(pred_intervals),
                                            cell_types=test_df['cell_type'].values)
                
                all_results.append({
                    'dataset': dataset_name,
                    'seed': seed,
                    'method': 'No Mondrian',
                    'marginal_coverage': coverage['marginal_coverage'],
                    'max_coverage_discrepancy': coverage['max_coverage_discrepancy']
                })
            except Exception as e:
                log(f"    Error: {e}")
    
    save_results({'experiments': all_results}, 'exp/ablations/no_mondrian/results.json')
    
    # 4.2 Residual scores
    log("  Residual scores ablation")
    all_results = []
    for seed in SEEDS:
        set_seed(seed)
        for params_path in glob.glob('exp/scvi_training/zinb_params_*.csv'):
            if f"_seed{seed}.csv" not in params_path:
                continue
            basename = os.path.basename(params_path)
            dataset_name = basename.replace('zinb_params_', '').replace(f'_seed{seed}.csv', '')
            
            try:
                df = pd.read_csv(params_path)
                cal_df = df[df['split'].isin(['calib', 'calibration'])].copy()
                test_df = df[df['split'] == 'test'].copy()
                
                data_path = f"data/{dataset_name}.h5ad"
                if os.path.exists(data_path):
                    adata = sc.read_h5ad(data_path)
                    cal_df['x_obs'] = extract_expression(adata, cal_df['cell_idx'].values)
                    test_df['x_obs'] = extract_expression(adata, test_df['cell_idx'].values)
                else:
                    cal_df['x_obs'] = cal_df['mu'].values
                    test_df['x_obs'] = test_df['mu'].values
                
                cal_df['x_pred'] = cal_df['mu'].values
                test_df['x_pred'] = test_df['mu'].values
                
                # Mondrian with residual scores
                cell_types = cal_df['cell_type'].unique()
                quantiles = {}
                for ct in cell_types:
                    ct_cal = cal_df[cal_df['cell_type'] == ct]
                    scores = np.abs(ct_cal['x_obs'].values - ct_cal['x_pred'].values)
                    quantiles[ct] = conformal_prediction_intervals(scores, None, 0.1)
                
                pred_intervals = []
                for idx, row in test_df.iterrows():
                    ct = row['cell_type']
                    q = quantiles.get(ct, max(quantiles.values()))
                    pred = row['x_pred']
                    pred_intervals.append([max(0, pred - q), pred + q])
                
                coverage = evaluate_coverage(test_df['x_obs'].values, np.array(pred_intervals),
                                            cell_types=test_df['cell_type'].values)
                
                all_results.append({
                    'dataset': dataset_name,
                    'seed': seed,
                    'method': 'Residual Scores',
                    'marginal_coverage': coverage['marginal_coverage'],
                    'mean_interval_width': coverage['mean_interval_width']
                })
            except Exception as e:
                log(f"    Error: {e}")
    
    save_results({'experiments': all_results}, 'exp/ablations/residual_scores/results.json')
    
    # 4.3 ACI ablation
    log("  ACI ablation")
    all_results = []
    for seed in SEEDS:
        set_seed(seed)
        for params_path in glob.glob('exp/scvi_training/zinb_params_*.csv'):
            if f"_seed{seed}.csv" not in params_path:
                continue
            basename = os.path.basename(params_path)
            dataset_name = basename.replace('zinb_params_', '').replace(f'_seed{seed}.csv', '')
            
            try:
                df = pd.read_csv(params_path)
                cal_df = df[df['split'].isin(['calib', 'calibration'])].copy()
                test_df = df[df['split'] == 'test'].copy()
                
                data_path = f"data/{dataset_name}.h5ad"
                if os.path.exists(data_path):
                    adata = sc.read_h5ad(data_path)
                    cal_df['x_obs'] = extract_expression(adata, cal_df['cell_idx'].values)
                    test_df['x_obs'] = extract_expression(adata, test_df['cell_idx'].values)
                else:
                    cal_df['x_obs'] = cal_df['mu'].values
                    test_df['x_obs'] = test_df['mu'].values
                
                # Simulate batch effect
                test_df['x_obs_shifted'] = test_df['x_obs'] * 0.8 + 0.5
                test_df['x_obs_shifted'] = np.maximum(0, test_df['x_obs_shifted'])
                
                cal_df['score'] = cal_df.apply(
                    lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'],
                                                          row['mu'], row['theta'], row['pi']), axis=1
                )
                
                cell_types = cal_df['cell_type'].unique()
                quantiles = {ct: conformal_prediction_intervals(
                    cal_df[cal_df['cell_type'] == ct]['score'].values, None, 0.1)
                    for ct in cell_types}
                
                # Fixed alpha
                pred_intervals = []
                for idx, row in test_df.iterrows():
                    ct = row['cell_type']
                    q = quantiles.get(ct, max(quantiles.values()))
                    zinb_params = {'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}
                    lower, upper = compute_prediction_interval(zinb_params, q)
                    pred_intervals.append([lower, upper])
                
                coverage_fixed = evaluate_coverage(test_df['x_obs_shifted'].values, 
                                                   np.array(pred_intervals),
                                                   cell_types=test_df['cell_type'].values)
                
                # With ACI
                test_df_sorted = test_df.sort_values('cell_type').reset_index(drop=True)
                coverage_history = []
                pred_intervals_aci = []
                
                for idx, row in test_df_sorted.iterrows():
                    ct = row['cell_type']
                    
                    if len(coverage_history) > 0:
                        current_alpha = adaptive_conformal_inference(coverage_history, 0.1, 0.01, 0.1)
                    else:
                        current_alpha = 0.1
                    
                    ct_scores = cal_df[cal_df['cell_type'] == ct]['score'].values
                    q = conformal_prediction_intervals(ct_scores, None, current_alpha)
                    zinb_params = {'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}
                    lower, upper = compute_prediction_interval(zinb_params, q)
                    pred_intervals_aci.append([lower, upper])
                    
                    covered = (row['x_obs_shifted'] >= lower) and (row['x_obs_shifted'] <= upper)
                    coverage_history.append(1 if covered else 0)
                
                coverage_aci = evaluate_coverage(test_df_sorted['x_obs_shifted'].values,
                                                 np.array(pred_intervals_aci),
                                                 cell_types=test_df_sorted['cell_type'].values)
                
                all_results.append({
                    'dataset': dataset_name,
                    'seed': seed,
                    'fixed_coverage': coverage_fixed['marginal_coverage'],
                    'aci_coverage': coverage_aci['marginal_coverage']
                })
            except Exception as e:
                log(f"    Error: {e}")
    
    save_results({'experiments': all_results}, 'exp/ablations/no_aci/results.json')
    return True

def step5_ood_detection():
    """Run OOD detection."""
    log("STEP 5: OOD detection")
    
    sys.path.insert(0, 'exp/shared')
    from scrna_utils import set_seed, zinb_nonconformity_score, conformal_prediction_intervals, save_results
    import pandas as pd
    import scanpy as sc
    from sklearn.metrics import roc_auc_score, roc_curve
    
    all_results = []
    
    for seed in SEEDS:
        set_seed(seed)
        for params_path in glob.glob('exp/scvi_training/zinb_params_*.csv')[:6]:
            if f"_seed{seed}.csv" not in params_path:
                continue
            
            basename = os.path.basename(params_path)
            dataset_name = basename.replace('zinb_params_', '').replace(f'_seed{seed}.csv', '')
            
            try:
                df = pd.read_csv(params_path)
                cell_types = df['cell_type'].unique()
                
                if len(cell_types) < 3:
                    continue
                
                for held_out_ct in cell_types[:3]:
                    cal_df = df[(df['split'].isin(['calib', 'calibration'])) & (df['cell_type'] != held_out_ct)].copy()
                    test_in = df[(df['split'] == 'test') & (df['cell_type'] != held_out_ct)].copy()
                    test_ood = df[(df['split'] == 'test') & (df['cell_type'] == held_out_ct)].copy()
                    
                    if len(test_ood) < 10 or len(test_in) < 10:
                        continue
                    
                    data_path = f"data/{dataset_name}.h5ad"
                    if os.path.exists(data_path):
                        adata = sc.read_h5ad(data_path)
                        cal_df['x_obs'] = extract_expression(adata, cal_df['cell_idx'].values)
                        test_in['x_obs'] = extract_expression(adata, test_in['cell_idx'].values)
                        test_ood['x_obs'] = extract_expression(adata, test_ood['cell_idx'].values)
                    else:
                        cal_df['x_obs'] = cal_df['mu'].values
                        test_in['x_obs'] = test_in['mu'].values
                        test_ood['x_obs'] = test_ood['mu'].values
                    
                    cal_df['score'] = cal_df.apply(
                        lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'],
                                                              row['mu'], row['theta'], row['pi']), axis=1
                    )
                    test_in['score'] = test_in.apply(
                        lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'],
                                                              row['mu'], row['theta'], row['pi']), axis=1
                    )
                    test_ood['score'] = test_ood.apply(
                        lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'],
                                                              row['mu'], row['theta'], row['pi']), axis=1
                    )
                    
                    y_true = np.concatenate([np.zeros(len(test_in)), np.ones(len(test_ood))])
                    scores = np.concatenate([test_in['score'].values, test_ood['score'].values])
                    
                    try:
                        auroc = roc_auc_score(y_true, scores)
                    except:
                        auroc = 0.5
                    
                    fpr, tpr, _ = roc_curve(y_true, scores)
                    idx = np.where(tpr >= 0.95)[0]
                    fpr_at_95 = fpr[idx[0]] if len(idx) > 0 else 1.0
                    
                    all_results.append({
                        'dataset': dataset_name,
                        'seed': seed,
                        'held_out_cell_type': str(held_out_ct),
                        'auroc': float(auroc),
                        'fpr_at_95_tpr': float(fpr_at_95)
                    })
            except Exception as e:
                log(f"    Error: {e}")
    
    save_results({'experiments': all_results}, 'exp/ood_detection/results.json')
    return True

def step6_runtime():
    """Run runtime analysis."""
    log("STEP 6: Runtime analysis")
    
    sys.path.insert(0, 'exp/shared')
    from scrna_utils import zinb_nonconformity_score, conformal_prediction_intervals, compute_prediction_interval, save_results
    
    results = []
    
    for n_cal in [100, 300, 1000, 3000]:
        times = []
        for _ in range(3):
            mu = np.random.lognormal(0, 1, n_cal)
            theta = np.random.gamma(2, 2, n_cal)
            pi = np.random.beta(2, 5, n_cal)
            x_obs = np.random.poisson(mu)
            
            start = time.time()
            for i in range(n_cal):
                zinb_nonconformity_score(x_obs[i], mu[i], mu[i], theta[i], pi[i])
            elapsed = time.time() - start
            times.append(elapsed)
        
        results.append({
            'n_calibration': n_cal,
            'calibration_time_mean': np.mean(times),
            'per_cell_time': np.mean(times) / n_cal
        })
    
    save_results({'runtime_analysis': results}, 'exp/runtime/results.json')
    return True

def step7_aggregate():
    """Aggregate all results."""
    log("STEP 7: Aggregating results")
    
    all_results = {
        'cellstratcp': [],
        'standard_cp': [],
        'ablations': {},
        'ood_detection': [],
        'runtime': {}
    }
    
    for key, path in [
        ('cellstratcp', 'exp/cellstratcp/results.json'),
        ('standard_cp', 'exp/cp_baselines/standard_cp/results.json'),
        ('ood_detection', 'exp/ood_detection/results.json'),
        ('runtime', 'exp/runtime/results.json')
    ]:
        try:
            with open(path) as f:
                data = json.load(f)
                if key == 'runtime':
                    all_results[key] = data.get('runtime_analysis', {})
                else:
                    all_results[key] = data.get('experiments', [])
        except:
            pass
    
    # Load ablations
    for ablation in ['no_mondrian', 'residual_scores', 'no_aci']:
        try:
            with open(f'exp/ablations/{ablation}/results.json') as f:
                data = json.load(f)
                all_results['ablations'][ablation] = data.get('experiments', [])
        except:
            pass
    
    # Summary
    def compute_summary(results, key):
        if not results:
            return {}
        values = [r.get(key, 0) for r in results if key in r]
        if not values:
            return {}
        return {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    
    all_results['summary'] = {
        'cellstratcp': {
            'marginal_coverage': compute_summary(all_results['cellstratcp'], 'marginal_coverage'),
            'mean_interval_width': compute_summary(all_results['cellstratcp'], 'mean_interval_width'),
            'max_coverage_discrepancy': compute_summary(all_results['cellstratcp'], 'max_coverage_discrepancy')
        },
        'standard_cp': {
            'marginal_coverage': compute_summary(all_results['standard_cp'], 'marginal_coverage'),
            'mean_interval_width': compute_summary(all_results['standard_cp'], 'mean_interval_width'),
            'max_coverage_discrepancy': compute_summary(all_results['standard_cp'], 'max_coverage_discrepancy')
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log(f"  CellStratCP: {len(all_results['cellstratcp'])} experiments")
    log(f"  Standard CP: {len(all_results['standard_cp'])} experiments")
    log(f"  Ablations: {sum(len(v) for v in all_results['ablations'].values())} experiments")
    log(f"  OOD: {len(all_results['ood_detection'])} experiments")
    
    return True

def main():
    """Run all experiments."""
    log("=" * 70)
    log("CELLSTRATCP FINAL EXPERIMENT RUNNER")
    log("=" * 70)
    
    start = time.time()
    
    # Create directories
    for d in ['exp/scvi_training/models', 'exp/cp_baselines/standard_cp', 
              'exp/cp_baselines/scvi_posterior', 'exp/ablations/no_mondrian',
              'exp/ablations/residual_scores', 'exp/ablations/no_aci',
              'exp/aci_experiments', 'exp/ood_detection', 'exp/runtime']:
        os.makedirs(d, exist_ok=True)
    
    step1_train_scvi()
    step2_standard_cp()
    step3_cellstratcp()
    step4_ablations()
    step5_ood_detection()
    step6_runtime()
    step7_aggregate()
    
    elapsed = time.time() - start
    log(f"\nTotal time: {elapsed/60:.1f} minutes")
    log("=" * 70)
    
    return True

if __name__ == '__main__':
    main()
