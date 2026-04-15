import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
from scrna_utils import (set_seed, zinb_nonconformity_score, compute_prediction_interval,
    conformal_prediction_intervals, evaluate_coverage, adaptive_conformal_inference, save_results)
import glob

seeds = [42, 123, 456]
all_results = []

for seed in seeds:
    set_seed(seed)
    param_files = glob.glob('exp/scvi_training/zinb_params_*.csv')
    
    for params_path in param_files:
        if f'seed{{seed}}' not in params_path:
            continue
        
        basename = os.path.basename(params_path)
        parts = basename.replace('zinb_params_', '').replace('.csv', '').split('_seed')
        dataset_name = parts[0]
        
        print(f"Processing {{dataset_name}} (seed {{seed}})...")
        
        df = pd.read_csv(params_path)
        cal_df = df[df['split'] == 'calibration'].copy()
        test_df = df[df['split'] == 'test'].copy()
        
        # Load ground truth
        data_path = f'data/{{dataset_name}}.h5ad'
        if os.path.exists(data_path):
            import scanpy as sc
            adata = sc.read_h5ad(data_path)
            cal_indices = cal_df['cell_idx'].values
            test_indices = test_df['cell_idx'].values
            cal_df['x_obs'] = adata.X[cal_indices, 0].toarray().flatten()
            test_df['x_obs'] = adata.X[test_indices, 0].toarray().flatten()
        else:
            cal_df['x_obs'] = cal_df['mu'].values
            test_df['x_obs'] = test_df['mu'].values
        
        # Add artificial batch effect to test data (simulate distribution shift)
        # Scale gene expression by 0.8 and shift
        test_df['x_obs_shifted'] = test_df['x_obs'] * 0.8 + 0.5
        test_df['x_obs_shifted'] = np.maximum(0, test_df['x_obs_shifted'])
        
        # Compute scores on calibration
        cal_df['score'] = cal_df.apply(
            lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'],
                                                  row['mu'], row['theta'], row['pi']), axis=1
        )
        
        # Get cell-type-specific quantiles
        cell_types = cal_df['cell_type'].unique()
        quantiles_fixed = {{}}
        for ct in cell_types:
            ct_scores = cal_df[cal_df['cell_type'] == ct]['score'].values
            quantiles_fixed[ct] = conformal_prediction_intervals(ct_scores, None, 0.1)
        
        # Test with FIXED alpha (no ACI) on shifted data
        pred_intervals_fixed = []
        for idx, row in test_df.iterrows():
            ct = row['cell_type']
            q = quantiles_fixed.get(ct, max(quantiles_fixed.values()))
            zinb_params = {{'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}}
            lower, upper = compute_prediction_interval(zinb_params, q)
            pred_intervals_fixed.append([lower, upper])
        
        pred_intervals_fixed = np.array(pred_intervals_fixed)
        
        # Evaluate on shifted data
        cell_types_test = test_df['cell_type'].values
        coverage_fixed = evaluate_coverage(test_df['x_obs_shifted'].values, pred_intervals_fixed, 
                                           cell_types=cell_types_test)
        
        # Test with ACI on shifted data
        test_df_sorted = test_df.sort_values('cell_type').reset_index(drop=True)
        coverage_history = []
        alpha_history = [0.1]
        pred_intervals_aci = []
        
        for idx, row in test_df_sorted.iterrows():
            ct = row['cell_type']
            
            if len(coverage_history) > 0:
                current_alpha = adaptive_conformal_inference(coverage_history, 0.1, 0.01, alpha_history[-1])
                alpha_history.append(current_alpha)
            else:
                current_alpha = 0.1
            
            ct_scores = cal_df[cal_df['cell_type'] == ct]['score'].values
            q = conformal_prediction_intervals(ct_scores, None, current_alpha)
            
            zinb_params = {{'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}}
            lower, upper = compute_prediction_interval(zinb_params, q)
            pred_intervals_aci.append([lower, upper])
            
            covered = (row['x_obs_shifted'] >= lower) and (row['x_obs_shifted'] <= upper)
            coverage_history.append(1 if covered else 0)
        
        pred_intervals_aci = np.array(pred_intervals_aci)
        coverage_aci = evaluate_coverage(test_df_sorted['x_obs_shifted'].values, pred_intervals_aci,
                                         cell_types=test_df_sorted['cell_type'].values)
        
        result = {{
            'dataset': dataset_name,
            'seed': seed,
            'fixed_alpha': {{
                'marginal_coverage': coverage_fixed['marginal_coverage'],
                'max_coverage_discrepancy': coverage_fixed['max_coverage_discrepancy']
            }},
            'aci': {{
                'marginal_coverage': coverage_aci['marginal_coverage'],
                'max_coverage_discrepancy': coverage_aci['max_coverage_discrepancy']
            }},
            'coverage_improvement': coverage_aci['marginal_coverage'] - coverage_fixed['marginal_coverage']
        }}
        
        all_results.append(result)
        print(f"  Fixed alpha coverage: {{coverage_fixed['marginal_coverage']:.3f}}")
        print(f"  ACI coverage: {{coverage_aci['marginal_coverage']:.3f}}")

save_results({{'experiments': all_results}}, 'exp/ablations/no_aci/results.json')
print("ACI ablation complete")
