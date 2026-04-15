import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
from scrna_utils import (set_seed, conformal_prediction_intervals, evaluate_coverage, save_results)
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
        
        # Use predicted mean as point prediction
        cal_df['x_pred'] = cal_df['mu'].values
        test_df['x_pred'] = test_df['mu'].values
        
        # Mondrian stratification with residual scores
        cell_types = cal_df['cell_type'].unique()
        quantiles = {{}}
        
        for ct in cell_types:
            ct_cal = cal_df[cal_df['cell_type'] == ct]
            # Residual scores
            scores = np.abs(ct_cal['x_obs'].values - ct_cal['x_pred'].values)
            quantiles[ct] = conformal_prediction_intervals(scores, None, 0.1)
        
        # Compute prediction intervals for test
        pred_intervals = []
        for idx, row in test_df.iterrows():
            ct = row['cell_type']
            q = quantiles.get(ct, max(quantiles.values()))
            pred = row['x_pred']
            lower = max(0, pred - q)
            upper = pred + q
            pred_intervals.append([lower, upper])
        
        pred_intervals = np.array(pred_intervals)
        
        # Evaluate
        cell_types_test = test_df['cell_type'].values
        coverage_results = evaluate_coverage(test_df['x_obs'].values, pred_intervals, cell_types=cell_types_test)
        
        result = {{
            'dataset': dataset_name,
            'seed': seed,
            'method': 'CellStratCP (Residual Scores)',
            'marginal_coverage': coverage_results['marginal_coverage'],
            'mean_interval_width': coverage_results['mean_interval_width'],
            'max_coverage_discrepancy': coverage_results['max_coverage_discrepancy'],
            'conditional_coverage': coverage_results.get('conditional_coverage', {{}})
        }}
        
        all_results.append(result)
        print(f"  Coverage: {{coverage_results['marginal_coverage']:.3f}}")
        print(f"  Width: {{coverage_results['mean_interval_width']:.3f}}")

save_results({{'experiments': all_results}}, 'exp/ablations/residual_scores/results.json')
print("Residual scores ablation complete")
