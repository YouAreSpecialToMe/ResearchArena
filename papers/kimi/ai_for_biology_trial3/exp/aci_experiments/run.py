import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
from scrna_utils import (set_seed, zinb_nonconformity_score, compute_prediction_interval,
    conformal_prediction_intervals, evaluate_coverage, adaptive_conformal_inference, save_results)
import glob
import json

seeds = [42, 123, 456]
all_results = []

# Different batch effect severities
batch_effects = [
    {{'name': 'mild', 'scale': 0.9, 'shift': 0.3}},
    {{'name': 'moderate', 'scale': 0.8, 'shift': 0.5}},
    {{'name': 'severe', 'scale': 0.7, 'shift': 1.0}}
]

for seed in seeds:
    set_seed(seed)
    
    # Use a representative dataset
    for effect in batch_effects:
        param_files = glob.glob('exp/scvi_training/zinb_params_*.csv')
        
        for params_path in param_files[:4]:
            if f'seed{{seed}}' not in params_path:
                continue
            
            basename = os.path.basename(params_path)
            parts = basename.replace('zinb_params_', '').replace('.csv', '').split('_seed')
            dataset_name = parts[0]
            
            print(f"Processing {{dataset_name}}, {{effect['name']}} shift (seed {{seed}})...")
            
            df = pd.read_csv(params_path)
            cal_df = df[df['split'] == 'calibration'].copy()
            test_df = df[df['split'] == 'test'].copy()
            
            # Load ground truth
            data_path = f'data/{{dataset_name}}.h5ad'
            if not os.path.exists(data_path):
                continue
            
            import scanpy as sc
            adata = sc.read_h5ad(data_path)
            cal_indices = cal_df['cell_idx'].values
            test_indices = test_df['cell_idx'].values
            cal_df['x_obs'] = adata.X[cal_indices, 0].toarray().flatten()
            test_df['x_obs'] = adata.X[test_indices, 0].toarray().flatten()
            
            # Apply batch effect
            test_df['x_obs_shifted'] = test_df['x_obs'] * effect['scale'] + effect['shift']
            test_df['x_obs_shifted'] = np.maximum(0, test_df['x_obs_shifted'])
            
            # Compute calibration scores
            cal_df['score'] = cal_df.apply(
                lambda row: zinb_nonconformity_score(row['x_obs'], row['mu'],
                                                      row['mu'], row['theta'], row['pi']), axis=1
            )
            
            # Get cell-type quantiles
            cell_types = cal_df['cell_type'].unique()
            quantiles = {{ct: conformal_prediction_intervals(
                cal_df[cal_df['cell_type'] == ct]['score'].values, None, 0.1) 
                for ct in cell_types}}
            
            # Without ACI
            pred_intervals_no_aci = []
            for idx, row in test_df.iterrows():
                ct = row['cell_type']
                q = quantiles.get(ct, max(quantiles.values()))
                zinb_params = {{'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}}
                lower, upper = compute_prediction_interval(zinb_params, q)
                pred_intervals_no_aci.append([lower, upper])
            
            coverage_no_aci = evaluate_coverage(
                test_df['x_obs_shifted'].values, 
                np.array(pred_intervals_no_aci),
                cell_types=test_df['cell_type'].values
            )
            
            # With ACI
            test_df_sorted = test_df.sort_values('cell_type').reset_index(drop=True)
            coverage_history = []
            alpha_history = [0.1]
            pred_intervals_aci = []
            
            for idx, row in test_df_sorted.iterrows():
                ct = row['cell_type']
                
                if len(coverage_history) > 0:
                    current_alpha = adaptive_conformal_inference(coverage_history, 0.1, 0.005, alpha_history[-1])
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
            
            coverage_aci = evaluate_coverage(
                test_df_sorted['x_obs_shifted'].values,
                np.array(pred_intervals_aci),
                cell_types=test_df_sorted['cell_type'].values
            )
            
            result = {{
                'dataset': dataset_name,
                'seed': seed,
                'batch_effect': effect['name'],
                'scale': effect['scale'],
                'shift': effect['shift'],
                'no_aci_coverage': coverage_no_aci['marginal_coverage'],
                'aci_coverage': coverage_aci['marginal_coverage'],
                'no_aci_discrepancy': coverage_no_aci['max_coverage_discrepancy'],
                'aci_discrepancy': coverage_aci['max_coverage_discrepancy']
            }}
            
            all_results.append(result)
            print(f"  No ACI: {{coverage_no_aci['marginal_coverage']:.3f}}, ACI: {{coverage_aci['marginal_coverage']:.3f}}")

save_results({{'experiments': all_results}}, 'exp/aci_experiments/results.json')
print("ACI experiments complete")
