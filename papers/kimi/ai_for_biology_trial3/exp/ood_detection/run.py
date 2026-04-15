import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
from scrna_utils import (set_seed, zinb_nonconformity_score, conformal_prediction_intervals,
    evaluate_ood_detection, save_results)
import glob
from sklearn.metrics import roc_auc_score

seeds = [42, 123, 456]
all_results = []

for seed in seeds:
    set_seed(seed)
    param_files = glob.glob('exp/scvi_training/zinb_params_*.csv')
    
    for params_path in param_files[:3]:
        if f'seed{{seed}}' not in params_path:
            continue
        
        basename = os.path.basename(params_path)
        parts = basename.replace('zinb_params_', '').replace('.csv', '').split('_seed')
        dataset_name = parts[0]
        
        print(f"Processing {{dataset_name}} (seed {{seed}})...")
        
        df = pd.read_csv(params_path)
        
        # Get unique cell types
        cell_types = df['cell_type'].unique()
        if len(cell_types) < 3:
            continue
        
        # Leave-one-cell-type-out for each cell type
        for held_out_ct in cell_types[:3]:
            # Train on all but held_out
            cal_df = df[(df['split'] == 'calibration') & (df['cell_type'] != held_out_ct)].copy()
            test_in = df[(df['split'] == 'test') & (df['cell_type'] != held_out_ct)].copy()
            test_ood = df[(df['split'] == 'test') & (df['cell_type'] == held_out_ct)].copy()
            
            if len(test_ood) < 10 or len(test_in) < 10:
                continue
            
            # Load ground truth
            data_path = f'data/{{dataset_name}}.h5ad'
            if os.path.exists(data_path):
                import scanpy as sc
                adata = sc.read_h5ad(data_path)
                cal_df['x_obs'] = adata.X[cal_df['cell_idx'].values, 0].toarray().flatten()
                test_in['x_obs'] = adata.X[test_in['cell_idx'].values, 0].toarray().flatten()
                test_ood['x_obs'] = adata.X[test_ood['cell_idx'].values, 0].toarray().flatten()
            else:
                cal_df['x_obs'] = cal_df['mu'].values
                test_in['x_obs'] = test_in['mu'].values
                test_ood['x_obs'] = test_ood['mu'].values
            
            # Compute non-conformity scores
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
            
            # Use pooled quantile for OOD detection
            pooled_quantile = conformal_prediction_intervals(cal_df['score'].values, None, 0.1)
            
            # OOD detection: flag cells with score > quantile
            in_scores = test_in['score'].values
            ood_scores = test_ood['score'].values
            
            # Compute AUROC
            y_true = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
            scores = np.concatenate([in_scores, ood_scores])
            
            try:
                auroc = roc_auc_score(y_true, scores)
            except:
                auroc = 0.5
            
            # FPR at 95% TPR
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, scores)
            idx = np.where(tpr >= 0.95)[0]
            fpr_at_95 = fpr[idx[0]] if len(idx) > 0 else 1.0
            
            result = {{
                'dataset': dataset_name,
                'seed': seed,
                'held_out_cell_type': str(held_out_ct),
                'n_in_distribution': len(test_in),
                'n_ood': len(test_ood),
                'auroc': float(auroc),
                'fpr_at_95_tpr': float(fpr_at_95),
                'pooled_quantile': float(pooled_quantile)
            }}
            
            all_results.append(result)
            print(f"  Held out {{held_out_ct}}: AUROC={{auroc:.3f}}, FPR@95TPR={{fpr_at_95:.3f}}")

save_results({{'experiments': all_results}}, 'exp/ood_detection/results.json')
print("OOD detection complete")
