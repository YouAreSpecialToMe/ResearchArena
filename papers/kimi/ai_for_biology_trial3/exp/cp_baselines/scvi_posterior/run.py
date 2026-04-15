import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scrna_utils import set_seed, evaluate_coverage, save_results
import glob
import time

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
        test_df = df[df['split'] == 'test'].copy()
        
        # Load original data for observed counts
        data_path = f'data/{{dataset_name}}.h5ad'
        if not os.path.exists(data_path):
            continue
        
        adata = sc.read_h5ad(data_path)
        
        # Load trained model
        model_path = f'exp/scvi_training/models/scvi_{{dataset_name}}_seed{{seed}}'
        if not os.path.exists(model_path):
            continue
        
        model = scvi.model.SCVI.load(model_path, adata=adata)
        
        # Sample from posterior for uncertainty
        start_time = time.time()
        
        test_indices = test_df['cell_idx'].values
        adata_test = adata[test_indices]
        
        # Get posterior samples
        n_samples = 100
        samples = model.posterior_predictive_sample(adata_test, n_samples=n_samples)
        
        # Use first gene for evaluation
        gene_samples = samples[:, 0, :]
        
        # Compute 90% prediction intervals
        lowers = np.percentile(gene_samples, 5, axis=1)
        uppers = np.percentile(gene_samples, 95, axis=1)
        pred_intervals = np.column_stack([lowers, uppers])
        
        # Get observed values
        y_obs = adata_test.X[:, 0].toarray().flatten()
        
        runtime = time.time() - start_time
        
        # Evaluate
        cell_types = test_df['cell_type'].values
        coverage_results = evaluate_coverage(y_obs, pred_intervals, cell_types=cell_types)
        
        result = {{
            'dataset': dataset_name,
            'seed': seed,
            'method': 'scVI Posterior',
            'marginal_coverage': coverage_results['marginal_coverage'],
            'mean_interval_width': coverage_results['mean_interval_width'],
            'max_coverage_discrepancy': coverage_results['max_coverage_discrepancy'],
            'conditional_coverage': coverage_results.get('conditional_coverage', {{}}),
            'runtime': runtime
        }}
        
        all_results.append(result)
        print(f"  Coverage: {{coverage_results['marginal_coverage']:.3f}}")
        print(f"  Width: {{coverage_results['mean_interval_width']:.3f}}")

save_results({{'experiments': all_results}}, 'exp/cp_baselines/scvi_posterior/results.json')
print("scVI Posterior baseline complete")
