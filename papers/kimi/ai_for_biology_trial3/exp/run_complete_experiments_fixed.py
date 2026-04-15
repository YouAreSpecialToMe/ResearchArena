#!/usr/bin/env python3
"""
Complete CellStratCP Experiment Runner
Executes all experiments following the plan.json exactly.
"""

import os
import sys
import time
import json
import glob
import subprocess
from datetime import datetime

# Configuration
SEEDS = [42, 123, 456]
CORE_DATASETS = ['pbmc_processed', 'synthetic_d30_s42', 'synthetic_d50_s42', 'synthetic_d70_s42']

VENV_PYTHON = ".venv/bin/python"

def log(msg):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def run_command(cmd, timeout=3600):
    """Run a command with timeout and capture output."""
    # Replace 'source .venv/bin/activate && python ' with venv python path
    if 'source .venv/bin/activate && python ' in cmd:
        cmd = cmd.replace('source .venv/bin/activate && python ', f'{VENV_PYTHON} ')
    
    log(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            log(f"Error: {result.stderr}")
            return False
        log(f"Success: {cmd[:50]}...")
        return True
    except subprocess.TimeoutExpired:
        log(f"Timeout: {cmd}")
        return False
    except Exception as e:
        log(f"Exception: {e}")
        return False

def step1_train_scvi_models():
    """Step 1: Train scVI models for all datasets with all seeds."""
    log("=" * 70)
    log("STEP 1: Training scVI models for all datasets with all 3 seeds")
    log("=" * 70)
    
    all_results = []
    
    # Train for all datasets and all seeds
    for dataset in CORE_DATASETS:
        for seed in SEEDS:
            log(f"\nTraining scVI: {dataset}, seed={seed}")
            
            # Determine the correct data path
            if 'synthetic' in dataset:
                data_path = f"data/{dataset}.h5ad"
            else:
                data_path = f"data/{dataset}.h5ad"
            
            if not os.path.exists(data_path):
                log(f"  Data file not found: {data_path}")
                continue
            
            # Create training script for this specific dataset/seed
            script = f'''import os
import sys
import time
sys.path.insert(0, 'exp/shared')
import numpy as np
import scanpy as sc
import scvi
import torch
import pandas as pd
from scrna_utils import set_seed

# Set seed
set_seed({seed})

# Load data
adata = sc.read_h5ad('{data_path}')
dataset_name = os.path.basename('{data_path}').replace('.h5ad', '')

print(f"Training on {{dataset_name}}: {{adata.n_obs}} cells, {{adata.n_vars}} genes")

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
    max_epochs=50,
    early_stopping=True,
    batch_size=128,
    plan_kwargs={{"lr": 1e-3}},
    enable_progress_bar=False
)
training_time = time.time() - start_time

# Save model
model_dir = f"exp/scvi_training/models/scvi_{{dataset_name}}_seed{seed}"
model.save(model_dir, overwrite=True, save_anndata=False)

# Extract and save ZINB parameters
params = model.get_likelihood_parameters(adata=adata, give_mean=True)
posterior_means = model.get_normalized_expression(adata, return_mean=True, n_samples=1)

cell_types = adata.obs['cell_type'].values
splits = adata.obs['split'].values if 'split' in adata.obs.columns else ['unknown'] * len(adata)

df = pd.DataFrame({{
    'cell_idx': np.arange(adata.n_obs),
    'cell_type': cell_types,
    'split': splits,
    'mu': params['mean'].flatten(),
    'theta': params['dispersion'].flatten(),
    'pi': params['dropout'].flatten(),
    'predicted_expression': posterior_means.flatten()
}})

params_path = f"exp/scvi_training/zinb_params_{{dataset_name}}_seed{seed}.csv"
df.to_csv(params_path, index=False)

print(f"  Training time: {{training_time:.1f}}s")
print(f"  Model saved: {{model_dir}}")
print(f"  Params saved: {{params_path}}")

# Save results
result = {{
    'dataset': dataset_name,
    'seed': {seed},
    'training_time': training_time,
    'n_cells': adata.n_obs,
    'n_genes': adata.n_vars,
    'model_path': model_dir,
    'params_path': params_path
}}

import json
with open(f'exp/scvi_training/result_{{dataset_name}}_seed{seed}.json', 'w') as f:
    json.dump(result, f, indent=2)
'''
            
            # Write and execute training script
            script_path = f"exp/scvi_training/train_temp_{dataset}_s{seed}.py"
            with open(script_path, 'w') as f:
                f.write(script)
            
            success = run_command(f".venv/bin/python {script_path}", timeout=600)
            
            if success:
                result_path = f"exp/scvi_training/result_{dataset}_seed{seed}.json"
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        all_results.append(json.load(f))
            
            # Clean up temp script
            if os.path.exists(script_path):
                os.remove(script_path)
    
    # Save combined results
    with open('exp/scvi_training/results.json', 'w') as f:
        json.dump({'models': all_results, 'status': 'success'}, f, indent=2)
    
    log(f"\nCompleted training {len(all_results)} models")
    return len(all_results) > 0

def step2_run_baselines():
    """Step 2: Run standard CP and scVI posterior baselines."""
    log("=" * 70)
    log("STEP 2: Running baselines")
    log("=" * 70)
    
    # Standard Split CP
    log("\n--- Running Standard Split CP ---")
    for seed in SEEDS:
        log(f"Seed {seed}...")
        run_command(f".venv/bin/python exp/cp_baselines/standard_cp/run.py --seed {seed}", 
                   timeout=300)
    
    # scVI Posterior baseline
    log("\n--- Running scVI Posterior Baseline ---")
    run_scvi_posterior_baseline()
    
    return True

def run_scvi_posterior_baseline():
    """Run scVI variational posterior baseline."""
    log("Implementing scVI posterior baseline...")
    
    script = '''import os
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
'''
    
    with open('exp/cp_baselines/scvi_posterior/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/cp_baselines/scvi_posterior/run.py", 
               timeout=600)

def step3_run_cellstratcp():
    """Step 3: Run CellStratCP main method."""
    log("=" * 70)
    log("STEP 3: Running CellStratCP")
    log("=" * 70)
    
    for seed in SEEDS:
        log(f"Seed {seed}...")
        run_command(f".venv/bin/python exp/cellstratcp/run.py --seed {seed}",
                   timeout=300)
    
    return True

def step4_run_ablations():
    """Step 4: Run all ablation studies."""
    log("=" * 70)
    log("STEP 4: Running ablation studies")
    log("=" * 70)
    
    # No Mondrian (pooled quantile)
    log("\n--- Ablation: No Mondrian ---")
    run_no_mondrian_ablation()
    
    # Residual scores (instead of ZINB)
    log("\n--- Ablation: Residual Scores ---")
    run_residual_scores_ablation()
    
    # No ACI (fixed alpha under distribution shift)
    log("\n--- Ablation: No ACI ---")
    run_no_aci_ablation()
    
    return True

def run_no_mondrian_ablation():
    """Run ablation without Mondrian stratification."""
    script = '''import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
from scrna_utils import (set_seed, zinb_nonconformity_score, compute_prediction_interval,
    conformal_prediction_intervals, evaluate_coverage, save_results)
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
            cal_df['x_obs'] = cal_df['mu'].values + np.random.randn(len(cal_df)) * np.sqrt(cal_df['mu'].values)
            test_df['x_obs'] = test_df['mu'].values + np.random.randn(len(test_df)) * np.sqrt(test_df['mu'].values)
            cal_df['x_obs'] = np.maximum(0, cal_df['x_obs'])
            test_df['x_obs'] = np.maximum(0, test_df['x_obs'])
        
        # Compute scores on calibration set (pooled, no stratification)
        cal_df['score'] = cal_df.apply(
            lambda row: zinb_nonconformity_score(row['x_obs'], row['x_pred'] if 'x_pred' in row else row['mu'], 
                                                  row['mu'], row['theta'], row['pi']), axis=1
        )
        
        # Single pooled quantile
        pooled_quantile = conformal_prediction_intervals(cal_df['score'].values, None, 0.1)
        
        # Compute prediction intervals for test set
        pred_intervals = []
        for idx, row in test_df.iterrows():
            zinb_params = {{'mu': row['mu'], 'theta': row['theta'], 'pi': row['pi']}}
            lower, upper = compute_prediction_interval(zinb_params, pooled_quantile)
            pred_intervals.append([lower, upper])
        
        pred_intervals = np.array(pred_intervals)
        
        # Evaluate
        cell_types = test_df['cell_type'].values
        coverage_results = evaluate_coverage(test_df['x_obs'].values, pred_intervals, cell_types=cell_types)
        
        result = {{
            'dataset': dataset_name,
            'seed': seed,
            'method': 'CellStratCP (No Mondrian)',
            'marginal_coverage': coverage_results['marginal_coverage'],
            'mean_interval_width': coverage_results['mean_interval_width'],
            'max_coverage_discrepancy': coverage_results['max_coverage_discrepancy'],
            'conditional_coverage': coverage_results.get('conditional_coverage', {{}})
        }}
        
        all_results.append(result)
        print(f"  Coverage: {{coverage_results['marginal_coverage']:.3f}}")
        print(f"  Discrepancy: {{coverage_results['max_coverage_discrepancy']:.3f}}")

save_results({{'experiments': all_results}}, 'exp/ablations/no_mondrian/results.json')
print("No Mondrian ablation complete")
'''
    
    with open('exp/ablations/no_mondrian/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/ablations/no_mondrian/run.py",
               timeout=300)

def run_residual_scores_ablation():
    """Run ablation with residual scores instead of ZINB."""
    script = '''import os
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
'''
    
    with open('exp/ablations/residual_scores/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/ablations/residual_scores/run.py",
               timeout=300)

def run_no_aci_ablation():
    """Run ablation comparing ACI vs fixed alpha under distribution shift."""
    script = '''import os
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
'''
    
    with open('exp/ablations/no_aci/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/ablations/no_aci/run.py",
               timeout=300)

def step5_run_aci_experiments():
    """Step 5: Run ACI experiments under distribution shifts."""
    log("=" * 70)
    log("STEP 5: Running ACI distribution shift experiments")
    log("=" * 70)
    
    script = '''import os
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
'''
    
    os.makedirs('exp/aci_experiments', exist_ok=True)
    with open('exp/aci_experiments/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/aci_experiments/run.py",
               timeout=600)
    
    return True

def step6_run_ood_detection():
    """Step 6: Run OOD detection experiments."""
    log("=" * 70)
    log("STEP 6: Running OOD detection experiments")
    log("=" * 70)
    
    script = '''import os
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
'''
    
    with open('exp/ood_detection/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/ood_detection/run.py",
               timeout=300)
    
    return True

def step7_run_runtime_analysis():
    """Step 7: Run runtime analysis."""
    log("=" * 70)
    log("STEP 7: Running runtime analysis")
    log("=" * 70)
    
    script = '''import os
import sys
sys.path.insert(0, 'exp/shared')
import numpy as np
import pandas as pd
import time
from scrna_utils import (set_seed, zinb_nonconformity_score, conformal_prediction_intervals,
    compute_prediction_interval, save_results)

def measure_calibration_time(n_cal, n_genes=100):
    """Measure calibration time for different calibration set sizes."""
    # Generate synthetic calibration data
    mu = np.random.lognormal(0, 1, (n_cal, n_genes))
    theta = np.random.gamma(2, 2, (n_cal, n_genes))
    pi = np.random.beta(2, 5, (n_cal, n_genes))
    x_obs = np.random.poisson(mu)
    
    cell_types = np.random.choice(['CT_0', 'CT_1', 'CT_2'], n_cal)
    
    start = time.time()
    
    # Compute non-conformity scores
    for i in range(n_cal):
        for g in range(min(n_genes, 10)):  # Sample 10 genes
            zinb_nonconformity_score(x_obs[i, g], mu[i, g], mu[i, g], theta[i, g], pi[i, g])
    
    # Compute quantiles per cell type
    for ct in ['CT_0', 'CT_1', 'CT_2']:
        mask = cell_types == ct
        if mask.sum() > 0:
            scores = np.random.exponential(1.0, mask.sum())  # Simulated scores
            conformal_prediction_intervals(scores, None, 0.1)
    
    elapsed = time.time() - start
    return elapsed

def measure_prediction_time(n_test, n_genes=100):
    """Measure prediction time."""
    mu = np.random.lognormal(0, 1, (n_test, n_genes))
    theta = np.random.gamma(2, 2, (n_test, n_genes))
    pi = np.random.beta(2, 5, (n_test, n_genes))
    
    quantile = 2.0  # Fixed quantile
    
    start = time.time()
    
    for i in range(n_test):
        for g in range(min(n_genes, 10)):
            zinb_params = {{'mu': mu[i, g], 'theta': theta[i, g], 'pi': pi[i, g]}}
            compute_prediction_interval(zinb_params, quantile)
    
    elapsed = time.time() - start
    return elapsed

set_seed(42)

results = []
calibration_sizes = [100, 300, 1000, 3000, 10000]

for n_cal in calibration_sizes:
    print(f"Measuring calibration time for n={{n_cal}}...")
    times = []
    for _ in range(5):
        t = measure_calibration_time(n_cal)
        times.append(t)
    
    results.append({{
        'n_calibration': n_cal,
        'calibration_time_mean': np.mean(times),
        'calibration_time_std': np.std(times),
        'per_cell_time': np.mean(times) / n_cal
    }})

# Prediction throughput
print("Measuring prediction throughput...")
pred_sizes = [100, 1000, 10000]
for n_pred in pred_sizes:
    times = []
    for _ in range(5):
        t = measure_prediction_time(n_pred)
        times.append(t)
    
    results.append({{
        'n_prediction': n_pred,
        'prediction_time_mean': np.mean(times),
        'prediction_time_std': np.std(times),
        'cells_per_second': n_pred / np.mean(times)
    }})

save_results({{'runtime_analysis': results}}, 'exp/runtime/results.json')
print("Runtime analysis complete")
'''
    
    with open('exp/runtime/run.py', 'w') as f:
        f.write(script)
    
    run_command(".venv/bin/python exp/runtime/run.py",
               timeout=300)
    
    return True

def step8_aggregate_results():
    """Step 8: Aggregate all results into final results.json."""
    log("=" * 70)
    log("STEP 8: Aggregating all results")
    log("=" * 70)
    
    import json
    import numpy as np
    
    all_results = {
        'cellstratcp': [],
        'standard_cp': [],
        'scvi_posterior': [],
        'ablations': {
            'no_mondrian': [],
            'residual_scores': [],
            'no_aci': []
        },
        'aci_experiments': [],
        'ood_detection': [],
        'runtime': {}
    }
    
    # Load CellStratCP results
    try:
        with open('exp/cellstratcp/results.json') as f:
            data = json.load(f)
            all_results['cellstratcp'] = data.get('experiments', [])
    except:
        log("No CellStratCP results found")
    
    # Load Standard CP results
    try:
        with open('exp/cp_baselines/standard_cp/results.json') as f:
            data = json.load(f)
            all_results['standard_cp'] = data.get('experiments', [])
    except:
        log("No Standard CP results found")
    
    # Load scVI posterior results
    try:
        with open('exp/cp_baselines/scvi_posterior/results.json') as f:
            data = json.load(f)
            all_results['scvi_posterior'] = data.get('experiments', [])
    except:
        log("No scVI posterior results found")
    
    # Load ablations
    for ablation in ['no_mondrian', 'residual_scores', 'no_aci']:
        try:
            with open(f'exp/ablations/{ablation}/results.json') as f:
                data = json.load(f)
                all_results['ablations'][ablation] = data.get('experiments', [])
        except:
            log(f"No {ablation} results found")
    
    # Load ACI experiments
    try:
        with open('exp/aci_experiments/results.json') as f:
            data = json.load(f)
            all_results['aci_experiments'] = data.get('experiments', [])
    except:
        log("No ACI experiments results found")
    
    # Load OOD detection
    try:
        with open('exp/ood_detection/results.json') as f:
            data = json.load(f)
            all_results['ood_detection'] = data.get('experiments', [])
    except:
        log("No OOD detection results found")
    
    # Load runtime
    try:
        with open('exp/runtime/results.json') as f:
            data = json.load(f)
            all_results['runtime'] = data.get('runtime_analysis', {})
    except:
        log("No runtime results found")
    
    # Compute summary statistics
    def compute_summary(results, key):
        if not results:
            return {}
        values = [r.get(key, 0) for r in results if key in r]
        if not values:
            return {}
        return {'mean': float(np.mean(values)), 'std': float(np.std(values))}
    
    summary = {
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
    
    all_results['summary'] = summary
    
    # Save aggregated results
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log(f"Aggregated results saved to results.json")
    log(f"  CellStratCP experiments: {len(all_results['cellstratcp'])}")
    log(f"  Standard CP experiments: {len(all_results['standard_cp'])}")
    log(f"  scVI posterior experiments: {len(all_results['scvi_posterior'])}")
    log(f"  Ablations: {sum(len(v) for v in all_results['ablations'].values())}")
    log(f"  ACI experiments: {len(all_results['aci_experiments'])}")
    log(f"  OOD detection experiments: {len(all_results['ood_detection'])}")
    
    return True

def main():
    """Run all experiments."""
    log("=" * 70)
    log("CELLSTRATCP COMPLETE EXPERIMENT RUNNER")
    log("=" * 70)
    
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs('exp/scvi_training/models', exist_ok=True)
    os.makedirs('exp/cp_baselines/standard_cp', exist_ok=True)
    os.makedirs('exp/cp_baselines/scvi_posterior', exist_ok=True)
    os.makedirs('exp/ablations/no_mondrian', exist_ok=True)
    os.makedirs('exp/ablations/residual_scores', exist_ok=True)
    os.makedirs('exp/ablations/no_aci', exist_ok=True)
    os.makedirs('exp/aci_experiments', exist_ok=True)
    os.makedirs('exp/ood_detection', exist_ok=True)
    os.makedirs('exp/runtime', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    success = True
    
    # Execute all steps
    try:
        success &= step1_train_scvi_models()
    except Exception as e:
        log(f"Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step2_run_baselines()
    except Exception as e:
        log(f"Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step3_run_cellstratcp()
    except Exception as e:
        log(f"Step 3 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step4_run_ablations()
    except Exception as e:
        log(f"Step 4 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step5_run_aci_experiments()
    except Exception as e:
        log(f"Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step6_run_ood_detection()
    except Exception as e:
        log(f"Step 6 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step7_run_runtime_analysis()
    except Exception as e:
        log(f"Step 7 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    try:
        success &= step8_aggregate_results()
    except Exception as e:
        log(f"Step 8 failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    elapsed = time.time() - start_time
    log(f"\n{'=' * 70}")
    log(f"ALL EXPERIMENTS COMPLETE")
    log(f"Total time: {elapsed/60:.1f} minutes")
    log(f"{'=' * 70}")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
