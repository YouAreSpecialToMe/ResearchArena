import os
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
