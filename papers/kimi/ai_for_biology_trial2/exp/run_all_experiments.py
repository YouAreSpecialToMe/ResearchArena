#!/usr/bin/env python3
"""Run all experiments in sequence."""
import subprocess
import sys
import os
import json
import time

os.chdir('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01')

def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start
    
    print(f"Completed in {elapsed/60:.1f} minutes with return code {result.returncode}")
    return result.returncode == 0

# Step 1: Preprocess data
if not run_command(
    'source .venv/bin/activate && python exp/preprocess_data.py',
    "Data Preprocessing"
):
    print("ERROR: Data preprocessing failed!")
    sys.exit(1)

# Step 2: Run baselines
print("\n" + "="*60)
print("Running Baselines")
print("="*60)

# GENIE3 baseline
run_command(
    'source .venv/bin/activate && python exp/genie3_baseline/run.py',
    "GENIE3 Baseline"
)

# scMultiomeGRN baseline - 3 seeds
for seed in [42, 43, 44]:
    run_command(
        f'source .venv/bin/activate && python exp/scmultiomegrn_baseline/run.py --seed {seed} --output exp/scmultiomegrn_baseline/results_s{seed}.json',
        f"scMultiomeGRN Baseline (seed={seed})"
    )

# XATGRN baseline - 3 seeds
for seed in [42, 43, 44]:
    run_command(
        f'source .venv/bin/activate && python exp/xatgrn_baseline/run.py --seed {seed} --output exp/xatgrn_baseline/results_s{seed}.json',
        f"XATGRN Baseline (seed={seed})"
    )

# Step 3: Train CROSS-GRN main model - 3 seeds
print("\n" + "="*60)
print("Training CROSS-GRN Main Model")
print("="*60)

for seed in [42, 43, 44]:
    run_command(
        f'source .venv/bin/activate && python exp/crossgrn_main/train.py --seed {seed} --output exp/crossgrn_main/results_s{seed}.json --model_path models/crossgrn_s{seed}.pt',
        f"CROSS-GRN Main (seed={seed})"
    )

# Step 4: Run ablation studies
print("\n" + "="*60)
print("Running Ablation Studies")
print("="*60)

run_command(
    'source .venv/bin/activate && python exp/ablation_symmetric/run.py',
    "Ablation: Symmetric Attention"
)

run_command(
    'source .venv/bin/activate && python exp/ablation_no_celltype/run.py',
    "Ablation: No Cell-Type Conditioning"
)

run_command(
    'source .venv/bin/activate && python exp/ablation_no_sign/run.py',
    "Ablation: No Sign Prediction"
)

print("\n" + "="*60)
print("All experiments completed!")
print("="*60)
