#!/bin/bash
# Comprehensive experiment runner for FedSecure-CL

set -e

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/privacy_in_machine_learning/idea_01
source .venv/bin/activate

echo "Starting FedSecure-CL Experiments at $(date)"
echo "================================================"

# Create directories
mkdir -p results/models results/logs results/attacks figures

# Run experiments sequentially to avoid memory issues

# 1. Baseline FCL - CIFAR-10 (3 seeds)
echo "Running Baseline FCL CIFAR-10..."
python exp/baseline_fcl/run.py --seed 42 --dataset cifar10 2>&1 | tee results/logs/baseline_fcl_cifar10_seed42.log
python exp/baseline_fcl/run.py --seed 123 --dataset cifar10 2>&1 | tee results/logs/baseline_fcl_cifar10_seed123.log
python exp/baseline_fcl/run.py --seed 456 --dataset cifar10 2>&1 | tee results/logs/baseline_fcl_cifar10_seed456.log

# 2. Baseline FCL-AT - CIFAR-10 (3 seeds)
echo "Running Baseline FCL-AT CIFAR-10..."
python exp/baseline_fcl_at/run.py --seed 42 --dataset cifar10 2>&1 | tee results/logs/baseline_fcl_at_cifar10_seed42.log
python exp/baseline_fcl_at/run.py --seed 123 --dataset cifar10 2>&1 | tee results/logs/baseline_fcl_at_cifar10_seed123.log
python exp/baseline_fcl_at/run.py --seed 456 --dataset cifar10 2>&1 | tee results/logs/baseline_fcl_at_cifar10_seed456.log

# 3. Baseline FCL-DP - CIFAR-10 (3 seeds)
echo "Running Baseline FCL-DP CIFAR-10..."
python exp/baseline_fcl_dp/run.py --seed 42 2>&1 | tee results/logs/baseline_fcl_dp_seed42.log
python exp/baseline_fcl_dp/run.py --seed 123 2>&1 | tee results/logs/baseline_fcl_dp_seed123.log
python exp/baseline_fcl_dp/run.py --seed 456 2>&1 | tee results/logs/baseline_fcl_dp_seed456.log

# 4. FedSecure-CL - CIFAR-10 (3 seeds)
echo "Running FedSecure-CL CIFAR-10..."
python exp/fedsecure_cl/run.py --seed 42 --dataset cifar10 --ablation none 2>&1 | tee results/logs/fedsecure_cl_cifar10_seed42.log
python exp/fedsecure_cl/run.py --seed 123 --dataset cifar10 --ablation none 2>&1 | tee results/logs/fedsecure_cl_cifar10_seed123.log
python exp/fedsecure_cl/run.py --seed 456 --dataset cifar10 --ablation none 2>&1 | tee results/logs/fedsecure_cl_cifar10_seed456.log

# 5. Ablation studies (2 seeds each)
echo "Running Ablation Studies..."
python exp/fedsecure_cl/run.py --seed 42 --dataset cifar10 --ablation no_privacy 2>&1 | tee results/logs/ablation_no_privacy_seed42.log
python exp/fedsecure_cl/run.py --seed 123 --dataset cifar10 --ablation no_privacy 2>&1 | tee results/logs/ablation_no_privacy_seed123.log

python exp/fedsecure_cl/run.py --seed 42 --dataset cifar10 --ablation no_grad_noise 2>&1 | tee results/logs/ablation_no_grad_noise_seed42.log
python exp/fedsecure_cl/run.py --seed 123 --dataset cifar10 --ablation no_grad_noise 2>&1 | tee results/logs/ablation_no_grad_noise_seed123.log

python exp/fedsecure_cl/run.py --seed 42 --dataset cifar10 --ablation no_adv 2>&1 | tee results/logs/ablation_no_adv_seed42.log
python exp/fedsecure_cl/run.py --seed 123 --dataset cifar10 --ablation no_adv 2>&1 | tee results/logs/ablation_no_adv_seed123.log

# 6. CIFAR-100 validation (1 seed each)
echo "Running CIFAR-100 experiments..."
python exp/baseline_fcl/run.py --seed 42 --dataset cifar100 2>&1 | tee results/logs/baseline_fcl_cifar100_seed42.log
python exp/baseline_fcl_at/run.py --seed 42 --dataset cifar100 2>&1 | tee results/logs/baseline_fcl_at_cifar100_seed42.log
python exp/fedsecure_cl/run.py --seed 42 --dataset cifar100 --ablation none 2>&1 | tee results/logs/fedsecure_cl_cifar100_seed42.log

echo "================================================"
echo "All experiments completed at $(date)"

# Generate visualizations
echo "Generating visualizations..."
python exp/shared/visualize_results.py

echo "Done!"
