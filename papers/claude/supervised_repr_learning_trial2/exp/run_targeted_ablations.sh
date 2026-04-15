#!/bin/bash
# Targeted ablation: CCR experiments that actually activate the regularizer
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# CCR-fixed with high tau (tau=30, which is above CE's final spread of 25.6)
# This should maintain spread above 30 while training normally
echo "=== CCR-fixed, tau=30, lambda=0.01 (CIFAR-100, seed 42) ==="
python exp/train.py \
    --method ccr_fixed \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 42 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --lambda_ccr 0.01 \
    --tau 30.0 \
    --output_dir results/cifar100/ccr_fixed_tau30/seed_42 \
    --data_dir ./data

# CCR-fixed with moderate tau (tau=15, gentler constraint)
echo "=== CCR-fixed, tau=15, lambda=0.01 (CIFAR-100, seed 42) ==="
python exp/train.py \
    --method ccr_fixed \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 42 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --lambda_ccr 0.01 \
    --tau 15.0 \
    --output_dir results/cifar100/ccr_fixed_tau15/seed_42 \
    --data_dir ./data

# CCR-adaptive with high gamma (gamma=1.0) to actually activate the adaptive threshold
echo "=== CCR-adaptive, gamma=1.0, lambda=0.01 (CIFAR-100, seed 42) ==="
python exp/train.py \
    --method ccr_adaptive \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 42 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --lambda_ccr 0.01 \
    --gamma 1.0 \
    --output_dir results/cifar100/ccr_adaptive_gamma_1.0/seed_42 \
    --data_dir ./data

# CCR-adaptive with moderate gamma (gamma=0.5)
echo "=== CCR-adaptive, gamma=0.5, lambda=0.01 (CIFAR-100, seed 42) ==="
python exp/train.py \
    --method ccr_adaptive \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 42 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --lambda_ccr 0.01 \
    --gamma 0.5 \
    --output_dir results/cifar100/ccr_adaptive_gamma_0.5/seed_42 \
    --data_dir ./data

echo "=== All targeted ablation training complete ==="
