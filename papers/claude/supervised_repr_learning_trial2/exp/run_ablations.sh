#!/bin/bash
# Run focused ablation experiments on CIFAR-100, seed 42
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Lambda sweep with lower values (3 key points)
for LAMBDA in 0.001 0.01 0.05; do
    echo "=== Lambda sweep: $LAMBDA (CIFAR-100, seed 42) ==="
    python exp/train.py \
        --method ccr_adaptive \
        --dataset cifar100 \
        --arch resnet18 \
        --seed 42 \
        --epochs 100 \
        --batch_size 256 \
        --lr 0.1 \
        --lambda_ccr $LAMBDA \
        --gamma 0.1 \
        --output_dir results/cifar100/ccr_adaptive_lambda_${LAMBDA}/seed_42 \
        --data_dir ./data
done

# CCR-fixed ablation (using lambda=0.01 which is more reasonable)
echo "=== CCR-fixed ablation (CIFAR-100, seed 42, lambda=0.01) ==="
python exp/train.py \
    --method ccr_fixed \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 42 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --lambda_ccr 0.01 \
    --tau 1.0 \
    --output_dir results/cifar100/ccr_fixed/seed_42 \
    --data_dir ./data

# CCR-spectral ablation
echo "=== CCR-spectral ablation (CIFAR-100, seed 42, lambda=0.01) ==="
python exp/train.py \
    --method ccr_spectral \
    --dataset cifar100 \
    --arch resnet18 \
    --seed 42 \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.1 \
    --lambda_ccr 0.01 \
    --output_dir results/cifar100/ccr_spectral/seed_42 \
    --data_dir ./data

echo "=== All ablation training complete ==="
