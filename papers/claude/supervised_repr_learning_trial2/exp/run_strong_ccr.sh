#!/bin/bash
# Additional CCR runs with stronger regularization
# Goal: find settings where CCR actually activates and affects calibration
set -e
cd "$(dirname "$0")/.."

TRAIN="python exp/train.py"
DATA_DIR="./data"

echo "=== Strong CCR experiments — $(date) ==="

# CCR-fixed with tau matching actual spread scale
# CE spread at epoch 200 is ~12. tau=20 should prevent excessive collapse.
echo ">>> CCR-fixed tau=20, lambda=0.1"
$TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --tau 20.0 \
    --output_dir results/cifar100/ccr_fixed_tau20/seed_42 --data_dir $DATA_DIR

echo ">>> CCR-fixed tau=30, lambda=0.1"
$TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --tau 30.0 \
    --output_dir results/cifar100/ccr_fixed_tau30/seed_42 --data_dir $DATA_DIR

echo ">>> CCR-fixed tau=50, lambda=0.1"
$TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --tau 50.0 \
    --output_dir results/cifar100/ccr_fixed_tau50/seed_42 --data_dir $DATA_DIR

# CCR-soft with higher gamma (so adaptive tau is larger)
echo ">>> CCR-soft gamma=2.0, lambda=0.01"
$TRAIN --method ccr_soft --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.01 --gamma 2.0 \
    --output_dir results/cifar100/ccr_soft_gamma2/seed_42 --data_dir $DATA_DIR

echo ">>> CCR-soft gamma=5.0, lambda=0.005"
$TRAIN --method ccr_soft --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.005 --gamma 5.0 \
    --output_dir results/cifar100/ccr_soft_gamma5/seed_42 --data_dir $DATA_DIR

# If the best tau/gamma is found, run 3 seeds for it
# (will be determined after analyzing the sweep above)

echo "=== Strong CCR complete — $(date) ==="
