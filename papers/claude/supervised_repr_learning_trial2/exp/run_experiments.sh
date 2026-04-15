#!/bin/bash
# Optimized experiment runner with time monitoring
# Budget: 8 hours = 28800 seconds

set -e
cd "$(dirname "$0")/.."

TRAIN="python exp/train.py"
DATA_DIR="./data"
SEEDS=(42 123 456)
START_TIME=$(date +%s)
MAX_SECONDS=28000  # 7h47m, leaving buffer for analysis

elapsed() {
    echo $(( $(date +%s) - START_TIME ))
}

run_if_time() {
    local est_seconds=$1
    shift
    local e=$(elapsed)
    local remaining=$(( MAX_SECONDS - e ))
    if [ $remaining -lt $est_seconds ]; then
        echo "SKIP (need ${est_seconds}s, only ${remaining}s left): $@"
        return 1
    fi
    echo ""
    echo ">>> $(date '+%H:%M:%S') [$(( e/60 ))m elapsed, $(( remaining/60 ))m left] $@"
    "$@"
    return 0
}

echo "============================================"
echo "CCR Experiments — $(date)"
echo "Budget: $(( MAX_SECONDS/60 )) minutes"
echo "============================================"

# ===== PRIORITY 1: CIFAR-100 core experiments (200 epochs, ~28 min each) =====
# This is the primary dataset where CCR should show its benefit
echo ""
echo "=== PRIORITY 1: CIFAR-100 (CE + LS + CCR-soft × 3 seeds) ==="

for SEED in "${SEEDS[@]}"; do
    run_if_time 1700 $TRAIN --method ce --dataset cifar100 --seed $SEED --epochs 200 \
        --output_dir results/cifar100/ce/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1700 $TRAIN --method label_smoothing --dataset cifar100 --seed $SEED --epochs 200 \
        --output_dir results/cifar100/label_smoothing/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1800 $TRAIN --method ccr_soft --dataset cifar100 --seed $SEED --epochs 200 \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir results/cifar100/ccr_soft/seed_${SEED} --data_dir $DATA_DIR || true
done

echo ""
echo "=== PRIORITY 1 COMPLETE at $(( $(elapsed)/60 ))m ==="

# ===== PRIORITY 2: CIFAR-10 core experiments (200 epochs) =====
echo ""
echo "=== PRIORITY 2: CIFAR-10 (CE + LS + CCR-soft × 3 seeds) ==="

for SEED in "${SEEDS[@]}"; do
    run_if_time 1700 $TRAIN --method ce --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/ce/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1700 $TRAIN --method label_smoothing --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/label_smoothing/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1800 $TRAIN --method ccr_soft --dataset cifar10 --seed $SEED --epochs 200 \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir results/cifar10/ccr_soft/seed_${SEED} --data_dir $DATA_DIR || true
done

echo ""
echo "=== PRIORITY 2 COMPLETE at $(( $(elapsed)/60 ))m ==="

# ===== PRIORITY 3: Mixup baselines (3rd baseline) =====
echo ""
echo "=== PRIORITY 3: Mixup baselines ==="

for SEED in "${SEEDS[@]}"; do
    run_if_time 1700 $TRAIN --method mixup --dataset cifar100 --seed $SEED --epochs 200 \
        --output_dir results/cifar100/mixup/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1700 $TRAIN --method mixup --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/mixup/seed_${SEED} --data_dir $DATA_DIR || true
done

echo ""
echo "=== PRIORITY 3 COMPLETE at $(( $(elapsed)/60 ))m ==="

# ===== PRIORITY 4: TinyImageNet (3rd dataset) =====
echo ""
echo "=== PRIORITY 4: TinyImageNet ==="

run_if_time 1800 $TRAIN --method ce --dataset tinyimagenet --seed 42 --epochs 100 \
    --output_dir results/tinyimagenet/ce/seed_42 --data_dir $DATA_DIR || true

run_if_time 1800 $TRAIN --method label_smoothing --dataset tinyimagenet --seed 42 --epochs 100 \
    --output_dir results/tinyimagenet/label_smoothing/seed_42 --data_dir $DATA_DIR || true

run_if_time 1800 $TRAIN --method ccr_soft --dataset tinyimagenet --seed 42 --epochs 100 \
    --lambda_ccr 0.1 --gamma 0.1 \
    --output_dir results/tinyimagenet/ccr_soft/seed_42 --data_dir $DATA_DIR || true

echo ""
echo "=== PRIORITY 4 COMPLETE at $(( $(elapsed)/60 ))m ==="

# ===== PRIORITY 5: Key ablations on CIFAR-100 =====
echo ""
echo "=== PRIORITY 5: Ablations ==="

run_if_time 1800 $TRAIN --method ccr_adaptive --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.05 --gamma 0.1 \
    --output_dir results/cifar100/ccr_adaptive/seed_42 --data_dir $DATA_DIR || true

run_if_time 1800 $TRAIN --method ccr_curriculum --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --gamma 0.1 \
    --output_dir results/cifar100/ccr_curriculum/seed_42 --data_dir $DATA_DIR || true

run_if_time 1800 $TRAIN --method ccr_spectral --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 \
    --output_dir results/cifar100/ccr_spectral/seed_42 --data_dir $DATA_DIR || true

run_if_time 1800 $TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --tau 5.0 \
    --output_dir results/cifar100/ccr_fixed/seed_42 --data_dir $DATA_DIR || true

# Lambda sweep
for LAMBDA in 0.01 0.5; do
    run_if_time 1800 $TRAIN --method ccr_soft --dataset cifar100 --seed 42 --epochs 200 \
        --lambda_ccr $LAMBDA --gamma 0.1 \
        --output_dir results/cifar100/ccr_soft_lambda_${LAMBDA}/seed_42 --data_dir $DATA_DIR || true
done

echo ""
echo "============================================"
echo "EXPERIMENTS COMPLETE — $(date)"
echo "Total time: $(( $(elapsed)/60 )) minutes"
echo "============================================"
