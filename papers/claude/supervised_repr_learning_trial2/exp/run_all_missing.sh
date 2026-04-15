#!/bin/bash
# Master script to run all missing experiments
# Priority order: CIFAR-100 ccr_adaptive > CIFAR-100 mixup > CIFAR-10 ccr_adaptive >
#   CIFAR-10 mixup > CIFAR-10 ccr_fixed remaining > CIFAR-10 ccr_soft >
#   CIFAR-100 ccr_spectral > TinyImageNet

set -e
cd "$(dirname "$0")/.."

TRAIN="python exp/train.py"
COMMON="--arch resnet18 --epochs 200 --batch_size 256 --lr 0.1 --data_dir ./data"

echo "========================================="
echo "Starting all missing experiments"
echo "Time: $(date)"
echo "========================================="

# --- CIFAR-100 ccr_adaptive (primary method, 3 seeds) ---
echo ""
echo "=== CIFAR-100 ccr_adaptive ==="
for SEED in 42 123 456; do
    DIR="results/cifar100/ccr_adaptive/seed_${SEED}"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    echo "Running ccr_adaptive CIFAR-100 seed=${SEED} [$(date)]"
    $TRAIN --method ccr_adaptive --dataset cifar100 --seed $SEED \
        --lambda_ccr 0.1 --gamma 0.1 --tau 1.0 \
        --output_dir $DIR $COMMON
done

# --- CIFAR-100 mixup (missing baseline, 3 seeds) ---
echo ""
echo "=== CIFAR-100 mixup ==="
for SEED in 42 123 456; do
    DIR="results/cifar100/mixup/seed_${SEED}"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    echo "Running mixup CIFAR-100 seed=${SEED} [$(date)]"
    $TRAIN --method mixup --dataset cifar100 --seed $SEED \
        --mixup_alpha 0.2 \
        --output_dir $DIR $COMMON
done

# --- CIFAR-10 ccr_adaptive (primary method, 3 seeds) ---
echo ""
echo "=== CIFAR-10 ccr_adaptive ==="
for SEED in 42 123 456; do
    DIR="results/cifar10/ccr_adaptive/seed_${SEED}"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    echo "Running ccr_adaptive CIFAR-10 seed=${SEED} [$(date)]"
    $TRAIN --method ccr_adaptive --dataset cifar10 --seed $SEED \
        --lambda_ccr 0.1 --gamma 0.1 --tau 1.0 \
        --output_dir $DIR $COMMON
done

# --- CIFAR-10 mixup (missing baseline, 3 seeds) ---
echo ""
echo "=== CIFAR-10 mixup ==="
for SEED in 42 123 456; do
    DIR="results/cifar10/mixup/seed_${SEED}"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    echo "Running mixup CIFAR-10 seed=${SEED} [$(date)]"
    $TRAIN --method mixup --dataset cifar10 --seed $SEED \
        --mixup_alpha 0.2 \
        --output_dir $DIR $COMMON
done

# --- CIFAR-10 ccr_fixed remaining seeds ---
echo ""
echo "=== CIFAR-10 ccr_fixed remaining ==="
for SEED in 123 456; do
    DIR="results/cifar10/ccr_fixed_tau15/seed_${SEED}"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    echo "Running ccr_fixed CIFAR-10 seed=${SEED} [$(date)]"
    $TRAIN --method ccr_fixed --dataset cifar10 --seed $SEED \
        --lambda_ccr 0.1 --tau 15.0 --gamma 0.1 \
        --output_dir $DIR $COMMON
done

# --- CIFAR-10 ccr_soft (3 seeds) ---
echo ""
echo "=== CIFAR-10 ccr_soft ==="
for SEED in 42 123 456; do
    DIR="results/cifar10/ccr_soft/seed_${SEED}"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    echo "Running ccr_soft CIFAR-10 seed=${SEED} [$(date)]"
    $TRAIN --method ccr_soft --dataset cifar10 --seed $SEED \
        --lambda_ccr 0.1 --gamma 0.1 --tau 1.0 \
        --output_dir $DIR $COMMON
done

# --- CIFAR-100 ccr_spectral ablation (1 seed) ---
echo ""
echo "=== CIFAR-100 ccr_spectral ==="
DIR="results/cifar100/ccr_spectral/seed_42"
if [ -f "${DIR}/metrics.json" ]; then
    echo "SKIP: ${DIR} already has metrics.json"
else
    echo "Running ccr_spectral CIFAR-100 seed=42 [$(date)]"
    $TRAIN --method ccr_spectral --dataset cifar100 --seed 42 \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir $DIR $COMMON
fi

# --- TinyImageNet (CE, label_smoothing, mixup, ccr_adaptive, seed 42, 100 epochs) ---
echo ""
echo "=== TinyImageNet experiments ==="
TINY_COMMON="--arch resnet18 --epochs 100 --batch_size 256 --lr 0.1 --data_dir ./data"

for METHOD in ce label_smoothing mixup ccr_adaptive; do
    DIR="results/tinyimagenet/${METHOD}/seed_42"
    if [ -f "${DIR}/metrics.json" ]; then
        echo "SKIP: ${DIR} already has metrics.json"
        continue
    fi
    EXTRA=""
    if [ "$METHOD" = "ccr_adaptive" ]; then
        EXTRA="--lambda_ccr 0.1 --gamma 0.1 --tau 1.0"
    elif [ "$METHOD" = "mixup" ]; then
        EXTRA="--mixup_alpha 0.2"
    fi
    echo "Running ${METHOD} TinyImageNet seed=42 [$(date)]"
    $TRAIN --method $METHOD --dataset tinyimagenet --seed 42 \
        $EXTRA --output_dir $DIR $TINY_COMMON
done

echo ""
echo "========================================="
echo "All experiments completed!"
echo "Time: $(date)"
echo "========================================="
