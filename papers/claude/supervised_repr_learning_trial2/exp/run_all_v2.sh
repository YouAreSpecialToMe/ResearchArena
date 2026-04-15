#!/bin/bash
# Master training script for CCR experiments (v2)
# Estimated total: ~8 hours on 1x RTX A6000
# Priority order ensures essential results come first

set -e
cd "$(dirname "$0")/.."

TRAIN="python exp/train.py"
DATA_DIR="./data"
SEEDS=(42 123 456)
CIFAR_EPOCHS=200
TINY_EPOCHS=100

echo "============================================"
echo "CCR Experiment Suite v2 — $(date)"
echo "============================================"

# ===== PRIORITY 1: Core baselines + CCR on CIFAR-10/100 (24 runs, ~6.4 hrs) =====

# --- CIFAR-10 ---
for METHOD in ce label_smoothing mixup ccr_soft; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> ${METHOD} / cifar10 / seed=${SEED} — $(date '+%H:%M')"
        if [ "$METHOD" = "ccr_soft" ]; then
            $TRAIN --method $METHOD --dataset cifar10 --seed $SEED --epochs $CIFAR_EPOCHS \
                --lambda_ccr 0.1 --gamma 0.1 \
                --output_dir results/cifar10/${METHOD}/seed_${SEED} --data_dir $DATA_DIR
        else
            $TRAIN --method $METHOD --dataset cifar10 --seed $SEED --epochs $CIFAR_EPOCHS \
                --output_dir results/cifar10/${METHOD}/seed_${SEED} --data_dir $DATA_DIR
        fi
    done
done

# --- CIFAR-100 ---
for METHOD in ce label_smoothing mixup ccr_soft; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo ">>> ${METHOD} / cifar100 / seed=${SEED} — $(date '+%H:%M')"
        if [ "$METHOD" = "ccr_soft" ]; then
            $TRAIN --method $METHOD --dataset cifar100 --seed $SEED --epochs $CIFAR_EPOCHS \
                --lambda_ccr 0.1 --gamma 0.1 \
                --output_dir results/cifar100/${METHOD}/seed_${SEED} --data_dir $DATA_DIR
        else
            $TRAIN --method $METHOD --dataset cifar100 --seed $SEED --epochs $CIFAR_EPOCHS \
                --output_dir results/cifar100/${METHOD}/seed_${SEED} --data_dir $DATA_DIR
        fi
    done
done

echo ""
echo "=== PRIORITY 1 COMPLETE — $(date) ==="

# ===== PRIORITY 2: TinyImageNet (4 runs, ~1.2 hrs) =====
for METHOD in ce label_smoothing mixup ccr_soft; do
    echo ""
    echo ">>> ${METHOD} / tinyimagenet / seed=42 — $(date '+%H:%M')"
    if [ "$METHOD" = "ccr_soft" ]; then
        $TRAIN --method $METHOD --dataset tinyimagenet --seed 42 --epochs $TINY_EPOCHS \
            --lambda_ccr 0.1 --gamma 0.1 \
            --output_dir results/tinyimagenet/${METHOD}/seed_42 --data_dir $DATA_DIR
    else
        $TRAIN --method $METHOD --dataset tinyimagenet --seed 42 --epochs $TINY_EPOCHS \
            --output_dir results/tinyimagenet/${METHOD}/seed_42 --data_dir $DATA_DIR
    fi
done

echo ""
echo "=== PRIORITY 2 COMPLETE — $(date) ==="

# ===== PRIORITY 3: Key ablations on CIFAR-100, seed=42 (~1 hr) =====

# CCR-adaptive (original hinge, with lower lambda to avoid phase transition)
echo ""
echo ">>> ccr_adaptive / cifar100 / seed=42 — $(date '+%H:%M')"
$TRAIN --method ccr_adaptive --dataset cifar100 --seed 42 --epochs $CIFAR_EPOCHS \
    --lambda_ccr 0.05 --gamma 0.1 \
    --output_dir results/cifar100/ccr_adaptive/seed_42 --data_dir $DATA_DIR

# CCR-curriculum (hinge + linear warmup)
echo ""
echo ">>> ccr_curriculum / cifar100 / seed=42 — $(date '+%H:%M')"
$TRAIN --method ccr_curriculum --dataset cifar100 --seed 42 --epochs $CIFAR_EPOCHS \
    --lambda_ccr 0.1 --gamma 0.1 \
    --output_dir results/cifar100/ccr_curriculum/seed_42 --data_dir $DATA_DIR

# CCR-spectral
echo ""
echo ">>> ccr_spectral / cifar100 / seed=42 — $(date '+%H:%M')"
$TRAIN --method ccr_spectral --dataset cifar100 --seed 42 --epochs $CIFAR_EPOCHS \
    --lambda_ccr 0.1 \
    --output_dir results/cifar100/ccr_spectral/seed_42 --data_dir $DATA_DIR

# CCR-fixed
echo ""
echo ">>> ccr_fixed / cifar100 / seed=42 — $(date '+%H:%M')"
$TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs $CIFAR_EPOCHS \
    --lambda_ccr 0.1 --tau 5.0 \
    --output_dir results/cifar100/ccr_fixed/seed_42 --data_dir $DATA_DIR

echo ""
echo "=== PRIORITY 3 COMPLETE — $(date) ==="

# ===== PRIORITY 4: Lambda sweep for Pareto frontier (~30 min) =====
for LAMBDA in 0.01 0.5; do
    echo ""
    echo ">>> ccr_soft lambda=${LAMBDA} / cifar100 / seed=42 — $(date '+%H:%M')"
    $TRAIN --method ccr_soft --dataset cifar100 --seed 42 --epochs $CIFAR_EPOCHS \
        --lambda_ccr $LAMBDA --gamma 0.1 \
        --output_dir results/cifar100/ccr_soft_lambda_${LAMBDA}/seed_42 --data_dir $DATA_DIR
done

echo ""
echo "============================================"
echo "ALL TRAINING COMPLETE — $(date)"
echo "============================================"
