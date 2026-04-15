#!/bin/bash
# Run remaining experiments after CIFAR-100 CE+LS baselines are done
set -e
cd "$(dirname "$0")/.."

TRAIN="python exp/train.py"
DATA_DIR="./data"
SEEDS=(42 123 456)
START_TIME=$(date +%s)
MAX_SECONDS=21600  # 6 hours

elapsed() { echo $(( $(date +%s) - START_TIME )); }
remaining() { echo $(( MAX_SECONDS - $(elapsed) )); }

run_if_time() {
    local est=$1; shift
    local r=$(remaining)
    if [ $r -lt $est ]; then
        echo "SKIP (need ${est}s, ${r}s left): $@"
        return 1
    fi
    echo ""
    echo ">>> $(date '+%H:%M') [$(( $(elapsed)/60 ))m, $(( r/60 ))m left] $@"
    "$@"
}

echo "=== Remaining experiments — $(date) ==="
echo "Budget: $(( MAX_SECONDS/60 )) minutes"

# --- CCR-soft on CIFAR-100 (3 seeds, ~35 min each) ---
for SEED in "${SEEDS[@]}"; do
    run_if_time 2100 $TRAIN --method ccr_soft --dataset cifar100 --seed $SEED --epochs 200 \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir results/cifar100/ccr_soft/seed_${SEED} --data_dir $DATA_DIR || true
done

# --- CIFAR-10: CE + LS + CCR-soft (3 seeds each, ~20-35 min each) ---
for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method ce --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/ce/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method label_smoothing --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/label_smoothing/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 2100 $TRAIN --method ccr_soft --dataset cifar10 --seed $SEED --epochs 200 \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir results/cifar10/ccr_soft/seed_${SEED} --data_dir $DATA_DIR || true
done

# --- Mixup baselines ---
for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method mixup --dataset cifar100 --seed $SEED --epochs 200 \
        --output_dir results/cifar100/mixup/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method mixup --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/mixup/seed_${SEED} --data_dir $DATA_DIR || true
done

# --- TinyImageNet ---
run_if_time 2400 $TRAIN --method ce --dataset tinyimagenet --seed 42 --epochs 100 \
    --output_dir results/tinyimagenet/ce/seed_42 --data_dir $DATA_DIR || true

run_if_time 2400 $TRAIN --method label_smoothing --dataset tinyimagenet --seed 42 --epochs 100 \
    --output_dir results/tinyimagenet/label_smoothing/seed_42 --data_dir $DATA_DIR || true

run_if_time 2400 $TRAIN --method ccr_soft --dataset tinyimagenet --seed 42 --epochs 100 \
    --lambda_ccr 0.1 --gamma 0.1 \
    --output_dir results/tinyimagenet/ccr_soft/seed_42 --data_dir $DATA_DIR || true

# --- Ablations ---
run_if_time 2100 $TRAIN --method ccr_adaptive --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.05 --gamma 0.1 \
    --output_dir results/cifar100/ccr_adaptive/seed_42 --data_dir $DATA_DIR || true

run_if_time 2100 $TRAIN --method ccr_curriculum --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --gamma 0.1 \
    --output_dir results/cifar100/ccr_curriculum/seed_42 --data_dir $DATA_DIR || true

run_if_time 2100 $TRAIN --method ccr_spectral --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 \
    --output_dir results/cifar100/ccr_spectral/seed_42 --data_dir $DATA_DIR || true

run_if_time 2100 $TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
    --lambda_ccr 0.1 --tau 5.0 \
    --output_dir results/cifar100/ccr_fixed/seed_42 --data_dir $DATA_DIR || true

# Lambda sweep
for LAMBDA in 0.01 0.5; do
    run_if_time 2100 $TRAIN --method ccr_soft --dataset cifar100 --seed 42 --epochs 200 \
        --lambda_ccr $LAMBDA --gamma 0.1 \
        --output_dir results/cifar100/ccr_soft_lambda_${LAMBDA}/seed_42 --data_dir $DATA_DIR || true
done

echo ""
echo "=== ALL DONE at $(( $(elapsed)/60 ))m — $(date) ==="
