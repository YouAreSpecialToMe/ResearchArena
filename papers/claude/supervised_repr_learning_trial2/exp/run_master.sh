#!/bin/bash
# Master experiment script — optimized priority ordering
# Runs strong CCR sweep first (critical), then baselines, then ablations
set -e
cd "$(dirname "$0")/.."

TRAIN="python exp/train.py"
DATA_DIR="./data"
SEEDS=(42 123 456)
START_TIME=$(date +%s)
MAX_SECONDS=16200  # 4.5 hours

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

echo "=== Master experiment run — $(date) ==="
echo "Budget: $(( MAX_SECONDS/60 )) minutes"

# ============================================================
# PHASE 1: Quick strong CCR sweep on CIFAR-100 seed_42 (100 epochs)
# Goal: find tau where CCR actually activates and improves ECE
# CE spread at epoch 200 is ~12.9; we want tau near that scale
# ============================================================
echo ""
echo "=== PHASE 1: Strong CCR tau sweep (100 epochs each) ==="

for TAU in 10 15 20; do
    run_if_time 700 $TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 100 \
        --lambda_ccr 0.1 --tau ${TAU}.0 \
        --output_dir results/cifar100/ccr_fixed_tau${TAU}_100ep/seed_42 --data_dir $DATA_DIR || true
done

# Also test ccr_soft with stronger gamma
run_if_time 700 $TRAIN --method ccr_soft --dataset cifar100 --seed 42 --epochs 100 \
    --lambda_ccr 0.01 --gamma 2.0 \
    --output_dir results/cifar100/ccr_soft_gamma2_100ep/seed_42 --data_dir $DATA_DIR || true

echo ""
echo "=== PHASE 1 COMPLETE — analyzing sweep results ==="

# Analyze which setting is best
python3 -c "
import json, os, glob
best_ece = 1.0
best_dir = ''
for d in glob.glob('results/cifar100/ccr_*_100ep/seed_42/metrics.json'):
    try:
        m = json.load(open(d))
        ece = m.get('calibration', {}).get('ece', 1.0)
        acc = m.get('test_accuracy', 0)
        name = d.split('/')[-3]
        print(f'  {name}: ECE={ece:.4f}, Acc={acc:.4f}')
        if ece < best_ece:
            best_ece = ece
            best_dir = name
    except: pass
print(f'\n  BEST: {best_dir} (ECE={best_ece:.4f})')
# Save best setting
with open('results/cifar100/best_ccr_setting.txt', 'w') as f:
    f.write(best_dir)
" 2>/dev/null || echo "Sweep analysis failed, defaulting to tau=15"

# ============================================================
# PHASE 2: Full 200-epoch runs with best CCR setting (3 seeds)
# ============================================================
echo ""
echo "=== PHASE 2: Best CCR × 3 seeds on CIFAR-100 (200 epochs) ==="

# Read best setting (default to ccr_fixed tau=15 if analysis failed)
BEST=$(cat results/cifar100/best_ccr_setting.txt 2>/dev/null || echo "ccr_fixed_tau15_100ep")

# Parse best setting to extract method and params
if echo "$BEST" | grep -q "ccr_fixed_tau"; then
    BEST_TAU=$(echo "$BEST" | sed 's/ccr_fixed_tau\([0-9]*\).*/\1/')
    BEST_METHOD="ccr_fixed"
    BEST_LAMBDA="0.1"
    BEST_GAMMA="0.1"
    BEST_TAU_VAL="${BEST_TAU}.0"
    BEST_NAME="ccr_fixed_tau${BEST_TAU}"
elif echo "$BEST" | grep -q "ccr_soft_gamma"; then
    BEST_METHOD="ccr_soft"
    BEST_LAMBDA="0.01"
    BEST_GAMMA="2.0"
    BEST_TAU_VAL="1.0"
    BEST_NAME="ccr_soft_gamma2"
else
    BEST_METHOD="ccr_fixed"
    BEST_LAMBDA="0.1"
    BEST_GAMMA="0.1"
    BEST_TAU_VAL="15.0"
    BEST_NAME="ccr_fixed_tau15"
fi

echo "Using: method=$BEST_METHOD, tau=$BEST_TAU_VAL, lambda=$BEST_LAMBDA, gamma=$BEST_GAMMA"
echo "Output name: $BEST_NAME"

for SEED in "${SEEDS[@]}"; do
    run_if_time 1800 $TRAIN --method $BEST_METHOD --dataset cifar100 --seed $SEED --epochs 200 \
        --lambda_ccr $BEST_LAMBDA --tau $BEST_TAU_VAL --gamma $BEST_GAMMA \
        --output_dir results/cifar100/${BEST_NAME}/seed_${SEED} --data_dir $DATA_DIR || true
done

# ============================================================
# PHASE 3: CIFAR-10 baselines + best CCR (3 seeds each)
# ============================================================
echo ""
echo "=== PHASE 3: CIFAR-10 (CE + LS + best CCR × 3 seeds) ==="

for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method ce --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/ce/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method label_smoothing --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/label_smoothing/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1800 $TRAIN --method $BEST_METHOD --dataset cifar10 --seed $SEED --epochs 200 \
        --lambda_ccr $BEST_LAMBDA --tau $BEST_TAU_VAL --gamma $BEST_GAMMA \
        --output_dir results/cifar10/${BEST_NAME}/seed_${SEED} --data_dir $DATA_DIR || true
done

# ============================================================
# PHASE 4: TinyImageNet (1 seed each)
# ============================================================
echo ""
echo "=== PHASE 4: TinyImageNet ==="

run_if_time 2400 $TRAIN --method ce --dataset tinyimagenet --seed 42 --epochs 100 \
    --output_dir results/tinyimagenet/ce/seed_42 --data_dir $DATA_DIR || true

run_if_time 2400 $TRAIN --method label_smoothing --dataset tinyimagenet --seed 42 --epochs 100 \
    --output_dir results/tinyimagenet/label_smoothing/seed_42 --data_dir $DATA_DIR || true

run_if_time 2400 $TRAIN --method $BEST_METHOD --dataset tinyimagenet --seed 42 --epochs 100 \
    --lambda_ccr $BEST_LAMBDA --tau $BEST_TAU_VAL --gamma $BEST_GAMMA \
    --output_dir results/tinyimagenet/${BEST_NAME}/seed_42 --data_dir $DATA_DIR || true

# ============================================================
# PHASE 5: Mixup baselines (if time permits)
# ============================================================
echo ""
echo "=== PHASE 5: Mixup baselines ==="

for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method mixup --dataset cifar100 --seed $SEED --epochs 200 \
        --output_dir results/cifar100/mixup/seed_${SEED} --data_dir $DATA_DIR || true
done

for SEED in "${SEEDS[@]}"; do
    run_if_time 1500 $TRAIN --method mixup --dataset cifar10 --seed $SEED --epochs 200 \
        --output_dir results/cifar10/mixup/seed_${SEED} --data_dir $DATA_DIR || true
done

# ============================================================
# PHASE 6: Ablations (if time permits)
# ============================================================
echo ""
echo "=== PHASE 6: Ablations ==="

# Other tau values as ablations
for TAU in 10 20 30; do
    if [ "$TAU" != "$BEST_TAU" ] 2>/dev/null; then
        run_if_time 1800 $TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
            --lambda_ccr 0.1 --tau ${TAU}.0 \
            --output_dir results/cifar100/ccr_fixed_tau${TAU}/seed_42 --data_dir $DATA_DIR || true
    fi
done

# Lambda sweep
for LAMBDA in 0.01 0.5; do
    run_if_time 1800 $TRAIN --method ccr_fixed --dataset cifar100 --seed 42 --epochs 200 \
        --lambda_ccr $LAMBDA --tau $BEST_TAU_VAL \
        --output_dir results/cifar100/ccr_fixed_lambda${LAMBDA}/seed_42 --data_dir $DATA_DIR || true
done

# CCR-soft (original gamma=0.1) already done, keep as ablation reference

echo ""
echo "============================================"
echo "MASTER SCRIPT COMPLETE — $(date)"
echo "Total time: $(( $(elapsed)/60 )) minutes"
echo "============================================"
