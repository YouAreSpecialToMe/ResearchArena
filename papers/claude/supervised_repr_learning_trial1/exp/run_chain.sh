#!/bin/bash
# Chain experiment runner - runs experiments sequentially
# Priority order: SupCon, CGA-only, HardNeg, then remaining methods
# Each run takes ~150 min. In 8 hours, we get ~3 full 200-epoch runs.
# After 3 full runs, switch to 100-epoch runs for remaining methods.

set -e
cd "$(dirname "$0")/.."

PY=python3
TRAIN="exp/train.py"
EPOCHS=200
LINEAR=100
FAST_EPOCHS=100
FAST_LINEAR=50

echo "=== Starting experiment chain at $(date) ==="
echo "Full runs: ${EPOCHS}ep contrastive + ${LINEAR}ep linear"

# Phase 1: Priority experiments at full epochs (seed 42)
echo ""
echo "=== Phase 1: Priority experiments (200 epochs, seed 42) ==="

for method in supcon cga_only hardneg; do
    DIR="exp/${method}_v2"
    RESULT="${DIR}/results_seed42.json"
    if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
        echo "[SKIP] $method seed 42 already complete"
    else
        echo ""
        echo "--- Running $method seed 42 (${EPOCHS} epochs) ---"
        $PY $TRAIN --method $method --seed 42 --epochs $EPOCHS --linear_epochs $LINEAR \
            --output_dir $DIR --alpha 0.5 --lam 0.5 2>&1 | tail -5
    fi
done

echo ""
echo "=== Phase 1 complete at $(date) ==="

# Phase 2: Additional methods at reduced epochs (seed 42)
echo ""
echo "=== Phase 2: Additional methods (${FAST_EPOCHS} epochs, seed 42) ==="

for method in reweight varcon_t tcl; do
    DIR="exp/${method}_v2"
    RESULT="${DIR}/results_seed42.json"
    if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
        echo "[SKIP] $method seed 42 already complete"
    else
        echo ""
        echo "--- Running $method seed 42 (${FAST_EPOCHS} epochs) ---"
        $PY $TRAIN --method $method --seed 42 --epochs $FAST_EPOCHS --linear_epochs $FAST_LINEAR \
            --output_dir $DIR 2>&1 | tail -5
    fi
done

# CE baseline
DIR="exp/ce_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] ce seed 42 already complete"
else
    echo ""
    echo "--- Running CE seed 42 ---"
    $PY $TRAIN --method ce --seed 42 --epochs $EPOCHS --linear_epochs 0 \
        --output_dir $DIR 2>&1 | tail -5
fi

echo ""
echo "=== Phase 2 complete at $(date) ==="

# Phase 3: Additional seeds for top methods (reduced epochs)
echo ""
echo "=== Phase 3: Additional seeds (${FAST_EPOCHS} epochs) ==="

for method in supcon cga_only hardneg; do
    for seed in 43 44; do
        DIR="exp/${method}_v2"
        RESULT="${DIR}/results_seed${seed}.json"
        if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
            echo "[SKIP] $method seed $seed already complete"
        else
            echo ""
            echo "--- Running $method seed $seed (${FAST_EPOCHS} epochs) ---"
            $PY $TRAIN --method $method --seed $seed --epochs $FAST_EPOCHS --linear_epochs $FAST_LINEAR \
                --output_dir $DIR --alpha 0.5 --lam 0.5 2>&1 | tail -5
        fi
    done
done

echo ""
echo "=== Phase 3 complete at $(date) ==="

# Phase 4: CGA grid search (reduced epochs, 1 seed)
echo ""
echo "=== Phase 4: CGA Grid Search ==="

for alpha in 0.3 0.5 0.7; do
    for lam in 0.1 0.5 1.0; do
        DIR="exp/cga_grid_v2/a${alpha}_l${lam}"
        RESULT="${DIR}/results_seed42.json"
        if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
            echo "[SKIP] CGA alpha=$alpha lambda=$lam"
        else
            echo "--- CGA alpha=$alpha lambda=$lam ---"
            $PY $TRAIN --method cga_only --seed 42 --epochs $FAST_EPOCHS --linear_epochs $FAST_LINEAR \
                --alpha $alpha --lam $lam --output_dir $DIR 2>&1 | tail -3
        fi
    done
done

echo ""
echo "=== All experiments complete at $(date) ==="
