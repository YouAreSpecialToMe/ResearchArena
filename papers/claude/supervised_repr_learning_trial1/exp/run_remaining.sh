#!/bin/bash
# Run remaining experiments after SupCon v2 seed 42 completes
# Time budget: ~7 hours remaining
# Priority: CGA-only (200ep), HardNeg (100ep), then others at 100ep

set -e
cd "$(dirname "$0")/.."

PY=python3
TRAIN="exp/train.py"

echo "=== Starting remaining experiments at $(date) ==="

# 1. CGA-only seed 42 (150 epochs) — OUR PRIMARY METHOD ~2.5h
DIR="exp/cga_only_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] CGA-only seed 42 already complete"
else
    echo ""
    echo "--- Running CGA-only seed 42 (150 epochs) at $(date) ---"
    mkdir -p $DIR
    $PY $TRAIN --method cga_only --seed 42 --epochs 150 --linear_epochs 75 \
        --output_dir $DIR --alpha 0.5 --lam 0.5 2>&1 | tee ${DIR}/log_seed42.txt
fi

# 2. HardNeg seed 42 (75 epochs) — strongest baseline ~1.2h
DIR="exp/hardneg_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] HardNeg seed 42 already complete"
else
    echo ""
    echo "--- Running HardNeg seed 42 (75 epochs) at $(date) ---"
    mkdir -p $DIR
    $PY $TRAIN --method hardneg --seed 42 --epochs 75 --linear_epochs 50 \
        --output_dir $DIR 2>&1 | tee ${DIR}/log_seed42.txt
fi

# 3. TCL seed 42 (75 epochs) — FIXED implementation ~1.2h
DIR="exp/tcl_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] TCL seed 42 already complete"
else
    echo ""
    echo "--- Running TCL seed 42 (75 epochs) at $(date) ---"
    mkdir -p $DIR
    $PY $TRAIN --method tcl --seed 42 --epochs 75 --linear_epochs 50 \
        --output_dir $DIR 2>&1 | tee ${DIR}/log_seed42.txt
fi

# 4. Reweight seed 42 (75 epochs) ~1.2h
DIR="exp/reweight_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] Reweight seed 42 already complete"
else
    echo ""
    echo "--- Running Reweight seed 42 (75 epochs) at $(date) ---"
    mkdir -p $DIR
    $PY $TRAIN --method reweight --seed 42 --epochs 75 --linear_epochs 50 \
        --output_dir $DIR 2>&1 | tee ${DIR}/log_seed42.txt
fi

# 5. VarCon-T seed 42 (75 epochs) ~1.2h
DIR="exp/varcon_t_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] VarCon-T seed 42 already complete"
else
    echo ""
    echo "--- Running VarCon-T seed 42 (75 epochs) at $(date) ---"
    mkdir -p $DIR
    $PY $TRAIN --method varcon_t --seed 42 --epochs 75 --linear_epochs 50 \
        --output_dir $DIR 2>&1 | tee ${DIR}/log_seed42.txt
fi

# 6. CE baseline seed 42 (150 epochs, no linear eval) ~1.2h
DIR="exp/ce_v2"
RESULT="${DIR}/results_seed42.json"
if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
    echo "[SKIP] CE seed 42 already complete"
else
    echo ""
    echo "--- Running CE seed 42 (150 epochs) at $(date) ---"
    mkdir -p $DIR
    $PY $TRAIN --method ce --seed 42 --epochs 150 --linear_epochs 0 \
        --output_dir $DIR 2>&1 | tee ${DIR}/log_seed42.txt
fi

# 7. Additional seeds for SupCon and CGA-only (75 epochs each)
for method in supcon cga_only; do
    for seed in 43 44; do
        DIR="exp/${method}_v2"
        RESULT="${DIR}/results_seed${seed}.json"
        if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
            echo "[SKIP] $method seed $seed already complete"
        else
            echo ""
            echo "--- Running $method seed $seed (75 epochs) at $(date) ---"
            mkdir -p $DIR
            $PY $TRAIN --method $method --seed $seed --epochs 75 --linear_epochs 50 \
                --output_dir $DIR --alpha 0.5 --lam 0.5 2>&1 | tee ${DIR}/log_seed${seed}.txt
        fi
    done
done

# 8. CGA grid search (50 epochs, seed 42 only) — enough for relative comparison
for alpha in 0.3 0.5 0.7; do
    for lam in 0.1 0.5 1.0; do
        DIR="exp/cga_grid_v2/a${alpha}_l${lam}"
        RESULT="${DIR}/results_seed42.json"
        if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
            echo "[SKIP] CGA alpha=$alpha lambda=$lam"
        else
            echo "--- CGA alpha=$alpha lambda=$lam at $(date) ---"
            mkdir -p $DIR
            $PY $TRAIN --method cga_only --seed 42 --epochs 50 --linear_epochs 30 \
                --alpha $alpha --lam $lam --output_dir $DIR 2>&1 | tail -5
        fi
    done
done

# 9. CGA-full and AdaptTemp if time remains
for method in cga_full adaptive_temp; do
    DIR="exp/${method}_v2"
    RESULT="${DIR}/results_seed42.json"
    if [ -f "$RESULT" ] && python3 -c "import json; d=json.load(open('$RESULT')); assert d['top1'] > 10" 2>/dev/null; then
        echo "[SKIP] $method seed 42 already complete"
    else
        echo ""
        echo "--- Running $method seed 42 (75 epochs) at $(date) ---"
        mkdir -p $DIR
        $PY $TRAIN --method $method --seed 42 --epochs 75 --linear_epochs 50 \
            --output_dir $DIR --alpha 0.5 --lam 0.5 2>&1 | tee ${DIR}/log_seed42.txt
    fi
done

echo ""
echo "=== All experiments complete at $(date) ==="

# Aggregate results
echo "Aggregating results..."
$PY aggregate_results.py

# Generate figures
echo "Generating figures..."
$PY figures/create_figures.py
