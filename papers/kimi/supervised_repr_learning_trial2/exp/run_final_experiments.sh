#!/bin/bash
# Final optimized experiment runner - fits within 8-hour budget
# 25 epochs per experiment, 9 experiments = ~6 hours total

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

mkdir -p results figures logs

# Configuration - reduced epochs to fit time budget
DATASET="cifar100"
NOISE_RATE=0.4
EPOCHS=25
NUM_WORKERS=2

echo "==================================================="
echo "LASER-SCL Final Experiments - Starting at $(date)"
echo "==================================================="
echo "Dataset: $DATASET | Noise: $NOISE_RATE | Epochs: $EPOCHS"
echo ""

# Define experiments
declare -a EXPERIMENTS=(
    "supcon 42"
    "supcon 123"
    "supcon 456"
    "supcon_lr 42"
    "supcon_lr 123"
    "supcon_lr 456"
    "laser_scl 42"
    "laser_scl 123"
    "laser_scl 456"
)

TOTAL=${#EXPERIMENTS[@]}
COUNT=0

for exp in "${EXPERIMENTS[@]}"; do
    COUNT=$((COUNT + 1))
    read method seed <<< "$exp"
    
    echo ""
    echo "[$COUNT/$TOTAL] Running $method seed $seed at $(date)"
    
    # Check if result already exists
    result_file="results/${method}_${DATASET}_n${NOISE_RATE/./}_s${seed}.json"
    if [ -f "$result_file" ]; then
        echo "  Result exists, skipping"
        continue
    fi
    
    # Run experiment
    python exp/shared/train.py \
        --dataset $DATASET \
        --noise_rate $NOISE_RATE \
        --method $method \
        --epochs $EPOCHS \
        --seed $seed \
        --num_workers $NUM_WORKERS \
        --save_dir results \
        2>&1 | tee "logs/${method}_s${seed}.log"
    
    if [ -f "$result_file" ]; then
        echo "  ✓ Completed"
    else
        echo "  ✗ Failed - check logs"
    fi
done

echo ""
echo "==================================================="
echo "All experiments completed at $(date)"
echo "==================================================="

# Run analysis
if ls results/*.json 1>/dev/null 2>&1; then
    echo ""
    echo "Running analysis..."
    python analyze_results.py 2>&1 | tee logs/analysis.log
fi

echo ""
echo "Results summary:"
ls -la results/*.json 2>/dev/null
