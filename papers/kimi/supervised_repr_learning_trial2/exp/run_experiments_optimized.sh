#!/bin/bash
# Optimized experiment runner for LASER-SCL
# Runs critical experiments with proper error handling and result verification

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

# Create necessary directories
mkdir -p results figures logs

# Configuration
DATASET="cifar100"
NOISE_RATE=0.4
EPOCHS=50
NUM_WORKERS=2  # Reduced to avoid DataLoader memory issues

echo "==================================================="
echo "LASER-SCL Critical Experiments - Starting at $(date)"
echo "==================================================="
echo "Dataset: $DATASET"
echo "Noise Rate: $NOISE_RATE"
echo "Epochs: $EPOCHS"
echo ""

# Define experiments: method_name seed
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
    echo "[$COUNT/$TOTAL] Running $method with seed $seed at $(date)"
    echo "---------------------------------------------------"
    
    # Check if result already exists
    result_file="results/${method}_${DATASET}_n${NOISE_RATE/./}_s${seed}.json"
    if [ -f "$result_file" ]; then
        echo "Result already exists: $result_file"
        echo "Skipping..."
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
    
    # Verify result was created
    if [ -f "$result_file" ]; then
        echo "✓ Success: $result_file created"
        # Show final accuracy
        python -c "import json; d=json.load(open('$result_file')); print(f\"Final Accuracy: {d['final_accuracy']:.2f}%\")" 2>/dev/null || echo "Could not parse result"
    else
        echo "✗ Error: Result file not created!"
        echo "Check logs: logs/${method}_s${seed}.log"
    fi
    
    echo "[$COUNT/$TOTAL] Completed at $(date)"
done

echo ""
echo "==================================================="
echo "All experiments completed at $(date)"
echo "==================================================="
echo ""

# List all results
echo "Results summary:"
ls -la results/*.json 2>/dev/null || echo "No results found!"

# Run analysis if results exist
if ls results/*.json 1>/dev/null 2>&1; then
    echo ""
    echo "Running analysis..."
    python analyze_results.py
fi
