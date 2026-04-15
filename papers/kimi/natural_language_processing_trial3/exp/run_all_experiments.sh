#!/bin/bash
# Run all ESR experiments with proper scaling
# This script runs experiments sequentially to avoid GPU memory issues

set -e

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01
source .venv/bin/activate

# Configuration
SCALE=${1:-300}  # Default 300 problems, pass "full" for complete dataset
if [ "$SCALE" == "full" ]; then
    MAX_PROBLEMS=""
    echo "Running FULL dataset experiments (all problems)"
else
    MAX_PROBLEMS="--max_problems $SCALE"
    echo "Running experiments with $SCALE problems per dataset"
fi

MODEL="Qwen/Qwen3-1.7B"
DATASETS="gsm8k math500"
METHODS="vanilla esr entropy_only egl bestofn"
SEEDS="42 123"

echo "=================================================="
echo "ESR Complete Experiment Suite"
echo "=================================================="
echo "Model: $MODEL"
echo "Datasets: $DATASETS"
echo "Methods: $METHODS"
echo "Seeds: $SEEDS"
echo "=================================================="

TOTAL=0
COMPLETED=0
FAILED=0

# Calculate total experiments
for dataset in $DATASETS; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "Total experiments: $TOTAL"
echo ""

# Run experiments
COUNTER=0
for dataset in $DATASETS; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            COUNTER=$((COUNTER + 1))
            echo ""
            echo "=================================================="
            echo "[$COUNTER/$TOTAL] Running: $method on $dataset (seed=$seed)"
            echo "=================================================="
            
            START_TIME=$(date +%s)
            
            if python exp/run_batch_experiments.py \
                --method $method \
                --dataset $dataset \
                --model $MODEL \
                --seed $seed \
                $MAX_PROBLEMS \
                --batch_size 50 2>&1 | tee logs/${method}_${dataset}_seed${seed}.log; then
                
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                COMPLETED=$((COMPLETED + 1))
                echo "✓ Completed in ${DURATION}s"
            else
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                FAILED=$((FAILED + 1))
                echo "✗ Failed after ${DURATION}s"
            fi
            
            # Small delay to let GPU cool down
            sleep 2
        done
    done
done

echo ""
echo "=================================================="
echo "Experiment Suite Completed!"
echo "=================================================="
echo "Total: $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo ""

# Generate aggregate results
echo "Generating aggregate results..."
python exp/create_final_results.py 2>&1 | tee logs/aggregate_results.log || echo "Aggregation failed"

echo ""
echo "All done! Results in exp/results/"
