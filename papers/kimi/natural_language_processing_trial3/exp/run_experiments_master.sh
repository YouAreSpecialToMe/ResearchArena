#!/bin/bash
# Master experiment script - runs all experiments
# This script is designed to run for several hours

set -e

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01
source .venv/bin/activate

LOGDIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

echo "==============================================" | tee "$LOGDIR/master.log"
echo "ESR Master Experiment Run" | tee -a "$LOGDIR/master.log"
echo "Started: $(date)" | tee -a "$LOGDIR/master.log"
echo "==============================================" | tee -a "$LOGDIR/master.log"

# Configuration
PROBLEMS=100  # Number of problems per experiment
METHODS="vanilla esr entropy_only egl bestofn"
SEEDS="42 123"

echo "" | tee -a "$LOGDIR/master.log"
echo "Configuration:" | tee -a "$LOGDIR/master.log"
echo "  Problems per experiment: $PROBLEMS" | tee -a "$LOGDIR/master.log"
echo "  Methods: $METHODS" | tee -a "$LOGDIR/master.log"
echo "  Seeds: $SEEDS" | tee -a "$LOGDIR/master.log"
echo "" | tee -a "$LOGDIR/master.log"

TOTAL=0
for method in $METHODS; do
    for seed in $SEEDS; do
        TOTAL=$((TOTAL + 1))
    done
done

echo "Total experiments: $TOTAL" | tee -a "$LOGDIR/master.log"
echo "" | tee -a "$LOGDIR/master.log"

# Run experiments
COUNTER=0
for method in $METHODS; do
    for seed in $SEEDS; do
        COUNTER=$((COUNTER + 1))
        echo "[$COUNTER/$TOTAL] Running $method (seed=$seed)..." | tee -a "$LOGDIR/master.log"
        
        START=$(date +%s)
        
        if python exp/run_batch_experiments.py \
            --method $method \
            --dataset gsm8k \
            --seed $seed \
            --max_problems $PROBLEMS \
            --batch_size 25 \
            2>&1 | tee "$LOGDIR/${method}_seed${seed}.log"; then
            
            END=$(date +%s)
            DURATION=$((END - START))
            echo "  ✓ Completed in ${DURATION}s" | tee -a "$LOGDIR/master.log"
        else
            END=$(date +%s)
            DURATION=$((END - START))
            echo "  ✗ Failed after ${DURATION}s" | tee -a "$LOGDIR/master.log"
        fi
        
        # Brief pause between experiments
        sleep 5
    done
done

echo "" | tee -a "$LOGDIR/master.log"
echo "==============================================" | tee -a "$LOGDIR/master.log"
echo "Experiments completed at: $(date)" | tee -a "$LOGDIR/master.log"
echo "==============================================" | tee -a "$LOGDIR/master.log"

# Aggregate results
echo "" | tee -a "$LOGDIR/master.log"
echo "Aggregating results..." | tee -a "$LOGDIR/master.log"
python exp/aggregate_all_results.py 2>&1 | tee -a "$LOGDIR/master.log"

echo "" | tee -a "$LOGDIR/master.log"
echo "All done! Results in exp/results/" | tee -a "$LOGDIR/master.log"
