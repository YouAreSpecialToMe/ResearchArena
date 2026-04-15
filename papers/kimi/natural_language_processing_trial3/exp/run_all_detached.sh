#!/bin/bash
# Run all experiments completely detached

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01
source .venv/bin/activate

LOG="logs/detached_run_$(date +%Y%m%d_%H%M%S).log"
exec > "$LOG" 2>&1

echo "Starting detached experiment run at $(date)"
echo "==========================================="

# Run all experiments
for method in vanilla esr entropy_only egl bestofn; do
    for seed in 42 123; do
        echo ""
        echo "[$method / seed=$seed] Starting at $(date)"
        
        python exp/run_batch_experiments.py \
            --method $method \
            --dataset gsm8k \
            --seed $seed \
            --max_problems 50 \
            --batch_size 10
        
        echo "[$method / seed=$seed] Completed at $(date)"
        sleep 5
    done
done

echo ""
echo "All experiments completed at $(date)"
echo "Aggregating results..."

python exp/aggregate_all_results.py

echo "Done!"
