#!/bin/bash
# Master experiment runner script

set -e  # Exit on error

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01
source .venv/bin/activate

mkdir -p logs exp/results

echo "========================================"
echo "ESR Experiments - Real Model Inference"
echo "Started at: $(date)"
echo "========================================"

# Configuration
MAX_PROBLEMS=150  # Reduced for time constraints
MODEL="Qwen/Qwen3-1.7B"
SEEDS=(42 123)

# Function to run experiment
run_exp() {
    local method=$1
    local dataset=$2
    local seed=$3
    local logfile="logs/${method}_${dataset}_seed${seed}.log"
    
    echo ""
    echo "[$(date)] Running: $method on $dataset (seed $seed)"
    echo "Log: $logfile"
    
    case $method in
        "vanilla")
            python exp/vanilla_cot/run.py --dataset $dataset --seed $seed --max_problems $MAX_PROBLEMS 2>&1 | tee $logfile
            ;;
        "esr")
            python exp/esr/run.py --dataset $dataset --seed $seed --max_problems $MAX_PROBLEMS 2>&1 | tee $logfile
            ;;
        "entropy_only")
            python exp/entropy_only/run.py --dataset $dataset --seed $seed --max_problems $MAX_PROBLEMS 2>&1 | tee $logfile
            ;;
        "egb")
            python exp/egb_beam/run.py --dataset $dataset --seed $seed --max_problems $MAX_PROBLEMS 2>&1 | tee $logfile
            ;;
        "egl")
            python exp/egl_posthoc/run.py --dataset $dataset --seed $seed --max_problems $MAX_PROBLEMS 2>&1 | tee $logfile
            ;;
        "bestofn")
            python exp/bestofn/run.py --dataset $dataset --seed $seed --max_problems $MAX_PROBLEMS 2>&1 | tee $logfile
            ;;
        *)
            echo "Unknown method: $method"
            return 1
            ;;
    esac
    
    echo "[$(date)] Completed: $method on $dataset (seed $seed)"
}

# Main experiment loop
METHODS=("vanilla" "esr" "entropy_only" "egb" "egl" "bestofn")
DATASETS=("gsm8k")

TOTAL=0
COMPLETED=0
FAILED=0

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ((TOTAL++))
        done
    done
done

echo ""
echo "Total experiments to run: $TOTAL"
echo ""

for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo ""
            echo "========================================"
            echo "Progress: $((COMPLETED + FAILED + 1))/$TOTAL"
            echo "========================================"
            
            if run_exp $method $dataset $seed; then
                ((COMPLETED++))
            else
                ((FAILED++))
                echo "WARNING: Experiment failed - $method/$dataset/seed$seed"
            fi
            
            # Show current results
            echo ""
            echo "Current results:"
            ls -la exp/results/*.json 2>/dev/null | wc -l | xargs echo "  JSON files:"
        done
    done
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Completed: $COMPLETED, Failed: $FAILED"
echo "Finished at: $(date)"
echo "========================================"

# Generate aggregate results
echo ""
echo "Generating aggregate results..."
python exp/aggregate_results.py 2>&1 | tee logs/aggregate_results.log

echo ""
echo "Results summary:"
ls -la exp/results/*.json 2>/dev/null
