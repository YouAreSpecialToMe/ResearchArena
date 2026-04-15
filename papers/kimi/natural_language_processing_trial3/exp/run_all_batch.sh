#!/bin/bash
# Run all experiments in batch with progress saving

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01/exp
source ../.venv/bin/activate

MODEL="Qwen/Qwen3-1.7B"
MAX_PROBLEMS=150
OUTPUT_DIR="exp/results"

# Function to run a single experiment
run_exp() {
    local method=$1
    local dataset=$2
    local seed=$3
    local extra_args=$4
    
    echo "=========================================="
    echo "Running: $method on $dataset (seed=$seed)"
    echo "=========================================="
    
    python run_complete_experiments.py \
        --method $method \
        --dataset $dataset \
        --model "$MODEL" \
        --seed $seed \
        --max_problems $MAX_PROBLEMS \
        --output_dir "$OUTPUT_DIR" \
        $extra_args
    
    echo "Completed: $method on $dataset (seed=$seed)"
    echo ""
}

# Create results directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Run experiments for GSM8K with seed 42
echo "Starting experiments at $(date)"
run_exp "vanilla" "gsm8k" 42 ""
run_exp "esr" "gsm8k" 42 "--tau_h 1.5 --tau_v 0.8"
run_exp "entropy_only" "gsm8k" 42 "--tau_h 1.5"
run_exp "egl_posthoc" "gsm8k" 42 "--tau_h 1.5"

# Run experiments for GSM8K with seed 123
echo "Running experiments with seed 123..."
run_exp "vanilla" "gsm8k" 123 ""
run_exp "esr" "gsm8k" 123 "--tau_h 1.5 --tau_v 0.8"
run_exp "entropy_only" "gsm8k" 123 "--tau_h 1.5"

# Run experiments for MATH-500
echo "Running MATH-500 experiments..."
run_exp "vanilla" "math500" 42 ""
run_exp "esr" "math500" 42 "--tau_h 1.5 --tau_v 0.8"

echo "All experiments completed at $(date)"
