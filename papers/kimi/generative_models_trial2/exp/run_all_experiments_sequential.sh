#!/bin/bash
# Run all experiments sequentially to avoid resource conflicts

set -e  # Exit on error

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/generative_models/idea_01

LOGDIR="logs/experiments"
mkdir -p "$LOGDIR"

echo "==============================================="
echo "Starting Sequential Experiment Run"
echo "==============================================="
echo ""

# Function to run an experiment
run_exp() {
    local exp_path=$1
    local log_file=$2
    shift 2
    local args=$@
    
    echo "Running: $exp_path $args"
    echo "Log: $log_file"
    
    if python "$exp_path" $args > "$log_file" 2>&1; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $?"
    fi
    echo ""
}

# ================================================
# BASELINE EXPERIMENTS
# ================================================

echo "=== BASELINE 1: UNIFORM WEIGHTING ==="
run_exp "exp/baseline_uniform/run.py" "$LOGDIR/baseline_uniform_s42.log" --seed 42 --epochs 70 --batch_size 32
run_exp "exp/baseline_uniform/run.py" "$LOGDIR/baseline_uniform_s123.log" --seed 123 --epochs 70 --batch_size 32

echo "=== BASELINE 2: DENSITY WEIGHTING ==="
run_exp "exp/baseline_density/run.py" "$LOGDIR/baseline_density_s42.log" --seed 42 --epochs 70 --batch_size 32
run_exp "exp/baseline_density/run.py" "$LOGDIR/baseline_density_s123.log" --seed 123 --epochs 70 --batch_size 32

# ================================================
# MAIN METHOD: DISTFLOW-IDW
# ================================================

echo "=== MAIN: DISTFLOW-IDW ==="
run_exp "exp/distflow_idw/run.py" "$LOGDIR/distflow_idw_s42.log" --seed 42 --epochs 70 --batch_size 32
run_exp "exp/distflow_idw/run.py" "$LOGDIR/distflow_idw_s123.log" --seed 123 --epochs 70 --batch_size 32
run_exp "exp/distflow_idw/run.py" "$LOGDIR/distflow_idw_s456.log" --seed 456 --epochs 70 --batch_size 32

# ================================================
# DISTFLOW-LAW
# ================================================

echo "=== DISTFLOW-LAW ==="
run_exp "exp/distflow_law/run.py" "$LOGDIR/distflow_law_s42.log" --seed 42 --epochs 70 --batch_size 32
run_exp "exp/distflow_law/run.py" "$LOGDIR/distflow_law_s123.log" --seed 123 --epochs 70 --batch_size 32

# ================================================
# ABLATION EXPERIMENTS
# ================================================

echo "=== ABLATION: NO FILM ==="
run_exp "exp/ablation_no_film/run.py" "$LOGDIR/ablation_no_film_s42.log" --seed 42 --epochs 70 --batch_size 32
run_exp "exp/ablation_no_film/run.py" "$LOGDIR/ablation_no_film_s123.log" --seed 123 --epochs 70 --batch_size 32

echo "=== ABLATION: NO STRATIFICATION ==="
run_exp "exp/ablation_no_stratify/run.py" "$LOGDIR/ablation_no_stratify_s42.log" --seed 42 --epochs 70 --batch_size 32
run_exp "exp/ablation_no_stratify/run.py" "$LOGDIR/ablation_no_stratify_s123.log" --seed 123 --epochs 70 --batch_size 32

# ================================================
# SENSITIVITY ANALYSIS
# ================================================

echo "=== SENSITIVITY: BETA VALUES ==="
run_exp "exp/sensitivity_beta/run.py" "$LOGDIR/sensitivity_beta1.0.log" --seed 42 --epochs 40 --batch_size 32 --beta 1.0
run_exp "exp/sensitivity_beta/run.py" "$LOGDIR/sensitivity_beta1.5.log" --seed 42 --epochs 40 --batch_size 32 --beta 1.5
run_exp "exp/sensitivity_beta/run.py" "$LOGDIR/sensitivity_beta2.5.log" --seed 42 --epochs 40 --batch_size 32 --beta 2.5

echo "==============================================="
echo "All experiments completed!"
echo "==============================================="
