#!/bin/bash
# Run only the most critical experiments with proper thresholds

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/natural_language_processing/idea_01/exp
source ../.venv/bin/activate

MODEL="Qwen/Qwen3-1.7B"
MAX_PROBLEMS=150
OUTPUT_DIR="exp/results"

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "=========================================="
echo "Running CRITICAL experiments"
echo "=========================================="
echo "Started at $(date)"

# 1. Vanilla baseline (seed 42)
echo ""
echo "[1/6] Vanilla CoT (seed 42)..."
python run_complete_experiments.py \
    --method vanilla \
    --dataset gsm8k \
    --model "$MODEL" \
    --seed 42 \
    --max_problems $MAX_PROBLEMS \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/vanilla_seed42_new.log

# 2. ESR with proper thresholds (seed 42)
echo ""
echo "[2/6] ESR with tau_h=1.5, tau_v=0.8 (seed 42)..."
python run_complete_experiments.py \
    --method esr \
    --dataset gsm8k \
    --model "$MODEL" \
    --seed 42 \
    --max_problems $MAX_PROBLEMS \
    --tau_h 1.5 \
    --tau_v 0.8 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/esr_seed42_new.log

# 3. Entropy-only baseline (seed 42)
echo ""
echo "[3/6] Entropy-Only with tau_h=1.5 (seed 42)..."
python run_complete_experiments.py \
    --method entropy_only \
    --dataset gsm8k \
    --model "$MODEL" \
    --seed 42 \
    --max_problems $MAX_PROBLEMS \
    --tau_h 1.5 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/entropy_only_seed42_new.log

# 4. Vanilla (seed 123) for error bars
echo ""
echo "[4/6] Vanilla CoT (seed 123)..."
python run_complete_experiments.py \
    --method vanilla \
    --dataset gsm8k \
    --model "$MODEL" \
    --seed 123 \
    --max_problems $MAX_PROBLEMS \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/vanilla_seed123_new.log

# 5. ESR (seed 123) for error bars
echo ""
echo "[5/6] ESR with tau_h=1.5, tau_v=0.8 (seed 123)..."
python run_complete_experiments.py \
    --method esr \
    --dataset gsm8k \
    --model "$MODEL" \
    --seed 123 \
    --max_problems $MAX_PROBLEMS \
    --tau_h 1.5 \
    --tau_v 0.8 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/esr_seed123_new.log

# 6. MATH-500 ESR
echo ""
echo "[6/6] ESR on MATH-500 (seed 42)..."
python run_complete_experiments.py \
    --method esr \
    --dataset math500 \
    --model "$MODEL" \
    --seed 42 \
    --max_problems 100 \
    --tau_h 1.5 \
    --tau_v 0.8 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee logs/esr_math500_seed42.log

echo ""
echo "=========================================="
echo "All critical experiments completed!"
echo "Finished at $(date)"
echo "=========================================="
