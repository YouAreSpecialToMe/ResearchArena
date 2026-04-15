#!/bin/bash
# Sequential experiment runner for a single model
# Usage: ./run_model_experiments.sh <model_name>

set -e

MODEL=$1
if [ -z "$MODEL" ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 llama-3.1-8b"
    exit 1
fi

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01
source .venv/bin/activate

echo "======================================"
echo "Running experiments for model: $MODEL"
echo "Started at: $(date)"
echo "======================================"

# Create directories
mkdir -p logs
mkdir -p results/baseline_cot
mkdir -p results/baseline_sc16
mkdir -p results/baseline_com
mkdir -p results/cdhr_main
mkdir -p results/ablation_beta

# ============================================
# BASELINE COT: All datasets, 3 seeds
# ============================================
echo ""
echo "=== Baseline CoT ==="
for DATASET in gsm8k math gpqa; do
    for SEED in 42 123 456; do
        echo "[$MODEL] CoT on $DATASET (seed=$SEED)"
        python exp/baseline_cot/run.py \
            --model $MODEL \
            --dataset data/${DATASET}.json \
            --seed $SEED \
            --output results/baseline_cot/${MODEL}_${DATASET}_seed${SEED}.json \
            2>&1 | tee logs/cot_${MODEL}_${DATASET}_s${SEED}.log
    done
done

# ============================================
# CDHR MAIN: All datasets, 3 seeds
# ============================================
echo ""
echo "=== CDHR Main ==="
for DATASET in gsm8k math gpqa; do
    for SEED in 42 123 456; do
        echo "[$MODEL] CDHR on $DATASET (seed=$SEED)"
        python exp/cdhr_main/run_real.py \
            --model $MODEL \
            --dataset data/${DATASET}.json \
            --retrieval_index data/retrieval_index.pkl \
            --seed $SEED \
            --output results/cdhr_main/${MODEL}_${DATASET}_seed${SEED}.json \
            2>&1 | tee logs/cdhr_${MODEL}_${DATASET}_s${SEED}.log
    done
done

# ============================================
# BASELINE: Self-Consistency (16 samples)
# ============================================
echo ""
echo "=== Self-Consistency (16 samples) ==="
for DATASET in gsm8k math gpqa; do
    echo "[$MODEL] SC16 on $DATASET"
    python exp/baseline_sc16/run.py \
        --model $MODEL \
        --dataset data/${DATASET}.json \
        --samples 16 \
        --seed 42 \
        --output results/baseline_sc16/${MODEL}_${DATASET}_seed42.json \
        2>&1 | tee logs/sc16_${MODEL}_${DATASET}.log
done

# ============================================
# BASELINE: Chain of Mindset
# ============================================
echo ""
echo "=== Chain of Mindset ==="
for DATASET in gsm8k math gpqa; do
    echo "[$MODEL] CoM on $DATASET"
    python exp/baseline_com/run.py \
        --model $MODEL \
        --dataset data/${DATASET}.json \
        --seed 42 \
        --output results/baseline_com/${MODEL}_${DATASET}_seed42.json \
        2>&1 | tee logs/com_${MODEL}_${DATASET}.log
done

# ============================================
# ABLATION: Beta sensitivity (Llama only)
# ============================================
if [ "$MODEL" = "llama-3.1-8b" ]; then
    echo ""
    echo "=== Beta Sensitivity Ablation ==="
    for BETA in 0.0 0.25 0.5 0.75 1.0; do
        echo "[$MODEL] CDHR with beta=$BETA on GSM8K (200 samples)"
        python exp/cdhr_main/run_real.py \
            --model $MODEL \
            --dataset data/gsm8k.json \
            --retrieval_index data/retrieval_index.pkl \
            --seed 42 \
            --beta $BETA \
            --limit 200 \
            --output results/ablation_beta/beta${BETA}_${MODEL}_gsm8k.json \
            2>&1 | tee logs/ablation_beta_${BETA}.log
    done
fi

echo ""
echo "======================================"
echo "All experiments completed for $MODEL"
echo "Finished at: $(date)"
echo "======================================"
