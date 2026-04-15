#!/bin/bash
# Run all real CDHR experiments
# This script runs all baselines and CDHR on all datasets with all models

set -e

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01
source .venv/bin/activate

MODELS="llama-3.1-8b qwen2.5-7b deepseek-r1-7b"
DATASETS="data/gsm8k.json data/math.json data/gpqa.json"
SEEDS="42 123 456"

echo "======================================"
echo "Starting Real CDHR Experiments"
echo "======================================"
echo "Models: $MODELS"
echo "Datasets: $DATASETS"
echo "Seeds: $SEEDS"
echo "======================================"

# Create result directories
mkdir -p results/baseline_cot
mkdir -p results/baseline_sc16
mkdir -p results/baseline_com
mkdir -p results/cdhr_main
mkdir -p results/ablation_beta
mkdir -p results/ablation_strategies
mkdir -p results/ablation_dynamics
mkdir -p logs

# ============================================
# BASELINE: Standard CoT (all models, all datasets, 3 seeds)
# ============================================
echo ""
echo "======================================"
echo "Running Baseline: Standard CoT"
echo "======================================"

for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            dataset_name=$(basename $dataset .json)
            echo "Running CoT: $model on $dataset_name (seed=$seed)"
            python exp/baseline_cot/run.py \
                --model $model \
                --dataset $dataset \
                --seed $seed \
                > logs/cot_${model}_${dataset_name}_s${seed}.log 2>&1
        done
    done
done

echo "CoT baselines completed!"

# ============================================
# BASELINE: Self-Consistency (16 samples)
# Note: Run only on Llama for time constraints
# ============================================
echo ""
echo "======================================"
echo "Running Baseline: Self-Consistency (16 samples)"
echo "======================================"

for dataset in $DATASETS; do
    dataset_name=$(basename $dataset .json)
    echo "Running SC16: llama-3.1-8b on $dataset_name"
    python exp/baseline_sc16/run.py \
        --model llama-3.1-8b \
        --dataset $dataset \
        --samples 16 \
        > logs/sc16_llama-3.1-8b_${dataset_name}.log 2>&1
done

echo "Self-Consistency baselines completed!"

# ============================================
# BASELINE: Chain of Mindset
# Note: Run only on Llama for time constraints
# ============================================
echo ""
echo "======================================"
echo "Running Baseline: Chain of Mindset"
echo "======================================"

for dataset in $DATASETS; do
    dataset_name=$(basename $dataset .json)
    echo "Running CoM: llama-3.1-8b on $dataset_name"
    python exp/baseline_com/run.py \
        --model llama-3.1-8b \
        --dataset $dataset \
        > logs/com_llama-3.1-8b_${dataset_name}.log 2>&1
done

echo "Chain of Mindset baselines completed!"

# ============================================
# MAIN: CDHR (all models, all datasets, 3 seeds)
# ============================================
echo ""
echo "======================================"
echo "Running CDHR Main Experiments"
echo "======================================"

for model in $MODELS; do
    for dataset in $DATASETS; do
        for seed in $SEEDS; do
            dataset_name=$(basename $dataset .json)
            echo "Running CDHR: $model on $dataset_name (seed=$seed)"
            python exp/cdhr_main/run_real.py \
                --model $model \
                --dataset $dataset \
                --retrieval_index data/retrieval_index.pkl \
                --seed $seed \
                > logs/cdhr_${model}_${dataset_name}_s${seed}.log 2>&1
        done
    done
done

echo "CDHR main experiments completed!"

# ============================================
# ABLATION: Beta sensitivity
# ============================================
echo ""
echo "======================================"
echo "Running Ablation: Beta Sensitivity"
echo "======================================"

BETA_VALUES="0.0 0.25 0.5 0.75 1.0"
for beta in $BETA_VALUES; do
    echo "Running CDHR with beta=$beta on GSM8K subset"
    python exp/cdhr_main/run_real.py \
        --model llama-3.1-8b \
        --dataset data/gsm8k.json \
        --retrieval_index data/retrieval_index.pkl \
        --seed 42 \
        --beta $beta \
        --limit 200 \
        --output results/ablation_beta/beta${beta}_llama_gsm8k.json \
        > logs/ablation_beta_${beta}.log 2>&1
done

echo "Beta sensitivity ablation completed!"

echo ""
echo "======================================"
echo "All experiments completed!"
echo "======================================"
