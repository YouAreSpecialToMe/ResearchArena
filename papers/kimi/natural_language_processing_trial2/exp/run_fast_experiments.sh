#!/bin/bash
# Fast experiment runner using 100-problem subsets
# Generates real results quickly for verification

set -e

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01
source .venv/bin/activate

mkdir -p logs
mkdir -p results/baseline_cot
mkdir -p results/baseline_sc16
mkdir -p results/baseline_com
mkdir -p results/cdhr_main
mkdir -p results/ablation_beta

# Use 100 problems for fast results
LIMIT=100

echo "======================================"
echo "FAST CDHR EXPERIMENTS (100 problems each)"
echo "Started at: $(date)"
echo "======================================"

MODEL="llama-3.1-8b"
DATASETS="gsm8k math gpqa"
SEEDS="42 123 456"

# ============================================
# BASELINE COT: All datasets, all seeds
# ============================================
echo ""
echo "--- Baseline CoT (limit=$LIMIT) ---"
for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        echo "[$MODEL] CoT on $DATASET (seed=$SEED, limit=$LIMIT)"
        python exp/baseline_cot/run.py \
            --model $MODEL \
            --dataset data/${DATASET}.json \
            --seed $SEED \
            --limit $LIMIT \
            --output results/baseline_cot/${MODEL}_${DATASET}_seed${SEED}_limit${LIMIT}.json \
            2>&1 | tee logs/fast_cot_${MODEL}_${DATASET}_s${SEED}.log
    done
done

# ============================================
# CDHR MAIN: All datasets, all seeds
# ============================================
echo ""
echo "--- CDHR Main (limit=$LIMIT) ---"
for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        echo "[$MODEL] CDHR on $DATASET (seed=$SEED, limit=$LIMIT)"
        python exp/cdhr_main/run_real.py \
            --model $MODEL \
            --dataset data/${DATASET}.json \
            --retrieval_index data/retrieval_index.pkl \
            --seed $SEED \
            --limit $LIMIT \
            --output results/cdhr_main/${MODEL}_${DATASET}_seed${SEED}_limit${LIMIT}.json \
            2>&1 | tee logs/fast_cdhr_${MODEL}_${DATASET}_s${SEED}.log
    done
done

# ============================================
# BASELINE: Self-Consistency (16 samples)
# ============================================
echo ""
echo "--- Self-Consistency (limit=$LIMIT, 16 samples) ---"
for DATASET in $DATASETS; do
    echo "[$MODEL] SC16 on $DATASET (limit=$LIMIT)"
    python exp/baseline_sc16/run.py \
        --model $MODEL \
        --dataset data/${DATASET}.json \
        --samples 16 \
        --seed 42 \
        --limit $LIMIT \
        --output results/baseline_sc16/${MODEL}_${DATASET}_seed42_limit${LIMIT}.json \
        2>&1 | tee logs/fast_sc16_${MODEL}_${DATASET}.log
done

# ============================================
# BASELINE: Chain of Mindset
# ============================================
echo ""
echo "--- Chain of Mindset (limit=$LIMIT) ---"
for DATASET in $DATASETS; do
    echo "[$MODEL] CoM on $DATASET (limit=$LIMIT)"
    python exp/baseline_com/run.py \
        --model $MODEL \
        --dataset data/${DATASET}.json \
        --seed 42 \
        --limit $LIMIT \
        --output results/baseline_com/${MODEL}_${DATASET}_seed42_limit${LIMIT}.json \
        2>&1 | tee logs/fast_com_${MODEL}_${DATASET}.log
done

# ============================================
# ABLATION: Beta sensitivity
# ============================================
echo ""
echo "--- Beta Sensitivity Ablation (limit=50) ---"
for BETA in 0.0 0.25 0.5 0.75 1.0; do
    echo "[$MODEL] CDHR with beta=$BETA on GSM8K (50 samples)"
    python exp/cdhr_main/run_real.py \
        --model $MODEL \
        --dataset data/gsm8k.json \
        --retrieval_index data/retrieval_index.pkl \
        --seed 42 \
        --beta $BETA \
        --limit 50 \
        --output results/ablation_beta/beta${BETA}_${MODEL}_gsm8k_limit50.json \
        2>&1 | tee logs/fast_ablation_beta_${BETA}.log
done

echo ""
echo "======================================"
echo "FAST EXPERIMENTS COMPLETED"
echo "Finished at: $(date)"
echo "======================================"

# Generate results summary
python << 'EOF'
import json
import os
from glob import glob

results_summary = {
    'fast_experiments': True,
    'limit': 100,
    'baseline_cot': {},
    'cdhr': {},
    'baseline_sc16': {},
    'baseline_com': {},
    'ablation_beta': {}
}

# Collect CoT results
for f in glob('results/baseline_cot/*_limit100.json'):
    with open(f) as fp:
        data = json.load(fp)
        key = os.path.basename(f).replace('.json', '')
        results_summary['baseline_cot'][key] = data['metrics']

# Collect CDHR results
for f in glob('results/cdhr_main/*_limit100.json'):
    with open(f) as fp:
        data = json.load(fp)
        key = os.path.basename(f).replace('.json', '')
        results_summary['cdhr'][key] = data['metrics']

# Collect SC16 results
for f in glob('results/baseline_sc16/*_limit100.json'):
    with open(f) as fp:
        data = json.load(fp)
        key = os.path.basename(f).replace('.json', '')
        results_summary['baseline_sc16'][key] = data['metrics']

# Collect CoM results
for f in glob('results/baseline_com/*_limit100.json'):
    with open(f) as fp:
        data = json.load(fp)
        key = os.path.basename(f).replace('.json', '')
        results_summary['baseline_com'][key] = data['metrics']

# Collect ablation results
for f in glob('results/ablation_beta/*_limit50.json'):
    with open(f) as fp:
        data = json.load(fp)
        key = os.path.basename(f).replace('.json', '')
        results_summary['ablation_beta'][key] = data['metrics']

with open('results/fast_experiments_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nFast experiments summary saved to results/fast_experiments_summary.json")
EOF
