#!/bin/bash
# Run remaining experiments after the first one completes

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/natural_language_processing/idea_01
source .venv/bin/activate

MODEL="llama-3.1-8b"
LIMIT=100

echo "Starting remaining experiments at $(date)"

# Wait for the first experiment to complete
while pgrep -f "baseline_cot/run.py.*limit100" > /dev/null; do
    echo "Waiting for CoT baseline to complete..."
    sleep 60
done

echo "CoT baseline completed. Starting remaining experiments..."

# Run CDHR on all datasets with all seeds
for DATASET in math gpqa gsm8k; do
    for SEED in 42 123 456; do
        echo "Running CDHR on $DATASET (seed=$SEED)"
        python exp/cdhr_main/run_real.py \
            --model $MODEL \
            --dataset data/${DATASET}.json \
            --retrieval_index data/retrieval_index.pkl \
            --seed $SEED \
            --limit $LIMIT \
            --output results/cdhr_main/${MODEL}_${DATASET}_seed${SEED}_limit${LIMIT}.json \
            2>&1 | tee logs/cdhr_${MODEL}_${DATASET}_s${SEED}_limit${LIMIT}.log
    done
done

# Run Self-Consistency baseline
for DATASET in math gpqa gsm8k; do
    echo "Running SC16 on $DATASET"
    python exp/baseline_sc16/run.py \
        --model $MODEL \
        --dataset data/${DATASET}.json \
        --samples 16 \
        --seed 42 \
        --limit $LIMIT \
        --output results/baseline_sc16/${MODEL}_${DATASET}_seed42_limit${LIMIT}.json \
        2>&1 | tee logs/sc16_${MODEL}_${DATASET}_limit${LIMIT}.log
done

# Run Chain of Mindset baseline
for DATASET in math gpqa gsm8k; do
    echo "Running CoM on $DATASET"
    python exp/baseline_com/run.py \
        --model $MODEL \
        --dataset data/${DATASET}.json \
        --seed 42 \
        --limit $LIMIT \
        --output results/baseline_com/${MODEL}_${DATASET}_seed42_limit${LIMIT}.json \
        2>&1 | tee logs/com_${MODEL}_${DATASET}_limit${LIMIT}.log
done

# Run Beta ablation
for BETA in 0.0 0.25 0.5 0.75 1.0; do
    echo "Running Beta ablation with beta=$BETA"
    python exp/cdhr_main/run_real.py \
        --model $MODEL \
        --dataset data/gsm8k.json \
        --retrieval_index data/retrieval_index.pkl \
        --seed 42 \
        --beta $BETA \
        --limit 50 \
        --output results/ablation_beta/beta${BETA}_${MODEL}_gsm8k_limit50.json \
        2>&1 | tee logs/ablation_beta_${BETA}_limit50.log
done

# Aggregate results
echo "Aggregating results..."
python aggregate_results.py

# Generate figures
echo "Generating figures..."
python generate_figures.py

echo "All experiments completed at $(date)"
