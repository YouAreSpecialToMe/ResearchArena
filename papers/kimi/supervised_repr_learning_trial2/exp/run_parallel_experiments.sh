#!/bin/bash
# Parallel experiment execution script

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

mkdir -p results logs

echo "=========================================="
echo "LASER-SCL Parallel Experiments"
echo "Started at: $(date)"
echo "=========================================="

# Function to run experiment
run_exp() {
    local method=$1
    local dataset=$2
    local noise=$3
    local seed=$4
    local gpu=$5
    
    local logfile="logs/${method}_${dataset}_n${noise}_s${seed}.log"
    
    echo "[$method | $dataset | noise=$noise | seed=$seed] -> GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python exp/shared/train.py \
        --dataset $dataset \
        --noise_rate $noise \
        --method $method \
        --epochs 100 \
        --seed $seed \
        --save_dir results \
        > $logfile 2>&1
    
    echo "Completed: $method seed=$seed"
}

# Run critical CIFAR-100 experiments in parallel
# Using 1 GPU per experiment for stability

echo "Launching CIFAR-100 experiments (40% noise)..."

# Batch 1: Baselines (seeds 42, 123, 456)
run_exp supcon cifar100 0.4 42 0 &
run_exp supcon cifar100 0.4 123 0 &
run_exp supcon cifar100 0.4 456 0 &

run_exp supcon_lr cifar100 0.4 42 0 &
run_exp supcon_lr cifar100 0.4 123 0 &
run_exp supcon_lr cifar100 0.4 456 0 &

wait
echo "Baselines completed at $(date)"

# Batch 2: LASER-SCL (seeds 42, 123, 456)
run_exp laser_scl cifar100 0.4 42 0 &
run_exp laser_scl cifar100 0.4 123 0 &
run_exp laser_scl cifar100 0.4 456 0 &

wait
echo "LASER-SCL completed at $(date)"

# Batch 3: Ablations (seed 42 only - time constraints)
run_exp ablation_no_curriculum cifar100 0.4 42 0 &
run_exp ablation_no_elp cifar100 0.4 42 0 &
run_exp ablation_static cifar100 0.4 42 0 &

wait
echo "Ablations completed at $(date)"

echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="

# Summary
python -c "
import json
import glob
import os

print('\n=== EXPERIMENT RESULTS SUMMARY ===')
print(f'{'Method':<30} | {'Dataset':<10} | {'Noise':<6} | {'Acc':<6}')
print('-' * 70)

for f in sorted(glob.glob('results/*.json')):
    try:
        with open(f) as fp:
            data = json.load(fp)
            method = data.get('method', 'unknown')
            dataset = data.get('dataset', 'unknown')
            noise = data.get('noise_rate', 0)
            acc = data.get('final_accuracy', 0)
            print(f'{method:<30} | {dataset:<10} | {noise*100:>5.0f}% | {acc:>5.1f}%')
    except Exception as e:
        pass
"
