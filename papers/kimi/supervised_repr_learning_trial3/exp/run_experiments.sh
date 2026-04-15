#!/bin/bash
# Run all experiments sequentially

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_3/supervised_representation_learning/idea_01
source .venv/bin/activate

mkdir -p logs results

echo "Starting experiment suite..."
echo "==========================="

# Function to run experiment and wait
run_exp() {
    local name=$1
    local cmd=$2
    echo ""
    echo "[$name] Starting..."
    echo "Command: $cmd"
    eval "$cmd"
    wait
    echo "[$name] Completed!"
}

# 1. Cross-Entropy baseline seeds
run_exp "CE-42" "python exp/cifar100_crossentropy/run.py --seed 42 --epochs 100 --batch_size 512 --save_dir ./results > logs/ce_seed42.log 2>&1"
run_exp "CE-123" "python exp/cifar100_crossentropy/run.py --seed 123 --epochs 100 --batch_size 512 --save_dir ./results > logs/ce_seed123.log 2>&1"

# 2. SupCon baseline seeds
run_exp "SupCon-42" "python exp/cifar100_supcon/run.py --seed 42 --epochs 100 --batch_size 512 --save_dir ./results > logs/supcon_seed42.log 2>&1"
run_exp "SupCon-123" "python exp/cifar100_supcon/run.py --seed 123 --epochs 100 --batch_size 512 --save_dir ./results > logs/supcon_seed123.log 2>&1"

# 3. JD-CCL fixed seeds
run_exp "JD-CCL-42" "python exp/cifar100_jdccl/run.py --seed 42 --epochs 100 --batch_size 512 --save_dir ./results > logs/jdccl_seed42.log 2>&1"
run_exp "JD-CCL-123" "python exp/cifar100_jdccl/run.py --seed 123 --epochs 100 --batch_size 512 --save_dir ./results > logs/jdccl_seed123.log 2>&1"

# 4. CAG-HNM seeds
run_exp "CAG-HNM-42" "python exp/cifar100_caghnm/run.py --seed 42 --epochs 100 --batch_size 512 --save_dir ./results > logs/caghnm_seed42.log 2>&1"
run_exp "CAG-HNM-123" "python exp/cifar100_caghnm/run.py --seed 123 --epochs 100 --batch_size 512 --save_dir ./results > logs/caghnm_seed123.log 2>&1"
run_exp "CAG-HNM-456" "python exp/cifar100_caghnm/run.py --seed 456 --epochs 100 --batch_size 512 --save_dir ./results > logs/caghnm_seed456.log 2>&1"

# 5. Ablation study
run_exp "Ablation" "python exp/ablation_fixed_vs_curriculum/run.py --seed 42 --epochs 100 --batch_size 512 --save_dir ./results > logs/ablation.log 2>&1"

echo ""
echo "==========================="
echo "All experiments completed!"
