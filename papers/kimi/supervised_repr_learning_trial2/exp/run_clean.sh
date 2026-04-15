#!/bin/bash
cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

echo "Starting LASER-SCL experiments at $(date)"

# Critical experiments only (to fit in time budget)
EXPERIMENTS=(
    "supcon 42"
    "supcon 123" 
    "supcon 456"
    "supcon_lr 42"
    "supcon_lr 123"
    "supcon_lr 456"
    "laser_scl 42"
    "laser_scl 123"
    "laser_scl 456"
)

for exp in "${EXPERIMENTS[@]}"; do
    read method seed <<< "$exp"
    echo ""
    echo "[$(date)] Running $method with seed $seed..."
    
    python exp/shared/train.py \
        --dataset cifar100 \
        --noise_rate 0.4 \
        --method $method \
        --epochs 50 \
        --seed $seed \
        --save_dir results \
        2>&1 | tail -5
    
    echo "[$(date)] Completed $method seed $seed"
done

echo ""
echo "All experiments completed at $(date)"
