#!/bin/bash
cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

echo "=========================================="
echo "LASER-SCL Sequential Experiments"
echo "Started at: $(date)"
echo "=========================================="

# Run each experiment sequentially
EXPERIMENTS=(
    "supcon cifar100 0.4 42"
    "supcon cifar100 0.4 123"
    "supcon cifar100 0.4 456"
    "supcon_lr cifar100 0.4 42"
    "supcon_lr cifar100 0.4 123"
    "supcon_lr cifar100 0.4 456"
    "laser_scl cifar100 0.4 42"
    "laser_scl cifar100 0.4 123"
    "laser_scl cifar100 0.4 456"
)

i=1
for exp in "${EXPERIMENTS[@]}"; do
    read method dataset noise seed <<< "$exp"
    echo ""
    echo "[$i/9] Running $method on $dataset (noise=$noise, seed=$seed)..."
    
    python exp/shared/train.py \
        --dataset $dataset \
        --noise_rate $noise \
        --method $method \
        --epochs 100 \
        --seed $seed \
        --save_dir results \
        2>&1 | tail -5
    
    echo "Completed at $(date)"
    ((i++))
done

echo ""
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
