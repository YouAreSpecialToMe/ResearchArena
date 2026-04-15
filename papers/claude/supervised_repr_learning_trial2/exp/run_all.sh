#!/bin/bash
# Master experiment runner. Runs all experiments sequentially on single GPU.
# Seeds: 42, 123, 456

set -e
cd "$(dirname "$0")/.."

DATA_DIR="./data"
EPOCHS=200
TINY_EPOCHS=100
BATCH_SIZE=256
LR=0.1

echo "========================================="
echo "Starting all experiments at $(date)"
echo "========================================="

# ---- CIFAR-10 Baselines ----
for SEED in 42 123 456; do
    echo "=== CE / CIFAR-10 / seed=${SEED} ==="
    python3 exp/train.py --method ce --dataset cifar10 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --output_dir results/cifar10/ce/seed_${SEED} --data_dir ${DATA_DIR}

    echo "=== Label Smoothing / CIFAR-10 / seed=${SEED} ==="
    python3 exp/train.py --method label_smoothing --dataset cifar10 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --output_dir results/cifar10/label_smoothing/seed_${SEED} --data_dir ${DATA_DIR}

    echo "=== Mixup / CIFAR-10 / seed=${SEED} ==="
    python3 exp/train.py --method mixup --dataset cifar10 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --output_dir results/cifar10/mixup/seed_${SEED} --data_dir ${DATA_DIR}

    echo "=== CCR-adaptive / CIFAR-10 / seed=${SEED} ==="
    python3 exp/train.py --method ccr_adaptive --dataset cifar10 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir results/cifar10/ccr_adaptive/seed_${SEED} --data_dir ${DATA_DIR}
done

# ---- CIFAR-100 Baselines ----
for SEED in 42 123 456; do
    echo "=== CE / CIFAR-100 / seed=${SEED} ==="
    python3 exp/train.py --method ce --dataset cifar100 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --output_dir results/cifar100/ce/seed_${SEED} --data_dir ${DATA_DIR}

    echo "=== Label Smoothing / CIFAR-100 / seed=${SEED} ==="
    python3 exp/train.py --method label_smoothing --dataset cifar100 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --output_dir results/cifar100/label_smoothing/seed_${SEED} --data_dir ${DATA_DIR}

    echo "=== Mixup / CIFAR-100 / seed=${SEED} ==="
    python3 exp/train.py --method mixup --dataset cifar100 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --output_dir results/cifar100/mixup/seed_${SEED} --data_dir ${DATA_DIR}

    echo "=== CCR-adaptive / CIFAR-100 / seed=${SEED} ==="
    python3 exp/train.py --method ccr_adaptive --dataset cifar100 --arch resnet18 --seed ${SEED} \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --lambda_ccr 0.1 --gamma 0.1 \
        --output_dir results/cifar100/ccr_adaptive/seed_${SEED} --data_dir ${DATA_DIR}
done

# ---- TinyImageNet (seed 42 only) ----
echo "=== CE / TinyImageNet / seed=42 ==="
python3 exp/train.py --method ce --dataset tinyimagenet --arch resnet18 --seed 42 \
    --epochs ${TINY_EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --output_dir results/tinyimagenet/ce/seed_42 --data_dir ${DATA_DIR}

echo "=== Label Smoothing / TinyImageNet / seed=42 ==="
python3 exp/train.py --method label_smoothing --dataset tinyimagenet --arch resnet18 --seed 42 \
    --epochs ${TINY_EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --output_dir results/tinyimagenet/label_smoothing/seed_42 --data_dir ${DATA_DIR}

echo "=== Mixup / TinyImageNet / seed=42 ==="
python3 exp/train.py --method mixup --dataset tinyimagenet --arch resnet18 --seed 42 \
    --epochs ${TINY_EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --output_dir results/tinyimagenet/mixup/seed_42 --data_dir ${DATA_DIR}

echo "=== CCR-adaptive / TinyImageNet / seed=42 ==="
python3 exp/train.py --method ccr_adaptive --dataset tinyimagenet --arch resnet18 --seed 42 \
    --epochs ${TINY_EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --lambda_ccr 0.1 --gamma 0.1 \
    --output_dir results/tinyimagenet/ccr_adaptive/seed_42 --data_dir ${DATA_DIR}

# ---- Ablation: CCR variants on CIFAR-100 (seed 42) ----
echo "=== CCR-fixed / CIFAR-100 / seed=42 ==="
python3 exp/train.py --method ccr_fixed --dataset cifar100 --arch resnet18 --seed 42 \
    --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --lambda_ccr 0.1 --tau 1.0 \
    --output_dir results/cifar100/ccr_fixed/seed_42 --data_dir ${DATA_DIR}

echo "=== CCR-spectral / CIFAR-100 / seed=42 ==="
python3 exp/train.py --method ccr_spectral --dataset cifar100 --arch resnet18 --seed 42 \
    --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
    --lambda_ccr 0.1 \
    --output_dir results/cifar100/ccr_spectral/seed_42 --data_dir ${DATA_DIR}

# ---- Ablation: Lambda sweep on CIFAR-100 (seed 42) ----
for LAMBDA in 0.01 0.05 0.5 1.0; do
    echo "=== CCR-adaptive lambda=${LAMBDA} / CIFAR-100 / seed=42 ==="
    python3 exp/train.py --method ccr_adaptive --dataset cifar100 --arch resnet18 --seed 42 \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --lambda_ccr ${LAMBDA} --gamma 0.1 \
        --output_dir results/cifar100/ccr_adaptive_lambda_${LAMBDA}/seed_42 --data_dir ${DATA_DIR}
done

# ---- Ablation: Gamma sweep on CIFAR-100 (seed 42) ----
for GAMMA in 0.05 0.25; do
    echo "=== CCR-adaptive gamma=${GAMMA} / CIFAR-100 / seed=42 ==="
    python3 exp/train.py --method ccr_adaptive --dataset cifar100 --arch resnet18 --seed 42 \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR} \
        --lambda_ccr 0.1 --gamma ${GAMMA} \
        --output_dir results/cifar100/ccr_adaptive_gamma_${GAMMA}/seed_42 --data_dir ${DATA_DIR}
done

echo "========================================="
echo "All training complete at $(date)"
echo "========================================="
