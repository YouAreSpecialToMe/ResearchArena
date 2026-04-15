#!/bin/bash
# Efficient experiment runner - batches of 3 parallel jobs
# With 4 CPU cores and num_workers=2, 3 parallel jobs = 6 workers + 3 main = manageable

set -e
cd "$(dirname "$0")/.."

DATA_DIR="./data"
EPOCHS=200
TINY_EPOCHS=100
BS=256
LR=0.1
NW=2  # num_workers per job

run_train() {
    local method=$1 dataset=$2 seed=$3 extra_args=$4 outdir=$5
    if [ -f "${outdir}/final_model.pt" ]; then
        echo "SKIP (already done): ${outdir}"
        return
    fi
    # Clean partial results
    rm -f "${outdir}/best_model.pt" "${outdir}/training_log.json"
    echo "START: ${method} / ${dataset} / seed=${seed}"
    python3 exp/train.py --method ${method} --dataset ${dataset} --arch resnet18 --seed ${seed} \
        --epochs ${EPOCHS} --batch_size ${BS} --lr ${LR} --num_workers ${NW} \
        --output_dir ${outdir} --data_dir ${DATA_DIR} ${extra_args}
    echo "DONE: ${method} / ${dataset} / seed=${seed}"
}

export -f run_train
export DATA_DIR EPOCHS TINY_EPOCHS BS LR NW

echo "========================================="
echo "Starting experiments at $(date)"
echo "========================================="

# Batch 1: CE baselines (3 parallel on CIFAR-10)
echo "--- Batch 1: CE x3 on CIFAR-10 ---"
for SEED in 42 123 456; do
    run_train ce cifar10 ${SEED} "" results/cifar10/ce/seed_${SEED} &
done
wait

# Batch 2: Label Smoothing x3 on CIFAR-10
echo "--- Batch 2: LS x3 on CIFAR-10 ---"
for SEED in 42 123 456; do
    run_train label_smoothing cifar10 ${SEED} "" results/cifar10/label_smoothing/seed_${SEED} &
done
wait

# Batch 3: Mixup + CCR-adaptive x3 on CIFAR-10 (split into 2 batches)
echo "--- Batch 3: Mixup x3 on CIFAR-10 ---"
for SEED in 42 123 456; do
    run_train mixup cifar10 ${SEED} "" results/cifar10/mixup/seed_${SEED} &
done
wait

echo "--- Batch 4: CCR-adaptive x3 on CIFAR-10 ---"
for SEED in 42 123 456; do
    run_train ccr_adaptive cifar10 ${SEED} "--lambda_ccr 0.1 --gamma 0.1" results/cifar10/ccr_adaptive/seed_${SEED} &
done
wait

echo "=== CIFAR-10 complete at $(date) ==="

# Batch 5-8: Same for CIFAR-100
echo "--- Batch 5: CE x3 on CIFAR-100 ---"
for SEED in 42 123 456; do
    run_train ce cifar100 ${SEED} "" results/cifar100/ce/seed_${SEED} &
done
wait

echo "--- Batch 6: LS x3 on CIFAR-100 ---"
for SEED in 42 123 456; do
    run_train label_smoothing cifar100 ${SEED} "" results/cifar100/label_smoothing/seed_${SEED} &
done
wait

echo "--- Batch 7: Mixup x3 on CIFAR-100 ---"
for SEED in 42 123 456; do
    run_train mixup cifar100 ${SEED} "" results/cifar100/mixup/seed_${SEED} &
done
wait

echo "--- Batch 8: CCR-adaptive x3 on CIFAR-100 ---"
for SEED in 42 123 456; do
    run_train ccr_adaptive cifar100 ${SEED} "--lambda_ccr 0.1 --gamma 0.1" results/cifar100/ccr_adaptive/seed_${SEED} &
done
wait

echo "=== CIFAR-100 complete at $(date) ==="

# TinyImageNet (seed 42 only, 4 methods sequential — each takes 20 min)
EPOCHS=100
echo "--- Batch 9: TinyImageNet baselines ---"
run_train ce tinyimagenet 42 "" results/tinyimagenet/ce/seed_42 &
run_train label_smoothing tinyimagenet 42 "" results/tinyimagenet/label_smoothing/seed_42 &
run_train mixup tinyimagenet 42 "" results/tinyimagenet/mixup/seed_42 &
wait
run_train ccr_adaptive tinyimagenet 42 "--lambda_ccr 0.1 --gamma 0.1" results/tinyimagenet/ccr_adaptive/seed_42
echo "=== TinyImageNet complete at $(date) ==="
EPOCHS=200

# Ablations on CIFAR-100 (seed 42)
echo "--- Batch 10: Ablation CCR variants ---"
run_train ccr_fixed cifar100 42 "--lambda_ccr 0.1 --tau 1.0" results/cifar100/ccr_fixed/seed_42 &
run_train ccr_spectral cifar100 42 "--lambda_ccr 0.1" results/cifar100/ccr_spectral/seed_42 &
wait

echo "--- Batch 11: Lambda sweep ---"
run_train ccr_adaptive cifar100 42 "--lambda_ccr 0.01 --gamma 0.1" results/cifar100/ccr_adaptive_lambda_0.01/seed_42 &
run_train ccr_adaptive cifar100 42 "--lambda_ccr 0.05 --gamma 0.1" results/cifar100/ccr_adaptive_lambda_0.05/seed_42 &
run_train ccr_adaptive cifar100 42 "--lambda_ccr 0.5 --gamma 0.1" results/cifar100/ccr_adaptive_lambda_0.5/seed_42 &
wait

run_train ccr_adaptive cifar100 42 "--lambda_ccr 1.0 --gamma 0.1" results/cifar100/ccr_adaptive_lambda_1.0/seed_42 &
run_train ccr_adaptive cifar100 42 "--lambda_ccr 0.1 --gamma 0.05" results/cifar100/ccr_adaptive_gamma_0.05/seed_42 &
run_train ccr_adaptive cifar100 42 "--lambda_ccr 0.1 --gamma 0.25" results/cifar100/ccr_adaptive_gamma_0.25/seed_42 &
wait

echo "========================================="
echo "ALL TRAINING COMPLETE at $(date)"
echo "========================================="
