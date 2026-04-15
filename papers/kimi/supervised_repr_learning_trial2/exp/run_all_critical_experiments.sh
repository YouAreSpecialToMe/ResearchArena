#!/bin/bash
# Master script to run all critical experiments for LASER-SCL
# Reduced scope: 100 epochs, 1 seed, focused on CIFAR-100 with 40% noise

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

mkdir -p results logs

echo "=========================================="
echo "LASER-SCL Critical Experiments"
echo "Started at: $(date)"
echo "=========================================="

# 1. ELP Validation on CIFAR-100 (40% noise, 100 epochs)
echo "[1/7] Running ELP validation on CIFAR-100..."
python exp/elp_validation/run.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/elp_validation_cifar100.log

# 2. SupCon Vanilla on CIFAR-100 (40% noise, 100 epochs)
echo "[2/7] Running SupCon vanilla on CIFAR-100..."
python exp/shared/train.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --method supcon \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/supcon_vanilla_cifar100.log

# 3. SupCon + Loss Reweighting on CIFAR-100 (40% noise, 100 epochs)
echo "[3/7] Running SupCon+LR on CIFAR-100..."
python exp/shared/train.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --method supcon_lr \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/supcon_lr_cifar100.log

# 4. LASER-SCL Full on CIFAR-100 (40% noise, 100 epochs)
echo "[4/7] Running LASER-SCL on CIFAR-100..."
python exp/shared/train.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --method laser_scl \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/laser_scl_cifar100.log

# 5. Ablation: No Curriculum on CIFAR-100 (40% noise, 100 epochs)
echo "[5/7] Running Ablation (No Curriculum) on CIFAR-100..."
python exp/shared/train.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --method ablation_no_curriculum \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/ablation_no_curriculum_cifar100.log

# 6. Ablation: No ELP on CIFAR-100 (40% noise, 100 epochs)
echo "[6/7] Running Ablation (No ELP) on CIFAR-100..."
python exp/shared/train.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --method ablation_no_elp \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/ablation_no_elp_cifar100.log

# 7. Ablation: Static on CIFAR-100 (40% noise, 100 epochs)
echo "[7/7] Running Ablation (Static) on CIFAR-100..."
python exp/shared/train.py \
    --dataset cifar100 \
    --noise_rate 0.4 \
    --method ablation_static \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/ablation_static_cifar100.log

# Also run CIFAR-10 experiments for comparison (faster)
echo "[8/10] Running SupCon vanilla on CIFAR-10..."
python exp/shared/train.py \
    --dataset cifar10 \
    --noise_rate 0.4 \
    --method supcon \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/supcon_vanilla_cifar10.log

echo "[9/10] Running SupCon+LR on CIFAR-10..."
python exp/shared/train.py \
    --dataset cifar10 \
    --noise_rate 0.4 \
    --method supcon_lr \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/supcon_lr_cifar10.log

echo "[10/10] Running LASER-SCL on CIFAR-10..."
python exp/shared/train.py \
    --dataset cifar10 \
    --noise_rate 0.4 \
    --method laser_scl \
    --epochs 100 \
    --seed 42 \
    --save_dir results \
    2>&1 | tee logs/laser_scl_cifar10.log

echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="

# Generate results summary
python -c "
import json
import glob
import os

results = {}
for f in glob.glob('results/*.json'):
    try:
        with open(f) as fp:
            data = json.load(fp)
            results[os.path.basename(f)] = data
    except:
        pass

print('\n=== EXPERIMENT RESULTS SUMMARY ===')
for name, data in sorted(results.items()):
    method = data.get('method', 'unknown')
    dataset = data.get('dataset', 'unknown')
    noise = data.get('noise_rate', 0)
    acc = data.get('final_accuracy', 0)
    print(f'{method:25s} | {dataset:10s} | {noise*100:4.0f}% | {acc:6.2f}%')
"
