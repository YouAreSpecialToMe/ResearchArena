#!/bin/bash
# Sequential experiment execution for single GPU
# Reduces epochs to fit within 8-hour budget

cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_01
source .venv/bin/activate

mkdir -p results logs

echo "=========================================="
echo "LASER-SCL Sequential Experiments (Fast)"
echo "Started at: $(date)"
echo "=========================================="

# Run experiment function
run_exp() {
    local method=$1
    local dataset=$2
    local noise=$3
    local seed=$4
    local epochs=$5
    
    local logfile="logs/${method}_${dataset}_n${noise}_s${seed}.log"
    local start_time=$(date +%s)
    
    echo ""
    echo "[$EXP_NUM/$TOTAL_EXPS] Running $method on $dataset"
    echo "    noise=$noise, seed=$seed, epochs=$epochs"
    echo "    Started at: $(date)"
    
    python exp/shared/train.py \
        --dataset $dataset \
        --noise_rate $noise \
        --method $method \
        --epochs $epochs \
        --seed $seed \
        --save_dir results \
        2>&1 | tee $logfile | tail -20
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    
    echo "    Completed in ${minutes}m at: $(date)"
    ((EXP_NUM++))
}

# Configuration: 50 epochs per experiment (~1 hour each)
EPOCHS=50
EXP_NUM=1

# Count total experiments
TOTAL_EXPS=9  # 3 methods x 3 seeds

# Run critical experiments
echo ""
echo "=== Phase 1: SupCon Baseline ==="
run_exp supcon cifar100 0.4 42 $EPOCHS
run_exp supcon cifar100 0.4 123 $EPOCHS
run_exp supcon cifar100 0.4 456 $EPOCHS

echo ""
echo "=== Phase 2: SupCon+LR Baseline ==="
run_exp supcon_lr cifar100 0.4 42 $EPOCHS
run_exp supcon_lr cifar100 0.4 123 $EPOCHS
run_exp supcon_lr cifar100 0.4 456 $EPOCHS

echo ""
echo "=== Phase 3: LASER-SCL (Our Method) ==="
run_exp laser_scl cifar100 0.4 42 $EPOCHS
run_exp laser_scl cifar100 0.4 123 $EPOCHS
run_exp laser_scl cifar100 0.4 456 $EPOCHS

echo ""
echo "=========================================="
echo "Main experiments completed at: $(date)"
echo "=========================================="

# Run ablations if time permits (checked by calling script)
echo ""
echo "=== Phase 4: Ablations ==="
run_exp ablation_no_curriculum cifar100 0.4 42 $EPOCHS
run_exp ablation_no_elp cifar100 0.4 42 $EPOCHS
run_exp ablation_static cifar100 0.4 42 $EPOCHS

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED at: $(date)"
echo "=========================================="

# Generate results summary
python3 << 'PYEOF'
import json
import glob
import os

print('\n' + '='*70)
print('EXPERIMENT RESULTS SUMMARY')
print('='*70)
print(f"{'Method':<30} | {'Dataset':<10} | {'Noise':<6} | {'Seed':<5} | {'Accuracy':<8}")
print('-'*70)

results = []
for f in sorted(glob.glob('results/*.json')):
    try:
        with open(f) as fp:
            data = json.load(fp)
            method = data.get('method', 'unknown')
            dataset = data.get('dataset', 'unknown')
            noise = data.get('noise_rate', 0)
            seed = data.get('seed', 0)
            acc = data.get('final_accuracy', 0)
            results.append({
                'method': method,
                'dataset': dataset,
                'noise': noise,
                'seed': seed,
                'accuracy': acc,
                'file': os.path.basename(f)
            })
            print(f"{method:<30} | {dataset:<10} | {noise*100:>5.0f}% | {seed:>4} | {acc:>7.2f}%")
    except Exception as e:
        pass

# Compute statistics by method
print('\n' + '-'*70)
print('STATISTICS BY METHOD (mean ± std)')
print('-'*70)

from collections import defaultdict
import statistics

method_accs = defaultdict(list)
for r in results:
    method_accs[r['method']].append(r['accuracy'])

for method in sorted(method_accs.keys()):
    accs = method_accs[method]
    if len(accs) > 1:
        mean_acc = statistics.mean(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
        print(f"{method:<30}: {mean_acc:>6.2f}% ± {std_acc:>4.2f}% (n={len(accs)})")
    else:
        print(f"{method:<30}: {accs[0]:>6.2f}% (n=1)")

# Success criterion check
print('\n' + '='*70)
print('SUCCESS CRITERION CHECK')
print('='*70)

if 'laser_scl' in method_accs and 'supcon_lr' in method_accs:
    laser_mean = statistics.mean(method_accs['laser_scl'])
    lr_mean = statistics.mean(method_accs['supcon_lr'])
    diff = laser_mean - lr_mean
    
    print(f"LASER-SCL mean accuracy: {laser_mean:.2f}%")
    print(f"SupCon+LR mean accuracy: {lr_mean:.2f}%")
    print(f"Difference: {diff:+.2f}%")
    print(f"Target (≥2%): {'PASS ✓' if diff >= 2 else 'FAIL ✗'}")

PYEOF

echo ""
echo "Results saved to results/"
echo "Logs saved to logs/"
