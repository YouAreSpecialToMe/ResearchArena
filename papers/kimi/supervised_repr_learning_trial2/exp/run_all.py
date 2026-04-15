#!/usr/bin/env python3
"""
Master script to run all experiments for GC-SCL paper.
Optimized for 8-hour time limit on single A6000 GPU.
"""
import os
import sys
import subprocess
import json
import time
from datetime import datetime

# Experiment configurations - reduced to 200 epochs to fit time budget
# Each experiment should take ~10-15 minutes
EXPERIMENTS = [
    # === SUPCON BASELINE (3 seeds) ===
    # CIFAR-100 clean
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 42 --save_dir exp/supcon/cifar100_clean_s42', 'name': 'SupCon-CIFAR100-clean-s42', 'priority': 1},
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 43 --save_dir exp/supcon/cifar100_clean_s43', 'name': 'SupCon-CIFAR100-clean-s43', 'priority': 1},
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 44 --save_dir exp/supcon/cifar100_clean_s44', 'name': 'SupCon-CIFAR100-clean-s44', 'priority': 1},
    
    # CIFAR-100 20% symmetric noise
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 42 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/supcon/cifar100_sym20_s42', 'name': 'SupCon-CIFAR100-sym20-s42', 'priority': 1},
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 43 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/supcon/cifar100_sym20_s43', 'name': 'SupCon-CIFAR100-sym20-s43', 'priority': 1},
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 44 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/supcon/cifar100_sym20_s44', 'name': 'SupCon-CIFAR100-sym20-s44', 'priority': 1},
    
    # === GC-SCL FULL METHOD (3 seeds) ===
    # CIFAR-100 clean
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --save_dir exp/gcscl/cifar100_clean_s42', 'name': 'GCSCL-CIFAR100-clean-s42', 'priority': 1},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 43 --save_dir exp/gcscl/cifar100_clean_s43', 'name': 'GCSCL-CIFAR100-clean-s43', 'priority': 1},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 44 --save_dir exp/gcscl/cifar100_clean_s44', 'name': 'GCSCL-CIFAR100-clean-s44', 'priority': 1},
    
    # CIFAR-100 20% symmetric noise
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/gcscl/cifar100_sym20_s42', 'name': 'GCSCL-CIFAR100-sym20-s42', 'priority': 1},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 43 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/gcscl/cifar100_sym20_s43', 'name': 'GCSCL-CIFAR100-sym20-s43', 'priority': 1},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 44 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/gcscl/cifar100_sym20_s44', 'name': 'GCSCL-CIFAR100-sym20-s44', 'priority': 1},
    
    # === ADDITIONAL NOISE LEVELS (GC-SCL only for comparison) ===
    # CIFAR-100 40% symmetric noise
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --noise_type symmetric --noise_ratio 0.4 --save_dir exp/gcscl/cifar100_sym40_s42', 'name': 'GCSCL-CIFAR100-sym40-s42', 'priority': 2},
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 512 --seed 42 --noise_type symmetric --noise_ratio 0.4 --save_dir exp/supcon/cifar100_sym40_s42', 'name': 'SupCon-CIFAR100-sym40-s42', 'priority': 2},
    
    # === ABLATION STUDIES (single seed) ===
    # Ablation: No velocity tracking
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --no_velocity --save_dir exp/gcscl_ablation_align/cifar100_clean_s42', 'name': 'GCSCL-abl-no-velocity-clean', 'priority': 3},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --noise_type symmetric --noise_ratio 0.2 --no_velocity --save_dir exp/gcscl_ablation_align/cifar100_sym20_s42', 'name': 'GCSCL-abl-no-velocity-sym20', 'priority': 3},
    
    # Ablation: No curriculum
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --no_curriculum --save_dir exp/gcscl_ablation_nocurriculum/cifar100_clean_s42', 'name': 'GCSCL-abl-no-curriculum-clean', 'priority': 3},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --noise_type symmetric --noise_ratio 0.2 --no_curriculum --save_dir exp/gcscl_ablation_nocurriculum/cifar100_sym20_s42', 'name': 'GCSCL-abl-no-curriculum-sym20', 'priority': 3},
    
    # Ablation: Loss-based weighting
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --use_loss_weighting --save_dir exp/gcscl_ablation_loss/cifar100_clean_s42', 'name': 'GCSCL-abl-loss-weight-clean', 'priority': 3},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar100 --epochs 200 --batch_size 256 --seed 42 --noise_type symmetric --noise_ratio 0.2 --use_loss_weighting --save_dir exp/gcscl_ablation_loss/cifar100_sym20_s42', 'name': 'GCSCL-abl-loss-weight-sym20', 'priority': 3},
    
    # === CROSS ENTROPY BASELINE ===
    {'script': 'exp/cross_entropy/train.py', 'args': '--dataset cifar100 --epochs 200 --seed 42 --save_dir exp/cross_entropy/cifar100_clean_s42', 'name': 'CE-CIFAR100-clean', 'priority': 4},
    {'script': 'exp/cross_entropy/train.py', 'args': '--dataset cifar100 --epochs 200 --seed 42 --noise_type symmetric --noise_ratio 0.2 --save_dir exp/cross_entropy/cifar100_sym20_s42', 'name': 'CE-CIFAR100-sym20', 'priority': 4},
    
    # === CIFAR-10 (1 seed each for comparison) ===
    {'script': 'exp/supcon/train.py', 'args': '--dataset cifar10 --epochs 200 --batch_size 512 --seed 42 --save_dir exp/supcon/cifar10_clean_s42', 'name': 'SupCon-CIFAR10-clean-s42', 'priority': 5},
    {'script': 'exp/gcscl/train.py', 'args': '--dataset cifar10 --epochs 200 --batch_size 256 --seed 42 --save_dir exp/gcscl/cifar10_clean_s42', 'name': 'GCSCL-CIFAR10-clean-s42', 'priority': 5},
]


def run_experiment(exp, exp_id, total):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"[{exp_id}/{total}] Running: {exp['name']} (Priority: {exp['priority']})")
    print(f"{'='*80}")
    
    cmd = f"cd /home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/supervised_representation_learning/idea_02 && source .venv/bin/activate && python {exp['script']} {exp['args']}"
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, executable='/bin/bash')
    elapsed = time.time() - start_time
    
    status = 'SUCCESS' if result.returncode == 0 else 'FAILED'
    print(f"\nExperiment {exp['name']}: {status} (took {elapsed/60:.1f} minutes)")
    
    return {
        'name': exp['name'],
        'status': status,
        'elapsed_minutes': elapsed / 60,
        'priority': exp['priority']
    }


def main():
    print(f"Starting experiment suite at {datetime.now()}")
    
    # Sort by priority
    EXPERIMENTS.sort(key=lambda x: x['priority'])
    
    print(f"Total experiments: {len(EXPERIMENTS)}")
    print(f"Priority 1 (main comparison): {sum(1 for e in EXPERIMENTS if e['priority']==1)} experiments")
    print(f"Priority 2 (noise levels): {sum(1 for e in EXPERIMENTS if e['priority']==2)} experiments")
    print(f"Priority 3 (ablations): {sum(1 for e in EXPERIMENTS if e['priority']==3)} experiments")
    print(f"Priority 4+ (baselines): {sum(1 for e in EXPERIMENTS if e['priority']>=4)} experiments")
    
    # Create necessary directories
    for subdir in ['exp/supcon', 'exp/gcscl', 'exp/gcscl_ablation_align', 
                   'exp/gcscl_ablation_loss', 'exp/gcscl_ablation_nocurriculum',
                   'exp/cross_entropy', 'results', 'figures']:
        os.makedirs(subdir, exist_ok=True)
    
    results = []
    start_time = time.time()
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        # Check time budget (7 hours to leave room for analysis)
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours > 7.0:
            print(f"\n{'='*80}")
            print(f"Approaching time limit ({elapsed_hours:.1f} hours elapsed). Stopping experiments.")
            print(f"{'='*80}")
            break
        
        result = run_experiment(exp, i, len(EXPERIMENTS))
        results.append(result)
        
        # Save progress
        with open('results/experiment_progress.json', 'w') as f:
            json.dump({
                'completed': results,
                'remaining': len(EXPERIMENTS) - i,
                'elapsed_hours': elapsed_hours,
                'timestamp': str(datetime.now())
            }, f, indent=2)
    
    total_time = (time.time() - start_time) / 3600
    print(f"\n{'='*80}")
    print(f"Experiment batch completed in {total_time:.2f} hours")
    print(f"{'='*80}")
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nSuccess: {success_count}/{len(results)}")
    
    for priority in sorted(set(r['priority'] for r in results)):
        priority_results = [r for r in results if r['priority'] == priority]
        priority_success = sum(1 for r in priority_results if r['status'] == 'SUCCESS')
        print(f"\nPriority {priority}: {priority_success}/{len(priority_results)}")
        for r in priority_results:
            status_icon = "✓" if r['status'] == 'SUCCESS' else "✗"
            print(f"  {status_icon} {r['name']} ({r['elapsed_minutes']:.1f} min)")


if __name__ == '__main__':
    main()
