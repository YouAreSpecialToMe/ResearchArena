"""
Phase 1: Run experiments on Asia network (smallest, fastest).
Use this to verify the pipeline works before scaling up.
"""
import json
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from shared.data_loader import load_dataset, load_ground_truth
from shared.metrics import evaluate_pc_discovery, evaluate_mb_discovery
from ait_lcd.ait_lcd_fast import ait_lcd_learn_fast
from baselines.baseline_wrappers import SimpleIAMB, AdaptiveIAMB
from baselines.hiton_mb import HITONMB
from baselines.pcmb import PCMB
import time

print("="*60)
print("Phase 1: Asia Network Experiments")
print("="*60)

# Configuration
NETWORK = 'asia'
SAMPLE_SIZES = [100, 200, 500, 1000]
SEEDS = [1, 2, 3]
ALGORITHMS = ['AIT-LCD', 'IAMB', 'HITON-MB', 'PCMB', 'EAMB-inspired']

gt = load_ground_truth(NETWORK)
results = []

for n_samples in SAMPLE_SIZES:
    for seed in SEEDS:
        print(f"\nRunning: n={n_samples}, seed={seed}")
        data = load_dataset(NETWORK, n_samples, seed)
        
        # Test on all 8 nodes in Asia
        for target in gt['nodes']:
            true_pc = gt['pc_sets'][target]
            true_mb = gt['mb_sets'][target]
            
            # AIT-LCD
            try:
                t0 = time.time()
                result = ait_lcd_learn_fast(data, target, alpha=0.2, beta=10)
                runtime = time.time() - t0
                pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                results.append({
                    'network': NETWORK, 'n_samples': n_samples, 'seed': seed,
                    'target': target, 'algorithm': 'AIT-LCD',
                    'pc_f1': pc_m['f1'], 'mb_f1': mb_m['f1'],
                    'runtime': runtime
                })
            except Exception as e:
                print(f"  AIT-LCD error on {target}: {e}")
            
            # IAMB
            try:
                t0 = time.time()
                algo = SimpleIAMB(alpha=0.05)
                result = algo.fit(data, target)
                runtime = time.time() - t0
                pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                results.append({
                    'network': NETWORK, 'n_samples': n_samples, 'seed': seed,
                    'target': target, 'algorithm': 'IAMB',
                    'pc_f1': pc_m['f1'], 'mb_f1': mb_m['f1'],
                    'runtime': runtime
                })
            except Exception as e:
                print(f"  IAMB error on {target}: {e}")
            
            # HITON-MB
            try:
                t0 = time.time()
                algo = HITONMB(alpha=0.05, max_k=2)
                result = algo.fit(data, target)
                runtime = time.time() - t0
                pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                results.append({
                    'network': NETWORK, 'n_samples': n_samples, 'seed': seed,
                    'target': target, 'algorithm': 'HITON-MB',
                    'pc_f1': pc_m['f1'], 'mb_f1': mb_m['f1'],
                    'runtime': runtime
                })
            except Exception as e:
                print(f"  HITON-MB error on {target}: {e}")
            
            # PCMB
            try:
                t0 = time.time()
                algo = PCMB(alpha=0.05, max_k=2)
                result = algo.fit(data, target)
                runtime = time.time() - t0
                pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                results.append({
                    'network': NETWORK, 'n_samples': n_samples, 'seed': seed,
                    'target': target, 'algorithm': 'PCMB',
                    'pc_f1': pc_m['f1'], 'mb_f1': mb_m['f1'],
                    'runtime': runtime
                })
            except Exception as e:
                print(f"  PCMB error on {target}: {e}")
            
            # EAMB-inspired
            try:
                t0 = time.time()
                algo = AdaptiveIAMB(alpha=0.05)
                result = algo.fit(data, target)
                runtime = time.time() - t0
                pc_m = evaluate_pc_discovery(result['pc'], true_pc)
                mb_m = evaluate_mb_discovery(result['mb'], true_mb)
                results.append({
                    'network': NETWORK, 'n_samples': n_samples, 'seed': seed,
                    'target': target, 'algorithm': 'EAMB-inspired',
                    'pc_f1': pc_m['f1'], 'mb_f1': mb_m['f1'],
                    'runtime': runtime
                })
            except Exception as e:
                print(f"  EAMB error on {target}: {e}")

# Save results
import os
os.makedirs('results', exist_ok=True)
with open('results/phase1_asia.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("Phase 1 Complete!")
print("="*60)

for algo in ALGORITHMS:
    algo_results = [r for r in results if r['algorithm'] == algo]
    if algo_results:
        pc_f1 = [r['pc_f1'] for r in algo_results]
        runtime = [r['runtime'] for r in algo_results]
        print(f"{algo}: PC F1={np.mean(pc_f1):.3f}±{np.std(pc_f1):.3f}, "
              f"Time={np.mean(runtime):.3f}s")
