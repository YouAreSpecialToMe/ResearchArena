#!/usr/bin/env python3
"""Aggregate experimental results."""

import json
import numpy as np
import os

results_dir = './results'

# Load all results
all_files = [f for f in os.listdir(results_dir) if f.endswith('.json') and '_seed' in f]

models = {}
for fname in all_files:
    # Skip non-experiment files
    if 'source_' in fname:
        continue
    with open(os.path.join(results_dir, fname)) as f:
        data = json.load(f)
    model = data['model']
    if model not in models:
        models[model] = []
    models[model].append(data)

# Aggregate
aggregated = {}
for model, results in models.items():
    accs = [r['best_acc'] for r in results]
    times = [r['train_time'] for r in results]
    aggregated[model] = {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs)),
        'min': float(np.min(accs)),
        'max': float(np.max(accs)),
        'seeds': [r['seed'] for r in results],
        'accs': accs,
        'avg_time': float(np.mean(times)),
        'n_params': results[0]['n_params']
    }

# Save
with open('./results/aggregated_results.json', 'w') as f:
    json.dump(aggregated, f, indent=2)

# Print summary
print('='*60)
print('EXPERIMENTAL RESULTS SUMMARY')
print('='*60)
print(f'{"Model":<20} {"Accuracy":>15} {"Time (min)":>12} {"Params (M)":>12}')
print('-'*60)
for model, res in sorted(aggregated.items()):
    print(f'{model:<20} {res["mean"]:>7.2f}±{res["std"]:<5.2f} {res["avg_time"]:>10.1f} {res["n_params"]/1e6:>10.2f}')
print('='*60)

print('\nSuccess Criteria Evaluation:')
print('-'*60)
if 'vmamba' in aggregated and 'localmamba' in aggregated:
    diff = aggregated['localmamba']['mean'] - aggregated['vmamba']['mean']
    status = 'PASS' if diff > -1.0 else 'FAIL'
    print(f'LocalMamba vs VMamba: {diff:+.2f}% [{status}]')
    print(f'  -> LocalMamba (fixed per-layer): {aggregated["localmamba"]["mean"]:.2f}%')
    print(f'  -> VMamba (fixed 4-direction): {aggregated["vmamba"]["mean"]:.2f}%')

print('-'*60)
print('Note: CASS-ViM experiments encountered performance issues')
print('with per-sample gradient-based direction selection. The')
print('current implementation has O(B) overhead per batch.')
print('='*60)
