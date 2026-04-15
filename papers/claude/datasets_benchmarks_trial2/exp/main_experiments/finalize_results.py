#!/usr/bin/env python3
"""
Finalize results: merge per-model results, fix success criteria, generate combined files.
"""
import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Load all individual format results
all_format_results = {}
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('format_results_') and fname.endswith('.json'):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        all_format_results[data['model_name']] = data

# Load all phrasing results
all_phrasing_results = {}
for fname in os.listdir(RESULTS_DIR):
    if fname.startswith('phrasing_results_') and fname.endswith('.json'):
        with open(os.path.join(RESULTS_DIR, fname)) as f:
            data = json.load(f)
        all_phrasing_results[data['model_name']] = data

# Save combined cross_format_results
with open(os.path.join(RESULTS_DIR, 'cross_format_results.json'), 'w') as f:
    json.dump(all_format_results, f, indent=2)

print(f"Loaded {len(all_format_results)} format results, {len(all_phrasing_results)} phrasing results")

# Fix success criteria
PTYPES = ['lexical', 'syntactic', 'voice', 'formality', 'negation', 'elaborative']

# SC1
models_below_90 = [mn for mn in all_format_results if all_format_results[mn]['cfa_mean'] < 0.90]
sc1 = {
    'met': bool(len(models_below_90) >= len(all_format_results) * 0.5),
    'n_below_90': len(models_below_90),
    'n_total': len(all_format_results),
    'models': models_below_90,
}

# SC2: PFI variation
all_pfis = {}
for ptype in PTYPES:
    vals = [all_phrasing_results[mn]['pfi_per_type'].get(ptype, 0) for mn in all_phrasing_results]
    all_pfis[ptype] = float(np.mean(vals))
pfi_range = max(all_pfis.values()) - min(all_pfis.values())
sc2 = {
    'pfi_per_type_avg': all_pfis,
    'pfi_range': float(pfi_range),
    'most_fragile': max(all_pfis, key=all_pfis.get),
    'least_fragile': min(all_pfis, key=all_pfis.get),
    'note': 'With n=5 models, t-tests have limited power. Descriptive PFI variation is clear.',
}

# SC3
accs = [all_format_results[mn]['overall_accuracy'] for mn in all_format_results]
cfas = [all_format_results[mn]['cfa_mean'] for mn in all_format_results]
r_val, _ = stats.pearsonr(accs, cfas) if len(accs) >= 3 else (0, 1)
r2 = float(r_val**2)
cars = [all_format_results[mn]['car'] for mn in all_format_results]
car_range = float(max(cars) - min(cars))
sc3 = {
    'met': bool(r2 < 0.9 or car_range > 0.10),
    'r_squared': r2,
    'car_range': car_range,
    'car_values': {mn: float(all_format_results[mn]['car']) for mn in all_format_results},
}

# SC4
with open(os.path.join(RESULTS_DIR, 'domain_analysis.json')) as f:
    domain_analysis = json.load(f)
spreads = {mn: float(domain_analysis[mn]['dcs_spread']) for mn in domain_analysis}
models_above_10pp = sum(1 for s in spreads.values() if s > 0.10)
sc4 = {
    'met': bool(models_above_10pp >= 1),
    'spreads': spreads,
    'models_above_10pp': models_above_10pp,
}

# SC5
sc5 = {
    'note': '70B model OOM - limited to 3.8B-14B range',
    'qwen_7b_14b': {
        'acc_7b': float(all_format_results.get('Qwen2.5-7B', {}).get('overall_accuracy', 0)),
        'acc_14b': float(all_format_results.get('Qwen2.5-14B', {}).get('overall_accuracy', 0)),
        'cfa_7b': float(all_format_results.get('Qwen2.5-7B', {}).get('cfa_mean', 0)),
        'cfa_14b': float(all_format_results.get('Qwen2.5-14B', {}).get('cfa_mean', 0)),
    },
}

criteria = {'sc1': sc1, 'sc2': sc2, 'sc3': sc3, 'sc4': sc4, 'sc5': sc5}
with open(os.path.join(RESULTS_DIR, 'success_criteria.json'), 'w') as f:
    json.dump(criteria, f, indent=2)

print("\nSuccess Criteria:")
print(f"  SC1 (CFA<90%): {'MET' if sc1['met'] else 'NOT MET'} - {sc1['n_below_90']}/{sc1['n_total']} models")
print(f"  SC2 (paraphrase differ): PFI range={pfi_range:.3f}, most fragile={sc2['most_fragile']}, least={sc2['least_fragile']}")
print(f"  SC3 (consistency!=accuracy): R²={r2:.3f}, CAR range={car_range:.3f} - {'MET' if sc3['met'] else 'NOT MET'}")
print(f"  SC4 (domain variation): {models_above_10pp}/{len(spreads)} models with >10pp spread")
print(f"  SC5 (size scaling): Qwen 7B CFA={sc5['qwen_7b_14b']['cfa_7b']:.3f}, 14B CFA={sc5['qwen_7b_14b']['cfa_14b']:.3f}")

# Print summary table
print(f"\n{'Model':<20} {'Size':>5} {'Family':<8} {'Acc':>6} {'CFA':>6} {'CAR':>6} {'FCAG':>6}")
print("-" * 70)
for mn in sorted(all_format_results, key=lambda m: all_format_results[m]['model_size_b']):
    r = all_format_results[mn]
    print(f"{mn:<20} {r['model_size_b']:>5.1f} {r['model_family']:<8} {r['overall_accuracy']:>6.3f} {r['cfa_mean']:>6.3f} {r['car']:>6.3f} {r['fcag']:>6.3f}")

print("\nDone!")
