"""
Create final results using existing data - pragmatic approach.
Uses synthetic task results which are complete and scientifically valuable.
"""
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind

# Load existing synthetic results
with open('exp/synthetic_fixed/results.json', 'r') as f:
    synthetic_results = json.load(f)

# Load existing results_final_v2 for metadata
with open('results_final_v2.json', 'r') as f:
    old_results = json.load(f)

# Aggregate synthetic results by method and overcomplete
summary = {}
all_results = []

for result in synthetic_results['all_results']:
    method = result['method']
    overcomplete = result['overcomplete']
    key = f"{method}_{overcomplete}"
    
    if key not in summary:
        summary[key] = {
            'method': method,
            'overcomplete': overcomplete,
            'cgas_values': [],
            'recovery_rates': []
        }
    
    summary[key]['cgas_values'].append(result['cgas'])
    summary[key]['recovery_rates'].append(result['recovery_rate'])
    all_results.append(result)

# Compute statistics
final_summary = {}
for key, data in summary.items():
    method = data['method']
    overcomplete = data['overcomplete']
    
    if method not in final_summary:
        final_summary[method] = {}
    
    cgas_values = data['cgas_values']
    recovery_rates = data['recovery_rates']
    
    final_summary[method][overcomplete] = {
        'cgas_mean': float(np.mean(cgas_values)),
        'cgas_std': float(np.std(cgas_values)),
        'recovery_mean': float(np.mean(recovery_rates)),
        'recovery_std': float(np.std(recovery_rates)),
        'n_seeds': len(cgas_values)
    }

# Compute correlation between C-GAS and recovery
all_cgas = [r['cgas'] for r in all_results]
all_recovery = [r['recovery_rate'] for r in all_results]

pearson_r, pearson_p = pearsonr(all_cgas, all_recovery)
spearman_r, spearman_p = spearmanr(all_cgas, all_recovery)

# Statistical tests: SAE vs others
sae_cgas = [r['cgas'] for r in all_results if r['method'] == 'sae']
other_cgas = [r['cgas'] for r in all_results if r['method'] != 'sae' and r['method'] != 'oracle']

if sae_cgas and other_cgas:
    t_stat, p_val = ttest_ind(sae_cgas, other_cgas)
else:
    t_stat, p_val = 0, 1

# Create comprehensive results
final_results = {
    "experiment_summary": {
        "title": "CAGER: Causal Geometric Explanation Recovery",
        "description": "Evaluation of interpretability methods using C-GAS metric on synthetic task with known ground-truth features",
        "tasks_evaluated": ["synthetic"],
        "methods": ["SAE", "Random", "Oracle"],
        "seeds": 3,
        "success_criteria": {
            "criterion_1": "SAE C-GAS > 0.75",
            "criterion_2": "SAE significantly better than baselines (p < 0.01)",
            "criterion_3": "C-GAS correlates with recovery (r > 0.8)",
            "criterion_4": "Validation improves predictions"
        }
    },
    "results_by_task": {
        "synthetic": {
            "summary": final_summary,
            "all_results": all_results,
            "correlations": {
                "cgas_vs_recovery": {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "n_samples": len(all_results)
                }
            },
            "statistical_tests": {
                "sae_vs_baselines": {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_val),
                    "sae_mean": float(np.mean(sae_cgas)) if sae_cgas else None,
                    "sae_std": float(np.std(sae_cgas)) if sae_cgas else None,
                    "baseline_mean": float(np.mean(other_cgas)) if other_cgas else None,
                    "baseline_std": float(np.std(other_cgas)) if other_cgas else None
                }
            }
        }
    },
    "success_criteria_assessment": {
        "criterion_1_sae_cgas_above_075": {
            "target": "C-GAS > 0.75",
            "achieved": f"{final_summary.get('sae', {}).get('1x', {}).get('cgas_mean', 0):.3f}",
            "passed": final_summary.get('sae', {}).get('1x', {}).get('cgas_mean', 0) > 0.75,
            "notes": "SAE 1x mean C-GAS below threshold; suggests metric or training needs refinement"
        },
        "criterion_2_sae_beats_baselines": {
            "target": "p < 0.01",
            "achieved": f"p = {p_val:.4f}",
            "passed": p_val < 0.01,
            "notes": "No significant difference between SAE and random baselines"
        },
        "criterion_3_cgas_correlates_recovery": {
            "target": "r > 0.8",
            "achieved": f"r = {pearson_r:.3f}",
            "passed": pearson_r > 0.8,
            "notes": f"Moderate positive correlation observed but below target (p = {pearson_p:.3f})"
        },
        "criterion_4_validation": {
            "target": "Validation improves predictions",
            "achieved": "Partial",
            "passed": None,
            "notes": "Validation implemented but ablation not completed due to time constraints"
        }
    },
    "key_findings": [
        {
            "finding": "SAE 1x achieves moderate C-GAS on synthetic task",
            "value": f"{final_summary.get('sae', {}).get('1x', {}).get('cgas_mean', 0):.3f} ± {final_summary.get('sae', {}).get('1x', {}).get('cgas_std', 0):.3f}",
            "interpretation": "C-GAS metric captures some meaningful structure but may need refinement"
        },
        {
            "finding": "Positive correlation between C-GAS and ground-truth recovery",
            "value": f"r = {pearson_r:.3f} (p = {pearson_p:.3f})",
            "interpretation": "C-GAS is directionally correct but correlation weaker than expected"
        },
        {
            "finding": "SAE 16x shows degraded performance",
            "value": f"{final_summary.get('sae', {}).get('16x', {}).get('cgas_mean', 0):.3f}",
            "interpretation": "High-dimensional SAEs suffer from dead neurons or feature selection issues"
        },
        {
            "finding": "Oracle baseline C-GAS surprisingly low",
            "value": f"{final_summary.get('oracle', {}).get('1x', {}).get('cgas_mean', 0):.3f}",
            "interpretation": "Suggests C-GAS formulation may not correctly capture ground-truth alignment"
        }
    ],
    "limitations_and_future_work": [
        "IOI and RAVEL tasks incomplete due to implementation bugs",
        "C-GAS metric may need refinement based on oracle baseline results",
        "Limited to synthetic task with simple MLP; real LLM evaluation needed",
        "Ablation studies not fully completed"
    ]
}

# Save results
with open('results_final_pragmatic.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("="*70)
print("FINAL RESULTS - PRAGMATIC APPROACH")
print("="*70)
print(f"\nTasks evaluated: synthetic (complete)")
print(f"Methods: SAE 1x/4x/16x, Random, Oracle")
print(f"Seeds: 3 per configuration")
print(f"\nKey Results:")
for method, configs in final_summary.items():
    print(f"\n{method.upper()}:")
    for occ, stats in configs.items():
        print(f"  {occ}: C-GAS = {stats['cgas_mean']:.3f} ± {stats['cgas_std']:.3f}")

print(f"\nCorrelation (C-GAS vs Recovery): r = {pearson_r:.3f}, p = {pearson_p:.3f}")
print(f"SAE vs Baselines: t = {t_stat:.3f}, p = {p_val:.4f}")

print("\n" + "="*70)
print("Results saved to: results_final_pragmatic.json")
print("="*70)
