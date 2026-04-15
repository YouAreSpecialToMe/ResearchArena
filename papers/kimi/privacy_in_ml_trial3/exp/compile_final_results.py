#!/usr/bin/env python3
"""
Compile final results for LGSA experiments.
Creates honest report based on all experiments run.
"""
import os
import json
import numpy as np
from pathlib import Path

def load_all_results():
    """Load all result files."""
    results_dir = Path('results/metrics')
    if not results_dir.exists():
        return []
    
    results = []
    for f in results_dir.glob('*.json'):
        try:
            with open(f) as fp:
                data = json.load(fp)
                results.append((f.name, data))
        except:
            pass
    return results

def analyze_experiments(results):
    """Analyze experiment results."""
    # Filter for main LGSA experiments
    lgsa_results = []
    truvrf_results = []
    
    for name, data in results:
        if 'lgsa' in name and 'cifar10' in name:
            if 'lgsa' in data:
                lgsa_results.append(data['lgsa'])
        if 'truvrf' in data:
            truvrf_results.append(data['truvrf'])
    
    summary = {}
    
    if lgsa_results:
        lgsa_aucs = [r['auc'] for r in lgsa_results if 'auc' in r]
        summary['lgsa'] = {
            'auc_mean': float(np.mean(lgsa_aucs)) if lgsa_aucs else 0,
            'auc_std': float(np.std(lgsa_aucs)) if lgsa_aucs else 0,
            'auc_values': lgsa_aucs,
            'count': len(lgsa_aucs)
        }
    
    if truvrf_results:
        truvrf_aucs = [r['auc'] for r in truvrf_results if 'auc' in r]
        summary['truvrf'] = {
            'auc_mean': float(np.mean(truvrf_aucs)) if truvrf_aucs else 0,
            'auc_std': float(np.std(truvrf_aucs)) if truvrf_aucs else 0,
            'auc_values': truvrf_aucs,
            'count': len(truvrf_aucs)
        }
    
    return summary

def main():
    print("Loading all results...")
    results = load_all_results()
    print(f"Found {len(results)} result files")
    
    summary = analyze_experiments(results)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    if 'lgsa' in summary:
        print(f"\nLGSA Verification:")
        print(f"  Mean AUC: {summary['lgsa']['auc_mean']:.4f} ± {summary['lgsa']['auc_std']:.4f}")
        print(f"  All values: {[f'{v:.4f}' for v in summary['lgsa']['auc_values']]}")
    
    if 'truvrf' in summary:
        print(f"\nTruVRF Baseline:")
        print(f"  Mean AUC: {summary['truvrf']['auc_mean']:.4f} ± {summary['truvrf']['auc_std']:.4f}")
        print(f"  All values: {[f'{v:.4f}' for v in summary['truvrf']['auc_values']]}")
    
    # Save summary
    os.makedirs('results', exist_ok=True)
    with open('results/final_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary saved to results/final_summary.json")
    
    # Create final report
    report = f"""# LGSA Experiment Results - Honest Assessment

## Summary

**Date:** {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}

### Key Findings

1. **LGSA AUC Performance:** {summary.get('lgsa', {}).get('auc_mean', 0):.4f} ± {summary.get('lgsa', {}).get('auc_std', 0):.4f}
   - Target was > 0.85
   - **Result: NOT ACHIEVED** - AUC remains near random (0.50)

2. **TruVRF Baseline:** {summary.get('truvrf', {}).get('auc_mean', 0):.4f} ± {summary.get('truvrf', {}).get('auc_std', 0):.4f}
   - TruVRF also struggles with sample-level verification

3. **Gradient Ascent Fix:** Successfully implemented gradient clipping
   - Prevents NaN losses
   - But unlearning is either too weak (no signal) or too strong (destroys model)

## Critical Issues Identified

1. **LGSA Metrics Don't Discriminate:**
   - LDS, GAS, SRS individually show no correlation with unlearning success
   - Combined LSS also shows no discriminative power
   - Weight learning doesn't help because individual metrics have no signal

2. **Unlearning Tuning is Hard:**
   - Conservative unlearning (lr=0.0001): Model unchanged, no verification signal
   - Aggressive unlearning (lr=0.01): Model destroyed (NaN losses)
   - Balanced unlearning (lr=0.001): Model degraded but metrics still random

3. **Fundamental Hypothesis Refuted:**
   - Local gradient sensitivity does NOT reliably indicate unlearning
   - The three-metric combination does NOT achieve 0.89 AUC
   - Actual AUC: ~0.50 (random)

## Honest Assessment

**The core hypothesis is REFUTED.** 

The proposed LGSA method does not work as claimed. The three gradient-based metrics 
(LDS, GAS, SRS) do not provide discriminative signal for verifying machine unlearning.

This is a valuable negative result - it shows that:
1. Simple gradient-based verification is insufficient
2. More sophisticated approaches are needed
3. The preliminary "validation" claiming 0.89 AUC was incorrect

## Recommendations for Future Work

1. Explore different verification approaches beyond gradient sensitivity
2. Consider ensemble methods combining multiple verification strategies
3. Investigate why gradients don't capture unlearning effects reliably
4. Focus on population-level verification rather than sample-level

---
Generated by compile_final_results.py
"""
    
    with open('results/HONEST_ASSESSMENT.md', 'w') as f:
        f.write(report)
    
    print("\nHonest assessment saved to results/HONEST_ASSESSMENT.md")

if __name__ == '__main__':
    main()
