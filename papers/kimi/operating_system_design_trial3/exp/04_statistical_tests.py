"""
Statistical significance testing for KAPHE vs baselines.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
import pandas as pd
import numpy as np
from scipy import stats

def main():
    print("=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 60)
    
    exp_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp'
    
    # Load prediction results
    kaphe_preds = pd.read_csv(f'{exp_dir}/kaphe/predictions_v3.csv')
    default_preds = pd.read_csv(f'{exp_dir}/baseline_default/results.csv')
    expert_preds = pd.read_csv(f'{exp_dir}/baseline_expert/results.csv')
    mlkaps_preds = pd.read_csv(f'{exp_dir}/baseline_mlkaps/results.csv')
    
    # Get normalized scores
    kaphe_scores = kaphe_preds['normalized_score'].values
    default_scores = default_preds['normalized_score'].values
    expert_scores = expert_preds['normalized_score'].values
    mlkaps_scores = mlkaps_preds['dt_normalized'].values
    
    print("\nPaired Statistical Tests:")
    print("-" * 60)
    
    results = []
    
    # Test 1: KAPHE vs Default
    diff = kaphe_scores - default_scores
    t_stat, p_ttest = stats.ttest_rel(kaphe_scores, default_scores)
    _, p_wilcoxon = stats.wilcoxon(kaphe_scores, default_scores)
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
    
    print(f"\n1. KAPHE vs Default:")
    print(f"   Mean improvement: {np.mean(diff):.4f} ({np.mean(diff)/np.mean(default_scores)*100:+.1f}%)")
    print(f"   t-test p-value: {p_ttest:.2e}")
    print(f"   Wilcoxon p-value: {p_wilcoxon:.2e}")
    print(f"   Cohen's d: {cohens_d:.3f}")
    print(f"   Significant (p<0.05): {'YES ✓' if p_ttest < 0.05 else 'NO'}")
    
    results.append({
        'comparison': 'KAPHE vs Default',
        'mean_diff': float(np.mean(diff)),
        'p_value_ttest': float(p_ttest),
        'p_value_wilcoxon': float(p_wilcoxon),
        'cohens_d': float(cohens_d),
        'significant': bool(p_ttest < 0.05),
    })
    
    # Test 2: KAPHE vs Expert
    diff = kaphe_scores - expert_scores
    t_stat, p_ttest = stats.ttest_rel(kaphe_scores, expert_scores)
    _, p_wilcoxon = stats.wilcoxon(kaphe_scores, expert_scores)
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
    
    print(f"\n2. KAPHE vs Expert Heuristics:")
    print(f"   Mean improvement: {np.mean(diff):.4f} ({np.mean(diff)/np.mean(expert_scores)*100:+.1f}%)")
    print(f"   t-test p-value: {p_ttest:.2e}")
    print(f"   Wilcoxon p-value: {p_wilcoxon:.2e}")
    print(f"   Cohen's d: {cohens_d:.3f}")
    print(f"   Significant (p<0.05): {'YES ✓' if p_ttest < 0.05 else 'NO'}")
    
    results.append({
        'comparison': 'KAPHE vs Expert',
        'mean_diff': float(np.mean(diff)),
        'p_value_ttest': float(p_ttest),
        'p_value_wilcoxon': float(p_wilcoxon),
        'cohens_d': float(cohens_d),
        'significant': bool(p_ttest < 0.05),
    })
    
    # Test 3: KAPHE vs MLKAPS
    diff = kaphe_scores - mlkaps_scores
    t_stat, p_ttest = stats.ttest_rel(kaphe_scores, mlkaps_scores)
    _, p_wilcoxon = stats.wilcoxon(kaphe_scores, mlkaps_scores)
    cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
    
    print(f"\n3. KAPHE vs MLKAPS:")
    print(f"   Mean difference: {np.mean(diff):.4f} ({np.mean(diff)/np.mean(mlkaps_scores)*100:+.2f}%)")
    print(f"   t-test p-value: {p_ttest:.3f}")
    print(f"   Wilcoxon p-value: {p_wilcoxon:.3f}")
    print(f"   Cohen's d: {cohens_d:.3f}")
    print(f"   Significant (p<0.05): {'YES' if p_ttest < 0.05 else 'NO (equivalent performance)'}")
    
    results.append({
        'comparison': 'KAPHE vs MLKAPS',
        'mean_diff': float(np.mean(diff)),
        'p_value_ttest': float(p_ttest),
        'p_value_wilcoxon': float(p_wilcoxon),
        'cohens_d': float(cohens_d),
        'significant': bool(p_ttest < 0.05),
    })
    
    # Save results
    with open(f'{exp_dir}/statistical_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Statistical testing complete!")
    print(f"Results saved to {exp_dir}/statistical_tests.json")
    print("=" * 60)

if __name__ == '__main__':
    main()
