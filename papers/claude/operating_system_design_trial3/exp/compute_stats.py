#!/usr/bin/env python3
"""Compute summary statistics for the paper."""
import pandas as pd
import numpy as np

OUT = '/home/nw366/ResearchArena/outputs/claude_t3_operating_system_design/idea_01/exp'

print("=" * 60)
print("EXPERIMENT 1: Displacement Characterization")
print("=" * 60)
df = pd.read_csv(f'{OUT}/exp1_displacement/results.csv')
for mech in ['io_uring_io_wq', 'io_uring_sqpoll', 'softirq_network', 'workqueue_cmwq', 'mixed']:
    sub = df[df['mechanism'] == mech]
    print(f"\n{mech}:")
    print(f"  relay_cpu_fraction: {sub['relay_cpu_fraction'].mean():.3f} ± {sub['relay_cpu_fraction'].std():.3f}")
    print(f"  mean_alpha: {sub['mean_alpha'].mean():.3f} ± {sub['mean_alpha'].std():.3f}")
    print(f"  jain_effective: {sub['jain_effective'].mean():.3f} ± {sub['jain_effective'].std():.3f}")

print("\n" + "=" * 60)
print("EXPERIMENT 2: Fairness Violation")
print("=" * 60)
df = pd.read_csv(f'{OUT}/exp2_fairness/results.csv')
for N in [4, 8, 16, 32, 64, 128, 256]:
    sub = df[(df['N'] == N) & (df['M'] == 2)]
    print(f"\nN={N}:")
    print(f"  jain_effective: {sub['jain_effective'].mean():.4f} ± {sub['jain_effective'].std():.4f} (min={sub['jain_effective'].min():.4f})")
    print(f"  max_share_ratio: {sub['max_share_ratio'].mean():.2f} ± {sub['max_share_ratio'].std():.2f}")

print("\n" + "=" * 60)
print("EXPERIMENT 3: Cgroup Accounting")
print("=" * 60)
df = pd.read_csv(f'{OUT}/exp3_cgroup/results.csv')
for K in [2, 4, 8]:
    for policy in ['none', 'partial', 'full']:
        sub = df[(df['K'] == K) & (df['attribution_policy'] == policy)]
        print(f"\nK={K}, policy={policy}:")
        print(f"  leakage_fraction: {sub['leakage_fraction'].mean():.3f} ± {sub['leakage_fraction'].std():.3f}")

print("\n" + "=" * 60)
print("EXPERIMENT 4: CCP Evaluation")
print("=" * 60)
df = pd.read_csv(f'{OUT}/exp4_ccp/results.csv')
for strat in df['strategy'].unique():
    for pv in df[df['strategy'] == strat]['param_value'].unique():
        sub = df[(df['strategy'] == strat) & (df['param_value'] == pv)]
        print(f"\n{strat} ({pv}):")
        print(f"  jain_no_ccp: {sub['jain_no_ccp'].mean():.4f} ± {sub['jain_no_ccp'].std():.4f}")
        print(f"  jain_with_ccp: {sub['jain_with_ccp'].mean():.6f} ± {sub['jain_with_ccp'].std():.6f}")
        print(f"  overhead_pct: {sub['overhead_pct'].mean():.4f} ± {sub['overhead_pct'].std():.4f}")

print("\n" + "=" * 60)
print("EXPERIMENT 5: Trace Validation")
print("=" * 60)
df = pd.read_csv(f'{OUT}/exp5_traces/results.csv')
for scenario in df['trace_scenario'].unique():
    sub = df[df['trace_scenario'] == scenario]
    print(f"\n{scenario}:")
    print(f"  jain_effective: {sub['jain_effective'].mean():.4f} ± {sub['jain_effective'].std():.4f}")
    print(f"  jain_analytical: {sub['jain_analytical'].mean():.4f} ± {sub['jain_analytical'].std():.4f}")
    print(f"  prediction_error: {sub['prediction_error'].mean():.4f} ± {sub['prediction_error'].std():.4f}")
    print(f"  jain_with_ccp: {sub['jain_with_ccp'].mean():.4f} ± {sub['jain_with_ccp'].std():.4f}")

# R² between analytical and simulation
from scipy import stats
all_sim = df['jain_effective'].values
all_anal = df['jain_analytical'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(all_anal, all_sim)
print(f"\nR² (analytical vs simulation): {r_value**2:.4f}")
print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}")

print("\n" + "=" * 60)
print("ABLATION: CCP Components")
print("=" * 60)
df = pd.read_csv(f'{OUT}/ablation_ccp_components/results.csv')
for abl in ['no_ccp', 'full_ccp', 'no_propagation', 'no_tagging']:
    sub = df[df['ablation'] == abl]
    print(f"\n{abl}:")
    print(f"  jain_effective: {sub['jain_effective'].mean():.4f} ± {sub['jain_effective'].std():.4f}")
    print(f"  overhead_pct: {sub['overhead_pct'].mean():.4f} ± {sub['overhead_pct'].std():.4f}")

print("\n" + "=" * 60)
print("ABLATION: Variance Sensitivity")
print("=" * 60)
df = pd.read_csv(f'{OUT}/ablation_variance/results.csv')
for vl in ['low', 'medium', 'high', 'extreme']:
    sub = df[df['var_level'] == vl]
    print(f"\n{vl}:")
    print(f"  var_alpha: {sub['var_alpha'].mean():.4f} ± {sub['var_alpha'].std():.4f}")
    print(f"  fairness_gap: {sub['fairness_gap'].mean():.4f} ± {sub['fairness_gap'].std():.4f}")

# Regression of fairness_gap vs var_alpha
all_var = df['var_alpha'].values
all_gap = df['fairness_gap'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(all_var, all_gap)
print(f"\nR² (var_alpha vs fairness_gap): {r_value**2:.4f}")
print(f"Slope: {slope:.4f}")
