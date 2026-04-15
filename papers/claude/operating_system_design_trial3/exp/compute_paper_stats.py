#!/usr/bin/env python3
"""Compute all statistics needed for the paper from actual CSV files."""
import csv
import numpy as np
from collections import defaultdict

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def stats(vals):
    a = np.array(vals, dtype=float)
    return f"{np.mean(a):.3f} ± {np.std(a):.3f}"

print("=" * 60)
print("EXP1: Displacement Characterization")
print("=" * 60)
rows = read_csv('exp/exp1_displacement/results.csv')
by_mech = defaultdict(list)
for r in rows:
    by_mech[r['mechanism']].append(r)
for mech in ['io_uring_io_wq', 'io_uring_sqpoll', 'softirq_network', 'workqueue_cmwq', 'mixed']:
    vals = [float(r['relay_cpu_fraction']) for r in by_mech[mech]]
    alphas = [float(r['mean_alpha']) for r in by_mech[mech]]
    print(f"  {mech}: alpha={stats(alphas)}, relay_frac={stats(vals)}, N_seeds={len(vals)}")

print()
print("=" * 60)
print("EXP2: Fairness Violation (Heterogeneous Displacement)")
print("=" * 60)
rows = read_csv('exp/exp2_fairness/results.csv')
by_n = defaultdict(list)
for r in rows:
    if int(r['M']) == 2:
        by_n[int(r['N'])].append(r)
for N in sorted(by_n.keys()):
    j_eff = [float(r['jain_effective']) for r in by_n[N]]
    j_rep = [float(r['jain_reported']) for r in by_n[N]]
    share = [float(r['max_share_ratio']) for r in by_n[N]]
    print(f"  N={N:3d}: J_rep={stats(j_rep)}, J_eff={stats(j_eff)}, max_share_ratio={stats(share)}, N_seeds={len(j_eff)}")

print()
print("=" * 60)
print("EXP3: Cgroup Accounting Leakage")
print("=" * 60)
rows = read_csv('exp/exp3_cgroup/results.csv')
by_k = defaultdict(list)
for r in rows:
    by_k[int(r['K'])].append(float(r['leakage_fraction']))
for K in sorted(by_k.keys()):
    vals = by_k[K]
    print(f"  K={K}: leakage_frac={stats(vals)}, N_rows={len(vals)}")

print()
print("=" * 60)
print("EXP4: CCP Evaluation")
print("=" * 60)
rows = read_csv('exp/exp4_ccp/results.csv')
by_strat = defaultdict(list)
for r in rows:
    key = f"{r['strategy']}_{r['param_value']}"
    by_strat[key].append(r)
for key in sorted(by_strat.keys()):
    rr = by_strat[key]
    j_no = [float(r['jain_no_ccp']) for r in rr]
    j_with = [float(r['jain_with_ccp']) for r in rr]
    oh = [float(r['overhead_pct']) for r in rr]
    print(f"  {key}: J_noCCP={stats(j_no)}, J_withCCP={stats(j_with)}, overhead={stats(oh)}, N={len(rr)}")

print()
print("=" * 60)
print("EXP5: Trace-Driven Validation")
print("=" * 60)
rows = read_csv('exp/exp5_traces/results.csv')
by_trace = defaultdict(list)
for r in rows:
    by_trace[r['trace_scenario']].append(r)
for trace in ['ml_inference', 'webserver', 'database_ycsb', 'mixed_colocation']:
    rr = by_trace[trace]
    j_sim = [float(r['jain_effective']) for r in rr]
    j_ana = [float(r['jain_analytical']) for r in rr]
    j_ccp = [float(r['jain_with_ccp']) for r in rr]
    pe = [float(r['prediction_error']) for r in rr]
    var = [float(r['var_alpha']) for r in rr]
    print(f"  {trace}: J_sim={stats(j_sim)}, J_bound={stats(j_ana)}, pred_error={stats(pe)}, J_ccp={stats(j_ccp)}, var={np.mean(var):.4f}")

print()
print("=" * 60)
print("ABLATION: Variance")
print("=" * 60)
rows = read_csv('exp/ablation_variance/results.csv')
by_level = defaultdict(list)
for r in rows:
    by_level[r['var_level']].append(r)
for level in ['low', 'medium', 'high', 'extreme']:
    rr = by_level[level]
    var = [float(r['var_alpha']) for r in rr]
    j_eff = [float(r['jain_effective']) for r in rr]
    gap = [float(r['fairness_gap']) for r in rr]
    print(f"  {level}: var={stats(var)}, J_eff={stats(j_eff)}, gap={stats(gap)}, N={len(rr)}")

# Compute R^2 for variance vs gap
all_var = [float(r['var_alpha']) for r in read_csv('exp/ablation_variance/results.csv')]
all_gap = [float(r['fairness_gap']) for r in read_csv('exp/ablation_variance/results.csv')]
from numpy.polynomial.polynomial import polyfit
coefs = np.polyfit(all_var, all_gap, 1)
pred = np.polyval(coefs, all_var)
ss_res = np.sum((np.array(all_gap) - pred) ** 2)
ss_tot = np.sum((np.array(all_gap) - np.mean(all_gap)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"  Linear fit: slope={coefs[0]:.3f}, R^2={r2:.4f}")

print()
print("=" * 60)
print("ABLATION: Load (10 seeds)")
print("=" * 60)
rows = read_csv('exp/ablation_load/results.csv')
by_util = defaultdict(list)
for r in rows:
    by_util[r['target_utilization']].append(r)
for util in sorted(by_util.keys()):
    rr = by_util[util]
    j_eff = [float(r['jain_effective']) for r in rr]
    j_ccp = [float(r['jain_with_ccp']) for r in rr]
    oh = [float(r['ccp_overhead_pct']) for r in rr]
    print(f"  util={util}: J_eff={stats(j_eff)}, J_ccp={stats(j_ccp)}, overhead={stats(oh)}, N={len(rr)}")

print()
print("=" * 60)
print("ABLATION: Cores (10 seeds)")
print("=" * 60)
rows = read_csv('exp/ablation_cores/results.csv')
by_m = defaultdict(list)
for r in rows:
    by_m[int(r['M'])].append(r)
for M in sorted(by_m.keys()):
    rr = by_m[M]
    j_eff = [float(r['jain_effective']) for r in rr]
    j_ccp = [float(r['jain_with_ccp']) for r in rr]
    oh = [float(r['ccp_overhead_pct']) for r in rr]
    print(f"  M={M:2d}: J_eff={stats(j_eff)}, J_ccp={stats(j_ccp)}, overhead={stats(oh)}, N={len(rr)}")

print()
print("=" * 60)
print("ABLATION: CCP Components")
print("=" * 60)
rows = read_csv('exp/ablation_ccp_components/results.csv')
by_abl = defaultdict(list)
for r in rows:
    by_abl[r['ablation']].append(r)
for abl in ['no_ccp', 'full_ccp', 'no_propagation', 'no_tagging']:
    rr = by_abl[abl]
    j_eff = [float(r['jain_effective']) for r in rr]
    oh = [float(r['overhead_pct']) for r in rr]
    print(f"  {abl}: J_eff={stats(j_eff)}, overhead={stats(oh)}, N={len(rr)}")

print()
print("=" * 60)
print("BASELINES (10 seeds)")
print("=" * 60)
for bl in ['baseline1', 'baseline2']:
    rows = read_csv(f'exp/{bl}/results.csv')
    by_n = defaultdict(list)
    for r in rows:
        by_n[int(r['N'])].append(r)
    print(f"  {bl}:")
    key = 'jain_effective' if 'jain_effective' in rows[0] else 'jain_fairness'
    for N in sorted(by_n.keys()):
        j = [float(r[key]) for r in by_n[N]]
        print(f"    N={N:3d}: J={stats(j)}, N_seeds={len(j)}")
