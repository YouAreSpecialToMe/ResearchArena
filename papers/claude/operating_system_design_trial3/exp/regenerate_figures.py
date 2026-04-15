#!/usr/bin/env python3
"""Regenerate all paper figures from updated CSV data."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

sns.set_context('paper', font_scale=1.2)
sns.set_style('whitegrid')
COLORS = sns.color_palette('colorblind')

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def group_by(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return d


# Figure 1: Displacement Ratios
print("Fig 1: Displacement ratios")
rows = read_csv('exp/exp1_displacement/results.csv')
by_mech = group_by(rows, 'mechanism')
mechs = ['io_uring_io_wq', 'io_uring_sqpoll', 'softirq_network', 'workqueue_cmwq', 'mixed']
labels = ['io_uring\n(io-wq)', 'io_uring\n(SQPOLL)', 'Softirq/\nNetwork', 'Workqueue\n(cmwq)', 'Mixed']
means = [np.mean([float(r['relay_cpu_fraction']) for r in by_mech[m]]) for m in mechs]
stds = [np.std([float(r['relay_cpu_fraction']) for r in by_mech[m]]) for m in mechs]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(range(len(mechs)), means, yerr=stds, capsize=5, color=[COLORS[i] for i in range(5)], edgecolor='black', linewidth=0.5)
ax.axhline(y=0.10, color='red', linestyle='--', linewidth=1.5, label='10% threshold')
ax.set_xticks(range(len(mechs)))
ax.set_xticklabels(labels)
ax.set_ylabel('Relay CPU Fraction')
ax.set_ylim(0, 0.45)
ax.legend()
ax.set_title('Displacement Ratio by Async Kernel Mechanism')
plt.tight_layout()
plt.savefig('figures/fig1_displacement_ratios.pdf', bbox_inches='tight')
plt.savefig('figures/fig1_displacement_ratios.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 2: Fairness Scaling
print("Fig 2: Fairness scaling")
exp2 = read_csv('exp/exp2_fairness/results.csv')
bl1 = read_csv('exp/baseline1/results.csv')
bl2 = read_csv('exp/baseline2/results.csv')

def get_fairness_by_n(rows, key='jain_effective', filter_m=2):
    by_n = defaultdict(list)
    for r in rows:
        if 'M' in r and int(r['M']) != filter_m:
            continue
        by_n[int(r['N'])].append(float(r[key]))
    ns = sorted(by_n.keys())
    means = [np.mean(by_n[n]) for n in ns]
    stds = [np.std(by_n[n]) for n in ns]
    return ns, means, stds

fig, ax = plt.subplots(figsize=(7, 4.5))

ns1, m1, s1 = get_fairness_by_n(bl1, 'jain_effective')
ns2, m2, s2 = get_fairness_by_n(bl2, 'jain_effective')
ns3, m3, s3 = get_fairness_by_n(exp2, 'jain_effective')

ax.errorbar(ns1, m1, yerr=s1, marker='o', label='No displacement (baseline)', color=COLORS[0], capsize=3)
ax.errorbar(ns2, m2, yerr=s2, marker='s', label='Uniform displacement ($\\alpha=0.3$)', color=COLORS[1], capsize=3)
ax.errorbar(ns3, m3, yerr=s3, marker='^', label='Heterogeneous displacement', color=COLORS[2], capsize=3)

ax.set_xscale('log', base=2)
ax.set_xlabel('Number of Processes (N)')
ax.set_ylabel("Jain's Fairness Index")
ax.set_ylim(0.85, 1.02)
ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend(loc='lower left')
ax.set_title('Fairness Index vs. Process Count')
plt.tight_layout()
plt.savefig('figures/fig2_fairness_scaling.pdf', bbox_inches='tight')
plt.savefig('figures/fig2_fairness_scaling.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 3: Scheduler vs Reality (N=32)
print("Fig 3: Scheduler vs reality")
# We need per-task data. Simulate one run to get it.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.engine import run_simulation

rng = np.random.RandomState(42)
N = 32
alphas = list(rng.beta(3, 7, size=N // 2)) + [0.0] * (N // 2)
# Get per-task shares
from src.engine import SimulationEngine
from src.task import Task

engine = SimulationEngine(num_cores=2, sim_duration_us=5_000_000, seed=42, tick_us=200.0)
for i in range(N):
    t = Task(task_id=i, weight=1.0, cgroup_id=0, displacement_ratio=alphas[i])
    t.relay_type = 'io_uring_io_wq'
    engine.add_task(t)
engine.run()

sched_shares = []
eff_shares = []
task_types = []
for t in engine.tasks:
    total_time = engine.current_time
    sched_shares.append(t.direct_cpu_time / total_time)
    eff_shares.append((t.direct_cpu_time + t.displaced_cpu_time) / total_time)
    task_types.append('I/O-intensive' if t.displacement_ratio > 0 else 'CPU-bound')

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(N)
width = 0.35
colors_sched = [COLORS[0] if tt == 'I/O-intensive' else COLORS[1] for tt in task_types]
colors_eff = [COLORS[2] if tt == 'I/O-intensive' else COLORS[3] for tt in task_types]

ax.bar(x - width/2, sched_shares, width, label='Scheduler-reported share', color=COLORS[0], alpha=0.7)
ax.bar(x + width/2, eff_shares, width, label='Effective share (incl. displaced)', color=COLORS[2], alpha=0.7)
ax.axhline(y=1.0/N, color='gray', linestyle='--', linewidth=1, label=f'Fair share (1/{N})')
ax.axvline(x=15.5, color='black', linestyle=':', linewidth=1)
ax.text(7.5, max(eff_shares)*1.05, 'I/O-intensive', ha='center', fontsize=10)
ax.text(23.5, max(eff_shares)*1.05, 'CPU-bound', ha='center', fontsize=10)
ax.set_xlabel('Task ID')
ax.set_ylabel('CPU Share')
ax.legend(loc='upper right', fontsize=9)
ax.set_title('Scheduler View vs. Effective CPU Shares (N=32)')
plt.tight_layout()
plt.savefig('figures/fig3_scheduler_vs_reality.pdf', bbox_inches='tight')
plt.savefig('figures/fig3_scheduler_vs_reality.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 4: Cgroup Leakage
print("Fig 4: Cgroup leakage")
rows = read_csv('exp/exp3_cgroup/results.csv')
# K=4, aggregate by cgroup_type
k4 = [r for r in rows if int(r['K']) == 4]
by_type = group_by(k4, 'cgroup_type')
cg_types = ['io_uring_io_wq', 'io_uring_sqpoll', 'softirq_network', 'workqueue_cmwq']
cg_labels = ['io_uring\n(io-wq)', 'io_uring\n(SQPOLL)', 'Softirq/\nNetwork', 'Workqueue\n(cmwq)']

fig, ax = plt.subplots(figsize=(7, 4.5))
reported_means = []
actual_means = []
for ct in cg_types:
    rr = by_type[ct]
    reported_means.append(np.mean([float(r['reported_cpu']) for r in rr]))
    actual_means.append(np.mean([float(r['actual_cpu']) for r in rr]))

x = np.arange(len(cg_types))
width = 0.35
ax.bar(x - width/2, reported_means, width, label='Reported CPU', color=COLORS[0])
ax.bar(x + width/2, actual_means, width, label='Actual CPU (incl. displaced)', color=COLORS[2])
ax.set_xticks(x)
ax.set_xticklabels(cg_labels)
ax.set_ylabel('CPU Usage')
ax.legend()
ax.set_title('Cgroup Reported vs. Actual CPU Usage (K=4)')
plt.tight_layout()
plt.savefig('figures/fig4_cgroup_leakage.pdf', bbox_inches='tight')
plt.savefig('figures/fig4_cgroup_leakage.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 5: CCP Evaluation
print("Fig 5: CCP evaluation")
rows = read_csv('exp/exp4_ccp/results.csv')
by_strat = group_by(rows, 'strategy')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel a: Batched CCP fairness vs interval
batched = [r for r in rows if r['strategy'] == 'batched']
by_param = group_by(batched, 'param_value')
intervals = ['1ms', '5ms', '10ms', '50ms']
j_means = [np.mean([float(r['jain_with_ccp']) for r in by_param[i]]) for i in intervals]
oh_means = [np.mean([float(r['overhead_pct']) for r in by_param[i]]) for i in intervals]

ax1.plot([1, 5, 10, 50], j_means, 'o-', color=COLORS[0], label='Fairness (J)')
ax1.set_xlabel('Batch Interval (ms)')
ax1.set_ylabel("Jain's Fairness Index", color=COLORS[0])
ax1.set_ylim(0.99, 1.001)
ax1b = ax1.twinx()
ax1b.plot([1, 5, 10, 50], oh_means, 's--', color=COLORS[2], label='Overhead (%)')
ax1b.set_ylabel('Overhead (%)', color=COLORS[2])
ax1.set_title('Batched CCP: Fairness vs. Interval')

# Panel b: All strategies
strategies = ['immediate', 'batched', 'statistical']
strat_labels = ['Immediate', 'Batched\n(10ms)', 'Statistical\n(EMA 0.1)']
strat_rows = [
    [r for r in rows if r['strategy'] == 'immediate'],
    [r for r in rows if r['strategy'] == 'batched' and r['param_value'] == '10ms'],
    [r for r in rows if r['strategy'] == 'statistical' and r['param_value'] == 'ema=0.1'],
]
j_vals = [np.mean([float(r['jain_with_ccp']) for r in rr]) for rr in strat_rows]
oh_vals = [np.mean([float(r['overhead_pct']) for r in rr]) for rr in strat_rows]

ax2.scatter(oh_vals, j_vals, s=100, color=[COLORS[0], COLORS[1], COLORS[2]], zorder=5)
for i, lbl in enumerate(['Immediate', 'Batched 10ms', 'Statistical']):
    ax2.annotate(lbl, (oh_vals[i], j_vals[i]), textcoords="offset points", xytext=(10, -5), fontsize=9)
ax2.set_xlabel('Overhead (%)')
ax2.set_ylabel("Jain's Fairness Index")
ax2.set_title('CCP Strategy Comparison')
ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target J=0.95')
ax2.legend()

plt.tight_layout()
plt.savefig('figures/fig5_ccp_evaluation.pdf', bbox_inches='tight')
plt.savefig('figures/fig5_ccp_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 6: CCP Convergence (bursty workload) - regenerate from simulation
print("Fig 6: CCP convergence")
N = 32
rng = np.random.RandomState(42)
alphas_base = list(rng.beta(3, 7, size=N // 2)) + [0.0] * (N // 2)
# Run with different CCP strategies and record timeseries
fig, ax = plt.subplots(figsize=(7, 4.5))
for strat, params, label, color in [
    (None, {}, 'No CCP', COLORS[3]),
    ('immediate', {}, 'Immediate CCP', COLORS[0]),
    ('batched', {'batch_interval_us': 10000}, 'Batched (10ms)', COLORS[1]),
    ('statistical', {'ema_alpha': 0.1}, 'Statistical (EMA)', COLORS[2]),
]:
    r = run_simulation(N, 2, alphas_base, seed=42, sim_duration_us=5_000_000, tick_us=200.0,
                       ccp_strategy=strat, ccp_params=params, record_timeseries=True)
    ts = r.get('fairness_timeseries', [])
    if ts:
        times = [t[0] for t in ts]
        fairness = [t[1] for t in ts]
        ax.plot(times, fairness, label=label, color=color, alpha=0.8)

ax.set_xlabel('Time (ms)')
ax.set_ylabel("Jain's Fairness Index")
ax.set_title('CCP Convergence Over Time (N=32)')
ax.legend()
ax.set_ylim(0.85, 1.02)
plt.tight_layout()
plt.savefig('figures/fig6_ccp_convergence.pdf', bbox_inches='tight')
plt.savefig('figures/fig6_ccp_convergence.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 7: Sensitivity Analysis (2x2)
print("Fig 7: Sensitivity analysis")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) Variance vs fairness gap
var_rows = read_csv('exp/ablation_variance/results.csv')
all_var = [float(r['var_alpha']) for r in var_rows]
all_gap = [float(r['fairness_gap']) for r in var_rows]
by_level = group_by(var_rows, 'var_level')
level_colors = {'low': COLORS[0], 'medium': COLORS[1], 'high': COLORS[2], 'extreme': COLORS[3]}
for level in ['low', 'medium', 'high', 'extreme']:
    rr = by_level[level]
    v = [float(r['var_alpha']) for r in rr]
    g = [float(r['fairness_gap']) for r in rr]
    axes[0,0].scatter(v, g, label=level.capitalize(), color=level_colors[level], s=40)
# Linear fit
coefs = np.polyfit(all_var, all_gap, 1)
xx = np.linspace(0, max(all_var)*1.1, 100)
axes[0,0].plot(xx, np.polyval(coefs, xx), 'k--', alpha=0.5)
ss_res = np.sum((np.array(all_gap) - np.polyval(coefs, all_var))**2)
ss_tot = np.sum((np.array(all_gap) - np.mean(all_gap))**2)
r2 = 1 - ss_res/ss_tot
axes[0,0].set_xlabel('Var($\\alpha$)')
axes[0,0].set_ylabel('Fairness Gap')
axes[0,0].set_title(f'(a) Fairness Gap vs. Var($\\alpha$) ($R^2={r2:.2f}$)')
axes[0,0].legend(fontsize=9)

# (b) Load sensitivity - note: all identical due to simulator limitation
load_rows = read_csv('exp/ablation_load/results.csv')
by_util = group_by(load_rows, 'target_utilization')
utils = sorted(by_util.keys())
j_means = [np.mean([float(r['jain_effective']) for r in by_util[u]]) for u in utils]
j_stds = [np.std([float(r['jain_effective']) for r in by_util[u]]) for u in utils]
j_ccp = [np.mean([float(r['jain_with_ccp']) for r in by_util[u]]) for u in utils]
axes[0,1].errorbar([float(u) for u in utils], j_means, yerr=j_stds, marker='o', label='No CCP', color=COLORS[0], capsize=3)
axes[0,1].errorbar([float(u) for u in utils], j_ccp, marker='s', label='With CCP', color=COLORS[1], capsize=3)
axes[0,1].set_xlabel('Target Utilization')
axes[0,1].set_ylabel("Jain's Fairness Index")
axes[0,1].set_title('(b) Fairness vs. System Load')
axes[0,1].legend()
axes[0,1].set_ylim(0.85, 1.02)

# (c) Core count
core_rows = read_csv('exp/ablation_cores/results.csv')
by_m = group_by(core_rows, 'M')
ms = sorted(by_m.keys(), key=int)
j_eff = [np.mean([float(r['jain_effective']) for r in by_m[m]]) for m in ms]
j_eff_std = [np.std([float(r['jain_effective']) for r in by_m[m]]) for m in ms]
j_ccp = [np.mean([float(r['jain_with_ccp']) for r in by_m[m]]) for m in ms]
j_ccp_std = [np.std([float(r['jain_with_ccp']) for r in by_m[m]]) for m in ms]
axes[1,0].errorbar([int(m) for m in ms], j_eff, yerr=j_eff_std, marker='o', label='No CCP', color=COLORS[0], capsize=3)
axes[1,0].errorbar([int(m) for m in ms], j_ccp, yerr=j_ccp_std, marker='s', label='With CCP', color=COLORS[1], capsize=3)
axes[1,0].set_xlabel('Number of Cores (M)')
axes[1,0].set_ylabel("Jain's Fairness Index")
axes[1,0].set_title('(c) Fairness vs. Core Count (N=32)')
axes[1,0].legend()
axes[1,0].set_xscale('log', base=2)
axes[1,0].set_ylim(0.85, 1.02)

# (d) CCP component ablation
abl_rows = read_csv('exp/ablation_ccp_components/results.csv')
by_abl = group_by(abl_rows, 'ablation')
abl_names = ['no_ccp', 'full_ccp', 'no_propagation', 'no_tagging']
abl_labels = ['No CCP', 'Full CCP', 'No Propagation', 'No Tagging\n(Equal Dist.)']
j_means = [np.mean([float(r['jain_effective']) for r in by_abl[a]]) for a in abl_names]
j_stds = [np.std([float(r['jain_effective']) for r in by_abl[a]]) for a in abl_names]
bars = axes[1,1].bar(range(4), j_means, yerr=j_stds, capsize=5, color=[COLORS[i] for i in range(4)], edgecolor='black', linewidth=0.5)
axes[1,1].set_xticks(range(4))
axes[1,1].set_xticklabels(abl_labels, fontsize=9)
axes[1,1].set_ylabel("Jain's Fairness Index")
axes[1,1].set_title('(d) CCP Component Ablation')
axes[1,1].set_ylim(0.85, 1.02)

plt.tight_layout()
plt.savefig('figures/fig7_sensitivity.pdf', bbox_inches='tight')
plt.savefig('figures/fig7_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()


# Figure 8: Analytical Validation
print("Fig 8: Analytical validation")
trace_rows = read_csv('exp/exp5_traces/results.csv')
by_trace = group_by(trace_rows, 'trace_scenario')

fig, ax = plt.subplots(figsize=(7, 5))
trace_colors = {'ml_inference': COLORS[0], 'webserver': COLORS[1], 'database_ycsb': COLORS[2], 'mixed_colocation': COLORS[3]}
trace_labels = {'ml_inference': 'ML Inference', 'webserver': 'Web Server', 'database_ycsb': 'Database (YCSB)', 'mixed_colocation': 'Mixed Co-location'}

for trace, rr in by_trace.items():
    j_sim = [float(r['jain_effective']) for r in rr]
    j_bound = [float(r['jain_analytical']) for r in rr]
    ax.scatter(j_sim, j_bound, color=trace_colors.get(trace, 'gray'), label=trace_labels.get(trace, trace), s=50, alpha=0.7)

ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', alpha=0.5, label='y = x (perfect prediction)')
ax.set_xlabel('Simulation $J_{\\mathrm{effective}}$')
ax.set_ylabel('Analytical Bound $J_{\\mathrm{bound}}$')
ax.set_title('Analytical Bound vs. Simulation')
ax.legend(fontsize=9)
ax.set_xlim(0.85, 1.01)
ax.set_ylim(0.45, 1.01)

# Compute R^2
all_sim = [float(r['jain_effective']) for r in trace_rows]
all_bound = [float(r['jain_analytical']) for r in trace_rows]
coefs = np.polyfit(all_sim, all_bound, 1)
pred = np.polyval(coefs, all_sim)
ss_res = np.sum((np.array(all_bound) - pred)**2)
ss_tot = np.sum((np.array(all_bound) - np.mean(all_bound))**2)
r2 = 1 - ss_res/ss_tot
ax.text(0.86, 0.48, f'$R^2 = {r2:.2f}$', fontsize=12)

plt.tight_layout()
plt.savefig('figures/fig8_analytical_validation.pdf', bbox_inches='tight')
plt.savefig('figures/fig8_analytical_validation.png', dpi=150, bbox_inches='tight')
plt.close()

print("All figures regenerated!")
