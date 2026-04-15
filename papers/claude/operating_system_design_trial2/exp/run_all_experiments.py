#!/usr/bin/env python3
"""Run all 7 experiments for the Bandwidth Knapsack Scheduler (BKS) paper.

Parameters:
- 2M accesses per trace (plan called for 10M, scaled to 2M for CPU feasibility)
- 5 random seeds: [42, 123, 456, 789, 1024]
- Epoch length: 100K accesses (as planned)
- Fast tier: 30% of pages, 80ns read latency
- Slow tier: 70% of pages, 200ns read latency
- Default bandwidth: 20% of max migration rate
"""

import sys
import os
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fast_simulator import FastTieredMemorySystem, FastMultiTenantSystem
from src.trace_generator import (generate_archetype_trace, generate_zipf_trace,
                                  generate_adversarial_trace, generate_adversarial_varying_R,
                                  generate_multitenant_traces)

# --- Configuration ---
SEEDS = [42, 123, 456, 789, 1024]
TOTAL_ACCESSES = 2_000_000
EPOCH_LENGTH = 100_000
FAST_LATENCY = 80
SLOW_LATENCY = 200
FAST_FRACTION = 0.3

ARCHETYPES = ['mcf-like', 'lbm-like', 'xalancbmk-like', 'omnetpp-like',
              'bwaves-like', 'cactuBSSN-like']
SCHEDULERS = ['GreedyByRank', 'FIFO', 'Random', 'BKSDensity', 'BKSThreshold']
SCHEDULERS_WITH_OPT = SCHEDULERS + ['OfflineOptimal']
BW_FRACTIONS = [0.1, 0.2, 0.3, 0.5, 1.0]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Helper ---
def compute_bandwidth_budget(total_pages, fast_fraction, bw_fraction):
    fast_capacity = int(total_pages * fast_fraction)
    max_migration_bytes = fast_capacity * 4096
    return int(max_migration_bytes * bw_fraction)


def run_single_config(trace, page_sizes_dict, total_pages, ranker_name,
                      scheduler_name, bw_fraction, seed, compute_optimal=False):
    """Run a single simulator configuration."""
    page_sizes_array = np.zeros(total_pages, dtype=np.int32)
    for pid, size in page_sizes_dict.items():
        if pid < total_pages:
            page_sizes_array[pid] = size

    budget = compute_bandwidth_budget(total_pages, FAST_FRACTION, bw_fraction)

    sim = FastTieredMemorySystem(
        total_pages=total_pages,
        fast_fraction=FAST_FRACTION,
        fast_latency_ns=FAST_LATENCY,
        slow_latency_ns=SLOW_LATENCY,
        epoch_length=EPOCH_LENGTH,
        bandwidth_budget_bytes=budget,
        page_sizes_array=page_sizes_array,
        ranker_name=ranker_name,
        scheduler_name=scheduler_name,
        seed=seed,
        compute_optimal=compute_optimal,
    )
    return sim.run(trace)


# ============================
# EXPERIMENT 1: Single-tenant
# ============================
def run_exp1():
    print("=" * 60)
    print("EXPERIMENT 1: Single-tenant performance comparison")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp1_single_tenant')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    run_count = 0

    for arch in ARCHETYPES:
        print(f"  Generating trace: {arch} mixed...")
        trace, page_sizes = generate_archetype_trace(arch, 'mixed',
                                                      total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for bw in BW_FRACTIONS:
            # Include OfflineOptimal only at bw=0.2 to keep runtime manageable
            scheds = SCHEDULERS_WITH_OPT if bw == 0.2 else SCHEDULERS
            for sched in scheds:
                for seed in SEEDS:
                    t0 = time.time()
                    compute_opt = (sched == 'OfflineOptimal')
                    result = run_single_config(trace, page_sizes, total_pages,
                                               'Colloid', sched, bw, seed,
                                               compute_optimal=compute_opt)
                    elapsed = time.time() - t0
                    run_count += 1

                    row = {
                        'trace': arch,
                        'page_size_variant': 'mixed',
                        'scheduler': sched,
                        'ranker': 'Colloid',
                        'bandwidth_pct': bw,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                        'total_migrations': result['total_migrations'],
                        'competitive_ratio': result['competitive_ratio'],
                        'total_benefit': result['total_benefit'],
                        'runtime_s': elapsed,
                    }
                    results.append(row)

                    if run_count % 50 == 0:
                        print(f"    Progress: {run_count}/{run_count} runs done")

    # Also run 4 zipf-stable synthetic traces
    for psm_idx, psm in enumerate([0.0, 0.1, 0.3, 0.5]):
        name = f'zipf-stable-mix{psm}'
        print(f"  Generating trace: {name}...")
        trace, page_sizes = generate_zipf_trace(
            num_pages=10000, working_set_fraction=0.2, zipf_skew=1.0,
            page_size_mix=psm, total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for bw in BW_FRACTIONS:
            scheds = SCHEDULERS_WITH_OPT if bw == 0.2 else SCHEDULERS
            for sched in scheds:
                for seed in SEEDS:
                    compute_opt = (sched == 'OfflineOptimal')
                    result = run_single_config(trace, page_sizes, total_pages,
                                               'Colloid', sched, bw, seed,
                                               compute_optimal=compute_opt)
                    row = {
                        'trace': name,
                        'page_size_variant': f'mix-{psm}',
                        'scheduler': sched,
                        'ranker': 'Colloid',
                        'bandwidth_pct': bw,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                        'total_migrations': result['total_migrations'],
                        'competitive_ratio': result['competitive_ratio'],
                        'total_benefit': result['total_benefit'],
                        'runtime_s': 0,
                    }
                    results.append(row)

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp1 complete: {len(results)} results saved")
    return results


# ============================
# EXPERIMENT 2: Page size heterogeneity
# ============================
def run_exp2():
    print("=" * 60)
    print("EXPERIMENT 2: Page size heterogeneity impact")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp2_heterogeneity')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    variants = ['uniform-4KB', 'uniform-2MB', 'mixed']
    bw = 0.2

    for arch in ARCHETYPES:
        for variant in variants:
            print(f"  {arch} {variant}...")
            trace, page_sizes = generate_archetype_trace(arch, variant,
                                                          total_accesses=TOTAL_ACCESSES, seed=42)
            total_pages = max(page_sizes.keys()) + 1

            for sched in SCHEDULERS:
                for seed in SEEDS:
                    result = run_single_config(trace, page_sizes, total_pages,
                                               'Colloid', sched, bw, seed)
                    results.append({
                        'trace': arch,
                        'page_size_variant': variant,
                        'scheduler': sched,
                        'bandwidth_pct': bw,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                        'total_migrations': result['total_migrations'],
                        'competitive_ratio': result['competitive_ratio'],
                    })

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp2 complete: {len(results)} results saved")
    return results


# ============================
# EXPERIMENT 3: Multi-tenant
# ============================
def run_exp3():
    print("=" * 60)
    print("EXPERIMENT 3: Multi-tenant contention")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp3_multitenant')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    policies = ['equal-share', 'greedy-global', 'BKS-global', 'BKS-fair']
    bw_fraction = 0.2

    for num_tenants in [2, 4, 8]:
        for policy in policies:
            for seed in SEEDS:
                print(f"  {num_tenants}T, {policy}, seed={seed}...")
                tenant_configs = generate_multitenant_traces(
                    num_tenants, total_accesses_per_tenant=2_000_000, seed=seed)

                # Build per-tenant local arrays with proper offset traces.
                # Each tenant gets local page IDs [0, n_tenant_pages) remapped
                # to global IDs [offset, offset+n_tenant_pages) for the
                # concatenated layout that FastMultiTenantSystem expects.
                per_tenant_traces = []
                per_tenant_sizes = []
                offset = 0

                for tc in tenant_configs:
                    ps = tc['page_sizes']  # {global_pid: size}
                    t = tc['trace']        # numpy array of global page IDs

                    # Get this tenant's unique page IDs (sorted for determinism)
                    tenant_pids = sorted(ps.keys())
                    n_pages = len(tenant_pids)

                    # Create mapping: original global pid -> local index
                    pid_to_local = {pid: idx for idx, pid in enumerate(tenant_pids)}

                    # Remap trace to local indices, then offset to global position
                    local_trace = np.array([pid_to_local[p] for p in t], dtype=np.int64)
                    global_trace = local_trace + offset
                    per_tenant_traces.append(global_trace)

                    # Per-tenant sizes array (only this tenant's pages)
                    local_sizes = np.array([ps[pid] for pid in tenant_pids], dtype=np.int32)
                    per_tenant_sizes.append(local_sizes)

                    offset += n_pages

                total_pages = offset
                fast_cap = int(total_pages * 0.3)
                budget = int(fast_cap * 4096 * bw_fraction)

                mt = FastMultiTenantSystem(
                    tenant_traces=per_tenant_traces,
                    tenant_page_sizes=per_tenant_sizes,
                    tenant_page_offsets=None,
                    fast_latency_ns=FAST_LATENCY,
                    slow_latency_ns=SLOW_LATENCY,
                    epoch_length=EPOCH_LENGTH,
                    bandwidth_budget_bytes=budget,
                    policy=policy,
                    seed=seed,
                )
                result = mt.run()

                results.append({
                    'num_tenants': num_tenants,
                    'policy': policy,
                    'seed': seed,
                    'system_latency': result['system_latency'],
                    'per_tenant_latency': result['per_tenant_latency'],
                    'jain_fairness': result['jain_fairness'],
                    'total_migrations': result['total_migrations'],
                    'worst_case_ratio': result['worst_case_ratio'],
                })

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp3 complete: {len(results)} results saved")
    return results


# ============================
# EXPERIMENT 4: Ablation study
# ============================
def run_exp4():
    print("=" * 60)
    print("EXPERIMENT 4: Ablation study of BKS components")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp4_ablation')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    ablation_schedulers = ['BKSDecay', 'BKSNoDensity', 'BKSNoThreshold',
                           'BKSNoDecay', 'BKSDensity']

    for arch in ARCHETYPES:
        trace, page_sizes = generate_archetype_trace(arch, 'mixed',
                                                      total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for bw in [0.05, 0.1, 0.2, 0.5]:
            for sched in ablation_schedulers:
                for seed in SEEDS:
                    result = run_single_config(trace, page_sizes, total_pages,
                                               'Colloid', sched, bw, seed)
                    results.append({
                        'trace': arch,
                        'scheduler': sched,
                        'bandwidth_pct': bw,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                        'total_migrations': result['total_migrations'],
                    })

        print(f"  {arch} ablation complete")

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp4 complete: {len(results)} results saved")
    return results


# ============================
# EXPERIMENT 5: Sensitivity
# ============================
def run_exp5():
    print("=" * 60)
    print("EXPERIMENT 5: Sensitivity analysis")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp5_sensitivity')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    test_archetypes = ['mcf-like', 'lbm-like']

    # Sweep 1: Epoch length
    print("  Sweep: epoch length")
    for arch in test_archetypes:
        trace, page_sizes = generate_archetype_trace(arch, 'mixed',
                                                      total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for epoch_len in [10000, 50000, 100000, 500000, 1000000]:
            for sched in ['GreedyByRank', 'BKSThreshold']:
                for seed in SEEDS:
                    budget = compute_bandwidth_budget(total_pages, FAST_FRACTION, 0.2)
                    page_sizes_array = np.zeros(total_pages, dtype=np.int32)
                    for pid, size in page_sizes.items():
                        if pid < total_pages:
                            page_sizes_array[pid] = size
                    sim = FastTieredMemorySystem(
                        total_pages=total_pages, fast_fraction=FAST_FRACTION,
                        fast_latency_ns=FAST_LATENCY, slow_latency_ns=SLOW_LATENCY,
                        epoch_length=epoch_len, bandwidth_budget_bytes=budget,
                        page_sizes_array=page_sizes_array,
                        ranker_name='Colloid', scheduler_name=sched, seed=seed)
                    result = sim.run(trace)
                    results.append({
                        'sweep': 'epoch_length',
                        'trace': arch,
                        'param_value': epoch_len,
                        'scheduler': sched,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                    })

    # Sweep 2: Latency ratio
    print("  Sweep: latency ratio")
    for arch in test_archetypes:
        trace, page_sizes = generate_archetype_trace(arch, 'mixed',
                                                      total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for slow_lat in [120, 160, 200, 300, 400]:
            for sched in ['GreedyByRank', 'BKSThreshold']:
                for seed in SEEDS:
                    budget = compute_bandwidth_budget(total_pages, FAST_FRACTION, 0.2)
                    page_sizes_array = np.zeros(total_pages, dtype=np.int32)
                    for pid, size in page_sizes.items():
                        if pid < total_pages:
                            page_sizes_array[pid] = size
                    sim = FastTieredMemorySystem(
                        total_pages=total_pages, fast_fraction=FAST_FRACTION,
                        fast_latency_ns=FAST_LATENCY, slow_latency_ns=slow_lat,
                        epoch_length=EPOCH_LENGTH, bandwidth_budget_bytes=budget,
                        page_sizes_array=page_sizes_array,
                        ranker_name='Colloid', scheduler_name=sched, seed=seed)
                    result = sim.run(trace)
                    results.append({
                        'sweep': 'latency_ratio',
                        'trace': arch,
                        'param_value': slow_lat / FAST_LATENCY,
                        'scheduler': sched,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                    })

    # Sweep 3: Fast tier fraction
    print("  Sweep: fast tier fraction")
    for arch in test_archetypes:
        trace, page_sizes = generate_archetype_trace(arch, 'mixed',
                                                      total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for ff in [0.1, 0.2, 0.3, 0.5, 0.7]:
            for sched in ['GreedyByRank', 'BKSThreshold']:
                for seed in SEEDS:
                    budget = compute_bandwidth_budget(total_pages, ff, 0.2)
                    page_sizes_array = np.zeros(total_pages, dtype=np.int32)
                    for pid, size in page_sizes.items():
                        if pid < total_pages:
                            page_sizes_array[pid] = size
                    sim = FastTieredMemorySystem(
                        total_pages=total_pages, fast_fraction=ff,
                        fast_latency_ns=FAST_LATENCY, slow_latency_ns=SLOW_LATENCY,
                        epoch_length=EPOCH_LENGTH, bandwidth_budget_bytes=budget,
                        page_sizes_array=page_sizes_array,
                        ranker_name='Colloid', scheduler_name=sched, seed=seed)
                    result = sim.run(trace)
                    results.append({
                        'sweep': 'fast_fraction',
                        'trace': arch,
                        'param_value': ff,
                        'scheduler': sched,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                    })

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp5 complete: {len(results)} results saved")
    return results


# ============================
# EXPERIMENT 6: Composability
# ============================
def run_exp6():
    print("=" * 60)
    print("EXPERIMENT 6: Composability with different ranking policies")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp6_composability')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    rankers = ['TPP', 'Colloid', 'ALTO']
    comp_schedulers = ['GreedyByRank', 'BKSThreshold']
    bw = 0.2

    for arch in ARCHETYPES:
        trace, page_sizes = generate_archetype_trace(arch, 'mixed',
                                                      total_accesses=TOTAL_ACCESSES, seed=42)
        total_pages = max(page_sizes.keys()) + 1

        for ranker in rankers:
            for sched in comp_schedulers:
                for seed in SEEDS:
                    result = run_single_config(trace, page_sizes, total_pages,
                                               ranker, sched, bw, seed)
                    results.append({
                        'trace': arch,
                        'ranker': ranker,
                        'scheduler': sched,
                        'seed': seed,
                        'total_latency': result['total_latency'],
                        'total_migrations': result['total_migrations'],
                        'total_benefit': result['total_benefit'],
                    })

        print(f"  {arch} composability complete")

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp6 complete: {len(results)} results saved")
    return results


# ============================
# EXPERIMENT 7: Adversarial
# ============================
def run_exp7():
    print("=" * 60)
    print("EXPERIMENT 7: Adversarial analysis and competitive ratio")
    print("=" * 60)
    exp_dir = os.path.join(RESULTS_DIR, 'exp7_adversarial')
    os.makedirs(exp_dir, exist_ok=True)

    results = []
    adv_schedulers = ['GreedyByRank', 'BKSDensity', 'BKSThreshold']
    adv_accesses = 2_000_000

    # Part 1: Vary adversarial fraction
    print("  Part 1: Adversarial fraction sweep")
    for adv_frac in [0.1, 0.5, 1.0]:
        for seed in SEEDS:
            print(f"    adv_frac={adv_frac}, seed={seed}")
            trace, page_sizes = generate_adversarial_trace(
                num_pages=1500, total_accesses=adv_accesses,
                epoch_length=EPOCH_LENGTH, adversarial_fraction=adv_frac,
                page_size_ratio=512, seed=seed)
            total_pages = max(page_sizes.keys()) + 1

            # Set budget = one large page size (key to adversarial scenario)
            budget_bytes = 2097152  # 2MB = one large page

            page_sizes_array = np.zeros(total_pages, dtype=np.int32)
            for pid, size in page_sizes.items():
                if pid < total_pages:
                    page_sizes_array[pid] = size

            for sched in adv_schedulers:
                sim = FastTieredMemorySystem(
                    total_pages=total_pages, fast_fraction=FAST_FRACTION,
                    fast_latency_ns=FAST_LATENCY, slow_latency_ns=SLOW_LATENCY,
                    epoch_length=EPOCH_LENGTH,
                    bandwidth_budget_bytes=budget_bytes,
                    page_sizes_array=page_sizes_array,
                    ranker_name='Colloid', scheduler_name=sched, seed=seed,
                    compute_optimal=True)
                result = sim.run(trace)
                results.append({
                    'scenario': 'adv_fraction',
                    'adversarial_fraction': adv_frac,
                    'R': 512,
                    'scheduler': sched,
                    'seed': seed,
                    'total_latency': result['total_latency'],
                    'competitive_ratio': result['competitive_ratio'],
                    'total_benefit': result['total_benefit'],
                    'total_optimal_benefit': result['total_optimal_benefit'],
                })

    # Part 2: Vary R (page size ratio)
    print("  Part 2: Page size ratio sweep")
    for R in [2, 16, 128, 512]:
        for seed in SEEDS:
            print(f"    R={R}, seed={seed}")
            trace, page_sizes = generate_adversarial_varying_R(
                total_accesses=adv_accesses, epoch_length=EPOCH_LENGTH,
                R=R, seed=seed)
            total_pages = max(page_sizes.keys()) + 1
            budget_bytes = 4096 * R  # budget = one large page

            page_sizes_array = np.zeros(total_pages, dtype=np.int32)
            for pid, size in page_sizes.items():
                if pid < total_pages:
                    page_sizes_array[pid] = size

            for sched in adv_schedulers:
                sim = FastTieredMemorySystem(
                    total_pages=total_pages, fast_fraction=FAST_FRACTION,
                    fast_latency_ns=FAST_LATENCY, slow_latency_ns=SLOW_LATENCY,
                    epoch_length=EPOCH_LENGTH,
                    bandwidth_budget_bytes=budget_bytes,
                    page_sizes_array=page_sizes_array,
                    ranker_name='Colloid', scheduler_name=sched, seed=seed,
                    compute_optimal=True)
                result = sim.run(trace)
                results.append({
                    'scenario': 'varying_R',
                    'adversarial_fraction': 1.0,
                    'R': R,
                    'scheduler': sched,
                    'seed': seed,
                    'total_latency': result['total_latency'],
                    'competitive_ratio': result['competitive_ratio'],
                    'total_benefit': result['total_benefit'],
                    'total_optimal_benefit': result['total_optimal_benefit'],
                })

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Exp7 complete: {len(results)} results saved")
    return results


# ============================
# MAIN
# ============================
if __name__ == '__main__':
    start_time = time.time()

    all_results = {}

    print("\n" + "=" * 60)
    print("BKS EXPERIMENT SUITE")
    print(f"Seeds: {SEEDS}")
    print(f"Total accesses: {TOTAL_ACCESSES}")
    print(f"Epoch length: {EPOCH_LENGTH}")
    print("=" * 60 + "\n")

    all_results['exp1'] = run_exp1()
    all_results['exp2'] = run_exp2()
    all_results['exp3'] = run_exp3()
    all_results['exp4'] = run_exp4()
    all_results['exp5'] = run_exp5()
    all_results['exp6'] = run_exp6()
    all_results['exp7'] = run_exp7()

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time/60:.1f} minutes")
    print(f"{'=' * 60}")
