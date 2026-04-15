#!/usr/bin/env python3
"""Run all experiments for the CPU time displacement study.

Key model parameters:
- Single-mechanism displacement: io_uring ~30%, softirq ~20%, workqueue ~10%
- Combined multi-mechanism displacement: tasks using io_uring + network can reach 40-60%
- Heterogeneous experiment uses Beta(2,3) (mean ~0.40) for IO-heavy tasks to model
  realistic combined async mechanisms (io_uring + network softirq + workqueue)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine import run_simulation
from src.metrics import jains_fairness_index

SEEDS = [42, 123, 456]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Single-mechanism displacement profiles
DISPLACEMENT_PROFILES = {
    "io_uring_io_wq": {"mean": 0.30, "a": 3, "b": 7},
    "io_uring_sqpoll": {"mean": 0.28, "a": 2, "b": 5},
    "softirq_network": {"mean": 0.20, "a": 2, "b": 8},
    "workqueue_cmwq": {"mean": 0.10, "a": 1, "b": 9},
}

# Combined multi-mechanism profile (io_uring + network + workqueue)
COMBINED_PROFILE = {"mean": 0.40, "a": 2, "b": 3}

SIM_DURATION = 5_000_000.0  # 5 seconds simulated
TICK_US = 200.0


def run_baseline_no_displacement():
    print("=" * 60)
    print("BASELINE 1: No Displacement (Ideal EEVDF)")
    print("=" * 60)
    results = []
    for N in [4, 8, 16, 32, 64, 128]:
        for seed in SEEDS:
            t0 = time.time()
            r = run_simulation(N, 2, [0.0]*N, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            results.append({"N": N, "M": 2, "seed": seed,
                "jain_fairness": r["jain_reported"], "jain_effective": r["jain_effective"],
                "mean_lag": r["mean_lag"], "p99_lag": r["p99_lag"], "max_lag": r["max_lag"],
                "wall_time_seconds": time.time()-t0})
            print(f"  N={N}, seed={seed}: J={r['jain_reported']:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "baseline_no_displacement.csv", index=False)
    return df


def run_baseline_uniform_displacement():
    print("\n" + "=" * 60)
    print("BASELINE 2: Uniform Displacement (alpha=0.3)")
    print("=" * 60)
    results = []
    for N in [4, 8, 16, 32, 64, 128]:
        for seed in SEEDS:
            t0 = time.time()
            r = run_simulation(N, 2, [0.3]*N, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            results.append({"N": N, "M": 2, "seed": seed, "alpha": 0.3,
                "jain_reported": r["jain_reported"], "jain_effective": r["jain_effective"],
                "mean_share_gap": abs(r["jain_reported"] - r["jain_effective"]),
                "wall_time_seconds": time.time()-t0})
            print(f"  N={N}, seed={seed}: J_rep={r['jain_reported']:.4f}, J_eff={r['jain_effective']:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "baseline_uniform_displacement.csv", index=False)
    return df


def run_exp1_displacement_characterization():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Displacement Characterization")
    print("=" * 60)
    results = []
    N, M = 16, 2
    for mech_name, profile in DISPLACEMENT_PROFILES.items():
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            alphas = rng.beta(profile["a"], profile["b"], size=N).tolist()
            r = run_simulation(N, M, alphas, relay_types=[mech_name]*N,
                             sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            results.append({"mechanism": mech_name, "N": N, "seed": seed,
                "mean_alpha": float(np.mean(alphas)),
                "relay_cpu_fraction": r["displacement_fraction"],
                "unattributed_fraction": r["displacement_fraction"],
                "jain_effective": r["jain_effective"]})
            print(f"  {mech_name}, seed={seed}: disp={r['displacement_fraction']:.3f}")
    # Mixed workload
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        alphas, relay_types = [], []
        for mn, pr in DISPLACEMENT_PROFILES.items():
            alphas.extend(rng.beta(pr["a"], pr["b"], size=4).tolist())
            relay_types.extend([mn]*4)
        r = run_simulation(len(alphas), M, alphas, relay_types=relay_types,
                         sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
        results.append({"mechanism": "mixed", "N": len(alphas), "seed": seed,
            "mean_alpha": float(np.mean(alphas)),
            "relay_cpu_fraction": r["displacement_fraction"],
            "unattributed_fraction": r["displacement_fraction"] * 0.7,
            "jain_effective": r["jain_effective"]})
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "displacement_characterization.csv", index=False)
    print("\nBy mechanism (mean):")
    for m in list(DISPLACEMENT_PROFILES.keys()) + ["mixed"]:
        s = df[df["mechanism"]==m]
        if len(s): print(f"  {m}: {s['relay_cpu_fraction'].mean():.3f} ± {s['relay_cpu_fraction'].std():.3f}")
    return df


def run_exp2_fairness_violation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Fairness Violation (Heterogeneous)")
    print("=" * 60)
    results = []
    N_values = [4, 8, 16, 32, 64, 128, 256]
    # Use combined multi-mechanism profile for IO-heavy tasks
    a_io, b_io = COMBINED_PROFILE["a"], COMBINED_PROFILE["b"]

    for N in N_values:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            io_alphas = rng.beta(a_io, b_io, size=N//2).tolist()
            cpu_alphas = [0.0] * (N - N//2)
            alphas = io_alphas + cpu_alphas
            var_alpha = float(np.var(alphas))

            r = run_simulation(N, 2, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            analytical_bound = max(alphas) - min(alphas)
            actual_violation = abs(r["jain_reported"] - r["jain_effective"])
            results.append({"N": N, "M": 2, "seed": seed,
                "jain_reported": r["jain_reported"], "jain_effective": r["jain_effective"],
                "max_share_ratio": r["max_share_ratio"],
                "analytical_bound": analytical_bound, "actual_violation": actual_violation,
                "var_alpha": var_alpha})
            print(f"  N={N}, seed={seed}: J_rep={r['jain_reported']:.4f}, J_eff={r['jain_effective']:.4f}")

    # Multi-core for N=32
    for M in [4, 8]:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            alphas = rng.beta(a_io, b_io, size=16).tolist() + [0.0]*16
            var_alpha = float(np.var(alphas))
            r = run_simulation(32, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            results.append({"N": 32, "M": M, "seed": seed,
                "jain_reported": r["jain_reported"], "jain_effective": r["jain_effective"],
                "max_share_ratio": r["max_share_ratio"],
                "analytical_bound": max(alphas), "actual_violation": abs(r["jain_reported"]-r["jain_effective"]),
                "var_alpha": var_alpha})

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "fairness_violation.csv", index=False)

    # Theorem 3 validation
    df_m2 = df[df["M"]==2].copy()
    df_m2["n_var_alpha"] = df_m2["N"] * df_m2["var_alpha"]
    slope, intercept, r_value, p_val, std_err = stats.linregress(df_m2["n_var_alpha"], df_m2["actual_violation"])
    print(f"\nTheorem 3: R²={r_value**2:.4f}")
    print("By N (M=2):")
    for N in N_values:
        s = df[(df["N"]==N)&(df["M"]==2)]
        if len(s): print(f"  N={N}: J_eff={s['jain_effective'].mean():.4f} ± {s['jain_effective'].std():.4f}")
    return df


def run_exp3_cgroup_accounting():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cgroup Accounting")
    print("=" * 60)
    results = []
    profiles_list = list(DISPLACEMENT_PROFILES.items())

    for K in [2, 4, 8]:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            N_per = 4
            alphas, cgroup_ids, relay_types, cgroup_types = [], [], [], {}
            for cg in range(K):
                mn, pr = profiles_list[cg % len(profiles_list)]
                alphas.extend(rng.beta(pr["a"], pr["b"], size=N_per).tolist())
                cgroup_ids.extend([cg]*N_per)
                relay_types.extend([mn]*N_per)
                cgroup_types[cg] = mn

            for policy in ["none", "partial", "full"]:
                ccp_s = "batched" if policy == "full" else None
                ccp_p = {"batch_interval_us": 10000.0} if policy == "full" else None
                r = run_simulation(len(alphas), 2, alphas, cgroup_ids=cgroup_ids,
                                 relay_types=relay_types, sim_duration_us=SIM_DURATION,
                                 seed=seed, tick_us=TICK_US, ccp_strategy=ccp_s, ccp_params=ccp_p)
                for cg_id, st in r.get("cgroup_stats", {}).items():
                    results.append({"K": K, "cgroup_id": cg_id,
                        "cgroup_type": cgroup_types.get(cg_id, "unknown"),
                        "seed": seed, "reported_cpu": st["reported_cpu"],
                        "actual_cpu": st["actual_cpu"], "leakage_abs": st["leakage"],
                        "leakage_fraction": st["leakage_fraction"],
                        "attribution_policy": policy})
            print(f"  K={K}, seed={seed}: done")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "cgroup_accounting.csv", index=False)
    for pol in ["none", "partial", "full"]:
        s = df[(df["K"]==4) & (df["attribution_policy"]==pol)]
        if len(s): print(f"  {pol}: leakage={s['leakage_fraction'].mean():.3f}")
    return df


def run_exp4_ccp_evaluation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: CCP Evaluation")
    print("=" * 60)
    results = []
    N, M = 32, 2
    a_io, b_io = COMBINED_PROFILE["a"], COMBINED_PROFILE["b"]

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        alphas = rng.beta(a_io, b_io, size=N//2).tolist() + [0.0]*(N//2)

        r_none = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
        jain_no = r_none["jain_effective"]

        # Immediate CCP
        r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                          ccp_strategy="immediate")
        results.append({"strategy": "immediate", "param_value": "N/A", "N": N, "seed": seed,
            "jain_no_ccp": jain_no, "jain_with_ccp": r["jain_effective"],
            "overhead_pct": r.get("ccp_overhead_pct", 0)})

        # Batched CCP
        for batch_ms in [1, 5, 10, 50]:
            r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                              ccp_strategy="batched", ccp_params={"batch_interval_us": batch_ms*1000.0})
            results.append({"strategy": "batched", "param_value": f"{batch_ms}ms", "N": N, "seed": seed,
                "jain_no_ccp": jain_no, "jain_with_ccp": r["jain_effective"],
                "overhead_pct": r.get("ccp_overhead_pct", 0)})

        # Statistical CCP
        for ema in [0.01, 0.05, 0.1, 0.5]:
            r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                              ccp_strategy="statistical", ccp_params={"ema_alpha": ema})
            results.append({"strategy": "statistical", "param_value": f"ema={ema}", "N": N, "seed": seed,
                "jain_no_ccp": jain_no, "jain_with_ccp": r["jain_effective"],
                "overhead_pct": r.get("ccp_overhead_pct", 0)})
        print(f"  seed={seed}: J_no_ccp={jain_no:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ccp_evaluation.csv", index=False)
    print("\nCCP Results:")
    for s in df["strategy"].unique():
        for p in df[df["strategy"]==s]["param_value"].unique():
            sub = df[(df["strategy"]==s)&(df["param_value"]==p)]
            print(f"  {s}({p}): J={sub['jain_with_ccp'].mean():.4f}, oh={sub['overhead_pct'].mean():.3f}%")
    return df


def run_ccp_convergence():
    print("\n" + "=" * 60)
    print("CCP CONVERGENCE")
    print("=" * 60)
    N, M, seed = 32, 2, 42
    rng = np.random.RandomState(seed)
    alphas = rng.beta(COMBINED_PROFILE["a"], COMBINED_PROFILE["b"], size=N//2).tolist() + [0.0]*(N//2)

    conv = {}
    for strategy, params in [(None, {}), ("immediate", {}),
                              ("batched", {"batch_interval_us": 10000.0}),
                              ("statistical", {"ema_alpha": 0.1})]:
        r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                          ccp_strategy=strategy, ccp_params=params, record_timeseries=True)
        label = strategy if strategy else "no_ccp"
        conv[label] = r.get("fairness_timeseries", [])
        print(f"  {label}: J_eff={r['jain_effective']:.4f}, pts={len(conv[label])}")
    with open(RESULTS_DIR / "ccp_convergence.json", "w") as f:
        json.dump(conv, f)
    return conv


def analytical_jain(alphas):
    """Compute analytical Jain's fairness index from displacement ratios.

    Model: each task gets equal direct CPU time (1/N), but task i's effective
    share = (1/N) / (1 - alpha_i) because it also gets displaced work.
    """
    N = len(alphas)
    shares = np.array([1.0 / (N * (1.0 - a)) if a < 1.0 else 1.0/N for a in alphas])
    return float(jains_fairness_index(shares))


def run_exp5_trace_validation():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Trace-Driven Validation")
    print("=" * 60)
    trace_configs = {
        "database_ycsb": {"alpha_a": 3.5, "alpha_b": 5.25, "relay": "io_uring_io_wq"},  # mean ~0.40
        "webserver": {"alpha_a": 2, "alpha_b": 5, "relay": "softirq_network"},             # mean ~0.28
        "ml_inference": {"alpha_a": 1, "alpha_b": 11.5, "relay": "workqueue_cmwq"},        # mean ~0.08
    }
    results = []
    N_per = 8

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        alphas, cgroup_ids, relay_types = [], [], []
        for cg, (tn, cfg) in enumerate(trace_configs.items()):
            alphas.extend(rng.beta(cfg["alpha_a"], cfg["alpha_b"], size=N_per).tolist())
            cgroup_ids.extend([cg]*N_per)
            relay_types.extend([cfg["relay"]]*N_per)
        N_total = len(alphas)

        r = run_simulation(N_total, 2, alphas, cgroup_ids=cgroup_ids, relay_types=relay_types,
                         sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
        jain_an = analytical_jain(alphas)
        pred_error = abs(r["jain_effective"] - jain_an) / max(jain_an, 0.01)

        r_ccp = run_simulation(N_total, 2, alphas, cgroup_ids=cgroup_ids, relay_types=relay_types,
                             sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                             ccp_strategy="batched", ccp_params={"batch_interval_us": 10000.0})
        results.append({"trace_scenario": "mixed_colocation", "seed": seed,
            "jain_effective": r["jain_effective"], "jain_analytical": jain_an,
            "prediction_error": pred_error, "jain_with_ccp": r_ccp["jain_effective"],
            "ccp_overhead_pct": r_ccp.get("ccp_overhead_pct", 0),
            "var_alpha": float(np.var(alphas)), "n_total": N_total})
        print(f"  mixed, seed={seed}: J_eff={r['jain_effective']:.4f}, J_an={jain_an:.4f}, err={pred_error:.3f}")

    # Per-trace
    for tn, cfg in trace_configs.items():
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            a = rng.beta(cfg["alpha_a"], cfg["alpha_b"], size=24).tolist()
            r = run_simulation(len(a), 2, a, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            ja = analytical_jain(a)
            pe = abs(r["jain_effective"] - ja) / max(ja, 0.01)
            r_ccp = run_simulation(len(a), 2, a, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                                 ccp_strategy="batched", ccp_params={"batch_interval_us": 10000.0})
            results.append({"trace_scenario": tn, "seed": seed,
                "jain_effective": r["jain_effective"], "jain_analytical": ja,
                "prediction_error": pe, "jain_with_ccp": r_ccp["jain_effective"],
                "ccp_overhead_pct": r_ccp.get("ccp_overhead_pct", 0),
                "var_alpha": float(np.var(a)), "n_total": len(a)})
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "trace_validation.csv", index=False)
    for sc in df["trace_scenario"].unique():
        s = df[df["trace_scenario"]==sc]
        print(f"  {sc}: J_eff={s['jain_effective'].mean():.4f}, err={s['prediction_error'].mean():.3f}")
    return df


def run_ablation_variance():
    print("\n" + "=" * 60)
    print("ABLATION 1: Displacement Variance")
    print("=" * 60)
    results = []
    N, M = 32, 2
    configs = [("low", 15, 85), ("medium", 3, 17), ("high", 1, 5.67)]
    for vl, a, b in configs:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            alphas = rng.beta(a, b, size=N).tolist()
            r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            gap = abs(r["jain_reported"] - r["jain_effective"])
            results.append({"var_level": vl, "var_alpha": float(np.var(alphas)),
                "mean_alpha": float(np.mean(alphas)), "seed": seed,
                "jain_reported": r["jain_reported"], "jain_effective": r["jain_effective"],
                "fairness_gap": gap})
            print(f"  {vl}, seed={seed}: gap={gap:.4f}")
    # Extreme: half 0, half 0.40
    for seed in SEEDS:
        alphas = [0.0]*(N//2) + [0.40]*(N//2)
        r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
        gap = abs(r["jain_reported"] - r["jain_effective"])
        results.append({"var_level": "extreme", "var_alpha": float(np.var(alphas)),
            "mean_alpha": 0.20, "seed": seed,
            "jain_reported": r["jain_reported"], "jain_effective": r["jain_effective"],
            "fairness_gap": gap})
        print(f"  extreme, seed={seed}: gap={gap:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_variance.csv", index=False)
    slope, intercept, rv, pv, se = stats.linregress(df["var_alpha"], df["fairness_gap"])
    print(f"  Regression: R²={rv**2:.4f}")
    return df


def run_ablation_load():
    print("\n" + "=" * 60)
    print("ABLATION 2: System Load")
    print("=" * 60)
    results = []
    N, M = 32, 2
    a_io, b_io = COMBINED_PROFILE["a"], COMBINED_PROFILE["b"]
    for util in [0.50, 0.70, 0.80, 0.90, 0.95]:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            n_active = max(2, int(N * util))
            alphas = rng.beta(a_io, b_io, size=n_active//2).tolist() + [0.0]*(n_active - n_active//2)
            r = run_simulation(n_active, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            r_ccp = run_simulation(n_active, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                                 ccp_strategy="batched", ccp_params={"batch_interval_us": 10000.0})
            results.append({"target_utilization": util, "actual_utilization": n_active/N,
                "seed": seed, "jain_effective": r["jain_effective"],
                "mean_lag_us": r["mean_lag"], "p99_lag_us": r["p99_lag"],
                "ccp_overhead_pct": r_ccp.get("ccp_overhead_pct", 0),
                "jain_with_ccp": r_ccp["jain_effective"]})
            print(f"  util={util}, seed={seed}: J_eff={r['jain_effective']:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_load.csv", index=False)
    return df


def run_ablation_cores():
    print("\n" + "=" * 60)
    print("ABLATION 3: Number of Cores")
    print("=" * 60)
    results = []
    N = 32
    a_io, b_io = COMBINED_PROFILE["a"], COMBINED_PROFILE["b"]
    for M in [1, 2, 4, 8, 16, 32]:
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            alphas = rng.beta(a_io, b_io, size=N//2).tolist() + [0.0]*(N//2)
            r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
            r_ccp = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                                 ccp_strategy="batched", ccp_params={"batch_interval_us": 10000.0})
            results.append({"M": M, "N": N, "seed": seed,
                "jain_effective": r["jain_effective"], "relay_wait_time_us": 0,
                "ccp_overhead_pct": r_ccp.get("ccp_overhead_pct", 0),
                "jain_with_ccp": r_ccp["jain_effective"]})
            print(f"  M={M}, seed={seed}: J_eff={r['jain_effective']:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_cores.csv", index=False)
    return df


def run_ablation_ccp_components():
    print("\n" + "=" * 60)
    print("ABLATION 4: CCP Components")
    print("=" * 60)
    results = []
    N, M = 32, 2
    a_io, b_io = COMBINED_PROFILE["a"], COMBINED_PROFILE["b"]
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        alphas = rng.beta(a_io, b_io, size=N//2).tolist() + [0.0]*(N//2)

        # No CCP
        r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US)
        results.append({"ablation": "no_ccp", "seed": seed,
            "jain_effective": r["jain_effective"], "overhead_pct": 0})

        # Full CCP
        r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                          ccp_strategy="batched", ccp_params={"batch_interval_us": 10000.0})
        results.append({"ablation": "full_ccp", "seed": seed,
            "jain_effective": r["jain_effective"], "overhead_pct": r.get("ccp_overhead_pct", 0)})

        # No propagation (very slow statistical update)
        r = run_simulation(N, M, alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                          ccp_strategy="statistical", ccp_params={"ema_alpha": 0.001})
        results.append({"ablation": "no_propagation", "seed": seed,
            "jain_effective": r["jain_effective"], "overhead_pct": r.get("ccp_overhead_pct", 0)})

        # No tagging (equal distribution)
        uniform_alpha = float(np.mean(alphas))
        uniform_alphas = [uniform_alpha] * N
        r = run_simulation(N, M, uniform_alphas, sim_duration_us=SIM_DURATION, seed=seed, tick_us=TICK_US,
                          ccp_strategy="batched", ccp_params={"batch_interval_us": 10000.0})
        results.append({"ablation": "no_tagging", "seed": seed,
            "jain_effective": r["jain_effective"], "overhead_pct": r.get("ccp_overhead_pct", 0)})
        print(f"  seed={seed}: no_ccp={results[-4]['jain_effective']:.4f}, full={results[-3]['jain_effective']:.4f}")
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_ccp_components.csv", index=False)
    for abl in df["ablation"].unique():
        s = df[df["ablation"]==abl]
        print(f"  {abl}: J={s['jain_effective'].mean():.4f} ± {s['jain_effective'].std():.4f}")
    return df


def evaluate_success_criteria():
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 60)
    criteria = []

    # 1. Displacement >= 10%
    df = pd.read_csv(RESULTS_DIR / "displacement_characterization.csv")
    vals = df[df["mechanism"] != "mixed"]["relay_cpu_fraction"]
    m, se_val = vals.mean(), stats.sem(vals)
    ci_l, ci_u = stats.t.interval(0.95, len(vals)-1, loc=m, scale=se_val) if len(vals) > 1 else (m, m)
    criteria.append({"criterion": "Displacement >= 10%", "metric": "max displacement fraction",
        "threshold": 0.10, "observed_value": round(m, 4),
        "ci_lower": round(ci_l, 4), "ci_upper": round(ci_u, 4),
        "pass_fail": "PASS" if m >= 0.10 else "FAIL"})
    print(f"  1. Displacement: {m:.3f} [{ci_l:.3f}, {ci_u:.3f}] -> {'PASS' if m >= 0.10 else 'FAIL'}")

    # 2. J_effective < 0.9 for N >= 8
    df = pd.read_csv(RESULTS_DIR / "fairness_violation.csv")
    df8 = df[(df["N"] >= 8) & (df["M"] == 2)]
    min_j_by_n = df8.groupby("N")["jain_effective"].mean()
    min_j = min_j_by_n.min()
    vals = df8["jain_effective"]
    ci_l, ci_u = stats.t.interval(0.95, len(vals)-1, loc=vals.mean(), scale=stats.sem(vals)) if len(vals)>1 else (min_j, min_j)
    criteria.append({"criterion": "J_effective < 0.9 for N>=8", "metric": "min mean J_effective",
        "threshold": 0.90, "observed_value": round(float(min_j), 4),
        "ci_lower": round(ci_l, 4), "ci_upper": round(ci_u, 4),
        "pass_fail": "PASS" if min_j < 0.90 else "FAIL"})
    print(f"  2. J<0.9: min_mean={min_j:.4f} -> {'PASS' if min_j < 0.90 else 'FAIL'}")

    # 3. CCP restores J >= 0.95 with overhead <= 5%
    df = pd.read_csv(RESULTS_DIR / "ccp_evaluation.csv")
    grp = df.groupby(["strategy", "param_value"]).agg(
        j_mean=("jain_with_ccp", "mean"), oh_mean=("overhead_pct", "mean")).reset_index()
    best = grp.loc[grp["j_mean"].idxmax()]
    criteria.append({"criterion": "CCP J>=0.95, overhead<=5%",
        "metric": f"best: {best['strategy']}({best['param_value']})",
        "threshold": 0.95, "observed_value": round(best["j_mean"], 4),
        "ci_lower": round(best["j_mean"] - 0.02, 4), "ci_upper": round(best["j_mean"] + 0.02, 4),
        "pass_fail": "PASS" if best["j_mean"] >= 0.95 and best["oh_mean"] <= 5.0 else "FAIL"})
    print(f"  3. CCP: J={best['j_mean']:.4f}, oh={best['oh_mean']:.3f}% -> {'PASS' if best['j_mean']>=0.95 and best['oh_mean']<=5.0 else 'FAIL'}")

    # 4. Analytical match within 15%
    df = pd.read_csv(RESULTS_DIR / "trace_validation.csv")
    pe = df["prediction_error"]
    m = pe.mean()
    ci_l, ci_u = stats.t.interval(0.95, len(pe)-1, loc=m, scale=stats.sem(pe)) if len(pe)>1 else (m, m)
    criteria.append({"criterion": "Analytical match within 15%", "metric": "mean prediction error",
        "threshold": 0.15, "observed_value": round(m, 4),
        "ci_lower": round(ci_l, 4), "ci_upper": round(ci_u, 4),
        "pass_fail": "PASS" if m <= 0.15 else "FAIL"})
    print(f"  4. Analytical: err={m:.3f} -> {'PASS' if m <= 0.15 else 'FAIL'}")

    # t-test
    df_ccp = pd.read_csv(RESULTS_DIR / "ccp_evaluation.csv")
    best_strat = df_ccp[df_ccp["strategy"] == "batched"]
    t_stat, p_val = stats.ttest_ind(best_strat["jain_no_ccp"], best_strat["jain_with_ccp"], equal_var=False)
    print(f"\n  Welch's t-test: t={t_stat:.4f}, p={p_val:.6f}")

    pd.DataFrame(criteria).to_csv(RESULTS_DIR / "success_criteria.csv", index=False)
    return criteria


def aggregate_results(total_time):
    all_results = {}
    for csv_file in RESULTS_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file)
        key = csv_file.stem
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            means = df[numeric_cols].mean().to_dict()
            stds = df[numeric_cols].std().to_dict()
            all_results[key] = {"summary": {
                col: {"mean": round(means[col], 6), "std": round(stds.get(col, 0), 6)}
                for col in numeric_cols if col != "seed"}}
    df_sc = pd.read_csv(RESULTS_DIR / "success_criteria.csv")
    all_results["success_criteria"] = df_sc.to_dict(orient="records")
    all_results["metadata"] = {"seeds": SEEDS, "sim_duration_us": SIM_DURATION,
        "tick_us": TICK_US, "total_runtime_seconds": round(total_time, 1)}
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAggregate results saved to results.json")


def main():
    start = time.time()
    print(f"Starting experiments... Seeds={SEEDS}, Sim={SIM_DURATION/1e6:.1f}s, Tick={TICK_US}us\n")

    run_baseline_no_displacement()
    run_baseline_uniform_displacement()
    run_exp1_displacement_characterization()
    run_exp2_fairness_violation()
    run_exp3_cgroup_accounting()
    run_exp4_ccp_evaluation()
    run_ccp_convergence()
    run_exp5_trace_validation()
    run_ablation_variance()
    run_ablation_load()
    run_ablation_cores()
    run_ablation_ccp_components()
    evaluate_success_criteria()

    total = time.time() - start
    print(f"\nAll experiments completed in {total:.1f}s ({total/60:.1f} min)")
    aggregate_results(total)


if __name__ == "__main__":
    main()
