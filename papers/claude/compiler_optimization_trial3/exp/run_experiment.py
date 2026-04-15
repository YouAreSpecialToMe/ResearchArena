#!/usr/bin/env python3
"""
ShapleyPass: Complete experiment pipeline using PolyBench/C benchmarks.
Addresses all review feedback: real benchmarks, proper baselines, ablations, error bars.
"""
import sys, os, json, time, itertools, subprocess, tempfile, shutil
import numpy as np
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
BC_DIR = WORKSPACE / "data" / "polybench_bc"
RESULTS_DIR = WORKSPACE / "results" / "data"
FIGURES_DIR = WORKSPACE / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
BUDGETS = [5, 8, 10, 12, 15]

# 15 diverse PolyBench/C programs across domains
BENCHMARK_NAMES = [
    "2mm", "gemm", "atax", "bicg", "syr2k",        # linear algebra kernels
    "lu", "ludcmp", "gramschmidt",                   # linear algebra solvers
    "fdtd-2d", "jacobi-2d-imper", "seidel-2d", "adi",  # stencils
    "correlation", "floyd-warshall", "dynprog",      # datamining/medley
]

# 20 candidate LLVM passes
CANDIDATE_PASSES = [
    "mem2reg", "instcombine", "simplifycfg", "gvn", "licm",
    "loop-unroll", "loop-rotate", "loop-simplify", "indvars", "sroa",
    "sccp", "dce", "adce", "reassociate", "jump-threading",
    "correlated-propagation", "early-cse", "tailcallelim", "dse", "bdce",
]
N_PASSES = len(CANDIDATE_PASSES)

def count_ir_instructions(bc_path):
    try:
        result = subprocess.run(
            ["llvm-dis", str(bc_path), "-o", "-"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        count = 0
        for line in result.stdout.split('\n'):
            stripped = line.lstrip()
            if (stripped and not stripped.startswith(';') and not stripped.startswith('}')
                and not stripped.startswith('define') and not stripped.startswith('declare')
                and not stripped.startswith('source_filename') and not stripped.startswith('target')
                and not stripped.startswith('@') and not stripped.startswith('!')
                and not stripped.startswith('attributes') and not stripped.startswith('module')
                and not stripped.endswith(':') and line.startswith('  ')):
                count += 1
        return count
    except subprocess.TimeoutExpired:
        return None


def apply_passes_and_count(bc_path, passes):
    if not passes:
        return count_ir_instructions(bc_path)
    fd, tmp = tempfile.mkstemp(suffix=".bc")
    os.close(fd)
    try:
        pipeline = ",".join(passes)
        result = subprocess.run(
            ["opt", f"-passes={pipeline}", str(bc_path), "-o", tmp],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            return count_ir_instructions(tmp)
        # Fallback: one at a time
        shutil.copy2(str(bc_path), tmp)
        fd2, tmp2 = tempfile.mkstemp(suffix=".bc")
        os.close(fd2)
        try:
            for p in passes:
                r = subprocess.run(
                    ["opt", f"-passes={p}", tmp, "-o", tmp2],
                    capture_output=True, text=True, timeout=15
                )
                if r.returncode == 0:
                    tmp, tmp2 = tmp2, tmp
        finally:
            if os.path.exists(tmp2):
                os.remove(tmp2)
        return count_ir_instructions(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class CompilerGame:
    def __init__(self, bc_path, passes=None):
        self.bc_path = str(bc_path)
        self.passes = list(passes) if passes else list(CANDIDATE_PASSES)
        self.n_players = len(self.passes)
        self.cache = {}
        self.baseline_count = count_ir_instructions(self.bc_path)
        if not self.baseline_count:
            raise ValueError(f"Cannot count instructions: {self.bc_path}")

    def value(self, binary_vector):
        key = tuple(int(x) for x in binary_vector)
        if key in self.cache:
            return self.cache[key]
        selected = [self.passes[i] for i, on in enumerate(binary_vector) if on]
        if not selected:
            self.cache[key] = 0.0
            return 0.0
        opt_count = apply_passes_and_count(self.bc_path, selected)
        if opt_count is None:
            self.cache[key] = 0.0
            return 0.0
        reduction = (self.baseline_count - opt_count) / self.baseline_count
        self.cache[key] = reduction
        return reduction

    def __call__(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            return np.array([self.value(x)])
        return np.array([self.value(row) for row in x])


# ── Shapley computation ──

def compute_shapley(game, seed, budget=3000):
    import shapiq
    approx = shapiq.PermutationSamplingSII(
        n=game.n_players, max_order=3, random_state=seed
    )
    iv = approx.approximate(budget=budget, game=game)
    order1, order2, order3 = {}, {}, {}
    for iset in iv.interaction_lookup:
        if len(iset) == 0:
            continue
        val = float(iv[iset])
        if len(iset) == 1:
            order1[game.passes[iset[0]]] = val
        elif len(iset) == 2:
            order2[f"{game.passes[iset[0]]}+{game.passes[iset[1]]}"] = val
        elif len(iset) == 3:
            order3[f"{game.passes[iset[0]]}+{game.passes[iset[1]]}+{game.passes[iset[2]]}"] = val
    return {"order1": order1, "order2": order2, "order3": order3}


# ── Selection algorithms ──

def greedy_individual(phi1, passes, k):
    ranked = sorted(passes, key=lambda p: phi1.get(p, 0), reverse=True)
    return ranked[:k]


def greedy_pairwise(phi1, phi2, passes, k):
    selected, remaining = [], set(passes)
    for _ in range(min(k, len(passes))):
        best, best_score = None, -float('inf')
        for p in remaining:
            score = phi1.get(p, 0)
            for s in selected:
                score += phi2.get(f"{p}+{s}", 0) + phi2.get(f"{s}+{p}", 0)
            if score > best_score:
                best_score, best = score, p
        if best is None: break
        selected.append(best); remaining.remove(best)
    return selected


def greedy_interaction(phi1, phi2, phi3, passes, k):
    selected, remaining = [], set(passes)
    for _ in range(min(k, len(passes))):
        best, best_score = None, -float('inf')
        for p in remaining:
            score = phi1.get(p, 0)
            for s in selected:
                score += phi2.get(f"{p}+{s}", 0) + phi2.get(f"{s}+{p}", 0)
            for s1, s2 in itertools.combinations(selected, 2):
                for perm in itertools.permutations([p, s1, s2]):
                    score += phi3.get(f"{perm[0]}+{perm[1]}+{perm[2]}", 0)
            if score > best_score:
                best_score, best = score, p
        if best is None: break
        selected.append(best); remaining.remove(best)
    return selected


def synergy_seeded(phi1, phi2, phi3, passes, k):
    best_triple, best_score = None, -float('inf')
    for t in itertools.combinations(passes, 3):
        score = sum(phi1.get(p, 0) for p in t)
        for p1, p2 in itertools.combinations(t, 2):
            score += phi2.get(f"{p1}+{p2}", 0) + phi2.get(f"{p2}+{p1}", 0)
        for perm in itertools.permutations(t):
            score += phi3.get(f"{perm[0]}+{perm[1]}+{perm[2]}", 0)
        if score > best_score:
            best_score, best_triple = score, list(t)
    if best_triple is None or k <= 3:
        return greedy_interaction(phi1, phi2, phi3, passes, k)
    selected = list(best_triple[:min(k, 3)])
    remaining = set(passes) - set(selected)
    while len(selected) < k and remaining:
        best, best_sc = None, -float('inf')
        for p in remaining:
            sc = phi1.get(p, 0)
            for s in selected:
                sc += phi2.get(f"{p}+{s}", 0) + phi2.get(f"{s}+{p}", 0)
            for s1, s2 in itertools.combinations(selected, 2):
                for perm in itertools.permutations([p, s1, s2]):
                    sc += phi3.get(f"{perm[0]}+{perm[1]}+{perm[2]}", 0)
            if sc > best_sc:
                best_sc, best = sc, p
        if best is None: break
        selected.append(best); remaining.remove(best)
    return selected


def eval_subset(game, passes_selected):
    bv = np.zeros(game.n_players)
    for p in passes_selected:
        if p in game.passes:
            bv[game.passes.index(p)] = 1
    return game.value(bv)


# ── Baselines ──

def random_search(game, n_samples, k, seed):
    rng = np.random.RandomState(seed)
    n = game.n_players
    best_val, best_bv = -float('inf'), None
    for _ in range(n_samples):
        indices = rng.choice(n, size=k, replace=False)
        bv = np.zeros(n); bv[indices] = 1
        val = game.value(bv)
        if val > best_val:
            best_val, best_bv = val, bv.copy()
    return best_val


def ga_search(game, pop_size, n_gen, k, seed):
    rng = np.random.RandomState(seed)
    n = game.n_players

    def repair(bv):
        bv = bv.copy()
        ones = list(np.where(bv == 1)[0])
        zeros = list(np.where(bv == 0)[0])
        while len(ones) > k and zeros:
            idx = rng.choice(len(ones))
            bv[ones[idx]] = 0
            ones.pop(idx)
            zeros = list(np.where(bv == 0)[0])
        while len(ones) < k and zeros:
            idx = rng.choice(len(zeros))
            bv[zeros[idx]] = 1
            zeros.pop(idx)
            ones = list(np.where(bv == 1)[0])
        return bv

    pop = []
    for _ in range(pop_size):
        indices = rng.choice(n, size=k, replace=False)
        bv = np.zeros(n); bv[indices] = 1
        pop.append(bv)

    best_val, best_bv = -float('inf'), None
    for gen in range(n_gen):
        fitnesses = np.array([game.value(ind) for ind in pop])
        bi = np.argmax(fitnesses)
        if fitnesses[bi] > best_val:
            best_val, best_bv = fitnesses[bi], pop[bi].copy()
        new_pop = [pop[bi].copy()]
        while len(new_pop) < pop_size:
            ti = rng.choice(pop_size, size=3, replace=False)
            p1 = pop[ti[np.argmax(fitnesses[ti])]].copy()
            ti = rng.choice(pop_size, size=3, replace=False)
            p2 = pop[ti[np.argmax(fitnesses[ti])]].copy()
            child = np.where(rng.random(n) < 0.5, p1, p2) if rng.random() < 0.7 else p1.copy()
            for i in range(n):
                if rng.random() < 0.1: child[i] = 1 - child[i]
            new_pop.append(repair(child))
        pop = new_pop[:pop_size]
    return best_val


# ── Variance decomposition ──

def variance_decomposition(inter):
    ss1 = sum(v**2 for v in inter["order1"].values())
    ss2 = sum(v**2 for v in inter["order2"].values())
    ss3 = sum(v**2 for v in inter["order3"].values())
    total = ss1 + ss2 + ss3
    if total == 0:
        return {"order1_frac": 0, "order2_frac": 0, "order3_frac": 0}
    return {"order1_frac": ss1/total, "order2_frac": ss2/total, "order3_frac": ss3/total}


# ── Main pipeline ──

def get_opt_levels(bc_path):
    baseline = count_ir_instructions(bc_path)
    results = {"O0": baseline}
    for level in ["O1", "O2", "O3", "Os", "Oz"]:
        fd, tmp = tempfile.mkstemp(suffix=".bc"); os.close(fd)
        try:
            r = subprocess.run(["opt", f"-{level}", str(bc_path), "-o", tmp],
                              capture_output=True, text=True, timeout=60)
            results[level] = count_ir_instructions(tmp) if r.returncode == 0 else None
        finally:
            if os.path.exists(tmp): os.remove(tmp)
    reductions = {}
    for lvl, cnt in results.items():
        reductions[lvl] = (baseline - cnt) / baseline if cnt and baseline else 0
    return {"counts": results, "reductions": reductions}


def process_benchmark(bname, bc_path):
    """Process one benchmark: all Shapley, selection, baselines, ablations."""
    print(f"\n{'='*60}")
    print(f"Processing: {bname}")
    print(f"{'='*60}")
    t_start = time.time()

    game = CompilerGame(bc_path)
    print(f"  Baseline: {game.baseline_count} instructions")

    # 1. Pass screening
    screening = {}
    for p in CANDIDATE_PASSES:
        c = apply_passes_and_count(bc_path, [p])
        screening[p] = (game.baseline_count - c) / game.baseline_count if c else 0
    active = sum(1 for v in screening.values() if abs(v) > 0.001)
    print(f"  Screening: {active}/{N_PASSES} passes with effect")

    # 2. Optimization levels
    opt_lvl = get_opt_levels(bc_path)
    print(f"  O3 reduction: {opt_lvl['reductions'].get('O3', 0):.4f}")

    # 3. Shapley interactions (3 seeds)
    interactions = {}
    for seed in SEEDS:
        t0 = time.time()
        inter = compute_shapley(game, seed, budget=3000)
        elapsed = time.time() - t0
        interactions[str(seed)] = inter
        print(f"  Shapley seed={seed}: {elapsed:.1f}s, cache={len(game.cache)} entries")

    # 4. Selection algorithms (all seeds, all budgets)
    selection = {}
    for seed_str, inter in interactions.items():
        selection[seed_str] = {}
        phi1, phi2, phi3 = inter["order1"], inter["order2"], inter["order3"]
        for k in BUDGETS:
            methods = {}
            for name, func in [
                ("individual_greedy", lambda: greedy_individual(phi1, CANDIDATE_PASSES, k)),
                ("pairwise_greedy", lambda: greedy_pairwise(phi1, phi2, CANDIDATE_PASSES, k)),
                ("interaction_greedy", lambda: greedy_interaction(phi1, phi2, phi3, CANDIDATE_PASSES, k)),
                ("synergy_seeded", lambda: synergy_seeded(phi1, phi2, phi3, CANDIDATE_PASSES, k)),
            ]:
                sel = func()
                val = eval_subset(game, sel)
                methods[name] = {"value": val, "passes": sel}
            selection[seed_str][str(k)] = methods

    # 5. Baselines (all seeds, all budgets) - using SHARED game cache
    baselines = {}
    for seed in SEEDS:
        baselines[str(seed)] = {}
        for k in BUDGETS:
            rs_val = random_search(game, 1000, k, seed)
            ga_val = ga_search(game, 50, 20, k, seed)
            baselines[str(seed)][str(k)] = {
                "random_search": {"value": rs_val},
                "genetic_algorithm": {"value": ga_val}
            }
        print(f"  Baselines seed={seed}: done, cache={len(game.cache)}")

    # 6. Ablation: interaction order
    ablation_order = {}
    for seed_str, inter in interactions.items():
        ablation_order[seed_str] = {}
        phi1, phi2, phi3 = inter["order1"], inter["order2"], inter["order3"]
        for k in BUDGETS:
            sel1 = greedy_individual(phi1, CANDIDATE_PASSES, k)
            sel12 = greedy_pairwise(phi1, phi2, CANDIDATE_PASSES, k)
            sel123 = greedy_interaction(phi1, phi2, phi3, CANDIDATE_PASSES, k)
            v1 = eval_subset(game, sel1)
            v12 = eval_subset(game, sel12)
            v123 = eval_subset(game, sel123)
            ablation_order[seed_str][str(k)] = {
                "order1_only": v1, "order1_2": v12, "order1_2_3": v123,
                "improvement_2_over_1": v12 - v1, "improvement_3_over_12": v123 - v12,
            }

    elapsed_total = time.time() - t_start
    print(f"  Total: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min), "
          f"cache={len(game.cache)} unique evaluations")

    return {
        "screening": screening,
        "opt_levels": opt_lvl,
        "interactions": interactions,
        "selection": selection,
        "baselines": baselines,
        "ablation_order": ablation_order,
        "baseline_count": game.baseline_count,
        "n_cache_entries": len(game.cache),
        "elapsed_seconds": elapsed_total,
    }


def main():
    print("=" * 70)
    print("ShapleyPass: PolyBench/C Experiment Pipeline")
    print(f"Benchmarks: {len(BENCHMARK_NAMES)}, Passes: {N_PASSES}, Seeds: {SEEDS}")
    print("=" * 70)

    # Find available benchmarks
    benchmark_bcs = {}
    for name in BENCHMARK_NAMES:
        bc = BC_DIR / f"{name}.bc"
        if bc.exists():
            benchmark_bcs[name] = str(bc)
    print(f"Found {len(benchmark_bcs)}/{len(BENCHMARK_NAMES)} benchmarks")

    all_results = {}
    t_total_start = time.time()

    # Process each benchmark
    for bname, bc_path in benchmark_bcs.items():
        all_results[bname] = process_benchmark(bname, bc_path)

    # Save per-benchmark results
    (RESULTS_DIR / "interactions").mkdir(exist_ok=True)
    for bname, res in all_results.items():
        with open(RESULTS_DIR / "interactions" / f"{bname}.json", "w") as f:
            json.dump(res["interactions"], f, indent=2)

    # ── K-ablation (5 benchmarks, K=10,15,20) ──
    print("\n=== K-ABLATION ===")
    ablation_K = {}
    test_bmarks = list(benchmark_bcs.keys())[:5]
    for K_val in [10, 15, 20]:
        passes_sub = CANDIDATE_PASSES[:K_val]
        ablation_K[str(K_val)] = {}
        for bname in test_bmarks:
            ablation_K[str(K_val)][bname] = {}
            game = CompilerGame(benchmark_bcs[bname], passes_sub)
            for seed in SEEDS:
                t0 = time.time()
                inter = compute_shapley(game, seed, budget=3000)
                sel = greedy_interaction(inter["order1"], inter["order2"], inter["order3"],
                                         passes_sub, min(10, K_val))
                val = eval_subset(game, sel)
                vd = variance_decomposition(inter)
                ablation_K[str(K_val)][bname][str(seed)] = {
                    "selection_value": val, "variance_decomp": vd,
                    "elapsed_seconds": time.time() - t0
                }
            print(f"  K={K_val}, {bname}: done")

    # ── Budget ablation (5 benchmarks, budgets 500-5000) ──
    print("\n=== BUDGET ABLATION ===")
    ablation_budget = {}
    for budget in [500, 1000, 2000, 3000, 5000]:
        ablation_budget[str(budget)] = {}
        for bname in test_bmarks:
            ablation_budget[str(budget)][bname] = {}
            game = CompilerGame(benchmark_bcs[bname])
            for seed in SEEDS:
                t0 = time.time()
                inter = compute_shapley(game, seed, budget=budget)
                sel = greedy_interaction(inter["order1"], inter["order2"], inter["order3"],
                                         CANDIDATE_PASSES, 10)
                val = eval_subset(game, sel)
                ablation_budget[str(budget)][bname][str(seed)] = {
                    "selection_value": val, "elapsed_seconds": time.time() - t0
                }
            print(f"  budget={budget}, {bname}: done")

    # ── Transferability ──
    print("\n=== TRANSFERABILITY ===")
    avg_inter = {}
    for bname, res in all_results.items():
        avg1, avg2, avg3 = defaultdict(float), defaultdict(float), defaultdict(float)
        n = len(res["interactions"])
        for seed_str, inter in res["interactions"].items():
            for k, v in inter["order1"].items(): avg1[k] += v/n
            for k, v in inter["order2"].items(): avg2[k] += v/n
            for k, v in inter["order3"].items(): avg3[k] += v/n
        avg_inter[bname] = {"order1": dict(avg1), "order2": dict(avg2), "order3": dict(avg3)}

    all_keys = sorted(set(k for b in avg_inter.values()
                          for d in [b["order1"], b["order2"], b["order3"]]
                          for k in d))

    def to_vec(inter):
        merged = {**inter["order1"], **inter["order2"], **inter["order3"]}
        return np.array([merged.get(k, 0) for k in all_keys])

    transferability = {}
    bmarks = list(avg_inter.keys())
    for target in bmarks:
        phi1, phi2, phi3 = avg_inter[target]["order1"], avg_inter[target]["order2"], avg_inter[target]["order3"]
        oracle_sel = greedy_interaction(phi1, phi2, phi3, CANDIDATE_PASSES, 10)
        game = CompilerGame(benchmark_bcs[target])
        oracle_val = eval_subset(game, oracle_sel)

        others = [b for b in bmarks if b != target]
        t1 = defaultdict(float); t2 = defaultdict(float); t3 = defaultdict(float)
        for o in others:
            for k, v in avg_inter[o]["order1"].items(): t1[k] += v/len(others)
            for k, v in avg_inter[o]["order2"].items(): t2[k] += v/len(others)
            for k, v in avg_inter[o]["order3"].items(): t3[k] += v/len(others)
        transfer_sel = greedy_interaction(dict(t1), dict(t2), dict(t3), CANDIDATE_PASSES, 10)
        transfer_val = eval_subset(game, transfer_sel)

        target_vec = to_vec(avg_inter[target])
        sims = {}
        for o in others:
            ov = to_vec(avg_inter[o])
            dot = np.dot(target_vec, ov)
            nrm = np.linalg.norm(target_vec) * np.linalg.norm(ov)
            sims[o] = float(dot/nrm) if nrm > 0 else 0

        transferability[target] = {
            "oracle_value": oracle_val, "transfer_value": transfer_val,
            "transfer_ratio": transfer_val/oracle_val if oracle_val > 0 else 0,
            "cosine_similarities": sims
        }

    # ── Save all intermediate results ──
    save_all(all_results, ablation_K, ablation_budget, transferability, benchmark_bcs)

    # ── Statistical evaluation ──
    print("\n=== STATISTICAL EVALUATION ===")
    stats = evaluate_criteria(all_results)

    # ── Figures ──
    print("\n=== GENERATING FIGURES ===")
    generate_figures(all_results, ablation_K, ablation_budget, transferability)

    # ── Final results.json ──
    print("\n=== COMPILING RESULTS.JSON ===")
    compile_final(all_results, stats, ablation_K, ablation_budget, transferability)

    elapsed = time.time() - t_total_start
    print(f"\nTOTAL TIME: {elapsed:.0f}s ({elapsed/60:.1f}min, {elapsed/3600:.2f}hours)")


def save_all(all_results, ablation_K, ablation_budget, transferability, benchmark_bcs):
    # Screening
    screening = {b: r["screening"] for b, r in all_results.items()}
    with open(RESULTS_DIR / "pass_screening.json", "w") as f:
        json.dump(screening, f, indent=2)

    # Opt levels
    opt_levels = {b: r["opt_levels"] for b, r in all_results.items()}
    with open(RESULTS_DIR / "opt_levels.json", "w") as f:
        json.dump(opt_levels, f, indent=2)

    # All interactions
    all_inter = {b: r["interactions"] for b, r in all_results.items()}
    with open(RESULTS_DIR / "all_interactions.json", "w") as f:
        json.dump(all_inter, f, indent=2)

    # Variance decomposition
    var_decomp = {}
    for bname, res in all_results.items():
        fracs = {"order1": [], "order2": [], "order3": []}
        per_seed = {}
        for seed_str, inter in res["interactions"].items():
            vd = variance_decomposition(inter)
            per_seed[seed_str] = vd
            for o in ["order1", "order2", "order3"]:
                fracs[o].append(vd[f"{o}_frac"])
        mean = {}
        for o in ["order1", "order2", "order3"]:
            mean[f"{o}_frac_mean"] = float(np.mean(fracs[o]))
            mean[f"{o}_frac_std"] = float(np.std(fracs[o]))
        var_decomp[bname] = {"per_seed": per_seed, "mean": mean}
    with open(RESULTS_DIR / "variance_decomposition.json", "w") as f:
        json.dump(var_decomp, f, indent=2)

    # Selection results
    selection = {b: r["selection"] for b, r in all_results.items()}
    with open(RESULTS_DIR / "selection_results.json", "w") as f:
        json.dump(selection, f, indent=2)

    # Baselines
    baselines = {b: r["baselines"] for b, r in all_results.items()}
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # Ablation order
    abl_order = {b: r["ablation_order"] for b, r in all_results.items()}
    with open(RESULTS_DIR / "ablation_order.json", "w") as f:
        json.dump(abl_order, f, indent=2)

    # Other ablations
    with open(RESULTS_DIR / "ablation_num_passes.json", "w") as f:
        json.dump(ablation_K, f, indent=2)
    with open(RESULTS_DIR / "ablation_budget.json", "w") as f:
        json.dump(ablation_budget, f, indent=2)
    with open(RESULTS_DIR / "transferability.json", "w") as f:
        json.dump(transferability, f, indent=2)

    # Interaction structure
    structure = {}
    for bname, res in all_results.items():
        # Average across seeds
        avg2, avg3 = defaultdict(float), defaultdict(float)
        n = len(res["interactions"])
        for seed_str, inter in res["interactions"].items():
            for k, v in inter["order2"].items(): avg2[k] += v/n
            for k, v in inter["order3"].items(): avg3[k] += v/n
        sorted_pairs = sorted(avg2.items(), key=lambda x: -abs(x[1]))
        sorted_triples = sorted(avg3.items(), key=lambda x: -abs(x[1]))
        structure[bname] = {
            "top_synergistic_pairs": [(k,v) for k,v in sorted_pairs if v > 0][:10],
            "top_redundant_pairs": [(k,v) for k,v in sorted_pairs if v < 0][:10],
            "top_synergistic_triples": [(k,v) for k,v in sorted_triples if v > 0][:10],
            "top_redundant_triples": [(k,v) for k,v in sorted_triples if v < 0][:10],
        }
    with open(RESULTS_DIR / "interaction_structure.json", "w") as f:
        json.dump(structure, f, indent=2)


def evaluate_criteria(all_results):
    stats = {}

    # Criterion 1: significant order-3 interactions >= 30%
    sig_fracs = []
    for bname, res in all_results.items():
        inter_data = res["interactions"]
        if len(inter_data) < 2: continue
        all_triples = set()
        for s, inter in inter_data.items():
            all_triples.update(inter["order3"].keys())
        n_sig, n_total = 0, 0
        for triple in all_triples:
            vals = [inter_data[s]["order3"].get(triple, 0) for s in inter_data]
            n_total += 1
            if abs(np.mean(vals)) > 2 * np.std(vals) and abs(np.mean(vals)) > 1e-5:
                n_sig += 1
        sig_fracs.append(n_sig / n_total if n_total > 0 else 0)

    c1_mean = float(np.mean(sig_fracs)) if sig_fracs else 0
    stats["criterion1_significant_interactions"] = {
        "mean_significant_fraction": c1_mean,
        "per_benchmark": dict(zip(all_results.keys(), sig_fracs)),
        "threshold": 0.30, "met": c1_mean >= 0.30
    }

    # Criterion 2: order-3 variance >= 10%
    o3_fracs = []
    for bname, res in all_results.items():
        fracs = []
        for s, inter in res["interactions"].items():
            vd = variance_decomposition(inter)
            fracs.append(vd["order3_frac"])
        o3_fracs.append(float(np.mean(fracs)))

    c2_mean = float(np.mean(o3_fracs))
    stats["criterion2_variance_explained"] = {
        "mean_order3_frac": c2_mean,
        "per_benchmark": dict(zip(all_results.keys(), o3_fracs)),
        "threshold": 0.10, "met": c2_mean >= 0.10
    }

    # Criterion 3: interaction > pairwise win rate >= 60% (STRICT: ties are NOT wins)
    wins, ties, losses, total = 0, 0, 0, 0
    per_bm = {}
    for bname, res in all_results.items():
        ig_vals, pg_vals = [], []
        for s in res["selection"]:
            ig = res["selection"][s].get("10", {}).get("interaction_greedy", {}).get("value", 0)
            pg = res["selection"][s].get("10", {}).get("pairwise_greedy", {}).get("value", 0)
            ig_vals.append(ig); pg_vals.append(pg)
        if ig_vals:
            total += 1
            diff = np.mean(ig_vals) - np.mean(pg_vals)
            if diff > 0.001: wins += 1; per_bm[bname] = "win"
            elif diff < -0.001: losses += 1; per_bm[bname] = "loss"
            else: ties += 1; per_bm[bname] = "tie"

    c3_wr = wins / total if total > 0 else 0
    stats["criterion3_selection_performance"] = {
        "wins": wins, "ties": ties, "losses": losses, "total": total,
        "win_rate": c3_wr, "per_benchmark": per_bm,
        "threshold": 0.60, "met": c3_wr >= 0.60
    }

    for i, (key, val) in enumerate(stats.items(), 1):
        met = val.get("met", "?")
        metric = list(val.values())[0]
        print(f"  C{i}: {key} = {metric:.3f} => {'MET' if met else 'NOT MET'}")

    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def generate_figures(all_results, ablation_K, ablation_budget, transferability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update({
        'font.size': 12, 'axes.labelsize': 14, 'figure.dpi': 150,
        'savefig.bbox': 'tight', 'savefig.dpi': 300
    })

    benchmarks = sorted(all_results.keys())

    # Figure 1: Variance decomposition
    fig, ax = plt.subplots(figsize=(14, 5))
    o1, o2, o3 = [], [], []
    for b in benchmarks:
        fracs = [variance_decomposition(all_results[b]["interactions"][s])
                 for s in all_results[b]["interactions"]]
        o1.append(np.mean([f["order1_frac"] for f in fracs]))
        o2.append(np.mean([f["order2_frac"] for f in fracs]))
        o3.append(np.mean([f["order3_frac"] for f in fracs]))
    x = np.arange(len(benchmarks))
    ax.bar(x, o1, label='Order 1 (individual)', color='#4477AA')
    ax.bar(x, o2, bottom=o1, label='Order 2 (pairwise)', color='#EE7733')
    ax.bar(x, o3, bottom=np.array(o1)+np.array(o2), label='Order 3 (three-way)', color='#CC3311')
    ax.set_xticks(x); ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Fraction of Interaction Variance')
    ax.set_title('Variance Decomposition by Interaction Order (PolyBench/C)')
    ax.legend(); ax.set_ylim(0, 1.05)
    fig.savefig(FIGURES_DIR / "variance_decomposition.pdf")
    fig.savefig(FIGURES_DIR / "variance_decomposition.png"); plt.close(fig)
    print("  Fig 1: variance_decomposition")

    # Figure 2: Heatmaps
    for rep in benchmarks[:2]:
        first_seed = list(all_results[rep]["interactions"].keys())[0]
        phi2 = all_results[rep]["interactions"][first_seed]["order2"]
        n = N_PASSES; matrix = np.zeros((n, n))
        for key, val in phi2.items():
            parts = key.split('+')
            if len(parts) == 2:
                i = CANDIDATE_PASSES.index(parts[0]) if parts[0] in CANDIDATE_PASSES else -1
                j = CANDIDATE_PASSES.index(parts[1]) if parts[1] in CANDIDATE_PASSES else -1
                if i >= 0 and j >= 0: matrix[i][j] = val; matrix[j][i] = val
        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = max(abs(matrix.min()), abs(matrix.max())) or 0.01
        sns.heatmap(matrix, xticklabels=CANDIDATE_PASSES, yticklabels=CANDIDATE_PASSES,
                    cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax, ax=ax, square=True)
        ax.set_title(f'Pairwise Interactions: {rep}')
        plt.xticks(fontsize=7, rotation=45, ha='right'); plt.yticks(fontsize=7)
        fig.savefig(FIGURES_DIR / f"heatmap_{rep}.pdf")
        fig.savefig(FIGURES_DIR / f"heatmap_{rep}.png"); plt.close(fig)
        print(f"  Fig 2: heatmap_{rep}")

    # Figure 3: Top order-3 interactions
    all_o3 = defaultdict(list)
    for b in benchmarks:
        for s, inter in all_results[b]["interactions"].items():
            for k, v in inter["order3"].items(): all_o3[k].append(v)
    o3_summary = sorted([(k, np.mean(v), np.std(v)) for k, v in all_o3.items()],
                         key=lambda x: -abs(x[1]))[:20]
    if o3_summary:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = [t[0].replace('+', '\n+') for t in o3_summary]
        means = [t[1] for t in o3_summary]
        stds = [t[2] for t in o3_summary]
        colors = ['#CC3311' if m > 0 else '#4477AA' for m in means]
        y = np.arange(len(o3_summary))
        ax.barh(y, means, xerr=stds, color=colors, capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Mean Shapley Interaction Index')
        ax.set_title('Top 20 Order-3 Interactions (across benchmarks)')
        ax.axvline(x=0, color='black', linewidth=0.5); ax.invert_yaxis()
        fig.savefig(FIGURES_DIR / "top_order3_interactions.pdf")
        fig.savefig(FIGURES_DIR / "top_order3_interactions.png"); plt.close(fig)
        print("  Fig 3: top_order3_interactions")

    # Figure 4: Selection comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # 4a: Bar chart at k=10
    ax = axes[0]
    method_names = ["O3", "random_search", "genetic_algorithm",
                    "individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]
    method_labels = ["-O3", "Random\nSearch", "GA", "Indiv.\nGreedy",
                     "Pairwise\nGreedy", "Interact.\nGreedy", "Synergy\nSeeded"]
    method_means, method_stds = [], []
    for method in method_names:
        vals = []
        if method == "O3":
            for b in benchmarks: vals.append(all_results[b]["opt_levels"]["reductions"].get("O3", 0))
        elif method in ["random_search", "genetic_algorithm"]:
            for b in benchmarks:
                for s in all_results[b]["baselines"]:
                    vals.append(all_results[b]["baselines"][s].get("10", {}).get(method, {}).get("value", 0))
        else:
            for b in benchmarks:
                for s in all_results[b]["selection"]:
                    vals.append(all_results[b]["selection"][s].get("10", {}).get(method, {}).get("value", 0))
        method_means.append(np.mean(vals) if vals else 0)
        method_stds.append(np.std(vals) if vals else 0)
    colors = ['#888888', '#BBBBBB', '#AAAAAA', '#66CCEE', '#4477AA', '#CC3311', '#EE7733']
    ax.bar(np.arange(len(method_names)), method_means, yerr=method_stds, color=colors, capsize=4)
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_labels, fontsize=8)
    ax.set_ylabel('Mean IR Reduction'); ax.set_title('Method Comparison (k=10)')

    # 4b: Performance vs budget
    ax = axes[1]
    for method, label, color, ls in [
        ("interaction_greedy", "Interaction Greedy", '#CC3311', '-'),
        ("pairwise_greedy", "Pairwise Greedy", '#4477AA', '--'),
        ("individual_greedy", "Individual Greedy", '#66CCEE', ':'),
        ("random_search", "Random Search", '#BBBBBB', '-.'),
        ("genetic_algorithm", "GA", '#AAAAAA', '-.')
    ]:
        bm, bs = [], []
        for k in BUDGETS:
            vals = []
            if method in ["random_search", "genetic_algorithm"]:
                for b in benchmarks:
                    for s in all_results[b]["baselines"]:
                        vals.append(all_results[b]["baselines"][s].get(str(k), {}).get(method, {}).get("value", 0))
            else:
                for b in benchmarks:
                    for s in all_results[b]["selection"]:
                        vals.append(all_results[b]["selection"][s].get(str(k), {}).get(method, {}).get("value", 0))
            bm.append(np.mean(vals) if vals else 0)
            bs.append(np.std(vals) if vals else 0)
        ax.errorbar(BUDGETS, bm, yerr=bs, label=label, color=color, linestyle=ls, marker='o', capsize=3)
    ax.set_xlabel('Pass Budget (k)'); ax.set_ylabel('Mean IR Reduction')
    ax.set_title('Performance vs. Pass Budget'); ax.legend(fontsize=8); ax.set_xticks(BUDGETS)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "selection_comparison.pdf")
    fig.savefig(FIGURES_DIR / "selection_comparison.png"); plt.close(fig)
    print("  Fig 4: selection_comparison")

    # Figure 5: Network
    try:
        import networkx as nx
        rep = benchmarks[0]
        first_seed = list(all_results[rep]["interactions"].keys())[0]
        phi1 = all_results[rep]["interactions"][first_seed]["order1"]
        phi2 = all_results[rep]["interactions"][first_seed]["order2"]
        G = nx.Graph()
        for p in CANDIDATE_PASSES: G.add_node(p, weight=phi1.get(p, 0))
        for key, val in phi2.items():
            parts = key.split('+')
            if len(parts) == 2 and abs(val) > 0.001:
                G.add_edge(parts[0], parts[1], weight=val)
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, k=2, seed=42)
        nsizes = [300 + 3000*abs(phi1.get(n, 0)) for n in G.nodes()]
        ncolors = ['#CC3311' if phi1.get(n, 0) > 0 else '#4477AA' for n in G.nodes()]
        ewidths = [abs(d['weight'])*50 for _, _, d in G.edges(data=True)]
        ecolors = ['red' if d['weight'] > 0 else 'blue' for _, _, d in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos, node_size=nsizes, node_color=ncolors, alpha=0.8, ax=ax)
        if ewidths:
            nx.draw_networkx_edges(G, pos, width=ewidths, edge_color=ecolors, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
        ax.set_title(f'Pass Interaction Network: {rep}')
        fig.savefig(FIGURES_DIR / f"network_{rep}.pdf")
        fig.savefig(FIGURES_DIR / f"network_{rep}.png"); plt.close(fig)
        print(f"  Fig 5: network_{rep}")
    except Exception as e:
        print(f"  Fig 5 skipped: {e}")

    # Figure 6: Transferability
    if transferability:
        bt = sorted(transferability.keys()); nb = len(bt)
        sim = np.eye(nb)
        for i, b1 in enumerate(bt):
            for j, b2 in enumerate(bt):
                if b2 in transferability[b1].get("cosine_similarities", {}):
                    sim[i][j] = transferability[b1]["cosine_similarities"][b2]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim, xticklabels=bt, yticklabels=bt, cmap='YlOrRd', vmin=-1, vmax=1,
                    ax=ax, square=True, annot=True, fmt='.2f', annot_kws={'fontsize': 5})
        ax.set_title('Interaction Vector Cosine Similarity')
        plt.xticks(fontsize=7, rotation=45, ha='right'); plt.yticks(fontsize=7)
        fig.savefig(FIGURES_DIR / "transferability.pdf")
        fig.savefig(FIGURES_DIR / "transferability.png"); plt.close(fig)
        print("  Fig 6: transferability")

    # Figure 7: Convergence
    if ablation_budget:
        fig, ax = plt.subplots(figsize=(8, 5))
        budgets = sorted(int(b) for b in ablation_budget)
        means, stds = [], []
        for bg in budgets:
            vals = [ablation_budget[str(bg)][b][s]["selection_value"]
                    for b in ablation_budget[str(bg)] for s in ablation_budget[str(bg)][b]]
            means.append(np.mean(vals)); stds.append(np.std(vals))
        ax.errorbar(budgets, means, yerr=stds, marker='o', color='#CC3311', capsize=5)
        ax.set_xlabel('Shapley Estimation Budget'); ax.set_ylabel('Selection Performance')
        ax.set_title('Convergence: Quality vs. Estimation Budget')
        fig.savefig(FIGURES_DIR / "convergence.pdf")
        fig.savefig(FIGURES_DIR / "convergence.png"); plt.close(fig)
        print("  Fig 7: convergence")

    # Figure 8: Ablation order
    fig, ax = plt.subplots(figsize=(14, 5))
    for order_name, label, color, offset in [
        ("order1_only", "Order 1 only", '#66CCEE', -0.25),
        ("order1_2", "Order 1+2", '#4477AA', 0),
        ("order1_2_3", "Order 1+2+3", '#CC3311', 0.25)
    ]:
        means, stds_v = [], []
        for b in benchmarks:
            vals = [all_results[b]["ablation_order"][s].get("10", {}).get(order_name, 0)
                    for s in all_results[b]["ablation_order"]]
            means.append(np.mean(vals)); stds_v.append(np.std(vals))
        ax.bar(np.arange(len(benchmarks))+offset, means, 0.25, yerr=stds_v,
               label=label, color=color, capsize=3)
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('IR Reduction'); ax.set_title('Ablation: Interaction Order (k=10)')
    ax.legend()
    fig.savefig(FIGURES_DIR / "ablation_order.pdf")
    fig.savefig(FIGURES_DIR / "ablation_order.png"); plt.close(fig)
    print("  Fig 8: ablation_order")

    # Figure 9: K-ablation
    if ablation_K:
        fig, ax = plt.subplots(figsize=(8, 5))
        K_vals = sorted(int(k) for k in ablation_K)
        means, stds = [], []
        for K in K_vals:
            vals = [ablation_K[str(K)][b][s]["selection_value"]
                    for b in ablation_K[str(K)] for s in ablation_K[str(K)][b]]
            means.append(np.mean(vals)); stds.append(np.std(vals))
        ax.bar(range(len(K_vals)), means, yerr=stds, color=['#66CCEE', '#4477AA', '#CC3311'], capsize=5)
        ax.set_xticks(range(len(K_vals))); ax.set_xticklabels([f'K={k}' for k in K_vals])
        ax.set_ylabel('Selection Performance'); ax.set_title('Effect of Candidate Pass Count (K)')
        fig.savefig(FIGURES_DIR / "ablation_K.pdf")
        fig.savefig(FIGURES_DIR / "ablation_K.png"); plt.close(fig)
        print("  Fig 9: ablation_K")


def compile_final(all_results, stats, ablation_K, ablation_budget, transferability):
    benchmarks = sorted(all_results.keys())

    # Main comparison at k=10
    main_comp = {}
    for b in benchmarks:
        main_comp[b] = {}
        for lvl in ["O1", "O2", "O3", "Os", "Oz"]:
            v = all_results[b]["opt_levels"]["reductions"].get(lvl, 0)
            main_comp[b][lvl] = {"mean": v, "std": 0}
        for method in ["individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]:
            vals = [all_results[b]["selection"][s].get("10", {}).get(method, {}).get("value", 0)
                    for s in all_results[b]["selection"]]
            main_comp[b][method] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for method in ["random_search", "genetic_algorithm"]:
            vals = [all_results[b]["baselines"][s].get("10", {}).get(method, {}).get("value", 0)
                    for s in all_results[b]["baselines"]]
            main_comp[b][method] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Variance decomposition
    var_summary = {}
    for b in benchmarks:
        fracs = [variance_decomposition(all_results[b]["interactions"][s])
                 for s in all_results[b]["interactions"]]
        var_summary[b] = {}
        for i in [1, 2, 3]:
            var_summary[b][f"order{i}_frac_mean"] = float(np.mean([f[f"order{i}_frac"] for f in fracs]))
            var_summary[b][f"order{i}_frac_std"] = float(np.std([f[f"order{i}_frac"] for f in fracs]))

    # Ablation order summary
    abl_order_sum = {}
    for b in benchmarks:
        data = all_results[b]["ablation_order"]
        abl_order_sum[b] = {}
        for key in ["order1_only", "order1_2", "order1_2_3"]:
            vals = [data[s].get("10", {}).get(key, 0) for s in data]
            abl_order_sum[b][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # K-ablation summary
    k_abl = {}
    for K_str in ablation_K:
        vals = [ablation_K[K_str][b][s]["selection_value"]
                for b in ablation_K[K_str] for s in ablation_K[K_str][b]]
        k_abl[K_str] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Budget ablation summary
    bg_abl = {}
    for bg_str in ablation_budget:
        vals = [ablation_budget[bg_str][b][s]["selection_value"]
                for b in ablation_budget[bg_str] for s in ablation_budget[bg_str][b]]
        bg_abl[bg_str] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Transfer summary
    trans_sum = {}
    if transferability:
        ratios = [transferability[b]["transfer_ratio"] for b in transferability
                  if transferability[b]["transfer_ratio"] > 0]
        trans_sum = {
            "per_benchmark": {b: transferability[b]["transfer_ratio"] for b in transferability},
            "mean_transfer_ratio": float(np.mean(ratios)) if ratios else 0,
            "success_rate_90pct": float(np.mean([1 if r >= 0.9 else 0 for r in ratios])) if ratios else 0
        }

    # Main table CSV
    import csv
    csv_path = WORKSPACE / "results" / "tables" / "main_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    methods = ["O1", "O2", "O3", "Os", "Oz", "random_search", "genetic_algorithm",
               "individual_greedy", "pairwise_greedy", "interaction_greedy", "synergy_seeded"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark"] + methods)
        for b in benchmarks:
            row = [b]
            for m in methods:
                d = main_comp[b].get(m, {"mean": 0, "std": 0})
                if d["std"] > 0:
                    row.append(f"{d['mean']:.4f}+/-{d['std']:.4f}")
                else:
                    row.append(f"{d['mean']:.4f}")
            w.writerow(row)

    results = {
        "title": "ShapleyPass: Quantifying Higher-Order Interactions Among Compiler Optimization Passes",
        "benchmarks": benchmarks,
        "n_benchmarks": len(benchmarks),
        "benchmark_source": "PolyBench/C (PolyBench-ACC repository, HMPP subset)",
        "n_passes": N_PASSES,
        "passes": CANDIDATE_PASSES,
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "main_comparison_k10": main_comp,
        "variance_decomposition": var_summary,
        "success_criteria": stats,
        "ablation_interaction_order": abl_order_sum,
        "ablation_num_passes_K": k_abl,
        "ablation_estimation_budget": bg_abl,
        "transferability": trans_sum,
    }

    with open(WORKSPACE / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  results.json: {len(benchmarks)} benchmarks")
    print(f"  Criteria: C1={stats.get('criterion1_significant_interactions',{}).get('met','?')}, "
          f"C2={stats.get('criterion2_variance_explained',{}).get('met','?')}, "
          f"C3={stats.get('criterion3_selection_performance',{}).get('met','?')}")


if __name__ == "__main__":
    main()
