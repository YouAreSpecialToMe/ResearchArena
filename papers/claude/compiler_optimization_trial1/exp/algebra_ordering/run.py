"""Experiment 6: Algebra-guided pass ordering.

Uses algebraic properties (idempotency, interference, synergy) to construct
a pass ordering heuristic. The approach uses:
1. Idempotency pruning: remove passes that have no effect
2. Phase-based ordering: canonicalization -> core optimization -> cleanup
3. Synergy chaining: place constructively interfering passes adjacent
4. Anti-interference: keep destructive pairs separated
5. Key pass repetition: repeat high-impact passes (instcombine, simplifycfg)
"""
import sys
import csv
import json
import time
import random
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_interference_matrix():
    matrix = {}
    path = RESULTS_DIR / "interference_matrix.csv"
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        passes_list = header[1:]
        for row in reader:
            pi = row[0]
            for j, pj in enumerate(passes_list):
                matrix[(pi, pj)] = float(row[j+1])
    return matrix, passes_list


def load_idempotency_summary():
    summary = {}
    path = RESULTS_DIR / "idempotency_summary.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary[row['pass_name']] = {
                'structural_rate': float(row['structural_idempotency_rate']),
                'instcount_rate': float(row['instcount_idempotency_rate']),
                'classification': row['classification']
            }
    return summary


def load_single_pass_effects():
    effects = {}
    path = RESULTS_DIR / "interference_raw.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pi = row['pass_i']
            pj = row['pass_j']
            baseline = int(row['baseline_instcount'])
            for p, after_key in [(pi, 'after_i_instcount'), (pj, 'after_j_instcount')]:
                after = int(row[after_key])
                delta_pct = (baseline - after) / baseline * 100 if baseline > 0 else 0
                if p not in effects:
                    effects[p] = []
                effects[p].append(delta_pct)
    return {p: sum(v)/len(v) for p, v in effects.items()}


def load_commutativity_matrix():
    matrix = {}
    path = RESULTS_DIR / "commutativity_matrix.csv"
    if not path.exists():
        return matrix
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        passes_list = header[1:]
        for row in reader:
            pi = row[0]
            for j, pj in enumerate(passes_list):
                matrix[(pi, pj)] = float(row[j+1])
    return matrix


# Pass categories for phase-based ordering
CANONICALIZATION_PASSES = {
    'mem2reg', 'sroa', 'early-cse', 'simplifycfg', 'loop-simplify',
    'lcssa', 'loop-rotate', 'instsimplify',
}

CORE_OPTIMIZATION_PASSES = {
    'instcombine', 'gvn', 'newgvn', 'sccp', 'correlated-propagation',
    'jump-threading', 'licm', 'indvars', 'loop-idiom', 'loop-deletion',
    'loop-reduce', 'reassociate', 'aggressive-instcombine',
    'constraint-elimination', 'nary-reassociate', 'float2int',
    'div-rem-pairs', 'gvn-hoist', 'gvn-sink',
}

CLEANUP_PASSES = {
    'adce', 'bdce', 'dce', 'dse', 'deadargelim', 'globalopt',
    'constmerge', 'memcpyopt', 'tailcallelim', 'sink',
    'mergereturn',
}


def build_algebra_guided_sequence(passes, interf_matrix, idem_summary,
                                   pass_effects, seed, max_len=30):
    """Build a pass ordering using algebraic properties with phase-based approach."""
    rng = random.Random(seed)

    # Step 1: Prune useless passes
    useful_passes = []
    for p in passes:
        avg_effect = pass_effects.get(p, 0)
        idem = idem_summary.get(p, {})
        # Keep passes with any positive effect or that aren't strongly idempotent
        if avg_effect > 0.05 or idem.get('classification', '') != 'strongly_idempotent':
            useful_passes.append(p)
    if not useful_passes:
        useful_passes = passes[:]

    # Step 2: Categorize passes into phases
    canon = [p for p in useful_passes if p in CANONICALIZATION_PASSES]
    core = [p for p in useful_passes if p in CORE_OPTIMIZATION_PASSES]
    cleanup = [p for p in useful_passes if p in CLEANUP_PASSES]
    other = [p for p in useful_passes if p not in CANONICALIZATION_PASSES
             and p not in CORE_OPTIMIZATION_PASSES and p not in CLEANUP_PASSES]

    def synergy_sort(pass_list):
        """Sort passes within a phase by synergy chaining."""
        if len(pass_list) <= 1:
            return pass_list
        # Rank by individual benefit first
        ranked = sorted(pass_list, key=lambda p: -pass_effects.get(p, 0))
        result = [ranked[0]]
        remaining = set(ranked[1:])

        while remaining:
            current = result[-1]
            best_p = None
            best_score = float('-inf')
            for p in remaining:
                synergy = interf_matrix.get((current, p), 0)
                benefit = pass_effects.get(p, 0)
                # Penalty for destructive interference with recent passes
                anti = 0
                for recent in result[-2:]:
                    val = interf_matrix.get((recent, p), 0)
                    if val < -3:
                        anti += abs(val)
                score = synergy * 1.5 + benefit * 0.5 - anti + rng.random() * 0.05
                if score > best_score:
                    best_score = score
                    best_p = p
            result.append(best_p)
            remaining.discard(best_p)
        return result

    # Step 3: Order within each phase using synergy
    canon_ordered = synergy_sort(canon)
    core_ordered = synergy_sort(core)
    cleanup_ordered = synergy_sort(cleanup)

    # Step 4: Build the sequence with repetitions of key passes
    sequence = []

    # Phase 1: Canonicalization
    sequence.extend(canon_ordered)

    # Insert instcombine + simplifycfg after canonicalization (cleanup from canon)
    if 'instcombine' in useful_passes:
        sequence.append('instcombine')
    if 'simplifycfg' in useful_passes:
        sequence.append('simplifycfg')

    # Phase 2: Core optimization
    sequence.extend(core_ordered)

    # Insert cleanup pass after core optimization
    if 'simplifycfg' in useful_passes:
        sequence.append('simplifycfg')
    if 'instcombine' in useful_passes:
        sequence.append('instcombine')

    # Phase 3: Cleanup
    sequence.extend(cleanup_ordered)

    # Phase 4: Final canonicalization
    if 'simplifycfg' in useful_passes:
        sequence.append('simplifycfg')

    # Add other passes
    if other:
        other_sorted = sorted(other, key=lambda p: -pass_effects.get(p, 0))
        # Insert high-effect others after core, low-effect at end
        for p in other_sorted:
            if pass_effects.get(p, 0) > 1.0:
                # Insert before cleanup
                idx = len(canon_ordered) + len(core_ordered) + 4  # after inter-phase passes
                sequence.insert(min(idx, len(sequence)), p)
            else:
                sequence.append(p)

    # Trim to max_len
    if len(sequence) > max_len:
        sequence = sequence[:max_len]

    return sequence


def apply_sequence(benchmark_path, sequence):
    """Apply a sequence of passes to a benchmark."""
    ir_text = get_baseline_ir(benchmark_path)
    for pass_name in sequence:
        ir_new, _, _ = apply_pass_to_ir_text(ir_text, pass_name)
        if ir_new is None:
            continue  # Skip failed passes instead of breaking
        ir_text = ir_new
    return count_instructions(ir_text)


def main():
    benchmarks = get_benchmark_files()
    passes = get_pass_list()
    seeds = [42, 123, 456]

    # Load algebraic properties
    interf_matrix, _ = load_interference_matrix()
    idem_summary = load_idempotency_summary()
    pass_effects = load_single_pass_effects()

    print("=== Algebra-Guided Pass Ordering ===")
    print(f"Benchmarks: {len(benchmarks)}, Passes: {len(passes)}")

    log_file = open(LOG_DIR / "algebra_ordering.log", 'w')
    start_time = time.time()
    results = []

    for seed in seeds:
        sequence = build_algebra_guided_sequence(
            passes, interf_matrix, idem_summary, pass_effects, seed
        )
        print(f"Seed {seed}: sequence length = {len(sequence)}")
        print(f"  Sequence: {sequence}")
        log_file.write(f"Seed {seed}: {sequence}\n")

        for bm in benchmarks:
            ic = apply_sequence(str(bm), sequence)
            baseline_ic = count_instructions(get_baseline_ir(str(bm)))
            ratio = round(ic / baseline_ic, 4) if baseline_ic and baseline_ic > 0 and ic else None
            results.append({
                'benchmark': bm.stem,
                'seed': seed,
                'instcount': ic,
                'baseline_O0': baseline_ic,
                'num_passes': len(sequence),
                'ratio': ratio
            })
            log_file.write(f"  {bm.stem}: {ic}/{baseline_ic} = {ratio}\n")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")
    log_file.write(f"\nRuntime: {elapsed:.1f}s\n")
    log_file.close()

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "algebra_ordering.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['benchmark', 'seed', 'instcount',
                                                'baseline_O0', 'num_passes', 'ratio'])
        writer.writeheader()
        writer.writerows(results)

    # Compare with baselines
    print("\n=== Comparison ===")

    # Load baseline data
    std_data = {}
    with open(RESULTS_DIR / "baseline_opt_levels.csv") as f:
        for row in csv.DictReader(f):
            std_data[row['benchmark']] = row

    # Compute geometric mean ratios
    methods = {}

    # Algebra-guided (mean across seeds)
    alg_ratios = {}
    for r in results:
        bm = r['benchmark']
        if r['ratio'] is not None:
            if bm not in alg_ratios:
                alg_ratios[bm] = []
            alg_ratios[bm].append(r['ratio'])

    valid_alg_ratios = [sum(v)/len(v) for v in alg_ratios.values() if v]
    if valid_alg_ratios:
        alg_geo = math.exp(sum(math.log(max(r, 0.001)) for r in valid_alg_ratios) / len(valid_alg_ratios))
        methods['algebra'] = alg_geo

    # Standard levels
    for level in ['O1', 'O2', 'O3', 'Oz']:
        ratios = []
        for bm, data in std_data.items():
            if data.get(level) and data.get('O0') and float(data['O0']) > 0:
                ratios.append(float(data[level]) / float(data['O0']))
        if ratios:
            methods[level] = math.exp(sum(math.log(r) for r in ratios) / len(ratios))

    # Load greedy and random baselines
    greedy_path = RESULTS_DIR / "baseline_greedy.csv"
    if greedy_path.exists():
        greedy_ratios = []
        with open(greedy_path) as f:
            for row in csv.DictReader(f):
                bm = row['benchmark']
                if bm in std_data and std_data[bm].get('O0') and float(std_data[bm]['O0']) > 0:
                    greedy_ratios.append(float(row['instcount']) / float(std_data[bm]['O0']))
        if greedy_ratios:
            methods['greedy'] = math.exp(sum(math.log(r) for r in greedy_ratios) / len(greedy_ratios))

    random_path = RESULTS_DIR / "baseline_random.csv"
    if random_path.exists():
        rand_by_bm = {}
        with open(random_path) as f:
            for row in csv.DictReader(f):
                bm = row['benchmark']
                if bm not in rand_by_bm:
                    rand_by_bm[bm] = []
                rand_by_bm[bm].append(float(row['instcount']))
        rand_ratios = []
        for bm, ics in rand_by_bm.items():
            if bm in std_data and std_data[bm].get('O0') and float(std_data[bm]['O0']) > 0:
                rand_ratios.append(sum(ics)/len(ics) / float(std_data[bm]['O0']))
        if rand_ratios:
            methods['random'] = math.exp(sum(math.log(r) for r in rand_ratios) / len(rand_ratios))

    for method, geo in sorted(methods.items(), key=lambda x: x[1]):
        print(f"  {method:10s}: {geo:.4f} ({(1-geo)*100:.1f}% reduction)")

    # Per-benchmark comparison vs O2
    wins = ties = losses = 0
    for bm in alg_ratios:
        if bm not in std_data or not std_data[bm].get('O2') or not std_data[bm].get('O0'):
            continue
        alg_mean = sum(alg_ratios[bm]) / len(alg_ratios[bm])
        alg_ic = alg_mean * float(std_data[bm]['O0'])
        o2_ic = float(std_data[bm]['O2'])
        if alg_ic < o2_ic - 0.5:
            wins += 1
        elif alg_ic > o2_ic + 0.5:
            losses += 1
        else:
            ties += 1
    print(f"\n  vs -O2: {wins} wins, {ties} ties, {losses} losses")

    # Determine sequence length from last seed
    last_seq_len = results[-1]['num_passes'] if results else 0

    exp_results = {
        'experiment': 'algebra_ordering',
        'num_passes_in_sequence': last_seq_len,
        'geo_mean_ratio': round(methods.get('algebra', 1.0), 4),
        'reduction_pct': round((1 - methods.get('algebra', 1.0)) * 100, 2),
        'vs_O2_wins': wins, 'vs_O2_ties': ties, 'vs_O2_losses': losses,
        'runtime_seconds': round(elapsed, 1),
        'methods_comparison': {k: round(v, 4) for k, v in methods.items()}
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()
