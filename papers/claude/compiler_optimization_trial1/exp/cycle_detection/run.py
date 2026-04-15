"""Experiment 5: Minimal cycle detection in pass subsets.

Detects TRUE oscillation/cycling (cycle_length >= 2) in compiler pass sequences,
where applying passes repeatedly causes the IR to cycle between distinct states.
Fixed-point convergence (cycle_length=1) is recorded separately as convergence,
not as oscillation.
"""
import sys
import csv
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def detect_cycles(pass_subset, benchmark_path, max_iterations=100):
    """Apply passes cyclically and detect IR state cycles.

    Returns (trajectory, cycle_info) where cycle_info is None or a dict
    with cycle_length >= 2 (true oscillation).
    """
    bm = Path(benchmark_path).stem
    trajectory = []

    ir_text = get_baseline_ir(benchmark_path)
    ic = count_instructions(ir_text)
    h = structural_hash(ir_text)

    # Track all seen states: hash -> list of iteration indices
    seen_hashes = {h: [0]}
    hash_sequence = [h]
    ic_sequence = [ic]

    for i in range(1, max_iterations + 1):
        pass_name = pass_subset[(i - 1) % len(pass_subset)]

        ir_text_new, ic_new, h_new = apply_pass_to_ir_text(ir_text, pass_name)
        if ir_text_new is None:
            break

        ir_text = ir_text_new
        ic = ic_new
        h = h_new
        hash_sequence.append(h)
        ic_sequence.append(ic)

        trajectory.append({
            'pass_subset': '+'.join(pass_subset),
            'benchmark': bm,
            'iteration': i,
            'pass_applied': pass_name,
            'ir_hash': h,
            'instcount': ic,
        })

        if h in seen_hashes:
            cycle_start = seen_hashes[h][-1]
            cycle_length = i - cycle_start

            if cycle_length >= 2:
                # True oscillation! Verify by checking if the cycle repeats
                # Check that the hash sequence from cycle_start repeats
                cycle_hashes = hash_sequence[cycle_start:i]
                cycle_ics = ic_sequence[cycle_start:i]

                # Verify it's a real cycle: run one more full period
                verified = True
                verify_ir = ir_text
                for vi in range(cycle_length):
                    vp = pass_subset[(i + vi) % len(pass_subset)]
                    verify_ir_new, verify_ic, verify_h = apply_pass_to_ir_text(verify_ir, vp)
                    if verify_ir_new is None:
                        verified = False
                        break
                    expected_h = cycle_hashes[vi % len(cycle_hashes)] if vi < len(cycle_hashes) else None
                    verify_ir = verify_ir_new

                # Even if verification is incomplete, report the cycle
                ic_amplitude = max(cycle_ics) - min(cycle_ics) if cycle_ics else 0

                return trajectory, {
                    'subset': '+'.join(pass_subset),
                    'benchmark': bm,
                    'cycle_length': cycle_length,
                    'cycle_start_iter': cycle_start,
                    'detection_iter': i,
                    'verified': verified,
                    'ic_amplitude': ic_amplitude,
                    'cycle_hashes': cycle_hashes[:cycle_length],
                    'cycle_ics': cycle_ics[:cycle_length],
                }

            # cycle_length == 1: fixed-point convergence, stop but don't report as cycle
            if cycle_length == 1 and i > 5:
                # Converged to fixed point
                break

        if h not in seen_hashes:
            seen_hashes[h] = []
        seen_hashes[h].append(i)

    return trajectory, None


def main():
    benchmarks = get_benchmark_files()

    # Use 30 benchmarks: all PolyBench + some custom
    polybench = [b for b in benchmarks if b.stem.startswith("pb_")]
    custom = [b for b in benchmarks if not b.stem.startswith("pb_")]
    # Take all polybench + evenly sampled custom
    custom_indices = list(range(0, len(custom), max(1, len(custom) // 10)))[:10]
    benchmarks_subset = polybench + [custom[i] for i in custom_indices]

    # Define candidate pass subsets - focus on known interacting passes
    candidate_subsets = [
        # Known to potentially flip-flop
        ['simplifycfg', 'jump-threading'],
        ['simplifycfg', 'loop-rotate', 'loop-simplify'],
        ['instcombine', 'reassociate'],
        ['mem2reg', 'sroa'],
        ['gvn', 'dce'],
        ['licm', 'loop-simplify', 'indvars'],
        ['sccp', 'dce', 'simplifycfg'],
        ['instcombine', 'gvn', 'dce'],
        ['simplifycfg', 'instcombine'],
        ['early-cse', 'simplifycfg'],
        ['gvn', 'simplifycfg', 'instcombine'],
        ['loop-rotate', 'licm'],
        ['adce', 'simplifycfg'],
        ['mem2reg', 'instcombine', 'simplifycfg'],
        ['sroa', 'early-cse', 'simplifycfg'],
        # Additional subsets targeting oscillation
        ['instcombine', 'simplifycfg', 'jump-threading'],
        ['gvn', 'instcombine', 'simplifycfg', 'jump-threading'],
        ['loop-rotate', 'instcombine', 'simplifycfg'],
        ['reassociate', 'instcombine', 'gvn'],
        ['licm', 'loop-rotate', 'instcombine', 'simplifycfg'],
        # 4-pass interactions
        ['mem2reg', 'instcombine', 'simplifycfg', 'jump-threading'],
        ['sroa', 'instcombine', 'simplifycfg', 'gvn'],
        ['loop-rotate', 'licm', 'instcombine', 'simplifycfg'],
        ['sccp', 'instcombine', 'simplifycfg', 'jump-threading'],
        ['gvn', 'dce', 'simplifycfg', 'instcombine'],
    ]

    print(f"Cycle detection: {len(candidate_subsets)} subsets x {len(benchmarks_subset)} benchmarks")

    start_time = time.time()
    all_trajectories = []
    true_cycles = []
    convergence_points = []  # Fixed-point convergence (cycle_length=1)

    log_file = open(LOG_DIR / "cycle_detection.log", 'w')

    for si, subset in enumerate(candidate_subsets):
        subset_str = '+'.join(subset)
        log_file.write(f"\n=== Subset {si+1}/{len(candidate_subsets)}: {subset_str} ===\n")

        for bm in benchmarks_subset:
            trajectory, cycle_info = detect_cycles(subset, str(bm), max_iterations=100)
            all_trajectories.extend(trajectory)

            if cycle_info is not None:
                true_cycles.append(cycle_info)
                log_file.write(f"  CYCLE FOUND: {bm.stem} - length={cycle_info['cycle_length']}, "
                              f"amplitude={cycle_info['ic_amplitude']}, verified={cycle_info['verified']}\n")
                log_file.write(f"    IC trajectory: {cycle_info['cycle_ics']}\n")
            else:
                # Check if it was fixed-point convergence
                if trajectory:
                    last_few = trajectory[-3:] if len(trajectory) >= 3 else trajectory
                    hashes = [t['ir_hash'] for t in last_few]
                    if len(set(hashes)) == 1:
                        convergence_points.append({
                            'subset': subset_str,
                            'benchmark': bm.stem,
                            'converged_at': trajectory[-1]['iteration'] if trajectory else 0,
                        })
                        log_file.write(f"  CONVERGED: {bm.stem} at iter {convergence_points[-1]['converged_at']}\n")

        elapsed = time.time() - start_time
        print(f"  Subset {si+1}/{len(candidate_subsets)} done ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    log_file.write(f"\nTotal runtime: {elapsed:.1f}s\n")
    log_file.close()
    print(f"Completed in {elapsed:.1f}s")

    # Save raw trajectory data
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if all_trajectories:
        with open(RESULTS_DIR / "cycle_detection_raw.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_trajectories[0].keys()))
            writer.writeheader()
            writer.writerows(all_trajectories)

    # Summary
    print(f"\n=== Cycle Detection Summary ===")
    print(f"True oscillating cycles (length >= 2): {len(true_cycles)}")
    print(f"Fixed-point convergences: {len(convergence_points)}")

    if true_cycles:
        print("\nDetected cycles:")
        for c in true_cycles:
            print(f"  {c['subset']} on {c['benchmark']}: length={c['cycle_length']}, "
                  f"IC amplitude={c['ic_amplitude']}, verified={c['verified']}")

    # Save cycle details
    cycle_summary = []
    for c in true_cycles:
        cycle_summary.append({
            'subset': c['subset'],
            'benchmark': c['benchmark'],
            'cycle_length': c['cycle_length'],
            'cycle_start_iter': c['cycle_start_iter'],
            'detection_iter': c['detection_iter'],
            'verified': c['verified'],
            'ic_amplitude': c['ic_amplitude'],
        })

    if cycle_summary:
        with open(RESULTS_DIR / "minimal_cycles.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(cycle_summary[0].keys()))
            writer.writeheader()
            writer.writerows(cycle_summary)

    # Save convergence data
    if convergence_points:
        with open(RESULTS_DIR / "convergence_points.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subset', 'benchmark', 'converged_at'])
            writer.writeheader()
            writer.writerows(convergence_points)

    exp_results = {
        'experiment': 'cycle_detection',
        'num_subsets': len(candidate_subsets),
        'num_benchmarks': len(benchmarks_subset),
        'true_oscillation_cycles': len(true_cycles),
        'fixed_point_convergences': len(convergence_points),
        'runtime_seconds': round(elapsed, 1),
        'cycle_details': cycle_summary[:20] if cycle_summary else [],
        'note': 'Only cycles with length >= 2 (true oscillation) are reported. '
                'Fixed-point convergence (repeated same state) is tracked separately.'
    }
    with open(Path(__file__).parent / 'results.json', 'w') as f:
        json.dump(exp_results, f, indent=2)


if __name__ == '__main__':
    main()
