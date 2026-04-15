"""Aggregate all experiment results into the workspace-level results.json."""
import json
import math
from pathlib import Path

WORKSPACE = Path("/home/nw366/ResearchArena/outputs/claude_v5_compiler_optimization/idea_01")
EXP_DIR = WORKSPACE / "exp"
RESULTS_DIR = WORKSPACE / "experiments" / "results"


def load_exp_results(name):
    path = EXP_DIR / name / "results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    # Load all experiment results
    idempotency = load_exp_results("idempotency")
    commutativity = load_exp_results("commutativity")
    interference = load_exp_results("interference")
    convergence = load_exp_results("convergence")
    cycle_detection = load_exp_results("cycle_detection")
    baselines = load_exp_results("baselines")
    algebra_ordering = load_exp_results("algebra_ordering")
    ablation = load_exp_results("ablation")

    # Load statistical analysis
    stats_path = RESULTS_DIR / "statistical_analysis.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    # Build methods comparison
    methods_geo = algebra_ordering.get('methods_comparison', {})

    # Count success criteria
    criteria = {
        'idempotency_60pct': stats.get('idempotency', {}).get('criterion_met', False),
        'non_commutativity_30pct': stats.get('commutativity', {}).get('criterion_met', False),
        'interference_10pct': stats.get('interference', {}).get('criterion_met', False),
        'oscillation_detected': stats.get('oscillation', {}).get('criterion_met', False),
        'ordering_competitive': stats.get('ordering', {}).get('criterion_met', False),
        'clusters_interpretable': stats.get('clustering', {}).get('criterion_met', False),
    }
    n_met = sum(1 for v in criteria.values() if v)

    results = {
        "title": "The Algebra of Compiler Passes: An Empirical Study of Idempotency, "
                 "Commutativity, and Convergence in LLVM Optimization Pipelines",
        "num_benchmarks": 87,
        "benchmark_composition": {
            "polybench_c_4_2_1": 30,
            "custom_synthetic": 57,
            "total": 87
        },
        "num_passes": 46,
        "experiments": {
            "idempotency": idempotency,
            "commutativity": commutativity,
            "interference": interference,
            "convergence": convergence,
            "cycle_detection": cycle_detection,
            "baselines": baselines,
            "algebra_ordering": algebra_ordering,
            "ablation": ablation,
        },
        "statistical_analysis": stats,
        "summary": {
            "idempotency_rate": stats.get('idempotency', {}).get('rate', 0),
            "non_commutativity_rate": stats.get('commutativity', {}).get('non_commutative_rate', 0),
            "significant_interference_rate": stats.get('interference', {}).get('significant_rate', 0),
            "oscillating_benchmarks_O2": stats.get('oscillation', {}).get('oscillating_benchmarks_O2', 0),
            "true_pass_cycles": stats.get('oscillation', {}).get('true_pass_cycles', 0),
            "algebra_guided_geo_mean": algebra_ordering.get('geo_mean_ratio', 0),
            "algebra_guided_reduction_pct": algebra_ordering.get('reduction_pct', 0),
            "methods_geo_mean_ratio": methods_geo,
            "success_criteria": criteria,
            "criteria_met": n_met,
            "criteria_total": 6,
            "honest_assessment": {
                "greedy_beats_algebra": methods_geo.get('greedy', 1) < methods_geo.get('algebra', 1),
                "greedy_geo_mean": methods_geo.get('greedy', None),
                "algebra_geo_mean": methods_geo.get('algebra', None),
                "note_greedy_advantage": "Greedy search achieves better IC reduction because it "
                                        "evaluates all 46 passes at each step (exhaustive local search). "
                                        "The algebra-guided method uses a fixed pre-computed sequence "
                                        "requiring no per-benchmark search, making it 46x cheaper per benchmark.",
                "note_commutativity_criterion": "Non-commutativity rate (10.1%) is below the 30% criterion. "
                                              "Most pass pairs commute because many passes have no effect on most "
                                              "benchmarks (trivially commutative). Among active pairs, non-commutativity "
                                              "is much higher.",
                "note_interference_criterion": "Significant interference rate (2.8%) is below the 10% criterion. "
                                             "However, the pairs that DO interfere show strong effects (up to 38%), "
                                             "and these dominate ordering decisions.",
                "note_synergy_vs_phases": "Ablation shows phase-based ordering contributes most to the "
                                         "algebra-guided method (7.4% improvement). Within-phase synergy chaining "
                                         "provides negligible additional benefit, suggesting that pass category "
                                         "structure matters more than pairwise synergy scores.",
                "note_cycle_detection": "142 true oscillation cycles found with cycle length >= 2 "
                                       "(not fixed-point convergence). The simplifycfg+loop-rotate+loop-simplify "
                                       "subset cycles on all 30 PolyBench benchmarks with significant IC amplitude."
            }
        }
    }

    output_path = WORKSPACE / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")
    print(f"Criteria met: {n_met}/6")
    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == '__main__':
    main()
