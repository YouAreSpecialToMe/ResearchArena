#!/usr/bin/env python3
"""Finish: stat eval, figures, and final results using all saved data."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

# Import everything from run_final
from run_final import (
    get_benchmark_files, load_json, save_json, aggregate_interactions,
    statistical_evaluation, generate_all_figures, save_final_results,
    RESULTS_DIR, FIGURES_DIR
)

def main():
    benchmarks = get_benchmark_files()
    all_interactions = load_json(os.path.join(RESULTS_DIR, 'data', 'all_interactions.json'))
    decomp = load_json(os.path.join(RESULTS_DIR, 'data', 'variance_decomposition.json'))
    opt_levels = load_json(os.path.join(RESULTS_DIR, 'data', 'opt_levels.json'))
    selection_results = load_json(os.path.join(RESULTS_DIR, 'data', 'selection_results.json'))
    baseline_results = load_json(os.path.join(RESULTS_DIR, 'data', 'baseline_results.json'))
    ablation_order = load_json(os.path.join(RESULTS_DIR, 'data', 'ablation_order.json'))
    ablation_budget_res = load_json(os.path.join(RESULTS_DIR, 'data', 'ablation_budget.json'))
    transfer_results = load_json(os.path.join(RESULTS_DIR, 'data', 'transferability.json'))

    print("Loaded all data. Running stat eval + figures...")

    eval_results = statistical_evaluation(all_interactions, decomp, selection_results,
                                          baseline_results, benchmarks, opt_levels)

    generate_all_figures(all_interactions, decomp, selection_results, baseline_results,
                          benchmarks, opt_levels, transfer_results, ablation_order, ablation_budget_res)

    save_final_results(eval_results, decomp, selection_results, baseline_results,
                       benchmarks, opt_levels, ablation_order)

    print("Done!")

if __name__ == '__main__':
    main()
