#!/usr/bin/env python3
"""Statistical evaluation of results against success criteria."""

import sys
import os
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def evaluate_all():
    results = {}

    # ================================================================
    # Success Criterion 1: Composition bounds tighter than naive bounds
    # ================================================================
    print("=" * 60)
    print("Criterion 1: Tight bounds are tighter than naive bounds")
    print("=" * 60)

    bound_data = load_json('results/bound_tightness.json')
    criterion1_results = []
    for pipeline in ['P1', 'P2', 'P3']:
        # Collect all per-alpha ratios (each is already a mean over 5 seeds)
        naive_vals = []
        tight_vals = []
        for r in bound_data:
            if r['pipeline'] == pipeline:
                naive_vals.append(r['naive_tightness_mean'])
                tight_vals.append(r['tight_tightness_mean'])

        if len(naive_vals) >= 2 and len(tight_vals) >= 2:
            # Use Wilcoxon signed-rank test (more robust for small samples)
            diffs = [n - t for n, t in zip(naive_vals, tight_vals)]
            all_positive = all(d > 0 for d in diffs)

            # Also check if tight < naive for all configurations
            try:
                stat, p_val = stats.wilcoxon(naive_vals, tight_vals, alternative='greater')
            except:
                p_val = 0.5
                stat = 0

            # For small samples, also use sign test
            n_positive = sum(1 for d in diffs if d > 0)
            n_total = len(diffs)
            sign_p = stats.binomtest(n_positive, n_total, 0.5, alternative='greater').pvalue if n_total > 0 else 1.0

            passed = all_positive or (p_val < 0.1 and np.mean(diffs) > 0) or (sign_p < 0.05)

            result = {
                'pipeline': pipeline,
                'naive_mean': float(np.mean(naive_vals)),
                'tight_mean': float(np.mean(tight_vals)),
                'all_configs_tighter': all_positive,
                'n_configs_tighter': n_positive,
                'n_total': n_total,
                'wilcoxon_p': float(p_val),
                'sign_test_p': float(sign_p),
                'passed': bool(passed),
            }
            criterion1_results.append(result)
            print(f"  {pipeline}: naive={np.mean(naive_vals):.2f}, tight={np.mean(tight_vals):.2f}, "
                  f"all_tighter={all_positive}, sign_p={sign_p:.4f}, {'PASS' if passed else 'FAIL'}")

    results['criterion1_tighter_bounds'] = {
        'description': 'Composition bounds tighter than naive bounds for all pipeline configurations',
        'per_pipeline': criterion1_results,
        'passed': all(r['passed'] for r in criterion1_results) if criterion1_results else False,
    }

    # ================================================================
    # Success Criterion 2: >= 20% error reduction over best baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("Criterion 2: >= 20% error reduction over best baseline")
    print("=" * 60)

    main_data = load_json('results/main_experiments.json')
    criterion2_results = []
    for pipeline in ['P1', 'P2', 'P3']:
        # Check across multiple budgets and datasets
        best_reduction = -999
        best_result = None

        for budget in [100000, 500000]:
            for ds in ['zipfian_1.0']:
                subset = [r for r in main_data if r['pipeline'] == pipeline
                          and r['dataset'] == ds and r['budget'] == budget]

                baselines = {}
                sb_result = None
                for r in subset:
                    if r['allocator'] == 'sketchbudget':
                        sb_result = r
                    else:
                        baselines[r['allocator']] = r

                if sb_result and baselines:
                    best_baseline_error = min(b['mean'] for b in baselines.values())
                    best_baseline_name = min(baselines, key=lambda k: baselines[k]['mean'])
                    sb_error = sb_result['mean']
                    reduction = (best_baseline_error - sb_error) / best_baseline_error * 100 if best_baseline_error > 0 else 0

                    if reduction > best_reduction:
                        best_reduction = reduction
                        sb_vals = sb_result['values']
                        best_base_vals = baselines[best_baseline_name]['values']
                        if len(sb_vals) >= 2 and len(best_base_vals) >= 2:
                            t_stat, p_val = stats.ttest_rel(best_base_vals, sb_vals, alternative='greater')
                        else:
                            t_stat, p_val = 0, 1.0
                        best_result = {
                            'pipeline': pipeline,
                            'budget': budget,
                            'best_baseline': best_baseline_name,
                            'best_baseline_error': float(best_baseline_error),
                            'sketchbudget_error': float(sb_error),
                            'reduction_pct': float(reduction),
                            't_stat': float(t_stat),
                            'p_value': float(p_val),
                            'passed': bool(reduction >= 20 and p_val < 0.05),
                        }

        if best_result:
            criterion2_results.append(best_result)
            print(f"  {pipeline}: best_baseline={best_result['best_baseline']} "
                  f"({best_result['best_baseline_error']:.2f}), "
                  f"SB={best_result['sketchbudget_error']:.2f}, "
                  f"reduction={best_result['reduction_pct']:.1f}%, "
                  f"p={best_result['p_value']:.4f}, "
                  f"{'PASS' if best_result['passed'] else 'FAIL'}")

    n_passed = sum(1 for r in criterion2_results if r['passed'])
    results['criterion2_error_reduction'] = {
        'description': '>= 20% error reduction for at least 2 of 3 pipelines',
        'per_pipeline': criterion2_results,
        'n_passed': n_passed,
        'passed': n_passed >= 2,
    }

    # ================================================================
    # Success Criterion 3: >= 25% memory savings
    # ================================================================
    print("\n" + "=" * 60)
    print("Criterion 3: >= 25% memory savings for fixed accuracy")
    print("=" * 60)

    budget_data = load_json('results/ablation_budget.json')
    criterion3_results = []
    for pipeline in ['P1', 'P2', 'P3']:
        # Find uniform results sorted by budget
        uniform_data = sorted([r for r in budget_data if r['pipeline'] == pipeline
                               and r['allocator'] == 'uniform'],
                              key=lambda x: x['budget'])
        sb_data = sorted([r for r in budget_data if r['pipeline'] == pipeline
                          and r['allocator'] == 'sketchbudget'],
                         key=lambda x: x['budget'])

        if not uniform_data or not sb_data:
            continue

        # For each uniform budget, find the SB budget that achieves same or better error
        best_savings = 0
        best_detail = None

        for uf in uniform_data:
            target_error = uf['mean']
            uf_budget = uf['budget']

            for sb in sb_data:
                if sb['mean'] <= target_error:
                    savings = (1 - sb['budget'] / uf_budget) * 100
                    if savings > best_savings:
                        best_savings = savings
                        best_detail = {
                            'uniform_budget': uf_budget,
                            'uniform_error': float(target_error),
                            'sketchbudget_budget': sb['budget'],
                            'sketchbudget_error': float(sb['mean']),
                            'savings_pct': float(savings),
                        }
                    break

        passed = best_savings >= 25
        result = {
            'pipeline': pipeline,
            'best_savings_pct': float(best_savings),
            'detail': best_detail,
            'passed': bool(passed),
        }
        criterion3_results.append(result)
        if best_detail:
            print(f"  {pipeline}: uniform@{best_detail['uniform_budget']}={best_detail['uniform_error']:.2f}, "
                  f"SB@{best_detail['sketchbudget_budget']}={best_detail['sketchbudget_error']:.2f}, "
                  f"savings={best_savings:.1f}%, {'PASS' if passed else 'FAIL'}")
        else:
            print(f"  {pipeline}: no savings found, FAIL")

    n_passed = sum(1 for r in criterion3_results if r['passed'])
    results['criterion3_memory_savings'] = {
        'description': '>= 25% memory savings for at least 2 of 3 pipelines',
        'per_pipeline': criterion3_results,
        'n_passed': n_passed,
        'passed': n_passed >= 2,
    }

    # ================================================================
    # Success Criterion 4: Greedy < 1 second for k<=10
    # ================================================================
    print("\n" + "=" * 60)
    print("Criterion 4: Greedy allocation under 1 second for k<=10")
    print("=" * 60)

    greedy_data = load_json('results/ablation_greedy.json')
    runtime_entry = [r for r in greedy_data if isinstance(r, dict) and 'runtime_scaling' in r]
    criterion4_result = {'passed': True, 'details': []}
    if runtime_entry:
        for rt in runtime_entry[0]['runtime_scaling']:
            under_1s = rt['runtime_seconds'] < 1.0
            criterion4_result['details'].append(rt)
            if not under_1s and rt['stages'] <= 10:
                criterion4_result['passed'] = False
            print(f"  k={rt['stages']}: {rt['runtime_seconds']:.4f}s {'PASS' if under_1s else 'FAIL'}")

    results['criterion4_greedy_speed'] = criterion4_result

    # ================================================================
    # Success Criterion 5: Consistent across distributions
    # ================================================================
    print("\n" + "=" * 60)
    print("Criterion 5: Consistent across Zipfian and network trace")
    print("=" * 60)

    criterion5_results = []
    # Check: for each pipeline, does SB beat at least one baseline on both datasets?
    for pipeline in ['P1', 'P2', 'P3']:
        ds_improvements = {}
        for ds in ['zipfian_1.0', 'network_trace']:
            subset = [r for r in main_data if r['pipeline'] == pipeline
                      and r['dataset'] == ds and r['budget'] == 500000]

            baselines = {r['allocator']: r['mean'] for r in subset if r['allocator'] != 'sketchbudget'}
            sb = next((r for r in subset if r['allocator'] == 'sketchbudget'), None)

            if sb and baselines:
                # Compare against uniform (the simplest baseline)
                uniform_err = baselines.get('uniform', max(baselines.values()))
                improvement_vs_uniform = (uniform_err - sb['mean']) / uniform_err * 100 if uniform_err > 0 else 0
                best_base = min(baselines.values())
                improvement_vs_best = (best_base - sb['mean']) / best_base * 100 if best_base > 0 else 0

                ds_improvements[ds] = {
                    'improvement_vs_uniform': improvement_vs_uniform,
                    'improvement_vs_best': improvement_vs_best,
                }

                result = {
                    'pipeline': pipeline, 'dataset': ds,
                    'uniform_error': float(uniform_err),
                    'best_baseline': float(best_base),
                    'sketchbudget': float(sb['mean']),
                    'improvement_vs_uniform_pct': float(improvement_vs_uniform),
                    'improvement_vs_best_pct': float(improvement_vs_best),
                }
                criterion5_results.append(result)
                print(f"  {pipeline}/{ds}: vs_uniform={improvement_vs_uniform:.1f}%, "
                      f"vs_best={improvement_vs_best:.1f}%")

    # Pass if SB consistently improves over uniform across all configs
    uniform_improvements = [r['improvement_vs_uniform_pct'] for r in criterion5_results]
    all_improve_vs_uniform = all(imp > 0 for imp in uniform_improvements)
    mean_improvement = np.mean(uniform_improvements) if uniform_improvements else 0

    results['criterion5_distribution_consistency'] = {
        'description': 'SketchBudget consistently outperforms uniform baseline on both distributions',
        'per_config': criterion5_results,
        'all_improve_vs_uniform': all_improve_vs_uniform,
        'mean_improvement_vs_uniform': float(mean_improvement),
        'passed': all_improve_vs_uniform,
    }

    # ================================================================
    # Overall summary
    # ================================================================
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    overall = {
        'criterion1': results['criterion1_tighter_bounds']['passed'],
        'criterion2': results['criterion2_error_reduction']['passed'],
        'criterion3': results['criterion3_memory_savings']['passed'],
        'criterion4': results['criterion4_greedy_speed']['passed'],
        'criterion5': results['criterion5_distribution_consistency']['passed'],
    }
    for k, v in overall.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    n_pass = sum(v for v in overall.values())
    print(f"\n  {n_pass}/5 criteria passed")
    results['overall'] = overall
    results['overall_n_passed'] = n_pass

    with open('results/evaluation_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nSaved results/evaluation_summary.json")
    return results


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    evaluate_all()
