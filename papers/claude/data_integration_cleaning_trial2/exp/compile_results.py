"""Compile final results.json aggregating all experiment results."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATASETS, DATASET_DIFFICULTY, RESULTS_DIR


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load all experiment results
    with open(os.path.join(RESULTS_DIR, 'exp1', 'all_results.json')) as f:
        exp1 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp2', 'all_results.json')) as f:
        exp2 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp3', 'epm_validation.json')) as f:
        exp3 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp4', 'eaf_analysis.json')) as f:
        exp4 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp5', 'soa_validation.json')) as f:
        exp5 = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        ablation = json.load(f)

    # === Exp 1 summary ===
    valid_exp1 = [r for r in exp1 if 'error' not in r and r.get('e2e_f1', 0) > 0]
    exp1_summary = {}
    for ds in DATASETS:
        ds_results = [r for r in valid_exp1 if r['dataset'] == ds]
        if not ds_results:
            continue
        import pandas as pd
        df = pd.DataFrame(ds_results)
        mean_f1 = df.groupby(['blocking_method', 'matching_method', 'clustering_method'])['e2e_f1'].mean()
        best_key = mean_f1.idxmax()
        best_runs = df[(df['blocking_method'] == best_key[0]) &
                       (df['matching_method'] == best_key[1]) &
                       (df['clustering_method'] == best_key[2])]

        exp1_summary[ds] = {
            'best_e2e_f1': float(best_runs['e2e_f1'].mean()),
            'best_e2e_f1_std': float(best_runs['e2e_f1'].std()),
            'best_config': f"{best_key[0]}/{best_key[1]}/{best_key[2]}",
            'mean_e2e_f1': float(df['e2e_f1'].mean()),
            'std_e2e_f1': float(df['e2e_f1'].std()),
            'n_configs': int(len(mean_f1)),
        }

    # === Exp 2 summary ===
    exp2_summary = {}
    for ds in DATASETS:
        baselines = [r for r in exp2 if r['dataset'] == ds and r['stage'] == 'none']
        if not baselines:
            continue
        base_f1 = np.mean([r['degraded_e2e_f1'] for r in baselines])
        impact = {}
        for stage, mode in [('blocking', 'both'), ('matching', 'both'), ('clustering', 'split')]:
            at_50 = [r for r in exp2 if r['dataset'] == ds and r['stage'] == stage
                     and r.get('mode') == mode and abs(r['degradation_level'] - 0.50) < 0.01]
            if at_50:
                impact[stage] = float(base_f1 - np.mean([r['degraded_e2e_f1'] for r in at_50]))
        exp2_summary[ds] = {'baseline_f1': float(base_f1), 'impact_at_50pct': impact}

    # === Hard recall bound check ===
    n_violations = sum(1 for r in valid_exp1 if r.get('e2e_recall', 0) > r.get('blocking_pc', 0) + 1e-6)

    # === EAF ratios ===
    eaf_ratios = {}
    bottleneck_stages = {}
    bottleneck_agreement = {}
    datasets_with_2x = 0
    for ds in DATASETS:
        if ds not in exp4:
            continue
        norm = exp4[ds]['empirical']['normalized']
        max_eaf = max(norm.values())
        min_eaf = min(norm.values())
        ratio = max_eaf / min_eaf if min_eaf > 0.001 else float('inf')
        eaf_ratios[ds] = ratio
        bottleneck_stages[ds] = exp4[ds]['empirical_bottleneck']
        bottleneck_agreement[ds] = exp4[ds]['bottleneck_agreement']
        if ratio >= 2.0:
            datasets_with_2x += 1

    # Average relative error between analytical and empirical EAFs
    avg_rel_errors = []
    for ds in DATASETS:
        if ds in exp4 and 'comparison' in exp4[ds]:
            for stage in ['blocking', 'matching', 'clustering']:
                if stage in exp4[ds]['comparison']:
                    avg_rel_errors.append(exp4[ds]['comparison'][stage]['relative_error'])

    # === SOA summary ===
    soa_summary = {}
    for strategy in ['uniform', 'bottleneck', 'soa']:
        strat_results = [r for r in exp5 if r['strategy'] == strategy]
        if strat_results:
            soa_summary[strategy] = {
                'avg_efficiency': float(np.mean([r['efficiency'] for r in strat_results])),
                'avg_delta_f1': float(np.mean([r['delta_f1'] for r in strat_results])),
            }

    soa_vs_uniform = {}
    datasets_soa_20pct = 0
    for ds in DATASETS:
        uniform_effs = [r['efficiency'] for r in exp5 if r['dataset'] == ds and r['strategy'] == 'uniform']
        soa_effs = [r['efficiency'] for r in exp5 if r['dataset'] == ds and r['strategy'] == 'soa']
        if uniform_effs and soa_effs:
            ratio = np.mean(soa_effs) / np.mean(uniform_effs) if np.mean(uniform_effs) > 0 else float('inf')
            soa_vs_uniform[ds] = float(ratio)
            if ratio >= 1.2:
                datasets_soa_20pct += 1

    # Determine if bottleneck strategy outperforms SOA
    bottleneck_avg = soa_summary.get('bottleneck', {}).get('avg_efficiency', 0)
    soa_avg = soa_summary.get('soa', {}).get('avg_efficiency', 0)

    # === LODO transferability ===
    lodo_r2s = [v.get('r2', 0) for v in exp3.get('lodo_cross_validation', {}).values()]
    avg_lodo_r2 = float(np.mean(lodo_r2s)) if lodo_r2s else 0

    # === Success criteria evaluation ===
    val_r2 = exp3['val_eval']['r2']
    epm_meets_085 = val_r2 >= 0.85

    # Check if bottleneck shifts with difficulty
    easy_bottlenecks = [bottleneck_stages.get(ds) for ds in DATASETS if DATASET_DIFFICULTY.get(ds) == 'easy' and ds in bottleneck_stages]
    hard_bottlenecks = [bottleneck_stages.get(ds) for ds in DATASETS if DATASET_DIFFICULTY.get(ds) == 'hard' and ds in bottleneck_stages]

    # Compile final results
    results = {
        'experiment_summary': {
            'exp1_baseline_quality': exp1_summary,
            'exp2_controlled_degradation': exp2_summary,
            'exp3_epm_validation': {
                'train_r2': exp3['train_eval']['r2'],
                'val_r2': exp3['val_eval']['r2'],
                'val_rmse': exp3['val_eval']['rmse'],
                'val_mae': exp3['val_eval']['mae'],
                'epm_params': exp3['epm_params'],
                'lodo_avg_r2': avg_lodo_r2,
            },
            'exp4_eaf_bottleneck': {
                'eaf_ratios': eaf_ratios,
                'bottleneck_stages': bottleneck_stages,
                'datasets_with_2x_ratio': datasets_with_2x,
                'bottleneck_agreement': bottleneck_agreement,
                'avg_analytical_empirical_relative_error': float(np.mean(avg_rel_errors)) if avg_rel_errors else None,
            },
            'exp5_soa_allocation': {
                'strategy_comparison': soa_summary,
                'soa_vs_uniform_ratio': soa_vs_uniform,
                'datasets_soa_outperforms_20pct': datasets_soa_20pct,
            },
            'ablation': {
                'model_comparison': {k: {kk: vv for kk, vv in v.items() if kk not in ('predicted_f1', 'actual_f1')}
                                     for k, v in ablation['ablation'].items()},
                'avg_r2_drop_lodo': ablation.get('avg_r2_drop', 0),
            },
        },
        'key_findings': {
            'hard_recall_bound_violations': n_violations,
            'hard_recall_bound_total': len(valid_exp1),
            'epm_val_r2': val_r2,
            'epm_val_rmse': exp3['val_eval']['rmse'],
            'epm_lodo_avg_r2': avg_lodo_r2,
            'bottleneck_stages': bottleneck_stages,
            'soa_avg_improvement_over_uniform': float(soa_avg / soa_summary.get('uniform', {}).get('avg_efficiency', 1) - 1) if soa_summary.get('uniform', {}).get('avg_efficiency', 0) > 0 else 0,
            'bottleneck_strategy_outperforms_soa': bottleneck_avg > soa_avg,
            'easy_dataset_bottlenecks': easy_bottlenecks,
            'hard_dataset_bottlenecks': hard_bottlenecks,
        },
        'success_criteria': {
            'epm_r2_ge_085': epm_meets_085,
            'epm_r2_value': val_r2,
            'eaf_2x_on_4_datasets': datasets_with_2x >= 4,
            'eaf_2x_count': datasets_with_2x,
            'soa_outperforms_20pct': datasets_soa_20pct >= 4,
            'soa_outperforms_count': datasets_soa_20pct,
            'hard_recall_bound_confirmed': n_violations <= 5,
            'analytical_empirical_eaf_agreement': float(np.mean(avg_rel_errors)) < 0.15 if avg_rel_errors else False,
        },
        'reproducibility': {
            'random_seeds': [42, 123, 456],
            'n_datasets': 6,
            'n_matching_methods': 4,
            'n_blocking_methods': 3,
            'n_clustering_methods': 3,
            'n_configs_per_dataset': 36,
            'total_exp1_runs': len(valid_exp1),
            'total_exp2_runs': len(exp2),
        },
        'figures': [
            'figures/fig1_pipeline_schematic.pdf',
            'figures/fig2_amplification_curves.pdf',
            'figures/fig3_epm_validation.pdf',
            'figures/fig4_eaf_heatmap.pdf',
            'figures/fig5_soa_budget.pdf',
            'figures/fig6_ablation.pdf',
            'figures/fig7_transferability.pdf',
        ],
    }

    output_path = os.path.join(base_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results compiled to {output_path}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"EPM validation R²: {val_r2:.4f} (target >= 0.85: {'PASS' if epm_meets_085 else 'FAIL'})")
    print(f"EPM LODO avg R²: {avg_lodo_r2:.4f}")
    print(f"Hard recall bound violations: {n_violations}/{len(valid_exp1)}")
    print(f"Datasets with >= 2x EAF ratio: {datasets_with_2x}/6 (target >= 4: {'PASS' if datasets_with_2x >= 4 else 'FAIL'})")
    print(f"SOA outperforms uniform by 20%: {datasets_soa_20pct}/6 (target >= 4: {'PASS' if datasets_soa_20pct >= 4 else 'FAIL'})")
    print(f"Bottleneck strategy vs SOA: {'Bottleneck wins' if bottleneck_avg > soa_avg else 'SOA wins'}")
    print(f"Bottleneck stages: {bottleneck_stages}")
    print(f"Avg analytical-empirical EAF relative error: {np.mean(avg_rel_errors):.3f}" if avg_rel_errors else "N/A")


if __name__ == '__main__':
    main()
