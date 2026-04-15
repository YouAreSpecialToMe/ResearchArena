"""Experiment 4: Bottleneck identification, EAF analysis, and Thirumuruganathan comparison."""
import os
import sys
import json
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASETS, DATASET_DIFFICULTY, RESULTS_DIR, MATCHING_METHODS

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'exp4.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.epm.propagation_model import ErrorPropagationModel


def compute_empirical_eafs(exp2_results, dataset):
    """Compute empirical EAFs from degradation curves."""
    ds_results = [r for r in exp2_results if r['dataset'] == dataset]
    baseline = [r for r in ds_results if r['stage'] == 'none']
    if not baseline:
        return None

    base_f1 = np.mean([r['degraded_e2e_f1'] for r in baseline])

    eafs = {}
    per_level = {}
    for stage in ['blocking', 'matching', 'clustering']:
        slopes = []
        stage_results = [r for r in ds_results if r['stage'] == stage and r.get('mode') in ('both', 'split')]
        level_data = {}
        for r in stage_results:
            if r['degradation_level'] > 0:
                delta_f1 = base_f1 - r['degraded_e2e_f1']
                slope = delta_f1 / r['degradation_level']
                slopes.append(abs(slope))
                lev = r['degradation_level']
                if lev not in level_data:
                    level_data[lev] = []
                level_data[lev].append(r['degraded_e2e_f1'])
        eafs[stage] = np.mean(slopes) if slopes else 0.0
        per_level[stage] = {str(k): float(np.mean(v)) for k, v in level_data.items()}

    total = sum(eafs.values())
    eafs_norm = {k: v / total for k, v in eafs.items()} if total > 0 else {k: 1/3 for k in eafs}

    return {'raw': {k: float(v) for k, v in eafs.items()}, 'normalized': eafs_norm, 'per_level': per_level, 'base_f1': float(base_f1)}


def thirumuruganathan_comparison(exp1_results, all_eafs):
    """Compare pipeline-aware EAF analysis vs matching-only analysis.

    Thirumuruganathan et al. (2019) focus on cleaning strategies based on
    matching-stage analysis alone. We test whether pipeline-aware EAFs give
    different recommendations.
    """
    comparison = {}
    for dataset in DATASETS:
        if dataset not in all_eafs:
            continue

        # Matching-only analysis: bottleneck is always matching by definition
        matching_only_recommendation = 'matching'

        # Pipeline-aware analysis: bottleneck from EAFs
        pipeline_bottleneck = all_eafs[dataset]['empirical_bottleneck']

        # Check matching-stage metrics
        ds_results = [r for r in exp1_results if r['dataset'] == dataset and 'error' not in r and r.get('e2e_f1', 0) > 0]
        if not ds_results:
            continue

        avg_mp = np.mean([r['matching_precision'] for r in ds_results])
        avg_mr = np.mean([r['matching_recall'] for r in ds_results])
        avg_pc = np.mean([r['blocking_pc'] for r in ds_results])

        # Matching-only would recommend: improve precision if precision < recall, else recall
        if avg_mp < avg_mr:
            matching_focus = 'matching_precision'
        else:
            matching_focus = 'matching_recall'

        disagrees = pipeline_bottleneck != 'matching'

        comparison[dataset] = {
            'matching_only_bottleneck': matching_only_recommendation,
            'matching_only_focus': matching_focus,
            'pipeline_bottleneck': pipeline_bottleneck,
            'disagrees': disagrees,
            'avg_matching_precision': float(avg_mp),
            'avg_matching_recall': float(avg_mr),
            'avg_blocking_pc': float(avg_pc),
            'analysis': (
                f"Pipeline-aware analysis identifies {pipeline_bottleneck} as bottleneck "
                f"(PC={avg_pc:.3f}, MP={avg_mp:.3f}, MR={avg_mr:.3f}). "
                f"Matching-only analysis would always recommend improving matching. "
                f"{'These disagree.' if disagrees else 'These agree.'}"
            ),
        }

    return comparison


def method_specific_eafs(exp2_results, exp1_results, epm):
    """Compute EAFs for each matching method separately."""
    method_eafs = {}
    for method in MATCHING_METHODS:
        method_eafs[method] = {}
        for dataset in DATASETS:
            # Get exp1 results for this method
            ds_results = [r for r in exp1_results
                         if r['dataset'] == dataset and r.get('matching_method') == method
                         and 'error' not in r and r.get('e2e_f1', 0) > 0]
            if not ds_results:
                continue

            avg_metrics = {
                'blocking_pc': np.mean([r['blocking_pc'] for r in ds_results]),
                'matching_recall': np.mean([r['matching_recall'] for r in ds_results]),
                'matching_precision': np.mean([r['matching_precision'] for r in ds_results]),
                'cluster_recall': np.mean([r['cluster_recall'] for r in ds_results]),
                'cluster_precision': np.mean([r['cluster_precision'] for r in ds_results]),
            }

            _, eafs_norm = epm.compute_eaf_analytical(
                avg_metrics['blocking_pc'], avg_metrics['matching_recall'],
                avg_metrics['cluster_recall'], avg_metrics['matching_precision'],
                avg_metrics['cluster_precision']
            )

            bottleneck = max(eafs_norm, key=eafs_norm.get)
            method_eafs[method][dataset] = {
                'eafs': {k: float(v) for k, v in eafs_norm.items()},
                'bottleneck': bottleneck,
                'avg_e2e_f1': float(np.mean([r['e2e_f1'] for r in ds_results])),
            }

    return method_eafs


def main():
    results_dir = os.path.join(RESULTS_DIR, 'exp4')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, 'exp1', 'all_results.json')) as f:
        exp1_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp2', 'all_results.json')) as f:
        exp2_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp3', 'epm_validation.json')) as f:
        epm_info = json.load(f)

    epm = ErrorPropagationModel()
    epm.params = np.array([epm_info['epm_params'][n] for n in epm.param_names])

    all_eafs = {}
    for dataset in DATASETS:
        logger.info(f"\n=== {dataset} (difficulty: {DATASET_DIFFICULTY[dataset]}) ===")

        empirical = compute_empirical_eafs(exp2_results, dataset)
        if empirical is None:
            logger.warning("  No degradation data available")
            continue

        logger.info(f"  Empirical EAFs (normalized): {empirical['normalized']}")

        ds_exp1 = [r for r in exp1_results if r['dataset'] == dataset and 'error' not in r and r.get('e2e_f1', 0) > 0]
        if not ds_exp1:
            continue

        best_f1 = max(r['e2e_f1'] for r in ds_exp1)
        best_configs = [r for r in ds_exp1 if abs(r['e2e_f1'] - best_f1) < 0.01]
        avg_metrics = {
            'blocking_pc': np.mean([r['blocking_pc'] for r in best_configs]),
            'matching_recall': np.mean([r['matching_recall'] for r in best_configs]),
            'matching_precision': np.mean([r['matching_precision'] for r in best_configs]),
            'cluster_recall': np.mean([r['cluster_recall'] for r in best_configs]),
            'cluster_precision': np.mean([r['cluster_precision'] for r in best_configs]),
        }

        analytical_raw, analytical_norm = epm.compute_eaf_analytical(
            avg_metrics['blocking_pc'], avg_metrics['matching_recall'],
            avg_metrics['cluster_recall'], avg_metrics['matching_precision'],
            avg_metrics['cluster_precision']
        )
        logger.info(f"  Analytical EAFs (normalized): {analytical_norm}")

        comparison = {}
        for stage in ['blocking', 'matching', 'clustering']:
            emp = empirical['normalized'][stage]
            ana = analytical_norm[stage]
            rel_error = abs(emp - ana) / max(emp, 0.001)
            comparison[stage] = {
                'empirical': float(emp),
                'analytical': float(ana),
                'relative_error': float(rel_error),
            }
            logger.info(f"  {stage}: emp={emp:.3f}, ana={ana:.3f}, rel_err={rel_error:.3f}")

        emp_bottleneck = max(empirical['normalized'], key=empirical['normalized'].get)
        ana_bottleneck = max(analytical_norm, key=analytical_norm.get)

        all_eafs[dataset] = {
            'difficulty': DATASET_DIFFICULTY[dataset],
            'empirical': empirical,
            'analytical_raw': {k: float(v) for k, v in analytical_raw.items()},
            'analytical_normalized': {k: float(v) for k, v in analytical_norm.items()},
            'comparison': comparison,
            'empirical_bottleneck': emp_bottleneck,
            'analytical_bottleneck': ana_bottleneck,
            'bottleneck_agreement': emp_bottleneck == ana_bottleneck,
        }

    # Bottleneck by difficulty
    logger.info("\n=== Bottleneck by Difficulty ===")
    for diff in ['easy', 'medium', 'hard']:
        diff_datasets = [ds for ds in DATASETS if DATASET_DIFFICULTY[ds] == diff and ds in all_eafs]
        if diff_datasets:
            bottlenecks = [all_eafs[ds]['empirical_bottleneck'] for ds in diff_datasets]
            logger.info(f"  {diff}: {diff_datasets} -> bottlenecks: {bottlenecks}")

    # EAF ratio analysis
    logger.info("\n=== EAF Ratios (max/min) ===")
    datasets_with_2x = 0
    for ds, eaf_data in all_eafs.items():
        norm = eaf_data['empirical']['normalized']
        max_eaf = max(norm.values())
        min_eaf = min(norm.values())
        ratio = max_eaf / min_eaf if min_eaf > 0.001 else float('inf')
        logger.info(f"  {ds}: ratio={ratio:.2f}")
        if ratio >= 2.0:
            datasets_with_2x += 1

    logger.info(f"\nDatasets with >= 2x EAF ratio: {datasets_with_2x}/{len(all_eafs)}")

    # Thirumuruganathan comparison
    logger.info("\n=== Thirumuruganathan Comparison ===")
    thiru_comparison = thirumuruganathan_comparison(exp1_results, all_eafs)
    for ds, comp in thiru_comparison.items():
        logger.info(f"  {ds}: {comp['analysis']}")

    # Method-specific EAFs
    logger.info("\n=== Method-specific bottleneck analysis ===")
    method_eaf_results = method_specific_eafs(exp2_results, exp1_results, epm)
    for method, method_data in method_eaf_results.items():
        bottlenecks = [v['bottleneck'] for v in method_data.values()]
        logger.info(f"  {method}: bottlenecks = {bottlenecks}")

    # Save all results
    with open(os.path.join(results_dir, 'eaf_analysis.json'), 'w') as f:
        json.dump(all_eafs, f, indent=2)
    with open(os.path.join(results_dir, 'thirumuruganathan_comparison.json'), 'w') as f:
        json.dump(thiru_comparison, f, indent=2)
    with open(os.path.join(results_dir, 'method_specific_eafs.json'), 'w') as f:
        json.dump(method_eaf_results, f, indent=2)

    logger.info(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
