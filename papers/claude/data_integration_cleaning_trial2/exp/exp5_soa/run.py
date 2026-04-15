"""Experiment 5: Stage-Optimal Allocation (SOA) validation."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASETS, RANDOM_SEEDS, RESULTS_DIR
from src.epm.propagation_model import ErrorPropagationModel
from src.epm.soa import soa_allocate, uniform_allocate, bottleneck_allocate


def main():
    results_dir = os.path.join(RESULTS_DIR, 'exp5')
    os.makedirs(results_dir, exist_ok=True)

    # Load exp1 data and fitted EPM
    with open(os.path.join(RESULTS_DIR, 'exp1', 'all_results.json')) as f:
        exp1_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp3', 'epm_validation.json')) as f:
        epm_info = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'exp4', 'eaf_analysis.json')) as f:
        eaf_data = json.load(f)

    epm = ErrorPropagationModel()
    epm.params = np.array([epm_info['epm_params'][n] for n in epm.param_names])

    budget_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
    degradation_amount = 0.30  # Start from 30% degraded baseline

    all_results = []

    for dataset in DATASETS:
        print(f"\n=== {dataset} ===")

        # Get optimal metrics from best config
        ds_exp1 = [r for r in exp1_results if r['dataset'] == dataset and 'error' not in r and r.get('e2e_f1', 0) > 0]
        if not ds_exp1:
            continue

        # Get average metrics across the best configuration
        best_f1 = max(r['e2e_f1'] for r in ds_exp1)
        best_configs = [r for r in ds_exp1 if abs(r['e2e_f1'] - best_f1) < 0.05]

        optimal_metrics = {
            'blocking_pc': np.mean([r['blocking_pc'] for r in best_configs]),
            'matching_recall': np.mean([r['matching_recall'] for r in best_configs]),
            'matching_precision': np.mean([r['matching_precision'] for r in best_configs]),
            'cluster_recall': np.mean([r['cluster_recall'] for r in best_configs]),
            'cluster_precision': np.mean([r['cluster_precision'] for r in best_configs]),
        }

        # Create degraded baseline (30% degradation = reduce each metric by 30%)
        degraded_metrics = {
            'blocking_pc': optimal_metrics['blocking_pc'] * (1 - degradation_amount),
            'matching_recall': optimal_metrics['matching_recall'] * (1 - degradation_amount),
            'matching_precision': optimal_metrics['matching_precision'] * (1 - degradation_amount),
            'cluster_recall': optimal_metrics['cluster_recall'] * (1 - degradation_amount),
            'cluster_precision': optimal_metrics['cluster_precision'] * (1 - degradation_amount),
        }

        # Get EAFs for bottleneck strategy
        eafs = eaf_data.get(dataset, {}).get('empirical', {}).get('normalized', {'blocking': 1/3, 'matching': 1/3, 'clustering': 1/3})

        for seed in RANDOM_SEEDS:
            rng = np.random.RandomState(seed)

            for budget in budget_levels:
                for strategy_name in ['uniform', 'bottleneck', 'soa']:
                    if strategy_name == 'uniform':
                        alloc = uniform_allocate(budget)
                    elif strategy_name == 'bottleneck':
                        alloc = bottleneck_allocate(eafs, budget)
                    else:
                        result = soa_allocate(epm, degraded_metrics, budget)
                        alloc = result['allocation']

                    # Apply allocation: interpolate between degraded and optimal
                    improved_metrics = {}
                    improved_metrics['blocking_pc'] = min(
                        degraded_metrics['blocking_pc'] + alloc['blocking'] *
                        (optimal_metrics['blocking_pc'] - degraded_metrics['blocking_pc']) / degradation_amount,
                        optimal_metrics['blocking_pc']
                    )
                    improved_metrics['matching_recall'] = min(
                        degraded_metrics['matching_recall'] + alloc['matching'] / 2 *
                        (optimal_metrics['matching_recall'] - degraded_metrics['matching_recall']) / degradation_amount,
                        optimal_metrics['matching_recall']
                    )
                    improved_metrics['matching_precision'] = min(
                        degraded_metrics['matching_precision'] + alloc['matching'] / 2 *
                        (optimal_metrics['matching_precision'] - degraded_metrics['matching_precision']) / degradation_amount,
                        optimal_metrics['matching_precision']
                    )
                    improved_metrics['cluster_recall'] = min(
                        degraded_metrics['cluster_recall'] + alloc['clustering'] / 2 *
                        (optimal_metrics['cluster_recall'] - degraded_metrics['cluster_recall']) / degradation_amount,
                        optimal_metrics['cluster_recall']
                    )
                    improved_metrics['cluster_precision'] = min(
                        degraded_metrics['cluster_precision'] + alloc['clustering'] / 2 *
                        (optimal_metrics['cluster_precision'] - degraded_metrics['cluster_precision']) / degradation_amount,
                        optimal_metrics['cluster_precision']
                    )

                    # Predict F1 with improved metrics
                    pred_f1, _, _ = epm.predict_f1(
                        np.array([improved_metrics['blocking_pc']]),
                        np.array([improved_metrics['matching_recall']]),
                        np.array([improved_metrics['cluster_recall']]),
                        np.array([improved_metrics['matching_precision']]),
                        np.array([improved_metrics['cluster_precision']]),
                    )

                    # Also compute degraded baseline F1
                    base_f1, _, _ = epm.predict_f1(
                        np.array([degraded_metrics['blocking_pc']]),
                        np.array([degraded_metrics['matching_recall']]),
                        np.array([degraded_metrics['cluster_recall']]),
                        np.array([degraded_metrics['matching_precision']]),
                        np.array([degraded_metrics['cluster_precision']]),
                    )

                    delta_f1 = float(pred_f1[0] - base_f1[0])
                    efficiency = delta_f1 / budget if budget > 0 else 0

                    all_results.append({
                        'dataset': dataset,
                        'seed': seed,
                        'budget': budget,
                        'strategy': strategy_name,
                        'allocation': alloc,
                        'predicted_f1': float(pred_f1[0]),
                        'base_f1': float(base_f1[0]),
                        'delta_f1': delta_f1,
                        'efficiency': efficiency,
                    })

        # Print summary for this dataset
        for strategy in ['uniform', 'bottleneck', 'soa']:
            strat_results = [r for r in all_results if r['dataset'] == dataset and r['strategy'] == strategy]
            avg_eff = np.mean([r['efficiency'] for r in strat_results])
            print(f"  {strategy}: avg efficiency = {avg_eff:.4f}")

    # Compute overall comparison
    print("\n=== Overall Comparison ===")
    for strategy in ['uniform', 'bottleneck', 'soa']:
        strat_results = [r for r in all_results if r['strategy'] == strategy]
        avg_eff = np.mean([r['efficiency'] for r in strat_results])
        avg_delta = np.mean([r['delta_f1'] for r in strat_results])
        print(f"  {strategy}: avg efficiency = {avg_eff:.4f}, avg delta_F1 = {avg_delta:.4f}")

    # Per-dataset: SOA vs uniform ratio
    print("\n=== SOA vs Uniform Efficiency Ratio by Dataset ===")
    datasets_outperformed = 0
    for dataset in DATASETS:
        uniform_effs = [r['efficiency'] for r in all_results if r['dataset'] == dataset and r['strategy'] == 'uniform']
        soa_effs = [r['efficiency'] for r in all_results if r['dataset'] == dataset and r['strategy'] == 'soa']
        if uniform_effs and soa_effs:
            ratio = np.mean(soa_effs) / np.mean(uniform_effs) if np.mean(uniform_effs) > 0 else float('inf')
            print(f"  {dataset}: SOA/Uniform = {ratio:.2f}x")
            if ratio >= 1.2:
                datasets_outperformed += 1
    print(f"\nDatasets where SOA outperforms uniform by >= 20%: {datasets_outperformed}/{len(DATASETS)}")

    # Save results
    with open(os.path.join(results_dir, 'soa_validation.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
