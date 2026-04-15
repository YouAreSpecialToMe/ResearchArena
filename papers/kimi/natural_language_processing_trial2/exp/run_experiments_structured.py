#!/usr/bin/env python3
"""
Structured CDHR Experiments
Runs experiments in phases to ensure everything completes within time limits.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'exp'))

from run_complete_experiments import (
    load_model, LLMWrapper, run_cot_baseline, run_cdhr_experiment,
    run_self_consistency_baseline, run_refinement_baseline
)


def run_phase1_baselines(model_name: str = "llama-3.1-8b", backend: str = "transformers", dataset: str = "data/gsm8k.json", limit: int = 300):
    """Phase 1: Run all baselines with 3 seeds."""
    print("\n" + "="*70)
    print("PHASE 1: BASELINE EXPERIMENTS")
    print("="*70)
    
    model_backend = load_model(model_name, backend=backend)
    model = LLMWrapper(model_backend, model_name)
    
    results = {}
    seeds = [42, 123, 456]
    
    # Baseline CoT - 3 seeds
    for seed in seeds:
        result = run_cot_baseline(
            model=model,
            dataset_path=dataset,
            output_path=f"results/baseline_cot_seed{seed}.json",
            seed=seed,
            limit=limit
        )
        results[f'baseline_cot_seed{seed}'] = result['metrics']
    
    # Self-Consistency-16 - 1 seed (expensive)
    result = run_self_consistency_baseline(
        model=model,
        dataset_path=dataset,
        output_path=f"results/baseline_sc16.json",
        seed=42,
        num_samples=16,
        limit=min(limit, 150)  # Smaller for SC
    )
    results['baseline_sc16'] = result['metrics']
    
    # Refinement baseline
    result = run_refinement_baseline(
        model=model,
        dataset_path=dataset,
        output_path=f"results/baseline_refinement.json",
        seed=42,
        limit=min(limit, 150)
    )
    results['baseline_refinement'] = result['metrics']
    
    # Save phase results
    with open('results/phase1_baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPhase 1 Complete!")
    return results


def run_phase2_cdhr_main(model_name: str = "llama-3.1-8b", backend: str = "transformers", dataset: str = "data/gsm8k.json", limit: int = 300):
    """Phase 2: Run CDHR main with 3 seeds."""
    print("\n" + "="*70)
    print("PHASE 2: CDHR MAIN EXPERIMENTS")
    print("="*70)
    
    model_backend = load_model(model_name, backend=backend)
    model = LLMWrapper(model_backend, model_name)
    
    results = {}
    seeds = [42, 123, 456]
    
    for seed in seeds:
        result = run_cdhr_experiment(
            model=model,
            dataset_path=dataset,
            output_path=f"results/cdhr_main_seed{seed}.json",
            seed=seed,
            theta_v=0.05,
            theta_sigma=0.1,
            beta=0.5,
            limit=limit
        )
        results[f'cdhr_main_seed{seed}'] = result['metrics']
    
    # Save phase results
    with open('results/phase2_cdhr_main.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPhase 2 Complete!")
    return results


def run_phase3_ablations(model_name: str = "llama-3.1-8b", backend: str = "transformers", dataset: str = "data/gsm8k.json", limit: int = 200):
    """Phase 3: Run all ablation studies."""
    print("\n" + "="*70)
    print("PHASE 3: ABLATION STUDIES")
    print("="*70)
    
    model_backend = load_model(model_name, backend=backend)
    model = LLMWrapper(model_backend, model_name)
    
    results = {}
    
    # Ablation 1: Token-only confidence
    result = run_cdhr_experiment(
        model=model,
        dataset_path=dataset,
        output_path="results/ablation_token_only.json",
        seed=42,
        theta_v=0.05,
        theta_sigma=0.1,
        beta=1.0,
        use_consistency=False,
        limit=limit
    )
    results['ablation_token_only'] = result['metrics']
    
    # Ablation 2: Consistency-only
    result = run_cdhr_experiment(
        model=model,
        dataset_path=dataset,
        output_path="results/ablation_consistency_only.json",
        seed=42,
        theta_v=0.05,
        theta_sigma=0.1,
        beta=0.0,
        limit=limit
    )
    results['ablation_consistency_only'] = result['metrics']
    
    # Ablation 3: Beta sensitivity
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = run_cdhr_experiment(
            model=model,
            dataset_path=dataset,
            output_path=f"results/ablation_beta{beta}.json",
            seed=42,
            theta_v=0.05,
            theta_sigma=0.1,
            beta=beta,
            limit=150
        )
        results[f'ablation_beta{beta}'] = result['metrics']
    
    # Ablation 4: Threshold sensitivity
    for theta_v in [0.03, 0.05, 0.07]:
        for theta_sigma in [0.075, 0.10, 0.125]:
            result = run_cdhr_experiment(
                model=model,
                dataset_path=dataset,
                output_path=f"results/ablation_th{theta_v}_{theta_sigma}.json",
                seed=42,
                theta_v=theta_v,
                theta_sigma=theta_sigma,
                beta=0.5,
                limit=100
            )
            results[f'ablation_th{theta_v}_{theta_sigma}'] = result['metrics']
    
    # Save phase results
    with open('results/phase3_ablations.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPhase 3 Complete!")
    return results


def run_phase4_additional_datasets(model_name: str = "llama-3.1-8b", backend: str = "transformers", limit: int = 200):
    """Phase 4: Run on MATH and GPQA datasets."""
    print("\n" + "="*70)
    print("PHASE 4: ADDITIONAL DATASETS")
    print("="*70)
    
    model_backend = load_model(model_name, backend=backend)
    model = LLMWrapper(model_backend, model_name)
    
    results = {}
    
    # MATH dataset
    if os.path.exists("data/math.json"):
        for seed in [42]:  # Just 1 seed for additional datasets
            result_cot = run_cot_baseline(
                model=model,
                dataset_path="data/math.json",
                output_path=f"results/baseline_cot_math_seed{seed}.json",
                seed=seed,
                limit=limit
            )
            results[f'baseline_cot_math_seed{seed}'] = result_cot['metrics']
            
            result_cdhr = run_cdhr_experiment(
                model=model,
                dataset_path="data/math.json",
                output_path=f"results/cdhr_main_math_seed{seed}.json",
                seed=seed,
                theta_v=0.05,
                theta_sigma=0.1,
                beta=0.5,
                limit=limit
            )
            results[f'cdhr_main_math_seed{seed}'] = result_cdhr['metrics']
    
    # GPQA dataset
    if os.path.exists("data/gpqa.json"):
        for seed in [42]:
            result_cot = run_cot_baseline(
                model=model,
                dataset_path="data/gpqa.json",
                output_path=f"results/baseline_cot_gpqa_seed{seed}.json",
                seed=seed,
                limit=min(limit, 100)  # GPQA is small
            )
            results[f'baseline_cot_gpqa_seed{seed}'] = result_cot['metrics']
            
            result_cdhr = run_cdhr_experiment(
                model=model,
                dataset_path="data/gpqa.json",
                output_path=f"results/cdhr_main_gpqa_seed{seed}.json",
                seed=seed,
                theta_v=0.05,
                theta_sigma=0.1,
                beta=0.5,
                limit=min(limit, 100)
            )
            results[f'cdhr_main_gpqa_seed{seed}'] = result_cdhr['metrics']
    
    # Save phase results
    with open('results/phase4_additional.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nPhase 4 Complete!")
    return results


def aggregate_results():
    """Aggregate all results into results.json."""
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    all_results = {}
    
    # Load all result files
    results_dir = Path("results")
    for result_file in results_dir.glob("*.json"):
        if result_file.name in ["summary.json", "results.json"]:
            continue
        try:
            with open(result_file) as f:
                data = json.load(f)
                exp_name = result_file.stem
                if 'metrics' in data:
                    all_results[exp_name] = data['metrics']
                elif isinstance(data, dict):
                    all_results[exp_name] = data
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    # Compute statistics for multi-seed experiments
    def compute_stats(key_prefix):
        values = []
        for seed in [42, 123, 456]:
            key = f"{key_prefix}_seed{seed}"
            if key in all_results and 'accuracy' in all_results[key]:
                values.append(all_results[key]['accuracy'])
        if len(values) >= 2:
            return {
                'mean': float(sum(values) / len(values)),
                'std': float((sum((x - sum(values)/len(values))**2 for x in values) / len(values)) ** 0.5),
                'values': values
            }
        return None
    
    # Main results summary
    summary = {
        'baseline_cot': compute_stats('baseline_cot'),
        'cdhr_main': compute_stats('cdhr_main'),
        'baseline_cot_math': compute_stats('baseline_cot_math'),
        'cdhr_main_math': compute_stats('cdhr_main_math'),
        'baseline_cot_gpqa': compute_stats('baseline_cot_gpqa'),
        'cdhr_main_gpqa': compute_stats('cdhr_main_gpqa'),
    }
    
    # Add single-run experiments
    for key in ['baseline_sc16', 'baseline_refinement', 'ablation_token_only', 'ablation_consistency_only']:
        if key in all_results:
            summary[key] = all_results[key]
    
    # Beta ablation
    summary['beta_sensitivity'] = {}
    for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        key = f'ablation_beta{beta}'
        if key in all_results:
            summary['beta_sensitivity'][beta] = all_results[key]
    
    # Threshold ablation
    summary['threshold_sensitivity'] = {}
    for theta_v in [0.03, 0.05, 0.07]:
        for theta_sigma in [0.075, 0.10, 0.125]:
            key = f'ablation_th{theta_v}_{theta_sigma}'
            if key in all_results:
                summary['threshold_sensitivity'][f'{theta_v}_{theta_sigma}'] = all_results[key]
    
    # Full results
    final_output = {
        'summary': summary,
        'all_experiments': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('results.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print("\nResults aggregated to results.json")
    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    
    return final_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=0, help='Run specific phase (0=all)')
    parser.add_argument('--model', type=str, default='llama-3.1-8b')
    parser.add_argument('--backend', type=str, default='transformers')
    parser.add_argument('--dataset', type=str, default='data/gsm8k.json')
    parser.add_argument('--limit', type=int, default=300, help='Limit problems per dataset')
    args = parser.parse_args()
    
    print("="*70)
    print(f"CDHR Structured Experiments")
    print(f"Model: {args.model}, Backend: {args.backend}")
    print("="*70)
    
    start_time = time.time()
    
    if args.phase == 0 or args.phase == 1:
        run_phase1_baselines(args.model, args.backend, args.dataset, args.limit)
    
    if args.phase == 0 or args.phase == 2:
        run_phase2_cdhr_main(args.model, args.backend, args.dataset, args.limit)
    
    if args.phase == 0 or args.phase == 3:
        run_phase3_ablations(args.model, args.backend, args.dataset, min(args.limit, 200))
    
    if args.phase == 0 or args.phase == 4:
        run_phase4_additional_datasets(args.model, args.backend, min(args.limit, 200))
    
    if args.phase == 0 or args.phase == 5:
        aggregate_results()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE - Time: {elapsed/3600:.2f} hours")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
