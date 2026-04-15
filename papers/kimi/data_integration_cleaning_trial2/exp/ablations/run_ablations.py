"""
Run ablation studies for ProgramClean.
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np

from src.data_loader import load_hospital_dataset, load_beers_dataset, load_flights_dataset
from src.programclean import evaluate_programclean
from baselines.seed_baseline import evaluate_seed_baseline


def run_ablation_naive_codegen(dataset_name='hospital'):
    """
    Ablation: Compare ProgramClean with naive code generation (no semantic profiling).
    This tests the value of the semantic profiling step.
    """
    print(f"\n{'='*60}")
    print(f"Ablation: Naive CodeGen vs ProgramClean on {dataset_name}")
    print(f"{'='*60}")
    
    if dataset_name == 'hospital':
        dirty_df, clean_df = load_hospital_dataset()
    elif dataset_name == 'beers':
        dirty_df, clean_df = load_beers_dataset()
    
    # ProgramClean (with semantic profiling)
    print("\n--- ProgramClean (with profiling) ---")
    pc_metrics = evaluate_programclean(dirty_df, clean_df, seed=42)
    print(f"F1: {pc_metrics['overall']['f1']:.3f}")
    
    # SEED baseline (naive codegen, no profiling)
    print("\n--- Naive CodeGen (no profiling) ---")
    seed_metrics = evaluate_seed_baseline(dirty_df, clean_df, verbose=False)
    print(f"F1: {seed_metrics['overall']['f1']:.3f}")
    
    results = {
        'dataset': dataset_name,
        'programclean': pc_metrics,
        'naive_codegen': seed_metrics,
        'improvement': pc_metrics['overall']['f1'] - seed_metrics['overall']['f1'],
    }
    
    output_path = f'results/ablations/naive_vs_profiling_{dataset_name}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
    
    return results


def analyze_program_validity():
    """
    Analyze program validity rates across datasets.
    """
    print(f"\n{'='*60}")
    print(f"Program Validity Analysis")
    print(f"{'='*60}")
    
    datasets = ['hospital', 'flights', 'beers']
    all_stats = []
    
    for dataset_name in datasets:
        if dataset_name == 'hospital':
            dirty_df, clean_df = load_hospital_dataset()
        elif dataset_name == 'flights':
            dirty_df, clean_df = load_flights_dataset()
        elif dataset_name == 'beers':
            dirty_df, clean_df = load_beers_dataset()
        
        from src.programclean import ProgramClean
        cleaner = ProgramClean(verbose=False)
        cleaner.fit(dirty_df)
        
        stats = cleaner.get_stats()
        all_stats.append({
            'dataset': dataset_name,
            'columns': stats['columns_processed'],
            'validity_rate': stats['program_validity_rate'],
            'synthesis_time': stats['synthesis_time'],
        })
        
        print(f"{dataset_name}: {stats['program_validity_rate']:.1%} validity, "
              f"{stats['columns_processed']} columns, "
              f"{stats['synthesis_time']:.3f}s synthesis")
    
    with open('results/ablations/program_validity.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    return all_stats


def compare_llm_calls():
    """
    Compare LLM call counts across methods.
    """
    print(f"\n{'='*60}")
    print(f"LLM Call Comparison")
    print(f"{'='*60}")
    
    dirty_df, clean_df = load_hospital_dataset()
    n_rows = len(dirty_df)
    n_cols = len(dirty_df.columns)
    
    # ProgramClean: O(columns)
    from src.programclean import ProgramClean
    pc = ProgramClean(verbose=False)
    pc.fit(dirty_df)
    pc_calls = pc.llm_calls
    
    # Direct Validation: O(cells) - limited for comparison
    from baselines.direct_validation import DirectValidationBaseline
    dv = DirectValidationBaseline(max_cells=200, verbose=False)
    dv.fit_predict(dirty_df)
    dv_calls = dv.llm_calls
    
    # SEED: O(columns)
    from baselines.seed_baseline import SEEDBaseline
    seed = SEEDBaseline(verbose=False)
    seed.fit(dirty_df)
    seed_calls = seed.llm_calls
    
    print(f"Dataset: {n_rows} rows, {n_cols} columns = {n_rows * n_cols} cells")
    print(f"  ProgramClean:  {pc_calls} LLM calls (O(columns))")
    print(f"  SEED Baseline: {seed_calls} LLM calls (O(columns))")
    print(f"  Direct Valid:  {dv_calls} LLM calls (O(cells), sampled)")
    print(f"  Reduction:     {dv_calls/pc_calls:.1f}x fewer calls than Direct Validation")
    
    results = {
        'dataset': 'hospital',
        'rows': n_rows,
        'columns': n_cols,
        'total_cells': n_rows * n_cols,
        'programclean_calls': pc_calls,
        'seed_calls': seed_calls,
        'direct_val_calls': dv_calls,
        'reduction_factor': dv_calls / pc_calls,
    }
    
    with open('results/ablations/llm_call_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    # Run ablations
    run_ablation_naive_codegen('hospital')
    run_ablation_naive_codegen('beers')
    analyze_program_validity()
    compare_llm_calls()
    
    print(f"\n{'='*60}")
    print("All ablation studies complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
