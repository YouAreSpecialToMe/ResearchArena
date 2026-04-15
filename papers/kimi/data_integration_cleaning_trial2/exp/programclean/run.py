"""
Run ProgramClean experiments on all datasets.
"""
import sys
sys.path.insert(0, '.')

import json
import argparse
import time
import numpy as np
import pandas as pd

from src.data_loader import (
    load_hospital_dataset, 
    load_flights_dataset, 
    load_beers_dataset,
    create_novel_domain_data
)
from src.programclean import evaluate_programclean


def run_experiment(dataset_name, seed=42, output_path=None):
    """Run ProgramClean on a dataset."""
    print(f"\n{'='*60}")
    print(f"ProgramClean on {dataset_name} (seed={seed})")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    # Load dataset
    if dataset_name == 'hospital':
        dirty_df, clean_df = load_hospital_dataset()
    elif dataset_name == 'flights':
        dirty_df, clean_df = load_flights_dataset()
    elif dataset_name == 'beers':
        dirty_df, clean_df = load_beers_dataset()
    elif dataset_name == 'novel':
        dirty_df, labels = create_novel_domain_data()
        # For novel data, we need to create a synthetic "clean" version
        clean_df = dirty_df.copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Dataset: {len(dirty_df)} rows, {len(dirty_df.columns)} columns")
    
    # Run evaluation
    start_time = time.time()
    metrics = evaluate_programclean(dirty_df, clean_df, seed=seed)
    total_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Precision: {metrics['overall']['precision']:.3f}")
    print(f"  Recall:    {metrics['overall']['recall']:.3f}")
    print(f"  F1:        {metrics['overall']['f1']:.3f}")
    print(f"  Runtime:   {total_time:.2f}s")
    print(f"  LLM calls: {metrics['stats']['llm_calls']}")
    
    # Save results
    results = {
        'experiment': 'programclean',
        'dataset': dataset_name,
        'seed': seed,
        'metrics': metrics,
        'total_time': total_time,
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['hospital', 'flights', 'beers', 'novel', 'all'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    
    args = parser.parse_args()
    
    datasets = ['hospital', 'flights', 'beers'] if args.dataset == 'all' else [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        for seed in args.seeds:
            output_path = args.output or f"results/programclean/{dataset}_seed{seed}.json"
            result = run_experiment(dataset, seed=seed, output_path=output_path)
            all_results.append(result)
    
    # Aggregate across seeds
    if len(args.seeds) > 1:
        print(f"\n{'='*60}")
        print("Aggregate Results (mean ± std)")
        print(f"{'='*60}")
        
        for dataset in datasets:
            dataset_results = [r for r in all_results if r['dataset'] == dataset]
            f1s = [r['metrics']['overall']['f1'] for r in dataset_results]
            precs = [r['metrics']['overall']['precision'] for r in dataset_results]
            recs = [r['metrics']['overall']['recall'] for r in dataset_results]
            
            print(f"{dataset}:")
            print(f"  F1:        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
            print(f"  Precision: {np.mean(precs):.3f} ± {np.std(precs):.3f}")
            print(f"  Recall:    {np.mean(recs):.3f} ± {np.std(recs):.3f}")


if __name__ == '__main__':
    main()
