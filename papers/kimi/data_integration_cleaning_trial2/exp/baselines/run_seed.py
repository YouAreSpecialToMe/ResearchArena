"""
Run SEED-style baseline experiments.
"""
import sys
sys.path.insert(0, '.')

import json
import argparse
import numpy as np

from src.data_loader import (
    load_hospital_dataset, 
    load_flights_dataset, 
    load_beers_dataset
)
from baselines.seed_baseline import evaluate_seed_baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['hospital', 'flights', 'beers', 'all'])
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    datasets = ['hospital', 'flights', 'beers'] if args.dataset == 'all' else [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"SEED baseline on {dataset}")
        print(f"{'='*60}")
        
        # Load dataset
        if dataset == 'hospital':
            dirty_df, clean_df = load_hospital_dataset()
        elif dataset == 'flights':
            dirty_df, clean_df = load_flights_dataset()
        elif dataset == 'beers':
            dirty_df, clean_df = load_beers_dataset()
        
        print(f"Dataset: {len(dirty_df)} rows, {len(dirty_df.columns)} columns")
        
        # Run evaluation (with profiling for fair comparison)
        metrics = evaluate_seed_baseline(dirty_df, clean_df, verbose=True, use_profiling=True)
        
        print(f"\nResults:")
        print(f"  Precision:  {metrics['overall']['precision']:.3f}")
        print(f"  Recall:     {metrics['overall']['recall']:.3f}")
        print(f"  F1:         {metrics['overall']['f1']:.3f}")
        print(f"  LLM calls:  {metrics['llm_calls']}")
        
        results = {
            'experiment': 'seed_baseline',
            'dataset': dataset,
            'metrics': metrics,
        }
        
        output_path = args.output or f"results/seed/{dataset}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")
        
        all_results.append(results)
    
    return all_results


if __name__ == '__main__':
    main()
