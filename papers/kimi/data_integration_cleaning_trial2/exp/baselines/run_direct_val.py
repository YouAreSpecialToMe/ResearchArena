"""
Run Direct Validation baseline experiments.
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
from baselines.direct_validation import evaluate_direct_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['hospital', 'flights', 'beers', 'all'])
    parser.add_argument('--max-cells', type=int, default=100)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    datasets = ['hospital', 'flights', 'beers'] if args.dataset == 'all' else [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Direct Validation baseline on {dataset}")
        print(f"{'='*60}")
        
        # Load dataset
        if dataset == 'hospital':
            dirty_df, clean_df = load_hospital_dataset()
        elif dataset == 'flights':
            dirty_df, clean_df = load_flights_dataset()
        elif dataset == 'beers':
            dirty_df, clean_df = load_beers_dataset()
        
        print(f"Dataset: {len(dirty_df)} rows, {len(dirty_df.columns)} columns")
        print(f"Max cells to validate: {args.max_cells}")
        
        # Run evaluation
        metrics = evaluate_direct_validation(dirty_df, clean_df, 
                                            max_cells=args.max_cells, 
                                            verbose=True)
        
        print(f"\nResults:")
        print(f"  Precision:  {metrics['overall']['precision']:.3f}")
        print(f"  Recall:     {metrics['overall']['recall']:.3f}")
        print(f"  F1:         {metrics['overall']['f1']:.3f}")
        print(f"  LLM calls:  {metrics['llm_calls']}")
        
        results = {
            'experiment': 'direct_validation',
            'dataset': dataset,
            'max_cells': args.max_cells,
            'metrics': metrics,
        }
        
        output_path = args.output or f"results/direct_val/{dataset}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {output_path}")
        
        all_results.append(results)
    
    return all_results


if __name__ == '__main__':
    main()
