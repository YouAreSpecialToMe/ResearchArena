#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for EVOLVE experiments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from exp.shared.data_loader import (
    download_mmlu, download_gsm8k, save_questions, 
    split_dataset, compute_dataset_statistics
)
import json


def main():
    print("=" * 60)
    print("Downloading and Preparing Datasets")
    print("=" * 60)
    
    # Download MMLU
    print("\n--- MMLU Dataset ---")
    mmlu_questions = download_mmlu(sample_size=2000, seed=42)
    mmlu_dev, mmlu_test = split_dataset(mmlu_questions, dev_ratio=0.25, seed=42)
    
    save_questions(mmlu_questions, 'data/mmlu_all.json')
    save_questions(mmlu_dev, 'data/mmlu_dev.json')
    save_questions(mmlu_test, 'data/mmlu_test.json')
    
    mmlu_stats = compute_dataset_statistics(mmlu_questions)
    print(f"MMLU statistics: {json.dumps(mmlu_stats, indent=2)}")
    
    # Download GSM8K
    print("\n--- GSM8K Dataset ---")
    gsm8k_questions = download_gsm8k(sample_size=1000, seed=42)
    gsm8k_dev, gsm8k_test = split_dataset(gsm8k_questions, dev_ratio=0.3, seed=42)
    
    save_questions(gsm8k_questions, 'data/gsm8k_all.json')
    save_questions(gsm8k_dev, 'data/gsm8k_dev.json')
    save_questions(gsm8k_test, 'data/gsm8k_test.json')
    
    gsm8k_stats = compute_dataset_statistics(gsm8k_questions)
    print(f"GSM8K statistics: {json.dumps(gsm8k_stats, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    
    # Save metadata
    metadata = {
        'mmlu': {
            'total': len(mmlu_questions),
            'dev': len(mmlu_dev),
            'test': len(mmlu_test),
            'statistics': mmlu_stats
        },
        'gsm8k': {
            'total': len(gsm8k_questions),
            'dev': len(gsm8k_dev),
            'test': len(gsm8k_test),
            'statistics': gsm8k_stats
        }
    }
    
    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
