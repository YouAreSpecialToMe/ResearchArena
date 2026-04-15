"""
Step 1: Generate synthetic workload signatures.
"""

import sys
sys.path.insert(0, '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp')

import os
import json
from shared.workload_generator import WorkloadGenerator

def main():
    print("=" * 60)
    print("STEP 1: Generating Synthetic Workload Signatures")
    print("=" * 60)
    
    output_dir = '/home/nw366/ResearchArena/outputs/kimi_t3_operating_system_design/idea_01/exp/data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = WorkloadGenerator(random_seed=42)
    
    # Generate and save datasets
    print("\nGenerating training set (240 workloads)...")
    train_df, test_df, real_df = generator.save_datasets(output_dir)
    
    print(f"  Training workloads: {len(train_df)}")
    print(f"  Test workloads: {len(test_df)}")
    print(f"  Real validation workloads: {len(real_df)}")
    
    # Compute and log statistics per category
    print("\nWorkload Statistics by Category:")
    print("-" * 60)
    stats = {}
    for category in train_df['category'].unique():
        cat_data = train_df[train_df['category'] == category]
        cat_stats = {
            'count': len(cat_data),
            'alloc_rate': {'mean': cat_data['alloc_rate'].mean(), 'std': cat_data['alloc_rate'].std()},
            'working_set_MB': {'mean': cat_data['working_set_MB'].mean(), 'std': cat_data['working_set_MB'].std()},
            'thread_churn_per_sec': {'mean': cat_data['thread_churn_per_sec'].mean(), 'std': cat_data['thread_churn_per_sec'].std()},
            'io_sequentiality_ratio': {'mean': cat_data['io_sequentiality_ratio'].mean(), 'std': cat_data['io_sequentiality_ratio'].std()},
            'syscall_rate_per_sec': {'mean': cat_data['syscall_rate_per_sec'].mean(), 'std': cat_data['syscall_rate_per_sec'].std()},
        }
        stats[category] = cat_stats
        print(f"\n{category}:")
        print(f"  Count: {cat_stats['count']}")
        print(f"  alloc_rate: {cat_stats['alloc_rate']['mean']:.1f} ± {cat_stats['alloc_rate']['std']:.1f}")
        print(f"  working_set_MB: {cat_stats['working_set_MB']['mean']:.1f} ± {cat_stats['working_set_MB']['std']:.1f}")
    
    # Save statistics
    with open(f'{output_dir}/workload_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Workload generation complete!")
    print(f"Data saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
