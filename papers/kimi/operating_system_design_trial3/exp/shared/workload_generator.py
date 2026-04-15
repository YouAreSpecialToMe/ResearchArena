"""
Workload Signature Generator for KAPHE
Generates synthetic workload signatures representing different application types.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .kernel_simulator import WorkloadSignature


class WorkloadGenerator:
    """Generates diverse workload signatures for experiments."""
    
    # Workload category definitions with characteristic distributions
    CATEGORIES = {
        'in_mem_db': {
            'description': 'In-memory databases (Redis, Memcached)',
            'alloc_rate': (1000, 5000),          # High allocation rate
            'working_set_MB': (100, 800),         # Small working set
            'thread_count': (4, 32),
            'thread_churn_per_sec': (5, 20),      # Low churn
            'io_read_MBps': (10, 50),
            'io_write_MBps': (50, 200),
            'io_sequentiality_ratio': (0.3, 0.6),
            'syscall_rate_per_sec': (5000, 15000),
        },
        'analytics': {
            'description': 'Analytics workloads (TPC-H, ClickHouse)',
            'alloc_rate': (100, 500),             # Medium allocation
            'working_set_MB': (5000, 50000),      # Large working set
            'thread_count': (8, 64),
            'thread_churn_per_sec': (1, 10),      # Very low churn
            'io_read_MBps': (200, 1000),
            'io_write_MBps': (50, 300),
            'io_sequentiality_ratio': (0.7, 0.95), # High sequentiality
            'syscall_rate_per_sec': (1000, 5000),
        },
        'web_service': {
            'description': 'Web services (Nginx)',
            'alloc_rate': (200, 800),             # Medium allocation
            'working_set_MB': (500, 3000),        # Medium working set
            'thread_count': (16, 128),
            'thread_churn_per_sec': (50, 200),    # High churn (connections)
            'io_read_MBps': (20, 100),
            'io_write_MBps': (10, 50),
            'io_sequentiality_ratio': (0.4, 0.7),
            'syscall_rate_per_sec': (10000, 50000), # Very high syscall rate
        },
        'build_compile': {
            'description': 'Build/compile workloads',
            'alloc_rate': (50, 300),              # Lower allocation
            'working_set_MB': (1000, 8000),       # Medium-large working set
            'thread_count': (8, 32),
            'thread_churn_per_sec': (100, 500),   # Very high churn (processes)
            'io_read_MBps': (50, 300),
            'io_write_MBps': (20, 100),
            'io_sequentiality_ratio': (0.5, 0.8),
            'syscall_rate_per_sec': (2000, 10000),
        },
    }
    
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)
    
    def generate_workload(self, category: str) -> WorkloadSignature:
        """Generate a single workload signature for a category."""
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        
        params = self.CATEGORIES[category]
        
        return WorkloadSignature(
            alloc_rate=self._sample_log(params['alloc_rate']),
            working_set_MB=self._sample_log(params['working_set_MB']),
            thread_count=self.rng.randint(params['thread_count'][0], params['thread_count'][1] + 1),
            thread_churn_per_sec=self._sample_log(params['thread_churn_per_sec']),
            io_read_MBps=self._sample_log(params['io_read_MBps']),
            io_write_MBps=self._sample_log(params['io_write_MBps']),
            io_sequentiality_ratio=self.rng.uniform(*params['io_sequentiality_ratio']),
            syscall_rate_per_sec=self._sample_log(params['syscall_rate_per_sec']),
            category=category,
        )
    
    def _sample_log(self, range_tuple: tuple) -> float:
        """Sample from log-uniform distribution."""
        log_min = np.log10(range_tuple[0])
        log_max = np.log10(range_tuple[1])
        return 10 ** self.rng.uniform(log_min, log_max)
    
    def generate_dataset(self, n_per_category: int, seed: int = None) -> pd.DataFrame:
        """Generate a dataset with n workloads per category."""
        if seed is not None:
            old_rng = self.rng
            self.rng = np.random.RandomState(seed)
        
        workloads = []
        for category in self.CATEGORIES:
            for i in range(n_per_category):
                wl = self.generate_workload(category)
                row = wl.to_feature_vector()
                workloads.append({
                    'workload_id': f"{category}_{i:03d}",
                    'category': category,
                    'alloc_rate': wl.alloc_rate,
                    'working_set_MB': wl.working_set_MB,
                    'thread_count': wl.thread_count,
                    'thread_churn_per_sec': wl.thread_churn_per_sec,
                    'io_read_MBps': wl.io_read_MBps,
                    'io_write_MBps': wl.io_write_MBps,
                    'io_sequentiality_ratio': wl.io_sequentiality_ratio,
                    'syscall_rate_per_sec': wl.syscall_rate_per_sec,
                })
        
        if seed is not None:
            self.rng = old_rng
        
        return pd.DataFrame(workloads)
    
    def save_datasets(self, output_dir: str):
        """Generate and save train, test, and validation datasets."""
        # Training set: 240 workloads (60 per category)
        train_df = self.generate_dataset(n_per_category=60, seed=42)
        train_df.to_csv(f"{output_dir}/workloads_train.csv", index=False)
        
        # Test set: 48 workloads (12 per category) with different seeds
        test_dfs = []
        for seed in [123, 456, 789]:
            test_dfs.append(self.generate_dataset(n_per_category=4, seed=seed))
        test_df = pd.concat(test_dfs, ignore_index=True)
        test_df['workload_id'] = [f"test_{i:03d}" for i in range(len(test_df))]
        test_df.to_csv(f"{output_dir}/workloads_test.csv", index=False)
        
        # Real validation set: 16 workloads (4 per category)
        real_df = self.generate_dataset(n_per_category=4, seed=999)
        real_df['workload_id'] = [f"real_{i:03d}" for i in range(len(real_df))]
        real_df.to_csv(f"{output_dir}/workloads_real.csv", index=False)
        
        return train_df, test_df, real_df


def load_workload_from_row(row: pd.Series) -> WorkloadSignature:
    """Load a WorkloadSignature from a DataFrame row."""
    return WorkloadSignature(
        alloc_rate=row['alloc_rate'],
        working_set_MB=row['working_set_MB'],
        thread_count=int(row['thread_count']),
        thread_churn_per_sec=row['thread_churn_per_sec'],
        io_read_MBps=row['io_read_MBps'],
        io_write_MBps=row['io_write_MBps'],
        io_sequentiality_ratio=row['io_sequentiality_ratio'],
        syscall_rate_per_sec=row['syscall_rate_per_sec'],
        category=row['category'],
    )
