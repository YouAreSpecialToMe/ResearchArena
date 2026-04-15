"""Experiment 1: Baseline per-stage quality measurement.
Run all pipeline configurations on all datasets with 3 seeds.
"""
import os
import sys
import json
import time
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import (DATASETS, BLOCKING_METHODS, MATCHING_METHODS,
                         CLUSTERING_METHODS, RANDOM_SEEDS, RESULTS_DIR)
from src.pipeline.pipeline import run_pipeline

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'exp1.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_single_config(args):
    """Run a single pipeline configuration."""
    dataset, blocking, matching, clustering, seed = args
    try:
        start = time.time()
        result = run_pipeline(dataset, blocking, matching, clustering, seed=seed)
        result['runtime_seconds'] = time.time() - start
        return result
    except Exception as e:
        return {
            'dataset': dataset,
            'blocking_method': blocking,
            'matching_method': matching,
            'clustering_method': clustering,
            'seed': seed,
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def main():
    results_dir = os.path.join(RESULTS_DIR, 'exp1')
    os.makedirs(results_dir, exist_ok=True)

    configs = []
    for dataset in DATASETS:
        for blocking in BLOCKING_METHODS:
            for matching in MATCHING_METHODS:
                for clustering in CLUSTERING_METHODS:
                    for seed in RANDOM_SEEDS:
                        configs.append((dataset, blocking, matching, clustering, seed))

    logger.info(f"Total configurations: {len(configs)}")

    all_results = []
    total_start = time.time()

    for dataset in DATASETS:
        ds_configs = [c for c in configs if c[0] == dataset]
        logger.info(f"\n=== {dataset} ({len(ds_configs)} configs) ===")
        ds_start = time.time()

        for i, cfg in enumerate(ds_configs):
            result = run_single_config(cfg)
            all_results.append(result)

            if 'error' in result:
                logger.warning(f"  [{i+1}/{len(ds_configs)}] {cfg[1]}/{cfg[2]}/{cfg[3]}/s{cfg[4]} - ERROR: {result['error'][:80]}")
            else:
                logger.info(f"  [{i+1}/{len(ds_configs)}] {cfg[1]}/{cfg[2]}/{cfg[3]}/s{cfg[4]} - "
                      f"PC={result['blocking_pc']:.3f} MF1={result['matching_f1']:.3f} "
                      f"CF1={result['cluster_f1']:.3f} E2E={result['e2e_f1']:.3f} "
                      f"({result.get('runtime_seconds', 0):.1f}s)")

            fname = f"{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}_{cfg[4]}.json"
            with open(os.path.join(results_dir, fname), 'w') as f:
                json.dump(result, f, indent=2)

        ds_time = time.time() - ds_start
        logger.info(f"  Dataset total: {ds_time:.1f}s")

    # Save all results
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Compute summary statistics
    import pandas as pd
    df = pd.DataFrame([r for r in all_results if 'error' not in r])
    if len(df) > 0:
        summary = df.groupby(['dataset', 'blocking_method', 'matching_method', 'clustering_method']).agg({
            'blocking_pc': ['mean', 'std'],
            'matching_f1': ['mean', 'std'],
            'cluster_f1': ['mean', 'std'],
            'e2e_f1': ['mean', 'std'],
            'e2e_precision': ['mean', 'std'],
            'e2e_recall': ['mean', 'std'],
        }).round(4)
        summary.columns = ['_'.join(col) for col in summary.columns]
        summary.to_csv(os.path.join(results_dir, 'exp1_summary.csv'))

        # Best config per dataset
        best_per_ds = {}
        for ds in DATASETS:
            ds_df = df[df['dataset'] == ds]
            if len(ds_df) == 0:
                continue
            mean_f1 = ds_df.groupby(['blocking_method', 'matching_method', 'clustering_method'])['e2e_f1'].mean()
            best_key = mean_f1.idxmax()
            best_per_ds[ds] = {
                'config': f"{best_key[0]}/{best_key[1]}/{best_key[2]}",
                'mean_e2e_f1': float(mean_f1.max()),
            }
            logger.info(f"  Best {ds}: {best_key[0]}/{best_key[1]}/{best_key[2]} -> E2E F1={mean_f1.max():.4f}")

        # Verify hard recall bound
        violations = df[df['e2e_recall'] > df['blocking_pc'] + 1e-6]
        logger.info(f"\nRecall bound violations (e2e_recall > PC): {len(violations)}/{len(df)}")

        # Correlation matrix
        import numpy as np
        metric_cols = ['blocking_pc', 'matching_f1', 'cluster_f1', 'e2e_f1']
        corr = df[metric_cols].corr()
        logger.info(f"\nCorrelation with e2e_f1:")
        for col in ['blocking_pc', 'matching_f1', 'cluster_f1']:
            logger.info(f"  {col}: {corr.loc[col, 'e2e_f1']:.4f}")

    total_time = time.time() - total_start
    logger.info(f"\nTotal time: {total_time:.1f}s, Results: {len(all_results)}")


if __name__ == '__main__':
    main()
