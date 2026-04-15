"""Experiment 2: Controlled stage degradation.
Degrade each stage independently while keeping others at best configuration.
"""
import os
import sys
import json
import time
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASETS, RANDOM_SEEDS, DEGRADATION_LEVELS, RESULTS_DIR, DATASET_KEY_ATTRS
from src.pipeline.pipeline import load_dataset_for_pipeline, compute_e2e_metrics
from src.blocking.blocker import run_blocking
from src.matching.matcher import run_matching
from src.clustering.clusterer import run_clustering, compute_cluster_metrics
from src.degradation.degrade import degrade_blocking, degrade_matching, degrade_clustering

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'exp2.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_best_config(dataset):
    """Get best pipeline configuration from Experiment 1."""
    exp1_dir = os.path.join(RESULTS_DIR, 'exp1')
    with open(os.path.join(exp1_dir, 'all_results.json')) as f:
        all_results = json.load(f)

    ds_results = [r for r in all_results if r.get('dataset') == dataset and 'error' not in r and r.get('e2e_f1', 0) > 0]
    if not ds_results:
        return None

    configs = {}
    for r in ds_results:
        key = (r['blocking_method'], r['matching_method'], r['clustering_method'])
        if key not in configs:
            configs[key] = []
        configs[key].append(r['e2e_f1'])

    best_key = max(configs, key=lambda k: np.mean(configs[k]))
    return {
        'blocking_method': best_key[0],
        'matching_method': best_key[1],
        'clustering_method': best_key[2],
        'mean_e2e_f1': float(np.mean(configs[best_key])),
    }


def run_optimal_pipeline_cached(dataset_name, best_config, seed):
    """Run the optimal pipeline and cache intermediate results."""
    tableA, tableB, ground_truth, splits, emb_A, emb_B = load_dataset_for_pipeline(dataset_name)
    key_attrs = DATASET_KEY_ATTRS[dataset_name]

    candidates, pc, rr = run_blocking(best_config['blocking_method'], tableA, tableB, key_attrs, ground_truth)

    predicted_matches, mp, mr, mf1 = run_matching(
        best_config['matching_method'], candidates, tableA, tableB, ground_truth,
        splits=splits, embeddings_A=emb_A, embeddings_B=emb_B, seed=seed
    )

    clusters, cp, cr, cf1 = run_clustering(best_config['clustering_method'], predicted_matches, ground_truth)
    e2e_p, e2e_r, e2e_f1 = compute_e2e_metrics(clusters, ground_truth)

    return {
        'candidates': candidates,
        'predicted_matches': predicted_matches,
        'clusters': clusters,
        'ground_truth': ground_truth,
        'tableA': tableA, 'tableB': tableB, 'key_attrs': key_attrs,
        'splits': splits, 'emb_A': emb_A, 'emb_B': emb_B,
        'metrics': {
            'blocking_pc': pc, 'blocking_rr': rr,
            'matching_precision': mp, 'matching_recall': mr, 'matching_f1': mf1,
            'cluster_precision': cp, 'cluster_recall': cr, 'cluster_f1': cf1,
            'e2e_precision': e2e_p, 'e2e_recall': e2e_r, 'e2e_f1': e2e_f1,
        }
    }


def main():
    results_dir = os.path.join(RESULTS_DIR, 'exp2')
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for dataset in DATASETS:
        logger.info(f"\n=== {dataset} ===")
        best_config = get_best_config(dataset)
        if best_config is None:
            logger.warning(f"  No results from exp1 for {dataset}, skipping")
            continue

        logger.info(f"  Best config: {best_config['blocking_method']}/{best_config['matching_method']}/{best_config['clustering_method']} (F1={best_config['mean_e2e_f1']:.4f})")

        for seed in RANDOM_SEEDS:
            logger.info(f"  Seed {seed}:")
            cache = run_optimal_pipeline_cached(dataset, best_config, seed)
            base_metrics = cache['metrics']
            logger.info(f"    Baseline E2E F1: {base_metrics['e2e_f1']:.4f}")

            all_results.append({
                'dataset': dataset, 'seed': seed,
                'stage': 'none', 'degradation_level': 0.0,
                'mode': 'baseline',
                **{f'degraded_{k}': v for k, v in base_metrics.items()},
                'config': best_config,
            })

            for level in DEGRADATION_LEVELS:
                # === Blocking degradation ===
                deg_candidates = degrade_blocking(cache['candidates'], cache['ground_truth'], level, seed)
                deg_matches, mp, mr, mf1 = run_matching(
                    best_config['matching_method'], deg_candidates,
                    cache['tableA'], cache['tableB'], cache['ground_truth'],
                    splits=cache['splits'], embeddings_A=cache['emb_A'], embeddings_B=cache['emb_B'], seed=seed
                )
                if deg_matches:
                    deg_clusters, cp, cr, cf1 = run_clustering(best_config['clustering_method'], deg_matches, cache['ground_truth'])
                    e2e_p, e2e_r, e2e_f1 = compute_e2e_metrics(deg_clusters, cache['ground_truth'])
                else:
                    cp, cr, cf1 = 0, 0, 0
                    e2e_p, e2e_r, e2e_f1 = 0, 0, 0

                pc_deg = len(deg_candidates & cache['ground_truth']) / len(cache['ground_truth']) if cache['ground_truth'] else 0
                all_results.append({
                    'dataset': dataset, 'seed': seed,
                    'stage': 'blocking', 'degradation_level': level,
                    'mode': 'both',
                    'degraded_blocking_pc': pc_deg,
                    'degraded_matching_precision': mp, 'degraded_matching_recall': mr,
                    'degraded_matching_f1': mf1,
                    'degraded_cluster_precision': cp, 'degraded_cluster_recall': cr,
                    'degraded_cluster_f1': cf1,
                    'degraded_e2e_f1': e2e_f1,
                    'degraded_e2e_precision': e2e_p,
                    'degraded_e2e_recall': e2e_r,
                })

                # === Matching degradation ===
                for mode in ['both', 'fn_only', 'fp_only']:
                    deg_matches_m = degrade_matching(
                        cache['predicted_matches'], cache['candidates'],
                        cache['ground_truth'], level, mode=mode, seed=seed
                    )
                    if deg_matches_m:
                        deg_clusters_m, cp_m, cr_m, cf1_m = run_clustering(best_config['clustering_method'], deg_matches_m, cache['ground_truth'])
                        e2e_p_m, e2e_r_m, e2e_f1_m = compute_e2e_metrics(deg_clusters_m, cache['ground_truth'])
                        # Compute matching metrics for degraded predictions
                        gt_in_cand = cache['ground_truth'] & cache['candidates']
                        tp_m = deg_matches_m & cache['ground_truth']
                        mp_m = len(tp_m) / len(deg_matches_m) if deg_matches_m else 0
                        mr_m = len(tp_m) / len(gt_in_cand) if gt_in_cand else 0
                        mf1_m = 2 * mp_m * mr_m / (mp_m + mr_m) if (mp_m + mr_m) > 0 else 0
                    else:
                        cp_m, cr_m, cf1_m = 0, 0, 0
                        e2e_p_m, e2e_r_m, e2e_f1_m = 0, 0, 0
                        mp_m, mr_m, mf1_m = 0, 0, 0

                    all_results.append({
                        'dataset': dataset, 'seed': seed,
                        'stage': 'matching', 'degradation_level': level,
                        'mode': mode,
                        'degraded_blocking_pc': base_metrics['blocking_pc'],
                        'degraded_matching_precision': mp_m, 'degraded_matching_recall': mr_m,
                        'degraded_matching_f1': mf1_m,
                        'degraded_cluster_precision': cp_m, 'degraded_cluster_recall': cr_m,
                        'degraded_cluster_f1': cf1_m,
                        'degraded_e2e_f1': e2e_f1_m,
                        'degraded_e2e_precision': e2e_p_m,
                        'degraded_e2e_recall': e2e_r_m,
                    })

                # === Clustering degradation ===
                for mode in ['split', 'merge']:
                    deg_clusters_c = degrade_clustering(cache['clusters'], cache['ground_truth'], level, mode=mode, seed=seed)
                    cp_c, cr_c, cf1_c = compute_cluster_metrics(deg_clusters_c, cache['ground_truth'])
                    e2e_p_c, e2e_r_c, e2e_f1_c = compute_e2e_metrics(deg_clusters_c, cache['ground_truth'])

                    all_results.append({
                        'dataset': dataset, 'seed': seed,
                        'stage': 'clustering', 'degradation_level': level,
                        'mode': mode,
                        'degraded_blocking_pc': base_metrics['blocking_pc'],
                        'degraded_matching_precision': base_metrics['matching_precision'],
                        'degraded_matching_recall': base_metrics['matching_recall'],
                        'degraded_matching_f1': base_metrics['matching_f1'],
                        'degraded_cluster_precision': cp_c, 'degraded_cluster_recall': cr_c,
                        'degraded_cluster_f1': cf1_c,
                        'degraded_e2e_f1': e2e_f1_c,
                        'degraded_e2e_precision': e2e_p_c,
                        'degraded_e2e_recall': e2e_r_c,
                    })

            logger.info(f"    Completed all degradation levels")

    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nTotal degradation results: {len(all_results)}")


if __name__ == '__main__':
    main()
