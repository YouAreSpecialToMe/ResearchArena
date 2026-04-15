#!/usr/bin/env python3
"""Run pipeline evaluation with RandomForest as a second downstream model.

Evaluates exhaustive search, IAPO, and random search with RandomForest (fast=False)
on all 18 datasets with seed=42 only, to validate that interaction patterns
generalize across downstream models.
"""
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_loader import load_dataset, load_dataset_features, DATASET_SPECS
from src.operators import get_all_operators
from src.interaction import apply_pipeline
from src.evaluation import evaluate_quality
from src.baselines import exhaustive_search, random_search, greedy_forward, canonical_order
from src.iapo import iapo_optimize
from src.config import RESULTS_DIR

SEED = 42
DATASET_NAMES = [spec[0] for spec in DATASET_SPECS]


def run_randomforest_experiment():
    """Run all methods with RandomForest evaluation."""
    out_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    results = {
        'exhaustive': [],
        'random': [],
        'iapo': [],
        'greedy': [],
        'canonical': [],
    }

    # Load interaction data and rules for IAPO
    interactions_path = os.path.join(RESULTS_DIR, 'interaction_profiles', 'aggregated_interactions.csv')
    rules_path = os.path.join(RESULTS_DIR, 'interaction_profiles', 'interaction_rules.json')
    features_path = os.path.join(RESULTS_DIR, '..', 'data', 'profiles', 'dataset_features.json')

    all_interactions = pd.read_csv(interactions_path) if os.path.exists(interactions_path) else pd.DataFrame()
    rules = json.load(open(rules_path)) if os.path.exists(rules_path) else []
    all_features = json.load(open(features_path)) if os.path.exists(features_path) else {}

    for ds_name in DATASET_NAMES:
        print(f"Processing {ds_name} with RandomForest (fast=False)...")
        try:
            # Exhaustive search (120 perms for 5 operators)
            r = exhaustive_search(ds_name, seed=SEED, fast=False)
            r['dataset'] = ds_name
            results['exhaustive'].append(r)
            print(f"  Exhaustive: F1={r['quality']:.4f} ({r['n_evaluations']} evals)")

            # Random search (50 samples)
            r = random_search(ds_name, n_samples=50, seed=SEED, fast=False)
            r['dataset'] = ds_name
            results['random'].append(r)
            print(f"  Random:     F1={r['quality']:.4f}")

            # Greedy
            r = greedy_forward(ds_name, seed=SEED, fast=False)
            r['dataset'] = ds_name
            results['greedy'].append(r)
            print(f"  Greedy:     F1={r['quality']:.4f}")

            # Canonical
            r = canonical_order(ds_name, seed=SEED, fast=False)
            r['dataset'] = ds_name
            results['canonical'].append(r)
            print(f"  Canonical:  F1={r['quality']:.4f}")

            # IAPO (leave-one-out)
            if ds_name in all_features and len(all_interactions) > 0:
                train_interactions = all_interactions[all_interactions['dataset'] != ds_name]
                train_features = {k: v for k, v in all_features.items() if k != ds_name}
                r = iapo_optimize(
                    ds_name, all_features[ds_name],
                    train_interactions, train_features,
                    rules, K=10, seed=SEED, fast=False
                )
                r['dataset'] = ds_name
                results['iapo'].append(r)
                print(f"  IAPO:       F1={r['quality']:.4f}")

        except Exception as e:
            print(f"  Error on {ds_name}: {e}")

    # Compute summary
    summary = {}
    for method, res_list in results.items():
        if res_list:
            qualities = [r['quality'] for r in res_list]
            summary[method] = {
                'mean_f1': float(np.mean(qualities)),
                'std_f1': float(np.std(qualities)),
                'n_datasets': len(qualities),
            }
            print(f"\n{method}: mean_f1={np.mean(qualities):.4f} +/- {np.std(qualities):.4f}")

    # Compute quality ratios
    if 'exhaustive' in summary:
        exh_mean = summary['exhaustive']['mean_f1']
        for method in summary:
            summary[method]['quality_ratio'] = summary[method]['mean_f1'] / exh_mean if exh_mean > 0 else 0

    # Save results
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump({
            'experiment': 'randomforest_evaluation',
            'downstream_model': 'RandomForestClassifier(n_estimators=100, max_depth=10)',
            'seed': SEED,
            'summary': summary,
            'per_dataset': {
                method: {r['dataset']: r['quality'] for r in res_list}
                for method, res_list in results.items()
            }
        }, f, indent=2)

    print("\nResults saved to", os.path.join(out_dir, 'results.json'))
    return summary


if __name__ == '__main__':
    run_randomforest_experiment()
