"""Ablation study: EPM variants and transferability analysis."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASETS, RESULTS_DIR
from src.epm.propagation_model import (
    ErrorPropagationModel, LinearEPM, MultiplicativeEPM,
    NoTransitiveEPM, NoTopologyEPM
)


def load_exp1_data():
    with open(os.path.join(RESULTS_DIR, 'exp1', 'all_results.json')) as f:
        raw = json.load(f)
    data = []
    for r in raw:
        if 'error' in r or r.get('e2e_f1', 0) == 0:
            continue
        data.append({
            'dataset': r['dataset'],
            'blocking_pc': r['blocking_pc'],
            'matching_recall': r['matching_recall'],
            'matching_precision': r['matching_precision'],
            'matching_f1': r['matching_f1'],
            'cluster_recall': r['cluster_recall'],
            'cluster_precision': r['cluster_precision'],
            'cluster_f1': r['cluster_f1'],
            'e2e_f1': r['e2e_f1'],
        })
    return data


def main():
    results_dir = os.path.join(RESULTS_DIR, 'ablation')
    os.makedirs(results_dir, exist_ok=True)

    data = load_exp1_data()
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(data))
    split_idx = int(0.7 * len(data))
    train_data = [data[i] for i in indices[:split_idx]]
    val_data = [data[i] for i in indices[split_idx:]]

    print(f"Data: {len(data)} total, {len(train_data)} train, {len(val_data)} val")

    # === EPM Ablation ===
    ablation_results = {}

    # Full EPM
    epm_full = ErrorPropagationModel()
    epm_full.fit(train_data)
    full_eval = epm_full.evaluate(val_data)
    ablation_results['full_epm'] = full_eval
    print(f"\nFull EPM: R2={full_eval['r2']:.4f}, RMSE={full_eval['rmse']:.4f}")

    # Linear EPM
    epm_linear = LinearEPM()
    epm_linear.fit(train_data)
    linear_eval = epm_linear.evaluate(val_data)
    ablation_results['linear'] = linear_eval
    print(f"Linear:   R2={linear_eval['r2']:.4f}, RMSE={linear_eval['rmse']:.4f}")

    # Multiplicative EPM
    epm_mult = MultiplicativeEPM()
    epm_mult.fit(train_data)
    mult_eval = epm_mult.evaluate(val_data)
    ablation_results['multiplicative'] = mult_eval
    print(f"Multiplicative: R2={mult_eval['r2']:.4f}, RMSE={mult_eval['rmse']:.4f}")

    # No-transitive EPM
    epm_nt = NoTransitiveEPM()
    epm_nt.fit(train_data)
    nt_eval = epm_nt.evaluate(val_data)
    ablation_results['no_transitive'] = nt_eval
    print(f"No-transitive:  R2={nt_eval['r2']:.4f}, RMSE={nt_eval['rmse']:.4f}")

    # No-topology EPM
    epm_ntop = NoTopologyEPM()
    epm_ntop.fit(train_data)
    ntop_eval = epm_ntop.evaluate(val_data)
    ablation_results['no_topology'] = ntop_eval
    print(f"No-topology:    R2={ntop_eval['r2']:.4f}, RMSE={ntop_eval['rmse']:.4f}")

    # Per-dataset ablation
    per_dataset_ablation = {}
    for ds in DATASETS:
        ds_val = [d for d in val_data if d['dataset'] == ds]
        if len(ds_val) < 2:
            continue

        per_dataset_ablation[ds] = {
            'full_epm': epm_full.evaluate(ds_val),
            'linear': epm_linear.evaluate(ds_val),
            'multiplicative': epm_mult.evaluate(ds_val),
            'no_transitive': epm_nt.evaluate(ds_val),
            'no_topology': epm_ntop.evaluate(ds_val),
        }

    # === Transferability Analysis (Leave-One-Dataset-Out) ===
    print("\n=== Transferability (LODO) ===")
    transferability = {}
    for held_out in DATASETS:
        train_cv = [d for d in data if d['dataset'] != held_out]
        test_cv = [d for d in data if d['dataset'] == held_out]
        if not test_cv:
            continue

        # Fit full EPM on remaining datasets
        epm_cv = ErrorPropagationModel()
        epm_cv.fit(train_cv)
        cv_eval = epm_cv.evaluate(test_cv)

        # Within-dataset evaluation (using global model)
        within_eval = epm_full.evaluate(test_cv)

        r2_drop = within_eval['r2'] - cv_eval['r2']
        transferability[held_out] = {
            'lodo_r2': cv_eval['r2'],
            'lodo_rmse': cv_eval['rmse'],
            'within_r2': within_eval['r2'],
            'r2_drop': r2_drop,
        }
        print(f"  {held_out}: LODO R2={cv_eval['r2']:.4f}, within R2={within_eval['r2']:.4f}, drop={r2_drop:.4f}")

    avg_r2_drop = np.mean([v['r2_drop'] for v in transferability.values()])
    print(f"\nAverage R2 drop: {avg_r2_drop:.4f}")

    # Save all results
    results = {
        'ablation': {k: {kk: vv for kk, vv in v.items() if kk not in ('predicted_f1', 'actual_f1')}
                     for k, v in ablation_results.items()},
        'per_dataset_ablation': {ds: {k: {kk: vv for kk, vv in v.items() if kk not in ('predicted_f1', 'actual_f1')}
                                       for k, v in ds_data.items()}
                                  for ds, ds_data in per_dataset_ablation.items()},
        'transferability': transferability,
        'avg_r2_drop': float(avg_r2_drop),
    }

    with open(os.path.join(results_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
