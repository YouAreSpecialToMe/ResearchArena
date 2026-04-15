"""Experiment 3: Error Propagation Model (EPM) validation.
Fit EPM on 70% of Exp1 data, validate on 30% + all Exp2 data.
Cross-validate with leave-one-dataset-out.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATASETS, RESULTS_DIR
from src.epm.propagation_model import ErrorPropagationModel


def load_exp1_data():
    """Load all Experiment 1 results as EPM-compatible dicts."""
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
            'e2e_recall': r['e2e_recall'],
            'e2e_precision': r['e2e_precision'],
        })
    return data


def main():
    results_dir = os.path.join(RESULTS_DIR, 'exp3')
    os.makedirs(results_dir, exist_ok=True)

    data = load_exp1_data()
    print(f"Loaded {len(data)} valid configurations from Experiment 1")

    # Split into 70% train, 30% validation
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(data))
    split_idx = int(0.7 * len(data))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]

    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")

    # Fit EPM
    epm = ErrorPropagationModel()
    result = epm.fit(train_data)
    print(f"EPM fitted. Parameters: {dict(zip(epm.param_names, epm.params))}")

    # Evaluate on training data
    train_eval = epm.evaluate(train_data)
    print(f"\nTraining set: R2={train_eval['r2']:.4f}, RMSE={train_eval['rmse']:.4f}, MAE={train_eval['mae']:.4f}")

    # Evaluate on validation data
    val_eval = epm.evaluate(val_data)
    print(f"Validation set: R2={val_eval['r2']:.4f}, RMSE={val_eval['rmse']:.4f}, MAE={val_eval['mae']:.4f}")

    # Evaluate on Experiment 2 data (if available)
    exp2_eval = None
    exp2_path = os.path.join(RESULTS_DIR, 'exp2', 'all_results.json')
    if os.path.exists(exp2_path):
        with open(exp2_path) as f:
            exp2_raw = json.load(f)
        exp2_data = []
        for r in exp2_raw:
            if r.get('degraded_e2e_f1', 0) > 0 and r.get('degraded_blocking_pc') is not None:
                mr = r.get('degraded_matching_recall') or r.get('degraded_matching_f1', 0.5)
                mp = r.get('degraded_matching_precision') or r.get('degraded_matching_f1', 0.5)
                cr = r.get('degraded_cluster_recall') or r.get('degraded_cluster_f1', 0.5)
                cp = r.get('degraded_cluster_precision') or r.get('degraded_cluster_f1', 0.5)
                if mr is None or mp is None or cr is None or cp is None:
                    continue
                exp2_data.append({
                    'blocking_pc': r['degraded_blocking_pc'],
                    'matching_recall': mr,
                    'matching_precision': mp,
                    'matching_f1': r.get('degraded_matching_f1', 0.5),
                    'cluster_recall': cr,
                    'cluster_precision': cp,
                    'cluster_f1': r.get('degraded_cluster_f1', 0.5),
                    'e2e_f1': r['degraded_e2e_f1'],
                })
        if exp2_data:
            exp2_eval = epm.evaluate(exp2_data)
            print(f"Exp2 degradation data: R2={exp2_eval['r2']:.4f}, RMSE={exp2_eval['rmse']:.4f}")

    # Per-dataset evaluation
    per_dataset = {}
    for ds in DATASETS:
        ds_data = [d for d in data if d['dataset'] == ds]
        if len(ds_data) > 2:
            ds_eval = epm.evaluate(ds_data)
            per_dataset[ds] = ds_eval
            print(f"  {ds}: R2={ds_eval['r2']:.4f}, RMSE={ds_eval['rmse']:.4f}")

    # Leave-one-dataset-out cross-validation
    lodo_results = {}
    for held_out in DATASETS:
        train_cv = [d for d in data if d['dataset'] != held_out]
        test_cv = [d for d in data if d['dataset'] == held_out]
        if not train_cv or not test_cv:
            continue

        epm_cv = ErrorPropagationModel()
        epm_cv.fit(train_cv)
        cv_eval = epm_cv.evaluate(test_cv)
        lodo_results[held_out] = cv_eval
        print(f"  LODO {held_out}: R2={cv_eval['r2']:.4f}, RMSE={cv_eval['rmse']:.4f}")

    # Save all results
    results = {
        'train_eval': train_eval,
        'val_eval': val_eval,
        'exp2_eval': exp2_eval,
        'per_dataset': per_dataset,
        'lodo_cross_validation': lodo_results,
        'epm_params': dict(zip(epm.param_names, [float(p) for p in epm.params])),
        'n_train': len(train_data),
        'n_val': len(val_data),
    }

    with open(os.path.join(results_dir, 'epm_validation.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save predictions vs actuals for plotting
    all_eval = epm.evaluate(data)
    with open(os.path.join(results_dir, 'predictions_vs_actuals.json'), 'w') as f:
        json.dump({
            'predicted': all_eval['predicted_f1'],
            'actual': all_eval['actual_f1'],
            'datasets': [d['dataset'] for d in data],
            'is_train': [i in set(train_idx) for i in range(len(data))],
        }, f)

    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
