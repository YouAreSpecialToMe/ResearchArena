"""Generate results tables for the paper."""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATASETS, DATASET_DIFFICULTY, RESULTS_DIR, MATCHING_METHODS


def table1_best_configs():
    """Table 1: Best pipeline configuration per dataset."""
    with open(os.path.join(RESULTS_DIR, 'exp1', 'all_results.json')) as f:
        results = json.load(f)

    rows = []
    for dataset in DATASETS:
        ds = [r for r in results if r['dataset'] == dataset and 'error' not in r and r.get('e2e_f1', 0) > 0]
        if not ds:
            continue

        df = pd.DataFrame(ds)
        # Best config by mean e2e_f1
        mean_f1 = df.groupby(['blocking_method', 'matching_method', 'clustering_method'])['e2e_f1'].mean()
        best_key = mean_f1.idxmax()
        best_runs = df[(df['blocking_method'] == best_key[0]) &
                       (df['matching_method'] == best_key[1]) &
                       (df['clustering_method'] == best_key[2])]

        rows.append({
            'Dataset': dataset,
            'Difficulty': DATASET_DIFFICULTY[dataset],
            'Blocking': best_key[0],
            'Matching': best_key[1],
            'Clustering': best_key[2],
            'PC': f"{best_runs['blocking_pc'].mean():.3f}",
            'MF1': f"{best_runs['matching_f1'].mean():.3f}",
            'CF1': f"{best_runs['cluster_f1'].mean():.3f}",
            'E2E_F1': f"{best_runs['e2e_f1'].mean():.3f} +/- {best_runs['e2e_f1'].std():.3f}",
            'E2E_P': f"{best_runs['e2e_precision'].mean():.3f}",
            'E2E_R': f"{best_runs['e2e_recall'].mean():.3f}",
        })

    return pd.DataFrame(rows)


def table2_epm_validation():
    """Table 2: EPM validation results."""
    with open(os.path.join(RESULTS_DIR, 'exp3', 'epm_validation.json')) as f:
        epm = json.load(f)

    rows = [
        {'Split': 'Training', 'R2': f"{epm['train_eval']['r2']:.4f}",
         'RMSE': f"{epm['train_eval']['rmse']:.4f}", 'MAE': f"{epm['train_eval']['mae']:.4f}"},
        {'Split': 'Validation', 'R2': f"{epm['val_eval']['r2']:.4f}",
         'RMSE': f"{epm['val_eval']['rmse']:.4f}", 'MAE': f"{epm['val_eval']['mae']:.4f}"},
    ]
    if epm.get('exp2_eval'):
        rows.append({'Split': 'Exp2 (degradation)', 'R2': f"{epm['exp2_eval']['r2']:.4f}",
                     'RMSE': f"{epm['exp2_eval']['rmse']:.4f}", 'MAE': f"{epm['exp2_eval']['mae']:.4f}"})

    # Per-dataset
    for ds in DATASETS:
        if ds in epm.get('per_dataset', {}):
            d = epm['per_dataset'][ds]
            rows.append({'Split': f'Per-DS: {ds}', 'R2': f"{d['r2']:.4f}",
                         'RMSE': f"{d['rmse']:.4f}", 'MAE': f"{d['mae']:.4f}"})

    # LODO
    for ds in DATASETS:
        if ds in epm.get('lodo_cross_validation', {}):
            d = epm['lodo_cross_validation'][ds]
            rows.append({'Split': f'LODO: {ds}', 'R2': f"{d['r2']:.4f}",
                         'RMSE': f"{d['rmse']:.4f}", 'MAE': f"{d['mae']:.4f}"})

    return pd.DataFrame(rows)


def table3_eafs():
    """Table 3: EAFs per stage per dataset."""
    with open(os.path.join(RESULTS_DIR, 'exp4', 'eaf_analysis.json')) as f:
        eaf_data = json.load(f)

    rows = []
    for dataset in DATASETS:
        if dataset not in eaf_data:
            continue
        d = eaf_data[dataset]
        emp = d['empirical']['normalized']
        ana = d['analytical_normalized']
        comp = d['comparison']

        rows.append({
            'Dataset': dataset,
            'Difficulty': d['difficulty'],
            'Emp_Blocking': f"{emp['blocking']:.3f}",
            'Emp_Matching': f"{emp['matching']:.3f}",
            'Emp_Clustering': f"{emp['clustering']:.3f}",
            'Ana_Blocking': f"{ana['blocking']:.3f}",
            'Ana_Matching': f"{ana['matching']:.3f}",
            'Ana_Clustering': f"{ana['clustering']:.3f}",
            'Emp_Bottleneck': d['empirical_bottleneck'],
            'Ana_Bottleneck': d['analytical_bottleneck'],
            'Agreement': d['bottleneck_agreement'],
            'Avg_RelError': f"{np.mean([comp[s]['relative_error'] for s in comp]):.3f}",
        })

    return pd.DataFrame(rows)


def table4_soa():
    """Table 4: SOA vs baselines."""
    with open(os.path.join(RESULTS_DIR, 'exp5', 'soa_validation.json')) as f:
        results = json.load(f)

    rows = []
    for dataset in DATASETS:
        ds_results = [r for r in results if r['dataset'] == dataset]
        if not ds_results:
            continue
        for strategy in ['uniform', 'bottleneck', 'soa']:
            strat = [r for r in ds_results if r['strategy'] == strategy]
            if strat:
                rows.append({
                    'Dataset': dataset,
                    'Strategy': strategy,
                    'Avg_Efficiency': f"{np.mean([r['efficiency'] for r in strat]):.4f}",
                    'Avg_DeltaF1': f"{np.mean([r['delta_f1'] for r in strat]):.4f}",
                    'Std_DeltaF1': f"{np.std([r['delta_f1'] for r in strat]):.4f}",
                })

    return pd.DataFrame(rows)


def table5_ablation():
    """Table 5: Ablation study results."""
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        results = json.load(f)

    rows = []
    for variant, label in [
        ('full_epm', 'Full EPM'),
        ('no_transitive', 'No Transitive'),
        ('no_topology', 'No Topology'),
        ('linear', 'Linear'),
        ('multiplicative', 'Multiplicative'),
    ]:
        d = results['ablation'][variant]
        rows.append({
            'Model': label,
            'R2': f"{d['r2']:.4f}",
            'RMSE': f"{d['rmse']:.4f}",
            'MAE': f"{d['mae']:.4f}",
        })

    return pd.DataFrame(rows)


def table6_transferability():
    """Table 6: Transferability analysis."""
    with open(os.path.join(RESULTS_DIR, 'ablation', 'ablation_results.json')) as f:
        results = json.load(f)

    rows = []
    for ds in DATASETS:
        if ds in results.get('transferability', {}):
            d = results['transferability'][ds]
            rows.append({
                'Held_Out': ds,
                'Difficulty': DATASET_DIFFICULTY[ds],
                'LODO_R2': f"{d['lodo_r2']:.4f}",
                'Within_R2': f"{d['within_r2']:.4f}",
                'R2_Drop': f"{d['r2_drop']:.4f}",
            })

    return pd.DataFrame(rows)


def main():
    tables_dir = os.path.join(RESULTS_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    table_funcs = [
        ('table1_best_configs.csv', table1_best_configs),
        ('table2_epm_validation.csv', table2_epm_validation),
        ('table3_eafs.csv', table3_eafs),
        ('table4_soa.csv', table4_soa),
        ('table5_ablation.csv', table5_ablation),
        ('table6_transferability.csv', table6_transferability),
    ]

    for fname, func in table_funcs:
        try:
            df = func()
            path = os.path.join(tables_dir, fname)
            df.to_csv(path, index=False)
            print(f"  Saved {fname} ({len(df)} rows)")
        except Exception as e:
            print(f"  Error generating {fname}: {e}")


if __name__ == '__main__':
    main()
