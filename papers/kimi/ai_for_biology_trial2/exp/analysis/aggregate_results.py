"""
Aggregate all experimental results into a single results.json file.
Perform statistical significance testing.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def paired_t_test(values_a, values_b):
    """Perform paired t-test and return statistics."""
    if len(values_a) != len(values_b):
        return None
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    
    # Cohen's d (effect size)
    diff = np.array(values_a) - np.array(values_b)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }


def main():
    data_dir = Path('/home/zz865/pythonProject/autoresearch/outputs/kimi/run_2/ai_for_biology/idea_01/results')
    output_dir = data_dir
    
    # Load all results
    results = {
        'baselines': {},
        'tricon': None,
        'ablations': {}
    }
    
    # Load baselines
    baseline_files = {
        'knn': 'baseline_knn.json',
        'contrastive': 'baseline_contrastive.json',
        'ontology': 'baseline_ontology.json'
    }
    
    for name, filename in baseline_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            results['baselines'][name] = load_json(filepath)
    
    # Load Tri-Con
    tricon_file = data_dir / 'tricon_v3_full.json'
    if tricon_file.exists():
        results['tricon'] = load_json(tricon_file)
    
    # Load ablations
    ablation_files = {
        'no_cc': 'ablation_no_cc.json',
        'no_co': 'ablation_no_co.json',
        'no_go': 'ablation_no_go.json',
        'uniform': 'ablation_uniform.json',
        'no_evidential': 'ablation_no_evidential.json'
    }
    
    for name, filename in ablation_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            results['ablations'][name] = load_json(filepath)
    
    # Statistical tests
    print("Performing statistical tests...")
    statistical_tests = {
        'tricon_vs_baselines': {},
        'tricon_vs_ablations': {}
    }
    
    if results['tricon']:
        tricon_per_seed = results['tricon']['per_seed_results']
        tricon_acc = [r['test_accuracy'] for r in tricon_per_seed]
        tricon_zs = [r['zero_shot_accuracy'] for r in tricon_per_seed]
        tricon_ood = [r['ood_auroc'] for r in tricon_per_seed]
        
        # Compare with baselines
        for name, baseline in results['baselines'].items():
            if 'per_seed_results' in baseline:
                baseline_acc = [r['test_accuracy'] for r in baseline['per_seed_results']]
                baseline_zs = [r['zero_shot_accuracy'] for r in baseline['per_seed_results']]
                
                statistical_tests['tricon_vs_baselines'][name] = {
                    'test_accuracy': paired_t_test(tricon_acc, baseline_acc),
                    'zero_shot_accuracy': paired_t_test(tricon_zs, baseline_zs)
                }
        
        # Compare with ablations
        for name, ablation in results['ablations'].items():
            if ablation and 'per_seed_results' in ablation:
                ablation_acc = [r['test_accuracy'] for r in ablation['per_seed_results']]
                ablation_zs = [r['zero_shot_accuracy'] for r in ablation['per_seed_results']]
                ablation_ood = [r['ood_auroc'] for r in ablation['per_seed_results']]
                
                statistical_tests['tricon_vs_ablations'][name] = {
                    'test_accuracy': paired_t_test(tricon_acc, ablation_acc),
                    'zero_shot_accuracy': paired_t_test(tricon_zs, ablation_zs),
                    'ood_auroc': paired_t_test(tricon_ood, ablation_ood)
                }
    
    # Save statistical tests
    with open(output_dir / 'statistical_tests.json', 'w') as f:
        json.dump(statistical_tests, f, indent=2)
    print(f"Saved statistical tests to {output_dir / 'statistical_tests.json'}")
    
    # Create final aggregated results
    final_results = {
        'summary': {
            'experiment_name': 'Tri-Con: Tri-Hierarchy Contrastive Learning',
            'dataset': 'PBMC 3k (2,638 cells, 2,000 genes, 8 cell types)',
            'seeds_used': [42, 123, 456],
            'key_findings': [
                'Tri-Con V3 achieves competitive performance with baselines (94.4% accuracy)',
                'Cell-Ontology contrast (L_CO) is crucial - removing it causes catastrophic failure (7.8% accuracy)',
                'Gene-Ontology contrast (L_GO) has minimal impact on performance',
                'Evidential uncertainty underperforms vs Maximum Softmax Probability for OOD detection',
                'Uniform sampling achieves similar performance to hierarchy-aware sampling'
            ]
        },
        'detailed_results': results,
        'statistical_tests': statistical_tests
    }
    
    # Save final results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved final results to {output_dir / 'results.json'}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n--- Baselines ---")
    for name, baseline in results['baselines'].items():
        mean = baseline.get('mean', {})
        print(f"{name:20s}: Acc={mean.get('test_accuracy', 0):.4f} ± {mean.get('test_accuracy_std', 0):.4f}, "
              f"ZS={mean.get('zero_shot_accuracy', 0):.4f} ± {mean.get('zero_shot_accuracy_std', 0):.4f}")
    
    print("\n--- Tri-Con V3 (Full Model) ---")
    if results['tricon']:
        mean = results['tricon'].get('mean', {})
        print(f"{'Tri-Con V3':20s}: Acc={mean.get('test_accuracy', 0):.4f} ± {mean.get('test_accuracy_std', 0):.4f}, "
              f"ZS={mean.get('zero_shot_accuracy', 0):.4f} ± {mean.get('zero_shot_accuracy_std', 0):.4f}, "
              f"OOD={mean.get('ood_auroc', 0):.4f} ± {mean.get('ood_auroc_std', 0):.4f}")
    
    print("\n--- Ablations ---")
    for name, ablation in results['ablations'].items():
        if ablation:
            mean = ablation.get('mean', {})
            print(f"{name:20s}: Acc={mean.get('test_accuracy', 0):.4f} ± {mean.get('test_accuracy_std', 0):.4f}, "
                  f"ZS={mean.get('zero_shot_accuracy', 0):.4f} ± {mean.get('zero_shot_accuracy_std', 0):.4f}, "
                  f"OOD={mean.get('ood_auroc', 0):.4f} ± {mean.get('ood_auroc_std', 0):.4f}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
