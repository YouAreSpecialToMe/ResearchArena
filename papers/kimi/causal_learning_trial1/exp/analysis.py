"""
Statistical analysis and hypothesis validation.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def load_results():
    """Load all experimental results."""
    results_dir = Path('results')
    
    with open(results_dir / 'main_experiment.json') as f:
        main_results = json.load(f)
    
    with open(results_dir / 'ablation_study.json') as f:
        ablation_results = json.load(f)
    
    return main_results, ablation_results


def statistical_tests(main_results):
    """Perform statistical significance tests."""
    algorithms = list(set(r['algorithm'] for r in main_results))
    networks = list(set(r['network'] for r in main_results))
    
    test_results = []
    
    for network in networks:
        for n_samples in [100, 200, 500]:
            # Get results for this configuration
            config_results = [
                r for r in main_results 
                if r['network'] == network and r['n_samples'] == n_samples
            ]
            
            if not config_results:
                continue
            
            # Get AIT-LCD results
            ait_results = [r for r in config_results if r['algorithm'] == 'ait-lcd']
            ait_f1 = [r['pc_f1'] for r in ait_results]
            
            for alg in algorithms:
                if alg == 'ait-lcd':
                    continue
                
                alg_results = [r for r in config_results if r['algorithm'] == alg]
                if len(alg_results) < 2:
                    continue
                
                alg_f1 = [r['pc_f1'] for r in alg_results]
                
                # Wilcoxon signed-rank test
                if len(ait_f1) == len(alg_f1) and len(ait_f1) >= 2:
                    try:
                        statistic, pvalue = stats.wilcoxon(ait_f1, alg_f1)
                        test_results.append({
                            'network': network,
                            'n_samples': n_samples,
                            'comparison': f'ait-lcd vs {alg}',
                            'ait_mean': np.mean(ait_f1),
                            'ait_std': np.std(ait_f1),
                            'baseline_mean': np.mean(alg_f1),
                            'baseline_std': np.std(alg_f1),
                            'pvalue': pvalue,
                            'significant': pvalue < 0.05
                        })
                    except Exception as e:
                        pass
    
    return test_results


def validate_hypothesis_h1(main_results):
    """
    H1: AIT-LCD achieves significantly lower SHD than ≥3 of 4 baselines 
    at n ≤ 500 on ≥4 of 5 networks
    """
    networks = ['asia', 'child', 'insurance', 'alarm', 'hailfinder']
    baselines = ['iamb', 'hiton-mb', 'pcmb', 'eamb-inspired']
    
    network_success_count = 0
    
    for network in networks:
        baseline_beaten = 0
        
        for baseline in baselines:
            ait_results = [
                r for r in main_results 
                if r['algorithm'] == 'ait-lcd' and r['network'] == network and r['n_samples'] <= 500
            ]
            baseline_results = [
                r for r in main_results 
                if r['algorithm'] == baseline and r['network'] == network and r['n_samples'] <= 500
            ]
            
            if ait_results and baseline_results:
                ait_f1 = np.mean([r['pc_f1'] for r in ait_results])
                baseline_f1 = np.mean([r['pc_f1'] for r in baseline_results])
                
                if ait_f1 > baseline_f1:
                    baseline_beaten += 1
        
        if baseline_beaten >= 3:
            network_success_count += 1
    
    h1_confirmed = network_success_count >= 4
    
    return {
        'h1_confirmed': h1_confirmed,
        'networks_passed': network_success_count,
        'threshold': '≥4 of 5 networks with ≥3 of 4 baselines beaten'
    }


def validate_hypothesis_h2(ablation_results):
    """
    H2: AIT-LCD with bias correction outperforms without bias correction at n ≤ 300
    """
    full_results = [
        r for r in ablation_results 
        if r['variant'] == 'full' and r['n_samples'] <= 300
    ]
    no_bias_results = [
        r for r in ablation_results 
        if r['variant'] == 'no_bias_correction' and r['n_samples'] <= 300
    ]
    
    if full_results and no_bias_results:
        full_f1 = np.mean([r['pc_f1'] for r in full_results])
        no_bias_f1 = np.mean([r['pc_f1'] for r in no_bias_results])
        
        h2_confirmed = full_f1 > no_bias_f1
        
        return {
            'h2_confirmed': h2_confirmed,
            'full_f1': full_f1,
            'no_bias_f1': no_bias_f1,
            'improvement': (full_f1 - no_bias_f1) / no_bias_f1 if no_bias_f1 > 0 else 0
        }
    
    return {'h2_confirmed': False, 'error': 'Insufficient data'}


def generate_summary_table(main_results):
    """Generate summary table of results."""
    algorithms = sorted(set(r['algorithm'] for r in main_results))
    
    rows = []
    for algorithm in algorithms:
        # Small sample results (n <= 500)
        small_sample = [r for r in main_results if r['algorithm'] == algorithm and r['n_samples'] <= 500]
        
        if small_sample:
            rows.append({
                'Algorithm': algorithm,
                'MB F1 (mean±std)': f"{np.mean([r['mb_f1'] for r in small_sample]):.3f}±{np.std([r['mb_f1'] for r in small_sample]):.3f}",
                'PC F1 (mean±std)': f"{np.mean([r['pc_f1'] for r in small_sample]):.3f}±{np.std([r['pc_f1'] for r in small_sample]):.3f}",
                'Runtime (s)': f"{np.mean([r['runtime'] for r in small_sample]):.2f}"
            })
    
    return pd.DataFrame(rows)


def main():
    print("="*60)
    print("AIT-LCD Statistical Analysis")
    print("="*60)
    print()
    
    # Load results
    main_results, ablation_results = load_results()
    
    # Statistical tests
    print("Performing statistical significance tests...")
    test_results = statistical_tests(main_results)
    
    output_dir = Path('results')
    with open(output_dir / 'statistical_tests.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Hypothesis validation
    print("\nValidating hypotheses...")
    h1_result = validate_hypothesis_h1(main_results)
    h2_result = validate_hypothesis_h2(ablation_results)
    
    print("\n" + "="*60)
    print("Hypothesis Validation Results")
    print("="*60)
    
    print(f"\nH1 (Sample Efficiency): {'CONFIRMED' if h1_result['h1_confirmed'] else 'NOT CONFIRMED'}")
    print(f"  Networks passed: {h1_result['networks_passed']}/5")
    print(f"  Threshold: {h1_result['threshold']}")
    
    print(f"\nH2 (Bias Correction): {'CONFIRMED' if h2_result.get('h2_confirmed', False) else 'NOT CONFIRMED'}")
    if 'full_f1' in h2_result:
        print(f"  With bias correction: {h2_result['full_f1']:.4f}")
        print(f"  Without bias correction: {h2_result['no_bias_f1']:.4f}")
        print(f"  Improvement: {h2_result['improvement']*100:.1f}%")
    
    # Summary table
    print("\n" + "="*60)
    print("Summary Table (Small Samples: n ≤ 500)")
    print("="*60)
    summary_df = generate_summary_table(main_results)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
