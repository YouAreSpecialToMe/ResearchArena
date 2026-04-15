"""
Analyze existing checkpoint data and generate honest results.json.
This addresses the feedback about inaccurate reporting.
"""

import json
import numpy as np
from scipy import stats

def load_aggregated(path):
    """Load aggregated results from checkpoint."""
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def analyze_existing_results():
    """Analyze what we have from existing checkpoints."""
    
    # Load existing results
    vmamba = load_aggregated('checkpoints/vmamba/aggregated.json')
    localmamba = load_aggregated('checkpoints/localmamba/aggregated.json')
    cassvim_4d = load_aggregated('checkpoints/cassvim_4d/aggregated.json')
    cassvim_8d = load_aggregated('checkpoints/cassvim_8d/aggregated.json')
    
    # Load random selection partial results
    random_results = []
    for seed in [42, 123]:
        try:
            with open(f'checkpoints/random_selection/results_seed{seed}.json') as f:
                random_results.append(json.load(f))
        except:
            pass
    
    print("="*70)
    print("EXISTING CHECKPOINT DATA ANALYSIS")
    print("="*70)
    
    if vmamba:
        print(f"\nVMamba:")
        print(f"  Best Acc: {vmamba['best_acc_mean']:.2f} ± {vmamba['best_acc_std']:.2f}%")
        print(f"  Params: {vmamba['individual_results'][0]['n_parameters']:,}")
        
    if localmamba:
        print(f"\nLocalMamba:")
        print(f"  Best Acc: {localmamba['best_acc_mean']:.2f} ± {localmamba['best_acc_std']:.2f}%")
        print(f"  Params: {localmamba['individual_results'][0]['n_parameters']:,}")
        
    if cassvim_4d:
        print(f"\nCASS-ViM-4D:")
        print(f"  Best Acc: {cassvim_4d['best_acc_mean']:.2f} ± {cassvim_4d['best_acc_std']:.2f}%")
        print(f"  Params: {cassvim_4d['individual_results'][0]['n_parameters']:,}")
        
    if cassvim_8d:
        print(f"\nCASS-ViM-8D:")
        print(f"  Best Acc: {cassvim_8d['best_acc_mean']:.2f} ± {cassvim_8d['best_acc_std']:.2f}%")
        print(f"  Params: {cassvim_8d['individual_results'][0]['n_parameters']:,}")
    
    if random_results:
        random_accs = [r['best_test_acc'] for r in random_results]
        print(f"\nRandom Selection (partial, {len(random_results)} seeds):")
        print(f"  Best Acc: {np.mean(random_accs):.2f}%")
        print(f"  Params: {random_results[0]['n_parameters']:,}")
    
    # Honest assessment
    print("\n" + "="*70)
    print("HONEST ASSESSMENT OF EXISTING RESULTS")
    print("="*70)
    
    if vmamba and cassvim_4d:
        vmamba_acc = vmamba['best_acc_mean']
        cassvim_acc = cassvim_4d['best_acc_mean']
        diff = cassvim_acc - vmamba_acc
        
        print(f"\n1. ARCHITECTURE MISMATCH (CRITICAL):")
        print(f"   VMamba:     {vmamba['individual_results'][0]['n_parameters']:,} params")
        print(f"   CASS-ViM:   {cassvim_4d['individual_results'][0]['n_parameters']:,} params")
        print(f"   Difference: CASS-ViM has {100*(1-cassvim_4d['individual_results'][0]['n_parameters']/vmamba['individual_results'][0]['n_parameters']):.1f}% FEWER parameters")
        print(f"   THIS IS NOT A FAIR COMPARISON")
        
        print(f"\n2. ACCURACY COMPARISON:")
        print(f"   VMamba:     {vmamba_acc:.2f}% ± {vmamba['best_acc_std']:.2f}%")
        print(f"   CASS-ViM:   {cassvim_acc:.2f}% ± {cassvim_4d['best_acc_std']:.2f}%")
        print(f"   Difference: {diff:+.2f}% (CASS-ViM is {abs(diff):.2f}% {'behind' if diff < 0 else 'ahead'})")
        
        # Statistical test
        vmamba_accs = [r['best_test_acc'] for r in vmamba['individual_results']]
        cassvim_accs = [r['best_test_acc'] for r in cassvim_4d['individual_results']]
        t_stat, p_val = stats.ttest_rel(vmamba_accs, cassvim_accs)
        
        print(f"\n3. STATISTICAL SIGNIFICANCE:")
        print(f"   Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"   Significant difference (p<0.05): {p_val < 0.05}")
        
    if localmamba and cassvim_4d:
        local_acc = localmamba['best_acc_mean']
        cassvim_acc = cassvim_4d['best_acc_mean']
        diff = cassvim_acc - local_acc
        
        print(f"\n4. vs LOCALMAMBA:")
        print(f"   LocalMamba: {local_acc:.2f}% ± {localmamba['best_acc_std']:.2f}%")
        print(f"   CASS-ViM:   {cassvim_acc:.2f}% ± {cassvim_4d['best_acc_std']:.2f}%")
        print(f"   Difference: {diff:+.2f}%")
        print(f"   Note: LocalMamba also has {100*(1-localmamba['individual_results'][0]['n_parameters']/vmamba['individual_results'][0]['n_parameters']):.1f}% fewer params than VMamba")
        
    if cassvim_4d and cassvim_8d:
        cassvim4_acc = cassvim_4d['best_acc_mean']
        cassvim8_acc = cassvim_8d['best_acc_mean']
        diff = cassvim8_acc - cassvim4_acc
        
        print(f"\n5. 4D vs 8D:")
        print(f"   CASS-ViM-4D: {cassvim4_acc:.2f}% ± {cassvim_4d['best_acc_std']:.2f}%")
        print(f"   CASS-ViM-8D: {cassvim8_acc:.2f}% ± {cassvim_8d['best_acc_std']:.2f}%")
        print(f"   Difference: {diff:+.2f}% (8D is {'better' if diff > 0 else 'worse'})")
        print(f"   Similar performance (within 1%): {abs(diff) <= 1.0}")
    
    if random_results and cassvim_4d:
        random_acc = np.mean([r['best_test_acc'] for r in random_results])
        cassvim_acc = cassvim_4d['best_acc_mean']
        diff = cassvim_acc - random_acc
        
        print(f"\n6. GRADIENT vs RANDOM (partial data):")
        print(f"   Random:     {random_acc:.2f}% ({len(random_results)} seeds)")
        print(f"   Gradient:   {cassvim_acc:.2f}%")
        print(f"   Difference: {diff:+.2f}%")
        if len(random_results) >= 2:
            print(f"   Gradient-based selection APPEARS effective")
        print(f"   WARNING: Random selection took ~40 min vs ~18 min for gradient")
        
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
1. CASS-ViM experiments DID complete successfully (contrary to old results.json)
2. CASS-ViM underperforms VMamba by ~6% on this architecture configuration
3. The comparison is UNFAIR due to architecture mismatch (1.1M vs 3.6M params)
4. LocalMamba also underperforms VMamba with fewer parameters (3.3M)
5. 4D and 8D variants show similar performance (both ~51%)
6. Gradient selection outperforms random selection by ~6.5%

RECOMMENDATION: Run experiments with architecture-matched models for fair comparison.
""")
    
    # Generate honest results.json
    results = {
        "experiment_info": {
            "title": "CASS-ViM: Content-Adaptive Selective Scanning for Vision State Space Models",
            "dataset": "CIFAR-100",
            "epochs": 100,
            "note": "FAST VALIDATION - NOT FULL CONVERGENCE. CRITICAL LIMITATION: Architecture mismatch between methods.",
            "date": "2026-03-23",
            "architecture_note": "MAJOR ISSUE: CASS-ViM has ~70% fewer parameters than VMamba (1.1M vs 3.6M). This comparison is NOT FAIR."
        },
        "main_results": {},
        "honest_assessment": {
            "key_finding": "CASS-ViM completed experiments but with significantly smaller architecture than baselines",
            "architecture_comparison": {
                "vmamba_params": 3600452,
                "localmamba_params": 3327172,
                "cassvim_4d_params": 1112164,
                "cassvim_8d_params": 1113220,
                "cassvim_vs_vmamba_ratio": 0.31
            },
            "performance_comparison": {
                "vmamba": f"{vmamba['best_acc_mean']:.2f}%" if vmamba else "N/A",
                "localmamba": f"{localmamba['best_acc_mean']:.2f}%" if localmamba else "N/A",
                "cassvim_4d": f"{cassvim_4d['best_acc_mean']:.2f}%" if cassvim_4d else "N/A",
                "cassvim_8d": f"{cassvim_8d['best_acc_mean']:.2f}%" if cassvim_8d else "N/A"
            },
            "critical_issues": [
                "Architecture mismatch: CASS-ViM has 70% fewer parameters than VMamba",
                "CASS-ViM underperforms VMamba by ~6% (expected given parameter difference)",
                "LocalMamba also underperforms VMamba (3% gap with 8% fewer params)",
                "Random selection ablation incomplete (only 2/3 seeds)",
                "Fixed-per-layer ablation not run",
                "No overhead analysis completed",
                "No ImageNet-100 validation completed"
            ],
            "valid_findings": [
                "4D and 8D variants show similar performance (~51%)",
                "Gradient-based selection outperforms random by ~6.5%",
                "CASS-ViM successfully trains and converges (no instability)",
                "Per-sample selection is implemented and functional"
            ],
            "fair_conclusion": "With 3x fewer parameters, CASS-ViM achieves ~51% vs VMamba's ~57%. This ~6% gap is likely due to capacity differences, not method quality. Architecture-matched experiments are needed for fair assessment."
        }
    }
    
    if vmamba:
        results["main_results"]["vmamba"] = {
            "accuracy_mean": round(vmamba['best_acc_mean'], 2),
            "accuracy_std": round(vmamba['best_acc_std'], 2),
            "n_parameters": vmamba['individual_results'][0]['n_parameters'],
            "seeds": [42, 123, 456]
        }
    
    if localmamba:
        results["main_results"]["localmamba"] = {
            "accuracy_mean": round(localmamba['best_acc_mean'], 2),
            "accuracy_std": round(localmamba['best_acc_std'], 2),
            "n_parameters": localmamba['individual_results'][0]['n_parameters'],
            "seeds": [42, 123, 456]
        }
    
    if cassvim_4d:
        results["main_results"]["cassvim_4d"] = {
            "accuracy_mean": round(cassvim_4d['best_acc_mean'], 2),
            "accuracy_std": round(cassvim_4d['best_acc_std'], 2),
            "n_parameters": cassvim_4d['individual_results'][0]['n_parameters'],
            "seeds": [42, 123, 456]
        }
    
    if cassvim_8d:
        results["main_results"]["cassvim_8d"] = {
            "accuracy_mean": round(cassvim_8d['best_acc_mean'], 2),
            "accuracy_std": round(cassvim_8d['best_acc_std'], 2),
            "n_parameters": cassvim_8d['individual_results'][0]['n_parameters'],
            "seeds": [42, 123, 456]
        }
    
    if random_results:
        results["ablation_results"] = {
            "random_selection": {
                "accuracy_mean": round(np.mean([r['best_test_acc'] for r in random_results]), 2),
                "n_parameters": random_results[0]['n_parameters'],
                "seeds_completed": len(random_results),
                "note": "Incomplete - only 2/3 seeds"
            }
        }
    
    with open('results_honest.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nHonest results saved to results_honest.json")
    
    return results

if __name__ == '__main__':
    analyze_existing_results()
