"""Create final statistical analysis and visualizations."""
import sys
sys.path.insert(0, '/home/zz865/pythonProject/autoresearch/outputs/kimi/interpretability_of_learned_representations_20260321_115956/idea_01')

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, pearsonr, spearmanr, f_oneway
from scipy import stats
import os


def load_results():
    """Load all experiment results."""
    results = {}
    
    # Synthetic results
    try:
        with open('exp/synthetic/cgas/results.json', 'r') as f:
            results['synthetic'] = json.load(f)
    except:
        print("Warning: Could not load synthetic results")
        results['synthetic'] = None
    
    # IOI results
    try:
        with open('exp/ioi/cgas/results_validated.json', 'r') as f:
            results['ioi'] = json.load(f)
    except:
        print("Warning: Could not load IOI validated results")
        try:
            with open('exp/ioi/cgas/results.json', 'r') as f:
                results['ioi'] = json.load(f)
        except:
            results['ioi'] = None
    
    # RAVEL results
    try:
        with open('exp/ravel/cgas/results.json', 'r') as f:
            results['ravel'] = json.load(f)
    except:
        print("Warning: Could not load RAVEL results")
        results['ravel'] = None
    
    # Ablation results
    try:
        with open('exp/synthetic/ablation_validation/results.json', 'r') as f:
            results['ablation'] = json.load(f)
    except:
        print("Warning: Could not load ablation results")
        results['ablation'] = None
    
    return results


def statistical_analysis(results):
    """Perform comprehensive statistical analysis."""
    print("="*60)
    print("Comprehensive Statistical Analysis")
    print("="*60)
    
    stats_summary = {}
    
    # Synthetic task analysis
    if results['synthetic']:
        print("\n" + "-"*40)
        print("Synthetic Task Analysis")
        print("-"*40)
        
        all_results = results['synthetic']['all_results']
        
        # Group by method
        sae_results = [r for r in all_results if r['method'] == 'sae']
        random_results = [r for r in all_results if r['method'] == 'random']
        pca_results = [r for r in all_results if r['method'] == 'pca']
        
        sae_cgas = [r['cgas'] for r in sae_results]
        random_cgas = [r['cgas'] for r in random_results]
        pca_cgas = [r['cgas'] for r in pca_results]
        
        # T-tests
        t_sae_random, p_sae_random = ttest_ind(sae_cgas, random_cgas)
        t_sae_pca, p_sae_pca = ttest_ind(sae_cgas, pca_cgas)
        
        # ANOVA
        f_stat, p_anova = f_oneway(sae_cgas, random_cgas, pca_cgas)
        
        print(f"SAE mean C-GAS: {np.mean(sae_cgas):.4f} ± {np.std(sae_cgas):.4f}")
        print(f"Random mean C-GAS: {np.mean(random_cgas):.4f} ± {np.std(random_cgas):.4f}")
        print(f"PCA mean C-GAS: {np.mean(pca_cgas):.4f} ± {np.std(pca_cgas):.4f}")
        print(f"\nSAE vs Random: t={t_sae_random:.4f}, p={p_sae_random:.4f}")
        print(f"SAE vs PCA: t={t_sae_pca:.4f}, p={p_sae_pca:.4f}")
        print(f"ANOVA: F={f_stat:.4f}, p={p_anova:.4f}")
        
        # Correlation with recovery
        all_cgas = [r['cgas'] for r in all_results]
        all_recovery = [r['recovery_rate'] for r in all_results]
        corr, p_corr = pearsonr(all_cgas, all_recovery)
        print(f"\nC-GAS vs Recovery correlation: r={corr:.4f}, p={p_corr:.4f}")
        
        stats_summary['synthetic'] = {
            'sae_mean': float(np.mean(sae_cgas)),
            'sae_std': float(np.std(sae_cgas)),
            'random_mean': float(np.mean(random_cgas)),
            'random_std': float(np.std(random_cgas)),
            'pca_mean': float(np.mean(pca_cgas)),
            'pca_std': float(np.std(pca_cgas)),
            'sae_vs_random': {'t_stat': float(t_sae_random), 'p_value': float(p_sae_random)},
            'sae_vs_pca': {'t_stat': float(t_sae_pca), 'p_value': float(p_sae_pca)},
            'anova': {'f_stat': float(f_stat), 'p_value': float(p_anova)},
            'recovery_correlation': {'r': float(corr), 'p_value': float(p_corr)}
        }
    
    # IOI analysis
    if results['ioi']:
        print("\n" + "-"*40)
        print("IOI Task Analysis")
        print("-"*40)
        
        cgas_all = results['ioi'].get('cgas_all', [])
        
        if cgas_all:
            sae_results = [r for r in cgas_all if r['method'] == 'sae']
            random_results = [r for r in cgas_all if r['method'] == 'random']
            pca_results = [r for r in cgas_all if r['method'] == 'pca']
            
            sae_cgas = [r['cgas'] for r in sae_results]
            random_cgas = [r['cgas'] for r in random_results]
            pca_cgas = [r['cgas'] for r in pca_results]
            
            if sae_cgas and random_cgas:
                t_stat, p_val = ttest_ind(sae_cgas, random_cgas)
                print(f"SAE vs Random: t={t_stat:.4f}, p={p_val:.4f}")
                
                stats_summary['ioi'] = {
                    'sae_mean': float(np.mean(sae_cgas)) if sae_cgas else 0,
                    'sae_std': float(np.std(sae_cgas)) if sae_cgas else 0,
                    'random_mean': float(np.mean(random_cgas)) if random_cgas else 0,
                    'random_std': float(np.std(random_cgas)) if random_cgas else 0,
                    'sae_vs_random': {'t_stat': float(t_stat), 'p_value': float(p_val)}
                }
        
        # Statistical tests from results
        if 'statistical_tests' in results['ioi']:
            print("\nPre-computed statistical tests:")
            for test_name, test_result in results['ioi']['statistical_tests'].items():
                print(f"  {test_name}: t={test_result.get('t_stat', 'N/A'):.4f}, "
                      f"p={test_result.get('p_value', 'N/A'):.4f}")
    
    # Ablation analysis
    if results['ablation']:
        print("\n" + "-"*40)
        print("Ablation Study: Validation Impact")
        print("-"*40)
        
        ab_stats = results['ablation'].get('statistical_tests', {})
        
        if 'cgas_comparison' in ab_stats:
            cgas_comp = ab_stats['cgas_comparison']
            print(f"Validated C-GAS: {cgas_comp.get('validated_mean', 0):.4f} ± "
                  f"{cgas_comp.get('validated_std', 0):.4f}")
            print(f"Unvalidated C-GAS: {cgas_comp.get('unvalidated_mean', 0):.4f} ± "
                  f"{cgas_comp.get('unvalidated_std', 0):.4f}")
            print(f"Paired t-test: t={cgas_comp.get('t_statistic', 0):.4f}, "
                  f"p={cgas_comp.get('p_value', 0):.4f}")
        
        if 'correlation_comparison' in ab_stats:
            corr_comp = ab_stats['correlation_comparison']
            print(f"\nValidated correlation with recovery: "
                  f"r={corr_comp.get('validated_correlation', 0):.4f}")
            print(f"Unvalidated correlation with recovery: "
                  f"r={corr_comp.get('unvalidated_correlation', 0):.4f}")
    
    return stats_summary


def create_visualizations(results):
    """Create publication-ready figures."""
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: C-GAS comparison across methods and tasks
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    tasks = ['synthetic', 'ioi', 'ravel']
    task_names = ['Synthetic', 'IOI', 'RAVEL']
    
    for idx, (task, task_name) in enumerate(zip(tasks, task_names)):
        if results[task] is None:
            continue
        
        ax = axes[idx]
        
        # Extract data
        if task == 'synthetic':
            all_results = results[task]['all_results']
            summary = results[task].get('summary', {})
            
            methods = ['random', 'pca', 'sae']
            method_labels = ['Random', 'PCA', 'SAE']
            overcompletes = ['1x', '4x', '16x']
            
            x_pos = []
            means = []
            stds = []
            labels = []
            
            for i, method in enumerate(methods):
                for j, oc in enumerate(overcompletes):
                    if method in summary and oc in summary[method]:
                        x_pos.append(i * 4 + j)
                        means.append(summary[method][oc]['cgas_mean'])
                        stds.append(summary[method][oc]['cgas_std'])
                        labels.append(f"{method_labels[i]}\n{oc}")
            
            ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0.75, color='r', linestyle='--', label='Threshold (0.75)')
            ax.set_ylabel('C-GAS', fontsize=10)
            ax.set_title(f'{task_name} Task', fontsize=12)
            ax.set_ylim(0, 1.5)
            ax.legend(fontsize=8)
            
        else:
            # IOI or RAVEL
            summary = results[task].get('summary', {})
            
            methods = ['random', 'pca', 'sae']
            method_labels = ['Random', 'PCA', 'SAE']
            overcompletes = ['1x', '4x']
            
            x_pos = []
            means = []
            stds = []
            labels = []
            
            for i, method in enumerate(methods):
                for j, oc in enumerate(overcompletes):
                    if method in summary and oc in summary[method]:
                        x_pos.append(i * 3 + j)
                        means.append(summary[method][oc]['mean'])
                        stds.append(summary[method][oc]['std'])
                        labels.append(f"{method_labels[i]}\n{oc}")
            
            ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0.75, color='r', linestyle='--', label='Threshold (0.75)')
            ax.set_ylabel('C-GAS', fontsize=10)
            ax.set_title(f'{task_name} Task', fontsize=12)
            ax.set_ylim(0, 1.5)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures/cgas_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: figures/cgas_comparison.png")
    plt.close()
    
    # Figure 2: Validation impact (if ablation results available)
    if results['ablation']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Load ablation data
        val_results = results['ablation']['results_validated']
        unval_results = results['ablation']['results_unvalidated']
        
        # Left plot: C-GAS distribution
        ax1 = axes[0]
        val_cgas = [r['cgas'] for r in val_results]
        unval_cgas = [r['cgas'] for r in unval_results]
        
        ax1.boxplot([unval_cgas, val_cgas], labels=['Unvalidated', 'Validated'])
        ax1.set_ylabel('C-GAS', fontsize=10)
        ax1.set_title('C-GAS: Validated vs Unvalidated', fontsize=12)
        ax1.axhline(y=0.75, color='r', linestyle='--', alpha=0.5)
        
        # Right plot: Correlation with recovery
        ax2 = axes[1]
        
        val_recovery = [(r['cgas'], r['recovery_rate']) for r in val_results]
        unval_recovery = [(r['cgas'], r['recovery_rate']) for r in unval_results]
        
        ax2.scatter([x[0] for x in unval_recovery], [x[1] for x in unval_recovery],
                   alpha=0.5, label='Unvalidated', s=50)
        ax2.scatter([x[0] for x in val_recovery], [x[1] for x in val_recovery],
                   alpha=0.5, label='Validated', s=50)
        
        ax2.set_xlabel('C-GAS', fontsize=10)
        ax2.set_ylabel('Ground Truth Recovery Rate', fontsize=10)
        ax2.set_title('C-GAS vs Recovery Correlation', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('figures/validation_impact.png', dpi=300, bbox_inches='tight')
        print("  Saved: figures/validation_impact.png")
        plt.close()
    
    # Figure 3: Recovery rate comparison (Synthetic only)
    if results['synthetic']:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        all_results = results['synthetic']['all_results']
        
        methods = ['random', 'pca', 'sae']
        method_labels = ['Random', 'PCA', 'SAE']
        
        for method, label in zip(methods, method_labels):
            method_data = [(r['cgas'], r['recovery_rate']) 
                          for r in all_results if r['method'] == method]
            if method_data:
                ax.scatter([x[0] for x in method_data], [x[1] for x in method_data],
                          label=label, alpha=0.6, s=100)
        
        ax.set_xlabel('C-GAS', fontsize=12)
        ax.set_ylabel('Ground Truth Recovery Rate', fontsize=12)
        ax.set_title('C-GAS vs Ground Truth Feature Recovery', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/cgas_vs_recovery.png', dpi=300, bbox_inches='tight')
        print("  Saved: figures/cgas_vs_recovery.png")
        plt.close()


def compile_final_results(results, stats_summary):
    """Compile final results.json."""
    print("\n" + "="*60)
    print("Compiling Final Results")
    print("="*60)
    
    final_results = {
        "experiment_summary": {
            "description": "CAGER: Causal Geometric Explanation Recovery",
            "tasks_evaluated": ["synthetic", "ioi", "ravel"],
            "methods_evaluated": ["SAE", "PCA", "Random"],
            "validation_threshold": ">=2 of 3 checks",
            "n_seeds": 3
        },
        "results_by_task": {},
        "statistical_tests": stats_summary,
        "success_criteria": {
            "criterion_1_sae_threshold": {"description": "SAEs achieve C-GAS > 0.75", "status": "TBD"},
            "criterion_2_baseline_comparison": {"description": "SAEs significantly higher than baselines (p < 0.01)", "status": "TBD"},
            "criterion_3_recovery_correlation": {"description": "C-GAS correlates with ground truth recovery (r > 0.8)", "status": "TBD"},
            "criterion_4_validation_matters": {"description": "Validation improves C-GAS predictive power", "status": "TBD"}
        }
    }
    
    # Add task results
    if results['synthetic']:
        summary = results['synthetic'].get('summary', {})
        final_results['results_by_task']['synthetic'] = summary
        
        # Check criterion 1
        if 'sae' in summary:
            sae_means = [summary['sae'][k]['cgas_mean'] for k in summary['sae'] if 'cgas_mean' in summary['sae'][k]]
            if sae_means and np.mean(sae_means) > 0.75:
                final_results['success_criteria']['criterion_1_sae_threshold']['status'] = 'PASS'
                final_results['success_criteria']['criterion_1_sae_threshold']['mean_cgas'] = float(np.mean(sae_means))
            else:
                final_results['success_criteria']['criterion_1_sae_threshold']['status'] = 'FAIL'
                final_results['success_criteria']['criterion_1_sae_threshold']['mean_cgas'] = float(np.mean(sae_means)) if sae_means else 0
        
        # Check criterion 2
        if 'synthetic' in stats_summary:
            p_val = stats_summary['synthetic'].get('sae_vs_random', {}).get('p_value', 1)
            if p_val < 0.01:
                final_results['success_criteria']['criterion_2_baseline_comparison']['status'] = 'PASS'
            else:
                final_results['success_criteria']['criterion_2_baseline_comparison']['status'] = 'FAIL'
            final_results['success_criteria']['criterion_2_baseline_comparison']['p_value'] = float(p_val)
        
        # Check criterion 3
        if 'synthetic' in stats_summary:
            r = stats_summary['synthetic'].get('recovery_correlation', {}).get('r', 0)
            if r > 0.8:
                final_results['success_criteria']['criterion_3_recovery_correlation']['status'] = 'PASS'
            else:
                final_results['success_criteria']['criterion_3_recovery_correlation']['status'] = 'FAIL'
            final_results['success_criteria']['criterion_3_recovery_correlation']['correlation'] = float(r)
    
    if results['ioi']:
        final_results['results_by_task']['ioi'] = results['ioi'].get('summary', {})
    
    if results['ravel']:
        final_results['results_by_task']['ravel'] = results['ravel'].get('summary', {})
    
    # Check criterion 4 (validation matters)
    if results['ablation']:
        corr_comp = results['ablation'].get('statistical_tests', {}).get('correlation_comparison', {})
        val_corr = corr_comp.get('validated_correlation', 0)
        unval_corr = corr_comp.get('unvalidated_correlation', 0)
        
        if val_corr > unval_corr:
            final_results['success_criteria']['criterion_4_validation_matters']['status'] = 'PASS'
        else:
            final_results['success_criteria']['criterion_4_validation_matters']['status'] = 'FAIL'
        
        final_results['success_criteria']['criterion_4_validation_matters']['validated_corr'] = float(val_corr)
        final_results['success_criteria']['criterion_4_validation_matters']['unvalidated_corr'] = float(unval_corr)
    
    # Save
    with open('results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nFinal results saved to results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("Success Criteria Summary")
    print("="*60)
    for criterion, info in final_results['success_criteria'].items():
        status = info['status']
        symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "?"
        print(f"{symbol} {criterion}: {status}")


def main():
    print("="*60)
    print("CAGER Final Analysis")
    print("="*60)
    
    # Load results
    results = load_results()
    
    # Perform statistical analysis
    stats_summary = statistical_analysis(results)
    
    # Create visualizations
    create_visualizations(results)
    
    # Compile final results
    compile_final_results(results, stats_summary)
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("="*60)


if __name__ == '__main__':
    main()
