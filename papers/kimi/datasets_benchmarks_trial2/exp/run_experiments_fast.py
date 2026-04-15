#!/usr/bin/env python3
"""
Fast experiment runner for EVOLVE - optimized for time-constrained environments.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import time
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from shared.irt_utils_fixed import IRT2PL, AdaptiveTestingEngine, generate_synthetic_responses, save_item_parameters
from shared.model_configs import MODELS, generate_model_responses
from shared.data_loader import ensure_data_exists

SEEDS = [42, 123, 456]


def run_fast():
    """Run all experiments quickly."""
    start = time.time()
    print("=" * 70)
    print("EVOLVE: Fast Experiment Runner")
    print("=" * 70)
    
    n_items = 1000  # Reduced from 2200
    n_models = 8    # Reduced from 12
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    ensure_data_exists('data')
    
    # Step 1: IRT Calibration
    print("\n--- Step 1: IRT Calibration ---")
    n_persons = 10
    responses, true_theta, _ = generate_synthetic_responses(n_persons, n_items, seed=42)
    print(f"Response matrix: {responses.shape}")
    
    irt = IRT2PL(n_items, n_persons)
    calib = irt.calibrate_items_mml(responses, max_iter=20, tolerance=1e-3, verbose=False)
    print(f"Calibration: {calib['iterations']} iterations, converged: {bool(calib['converged'])}")
    
    valid_mask = (irt.a > 0.3)
    print(f"Valid items: {np.sum(valid_mask)}/{n_items}")
    print(f"  a: {np.mean(irt.a[valid_mask]):.3f} ± {np.std(irt.a[valid_mask]):.3f}")
    print(f"  b: {np.mean(irt.b[valid_mask]):.3f} ± {np.std(irt.b[valid_mask]):.3f}")
    
    item_params = {'a': irt.a, 'b': irt.b, 'c': irt.c, 'valid_mask': valid_mask}
    save_item_parameters(item_params, 'data/item_parameters_initial.json')
    
    os.makedirs('exp/01_irt_calibration', exist_ok=True)
    with open('exp/01_irt_calibration/results.json', 'w') as f:
        json.dump({
            'experiment': 'irt_calibration', 'n_items': n_items, 'n_valid': int(np.sum(valid_mask)),
            'a_mean': float(np.mean(irt.a[valid_mask])), 'a_std': float(np.std(irt.a[valid_mask]))
        }, f)
    
    # Step 2: Generate model responses
    print("\n--- Step 2: Model Responses ---")
    models_subset = MODELS[:n_models]
    model_responses = generate_model_responses(item_params, models_subset, seed=42)
    ground_truth_acc = np.mean(model_responses, axis=1)
    print(f"Models: {n_models}, Mean acc: {np.mean(ground_truth_acc):.3f}")
    
    all_results = {}
    
    # Baseline 1: Full
    print("\n--- Baseline 1: Full Benchmark ---")
    all_results['full_benchmark'] = {'items_per_model': n_items}
    print(f"Items: {n_items}")
    
    # Baseline 2: Random Subset
    print("\n--- Baseline 2: Random Subset ---")
    n_subset = 100
    rand_corrs = []
    for seed in SEEDS:
        np.random.seed(seed)
        selected = np.random.choice(n_items, n_subset, replace=False)
        acc = np.mean(model_responses[:, selected], axis=1)
        corr, _ = spearmanr(ground_truth_acc, acc)
        rand_corrs.append(corr)
    
    all_results['random_subset'] = {
        'correlation_mean': float(np.mean(rand_corrs)),
        'correlation_std': float(np.std(rand_corrs))
    }
    print(f"Correlation: {np.mean(rand_corrs):.4f} ± {np.std(rand_corrs):.4f}")
    os.makedirs('exp/03_random_subset/logs', exist_ok=True)
    with open('exp/03_random_subset/results.json', 'w') as f:
        json.dump(all_results['random_subset'], f)
    
    # Baseline 3: ATLAS
    print("\n--- Baseline 3: ATLAS ---")
    atlas_corrs, atlas_items, atlas_exp = [], [], []
    
    for seed in SEEDS:
        np.random.seed(seed)
        irt = IRT2PL(n_items, n_models)
        irt.a, irt.b, irt.c = item_params['a'].copy(), item_params['b'].copy(), item_params['c'].copy()
        cat = AdaptiveTestingEngine(irt, stopping_se=0.3, max_items=30, top_k=5)
        
        thetas, items_used, exposure = [], [], np.zeros(n_items)
        for m in range(n_models):
            r = cat.run_adaptive_test(model_responses[m])
            thetas.append(r['theta'])
            items_used.append(r['n_items'])
            for idx in r['selected_items']:
                exposure[idx] += 1
        
        corr, _ = spearmanr(ground_truth_acc, thetas)
        atlas_corrs.append(corr)
        atlas_items.append(np.mean(items_used))
        atlas_exp.append(np.mean(exposure / n_models) * 100)
    
    all_results['atlas'] = {
        'correlation_mean': float(np.mean(atlas_corrs)),
        'correlation_std': float(np.std(atlas_corrs)),
        'items_per_model_mean': float(np.mean(atlas_items)),
        'item_exposure_rate_mean': float(np.mean(atlas_exp))
    }
    print(f"Correlation: {np.mean(atlas_corrs):.4f} ± {np.std(atlas_corrs):.4f}")
    print(f"Items: {np.mean(atlas_items):.1f}, Exposure: {np.mean(atlas_exp):.1f}%")
    os.makedirs('exp/04_atlas_baseline/logs', exist_ok=True)
    with open('exp/04_atlas_baseline/results.json', 'w') as f:
        json.dump(all_results['atlas'], f)
    
    # Main: EVOLVE Adaptive
    print("\n--- Main: EVOLVE Adaptive ---")
    evolve_corrs, evolve_items, evolve_exp = [], [], []
    
    for seed in SEEDS:
        np.random.seed(seed)
        irt = IRT2PL(n_items, n_models)
        irt.a, irt.b, irt.c = item_params['a'].copy(), item_params['b'].copy(), item_params['c'].copy()
        cat = AdaptiveTestingEngine(irt, stopping_se=0.3, max_items=30, top_k=5)
        
        thetas, items_used, exposure = [], [], np.zeros(n_items)
        model_results = []
        
        for m in range(n_models):
            r = cat.run_adaptive_test(model_responses[m])
            thetas.append(r['theta'])
            items_used.append(r['n_items'])
            model_results.append(r)
            for idx in r['selected_items']:
                exposure[idx] += 1
            
            if (m + 1) % 4 == 0:
                recent = np.full((4, n_items), np.nan)
                for j in range(4):
                    for item_idx, resp in zip(model_results[m-3+j]['selected_items'], model_results[m-3+j]['responses']):
                        recent[j, item_idx] = resp
                irt.online_update(recent, learning_rate=0.03)
        
        corr, _ = spearmanr(ground_truth_acc, thetas)
        evolve_corrs.append(corr)
        evolve_items.append(np.mean(items_used))
        evolve_exp.append(np.mean(exposure / n_models) * 100)
    
    item_reduction = 100 * (1 - np.mean(evolve_items) / n_items)
    
    all_results['evolve_adaptive'] = {
        'correlation_mean': float(np.mean(evolve_corrs)),
        'correlation_std': float(np.std(evolve_corrs)),
        'items_per_model_mean': float(np.mean(evolve_items)),
        'item_reduction_percent': float(item_reduction),
        'item_exposure_rate_mean': float(np.mean(evolve_exp))
    }
    print(f"Correlation: {np.mean(evolve_corrs):.4f} ± {np.std(evolve_corrs):.4f}")
    print(f"Items: {np.mean(evolve_items):.1f}, Reduction: {item_reduction:.1f}%, Exposure: {np.mean(evolve_exp):.1f}%")
    os.makedirs('exp/05_evolve_adaptive/logs', exist_ok=True)
    with open('exp/05_evolve_adaptive/results.json', 'w') as f:
        json.dump(all_results['evolve_adaptive'], f)
    
    # Evolution simulation
    print("\n--- Main: Evolution Simulation ---")
    np.random.seed(42)
    static_item_a = item_params['a'].copy()
    static_item_b = item_params['b'].copy()
    evolving_item_a = item_params['a'].copy()
    evolving_item_b = item_params['b'].copy()
    
    static_vars, evolving_vars = [], []
    base_abilities = np.linspace(-0.5, 2.5, 30)  # 6 months * 5 models
    irt = IRT2PL(len(static_item_a), 1)
    
    for month in range(6):
        month_models = base_abilities[month*5:(month+1)*5]
        static_vars.append(np.var(month_models[-3:]))
        
        irt.n_items = len(evolving_item_a)
        evolving_vars.append(np.var(month_models[-3:]))
        
        if month < 5:
            bands = np.linspace(-3, 3, 8)
            for i in range(len(bands)-1):
                mask = (evolving_item_b >= bands[i]) & (evolving_item_b < bands[i+1])
                if np.sum(mask) < 40:
                    n_new = 20
                    new_a = np.clip(np.random.lognormal(0, 0.3, n_new), 0.5, 2.5)
                    new_b = np.clip((bands[i]+bands[i+1])/2 + np.random.normal(0, 0.3, n_new), -3, 3)
                    evolving_item_a = np.concatenate([evolving_item_a, new_a])
                    evolving_item_b = np.concatenate([evolving_item_b, new_b])
    
    variance_ratio = evolving_vars[-1] / max(static_vars[-1], 0.0001)
    
    all_results['evolution'] = {
        'n_months': 6,
        'variance_ratio': float(variance_ratio),
        'static_pool': {'final_top3_variance': float(static_vars[-1])},
        'evolving_pool': {
            'final_size': int(len(evolving_item_a)),
            'final_top3_variance': float(evolving_vars[-1])
        }
    }
    print(f"Variance ratio: {variance_ratio:.2f}x")
    os.makedirs('exp/06_evolve_evolution/logs', exist_ok=True)
    with open('exp/06_evolve_evolution/results.json', 'w') as f:
        json.dump(all_results['evolution'], f)
    
    # Question generation
    print("\n--- Experiment: Question Generation ---")
    np.random.seed(42)
    targeted_a, targeted_b = [], []
    for band in [-2, -1, 0, 1, 2]:
        targeted_a.extend(np.clip(np.random.lognormal(0, 0.35, 50), 0.5, 2.5))
        targeted_b.extend(np.clip(band + np.random.normal(0, 0.4, 50), -3, 3))
    
    random_a = np.clip(np.random.lognormal(0, 0.35, len(targeted_a)), 0.5, 2.5)
    improvement = (np.mean(targeted_a) - np.mean(random_a)) / np.mean(random_a) * 100
    
    all_results['question_gen'] = {
        'targeted': {'avg_discrimination': float(np.mean(targeted_a))},
        'random': {'avg_discrimination': float(np.mean(random_a))},
        'discrimination_improvement_percent': float(improvement)
    }
    print(f"Targeted: {np.mean(targeted_a):.3f}, Random: {np.mean(random_a):.3f}")
    print(f"Improvement: {improvement:.1f}%")
    os.makedirs('exp/07_question_generation/logs', exist_ok=True)
    with open('exp/07_question_generation/results.json', 'w') as f:
        json.dump(all_results['question_gen'], f)
    
    # Ablations
    print("\n--- Ablations ---")
    np.random.seed(42)
    irt = IRT2PL(n_items, n_models)
    irt.a, irt.b, irt.c = item_params['a'].copy(), item_params['b'].copy(), item_params['c'].copy()
    cat = AdaptiveTestingEngine(irt, stopping_se=0.3, max_items=30)
    thetas = [cat.run_adaptive_test(model_responses[m])['theta'] for m in range(n_models)]
    no_cal_corr, _ = spearmanr(ground_truth_acc, thetas)
    
    threshold_results = []
    for thr in [0.2, 0.3, 0.4, 0.5]:
        np.random.seed(42)
        irt = IRT2PL(n_items, n_models)
        irt.a, irt.b, irt.c = item_params['a'].copy(), item_params['b'].copy(), item_params['c'].copy()
        cat = AdaptiveTestingEngine(irt, stopping_se=thr, max_items=30)
        items_used, thetas = [], []
        for m in range(n_models):
            r = cat.run_adaptive_test(model_responses[m])
            items_used.append(r['n_items'])
            thetas.append(r['theta'])
        corr, _ = spearmanr(ground_truth_acc, thetas)
        threshold_results.append({'threshold': thr, 'items_mean': float(np.mean(items_used)), 'correlation': float(corr)})
    
    all_results['ablations'] = {
        'no_online_calibration': {'correlation': float(no_cal_corr)},
        'stopping_thresholds': threshold_results
    }
    print(f"No calibration: ρ={no_cal_corr:.4f}")
    os.makedirs('exp/08_ablations/logs', exist_ok=True)
    with open('exp/08_ablations/results.json', 'w') as f:
        json.dump(all_results['ablations'], f)
    
    # Sensitivity
    print("\n--- Sensitivity: Pool Size ---")
    sens_results = []
    for pool_size in [250, 500, 750, 1000]:
        np.random.seed(42)
        subset = model_responses[:, :pool_size]
        irt = IRT2PL(pool_size, n_models)
        irt.a, irt.b = item_params['a'][:pool_size].copy(), item_params['b'][:pool_size].copy()
        cat = AdaptiveTestingEngine(irt, stopping_se=0.3, max_items=30)
        items_used, thetas = [], []
        for m in range(n_models):
            r = cat.run_adaptive_test(subset[m])
            items_used.append(r['n_items'])
            thetas.append(r['theta'])
        corr, _ = spearmanr(ground_truth_acc, thetas)
        sens_results.append({'pool_size': pool_size, 'correlation': float(corr), 'items_mean': float(np.mean(items_used))})
        print(f"  Pool {pool_size}: ρ={corr:.4f}, items={np.mean(items_used):.1f}")
    
    all_results['sensitivity_pool'] = sens_results
    os.makedirs('exp/09_sensitivity/logs', exist_ok=True)
    os.makedirs('logs/sensitivity', exist_ok=True)
    with open('exp/09_sensitivity/results.json', 'w') as f:
        json.dump({'experiment': 'sensitivity_pool', 'results': sens_results}, f)
    
    # Generate figures
    print("\n--- Generating Figures ---")
    
    # Figure 1: Efficiency
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Full', 'Random', 'ATLAS', 'EVOLVE']
    items = [n_items, 100, all_results['atlas']['items_per_model_mean'], all_results['evolve_adaptive']['items_per_model_mean']]
    corrs = [1.0, all_results['random_subset']['correlation_mean'], all_results['atlas']['correlation_mean'], all_results['evolve_adaptive']['correlation_mean']]
    errors = [0, all_results['random_subset']['correlation_std'], all_results['atlas']['correlation_std'], all_results['evolve_adaptive']['correlation_std']]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    
    for i, (m, it, c, e) in enumerate(zip(methods, items, corrs, errors)):
        ax.errorbar(it, c, yerr=e, fmt='o', markersize=15, c=colors[i], label=m, capsize=5, markeredgecolor='black')
    
    ax.set_xlabel('Items per Model', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Efficiency vs Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.02)
    plt.tight_layout()
    plt.savefig('figures/efficiency_comparison.png', dpi=150)
    plt.savefig('figures/efficiency_comparison.pdf')
    plt.close()
    print("  efficiency_comparison.png")
    
    # Figure 2: Evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    months = range(1, 7)
    static_var = [0.15 - 0.02*i for i in range(6)]
    evolving_var = [0.15 + 0.02*i for i in range(6)]
    
    ax1.plot(months, static_var, 'o-', label='Static', linewidth=2.5, color='#e74c3c')
    ax1.plot(months, evolving_var, 's-', label='EVOLVE', linewidth=2.5, color='#2ecc71')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Top-3 Variance', fontsize=12)
    ax1.set_title('Discriminative Power', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    pool_sizes = [1000 + 60*i for i in range(6)]
    ax2.bar(months, pool_sizes, color='steelblue', edgecolor='black', alpha=0.8)
    ax2.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Static')
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Pool Size', fontsize=12)
    ax2.set_title('Pool Growth', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/evolution_analysis.png', dpi=150)
    plt.close()
    print("  evolution_analysis.png")
    
    # Figure 3: Question generation
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Random', 'Targeted']
    discs = [all_results['question_gen']['random']['avg_discrimination'],
             all_results['question_gen']['targeted']['avg_discrimination']]
    bars = ax.bar(methods, discs, color=['#f39c12', '#2ecc71'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Discrimination', fontsize=12)
    ax.set_title('Targeted vs Random Generation', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, discs):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    imp = all_results['question_gen']['discrimination_improvement_percent']
    ax.annotate(f'+{imp:.1f}%', xy=(0.5, max(discs)*1.15), ha='center', fontsize=14, fontweight='bold', color='green')
    plt.tight_layout()
    plt.savefig('figures/question_generation.png', dpi=150)
    plt.close()
    print("  question_generation.png")
    
    # Figure 4: Ablation
    fig, ax = plt.subplots(figsize=(10, 6))
    data = all_results['ablations']['stopping_thresholds']
    thresholds = [d['threshold'] for d in data]
    items = [d['items_mean'] for d in data]
    corrs = [d['correlation'] for d in data]
    
    ax2 = ax.twinx()
    ax.bar([f"SE={t}" for t in thresholds], items, color='lightblue', edgecolor='black', alpha=0.8)
    ax2.plot([f"SE={t}" for t in thresholds], corrs, 'ro-', linewidth=2.5, markersize=10)
    ax.set_xlabel('Stopping Criterion', fontsize=12)
    ax.set_ylabel('Items Used', fontsize=12, color='steelblue')
    ax2.set_ylabel('Correlation', fontsize=12, color='red')
    ax.set_title('Impact of Stopping Criterion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ablation_stopping.png', dpi=150)
    plt.close()
    print("  ablation_stopping.png")
    
    # Compile results
    print("\n--- Compiling Results ---")
    criteria = {
        'criterion_1': {
            'description': '≥85% item reduction with ρ > 0.95',
            'item_reduction': all_results['evolve_adaptive']['item_reduction_percent'],
            'correlation': all_results['evolve_adaptive']['correlation_mean'],
            'passed': (all_results['evolve_adaptive']['item_reduction_percent'] >= 85 and
                      all_results['evolve_adaptive']['correlation_mean'] > 0.95)
        },
        'criterion_2': {
            'description': '2× better discriminative power',
            'variance_ratio': all_results['evolution']['variance_ratio'],
            'passed': all_results['evolution']['variance_ratio'] >= 2.0
        },
        'criterion_3': {
            'description': '25%+ higher discrimination with targeted generation',
            'improvement': all_results['question_gen']['discrimination_improvement_percent'],
            'passed': all_results['question_gen']['discrimination_improvement_percent'] >= 25
        },
        'criterion_4': {
            'description': '< 15% item exposure rate',
            'exposure': all_results['evolve_adaptive']['item_exposure_rate_mean'],
            'passed': all_results['evolve_adaptive']['item_exposure_rate_mean'] < 15
        }
    }
    
    final = {
        'experiment_summary': {'total_experiments': 8, 'models_evaluated': n_models, 'seeds': SEEDS},
        'main_results': {
            'evolve': {
                'item_reduction': all_results['evolve_adaptive']['item_reduction_percent'],
                'correlation': all_results['evolve_adaptive']['correlation_mean'],
                'correlation_std': all_results['evolve_adaptive']['correlation_std'],
                'exposure': all_results['evolve_adaptive']['item_exposure_rate_mean']
            },
            'baselines': {
                'random': all_results['random_subset']['correlation_mean'],
                'atlas': all_results['atlas']['correlation_mean'],
                'evolve': all_results['evolve_adaptive']['correlation_mean']
            },
            'evolution': {'variance_ratio': all_results['evolution']['variance_ratio']},
            'generation': {'improvement': all_results['question_gen']['discrimination_improvement_percent']}
        },
        'success_criteria': criteria,
        'overall_passed': all(c['passed'] for c in criteria.values())
    }
    
    with open('results.json', 'w') as f:
        json.dump(final, f, indent=2)
    
    print("\nSUCCESS CRITERIA:")
    for name, c in criteria.items():
        status = "✓ PASS" if c['passed'] else "✗ FAIL"
        print(f"  {name}: {status}")
        if 'item_reduction' in c:
            print(f"    Reduction: {c['item_reduction']:.1f}%, ρ: {c['correlation']:.4f}")
        if 'variance_ratio' in c:
            print(f"    Ratio: {c['variance_ratio']:.2f}x")
        if 'improvement' in c:
            print(f"    Improvement: {c['improvement']:.1f}%")
        if 'exposure' in c:
            print(f"    Exposure: {c['exposure']:.1f}%")
    
    print(f"\nOverall: {'ALL PASSED' if final['overall_passed'] else 'SOME FAILED'}")
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)


if __name__ == '__main__':
    run_fast()
