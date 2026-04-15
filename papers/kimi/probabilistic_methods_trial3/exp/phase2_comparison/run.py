"""
Phase 2: Head-to-Head Comparison
Systematic comparison of all adaptive methods on standardized problems.
"""
import numpy as np
import sys
import os
import json
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import LatticeIsingModel, RandomIsingModel
from samplers import GibbsWithGradients, ALBP, ABSampler, ACS, GridSearchHeuristic
from metrics import compute_ess_spectral, compute_cohens_d, mann_whitney_test


def run_single_experiment(args):
    """Run a single experiment (for parallel execution)."""
    problem_name, method_name, config = args
    
    # Unpack config
    dim = config['dim']
    n_steps = config.get('n_steps', 5000)
    warmup = config.get('warmup', 2000)
    seed = config['seed']
    problem_type = config['problem_type']
    problem_params = config.get('problem_params', {})
    
    rng = np.random.default_rng(seed)
    
    # Create problem
    if problem_type == 'lattice_ising':
        L = int(np.sqrt(dim))
        problem = LatticeIsingModel(L, problem_params['J'], seed=seed)
    elif problem_type == 'random_ising':
        problem = RandomIsingModel(dim, problem_params['J_mean'], 0.1, 
                                   problem_params.get('frustration', 0.0), seed=seed)
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    # Initialize
    x_init = (rng.random(dim) < 0.5).astype(float)
    
    # Run method
    start_time = time.time()
    
    if method_name == 'fixed_gwg':
        sigma = config.get('sigma', 0.1)
        sampler = GibbsWithGradients(problem, step_size=sigma, balancing='barker', seed=seed)
        samples = sampler.sample(x_init, n_steps, warmup)
        
    elif method_name == 'albp':
        sampler = ALBP(problem, target_rate=0.574, eta_0=0.1, tau=1000, seed=seed)
        samples = sampler.sample(x_init, n_steps, warmup)
        
    elif method_name == 'ab_sampler':
        sampler = ABSampler(problem, sigma_init=0.1, window_size=100, seed=seed)
        samples = sampler.sample(x_init, n_steps, warmup)
        
    elif method_name == 'acs':
        sampler = ACS(problem, sigma_min=0.05, sigma_max=0.5, cycle_length=1000, seed=seed)
        samples = sampler.sample(x_init, n_steps, warmup)
        
    elif method_name == 'grid_search':
        step_sizes = config.get('step_sizes', [0.01, 0.05, 0.1, 0.2, 0.5])
        pilot_steps = config.get('pilot_steps', 500)
        sampler = GridSearchHeuristic(problem, step_sizes, pilot_steps, seed=seed)
        sampler.run_pilots(x_init)
        samples = sampler.sample(x_init, n_steps, warmup)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    runtime = time.time() - start_time
    
    # Compute metrics
    _, ess = compute_ess_spectral(samples)
    
    result = {
        'problem': problem_name,
        'method': method_name,
        'seed': seed,
        'dim': dim,
        'ess': float(ess),
        'runtime': float(runtime),
        'ess_per_sec': float(ess / runtime) if runtime > 0 else 0,
        'n_samples': len(samples)
    }
    
    # Add method-specific metrics
    if hasattr(sampler, 'acceptance_history') and len(sampler.acceptance_history) > 0:
        result['final_accept_rate'] = float(np.mean(sampler.acceptance_history[-500:]))
    if hasattr(sampler, 'jump_distance_history') and len(sampler.jump_distance_history) > 0:
        result['avg_jump'] = float(np.mean(sampler.jump_distance_history[warmup:]))
    if hasattr(sampler, 'sigma_history') and len(sampler.sigma_history) > 0:
        result['final_sigma'] = float(sampler.sigma_history[-1])
    if hasattr(sampler, 'R_history') and len(sampler.R_history) > 0:
        result['final_R'] = float(sampler.R_history[-1])
    if hasattr(sampler, 'selected_sigma') and sampler.selected_sigma is not None:
        result['selected_sigma'] = float(sampler.selected_sigma)
    
    return result


def run_comparison_experiment(problem_configs, method_names, n_seeds=10, n_workers=2):
    """
    Run full comparison across problems, methods, and seeds.
    """
    all_configs = []
    
    # Create all experiment configurations
    for problem_name, problem_config in problem_configs.items():
        for method_name in method_names:
            for seed_offset in range(n_seeds):
                config = problem_config.copy()
                config['seed'] = 42 + seed_offset
                config['method'] = method_name
                all_configs.append((problem_name, method_name, config))
    
    print(f"Running {len(all_configs)} experiments total...")
    print(f"Problems: {list(problem_configs.keys())}")
    print(f"Methods: {method_names}")
    print(f"Seeds per config: {n_seeds}")
    print(f"Workers: {n_workers}")
    
    # Run in parallel
    results = []
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(run_single_experiment, all_configs)
    else:
        for config in all_configs:
            result = run_single_experiment(config)
            results.append(result)
    
    return results


def aggregate_results_by_config(results):
    """Aggregate results by problem and method."""
    from collections import defaultdict
    
    grouped = defaultdict(list)
    for r in results:
        key = (r['problem'], r['method'])
        grouped[key].append(r)
    
    aggregated = {}
    for key, group in grouped.items():
        problem, method = key
        ess_values = [r['ess'] for r in group]
        runtime_values = [r['runtime'] for r in group]
        
        aggregated[key] = {
            'problem': problem,
            'method': method,
            'n_seeds': len(group),
            'ess_mean': float(np.mean(ess_values)),
            'ess_std': float(np.std(ess_values, ddof=1)),
            'ess_min': float(np.min(ess_values)),
            'ess_max': float(np.max(ess_values)),
            'ess_cv': float(np.std(ess_values, ddof=1) / np.mean(ess_values)) if np.mean(ess_values) > 0 else 0,
            'runtime_mean': float(np.mean(runtime_values)),
            'runtime_std': float(np.std(runtime_values, ddof=1)),
        }
        
        # Add acceptance rate if available
        accept_rates = [r.get('final_accept_rate', None) for r in group if 'final_accept_rate' in r]
        if accept_rates:
            aggregated[key]['accept_rate_mean'] = float(np.mean(accept_rates))
            aggregated[key]['accept_rate_std'] = float(np.std(accept_rates, ddof=1))
    
    return aggregated


def compute_pairwise_comparisons(results):
    """Compute effect sizes and statistical tests between methods."""
    from collections import defaultdict
    
    # Group by problem
    by_problem = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_problem[r['problem']][r['method']].append(r['ess'])
    
    comparisons = []
    methods = ['albp', 'ab_sampler', 'acs', 'fixed_gwg', 'grid_search']
    
    for problem in by_problem:
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                if method1 in by_problem[problem] and method2 in by_problem[problem]:
                    ess1 = by_problem[problem][method1]
                    ess2 = by_problem[problem][method2]
                    
                    if len(ess1) > 0 and len(ess2) > 0:
                        cohens_d = compute_cohens_d(ess1, ess2)
                        _, pvalue = mann_whitney_test(ess1, ess2)
                        
                        comparisons.append({
                            'problem': problem,
                            'method1': method1,
                            'method2': method2,
                            'cohens_d': float(cohens_d),
                            'pvalue': float(pvalue),
                            'significant': pvalue < 0.05,
                            'meaningful': abs(cohens_d) > 0.5,
                            'winner': method1 if np.mean(ess1) > np.mean(ess2) else method2
                        })
    
    return comparisons


def main():
    print("=" * 70)
    print("PHASE 2: HEAD-TO-HEAD COMPARISON")
    print("=" * 70)
    
    # Define problem configurations
    problem_configs = {
        'lattice_100_below': {
            'dim': 100,
            'problem_type': 'lattice_ising',
            'problem_params': {'J': 0.2},
            'n_steps': 3000,
            'warmup': 1000
        },
        'lattice_100_critical': {
            'dim': 100,
            'problem_type': 'lattice_ising',
            'problem_params': {'J': 0.44},
            'n_steps': 5000,
            'warmup': 2000
        },
        'lattice_100_above': {
            'dim': 100,
            'problem_type': 'lattice_ising',
            'problem_params': {'J': 0.6},
            'n_steps': 3000,
            'warmup': 1000
        },
        'lattice_400_below': {
            'dim': 400,
            'problem_type': 'lattice_ising',
            'problem_params': {'J': 0.2},
            'n_steps': 3000,
            'warmup': 1000
        },
        'random_100_uniform': {
            'dim': 100,
            'problem_type': 'random_ising',
            'problem_params': {'J_mean': 0.2, 'frustration': 0.0},
            'n_steps': 3000,
            'warmup': 1000
        },
        'random_100_frustrated': {
            'dim': 100,
            'problem_type': 'random_ising',
            'problem_params': {'J_mean': 0.4, 'frustration': 0.5},
            'n_steps': 5000,
            'warmup': 2000
        }
    }
    
    # Methods to compare
    method_names = ['fixed_gwg', 'grid_search', 'albp', 'ab_sampler', 'acs']
    
    # Run experiments
    n_seeds = 10
    n_workers = 2  # Use 2 cores
    
    print(f"\nRunning comparison with {n_seeds} seeds per configuration...")
    print(f"Total experiments: {len(problem_configs) * len(method_names) * n_seeds}")
    
    start = time.time()
    results = run_comparison_experiment(problem_configs, method_names, n_seeds, n_workers)
    total_time = time.time() - start
    
    print(f"\nCompleted in {total_time:.1f} seconds")
    
    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results_by_config(results)
    comparisons = compute_pairwise_comparisons(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)
    
    problems = sorted(set(r['problem'] for r in results))
    methods = sorted(set(r['method'] for r in results))
    
    for problem in problems:
        print(f"\n{problem}:")
        for method in methods:
            key = (problem, method)
            if key in aggregated:
                agg = aggregated[key]
                print(f"  {method:15s}: ESS = {agg['ess_mean']:6.1f} ± {agg['ess_std']:5.1f} "
                      f"(CV={agg['ess_cv']:.2f}, n={agg['n_seeds']})")
    
    # Save results
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'phase2_results.json'), 'w') as f:
        json.dump({
            'raw_results': results,
            'aggregated': {f"{k[0]}_{k[1]}": v for k, v in aggregated.items()},
            'comparisons': comparisons,
            'metadata': {
                'n_problems': len(problem_configs),
                'n_methods': len(method_names),
                'n_seeds': n_seeds,
                'total_runtime': total_time
            }
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating, np.bool_)) else x)
    
    print(f"\nResults saved to {output_dir}/phase2_results.json")
    
    return results, aggregated, comparisons


if __name__ == '__main__':
    main()
