"""
Phase 3: Failure Mode Analysis
Characterize when each method fails.
"""
import numpy as np
import sys
import os
import json
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import LatticeIsingModel, create_multimodal_problem
from samplers import ALBP, ABSampler, ACS, GibbsWithGradients
from metrics import compute_ess_spectral


def phase_transition_stress_test(args):
    """Test at critical temperature J=0.44."""
    dim, seed = args
    rng = np.random.default_rng(seed)
    
    print(f"  Phase transition test: dim={dim}, seed={seed}")
    
    L = int(np.sqrt(dim))
    
    # Critical temperature
    problem_critical = LatticeIsingModel(L, J_coupling=0.44, seed=seed)
    x_init = (rng.random(dim) < 0.5).astype(float)
    
    # Off-critical (below)
    problem_below = LatticeIsingModel(L, J_coupling=0.2, seed=seed)
    
    results = {'dim': dim, 'seed': seed, 'tests': {}}
    
    for label, problem in [('critical', problem_critical), ('below', problem_below)]:
        results['tests'][label] = {}
        
        for method_name, sampler_class, kwargs in [
            ('albp', ALBP, {'target_rate': 0.574}),
            ('ab_sampler', ABSampler, {'sigma_init': 0.1}),
            ('acs', ACS, {'sigma_min': 0.05, 'sigma_max': 0.5})
        ]:
            sampler = sampler_class(problem, seed=seed, **kwargs)
            samples = sampler.sample(x_init.copy(), 5000, warmup=2000)
            
            _, ess = compute_ess_spectral(samples)
            
            # Track acceptance rate trajectory
            accept_traj = []
            window = 100
            for i in range(0, len(sampler.acceptance_history), window):
                accept_traj.append(np.mean(sampler.acceptance_history[i:i+window]))
            
            results['tests'][label][method_name] = {
                'ess': float(ess),
                'final_accept': float(np.mean(sampler.acceptance_history[-500:])),
                'accept_trajectory': [float(a) for a in accept_traj[:20]],  # First 20 windows
                'sigma_final': float(sampler.sigma_history[-1]) if hasattr(sampler, 'sigma_history') else None
            }
    
    # Compute degradation at critical
    for method_name in ['albp', 'ab_sampler', 'acs']:
        ess_critical = results['tests']['critical'][method_name]['ess']
        ess_below = results['tests']['below'][method_name]['ess']
        degradation = (ess_below - ess_critical) / ess_below if ess_below > 0 else 0
        results['tests']['critical'][method_name]['degradation_pct'] = float(degradation * 100)
    
    return results


def preconvergence_stability_test(args):
    """Test adaptation stability with different initializations."""
    dim, init_type, seed = args
    rng = np.random.default_rng(seed)
    
    print(f"  Stability test: dim={dim}, init={init_type}, seed={seed}")
    
    L = int(np.sqrt(dim))
    problem = LatticeIsingModel(L, J_coupling=0.2, seed=seed)
    
    # Different initializations
    if init_type == 'zeros':
        x_init = np.zeros(dim)
    elif init_type == 'random':
        x_init = (rng.random(dim) < 0.5).astype(float)
    elif init_type == 'high_energy':
        # Start far from typical set (all ones when ground state is mixed)
        x_init = np.ones(dim)
    else:
        x_init = (rng.random(dim) < 0.5).astype(float)
    
    results = {'dim': dim, 'init_type': init_type, 'seed': seed, 'methods': {}}
    
    for method_name, sampler_class, kwargs in [
        ('albp', ALBP, {'target_rate': 0.574}),
        ('ab_sampler', ABSampler, {'sigma_init': 0.1})
    ]:
        sampler = sampler_class(problem, seed=seed, **kwargs)
        
        # Run and track early adaptation
        sampler.sample(x_init.copy(), 3000, warmup=0)
        
        # Compute variance in early phase (first 1000 iterations)
        early_phase = 1000
        if hasattr(sampler, 'sigma_history'):
            early_sigma = sampler.sigma_history[:early_phase]
        elif hasattr(sampler, 'R_history'):
            early_sigma = [np.exp(r) for r in sampler.R_history[:early_phase]]
        else:
            early_sigma = []
        
        if len(early_sigma) > 100:
            early_var = np.var(early_sigma[100:])  # Skip first 100
            late_var = np.var(sampler.sigma_history[-1000:]) if len(sampler.sigma_history) > 1000 else 0
        else:
            early_var = 0
            late_var = 0
        
        results['methods'][method_name] = {
            'early_sigma_variance': float(early_var),
            'late_sigma_variance': float(late_var),
            'sigma_trajectory': [float(s) for s in sampler.sigma_history[::100]]
        }
    
    # Compute stability ratio
    albp_early = results['methods']['albp']['early_sigma_variance']
    ab_early = results['methods']['ab_sampler']['early_sigma_variance']
    if ab_early > 0:
        results['variance_ratio_ab_to_albp'] = float(ab_early / albp_early)
    else:
        results['variance_ratio_ab_to_albp'] = None
    
    return results


def multimodality_test(args):
    """Test mode discovery on multimodal targets."""
    dim, n_modes, seed = args
    rng = np.random.default_rng(seed)
    
    print(f"  Multimodality test: dim={dim}, modes={n_modes}, seed={seed}")
    
    problem = create_multimodal_problem(dim, n_modes, separation=0.5, seed=seed)
    x_init = (rng.random(dim) < 0.5).astype(float)
    
    results = {'dim': dim, 'n_modes': n_modes, 'seed': seed, 'methods': {}}
    
    for method_name, sampler_class, kwargs in [
        ('albp', ALBP, {'target_rate': 0.574}),
        ('ab_sampler', ABSampler, {'sigma_init': 0.1}),
        ('acs', ACS, {'sigma_min': 0.05, 'sigma_max': 0.5})
    ]:
        sampler = sampler_class(problem, seed=seed, **kwargs)
        samples = sampler.sample(x_init.copy(), 5000, warmup=2000)
        
        _, ess = compute_ess_spectral(samples)
        
        # Count mode discoveries
        # Simple heuristic: cluster by nearest mode
        modes_discovered = 0
        for mode_idx, mode_center in enumerate(problem.means):
            distances = np.linalg.norm(samples - mode_center, axis=1)
            if np.min(distances) < 0.3 * np.sqrt(dim):  # Within reasonable distance
                modes_discovered += 1
        
        results['methods'][method_name] = {
            'ess': float(ess),
            'modes_discovered': modes_discovered,
            'total_modes': n_modes,
            'discovery_rate': float(modes_discovered / n_modes)
        }
    
    return results


def main():
    print("=" * 70)
    print("PHASE 3: FAILURE MODE ANALYSIS")
    print("=" * 70)
    
    all_results = {}
    
    # Test 1: Phase transition stress test
    print("\n1. Phase Transition Stress Test (critical temperature)")
    print("-" * 50)
    phase_args = [(100, 42 + i) for i in range(5)]  # 5 seeds
    
    phase_results = []
    for args in phase_args:
        phase_results.append(phase_transition_stress_test(args))
    
    all_results['phase_transition'] = phase_results
    
    # Aggregate phase transition results
    print("\n  Phase Transition Summary:")
    for method in ['albp', 'ab_sampler', 'acs']:
        degradations = [r['tests']['critical'][method]['degradation_pct'] 
                       for r in phase_results if method in r['tests']['critical']]
        if degradations:
            print(f"    {method:12s}: degradation = {np.mean(degradations):.1f}% ± {np.std(degradations):.1f}%")
    
    # Test 2: Pre-convergence stability
    print("\n2. Pre-Convergence Stability Test")
    print("-" * 50)
    stability_args = [(100, init, 42 + i) 
                      for init in ['zeros', 'random', 'high_energy']
                      for i in range(3)]  # 3 seeds per init
    
    stability_results = []
    for args in stability_args:
        stability_results.append(preconvergence_stability_test(args))
    
    all_results['stability'] = stability_results
    
    # Aggregate stability results
    print("\n  Stability Summary:")
    for init_type in ['zeros', 'random', 'high_energy']:
        ratios = [r['variance_ratio_ab_to_albp'] for r in stability_results 
                 if r['init_type'] == init_type and r['variance_ratio_ab_to_albp'] is not None]
        if ratios:
            print(f"    {init_type:12s}: AB/ALBP variance ratio = {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
    
    # Test 3: Multimodality
    print("\n3. Multimodality Challenge")
    print("-" * 50)
    multimodal_args = [(50, n_modes, 42 + i) 
                       for n_modes in [2, 4, 6]
                       for i in range(3)]  # 3 seeds per mode count
    
    multimodal_results = []
    for args in multimodal_args:
        multimodal_results.append(multimodality_test(args))
    
    all_results['multimodality'] = multimodal_results
    
    # Aggregate multimodal results
    print("\n  Multimodality Summary:")
    for n_modes in [2, 4, 6]:
        for method in ['albp', 'ab_sampler', 'acs']:
            discoveries = [r['methods'][method]['discovery_rate'] 
                          for r in multimodal_results 
                          if r['n_modes'] == n_modes and method in r['methods']]
            if discoveries:
                print(f"    {n_modes} modes, {method:12s}: discovery rate = {np.mean(discoveries):.2f} ± {np.std(discoveries):.2f}")
    
    # Save results
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'phase3_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, 
                  default=lambda x: float(x) if isinstance(x, (np.integer, np.floating, np.bool_)) else x)
    
    print(f"\nResults saved to {output_dir}/phase3_results.json")
    
    return all_results


if __name__ == '__main__':
    main()
