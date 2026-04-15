"""
Phase 1: Pilot Validation
Validates theoretical predictions on tractable problems.
"""
import numpy as np
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from models import BernoulliDistribution, GaussianDistribution
from samplers import GibbsWithGradients, ALBP, ABSampler
from metrics import compute_ess_spectral, compute_summary_statistics


def test_bernoulli_optimal_rate(dim=100, n_steps=5000, seed=42):
    """
    Reproduce Sun et al. 2022: Test optimal acceptance rate on Bernoulli.
    Theory predicts optimal rate is 0.574.
    """
    print(f"\n=== Bernoulli Optimal Rate Test (d={dim}) ===")
    rng = np.random.default_rng(seed)
    
    # Create Bernoulli problem
    probs = rng.uniform(0.3, 0.7, dim)
    model = BernoulliDistribution(probs)
    
    # Test different step sizes to find optimal acceptance rate
    step_sizes = np.logspace(-2, 0, 15)  # 0.01 to 1.0
    results = []
    
    for sigma in step_sizes:
        sampler = GibbsWithGradients(model, step_size=sigma, balancing='barker', seed=seed)
        x_init = (rng.random(dim) < 0.5).astype(float)
        
        start = time.time()
        samples = sampler.sample(x_init, n_steps, warmup=1000)
        runtime = time.time() - start
        
        # Compute metrics
        ess_per_dim, mean_ess = compute_ess_spectral(samples)
        accept_rate = np.mean(sampler.acceptance_history[1000:])
        avg_jump = np.mean(sampler.jump_distance_history[1000:])
        
        results.append({
            'sigma': float(sigma),
            'accept_rate': float(accept_rate),
            'ess': float(mean_ess),
            'ess_per_sec': float(mean_ess / runtime) if runtime > 0 else 0,
            'avg_jump': float(avg_jump)
        })
        
        print(f"  sigma={sigma:.4f}: accept={accept_rate:.3f}, ESS={mean_ess:.1f}, jump={avg_jump:.3f}")
    
    # Find optimal
    best_idx = np.argmax([r['ess'] for r in results])
    optimal = results[best_idx]
    
    print(f"\n  Optimal: sigma={optimal['sigma']:.4f}, accept_rate={optimal['accept_rate']:.3f}")
    print(f"  Theoretical optimal rate: 0.574")
    print(f"  Difference from theory: {abs(optimal['accept_rate'] - 0.574):.3f}")
    
    return {
        'test': 'bernoulli_optimal_rate',
        'dim': dim,
        'optimal_accept_rate': optimal['accept_rate'],
        'optimal_sigma': optimal['sigma'],
        'target_rate': 0.574,
        'difference': abs(optimal['accept_rate'] - 0.574),
        'within_tolerance': abs(optimal['accept_rate'] - 0.574) < 0.05,
        'all_results': results
    }


def test_albp_convergence(dim=100, n_steps=3000, seed=42):
    """
    Test ALBP adaptation converges to target rate.
    """
    print(f"\n=== ALBP Convergence Test (d={dim}) ===")
    rng = np.random.default_rng(seed)
    
    probs = rng.uniform(0.3, 0.7, dim)
    model = BernoulliDistribution(probs)
    
    sampler = ALBP(model, target_rate=0.574, eta_0=0.1, tau=1000, seed=seed)
    x_init = (rng.random(dim) < 0.5).astype(float)
    
    start = time.time()
    samples = sampler.sample(x_init, n_steps, warmup=0)
    runtime = time.time() - start
    
    # Analyze adaptation
    # Split into windows and compute acceptance rate per window
    window_size = 500
    n_windows = n_steps // window_size
    window_rates = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        rate = np.mean(sampler.acceptance_history[start_idx:end_idx])
        window_rates.append(rate)
    
    final_rate = np.mean(sampler.acceptance_history[-window_size:])
    final_sigma = sampler.sigma_history[-1]
    
    print(f"  Final acceptance rate: {final_rate:.3f}")
    print(f"  Target rate: 0.574")
    print(f"  Final sigma: {final_sigma:.4f}")
    print(f"  Rate trajectory: {[f'{r:.3f}' for r in window_rates]}")
    
    return {
        'test': 'albp_convergence',
        'dim': dim,
        'final_accept_rate': float(final_rate),
        'target_rate': 0.574,
        'difference': float(abs(final_rate - 0.574)),
        'within_tolerance': bool(abs(final_rate - 0.574) < 0.05),
        'final_sigma': float(final_sigma),
        'rate_trajectory': [float(r) for r in window_rates],
        'sigma_trajectory': [float(s) for s in sampler.sigma_history[::100]],
        'runtime': runtime
    }


def test_albp_vs_ab_sampler(dim=100, n_steps=3000, seed=42):
    """
    Compare ALBP vs AB-sampler on Gaussian distribution.
    """
    print(f"\n=== ALBP vs AB-Sampler Test (d={dim}) ===")
    rng = np.random.default_rng(seed)
    
    # Create Gaussian problem
    A = np.eye(dim) * 2 + rng.standard_normal((dim, dim)) * 0.1
    A = (A + A.T) / 2
    b = rng.standard_normal(dim)
    model = GaussianDistribution(A, b)
    
    results = {}
    
    # ALBP
    print("  Running ALBP...")
    sampler_albp = ALBP(model, target_rate=0.574, seed=seed)
    x_init = (rng.random(dim) < 0.5).astype(float)
    start = time.time()
    samples_albp = sampler_albp.sample(x_init, n_steps, warmup=1000)
    runtime_albp = time.time() - start
    
    _, ess_albp = compute_ess_spectral(samples_albp)
    final_accept_albp = np.mean(sampler_albp.acceptance_history[-500:])
    final_sigma_albp = sampler_albp.sigma_history[-1]
    
    results['albp'] = {
        'ess': float(ess_albp),
        'ess_per_sec': float(ess_albp / runtime_albp),
        'final_accept_rate': float(final_accept_albp),
        'final_sigma': float(final_sigma_albp),
        'runtime': runtime_albp
    }
    print(f"    ESS={ess_albp:.1f}, accept={final_accept_albp:.3f}, sigma={final_sigma_albp:.4f}")
    
    # AB-sampler
    print("  Running AB-sampler...")
    sampler_ab = ABSampler(model, sigma_init=0.1, seed=seed)
    x_init = (rng.random(dim) < 0.5).astype(float)
    start = time.time()
    samples_ab = sampler_ab.sample(x_init, n_steps, warmup=1000)
    runtime_ab = time.time() - start
    
    _, ess_ab = compute_ess_spectral(samples_ab)
    final_accept_ab = np.mean(sampler_ab.acceptance_history[-500:])
    final_sigma_ab = sampler_ab.sigma_history[-1]
    avg_jump_ab = np.mean(sampler_ab.jump_distance_history[-500:])
    
    results['ab_sampler'] = {
        'ess': float(ess_ab),
        'ess_per_sec': float(ess_ab / runtime_ab),
        'final_accept_rate': float(final_accept_ab),
        'final_sigma': float(final_sigma_ab),
        'avg_jump': float(avg_jump_ab),
        'runtime': runtime_ab
    }
    print(f"    ESS={ess_ab:.1f}, accept={final_accept_ab:.3f}, sigma={final_sigma_ab:.4f}, jump={avg_jump_ab:.3f}")
    
    # Compare
    ess_ratio = ess_ab / ess_albp if ess_albp > 0 else float('inf')
    print(f"\n  ESS ratio (AB/ALBP): {ess_ratio:.3f}")
    
    results['comparison'] = {
        'ess_ratio_ab_to_albp': float(ess_ratio),
        'albp_better': ess_albp > ess_ab
    }
    
    return {
        'test': 'albp_vs_ab',
        'dim': dim,
        'results': results
    }


def main():
    print("=" * 60)
    print("PHASE 1: PILOT VALIDATION")
    print("=" * 60)
    
    all_results = {}
    
    # Test 1: Bernoulli optimal rate
    result1 = test_bernoulli_optimal_rate(dim=100, n_steps=5000, seed=42)
    all_results['bernoulli_optimal_rate'] = result1
    
    # Test 2: ALBP convergence
    result2 = test_albp_convergence(dim=100, n_steps=3000, seed=42)
    all_results['albp_convergence'] = result2
    
    # Test 3: ALBP vs AB-sampler
    result3 = test_albp_vs_ab_sampler(dim=100, n_steps=3000, seed=42)
    all_results['albp_vs_ab'] = result3
    
    # Summary
    print("\n" + "=" * 60)
    print("PILOT VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"\n1. Bernoulli Optimal Rate:")
    print(f"   Found: {result1['optimal_accept_rate']:.3f}, Target: 0.574")
    print(f"   Within ±0.05 tolerance: {result1['within_tolerance']}")
    
    print(f"\n2. ALBP Convergence:")
    print(f"   Final rate: {result2['final_accept_rate']:.3f}, Target: 0.574")
    print(f"   Within ±0.05 tolerance: {bool(result2['within_tolerance'])}")
    
    print(f"\n3. ALBP vs AB-sampler:")
    print(f"   ALBP ESS: {result3['results']['albp']['ess']:.1f}")
    print(f"   AB-sampler ESS: {result3['results']['ab_sampler']['ess']:.1f}")
    print(f"   Winner: {'ALBP' if result3['results']['comparison']['albp_better'] else 'AB-sampler'}")
    
    # Save results
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), 'pilot_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating, np.bool_)) else x)
    
    print("\nResults saved to pilot_results.json")
    
    return all_results


if __name__ == '__main__':
    main()
