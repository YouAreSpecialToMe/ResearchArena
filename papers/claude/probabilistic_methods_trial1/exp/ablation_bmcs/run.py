"""Ablation studies for BMCS components."""
import sys
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.targets import GaussianTarget, MixtureGaussianTarget, BayesianLogisticRegression
from src.samplers import rwmh_sample
from src.confidence_sequences import BatchMeansCS


class BatchMeansCS_FixedBatch(BatchMeansCS):
    """BMCS with fixed batch size (ablation)."""
    def __init__(self, alpha=0.05, fixed_batch_size=None, exponent=0.5):
        super().__init__(alpha=alpha)
        self.fixed_batch_size = fixed_batch_size
        self.exponent = exponent

    def _get_batch_size(self, k):
        if self.fixed_batch_size is not None:
            return self.fixed_batch_size
        # n^exponent based on running count
        n_est = max(100, self.running_count + 100)
        return max(10, int(n_est ** self.exponent))


class BatchMeansCS_NoCorrection(BatchMeansCS):
    """BMCS without dependence correction (ablation)."""
    def get_cs(self):
        k = len(self.batch_means)
        if k < 2:
            return self.running_sum / max(1, self.running_count), np.inf

        bm_arr = np.array(self.batch_means)
        center = np.mean(bm_arr)
        V_k = np.var(bm_arr, ddof=1)

        log_term = np.log(max(1.0, np.log(2 * k))) + np.log(2.0 / self.alpha)
        B = np.max(np.abs(bm_arr - center)) + np.std(bm_arr)
        half_width = np.sqrt(2 * V_k * log_term / k) + B * log_term / (3 * k)
        # No dependence correction
        return center, half_width


class HoeffdingCS:
    """Hoeffding-style CS (no variance adaptation)."""
    def __init__(self, alpha=0.05, bound=10.0):
        self.alpha = alpha
        self.bound = bound
        self.values = []

    def update(self, values):
        if np.isscalar(values):
            self.values.append(values)
        else:
            self.values.extend(values)

    def get_cs(self):
        n = len(self.values)
        if n < 2:
            return np.mean(self.values) if self.values else 0.0, np.inf
        center = np.mean(self.values)
        log_term = np.log(max(1.0, np.log(2 * n))) + np.log(2.0 / self.alpha)
        half_width = self.bound * np.sqrt(2 * log_term / n)
        return center, half_width


def run_ablation_replicate(target, d, seed, rep_idx, n_samples, alpha, ablation_type):
    rng = np.random.default_rng(seed * 10000 + rep_idx)
    true_mean = 0.0

    h = 2.38 / np.sqrt(d) if d >= 2 else 1.0
    x0 = rng.standard_normal(d)
    chain, _ = rwmh_sample(target, x0, n_samples, h, burn_in=1000, rng=rng)
    f_values = chain[:, 0]

    monitor_times = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    monitor_times = [t for t in monitor_times if t <= n_samples]

    results = {}

    if ablation_type == 'batch_size':
        variants = {
            'adaptive': BatchMeansCS(alpha=alpha),
            'sqrt_n': BatchMeansCS_FixedBatch(alpha=alpha, exponent=0.5),
            'cbrt_n': BatchMeansCS_FixedBatch(alpha=alpha, exponent=1/3),
        }
    elif ablation_type == 'correction':
        variants = {
            'with_correction': BatchMeansCS(alpha=alpha),
            'no_correction': BatchMeansCS_NoCorrection(alpha=alpha),
        }
    elif ablation_type == 'cs_type':
        variants = {
            'empirical_bernstein': BatchMeansCS(alpha=alpha),
            'hoeffding': HoeffdingCS(alpha=alpha),
        }
    elif ablation_type == 'burnin':
        # Different burn-in fractions
        variants = {}
        for frac_name, frac in [('10pct', 0.1), ('0pct', 0.0), ('20pct', 0.2)]:
            variants[frac_name] = {'frac': frac}

    if ablation_type != 'burnin':
        for name, cs in variants.items():
            prev_t = 0
            covers = {}
            widths = {}
            for t in monitor_times:
                cs.update(f_values[prev_t:t])
                c, hw = cs.get_cs()
                covers[t] = bool(abs(c - true_mean) <= hw)
                widths[t] = float(hw)
                prev_t = t
            results[name] = {'covers': covers, 'widths': widths}
    else:
        for name, cfg in variants.items():
            frac = cfg['frac']
            burn = int(n_samples * frac)
            f_post = f_values[burn:]
            cs = BatchMeansCS(alpha=alpha)
            covers = {}
            widths = {}
            adj_monitor = [t for t in monitor_times if t <= len(f_post)]
            prev_t = 0
            for t in adj_monitor:
                cs.update(f_post[prev_t:t])
                c, hw = cs.get_cs()
                covers[t] = bool(abs(c - true_mean) <= hw)
                widths[t] = float(hw)
                prev_t = t
            results[name] = {'covers': covers, 'widths': widths}

    return results


def run_ablation(ablation_type, target_name, target, d, seeds, n_reps, n_samples, alpha):
    all_results = []
    for seed in seeds:
        results = Parallel(n_jobs=2, backend='loky')(
            delayed(run_ablation_replicate)(
                target, d, seed, rep, n_samples, alpha, ablation_type
            ) for rep in range(n_reps)
        )
        all_results.extend(results)

    # Aggregate
    summary = {}
    variant_names = list(all_results[0].keys())
    for name in variant_names:
        all_covers = []
        all_widths = []
        for r in all_results:
            if name in r:
                for t, c in r[name]['covers'].items():
                    all_covers.append(c)
                # Final width
                max_t = max(r[name]['widths'].keys())
                all_widths.append(r[name]['widths'][max_t])

        # Time-uniform coverage
        covers_by_time = {}
        for r in all_results:
            if name in r:
                for t, c in r[name]['covers'].items():
                    covers_by_time.setdefault(t, []).append(c)
        per_time_cov = {str(t): float(np.mean(v)) for t, v in covers_by_time.items()}
        time_uniform = min(per_time_cov.values()) if per_time_cov else 0.0

        summary[name] = {
            'time_uniform_coverage': time_uniform,
            'per_time_coverage': per_time_cov,
            'mean_final_width': float(np.mean(all_widths)),
            'std_final_width': float(np.std(all_widths)),
        }

    return summary


def main():
    start_time = time.time()
    seeds = [42, 123, 456]
    n_reps = 200
    alpha = 0.05

    results = {}

    # A1: Batch size strategy
    print("Ablation A1: Batch size strategy...", flush=True)
    target = GaussianTarget(10)
    results['A1_batch_size'] = run_ablation(
        'batch_size', 'gaussian_d10', target, 10, seeds, n_reps, 50000, alpha
    )
    print(f"  Done. Results: {json.dumps({k: v['time_uniform_coverage'] for k, v in results['A1_batch_size'].items()})}")

    # A2: Dependence correction
    print("Ablation A2: Dependence correction...", flush=True)
    results['A2_correction_gauss'] = run_ablation(
        'correction', 'gaussian_d10', GaussianTarget(10), 10, seeds, n_reps, 50000, alpha
    )
    target_mm = MixtureGaussianTarget(5, 2.0)
    results['A2_correction_multimodal'] = run_ablation(
        'correction', 'mixture_d5', target_mm, 5, seeds, n_reps, 50000, alpha
    )
    print(f"  Done.")

    # A3: CS type
    print("Ablation A3: CS type...", flush=True)
    results['A3_cs_type'] = run_ablation(
        'cs_type', 'gaussian_d10', GaussianTarget(10), 10, seeds, n_reps, 50000, alpha
    )
    print(f"  Done.")

    # A4: Burn-in handling
    print("Ablation A4: Burn-in handling...", flush=True)
    results['A4_burnin'] = run_ablation(
        'burnin', 'gaussian_d10', GaussianTarget(10), 10, seeds, n_reps, 50000, alpha
    )
    print(f"  Done.")

    elapsed = time.time() - start_time
    output = {
        'experiment': 'ablation_bmcs',
        'config': {'seeds': seeds, 'n_reps': n_reps, 'alpha': alpha},
        'results': results,
        'runtime_minutes': elapsed / 60,
    }

    with open(os.path.join(os.path.dirname(__file__), 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nBMCS Ablations completed in {elapsed/60:.1f} minutes")
    for ablation_name, ablation_data in results.items():
        print(f"\n{ablation_name}:")
        for variant, metrics in ablation_data.items():
            print(f"  {variant}: coverage={metrics['time_uniform_coverage']:.3f}, "
                  f"width={metrics['mean_final_width']:.4f}")


if __name__ == '__main__':
    main()
