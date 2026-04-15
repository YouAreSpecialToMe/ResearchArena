"""Statistical testing and success criteria evaluation for FlipBench."""

import json
import os
import sys
import numpy as np
from scipy import stats
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_DIR = os.path.join(BASE_DIR, 'data')

DOMAINS = ['propositional_logic', 'arithmetic_reasoning',
           'relational_reasoning', 'function_computation']

# Models evaluated on seed_42
SINGLE_SEED_MODELS = ['phi35', 'llama31_8b', 'qwen25_7b', 'deepseek_r1_7b', 'qwen25_32b']
# Models evaluated on all 3 seeds
MULTI_SEED_MODELS = ['llama31_8b', 'deepseek_r1_7b']


def load_parsed_results(model_short, seed='seed_42', cot=False):
    suffix = '_cot' if cot else ''
    path = os.path.join(RESULTS_DIR, 'parsed', f'{model_short}{suffix}_{seed}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_raw_results(model_short, seed='seed_42', cot=False):
    suffix = '_cot' if cot else ''
    path = os.path.join(RESULTS_DIR, 'raw', f'{model_short}{suffix}_{seed}.jsonl')
    if not os.path.exists(path):
        return None
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def mcnemar_test(fwd_correct, bwd_correct):
    """McNemar's test for paired binary data.

    Args:
        fwd_correct: list of bool for forward direction
        bwd_correct: list of bool for backward direction

    Returns:
        test_statistic, p_value
    """
    # Count discordant pairs
    b = sum(1 for f, bk in zip(fwd_correct, bwd_correct) if f and not bk)  # fwd correct, bwd wrong
    c = sum(1 for f, bk in zip(fwd_correct, bwd_correct) if not f and bk)  # fwd wrong, bwd correct

    if b + c == 0:
        return 0, 1.0

    # One-sided: test if b > c (forward better than backward)
    # Use exact binomial test
    p_value = stats.binomtest(b, b + c, 0.5, alternative='greater').pvalue
    test_stat = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0

    return float(test_stat), float(p_value)


def run_all_tests():
    results = {}

    # 1. McNemar's test for DRG > 0 per model × domain
    print("=== McNemar's Test: DRG > 0 ===")
    mcnemar_results = {}
    n_tests = len(SINGLE_SEED_MODELS) * len(DOMAINS)
    bonferroni = n_tests

    for model in SINGLE_SEED_MODELS:
        raw = load_raw_results(model, 'seed_42')
        if raw is None:
            print(f"  Skipping {model} (no raw results)")
            continue

        mcnemar_results[model] = {}
        # Group by matched pair, also store domain
        pairs = defaultdict(dict)
        pair_domain = {}
        for r in raw:
            pairs[r['matched_pair_id']][r['direction']] = r['correct']
            pair_domain[r['matched_pair_id']] = r['domain']

        for domain in DOMAINS:
            domain_pairs = {k: v for k, v in pairs.items() if pair_domain.get(k) == domain}
            fwd_correct = [v.get('forward', False) for v in domain_pairs.values()]
            bwd_correct = [v.get('backward', False) for v in domain_pairs.values()]

            stat, p_raw = mcnemar_test(fwd_correct, bwd_correct)
            p_corrected = min(p_raw * bonferroni, 1.0)

            parsed = load_parsed_results(model, 'seed_42')
            drg = parsed[domain]['drg'] if parsed and domain in parsed else None

            mcnemar_results[model][domain] = {
                'test_statistic': round(stat, 4),
                'p_value_raw': round(p_raw, 6),
                'p_value_corrected': round(p_corrected, 6),
                'significant_at_005': p_corrected < 0.05,
                'drg': drg,
                'n_pairs': len(fwd_correct)
            }

            sig = "***" if p_corrected < 0.001 else "**" if p_corrected < 0.01 else "*" if p_corrected < 0.05 else "ns"
            print(f"  {model:20s} × {domain:25s}: DRG={drg:.3f}, p={p_corrected:.4f} {sig}")

    results['mcnemar_tests'] = mcnemar_results

    # 2. Success criterion 1: DRG > 5pp in at least 3/4 domains across majority of models
    print("\n=== Success Criterion 1: DRG > 5pp in ≥3/4 domains ===")
    criterion1 = {}
    for model in SINGLE_SEED_MODELS:
        parsed = load_parsed_results(model, 'seed_42')
        if not parsed:
            continue
        domains_above_5pp = sum(1 for d in DOMAINS if parsed.get(d, {}).get('drg', 0) > 0.05)
        criterion1[model] = {
            'domains_above_5pp': domains_above_5pp,
            'passes': domains_above_5pp >= 3
        }
        status = "PASS" if domains_above_5pp >= 3 else "FAIL"
        print(f"  {model:20s}: {domains_above_5pp}/4 domains > 5pp [{status}]")

    models_passing = sum(1 for v in criterion1.values() if v['passes'])
    criterion1['summary'] = {
        'models_passing': models_passing,
        'total_models': len(criterion1) - 1,
        'majority_passes': models_passing > (len(criterion1) - 1) / 2
    }
    results['criterion1_drg_above_5pp'] = criterion1

    # 3. Success criterion 2: DRG increases with difficulty in ≥2 domains
    print("\n=== Success Criterion 2: DRG increases with difficulty ===")
    criterion2 = {}
    for domain in DOMAINS:
        drg_by_diff = {1: [], 2: [], 3: []}
        for model in SINGLE_SEED_MODELS:
            parsed = load_parsed_results(model, 'seed_42')
            if not parsed or domain not in parsed:
                continue
            for diff in [1, 2, 3]:
                key = f'difficulty_{diff}'
                if key in parsed[domain]:
                    drg_by_diff[diff].append(parsed[domain][key]['drg'])

        # Test trend: DRG increases with difficulty
        if drg_by_diff[1] and drg_by_diff[3]:
            mean_easy = np.mean(drg_by_diff[1])
            mean_hard = np.mean(drg_by_diff[3])
            # Wilcoxon signed-rank test on paired differences
            diffs = [h - e for h, e in zip(drg_by_diff[3], drg_by_diff[1])]
            if len(diffs) >= 3:
                try:
                    stat, p_val = stats.wilcoxon(diffs, alternative='greater')
                except:
                    stat, p_val = 0, 1.0
            else:
                stat, p_val = 0, 1.0

            amplification = mean_hard / mean_easy if mean_easy != 0 else float('inf')
            criterion2[domain] = {
                'mean_drg_easy': round(float(mean_easy), 4),
                'mean_drg_hard': round(float(mean_hard), 4),
                'amplification_factor': round(float(amplification), 4),
                'increases': mean_hard > mean_easy,
                'wilcoxon_p': round(float(p_val), 4)
            }
            trend = "↑" if mean_hard > mean_easy else "↓"
            print(f"  {domain:25s}: easy={mean_easy:.3f} hard={mean_hard:.3f} {trend} amp={amplification:.2f}x p={p_val:.4f}")

    domains_increasing = sum(1 for v in criterion2.values() if v.get('increases', False))
    criterion2['summary'] = {
        'domains_increasing': domains_increasing,
        'passes': domains_increasing >= 2
    }
    results['criterion2_difficulty_scaling'] = criterion2

    # 4. Success criterion 3: Meaningful differences between model families
    print("\n=== Success Criterion 3: Standard vs Reasoning-optimized ===")
    criterion3 = {}
    # Compare Qwen2.5-7B-Instruct vs DeepSeek-R1-Distill-Qwen-7B
    standard = load_parsed_results('qwen25_7b', 'seed_42')
    reasoning = load_parsed_results('deepseek_r1_7b', 'seed_42')

    if standard and reasoning:
        for domain in DOMAINS:
            std_drg = standard.get(domain, {}).get('drg', 0)
            reas_drg = reasoning.get(domain, {}).get('drg', 0)
            diff = std_drg - reas_drg
            criterion3[domain] = {
                'standard_drg': round(std_drg, 4),
                'reasoning_drg': round(reas_drg, 4),
                'difference': round(diff, 4),
                'reasoning_lower': reas_drg < std_drg
            }
            arrow = "↓" if reas_drg < std_drg else "↑"
            print(f"  {domain:25s}: standard={std_drg:.3f} reasoning={reas_drg:.3f} {arrow} diff={diff:.3f}")
    results['criterion3_model_families'] = criterion3

    # 5. Consistency rate < min(FA, BA)
    print("\n=== Secondary Criterion: CR < min(FA, BA) ===")
    criterion_cr = {}
    for model in SINGLE_SEED_MODELS:
        parsed = load_parsed_results(model, 'seed_42')
        if not parsed:
            continue
        criterion_cr[model] = {}
        for domain in DOMAINS:
            d = parsed.get(domain, {})
            fa = d.get('forward_accuracy', 0)
            ba = d.get('backward_accuracy', 0)
            cr = d.get('consistency_rate', 0)
            min_acc = min(fa, ba)
            criterion_cr[model][domain] = {
                'cr': round(cr, 4),
                'min_fa_ba': round(min_acc, 4),
                'cr_below_min': cr < min_acc
            }

    results['criterion_consistency'] = criterion_cr

    # 6. Cross-seed stability
    print("\n=== Cross-seed Stability ===")
    stability = {}
    for model in MULTI_SEED_MODELS:
        stability[model] = {}
        seed_results = []
        for seed in ['seed_42', 'seed_123', 'seed_456']:
            parsed = load_parsed_results(model, seed)
            if parsed:
                seed_results.append(parsed)

        if len(seed_results) >= 2:
            for domain in DOMAINS:
                drgs = [sr[domain]['drg'] for sr in seed_results if domain in sr]
                if drgs:
                    mean_drg = float(np.mean(drgs))
                    std_drg = float(np.std(drgs))
                    stability[model][domain] = {
                        'drg_values': [round(d, 4) for d in drgs],
                        'mean': round(mean_drg, 4),
                        'std': round(std_drg, 4),
                        'stable': std_drg < 0.03
                    }
                    status = "stable" if std_drg < 0.03 else "variable"
                    print(f"  {model:20s} × {domain:25s}: DRG={mean_drg:.3f}±{std_drg:.3f} [{status}]")

    results['cross_seed_stability'] = stability

    # Save all
    outpath = os.path.join(RESULTS_DIR, 'aggregated', 'statistical_tests.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nAll statistical results saved to {outpath}")

    return results


if __name__ == '__main__':
    run_all_tests()
