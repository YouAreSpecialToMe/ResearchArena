"""Statistical analysis and success criteria verification."""
import sys
import csv
import json
import math
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    def adjusted_rand_score(a, b):
        return 0.0
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from exp.shared.utils import *

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def analyze_idempotency():
    summary = []
    with open(RESULTS_DIR / "idempotency_summary.csv") as f:
        for row in csv.DictReader(f):
            summary.append(row)

    n_total = len(summary)
    n_strongly = sum(1 for s in summary if s['classification'] == 'strongly_idempotent')
    n_weakly = sum(1 for s in summary if s['classification'] == 'weakly_idempotent')
    n_idem = n_strongly + n_weakly
    rate = n_idem / n_total if n_total > 0 else 0

    binom_p = 1 - stats.binom.cdf(n_idem - 1, n_total, 0.6)
    # Wilson score confidence interval
    from statsmodels.stats.proportion import proportion_confint as _pci
    try:
        ci = _pci(n_idem, n_total, alpha=0.05, method='wilson')
    except Exception:
        p_hat = n_idem / n_total if n_total > 0 else 0
        z = 1.96
        denom = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denom
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
        ci = (max(0, center - margin), min(1, center + margin))

    return {
        'rate': round(rate, 4),
        'n_idempotent': n_idem,
        'n_strongly': n_strongly,
        'n_weakly': n_weakly,
        'n_total': n_total,
        'criterion_value': 0.6,
        'criterion_met': rate > 0.6,
        'binomial_p_value': round(float(binom_p), 6),
        'ci_95': [round(ci[0], 4), round(ci[1], 4)]
    }


def analyze_commutativity():
    with open(RESULTS_DIR / "commutativity_matrix.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        passes = header[1:]
        matrix = {}
        for row in reader:
            pi = row[0]
            for j, pj in enumerate(passes):
                matrix[(pi, pj)] = float(row[j + 1])

    from itertools import combinations
    pairs = list(combinations(passes, 2))
    n_total = len(pairs)
    n_non_comm = sum(1 for pi, pj in pairs if matrix.get((pi, pj), 1.0) < 0.5)
    rate = n_non_comm / n_total if n_total > 0 else 0

    binom_p = 1 - stats.binom.cdf(n_non_comm - 1, n_total, 0.3) if n_non_comm > 0 else 1.0
    try:
        from statsmodels.stats.proportion import proportion_confint as _pci
        ci = _pci(n_non_comm, n_total, alpha=0.05, method='wilson')
    except Exception:
        p_hat = n_non_comm / n_total if n_total > 0 else 0
        z = 1.96
        denom = 1 + z**2 / n_total
        center = (p_hat + z**2 / (2 * n_total)) / denom
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2)) / denom
        ci = (max(0, center - margin), min(1, center + margin))

    return {
        'non_commutative_rate': round(rate, 4),
        'n_non_commutative': n_non_comm,
        'n_total_pairs': n_total,
        'criterion_value': 0.3,
        'criterion_met': rate > 0.3,
        'binomial_p_value': round(float(binom_p), 6),
        'ci_95': [round(ci[0], 4), round(ci[1], 4)]
    }


def analyze_interference():
    interf_raw = []
    with open(RESULTS_DIR / "interference_raw.csv") as f:
        for row in csv.DictReader(f):
            interf_raw.append(row)

    from itertools import combinations
    passes = get_pass_list()
    pairs = list(combinations(passes, 2))
    n_total = len(pairs)

    pair_interf = {}
    for row in interf_raw:
        key = (row['pass_i'], row['pass_j'])
        baseline = int(row['baseline_instcount'])
        interf = float(row['interference'])
        pct = abs(interf) / baseline * 100 if baseline > 0 else 0
        if key not in pair_interf:
            pair_interf[key] = []
        pair_interf[key].append(pct)

    n_significant = 0
    n_constructive = 0
    n_destructive = 0
    for key, pcts in pair_interf.items():
        mean_pct = sum(pcts) / len(pcts)
        if mean_pct > 5:
            n_significant += 1
            row_vals = [float(r['interference']) for r in interf_raw
                       if (r['pass_i'], r['pass_j']) == key]
            if sum(row_vals) > 0:
                n_constructive += 1
            else:
                n_destructive += 1

    rate = n_significant / n_total if n_total > 0 else 0

    return {
        'significant_rate': round(rate, 4),
        'n_significant': n_significant,
        'n_constructive': n_constructive,
        'n_destructive': n_destructive,
        'n_total_pairs': n_total,
        'criterion_value': 0.1,
        'criterion_met': rate > 0.1
    }


def analyze_oscillation():
    cycle_file = Path(__file__).parent.parent / "cycle_detection" / "results.json"
    n_true_cycles = 0
    if cycle_file.exists():
        with open(cycle_file) as f:
            cycle_data = json.load(f)
            n_true_cycles = cycle_data.get('true_oscillation_cycles', 0)

    conv_json = Path(__file__).parent.parent / "convergence" / "results.json"
    conv_results = {}
    if conv_json.exists():
        with open(conv_json) as f:
            conv_results = json.load(f)

    n_oscillating_O2 = conv_results.get('pipelines', {}).get('-O2', {}).get('oscillating', 0)

    return {
        'oscillating_benchmarks_O2': n_oscillating_O2,
        'true_pass_cycles': n_true_cycles,
        'criterion_met': n_oscillating_O2 > 0 or n_true_cycles > 0
    }


def analyze_ordering():
    alg_ratios = {}
    with open(RESULTS_DIR / "algebra_ordering.csv") as f:
        for row in csv.DictReader(f):
            bm = row['benchmark']
            if row['ratio'] and row['ratio'] != 'None':
                if bm not in alg_ratios:
                    alg_ratios[bm] = []
                alg_ratios[bm].append(float(row['ratio']))

    o2_ratios = {}
    with open(RESULTS_DIR / "baseline_opt_levels.csv") as f:
        for row in csv.DictReader(f):
            if row['O2'] and row['O0'] and float(row['O0']) > 0:
                o2_ratios[row['benchmark']] = float(row['O2']) / float(row['O0'])

    common = set(alg_ratios.keys()) & set(o2_ratios.keys())
    alg_vals = [sum(alg_ratios[bm]) / len(alg_ratios[bm]) for bm in common]
    o2_vals = [o2_ratios[bm] for bm in common]

    alg_geo = math.exp(sum(math.log(max(v, 0.001)) for v in alg_vals) / len(alg_vals)) if alg_vals else 1
    o2_geo = math.exp(sum(math.log(max(v, 0.001)) for v in o2_vals) / len(o2_vals)) if o2_vals else 1

    diffs = [a - o for a, o in zip(alg_vals, o2_vals)]
    try:
        stat, p_val = stats.wilcoxon(diffs, alternative='two-sided')
    except Exception:
        stat, p_val = 0, 1.0

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 1
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    within_5pct = abs(alg_geo - o2_geo) / o2_geo < 0.05
    improves = alg_geo < o2_geo

    return {
        'algebra_geo_mean': round(alg_geo, 4),
        'O2_geo_mean': round(o2_geo, 4),
        'wilcoxon_stat': float(stat),
        'wilcoxon_p': float(p_val),
        'cohens_d': round(float(cohens_d), 4),
        'within_5pct_of_O2': within_5pct,
        'improves_over_O2': improves,
        'criterion_met': within_5pct or improves
    }


def analyze_clustering():
    passes = get_pass_list()
    interf = np.zeros((len(passes), len(passes)))
    with open(RESULTS_DIR / "interference_matrix.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            for j in range(1, len(row)):
                interf[i][j-1] = float(row[j])

    dist = np.abs(interf)
    np.fill_diagonal(dist, 0)
    condensed = []
    for i in range(len(passes)):
        for j in range(i + 1, len(passes)):
            condensed.append(1 - dist[i][j] / (dist.max() + 1e-10))
    condensed = np.array(condensed)

    Z = linkage(condensed, method='ward')
    labels = fcluster(Z, t=4, criterion='maxclust')

    loop_passes = {'indvars', 'licm', 'loop-deletion', 'loop-idiom', 'loop-reduce',
                   'loop-rotate', 'loop-simplify', 'loop-sink', 'loop-fusion',
                   'loop-distribute', 'loop-unroll', 'lcssa'}
    scalar_passes = {'instcombine', 'gvn', 'newgvn', 'sccp', 'reassociate',
                     'aggressive-instcombine', 'early-cse', 'float2int',
                     'constraint-elimination', 'nary-reassociate', 'instsimplify',
                     'div-rem-pairs', 'gvn-hoist', 'gvn-sink', 'correlated-propagation'}
    memory_passes = {'mem2reg', 'sroa', 'dse', 'memcpyopt'}
    cfg_passes = {'simplifycfg', 'jump-threading', 'adce', 'bdce', 'dce',
                  'sink', 'mergereturn', 'flattencfg'}

    true_labels = []
    for p in passes:
        if p in loop_passes:
            true_labels.append(0)
        elif p in scalar_passes:
            true_labels.append(1)
        elif p in memory_passes:
            true_labels.append(2)
        elif p in cfg_passes:
            true_labels.append(3)
        else:
            true_labels.append(4)

    ari = adjusted_rand_score(true_labels, labels)

    return {
        'num_clusters': 4,
        'adjusted_rand_index': round(ari, 4),
        'criterion_met': ari > 0
    }


def main():
    log_file = open(LOG_DIR / "analysis.log", 'w')
    print("=== Statistical Analysis ===")

    idem = analyze_idempotency()
    log_file.write(f"Idempotency: {json.dumps(idem, indent=2)}\n\n")
    print(f"  Idempotency rate: {idem['rate']*100:.1f}%, criterion (>60%): {'PASS' if idem['criterion_met'] else 'FAIL'}")

    comm = analyze_commutativity()
    log_file.write(f"Commutativity: {json.dumps(comm, indent=2)}\n\n")
    print(f"  Non-commutative rate: {comm['non_commutative_rate']*100:.1f}%, criterion (>30%): {'PASS' if comm['criterion_met'] else 'FAIL'}")

    interf = analyze_interference()
    log_file.write(f"Interference: {json.dumps(interf, indent=2)}\n\n")
    print(f"  Significant interference rate: {interf['significant_rate']*100:.1f}%, criterion (>10%): {'PASS' if interf['criterion_met'] else 'FAIL'}")

    osc = analyze_oscillation()
    log_file.write(f"Oscillation: {json.dumps(osc, indent=2)}\n\n")
    print(f"  Oscillating (O2): {osc['oscillating_benchmarks_O2']}, cycles: {osc['true_pass_cycles']}, criterion: {'PASS' if osc['criterion_met'] else 'FAIL'}")

    order = analyze_ordering()
    log_file.write(f"Ordering: {json.dumps(order, indent=2)}\n\n")
    print(f"  Algebra: {order['algebra_geo_mean']:.4f}, O2: {order['O2_geo_mean']:.4f}, criterion: {'PASS' if order['criterion_met'] else 'FAIL'}")

    clust = analyze_clustering()
    log_file.write(f"Clustering: {json.dumps(clust, indent=2)}\n\n")
    print(f"  ARI: {clust['adjusted_rand_index']:.4f}, criterion: {'PASS' if clust['criterion_met'] else 'FAIL'}")

    results = {
        'idempotency': idem,
        'commutativity': comm,
        'interference': interf,
        'oscillation': osc,
        'ordering': order,
        'clustering': clust
    }

    with open(RESULTS_DIR / "statistical_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    log_file.close()
    print("\nSaved to statistical_analysis.json")


if __name__ == '__main__':
    main()
