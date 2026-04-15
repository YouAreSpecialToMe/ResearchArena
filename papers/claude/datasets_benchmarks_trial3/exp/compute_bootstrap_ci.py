"""Compute bootstrap 95% CIs for main table accuracies and composition gaps.
Also investigate negative CG values."""
import json
import numpy as np
import os

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/results"
SKILLS = ["AR", "CO", "CN", "LD", "SP", "ST", "TE", "SE"]

def load_jsonl(path):
    import jsonlines
    with jsonlines.open(path) as reader:
        return list(reader)

def bootstrap_accuracy(correct_array, n_bootstrap=10000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(correct_array)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        means.append(np.mean(correct_array[idx]))
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return float(np.mean(correct_array)), float(lo), float(hi)

def get_level(cat):
    parts = cat.split("+")
    return len(parts)

def compute_cg(single_accs, cat, cat_acc):
    parts = cat.split("+")
    min_single = min(single_accs.get(p, 0) for p in parts)
    return min_single - cat_acc

def analyze_model(summary_path, jsonl_path):
    with open(summary_path) as f:
        summary = json.load(f)

    per_cat = summary["per_category"]

    # Single skill accuracies
    single_accs = {}
    for s in SKILLS:
        if s in per_cat:
            single_accs[s] = per_cat[s]["accuracy"]

    # Bootstrap CIs for level averages
    level_items = {1: [], 2: [], 3: []}
    for cat, info in per_cat.items():
        level = get_level(cat)
        level_items[level].append(info["accuracy"])

    level_stats = {}
    for lv in [1, 2, 3]:
        arr = np.array(level_items[lv])
        mean, lo, hi = bootstrap_accuracy(arr)
        level_stats[f"L{lv}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(arr)}

    # Bootstrap CIs for average composition gap (pairwise)
    cg_values = []
    negative_cg_cats = []
    for cat, info in per_cat.items():
        if get_level(cat) == 2:
            cg = compute_cg(single_accs, cat, info["accuracy"])
            cg_values.append(cg)
            if cg < 0:
                negative_cg_cats.append({"pair": cat, "cg": cg, "composed_acc": info["accuracy"],
                    "min_single": min(single_accs.get(p, 0) for p in cat.split("+"))})

    cg_arr = np.array(cg_values)
    cg_mean, cg_lo, cg_hi = bootstrap_accuracy(cg_arr)

    # Negative CG analysis
    negative_cg_cats.sort(key=lambda x: x["cg"])

    return {
        "level_stats": level_stats,
        "avg_cg": {"mean": cg_mean, "ci_lo": cg_lo, "ci_hi": cg_hi},
        "negative_cg_pairs": negative_cg_cats,
        "n_negative_cg": len(negative_cg_cats),
        "n_total_pairs": len(cg_values),
        "fraction_negative": len(negative_cg_cats) / len(cg_values) if cg_values else 0
    }

# Analyze all direct models on seed 42
models = [
    ("Qwen-0.5B", "qwen0.5b_direct_seed42"),
    ("Qwen-1.5B", "qwen1.5b_direct_seed42"),
    ("Qwen-3B", "qwen3b_direct_seed42"),
    ("Llama-8B", "llama8b_direct_seed42"),
    ("Qwen-7B", "qwen7b_direct_seed42"),
    ("DS-R1-7B", "deepseek7b_direct_seed42"),
    ("Qwen-14B", "qwen14b_direct_seed42"),
    ("Qwen-32B", "qwen32b_direct_seed42"),
]

cot_models = [
    ("Qwen-1.5B-CoT", "qwen1.5b_cot_seed42"),
    ("Qwen-7B-CoT", "qwen7b_cot_seed42"),
    ("DS-R1-7B-CoT", "deepseek7b_cot_seed42"),
    ("Qwen-14B-CoT", "qwen14b_cot_seed42"),
    ("Qwen-32B-CoT", "qwen32b_cot_seed42"),
]

all_results = {}
for name, prefix in models + cot_models:
    summary_path = os.path.join(RESULTS_DIR, f"{prefix}_summary.json")
    if os.path.exists(summary_path):
        all_results[name] = analyze_model(summary_path, None)

# Print summary
print("=" * 80)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 80)
for name, res in all_results.items():
    ls = res["level_stats"]
    print(f"\n{name}:")
    for lv in ["L1", "L2", "L3"]:
        s = ls[lv]
        print(f"  {lv}: {s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] (n={s['n']})")
    cg = res["avg_cg"]
    print(f"  Avg CG: {cg['mean']:.3f} [{cg['ci_lo']:.3f}, {cg['ci_hi']:.3f}]")

print("\n" + "=" * 80)
print("NEGATIVE COMPOSITION GAP ANALYSIS")
print("=" * 80)
for name, res in all_results.items():
    neg = res["negative_cg_pairs"]
    if neg:
        print(f"\n{name}: {res['n_negative_cg']}/{res['n_total_pairs']} pairs have negative CG ({res['fraction_negative']:.1%})")
        for item in neg[:5]:
            parts = item["pair"].split("+")
            print(f"  {item['pair']}: CG={item['cg']:.3f} (composed={item['composed_acc']:.2f}, min_single={item['min_single']:.2f})")

# Aggregate: which pairs are most commonly negative across models?
pair_neg_count = {}
pair_neg_values = {}
for name, res in all_results.items():
    if "CoT" in name:
        continue  # only count direct
    for item in res["negative_cg_pairs"]:
        p = item["pair"]
        pair_neg_count[p] = pair_neg_count.get(p, 0) + 1
        pair_neg_values.setdefault(p, []).append(item["cg"])

print("\n\nPairs most commonly negative across direct models:")
for p, count in sorted(pair_neg_count.items(), key=lambda x: -x[1]):
    avg_neg = np.mean(pair_neg_values[p])
    print(f"  {p}: negative in {count}/8 models, avg CG when negative: {avg_neg:.3f}")

# Save results
with open(os.path.join(RESULTS_DIR, "bootstrap_ci_analysis.json"), "w") as f:
    json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

print("\nResults saved to bootstrap_ci_analysis.json")
