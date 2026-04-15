from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score

LABELS = ["context", "memory", "abstain"]


def utility_from_answer(pred_route: str, answer_ok: bool) -> float:
    if pred_route == "abstain":
        return 0.0
    if answer_ok:
        return 1.0
    return -2.0


def summarize_predictions(rows: List[Dict], regime_name: str) -> Dict:
    gold = []
    route_preds = []
    alias_hits = []
    naive_hits = []
    utilities = []
    harmful = 0
    abstain = 0
    for row in rows:
        gold_route = row.get("gold_route")
        pred_route = row["pred_route"]
        answer_ok = bool(row["answer_ok"])
        answer_ok_naive = bool(row.get("answer_ok_naive", answer_ok))
        if gold_route is not None:
            gold.append(gold_route)
            route_preds.append(pred_route)
        alias_hits.append(1.0 if answer_ok else 0.0)
        naive_hits.append(1.0 if answer_ok_naive else 0.0)
        util = utility_from_answer(pred_route, answer_ok)
        utilities.append(util)
        if pred_route != "abstain" and not answer_ok:
            harmful += 1
        if pred_route == "abstain":
            abstain += 1
    route_acc = float(np.mean([g == p for g, p in zip(gold, route_preds)])) if gold else None
    macro_f1 = float(f1_score(gold, route_preds, labels=LABELS, average="macro", zero_division=0)) if gold else None
    return {
        "regime": regime_name,
        "n": len(rows),
        "route_accuracy": route_acc,
        "route_macro_f1": macro_f1,
        "alias_aware_exact_match": float(np.mean(alias_hits)) if alias_hits else 0.0,
        "naive_exact_match": float(np.mean(naive_hits)) if naive_hits else 0.0,
        "benchmark_native_accuracy": float(np.mean(alias_hits)) if alias_hits else 0.0,
        "harmful_answer_rate": harmful / len(rows) if rows else 0.0,
        "abstention_rate": abstain / len(rows) if rows else 0.0,
        "coverage": 1.0 - (abstain / len(rows) if rows else 0.0),
        "expected_utility": float(np.mean(utilities)) if utilities else 0.0,
        "label_balance": {k: int(v) for k, v in Counter(gold).items()} if gold else None,
    }


def ranking_stats(rank_a: Dict[str, float], rank_b: Dict[str, float]) -> Dict:
    policies = sorted(set(rank_a) & set(rank_b))
    a = [rank_a[p] for p in policies]
    b = [rank_b[p] for p in policies]
    tau = float(kendalltau(a, b).statistic)
    rho = float(spearmanr(a, b).statistic)
    reversals = 0
    for i, pa in enumerate(policies):
        for pb in policies[i + 1 :]:
            rel_a = rank_a[pa] - rank_a[pb]
            rel_b = rank_b[pa] - rank_b[pb]
            if rel_a * rel_b < 0:
                reversals += 1
    top_change = min(rank_a, key=rank_a.get) != min(rank_b, key=rank_b.get)
    return {
        "kendall_tau": tau,
        "spearman_rho": rho,
        "pairwise_reversals": reversals,
        "top_policy_change": top_change,
    }


def bootstrap_ci(values: List[float], rng: np.random.Generator, n_resamples: int = 1000) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(float(np.mean(arr[idx])))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)
