from __future__ import annotations

import itertools

import pandas as pd


def ranking_frame(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(
        ["miss_rate", "writeback_count", "mean_eviction_age", "policy"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    ordered["rank"] = range(1, len(ordered) + 1)
    return ordered


def _rank_dict(df: pd.DataFrame) -> dict[str, int]:
    ranked = ranking_frame(df)
    return {row["policy"]: int(row["rank"]) for _, row in ranked.iterrows()}


def spearman_rho(a: dict[str, int], b: dict[str, int]) -> float:
    n = len(a)
    diff = sum((a[p] - b[p]) ** 2 for p in a)
    return 1.0 - 6.0 * diff / (n * (n * n - 1))


def kendall_tau(a: dict[str, int], b: dict[str, int]) -> float:
    policies = list(a)
    conc = 0
    disc = 0
    for i, j in itertools.combinations(policies, 2):
        va = a[i] - a[j]
        vb = b[i] - b[j]
        if va * vb > 0:
            conc += 1
        elif va * vb < 0:
            disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def topk_set_recall(a: dict[str, int], b: dict[str, int], k: int) -> float:
    top_a = {p for p, r in a.items() if r <= k}
    top_b = {p for p, r in b.items() if r <= k}
    return len(top_a & top_b) / max(1, len(top_a))


def compare_rankings(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, suffix: str = "") -> dict[str, float]:
    ref = _rank_dict(reference_df)
    cand = _rank_dict(candidate_df)
    ranked_ref = ranking_frame(reference_df)
    ranked_cand = ranking_frame(candidate_df)
    best_ref = ranked_ref.iloc[0]
    best_cand = ranked_cand.iloc[0]
    key = f"_{suffix}" if suffix else ""
    return {
        f"Kendall_tau_6{key}": kendall_tau(ref, cand),
        f"Spearman_rho_6{key}": spearman_rho(ref, cand),
        f"top1_agreement{key}": float(ranked_ref.iloc[0]["policy"] == ranked_cand.iloc[0]["policy"]),
        f"top2_set_recall{key}": topk_set_recall(ref, cand, 2),
        f"best_policy_regret{key}": float(best_cand["miss_rate"] - best_ref["miss_rate"]),
    }
