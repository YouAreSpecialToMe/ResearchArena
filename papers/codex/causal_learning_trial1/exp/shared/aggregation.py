from __future__ import annotations

import itertools
import math
from collections import defaultdict

import numpy as np

from .graph_utils import build_cpdag_from_claims


def contradiction_stats(claims_a: dict[str, set], claims_b: dict[str, set], overlap: set[int]) -> dict[str, int]:
    by_family = {
        "adj": {"comparable": 0, "contradictions": 0},
        "dir": {"comparable": 0, "contradictions": 0},
        "coll": {"comparable": 0, "contradictions": 0},
    }
    overlap_pairs = {pair for pair in claims_a["adj"] | claims_a["nonadj"] | claims_b["adj"] | claims_b["nonadj"] if set(pair) <= overlap}
    for pair in overlap_pairs:
        a_adj = pair in claims_a["adj"]
        a_non = pair in claims_a["nonadj"]
        b_adj = pair in claims_b["adj"]
        b_non = pair in claims_b["nonadj"]
        if (a_adj or a_non) and (b_adj or b_non):
            by_family["adj"]["comparable"] += 1
            if (a_adj and b_non) or (a_non and b_adj):
                by_family["adj"]["contradictions"] += 1

    ordered_pairs = {
        pair for pair in claims_a["dir"] | claims_b["dir"] if {pair[0], pair[1]} <= overlap
    }
    for i, j in ordered_pairs:
        reverse = (j, i)
        pair = (min(i, j), max(i, j))
        a_has = (i, j) in claims_a["dir"]
        b_has = (i, j) in claims_b["dir"]
        a_rev = reverse in claims_a["dir"]
        b_rev = reverse in claims_b["dir"]
        a_non = pair in claims_a["nonadj"]
        b_non = pair in claims_b["nonadj"]
        if a_has and (b_has or b_rev or b_non):
            by_family["dir"]["comparable"] += 1
            if b_rev or b_non:
                by_family["dir"]["contradictions"] += 1
        if b_has and (a_has or a_rev or a_non):
            by_family["dir"]["comparable"] += 1
            if a_rev or a_non:
                by_family["dir"]["contradictions"] += 1

    for coll in claims_a["coll"] | claims_b["coll"]:
        if set(coll) <= overlap:
            has_a = coll in claims_a["coll"]
            has_b = coll in claims_b["coll"]
            if not (has_a and has_b):
                continue
            by_family["coll"]["comparable"] += 1
            i, k, j = coll
            incompatible = (
                (k, i) in claims_a["dir"]
                or (k, j) in claims_a["dir"]
                or (k, i) in claims_b["dir"]
                or (k, j) in claims_b["dir"]
            )
            if incompatible:
                by_family["coll"]["contradictions"] += 1
    comparable = sum(value["comparable"] for value in by_family.values())
    contradictions = sum(value["contradictions"] for value in by_family.values())
    return {
        "comparable": comparable,
        "contradictions": contradictions,
        "adj_comparable": by_family["adj"]["comparable"],
        "adj_contradictions": by_family["adj"]["contradictions"],
        "dir_comparable": by_family["dir"]["comparable"],
        "dir_contradictions": by_family["dir"]["contradictions"],
        "coll_comparable": by_family["coll"]["comparable"],
        "coll_contradictions": by_family["coll"]["contradictions"],
    }


def bootstrap_stability_score(claims_a: dict[str, set], claims_b: dict[str, set]) -> float:
    stats = contradiction_stats(claims_a, claims_b, set(range(10_000)))
    if stats["comparable"] == 0:
        return 1.0
    return 1.0 - stats["contradictions"] / stats["comparable"]


def wilson_accept(support: float, oppose: float, n_eff: float) -> tuple[bool, float]:
    total = support + oppose
    if total == 0 or n_eff <= 0:
        return False, 0.0
    p_hat = support / total
    z = 1.2815515655446004
    denom = 1.0 + z * z / n_eff
    center = p_hat + z * z / (2.0 * n_eff)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4.0 * n_eff)) / n_eff)
    lower = (center - margin) / denom
    return lower > 0.5, lower


def merge_local_graphs(
    p: int,
    local_entries: list[dict],
    weights: dict[str, float],
    merge_rule: str,
    include_dir: bool = True,
    include_colliders: bool = True,
    weak_abstention: bool = False,
) -> tuple[np.ndarray, dict]:
    adj_scores: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    dir_scores: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    support_table = []
    all_weights = []
    for entry in local_entries:
        w = weights[entry["graph_id"]]
        all_weights.append(w)
        claims = entry["claims"]
        for pair in claims["adj"]:
            adj_scores[pair][0] += w
        for pair in claims["nonadj"]:
            adj_scores[pair][1] += w
        if include_dir:
            for i, j in claims["dir"]:
                key = (min(i, j), max(i, j))
                if (i, j) == key:
                    dir_scores[key][0] += w
                else:
                    dir_scores[key][1] += w
        if include_colliders:
            for i, k, j in claims["coll"]:
                for u, v in [(i, k), (j, k)]:
                    key = (min(u, v), max(u, v))
                    if (u, v) == key:
                        dir_scores[key][0] += 0.5 * w
                    else:
                        dir_scores[key][1] += 0.5 * w
    n_eff = (sum(all_weights) ** 2 / sum(w * w for w in all_weights)) if all_weights and sum(w * w for w in all_weights) else 0.0

    threshold = 0.6 if merge_rule == "DetThreshold" else 0.5
    filtered_adj = {}
    adjacency_order = []
    for pair, scores in adj_scores.items():
        support, oppose = scores
        ratio = support / (support + oppose) if support + oppose > 0 else 0.0
        accepted = False
        stat_value = ratio
        if merge_rule == "Wilson":
            accepted, stat_value = wilson_accept(support, oppose, n_eff)
        else:
            accepted = ratio > threshold
        filtered_adj[pair] = (support if accepted else 0.0, oppose if accepted else max(support, oppose))
        adjacency_order.append(pair)
        support_table.append(
            {
                "claim_type": "adj",
                "claim": f"{pair[0]}-{pair[1]}",
                "support": support,
                "opposition": oppose,
                "statistic": stat_value,
                "accepted": accepted,
            }
        )
    direction_order = []
    for pair, scores in dir_scores.items():
        fwd, rev = scores
        total = fwd + rev
        if total == 0:
            continue
        ratio = max(fwd, rev) / total
        accepted = ratio > (0.6 if merge_rule == "DetThreshold" else 0.5)
        if merge_rule == "Wilson":
            accepted, ratio = wilson_accept(max(fwd, rev), min(fwd, rev), n_eff)
        support_table.append(
            {
                "claim_type": "dir",
                "claim": f"{pair[0]}->{pair[1]}",
                "support": max(fwd, rev),
                "opposition": min(fwd, rev),
                "statistic": ratio,
                "accepted": accepted,
            }
        )
        if accepted:
            direction_order.append(pair)
    if merge_rule == "DetRank":
        adjacency_order = sorted(
            adjacency_order,
            key=lambda pair: (
                abs(filtered_adj[pair][0] - filtered_adj[pair][1]),
                filtered_adj[pair][0],
                -filtered_adj[pair][1],
            ),
            reverse=True,
        )
        direction_order = sorted(
            direction_order,
            key=lambda pair: (
                abs(dir_scores[pair][0] - dir_scores[pair][1]),
                max(dir_scores[pair][0], dir_scores[pair][1]),
            ),
            reverse=True,
        )
    cpdag = build_cpdag_from_claims(
        p=p,
        adjacency_scores={k: tuple(v) for k, v in filtered_adj.items()},
        direction_scores={k: tuple(v) for k, v in dir_scores.items()},
        threshold=0.5,
        adjacency_order=adjacency_order if merge_rule == "DetRank" else None,
        direction_order=direction_order if merge_rule == "DetRank" else None,
    )
    ratios = [row["support"] / (row["support"] + row["opposition"]) for row in support_table if (row["support"] + row["opposition"]) > 0]
    oppositions = [row["opposition"] for row in support_table]
    accepted_adjacency_claims = sum(1 for row in support_table if row["claim_type"] == "adj" and row["accepted"])
    accepted_orientation_claims = sum(1 for row in support_table if row["claim_type"] == "dir" and row["accepted"])
    return cpdag, {
        "support_table": support_table,
        "n_eff": n_eff,
        "accepted_adjacency_claims": accepted_adjacency_claims,
        "accepted_orientation_claims": accepted_orientation_claims,
        "median_weighted_support_ratio": float(np.median(ratios)) if ratios else 0.0,
        "median_weighted_opposition": float(np.median(oppositions)) if oppositions else 0.0,
    }


def compute_weights(local_entries: list[dict], scheme: str, weak_abstention: bool = False) -> dict[str, float]:
    if scheme == "Uniform":
        return {entry["graph_id"]: 1.0 for entry in local_entries}
    by_subset = defaultdict(list)
    for entry in local_entries:
        by_subset[entry["subset_id"]].append(entry)
    if scheme == "BootstrapStability":
        weights = {}
        for entries in by_subset.values():
            score = bootstrap_stability_score(entries[0]["claims"], entries[1]["claims"]) if len(entries) == 2 else 1.0
            for entry in entries:
                weights[entry["graph_id"]] = score
        return weights

    overlaps = defaultdict(list)
    for a, b in itertools.combinations(local_entries, 2):
        overlap = set(a["nodes"]) & set(b["nodes"])
        if len(overlap) < 2:
            continue
        stats = contradiction_stats(a["claims"], b["claims"], overlap)
        comparable = stats["comparable"]
        rate = stats["contradictions"] / comparable if comparable else (0.15 if weak_abstention else 0.0)
        overlaps[a["graph_id"]].append(rate)
        overlaps[b["graph_id"]].append(rate)
    compat = {entry["graph_id"]: float(np.mean(overlaps[entry["graph_id"]]) if overlaps[entry["graph_id"]] else 0.0) for entry in local_entries}
    ordered = sorted(compat.items(), key=lambda kv: kv[1])
    if scheme == "CompatExp":
        return {gid: math.exp(-2.0 * score) for gid, score in compat.items()}
    if scheme == "CompatRank":
        if len(ordered) == 1:
            return {ordered[0][0]: 1.0}
        return {gid: 1.0 - idx / (len(ordered) - 1) for idx, (gid, _) in enumerate(ordered)}
    if scheme == "CompatTopHalf":
        cutoff = math.ceil(len(ordered) / 2)
        keep = {gid for gid, _ in ordered[:cutoff]}
        return {entry["graph_id"]: 1.0 if entry["graph_id"] in keep else 0.0 for entry in local_entries}
    raise ValueError(scheme)
