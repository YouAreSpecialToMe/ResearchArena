from __future__ import annotations

import itertools

import numpy as np


def build_subset_bank(p: int, k: int, m: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    variable_counts = np.zeros(p, dtype=int)
    pair_counts = np.zeros((p, p), dtype=int)
    subsets: list[list[int]] = []
    for _ in range(m):
        best_subset = None
        best_score = None
        for _ in range(400):
            cand = np.sort(rng.choice(p, size=k, replace=False)).tolist()
            v_score = sum(max(0, 6 - variable_counts[v]) for v in cand)
            p_score = 0
            for i, j in itertools.combinations(cand, 2):
                p_score += max(0, 2 - pair_counts[i, j])
            score = (v_score, p_score, rng.random())
            if best_score is None or score > best_score:
                best_score = score
                best_subset = cand
        assert best_subset is not None
        subsets.append(best_subset)
        for v in best_subset:
            variable_counts[v] += 1
        for i, j in itertools.combinations(best_subset, 2):
            pair_counts[i, j] += 1
            pair_counts[j, i] += 1
    return subsets

