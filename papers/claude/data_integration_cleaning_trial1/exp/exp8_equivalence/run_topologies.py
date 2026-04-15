#!/usr/bin/env python3
"""Experiment 8b: Decomposition Equivalence with Controlled CIG Topologies.

Creates synthetic constraint sets with known chain, tree, and forest
interaction structures, then compares cell-level agreement between
CRIS decomposed repair and sequential (monolithic-proxy) repair.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from src.constraints import FD, DC, Predicate
from src.cris import CRIS
from src.baselines import SequentialIndependent
from src.utils import set_seed
from src.error_injection import inject_errors


def make_chain_constraints(n):
    """Chain topology: FD_0 -> FD_1 -> ... -> FD_{n-1}.
    Each FD shares its RHS with the next FD's LHS, creating a chain of interactions.
    """
    attrs = [f"X{i}" for i in range(n + 1)]
    constraints = []
    for i in range(n):
        fd = FD(lhs={attrs[i]}, rhs={attrs[i + 1]}, name=f"chain_fd_{i}")
        constraints.append(fd)
    return constraints, attrs


def make_tree_constraints(n):
    """Tree topology: root FD with branching children.
    Root: X0 -> X1. Branches from X1: X1->X2, X1->X3, X1->X4, etc.
    Then X2->X5, X3->X6, etc.
    """
    attrs_needed = n + 1
    attrs = [f"X{i}" for i in range(attrs_needed + 5)]
    constraints = []
    # Root
    constraints.append(FD(lhs={attrs[0]}, rhs={attrs[1]}, name="tree_fd_0"))
    # Branch from X1
    idx = 2
    parent_queue = [1]
    fd_count = 1
    while fd_count < n and parent_queue:
        parent = parent_queue.pop(0)
        for _ in range(min(2, n - fd_count)):  # binary branching
            constraints.append(FD(lhs={attrs[parent]}, rhs={attrs[idx]}, name=f"tree_fd_{fd_count}"))
            parent_queue.append(idx)
            idx += 1
            fd_count += 1
            if fd_count >= n:
                break
    return constraints, attrs[:idx]


def make_forest_constraints(n):
    """Forest topology: multiple disconnected trees (3-4 per tree)."""
    attrs = [f"X{i}" for i in range(n * 2 + 10)]
    constraints = []
    attr_idx = 0
    fd_count = 0
    while fd_count < n:
        tree_size = min(3, n - fd_count)
        root_attr = attrs[attr_idx]
        attr_idx += 1
        for _ in range(tree_size):
            child_attr = attrs[attr_idx]
            attr_idx += 1
            constraints.append(FD(lhs={root_attr}, rhs={child_attr}, name=f"forest_fd_{fd_count}"))
            fd_count += 1
    return constraints, attrs[:attr_idx]


def generate_data(constraints, attrs, n_tuples, seed):
    """Generate clean data satisfying the FD constraints, then inject errors."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame()
    for a in attrs:
        df[a] = rng.randint(0, 50, size=n_tuples)

    # Enforce FDs by grouping
    for c in constraints:
        if isinstance(c, FD):
            lhs_cols = sorted(c.lhs)
            rhs_cols = sorted(c.rhs)
            groups = df.groupby(lhs_cols)
            for _, group in groups:
                for col in rhs_cols:
                    val = group[col].iloc[0]
                    df.loc[group.index, col] = val

    clean_df = df.copy()
    dirty_df, error_mask = inject_errors(clean_df, constraints, 0.10, seed)
    return clean_df, dirty_df, error_mask


def compare_repairs(dirty_df, constraints, seed):
    """Compare CRIS (decomposed) vs Sequential (monolithic proxy)."""
    # CRIS decomposed repair (Level 1 - structural)
    set_seed(seed)
    cris = CRIS(constraints, safety_level=1, epsilon=0.01, seed=seed)
    repaired_cris, log_cris = cris.repair(dirty_df, track_cascades=False)

    # Sequential (monolithic proxy - same algorithms, fixed order)
    set_seed(seed)
    seq = SequentialIndependent()
    repaired_seq, log_seq = seq.repair(dirty_df, constraints, track_cascades=False)

    # Cell-level comparison
    n_total = dirty_df.shape[0] * dirty_df.shape[1]
    n_agree = 0
    for col in dirty_df.columns:
        n_agree += (repaired_cris[col].astype(str) == repaired_seq[col].astype(str)).sum()

    return {
        'n_total_cells': int(n_total),
        'n_agree': int(n_agree),
        'agreement_pct': round(n_agree / max(n_total, 1) * 100, 2),
        'n_components': log_cris.get('n_components', 0),
        'n_edges': log_cris.get('cig_stats', {}).get('n_edges', 0),
    }


def main():
    SEEDS = [42, 123, 456]
    N_TUPLES = 5000
    results = []

    topologies = {
        'Chain-6': (make_chain_constraints, 6),
        'Chain-10': (make_chain_constraints, 10),
        'Tree-8': (make_tree_constraints, 8),
        'Tree-12': (make_tree_constraints, 12),
        'Forest-6': (make_forest_constraints, 6),
        'Forest-10': (make_forest_constraints, 10),
    }

    for topo_name, (make_fn, n) in topologies.items():
        print(f"\n{'='*50}")
        print(f"Topology: {topo_name}")
        print(f"{'='*50}")

        constraints, attrs = make_fn(n)
        print(f"  Constraints: {len(constraints)}, Attributes: {len(attrs)}")

        for seed in SEEDS:
            clean_df, dirty_df, error_mask = generate_data(constraints, attrs, N_TUPLES, seed)
            result = compare_repairs(dirty_df, constraints, seed)
            result['topology'] = topo_name
            result['n_constraints'] = len(constraints)
            result['seed'] = seed
            results.append(result)
            print(f"  seed={seed}: agreement={result['agreement_pct']}%, "
                  f"edges={result['n_edges']}, components={result['n_components']}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results_topologies.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    from collections import defaultdict
    by_topo = defaultdict(list)
    for r in results:
        by_topo[r['topology']].append(r['agreement_pct'])
    for topo, agreements in by_topo.items():
        mean_agree = sum(agreements) / len(agreements)
        print(f"  {topo}: mean agreement = {mean_agree:.1f}% (range {min(agreements):.1f}-{max(agreements):.1f}%)")


if __name__ == '__main__':
    main()
