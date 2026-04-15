#!/usr/bin/env python3
"""Test equivalence with mixed FD+DC constraints that create denser CIGs."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from src.constraints import FD, DC, Predicate
from src.cris import CRIS
from src.baselines import SequentialIndependent
from src.utils import set_seed
from src.error_injection import inject_errors


def make_mixed_constraints():
    """Mixed FD+DC constraints with shared attributes creating dense interactions."""
    constraints = [
        FD(lhs={'A'}, rhs={'B'}, name="fd_AB"),
        FD(lhs={'B'}, rhs={'C'}, name="fd_BC"),
        FD(lhs={'C'}, rhs={'D'}, name="fd_CD"),
        FD(lhs={'E'}, rhs={'F'}, name="fd_EF"),
        FD(lhs={'F'}, rhs={'B'}, name="fd_FB"),  # Creates cycle: B->C, C->D, F->B
        DC(predicates=[Predicate(attr1='B', op='>', attr2='D', is_cross_tuple=False)],
           name="dc_BD"),
        DC(predicates=[Predicate(attr1='A', op='<', attr2='F', is_cross_tuple=False)],
           name="dc_AF"),
    ]
    attrs = ['A', 'B', 'C', 'D', 'E', 'F']
    return constraints, attrs


def make_dense_fds(n=8):
    """FDs with high attribute overlap - star topology from central attribute."""
    attrs = [f"X{i}" for i in range(n + 1)]
    constraints = []
    # Star: X0 determines everything
    for i in range(1, n + 1):
        constraints.append(FD(lhs={attrs[0]}, rhs={attrs[i]}, name=f"star_fd_{i}"))
    return constraints, attrs


def generate_data(constraints, attrs, n_tuples, seed):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame()
    for a in attrs:
        df[a] = rng.randint(0, 50, size=n_tuples)

    for c in constraints:
        if isinstance(c, FD):
            lhs_cols = sorted(c.lhs)
            rhs_cols = sorted(c.rhs)
            groups = df.groupby(lhs_cols)
            for _, group in groups:
                for col in rhs_cols:
                    df.loc[group.index, col] = group[col].iloc[0]

    for c in constraints:
        if isinstance(c, DC):
            for pred in c.predicates:
                if not pred.is_cross_tuple and pred.attr1 in df.columns and pred.attr2 in df.columns:
                    if pred.op in ['<', '<=']:
                        mask = df[pred.attr1] >= df[pred.attr2]
                        df.loc[mask, pred.attr1] = df.loc[mask, pred.attr2] - 1
                    elif pred.op in ['>', '>=']:
                        mask = df[pred.attr1] <= df[pred.attr2]
                        df.loc[mask, pred.attr1] = df.loc[mask, pred.attr2] + 1

    clean_df = df.copy()
    dirty_df, error_mask = inject_errors(clean_df, constraints, 0.10, seed)
    return clean_df, dirty_df, error_mask


def compare(dirty_df, constraints, seed):
    set_seed(seed)
    cris = CRIS(constraints, safety_level=1, epsilon=0.01, seed=seed)
    repaired_cris, log_cris = cris.repair(dirty_df, track_cascades=False)

    set_seed(seed)
    seq = SequentialIndependent()
    repaired_seq, log_seq = seq.repair(dirty_df, constraints, track_cascades=False)

    n_total = dirty_df.shape[0] * dirty_df.shape[1]
    n_agree = sum((repaired_cris[c].astype(str) == repaired_seq[c].astype(str)).sum()
                  for c in dirty_df.columns)
    return {
        'n_total_cells': int(n_total),
        'n_agree': int(n_agree),
        'agreement_pct': round(n_agree / max(n_total, 1) * 100, 2),
        'n_components': log_cris.get('n_components', 0),
        'n_edges': log_cris.get('cig_stats', {}).get('n_edges', 0),
    }


SEEDS = [42, 123, 456]
results = []

for name, make_fn in [('Mixed-7', make_mixed_constraints), ('Star-8', lambda: make_dense_fds(8))]:
    print(f"\n{name}:")
    constraints, attrs = make_fn()
    for seed in SEEDS:
        clean_df, dirty_df, _ = generate_data(constraints, attrs, 5000, seed)
        r = compare(dirty_df, constraints, seed)
        r['topology'] = name
        r['seed'] = seed
        r['n_constraints'] = len(constraints)
        results.append(r)
        print(f"  seed={seed}: agree={r['agreement_pct']}% edges={r['n_edges']} comp={r['n_components']}")

with open(os.path.join(os.path.dirname(__file__), 'results_mixed.json'), 'w') as f:
    json.dump(results, f, indent=2)
