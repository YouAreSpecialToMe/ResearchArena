from __future__ import annotations

import numpy as np

from .graph_utils import cpdag_colliders, cpdag_directed, cpdag_skeleton, precision_recall_f1, shd


def score_cpdag(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred_skel = cpdag_skeleton(pred)
    true_skel = cpdag_skeleton(truth)
    pred_dir = cpdag_directed(pred)
    true_dir = cpdag_directed(truth)
    pred_coll = cpdag_colliders(pred)
    true_coll = cpdag_colliders(truth)

    sk_prec, sk_rec, sk_f1 = precision_recall_f1(pred_skel, true_skel)
    dir_prec, dir_rec, _ = precision_recall_f1(pred_dir, true_dir)
    coll_prec, coll_rec, _ = precision_recall_f1(pred_coll, true_coll)
    false_orientation_rate = 1.0 - dir_prec if pred_dir else 0.0
    fraction_undirected = len({pair for pair in pred_skel if pair not in {(min(i, j), max(i, j)) for i, j in pred_dir}}) / len(pred_skel) if pred_skel else 0.0
    return {
        "shd": float(shd(pred, truth)),
        "skeleton_precision": float(sk_prec),
        "skeleton_recall": float(sk_rec),
        "skeleton_f1": float(sk_f1),
        "orientation_precision": float(dir_prec),
        "orientation_recall": float(dir_rec),
        "false_orientation_rate": float(false_orientation_rate),
        "fraction_undirected": float(fraction_undirected),
        "collider_precision": float(coll_prec),
        "collider_recall": float(coll_rec),
        "graph_density": float(len(pred_skel) / (pred.shape[0] * (pred.shape[0] - 1) / 2)),
        "oriented_edge_count": float(len(pred_dir)),
    }

