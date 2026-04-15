from __future__ import annotations

import itertools
from typing import Iterable

import networkx as nx
import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.DAG2CPDAG import dag2cpdag


def nx_dag_to_cl_dag(graph: nx.DiGraph) -> Dag:
    ordered_nodes = sorted(graph.nodes())
    nodes = [GraphNode(f"X{i}") for i in ordered_nodes]
    dag = Dag(nodes)
    node_map = {node_id: nodes[idx] for idx, node_id in enumerate(ordered_nodes)}
    for i, j in graph.edges():
        dag.add_directed_edge(node_map[i], node_map[j])
    return dag


def dag_to_cpdag_matrix(graph: nx.DiGraph) -> np.ndarray:
    cpdag = dag2cpdag(nx_dag_to_cl_dag(graph))
    return np.asarray(cpdag.graph, dtype=int)


def induced_subdag(graph: nx.DiGraph, nodes: list[int]) -> nx.DiGraph:
    return nx.DiGraph(graph.subgraph(nodes).copy())


def cpdag_skeleton(cpdag: np.ndarray) -> set[tuple[int, int]]:
    n = cpdag.shape[0]
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] != 0 or cpdag[j, i] != 0:
                edges.add((i, j))
    return edges


def cpdag_directed(cpdag: np.ndarray) -> set[tuple[int, int]]:
    n = cpdag.shape[0]
    directed = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cpdag[i, j] == -1 and cpdag[j, i] == 1:
                directed.add((i, j))
    return directed


def cpdag_undirected(cpdag: np.ndarray) -> set[tuple[int, int]]:
    n = cpdag.shape[0]
    undirected = set()
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] == -1 and cpdag[j, i] == -1:
                undirected.add((i, j))
    return undirected


def cpdag_colliders(cpdag: np.ndarray) -> set[tuple[int, int, int]]:
    n = cpdag.shape[0]
    colliders: set[tuple[int, int, int]] = set()
    skeleton = cpdag_skeleton(cpdag)
    directed = cpdag_directed(cpdag)
    for k in range(n):
        parents = [i for i in range(n) if (i, k) in directed]
        for i, j in itertools.combinations(sorted(parents), 2):
            if (min(i, j), max(i, j)) not in skeleton:
                colliders.add((i, k, j))
    return colliders


def cpdag_claims(cpdag: np.ndarray, global_nodes: list[int] | None = None) -> dict[str, set]:
    if global_nodes is None:
        global_nodes = list(range(cpdag.shape[0]))
    n = cpdag.shape[0]
    adj = set()
    nonadj = set()
    direction = set()
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = global_nodes[i], global_nodes[j]
            pair = (min(gi, gj), max(gi, gj))
            if cpdag[i, j] != 0 or cpdag[j, i] != 0:
                adj.add(pair)
            else:
                nonadj.add(pair)
            if cpdag[i, j] == -1 and cpdag[j, i] == 1:
                direction.add((gi, gj))
            elif cpdag[i, j] == 1 and cpdag[j, i] == -1:
                direction.add((gj, gi))
    coll = set()
    for i, k, j in cpdag_colliders(cpdag):
        coll.add((global_nodes[i], global_nodes[k], global_nodes[j]))
    return {"adj": adj, "nonadj": nonadj, "dir": direction, "coll": coll}


def restrict_cpdag(cpdag: np.ndarray, nodes: list[int]) -> np.ndarray:
    idx = np.asarray(nodes, dtype=int)
    return np.asarray(cpdag[np.ix_(idx, idx)], dtype=int)


def build_cpdag_from_claims(
    p: int,
    adjacency_scores: dict[tuple[int, int], tuple[float, float]],
    direction_scores: dict[tuple[int, int], tuple[float, float]],
    threshold: float,
    adjacency_order: list[tuple[int, int]] | None = None,
    direction_order: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(p))
    skeleton = []
    ordered_pairs = adjacency_order or sorted(adjacency_scores)
    for pair in ordered_pairs:
        support, oppose = adjacency_scores[pair]
        ratio = support / (support + oppose) if support + oppose > 0 else 0.0
        if ratio > threshold:
            skeleton.append(pair)
    for i, j in skeleton:
        dag.add_edge(i, j)
        dag.remove_edge(i, j)
    chosen = set()
    ordered_directions = direction_order or sorted(skeleton)
    for i, j in ordered_directions:
        if (i, j) not in skeleton:
            continue
        fwd_support, rev_support = direction_scores.get((i, j), (0.0, 0.0))
        if fwd_support > rev_support:
            cand = (i, j)
        elif rev_support > fwd_support:
            cand = (j, i)
        else:
            cand = (i, j)
        dag.add_edge(*cand)
        if nx.is_directed_acyclic_graph(dag):
            chosen.add((min(i, j), max(i, j), cand[0], cand[1]))
            continue
        dag.remove_edge(*cand)
    dag2 = nx.DiGraph()
    dag2.add_nodes_from(range(p))
    for i, j in skeleton:
        orient = next(((a, b) for x, y, a, b in chosen if (x, y) == (i, j)), None)
        if orient is not None:
            dag2.add_edge(*orient)
    order = list(nx.topological_sort(dag2))
    rank = {node: idx for idx, node in enumerate(order)}
    for i, j in skeleton:
        if dag2.has_edge(i, j) or dag2.has_edge(j, i):
            continue
        cand = (i, j) if rank.get(i, i) <= rank.get(j, j) else (j, i)
        dag2.add_edge(*cand)
        if not nx.is_directed_acyclic_graph(dag2):
            dag2.remove_edge(*cand)
            dag2.add_edge(cand[1], cand[0])
    return dag_to_cpdag_matrix(dag2)


def shd(pred: np.ndarray, truth: np.ndarray) -> int:
    n = pred.shape[0]
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_edge = pred[i, j] != 0 or pred[j, i] != 0
            true_edge = truth[i, j] != 0 or truth[j, i] != 0
            if pred_edge != true_edge:
                total += 1
                continue
            if not pred_edge:
                continue
            pred_dir = (
                1
                if pred[i, j] == -1 and pred[j, i] == 1
                else -1
                if pred[i, j] == 1 and pred[j, i] == -1
                else 0
            )
            true_dir = (
                1
                if truth[i, j] == -1 and truth[j, i] == 1
                else -1
                if truth[i, j] == 1 and truth[j, i] == -1
                else 0
            )
            if pred_dir != true_dir:
                total += 1
    return total


def precision_recall_f1(pred: set, truth: set) -> tuple[float, float, float]:
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def induced_pair_mapping(nodes: list[int]) -> dict[int, int]:
    return {node: idx for idx, node in enumerate(nodes)}
