from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .metrics import binary_entropy, compute_auec
from .posterior import (
    DatasetRecord,
    PosteriorState,
    fit_dag_parameters,
    simulate_from_posterior_dag,
    update_posterior,
)


@dataclass(frozen=True)
class Action:
    target: int
    family: str
    batch_size: int


def action_cost(action: Action, previous_family: str | None, cfg: dict, switch_matrix: list[list[float]]) -> tuple[float, float, float]:
    exec_cost = cfg["execution_cost"][action.family]
    sample_cost = cfg["sample_cost"][action.family] * action.batch_size
    if previous_family is None:
        switch_cost = 0.0
    else:
        i = 0 if previous_family == "hard" else 1
        j = 0 if action.family == "hard" else 1
        switch_cost = float(switch_matrix[i][j])
    return exec_cost, sample_cost, switch_cost


def feasible_actions(state: PosteriorState, remaining_budget: float, cfg: dict, switch_matrix: list[list[float]]) -> list[Action]:
    actions = []
    d = state.skeleton.shape[0]
    for target in range(d):
        for family in cfg["families"]:
            for batch in cfg["batch_sizes"]:
                action = Action(target, family, batch)
                if sum(action_cost(action, state.previous_family, cfg, switch_matrix)) <= remaining_budget + 1e-9:
                    actions.append(action)
    return actions


def _target_entropy_score(state: PosteriorState, action: Action) -> float:
    probs = state.orientation_probabilities()
    neighbors = np.flatnonzero(state.skeleton[action.target] > 0)
    if neighbors.size == 0:
        return 0.0
    outgoing = probs[action.target, neighbors]
    incoming = probs[neighbors, action.target]
    uncertainty = binary_entropy(np.clip(outgoing, 1e-9, 1 - 1e-9))
    if action.family == "hard":
        family_fit = 0.8 * outgoing + 0.2 * incoming
    else:
        family_fit = 0.8 * incoming + 0.2 * outgoing
    score = float(np.sum(uncertainty * (0.5 + family_fit)))
    score *= np.sqrt(action.batch_size / 25.0)
    return score


def _rank_actions(
    state: PosteriorState,
    remaining_budget: float,
    cfg: dict,
    switch_matrix: list[list[float]],
) -> list[Action]:
    actions = feasible_actions(state, remaining_budget, cfg, switch_matrix)
    def screening_score(action: Action) -> float:
        exec_cost, sample_cost, switch_cost = action_cost(action, state.previous_family, cfg, switch_matrix)
        total_cost = exec_cost + sample_cost + switch_cost
        affordability = max(0.05, 1.0 - total_cost / max(remaining_budget, 1e-9))
        switch_discount = 1.0 / (1.0 + switch_cost)
        # This is a screening heuristic for top-k search, not the final objective.
        return _target_entropy_score(state, action) * (0.4 + 0.6 * affordability) * switch_discount

    return sorted(actions, key=screening_score, reverse=True)


def _fit_params_cached(dag: np.ndarray, datasets: list[DatasetRecord]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    # A small in-process cache keyed by DAG bytes and the current number of datasets.
    cache = _fit_params_cached.__dict__.setdefault("_cache", {})
    key = (dag.tobytes(), len(datasets))
    if key not in cache:
        params = fit_dag_parameters(dag, datasets)
        cache[key] = (params.intercepts, params.coefs, params.noise_vars)
    return cache[key]


def evaluate_sequence(
    state: PosteriorState,
    sequence: tuple[Action, ...],
    cfg: dict,
    remaining_budget: float,
    switch_matrix: list[list[float]],
    intervention_params: dict,
    seed: int,
    rejuvenate: bool,
    rollouts: int,
) -> dict:
    initial_entropy = state.orientation_entropy()
    vb_values = []
    auec_values = []
    for rollout_idx in range(rollouts):
        rng = np.random.default_rng(seed + 7919 * (rollout_idx + 1))
        rollout_state = state.clone()
        budget = remaining_budget
        cumulative_cost = 0.0
        costs: list[float] = []
        entropies: list[float] = []
        valid = True
        for depth, action in enumerate(sequence):
            exec_cost, sample_cost, switch_cost = action_cost(action, rollout_state.previous_family, cfg, switch_matrix)
            cost = exec_cost + sample_cost + switch_cost
            if cost > budget + 1e-9:
                valid = False
                break
            dag_idx = int(rng.choice(len(rollout_state.dags), p=rollout_state.weights))
            dag = rollout_state.dags[dag_idx]
            intercepts, coefs, noise_vars = _fit_params_cached(dag, rollout_state.datasets)
            params = type("Params", (), {"intercepts": intercepts, "coefs": coefs, "noise_vars": noise_vars})
            data = simulate_from_posterior_dag(
                rng,
                dag,
                params,
                action.batch_size,
                {"target": action.target, "family": action.family, **intervention_params},
            )
            rollout_state = update_posterior(
                rollout_state,
                DatasetRecord(kind="intervention", target=action.target, family=action.family, data=data),
                rejuvenate=rejuvenate,
                seed=seed + depth * 101 + rollout_idx * 103,
            )
            budget -= cost
            cumulative_cost += cost
            costs.append(cumulative_cost)
            entropies.append(rollout_state.orientation_entropy())
        if not valid:
            return {"v_b": -1e12, "AUEC_partial": -1e12, "valid": False}
        terminal_entropy = rollout_state.orientation_entropy()
        vb_values.append(initial_entropy - terminal_entropy)
        auec_values.append(compute_auec(costs, entropies, initial_entropy, remaining_budget))
    return {
        "v_b": float(np.mean(vb_values)),
        "AUEC_partial": float(np.mean(auec_values)),
        "valid": True,
    }


def _generate_candidate_sequences(
    state: PosteriorState,
    remaining_budget: float,
    cfg: dict,
    switch_matrix: list[list[float]],
    horizon: int,
    top_k: int | None,
) -> list[tuple[Action, ...]]:
    ranked = _rank_actions(state, remaining_budget, cfg, switch_matrix)
    first_actions = ranked if top_k is None else ranked[:top_k]
    sequences: list[tuple[Action, ...]] = []

    def recurse(prefix: tuple[Action, ...], cur_state: PosteriorState, cur_budget: float, depth_left: int) -> None:
        next_ranked = _rank_actions(cur_state, cur_budget, cfg, switch_matrix)
        if depth_left == 0 or not next_ranked:
            if prefix:
                sequences.append(prefix)
            return
        next_actions = next_ranked if top_k is None else next_ranked[:top_k]
        for action in next_actions:
            cost = sum(action_cost(action, cur_state.previous_family, cfg, switch_matrix))
            if cost > cur_budget + 1e-9:
                continue
            next_state = cur_state.clone()
            next_state.previous_family = action.family
            recurse(prefix + (action,), next_state, cur_budget - cost, depth_left - 1)

    for action in first_actions:
        cost = sum(action_cost(action, state.previous_family, cfg, switch_matrix))
        if cost > remaining_budget + 1e-9:
            continue
        next_state = state.clone()
        next_state.previous_family = action.family
        recurse((action,), next_state, remaining_budget - cost, horizon - 1)
    return sequences


def _best_sequence(
    state: PosteriorState,
    cfg: dict,
    remaining_budget: float,
    switch_matrix: list[list[float]],
    intervention_params: dict,
    seed: int,
    horizon: int,
    top_k: int | None,
    rejuvenate: bool,
    rollouts: int,
) -> tuple[tuple[Action, ...] | None, bool]:
    candidates = _generate_candidate_sequences(state, remaining_budget, cfg, switch_matrix, horizon, top_k)
    if not candidates:
        return None, False
    best_sequence = None
    best_score = -1e12
    best_auec = -1e12
    best_switches = 10**9
    for idx, seq in enumerate(candidates):
        metrics = evaluate_sequence(
            state,
            seq,
            cfg,
            remaining_budget,
            switch_matrix,
            intervention_params,
            seed + idx * 17,
            rejuvenate=rejuvenate,
            rollouts=rollouts,
        )
        seq_switches = sum(int(prev.family != cur.family) for prev, cur in zip(seq[:-1], seq[1:]))
        better = (
            metrics["valid"]
            and (
                metrics["v_b"] > best_score + 1e-9
                or (
                    abs(metrics["v_b"] - best_score) <= 1e-9
                    and (
                        metrics["AUEC_partial"] > best_auec + 1e-9
                        or (
                            abs(metrics["AUEC_partial"] - best_auec) <= 1e-9
                            and seq_switches < best_switches
                        )
                    )
                )
            )
        )
        if better:
            best_score = metrics["v_b"]
            best_auec = metrics["AUEC_partial"]
            best_switches = seq_switches
            best_sequence = seq
    return best_sequence, False


def plan_sequence(
    method: str,
    state: PosteriorState,
    instance,
    cfg: dict,
    remaining_budget: float,
    switch_matrix: list[list[float]],
    seed: int,
    exact_branch_cap: int = 5000,
) -> tuple[tuple[Action, ...] | None, dict]:
    start = time.time()
    actions = feasible_actions(state, remaining_budget, cfg, switch_matrix)
    if not actions:
        return None, {"planner_runtime_sec": time.time() - start, "fallback_triggered": False}

    intervention_params = cfg["intervention_regimes"][instance.regime]
    exact_rollouts = cfg["rollouts_exact"] if state.exact else cfg["rollouts_main"]

    if method == "myopic_budgeted_gain":
        best, _ = _best_sequence(
            state,
            cfg,
            remaining_budget,
            switch_matrix,
            intervention_params,
            seed,
            horizon=1,
            top_k=None,
            rejuvenate=False,
            rollouts=exact_rollouts,
        )
        return best, {"planner_runtime_sec": time.time() - start, "fallback_triggered": False}

    if method == "ratio_objective":
        best_action = None
        best_ratio = -1e12
        for idx, action in enumerate(actions):
            metrics = evaluate_sequence(
                state,
                (action,),
                cfg,
                remaining_budget,
                switch_matrix,
                intervention_params,
                seed + idx * 31,
                rejuvenate=False,
                rollouts=exact_rollouts,
            )
            ratio = metrics["v_b"] / max(sum(action_cost(action, state.previous_family, cfg, switch_matrix)), 1e-9)
            if ratio > best_ratio:
                best_ratio = ratio
                best_action = action
        seq = None if best_action is None else (best_action,)
        return seq, {"planner_runtime_sec": time.time() - start, "fallback_triggered": False}

    planning_switch = [[0.0, 0.0], [0.0, 0.0]] if method == "additive_h2" else switch_matrix
    if method in {"additive_h2", "switching_h2", "myopic_switching"}:
        horizon = 1 if method == "myopic_switching" else 2
        best, _ = _best_sequence(
            state,
            cfg,
            remaining_budget,
            planning_switch,
            intervention_params,
            seed,
            horizon=horizon,
            top_k=cfg["top_k"],
            rejuvenate=False,
            rollouts=exact_rollouts,
        )
        return best, {"planner_runtime_sec": time.time() - start, "fallback_triggered": False}

    if method == "exact_dp":
        branch = len(actions) ** cfg.get("exact_search_horizon", 3)
        top_k = None if branch <= exact_branch_cap else cfg["top_k"]
        best, fallback_triggered = _best_sequence(
            state,
            cfg,
            remaining_budget,
            switch_matrix,
            intervention_params,
            seed,
            horizon=cfg.get("exact_search_horizon", 3),
            top_k=top_k,
            rejuvenate=False,
            rollouts=cfg["rollouts_exact"],
        )
        return best, {"planner_runtime_sec": time.time() - start, "fallback_triggered": fallback_triggered or top_k is not None}

    raise ValueError(f"Unknown method: {method}")


def choose_action(
    method: str,
    state: PosteriorState,
    instance,
    cfg: dict,
    remaining_budget: float,
    switch_matrix: list[list[float]],
    seed: int,
    rejuvenate: bool = True,
    exact_branch_cap: int = 5000,
) -> tuple[Action | None, dict]:
    start = time.time()
    actions = feasible_actions(state, remaining_budget, cfg, switch_matrix)
    if not actions:
        return None, {"planner_runtime_sec": time.time() - start, "fallback_triggered": False}

    if method == "random_feasible":
        rng = np.random.default_rng(seed)
        action = actions[int(rng.integers(len(actions)))]
        return action, {"planner_runtime_sec": time.time() - start, "fallback_triggered": False}

    sequence, meta = plan_sequence(
        method,
        state,
        instance,
        cfg,
        remaining_budget,
        switch_matrix,
        seed,
        exact_branch_cap=exact_branch_cap,
    )
    meta["planner_runtime_sec"] = time.time() - start
    return (None if sequence is None else sequence[0]), meta


def classify_regime(state: PosteriorState, instance, cfg: dict, budget: float, switch_matrix: list[list[float]], seed: int) -> str:
    additive_seq, _ = plan_sequence("additive_h2", state, instance, cfg, budget, switch_matrix, seed)
    switching_seq, _ = plan_sequence("switching_h2", state, instance, cfg, budget, switch_matrix, seed + 1)
    if additive_seq is None or switching_seq is None:
        return "invariant"
    additive_true_cost = 0.0
    additive_search_cost = 0.0
    prev_true = state.previous_family
    prev_search = state.previous_family
    zero_switch = [[0.0, 0.0], [0.0, 0.0]]
    for action in additive_seq:
        additive_true_cost += sum(action_cost(action, prev_true, cfg, switch_matrix))
        additive_search_cost += sum(action_cost(action, prev_search, cfg, zero_switch))
        prev_true = action.family
        prev_search = action.family
    if additive_seq != switching_seq and additive_true_cost <= budget + 1e-9:
        return "value_crossover"
    if additive_search_cost <= budget and additive_true_cost > budget:
        return "feasible_change"
    return "invariant"
