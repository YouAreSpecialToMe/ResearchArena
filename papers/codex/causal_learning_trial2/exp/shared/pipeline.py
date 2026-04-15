from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import resource

import numpy as np

from .common import BATCH_SIZES, CHECKPOINTS, TOTAL_BUDGET, now, save_json
from .discovery import (
    ambiguity_stats,
    best_particle,
    build_particles,
    detectability,
    expected_entropy_reduction,
    effective_sample_size,
    git_expected_gradient_score,
    graph_posterior_entropy,
    update_particles,
)
from .metrics import auc_over_samples, directed_metrics
from .sem import SEMInstance


@dataclass
class RolloutConfig:
    method: str
    epsilon_stop: float = 0.30
    tau_stop: float = 0.90
    eta_stop: float = 0.002
    fixed_batch_size: int | None = None
    disable_detectability: bool = False
    disable_early_stop: bool = False
    shortlist_k: int | None = None


def true_edge_label(instance: SEMInstance, edge: tuple[int, int]) -> str:
    i, j = edge
    if instance.adjacency[i, j] == 1:
        return "fwd"
    if instance.adjacency[j, i] == 1:
        return "rev"
    return "none"


def target_ambiguity_mass(stats: list[dict], p: int) -> np.ndarray:
    scores = np.zeros(p, dtype=float)
    for row in stats:
        i, j = row["edge"]
        mass = row["A"] * row["R"]
        scores[i] += mass
        scores[j] += mass
    return scores


def candidate_targets_for_stats(stats: list[dict], p: int, shortlist_k: int | None) -> np.ndarray:
    if not stats:
        base = np.arange(p)
    else:
        base = np.argsort(target_ambiguity_mass(stats, p))[::-1]
    if shortlist_k:
        base = base[:shortlist_k]
    return base


def select_action(instance: SEMInstance, particles, stats: list[dict], config: RolloutConfig, budget_used: int, seed: int) -> tuple[int, int, dict]:
    p = instance.p
    if config.method == "random_active":
        rng = np.random.default_rng(seed + budget_used + 123)
        target = int(rng.integers(0, p))
        return target, 50, {"best_score": None}
    candidate_targets = candidate_targets_for_stats(stats, p, config.shortlist_k)
    best = None
    cert_rows = []
    for target in candidate_targets:
        for batch_size in BATCH_SIZES:
            if config.fixed_batch_size is not None and batch_size != config.fixed_batch_size:
                continue
            if batch_size > TOTAL_BUDGET - budget_used:
                continue
            if config.method == "git":
                score = git_expected_gradient_score(particles, int(target), batch_size, seed + 1009 * (budget_used + 1) + 17 * int(target) + batch_size)
                gain_per_sample = score / batch_size
            elif config.method == "aoed_lite":
                score = expected_entropy_reduction(particles, int(target), batch_size, seed + 2003 * (budget_used + 1) + 19 * int(target) + batch_size)
                gain_per_sample = score / batch_size
            else:
                score = 0.0
                for row in stats:
                    edge = tuple(row["edge"])
                    dval = 1.0 if config.disable_detectability else detectability(particles, edge, int(target), batch_size)[0]
                    score += row["A"] * row["R"] * dval / batch_size
                gain_per_sample = score
            meta = {"target": int(target), "batch_size": int(batch_size), "score": float(score), "gain_per_sample": float(gain_per_sample)}
            cert_rows.append(meta)
            if best is None or score > best["score"]:
                best = meta
    if best is None:
        return int(candidate_targets[0]), min(BATCH_SIZES), {"best_score": 0.0}
    return best["target"], best["batch_size"], {"best_score": best["score"], "best_gain_per_sample": best["gain_per_sample"], "candidate_scores": cert_rows}


def certificate_table(instance: SEMInstance, particles, budget_remaining: int, config: RolloutConfig) -> tuple[list[dict], float]:
    stats = ambiguity_stats(particles)
    ess = effective_sample_size(particles)
    shortlist = set(candidate_targets_for_stats(stats, instance.p, config.shortlist_k).tolist())
    rows = []
    total_cert = 0.0
    for row in stats:
        edge = tuple(row["edge"])
        best_q = -1.0
        best_action = None
        max_local = 0
        for target in range(instance.p):
            if config.shortlist_k and target not in shortlist:
                continue
            for batch_size in BATCH_SIZES:
                if batch_size > budget_remaining:
                    continue
                dval, local_size = ((1.0, 0) if config.disable_detectability else detectability(particles, edge, target, batch_size))
                max_local = max(max_local, local_size)
                if dval > best_q:
                    best_q = dval
                    best_action = [target, batch_size]
        best_q = max(best_q, 0.0)
        total_cert += row["A"] * row["R"] * best_q
        rows.append(
            {
                **row,
                "q_e": float(best_q),
                "best_action": best_action,
                "low_trust_flag": bool(ess < 4.0 or max_local > 6),
                "unlikely_to_resolve_under_budget": bool(best_q < 0.35),
                "true_label": true_edge_label(instance, edge),
                "ess": float(ess),
                "local_neighborhood_size": int(max_local),
            }
        )
    return rows, float(total_cert)


def evaluate_state(instance: SEMInstance, particles, budget_used: int, config: RolloutConfig) -> dict:
    pred = best_particle(particles).adjacency
    metrics = directed_metrics(pred, instance.adjacency)
    table, certificate_mass = certificate_table(instance, particles, TOTAL_BUDGET - budget_used, config)
    metrics.update(
        {
            "cumulative_samples": budget_used,
            "unused_budget": TOTAL_BUDGET - budget_used,
            "stop_flag": False,
            "certificate_mass": certificate_mass,
            "graph_posterior_entropy": graph_posterior_entropy(particles),
            "ess": effective_sample_size(particles),
            "certificate_table": table,
        }
    )
    return metrics


def run_rollout(instance: SEMInstance, config: RolloutConfig, out_dir: Path, seed: int, initial_particles=None) -> dict:
    observations = instance.observational_data
    interventions: list[dict] = []
    offsets = {node: 0 for node in range(instance.p)}
    start = now()
    if config.method == "fges_only":
        particles = initial_particles if initial_particles is not None else build_particles(observations, [], seed)
        metrics = evaluate_state(instance, particles, 0, config)
        metrics["runtime_seconds"] = now() - start
        metrics["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        metrics["method"] = config.method
        metrics["checkpoints"] = [{k: v for k, v in metrics.items() if k != "certificate_table"}]
        metrics["stop_flag"] = True
        metrics["rollout_seed"] = seed
        metrics["stop_reason"] = "observational_only"
        metrics["config"] = {
            "epsilon_stop": config.epsilon_stop,
            "tau_stop": config.tau_stop,
            "eta_stop": config.eta_stop,
            "fixed_batch_size": config.fixed_batch_size,
            "disable_detectability": config.disable_detectability,
            "disable_early_stop": config.disable_early_stop,
            "shortlist_k": config.shortlist_k,
        }
        save_json(
            out_dir / "rollout_log.json",
            {
                "instance_id": instance.instance_id,
                "method": config.method,
                "seed": seed,
                "checkpoints": metrics["checkpoints"],
                "stop_reason": metrics["stop_reason"],
                "final_metrics": {k: v for k, v in metrics.items() if k not in {"certificate_table", "checkpoints"}},
            },
        )
        save_json(out_dir / "results.json", metrics)
        return metrics
    particles = initial_particles if initial_particles is not None else build_particles(observations, [], seed)
    checkpoints = []
    budget_used = 0
    stopped = False
    stop_reason = "budget_exhausted"
    final_state = None
    while budget_used <= TOTAL_BUDGET:
        state = evaluate_state(instance, particles, budget_used, config)
        should_save_state = budget_used in CHECKPOINTS or budget_used == TOTAL_BUDGET
        if should_save_state:
            checkpoints.append({k: v for k, v in state.items() if k != "certificate_table"})
            save_json(out_dir / f"certificate_{budget_used:03d}.json", state["certificate_table"])
        stats = ambiguity_stats(particles)
        action_meta = None
        if budget_used < TOTAL_BUDGET:
            _, _, action_meta = select_action(instance, particles, stats, config, budget_used, seed)
        if not config.disable_early_stop and config.method in {"pacer_cert", "pacer_no_d", "pacer_fixed_batch", "aoed_lite"} and budget_used > 0:
            if config.method == "aoed_lite":
                best_gain = 0.0 if action_meta is None else float(action_meta.get("best_gain_per_sample", 0.0))
                stop = best_particle(particles).weight >= config.tau_stop or best_gain < config.eta_stop
                stop_reason = "posterior_confidence" if best_particle(particles).weight >= config.tau_stop else "low_expected_gain"
            else:
                stop = state["certificate_mass"] < config.epsilon_stop
                stop_reason = "certificate_threshold"
            if stop:
                state["stop_flag"] = True
                if not should_save_state:
                    checkpoints.append({k: v for k, v in state.items() if k != "certificate_table"})
                    save_json(out_dir / f"certificate_{budget_used:03d}.json", state["certificate_table"])
                else:
                    checkpoints[-1]["stop_flag"] = True
                final_state = state
                stopped = True
                break
        if budget_used >= TOTAL_BUDGET:
            break
        target, batch_size, action_meta = select_action(instance, particles, stats, config, budget_used, seed)
        stream = instance.intervention_streams[target]
        start_idx = offsets[target]
        end_idx = start_idx + batch_size
        batch = stream[start_idx:end_idx]
        offsets[target] = end_idx
        interventions.append({"target": target, "data": batch, "batch_size": batch_size})
        budget_used += batch_size
        particles = update_particles(particles, observations, interventions, seed + budget_used + 13)
    if final_state is None:
        final_state = evaluate_state(instance, particles, budget_used, config)
    final_state["runtime_seconds"] = now() - start
    final_state["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    final_state["method"] = config.method
    final_state["rollout_seed"] = seed
    final_state["checkpoints"] = checkpoints
    final_state["stop_flag"] = bool(stopped)
    final_state["stop_reason"] = stop_reason
    final_state["config"] = {
        "epsilon_stop": config.epsilon_stop,
        "tau_stop": config.tau_stop,
        "eta_stop": config.eta_stop,
        "fixed_batch_size": config.fixed_batch_size,
        "disable_detectability": config.disable_detectability,
        "disable_early_stop": config.disable_early_stop,
        "shortlist_k": config.shortlist_k,
    }
    final_state["auc_directed_f1"] = auc_over_samples(
        [item["cumulative_samples"] for item in checkpoints],
        [item["directed_f1"] for item in checkpoints],
        TOTAL_BUDGET,
    )
    final_state["interventions"] = [{"target": int(x["target"]), "batch_size": int(x["batch_size"])} for x in interventions]
    save_json(
        out_dir / "rollout_log.json",
        {
            "instance_id": instance.instance_id,
            "method": config.method,
            "seed": seed,
            "checkpoints": checkpoints,
            "stop_reason": stop_reason,
            "final_metrics": {k: v for k, v in final_state.items() if k not in {"certificate_table", "checkpoints"}},
        },
    )
    save_json(out_dir / "results.json", final_state)
    return final_state
