from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from .config import ROOT, load_config
from .data import Instance, generate_instance, simulate_scm
from .metrics import bootstrap_ci, compute_auec, paired_bootstrap_ci
from .planners import action_cost, choose_action, classify_regime
from .posterior import DatasetRecord, dag_kl, initialize_posterior, tv_orientation_error, update_posterior


METHOD_LABELS = {
    "random_feasible": "Random feasible",
    "myopic_budgeted_gain": "Myopic budgeted gain",
    "additive_h2": "Horizon-2 additive-cost",
    "ratio_objective": "Ratio-objective selector",
    "switching_h2": "Proposed H=2 switching-aware",
    "exact_dp": "Exact finite-horizon budgeted DP",
    "myopic_switching": "Ablation B: myopic switching-aware",
}

PAIR_COLS = ["seed", "graph_family", "node_count", "intervention_regime", "budget", "switch_regime"]
RUN_COLS = PAIR_COLS + ["method", "skeleton_mode", "posterior_mode_requested"]


def _structure_for_mode(instance: Instance, skeleton_mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    if skeleton_mode == "learned":
        return instance.learned_skeleton, instance.learned_compelled, instance.meta["structure_backend"]
    compelled = np.zeros_like(instance.oracle_skeleton)
    return instance.oracle_skeleton, compelled, "oracle_skeleton"


def _calibration_brier(state, true_adj: np.ndarray) -> float:
    probs = state.orientation_probabilities()
    mask = np.triu(state.skeleton, 1) > 0
    truth = true_adj[mask].astype(float)
    return float(np.mean((probs[mask] - truth) ** 2))


def _make_summary(logs: list[dict], fallback_requested: bool, total_runtime_start: float, defaults: dict) -> dict:
    if logs:
        summary = logs[-1].copy()
    else:
        summary = defaults.copy()
        summary.update(
            {
                "round_idx": -1,
                "run_status": "no_feasible_action",
                "v_b": 0.0,
                "AUEC_partial": 0.0,
                "switch_count": 0,
                "planner_runtime_sec": 0.0,
                "total_runtime_sec": 0.0,
                "budget_spent_on_switching": 0.0,
            }
        )
    summary["is_summary"] = True
    summary["total_runtime_sec"] = time.time() - total_runtime_start
    summary["fallback_requested"] = fallback_requested
    summary["rounds_completed"] = int(len(logs))
    summary["exactness_status"] = (
        "requested_exact_and_used_exact"
        if bool(summary.get("exact_posterior_requested")) and bool(summary.get("exact_posterior_used"))
        else "requested_exact_but_fell_back"
        if bool(summary.get("exact_posterior_requested"))
        else "approximate_requested"
    )
    return summary


def _run_method(
    instance: Instance,
    method: str,
    skeleton_mode: str,
    posterior_mode: str,
    seed: int,
    cfg: dict,
    rejuvenate: bool = True,
    override_switch: list[list[float]] | None = None,
) -> tuple[list[dict], dict]:
    switch_matrix = override_switch or cfg["switch_regimes"][instance.switch_regime]
    skeleton, compelled, structure_backend = _structure_for_mode(instance, skeleton_mode)
    exact_requested = posterior_mode == "exact"
    particles = cfg["particle_count_main"] if instance.node_count in cfg["d_main"] else cfg["particle_count_stress"]
    state = initialize_posterior(skeleton, compelled, instance.obs_data, exact_requested, particles, cfg["exact_mec_cap"], seed)
    regime_label = classify_regime(state, instance, cfg, instance.budget, switch_matrix, seed + 7)
    initial_entropy = state.orientation_entropy()
    fallback_requested = exact_requested and not state.exact
    logs: list[dict] = []
    cumulative_cost = 0.0
    switching_spend = 0.0
    switch_count = 0
    total_runtime_start = time.time()

    for round_idx in range(cfg["t_max"]):
        remaining = instance.budget - cumulative_cost
        action, planner_meta = choose_action(
            method,
            state,
            instance,
            cfg,
            remaining,
            switch_matrix,
            seed + round_idx * 31,
            rejuvenate=rejuvenate,
            exact_branch_cap=cfg.get("exact_branch_cap", 5000),
        )
        if action is None:
            break
        prev_family = state.previous_family
        exec_cost, sample_cost, switch_cost = action_cost(action, prev_family, cfg, switch_matrix)
        spent = exec_cost + sample_cost + switch_cost
        if spent > remaining + 1e-9:
            break
        params = cfg["intervention_regimes"][instance.regime]
        data = simulate_scm(
            np.random.default_rng(seed * 101 + round_idx),
            instance.true_adj,
            instance.true_weights,
            instance.noise_vars,
            action.batch_size,
            {"target": action.target, "family": action.family, **params},
        )
        state = update_posterior(
            state,
            DatasetRecord(kind="intervention", target=action.target, family=action.family, data=data),
            rejuvenate=rejuvenate,
            seed=seed + 10_000 + round_idx,
        )
        if switch_cost > 0:
            switch_count += 1
            switching_spend += switch_cost
        cumulative_cost += spent
        entropy = state.orientation_entropy()
        costs = [row["cumulative_cost"] for row in logs] + [cumulative_cost]
        entropies = [row["orientation_entropy"] for row in logs] + [entropy]
        map_dag = state.map_dag()
        restricted_mask = np.triu(skeleton, 1) > 0
        true_dir = instance.true_adj[restricted_mask]
        pred_dir = map_dag[restricted_mask]
        row = {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "seed": instance.seed,
            "graph_family": instance.graph_family,
            "node_count": instance.node_count,
            "skeleton_mode": skeleton_mode,
            "posterior_mode": "exact" if state.exact else "approximate",
            "posterior_mode_requested": posterior_mode,
            "intervention_regime": instance.regime,
            "switch_regime": instance.switch_regime,
            "objective_mode": "budget_aligned" if method != "ratio_objective" else "ratio_secondary",
            "round_idx": round_idx,
            "previous_family": prev_family,
            "chosen_target": action.target,
            "chosen_family": action.family,
            "batch_size": action.batch_size,
            "immediate_exec_cost": exec_cost,
            "immediate_sample_cost": sample_cost,
            "immediate_switch_cost": switch_cost,
            "cumulative_cost": cumulative_cost,
            "remaining_budget": instance.budget - cumulative_cost,
            "orientation_entropy": entropy,
            "AUEC_partial": compute_auec(costs, entropies, initial_entropy, instance.budget),
            "oriented_edge_fraction": float(np.mean(np.maximum(state.orientation_probabilities()[restricted_mask], 1.0 - state.orientation_probabilities()[restricted_mask]) > 0.75)),
            "restricted_SHD": int(np.sum(true_dir != pred_dir)),
            "full_SHD": int(np.sum(np.abs(map_dag - instance.true_adj))),
            "switch_count": switch_count,
            "planner_runtime_sec": planner_meta["planner_runtime_sec"],
            "total_runtime_sec": time.time() - total_runtime_start,
            "ESS_before": state.ess_before,
            "ESS_after": state.ess_after,
            "MH_accept_rate": state.mh_accept_rate,
            "TV_orient_error": np.nan,
            "DAG_KL_error": np.nan,
            "calibration_brier": _calibration_brier(state, instance.true_adj),
            "feasibility_margin": remaining - spent,
            "fallback_triggered": planner_meta["fallback_triggered"],
            "run_status": "ok",
            "budget": instance.budget,
            "regime_label": regime_label,
            "initial_orientation_entropy": initial_entropy,
            "v_b": initial_entropy - entropy,
            "budget_spent_on_switching": switching_spend / max(cumulative_cost, 1e-9),
            "structure_backend": structure_backend,
            "structure_fallback": bool(instance.meta.get("structure_fallback", False)),
            "skeleton_shd": instance.meta["skeleton_shd"],
            "learned_undirected_edges": instance.meta["learned_undirected_edges"],
            "exact_posterior_requested": exact_requested,
            "exact_posterior_used": bool(state.exact),
            "particle_count": len(state.dags),
            "t_max": cfg["t_max"],
            "top_k": cfg["top_k"],
            "rollouts": cfg["rollouts_exact"] if state.exact else cfg["rollouts_main"],
            "exact_branch_cap": cfg.get("exact_branch_cap", 5000),
        }
        logs.append(row)

    defaults = {
        "method": METHOD_LABELS[method],
        "method_key": method,
        "seed": instance.seed,
        "graph_family": instance.graph_family,
        "node_count": instance.node_count,
        "skeleton_mode": skeleton_mode,
        "posterior_mode": posterior_mode,
        "posterior_mode_requested": posterior_mode,
        "intervention_regime": instance.regime,
        "switch_regime": instance.switch_regime,
        "objective_mode": "budget_aligned" if method != "ratio_objective" else "ratio_secondary",
        "budget": instance.budget,
        "regime_label": regime_label,
        "initial_orientation_entropy": initial_entropy,
        "structure_backend": structure_backend,
        "structure_fallback": bool(instance.meta.get("structure_fallback", False)),
        "skeleton_shd": instance.meta["skeleton_shd"],
        "learned_undirected_edges": instance.meta["learned_undirected_edges"],
        "exact_posterior_requested": exact_requested,
        "exact_posterior_used": bool(state.exact),
        "particle_count": len(state.dags),
        "t_max": cfg["t_max"],
        "top_k": cfg["top_k"],
        "rollouts": cfg["rollouts_exact"] if state.exact else cfg["rollouts_main"],
        "exact_branch_cap": cfg.get("exact_branch_cap", 5000),
    }
    summary = _make_summary(logs, fallback_requested, total_runtime_start, defaults)
    return logs, summary


def _write_experiment_json(name: str, out_dir: Path, summary_df: pd.DataFrame, cfg: dict, extra: dict | None = None) -> None:
    group_columns = [col for col in ["node_count", "graph_family", "intervention_regime", "budget", "switch_regime", "regime_label", "method"] if col in summary_df.columns]
    payload = {
        "experiment": name,
        "config": {
            "master_seeds_main": cfg["master_seeds_main"],
            "master_seeds_exact": cfg["master_seeds_exact"],
            "t_max": cfg["t_max"],
            "budgets_main": cfg["budgets_main"],
            "switch_regimes": cfg["switch_regimes"],
            "particle_count_main": cfg["particle_count_main"],
            "particle_count_stress": cfg["particle_count_stress"],
            "rollouts_main": cfg["rollouts_main"],
            "rollouts_exact": cfg["rollouts_exact"],
            "top_k": cfg["top_k"],
            "exact_search_horizon": cfg["exact_search_horizon"],
            "exact_branch_cap": cfg.get("exact_branch_cap", 5000),
            "cpdag_backend": cfg.get("cpdag_backend"),
        },
        "study_scope": "orientation_only_surrogate",
        "claims_guardrail": "Do not treat these results as evidence for a full CPDAG/DAG posterior pipeline.",
        "num_runs": int(len(summary_df)),
        "run_status_counts": summary_df["run_status"].value_counts(dropna=False).to_dict() if "run_status" in summary_df else {},
        "regime_counts": summary_df["regime_label"].value_counts(dropna=False).to_dict() if "regime_label" in summary_df else {},
        "methods": {},
        "group_summaries": [],
        "posterior_mode_counts": summary_df["posterior_mode"].value_counts(dropna=False).to_dict() if "posterior_mode" in summary_df else {},
        "structure_backend_counts": summary_df["structure_backend"].value_counts(dropna=False).to_dict() if "structure_backend" in summary_df else {},
        "exactness_status_counts": summary_df["exactness_status"].value_counts(dropna=False).to_dict() if "exactness_status" in summary_df else {},
    }
    if "exact_posterior_requested" in summary_df:
        payload["exact_posterior_fallback_rate"] = float(np.mean(summary_df["exact_posterior_requested"] & ~summary_df["exact_posterior_used"]))
    if "fallback_triggered" in summary_df:
        payload["planner_fallback_rate"] = float(summary_df["fallback_triggered"].fillna(False).mean())
    for method, group in summary_df.groupby("method"):
        payload["methods"][method] = {
            "n": int(len(group)),
            "v_b": {
                "mean": float(group["v_b"].mean()),
                "median": float(group["v_b"].median()),
                "std": float(group["v_b"].std(ddof=1)) if len(group) > 1 else 0.0,
                "ci95": list(bootstrap_ci(group["v_b"].tolist(), seed=0)),
            },
            "AUEC_partial": {
                "mean": float(group["AUEC_partial"].mean()),
                "median": float(group["AUEC_partial"].median()),
                "std": float(group["AUEC_partial"].std(ddof=1)) if len(group) > 1 else 0.0,
                "ci95": list(bootstrap_ci(group["AUEC_partial"].tolist(), seed=0)),
            },
            "oriented_edge_fraction": {
                "mean": float(group["oriented_edge_fraction"].mean()) if "oriented_edge_fraction" in group else 0.0,
                "std": float(group["oriented_edge_fraction"].std(ddof=1)) if "oriented_edge_fraction" in group and len(group) > 1 else 0.0,
            },
            "restricted_SHD": {
                "mean": float(group["restricted_SHD"].mean()) if "restricted_SHD" in group else 0.0,
                "std": float(group["restricted_SHD"].std(ddof=1)) if "restricted_SHD" in group and len(group) > 1 else 0.0,
            },
            "switch_count_mean": float(group["switch_count"].mean()) if "switch_count" in group else 0.0,
            "runtime_sec_mean": float(group["total_runtime_sec"].mean()) if "total_runtime_sec" in group else 0.0,
            "fallback_triggered_rate": float(group["fallback_triggered"].fillna(False).mean()) if "fallback_triggered" in group else 0.0,
            "fallback_requested_rate": float(group["fallback_requested"].fillna(False).mean()) if "fallback_requested" in group else 0.0,
            "tv_orient_error_mean": float(group["TV_orient_error"].dropna().mean()) if "TV_orient_error" in group else float("nan"),
            "dag_kl_error_mean": float(group["DAG_KL_error"].dropna().mean()) if "DAG_KL_error" in group else float("nan"),
            "calibration_brier_mean": float(group["calibration_brier"].dropna().mean()) if "calibration_brier" in group else float("nan"),
            "exact_posterior_rate": float(group["exact_posterior_used"].mean()) if "exact_posterior_used" in group else float("nan"),
            "structure_fallback_rate": float(group["structure_fallback"].mean()) if "structure_fallback" in group else float("nan"),
        }
    if group_columns:
        grouped = (
            summary_df.groupby(group_columns, dropna=False)
            .agg(
                n=("method", "size"),
                v_b_mean=("v_b", "mean"),
                v_b_std=("v_b", "std"),
                AUEC_mean=("AUEC_partial", "mean"),
                AUEC_std=("AUEC_partial", "std"),
                switch_count_mean=("switch_count", "mean"),
                runtime_sec_mean=("total_runtime_sec", "mean"),
            )
            .reset_index()
        )
        payload["group_summaries"] = grouped.fillna(0.0).to_dict(orient="records")
    if extra:
        payload.update(extra)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_run_logs(df: pd.DataFrame, out_dir: Path) -> None:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    run_df = df.copy()
    for col in RUN_COLS + ["round_idx", "is_summary"]:
        if col not in run_df.columns:
            run_df[col] = "na" if col != "round_idx" else -1
    for keys, group in run_df.groupby(RUN_COLS, dropna=False):
        seed, graph_family, node_count, regime, budget, switch_regime, method, skeleton_mode, posterior_mode_requested = keys
        slug = f"{method.replace(' ', '_')}_seed{seed}_{graph_family}_d{node_count}_{regime}_B{budget:g}_{switch_regime}_{skeleton_mode}_{posterior_mode_requested}"
        path = logs_dir / f"{slug}.jsonl"
        ordered = group.sort_values(by=["is_summary", "round_idx"], kind="stable")
        with path.open("w", encoding="utf-8") as f:
            for row in ordered.to_dict(orient="records"):
                f.write(json.dumps(row) + "\n")


def _paired_stats(df: pd.DataFrame, left_method: str, right_method: str) -> dict | None:
    left = df[df["method"] == left_method][PAIR_COLS + ["AUEC_partial", "v_b"]]
    right = df[df["method"] == right_method][PAIR_COLS + ["AUEC_partial", "v_b"]]
    merged = left.merge(right, on=PAIR_COLS, suffixes=("_left", "_right"))
    if merged.empty:
        return None
    diffs_auec = merged["AUEC_partial_left"] - merged["AUEC_partial_right"]
    diffs_vb = merged["v_b_left"] - merged["v_b_right"]
    stats = {
        "n": int(len(merged)),
        "AUEC_partial": {
            "mean_diff": float(diffs_auec.mean()),
            "median_diff": float(diffs_auec.median()),
            "ci95": list(paired_bootstrap_ci(diffs_auec.tolist(), seed=0)),
        },
        "v_b": {
            "mean_diff": float(diffs_vb.mean()),
            "median_diff": float(diffs_vb.median()),
            "ci95": list(paired_bootstrap_ci(diffs_vb.tolist(), seed=1)),
        },
    }
    for metric_name, values in [("AUEC_partial", diffs_auec), ("v_b", diffs_vb)]:
        try:
            stats[metric_name]["wilcoxon_p"] = float(wilcoxon(values).pvalue)
        except ValueError:
            stats[metric_name]["wilcoxon_p"] = None
    return stats


def _main_task(cfg: dict, d: int, graph_family: str, regime: str, budget: float, switch_regime: str, seed: int, methods: list[str]) -> list[tuple[list[dict], dict]]:
    instance = generate_instance(seed, graph_family, d, regime, budget, switch_regime, cfg)
    return [_run_method(instance, method, "learned", "approximate", seed + 13, cfg) for method in methods]


def _generate_exact_instance(cfg: dict, graph_family: str, regime: str, budget: float, switch_regime: str, seed: int) -> Instance:
    for attempt in range(32):
        trial_seed = seed + attempt * 1000
        instance = generate_instance(trial_seed, graph_family, cfg["d_exact"], regime, budget, switch_regime, cfg)
        skeleton = instance.oracle_skeleton
        compelled = np.zeros_like(skeleton)
        state = initialize_posterior(skeleton, compelled, instance.obs_data, True, cfg["particle_count_main"], cfg["exact_mec_cap"], trial_seed + 999)
        if state.exact:
            instance.seed = seed
            instance.meta["exact_trial_seed"] = trial_seed
            return instance
    raise RuntimeError(f"Failed to generate an exact d={cfg['d_exact']} instance within the MEC cap for seed={seed}.")


def _exact_task(cfg: dict, graph_family: str, regime: str, budget: float, switch_regime: str, seed: int, methods: list[str]) -> list[tuple[list[dict], dict]]:
    instance = _generate_exact_instance(cfg, graph_family, regime, budget, switch_regime, seed)
    return [_run_method(instance, method, "oracle", "exact", seed + 29, cfg) for method in methods]


def _skeleton_audit_task(cfg: dict, graph_family: str, regime: str, switch_regime: str, seed: int, skeleton_mode: str, method: str) -> dict:
    instance = generate_instance(seed, graph_family, 15, regime, 14.0, switch_regime, cfg)
    _, summary = _run_method(instance, method, skeleton_mode, "approximate", seed + 202, cfg)
    summary["audit_type"] = "skeleton"
    return summary


def _posterior_audit_task(cfg: dict, graph_family: str, regime: str, switch_regime: str, seed: int, method: str) -> list[dict]:
    instance = _generate_exact_instance(cfg, graph_family, regime, 14.0, switch_regime, seed)
    return _run_dual_posterior_audit(instance, method, cfg, seed + 101)


def _ablation_task(cfg: dict, graph_family: str, regime: str, switch_regime: str, seed: int, method: str, rejuvenate: bool, override_switch: list[list[float]] | None, tag: str) -> dict:
    instance = generate_instance(seed, graph_family, 15, regime, 14.0, switch_regime, cfg)
    _, summary = _run_method(instance, method, "learned", "approximate", seed + 303, cfg, rejuvenate=rejuvenate, override_switch=override_switch)
    summary["ablation_tag"] = tag
    return summary


def run_main_benchmark(out_dir: Path) -> pd.DataFrame:
    cfg = load_config()
    tasks = []
    methods_full = ["myopic_budgeted_gain", "additive_h2", "switching_h2"]
    for d in cfg["d_main"]:
        for graph_family in cfg["graph_families"]:
            for regime in cfg["intervention_regimes"]:
                for budget in cfg["budgets_main"]:
                    for switch_regime in cfg["switch_regimes"]:
                        for seed in cfg["master_seeds_main"]:
                            tasks.append((d, graph_family, regime, budget, switch_regime, seed, list(methods_full)))
    task_outputs = Parallel(n_jobs=cfg["n_jobs"], prefer="processes")(
        delayed(_main_task)(cfg, d, graph_family, regime, budget, switch_regime, seed, methods)
        for d, graph_family, regime, budget, switch_regime, seed, methods in tasks
    )
    rows: list[dict] = []
    for batch in task_outputs:
        for logs, summary in batch:
            rows.extend(logs)
            rows.append(summary)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[df["is_summary"] != True].to_csv(out_dir / "round_logs.csv", index=False)
    summary_df = df[df["is_summary"] == True].copy()
    summary_df.to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df, out_dir)
    pairwise = {
        "proposed_vs_additive": _paired_stats(summary_df, METHOD_LABELS["switching_h2"], METHOD_LABELS["additive_h2"]),
        "proposed_vs_myopic": _paired_stats(summary_df, METHOD_LABELS["switching_h2"], METHOD_LABELS["myopic_budgeted_gain"]),
    }
    _write_experiment_json("main_benchmark", out_dir, summary_df, cfg, extra={"paired_comparisons": pairwise})
    return df


def run_exact_validation(out_dir: Path) -> pd.DataFrame:
    cfg = load_config()
    methods = ["random_feasible", "myopic_budgeted_gain", "additive_h2", "ratio_objective", "switching_h2", "exact_dp"]
    tasks = []
    for graph_family in cfg["graph_families"]:
        for regime in cfg["intervention_regimes"]:
            for budget in cfg["budgets_main"]:
                for switch_regime in ["S2", "S3"]:
                    for seed in cfg["master_seeds_exact"]:
                        tasks.append((graph_family, regime, budget, switch_regime, seed, methods))
    task_outputs = Parallel(n_jobs=cfg["n_jobs"], prefer="processes")(
        delayed(_exact_task)(cfg, graph_family, regime, budget, switch_regime, seed, methods)
        for graph_family, regime, budget, switch_regime, seed, methods in tasks
    )
    rows: list[dict] = []
    for batch in task_outputs:
        for logs, summary in batch:
            rows.extend(logs)
            rows.append(summary)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df[df["is_summary"] != True].to_csv(out_dir / "round_logs.csv", index=False)
    summary_df = df[df["is_summary"] == True].copy()
    summary_df.to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df, out_dir)
    pairwise = {
        "proposed_vs_additive": _paired_stats(summary_df, METHOD_LABELS["switching_h2"], METHOD_LABELS["additive_h2"]),
        "exact_dp_vs_proposed": _paired_stats(summary_df, METHOD_LABELS["exact_dp"], METHOD_LABELS["switching_h2"]),
        "exact_dp_vs_additive": _paired_stats(summary_df, METHOD_LABELS["exact_dp"], METHOD_LABELS["additive_h2"]),
    }
    _write_experiment_json("exact_validation", out_dir, summary_df, cfg, extra={"paired_comparisons": pairwise})
    return df


def _run_dual_posterior_audit(instance: Instance, method: str, cfg: dict, seed: int) -> list[dict]:
    switch_matrix = cfg["switch_regimes"][instance.switch_regime]
    skeleton = instance.oracle_skeleton
    compelled = np.zeros_like(skeleton)
    approx_state = initialize_posterior(skeleton, compelled, instance.obs_data, False, cfg["particle_count_main"], cfg["exact_mec_cap"], seed)
    exact_state = initialize_posterior(skeleton, compelled, instance.obs_data, True, cfg["particle_count_main"], cfg["exact_mec_cap"], seed)
    rows: list[dict] = []
    cumulative_cost = 0.0
    for round_idx in range(cfg["t_max"]):
        remaining = instance.budget - cumulative_cost
        approx_action, _ = choose_action(method, approx_state, instance, cfg, remaining, switch_matrix, seed + round_idx)
        exact_action, _ = choose_action(method, exact_state, instance, cfg, remaining, switch_matrix, seed + round_idx)
        dp_action, dp_meta = choose_action(
            "exact_dp",
            exact_state,
            instance,
            cfg,
            remaining,
            switch_matrix,
            seed + round_idx,
            exact_branch_cap=cfg.get("exact_branch_cap", 5000),
        )
        rows.append(
            {
                "audit_type": "posterior",
                "method": METHOD_LABELS[method],
                "seed": instance.seed,
                "graph_family": instance.graph_family,
                "node_count": instance.node_count,
                "intervention_regime": instance.regime,
                "switch_regime": instance.switch_regime,
                "budget": instance.budget,
                "round_idx": round_idx,
                "TV_orient_error": tv_orientation_error(approx_state, exact_state),
                "DAG_KL_error": dag_kl(approx_state, exact_state),
                "calibration_brier": _calibration_brier(approx_state, instance.true_adj),
                "approx_action": None if approx_action is None else f"{approx_action.family}:{approx_action.target}:{approx_action.batch_size}",
                "exact_action": None if exact_action is None else f"{exact_action.family}:{exact_action.target}:{exact_action.batch_size}",
                "exact_dp_action": None if dp_action is None else f"{dp_action.family}:{dp_action.target}:{dp_action.batch_size}",
                "first_action_agreement_exact": bool(approx_action == exact_action),
                "first_action_agreement_exact_dp": bool(approx_action == dp_action),
                "fallback_triggered": dp_meta["fallback_triggered"],
                "posterior_mode": "approx_vs_exact",
                "structure_backend": "oracle_skeleton",
            }
        )
        if approx_action is None:
            break
        prev_family = approx_state.previous_family
        params = cfg["intervention_regimes"][instance.regime]
        data = simulate_scm(
            np.random.default_rng(seed * 131 + round_idx),
            instance.true_adj,
            instance.true_weights,
            instance.noise_vars,
            approx_action.batch_size,
            {"target": approx_action.target, "family": approx_action.family, **params},
        )
        rec = DatasetRecord(kind="intervention", target=approx_action.target, family=approx_action.family, data=data)
        approx_state = update_posterior(approx_state, rec, rejuvenate=True, seed=seed + 10_000 + round_idx)
        exact_state = update_posterior(exact_state, rec, rejuvenate=False, seed=seed + 20_000 + round_idx)
        cumulative_cost += sum(action_cost(approx_action, prev_family, cfg, switch_matrix))
    return rows


def run_audits(out_dir: Path) -> pd.DataFrame:
    cfg = load_config()
    rows: list[dict] = []
    posterior_tasks = []
    skeleton_tasks = []
    for graph_family in cfg["graph_families"]:
        for regime in cfg["intervention_regimes"]:
            for switch_regime in ["S2", "S3"]:
                for seed in cfg["master_seeds_exact"]:
                    posterior_tasks.append((graph_family, regime, switch_regime, seed, "additive_h2"))
                    posterior_tasks.append((graph_family, regime, switch_regime, seed, "switching_h2"))
                    for skeleton_mode in ["learned", "oracle"]:
                        for method in ["additive_h2", "switching_h2"]:
                            skeleton_tasks.append((graph_family, regime, switch_regime, seed, skeleton_mode, method))
    posterior_outputs = Parallel(n_jobs=cfg["n_jobs"], prefer="processes")(
        delayed(_posterior_audit_task)(cfg, graph_family, regime, switch_regime, seed, method)
        for graph_family, regime, switch_regime, seed, method in posterior_tasks
    )
    for batch in posterior_outputs:
        rows.extend(batch)
    skeleton_outputs = Parallel(n_jobs=cfg["n_jobs"], prefer="processes")(
        delayed(_skeleton_audit_task)(cfg, graph_family, regime, switch_regime, seed, skeleton_mode, method)
        for graph_family, regime, switch_regime, seed, skeleton_mode, method in skeleton_tasks
    )
    rows.extend(skeleton_outputs)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "round_logs.csv", index=False)
    summary_rows = []
    if "audit_type" in df.columns:
        for (audit_type, method), group in df.groupby(["audit_type", "method"], dropna=False):
            summary_rows.append(
                {
                    "audit_type": audit_type,
                    "method": method,
                    "n_rows": int(len(group)),
                    "TV_orient_error": float(group["TV_orient_error"].dropna().mean()) if "TV_orient_error" in group else np.nan,
                    "DAG_KL_error": float(group["DAG_KL_error"].dropna().mean()) if "DAG_KL_error" in group else np.nan,
                    "calibration_brier": float(group["calibration_brier"].dropna().mean()) if "calibration_brier" in group else np.nan,
                    "AUEC_partial": float(group["AUEC_partial"].dropna().mean()) if "AUEC_partial" in group else np.nan,
                    "v_b": float(group["v_b"].dropna().mean()) if "v_b" in group else np.nan,
                    "posterior_mode": group["posterior_mode"].iloc[0] if "posterior_mode" in group else "audit",
                    "structure_backend": group["structure_backend"].iloc[0] if "structure_backend" in group else "audit",
                    "run_status": "ok",
                }
            )
    pd.DataFrame(summary_rows).to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df.assign(is_summary=True, posterior_mode_requested="audit", skeleton_mode=df.get("skeleton_mode", "audit")), out_dir)
    skeleton_df = df[df["audit_type"] == "skeleton"].copy() if "audit_type" in df.columns else df.iloc[0:0].copy()
    tv_mean = float(df["TV_orient_error"].dropna().mean())
    first_action_dp = float(df["first_action_agreement_exact_dp"].dropna().mean())
    oracle_df = skeleton_df[skeleton_df["skeleton_mode"] == "oracle"].copy() if "skeleton_mode" in skeleton_df.columns else skeleton_df.iloc[0:0].copy()
    learned_df = skeleton_df[skeleton_df["skeleton_mode"] == "learned"].copy() if "skeleton_mode" in skeleton_df.columns else skeleton_df.iloc[0:0].copy()
    skeleton_retention = None
    if not oracle_df.empty and not learned_df.empty:
        merged = learned_df[PAIR_COLS + ["method", "AUEC_partial"]].merge(
            oracle_df[PAIR_COLS + ["method", "AUEC_partial"]],
            on=PAIR_COLS + ["method"],
            suffixes=("_learned", "_oracle"),
        )
        if not merged.empty:
            merged["retention"] = merged["AUEC_partial_learned"] / np.maximum(np.abs(merged["AUEC_partial_oracle"]), 1e-9)
            skeleton_retention = float(merged["retention"].mean())
    _write_experiment_json(
        "audits",
        out_dir,
        skeleton_df if not skeleton_df.empty else df,
        cfg,
        extra={
            "posterior_audit": {
                "tv_error_mean": tv_mean,
                "tv_error_ci95": list(bootstrap_ci(df["TV_orient_error"].dropna().tolist(), seed=0)),
                "dag_kl_mean": float(df["DAG_KL_error"].dropna().mean()),
                "first_action_agreement_exact_mean": float(df["first_action_agreement_exact"].dropna().mean()),
                "first_action_agreement_exact_dp_mean": first_action_dp,
                "thresholds": {
                    "tv_orient_error_max": 0.05,
                    "first_action_agreement_exact_dp_min": 0.85,
                },
                "threshold_failures": {
                    "tv_orient_error": bool(tv_mean >= 0.05),
                    "first_action_agreement_exact_dp": bool(first_action_dp < 0.85),
                },
            },
            "skeleton_audit": {
                "learned_vs_oracle_effect_retention_mean": skeleton_retention,
                "retention_threshold_min": 0.5,
                "threshold_failure": None if skeleton_retention is None else bool(skeleton_retention < 0.5),
            },
        },
    )
    return df


def run_ablations(out_dir: Path) -> pd.DataFrame:
    cfg = load_config()
    tasks = []
    for graph_family in cfg["graph_families"]:
        for regime in cfg["intervention_regimes"]:
            for switch_regime in ["S2", "S3"]:
                for seed in cfg["master_seeds_exact"]:
                    experiments = [
                        ("switching_h2", True, None, "proposed"),
                        ("additive_h2", True, None, "ablation_a"),
                        ("myopic_switching", True, None, "ablation_b"),
                        ("switching_h2", False, None, "ablation_c"),
                        ("switching_h2", True, cfg["switch_regimes_ablation"]["S2_sym"], "ablation_d"),
                    ]
                    for method, rejuvenate, override_switch, tag in experiments:
                        tasks.append((graph_family, regime, switch_regime, seed, method, rejuvenate, override_switch, tag))
    rows = Parallel(n_jobs=cfg["n_jobs"], prefer="processes")(
        delayed(_ablation_task)(cfg, graph_family, regime, switch_regime, seed, method, rejuvenate, override_switch, tag)
        for graph_family, regime, switch_regime, seed, method, rejuvenate, override_switch, tag in tasks
    )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    _write_run_logs(df.assign(is_summary=True, posterior_mode_requested="ablation", skeleton_mode="learned"), out_dir)
    paired = {}
    tagged_df = df.copy()
    tagged_df["method"] = tagged_df["ablation_tag"]
    for tag in sorted(tagged_df["ablation_tag"].unique()):
        if tag == "proposed":
            continue
        paired[f"proposed_vs_{tag}"] = _paired_stats(tagged_df, "proposed", tag)
    _write_experiment_json("ablations", out_dir, df, cfg, extra={"paired_comparisons": paired})
    return df


def aggregate_all(results: dict[str, pd.DataFrame]) -> dict:
    main_summary = results["main_benchmark"]
    exact_summary = results["exact_validation"]
    audit_summary = results["audits"]
    ablation_summary = results["ablations"]
    main_summary = main_summary[main_summary["is_summary"] == True].copy()
    exact_summary = exact_summary[exact_summary["is_summary"] == True].copy()
    summary = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": os.popen(f"{ROOT}/.venv/bin/python -V").read().strip(),
            "available_gpus": 0,
            "master_seeds_main": load_config()["master_seeds_main"],
            "master_seeds_exact": load_config()["master_seeds_exact"],
        },
        "experiments": {},
        "paired_comparisons": {},
        "audit_summary": {
            "tv_error_mean": float(audit_summary["TV_orient_error"].dropna().mean()) if "TV_orient_error" in audit_summary else None,
            "dag_kl_mean": float(audit_summary["DAG_KL_error"].dropna().mean()) if "DAG_KL_error" in audit_summary else None,
            "first_action_agreement_exact_mean": float(audit_summary["first_action_agreement_exact"].dropna().mean()) if "first_action_agreement_exact" in audit_summary else None,
            "first_action_agreement_exact_dp_mean": float(audit_summary["first_action_agreement_exact_dp"].dropna().mean()) if "first_action_agreement_exact_dp" in audit_summary else None,
            "threshold_failures": {
                "tv_orient_error": bool(float(audit_summary["TV_orient_error"].dropna().mean()) >= 0.05) if "TV_orient_error" in audit_summary and not audit_summary["TV_orient_error"].dropna().empty else None,
                "first_action_agreement_exact_dp": bool(float(audit_summary["first_action_agreement_exact_dp"].dropna().mean()) < 0.85) if "first_action_agreement_exact_dp" in audit_summary and not audit_summary["first_action_agreement_exact_dp"].dropna().empty else None,
            },
        },
    }
    for name, df in [("main_benchmark", main_summary), ("exact_validation", exact_summary), ("audits", audit_summary), ("ablations", ablation_summary)]:
        summary["experiments"][name] = {
            "rows": int(len(df)),
            "v_b_mean": float(df["v_b"].mean()) if "v_b" in df else None,
            "v_b_std": float(df["v_b"].std(ddof=1)) if "v_b" in df and len(df) > 1 else None,
            "AUEC_mean": float(df["AUEC_partial"].mean()) if "AUEC_partial" in df else None,
            "AUEC_std": float(df["AUEC_partial"].std(ddof=1)) if "AUEC_partial" in df and len(df) > 1 else None,
            "posterior_mode_counts": df["posterior_mode"].value_counts(dropna=False).to_dict() if "posterior_mode" in df else {},
            "structure_backend_counts": df["structure_backend"].value_counts(dropna=False).to_dict() if "structure_backend" in df else {},
            "regime_counts": df["regime_label"].value_counts(dropna=False).to_dict() if "regime_label" in df else {},
            "run_status_counts": df["run_status"].value_counts(dropna=False).to_dict() if "run_status" in df else {},
            "methods": {
                method: {
                    "n": int(len(group)),
                    "v_b_mean": float(group["v_b"].mean()) if "v_b" in group else None,
                    "v_b_std": float(group["v_b"].std(ddof=1)) if "v_b" in group and len(group) > 1 else None,
                    "AUEC_mean": float(group["AUEC_partial"].mean()) if "AUEC_partial" in group else None,
                    "AUEC_std": float(group["AUEC_partial"].std(ddof=1)) if "AUEC_partial" in group and len(group) > 1 else None,
                }
                for method, group in (df.groupby("method") if "method" in df else [])
            },
        }
    key_pairs = [
        ("main_proposed_vs_additive", main_summary, METHOD_LABELS["switching_h2"], METHOD_LABELS["additive_h2"]),
        ("main_proposed_vs_myopic", main_summary, METHOD_LABELS["switching_h2"], METHOD_LABELS["myopic_budgeted_gain"]),
        ("exact_proposed_vs_additive", exact_summary, METHOD_LABELS["switching_h2"], METHOD_LABELS["additive_h2"]),
        ("exact_dp_vs_proposed", exact_summary, METHOD_LABELS["exact_dp"], METHOD_LABELS["switching_h2"]),
    ]
    for name, df, left, right in key_pairs:
        stats = _paired_stats(df, left, right)
        if stats is not None:
            summary["paired_comparisons"][name] = stats
    return summary


def save_table(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    with path_base.with_suffix(".tex").open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))
