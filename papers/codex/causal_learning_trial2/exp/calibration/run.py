from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from exp.shared.common import BATCH_SIZES, CALIBRATION_CONTINUATIONS, TOTAL_BUDGET, ensure_dir, load_json, save_json, set_thread_env
from exp.shared.discovery import ambiguity_stats, build_particles, detectability, edge_state_probs, update_particles
from exp.shared.metrics import calibration_metrics
from exp.shared.pipeline import RolloutConfig, candidate_targets_for_stats, certificate_table, run_rollout, true_edge_label
from exp.shared.sem import load_instance


METHOD_CONFIGS = {
    "random_active": lambda shortlist_k, epsilon_stop, tau_stop, eta_stop: RolloutConfig(method="random_active"),
    "git": lambda shortlist_k, epsilon_stop, tau_stop, eta_stop: RolloutConfig(method="git", shortlist_k=shortlist_k),
    "aoed_lite": lambda shortlist_k, epsilon_stop, tau_stop, eta_stop: RolloutConfig(method="aoed_lite", tau_stop=tau_stop, eta_stop=eta_stop, shortlist_k=shortlist_k),
    "pacer_no_d": lambda shortlist_k, epsilon_stop, tau_stop, eta_stop: RolloutConfig(method="pacer_no_d", epsilon_stop=epsilon_stop, disable_detectability=True, shortlist_k=shortlist_k),
    "pacer_cert": lambda shortlist_k, epsilon_stop, tau_stop, eta_stop: RolloutConfig(method="pacer_cert", epsilon_stop=epsilon_stop, shortlist_k=shortlist_k),
}


def run_or_load_rollout(inst, cfg, out_dir: Path, seed: int, force_rerun: bool) -> dict:
    result_path = out_dir / "results.json"
    if result_path.exists() and not force_rerun:
        return load_json(result_path)
    return run_rollout(inst, cfg, out_dir, seed)


def top_state_label(particles, edge: tuple[int, int]) -> str:
    probs = edge_state_probs(particles)[edge]
    return max([("fwd", probs["fwd"]), ("rev", probs["rev"]), ("none", probs["none"])], key=lambda x: x[1])[0]


def empirical_resolvability(instance, base_particles, base_interventions, edge: tuple[int, int], action: tuple[int, int], seed: int) -> float:
    success = 0
    true_label = true_edge_label(instance, edge)
    for k in range(CALIBRATION_CONTINUATIONS):
        interventions = [{"target": batch["target"], "data": batch["data"].copy(), "batch_size": batch["batch_size"]} for batch in base_interventions]
        particles = [copy.deepcopy(particle) for particle in base_particles]
        target, batch_size = action
        rng = np.random.default_rng(seed + 1000 + k)
        stream = instance.intervention_streams[target]
        start = int(rng.integers(0, max(len(stream) - batch_size + 1, 1)))
        batch = stream[start : start + batch_size]
        interventions.append({"target": target, "data": batch, "batch_size": batch_size})
        particles = update_particles(particles, instance.observational_data, interventions, seed + 2000 + k)
        if top_state_label(particles, edge) == true_label:
            success += 1
    return success / CALIBRATION_CONTINUATIONS


def phase_for_budget(budget_used: int) -> str | None:
    if 25 <= budget_used <= 100:
        return "early"
    if 100 < budget_used <= 200:
        return "middle"
    if 200 < budget_used <= 300:
        return "late"
    return None


def reconstruct_state_candidates(
    aux_paths: list[str],
    shortlist_k: int | None,
    epsilon_stop: float,
    tau_stop: float,
    eta_stop: float,
    force_rerun: bool,
) -> list[dict]:
    candidates = []
    for path in aux_paths:
        inst = load_instance(Path(path))
        for method, builder in METHOD_CONFIGS.items():
            cfg = builder(shortlist_k, epsilon_stop, tau_stop, eta_stop)
            out_dir = Path(__file__).resolve().parent / "saved_states" / method / inst.instance_id
            result = run_or_load_rollout(inst, cfg, out_dir, inst.seed + 19, force_rerun)
            particles = build_particles(inst.observational_data, [], inst.seed + 29)
            running_interventions = []
            running_offsets = {node: 0 for node in range(inst.p)}
            budget_used = 0
            for step_idx, record in enumerate(result["interventions"], start=1):
                start = running_offsets[record["target"]]
                end = start + record["batch_size"]
                batch = inst.intervention_streams[record["target"]][start:end]
                running_offsets[record["target"]] = end
                running_interventions.append({"target": record["target"], "data": batch, "batch_size": record["batch_size"]})
                budget_used += record["batch_size"]
                particles = update_particles(particles, inst.observational_data, running_interventions, inst.seed + budget_used + 31)
                phase = phase_for_budget(budget_used)
                stats = ambiguity_stats(particles)
                if phase is None or not stats:
                    continue
                candidates.append(
                    {
                        "instance_path": path,
                        "instance_id": inst.instance_id,
                        "method": method,
                        "budget_used": budget_used,
                        "phase": phase,
                        "step_idx": step_idx,
                        "particles": [copy.deepcopy(particle) for particle in particles],
                        "interventions": [{"target": x["target"], "data": x["data"].copy(), "batch_size": x["batch_size"]} for x in running_interventions],
                        "stats": sorted(stats, key=lambda row: row["A"] * row["R"], reverse=True),
                    }
                )
            if result.get("stop_flag") and result.get("cumulative_samples", 0) > 0:
                stop_budget = int(result["cumulative_samples"])
                phase = phase_for_budget(stop_budget)
                if phase is not None:
                    matched = [row for row in candidates if row["instance_id"] == inst.instance_id and row["method"] == method and row["budget_used"] == stop_budget]
                    if not matched:
                        stats = ambiguity_stats(particles)
                        if stats:
                            candidates.append(
                                {
                                    "instance_path": path,
                                    "instance_id": inst.instance_id,
                                    "method": method,
                                    "budget_used": stop_budget,
                                    "phase": phase,
                                    "step_idx": len(result["interventions"]),
                                    "particles": [copy.deepcopy(particle) for particle in particles],
                                    "interventions": [{"target": x["target"], "data": x["data"].copy(), "batch_size": x["batch_size"]} for x in running_interventions],
                                    "stats": sorted(stats, key=lambda row: row["A"] * row["R"], reverse=True),
                                }
                            )
    return candidates


def select_saved_states(candidates: list[dict], total_states: int = 48) -> list[dict]:
    methods = ["pacer_cert", "random_active", "git", "aoed_lite", "pacer_no_d"]
    phases = ["early", "middle", "late"]
    combo_order = [("erdos_renyi", "weak"), ("erdos_renyi", "mixed"), ("scale_free", "weak"), ("scale_free", "mixed")]
    method_quota = {method: total_states // len(methods) for method in methods}
    for method in methods[: total_states - sum(method_quota.values())]:
        method_quota[method] += 1
    meta_cache: dict[str, tuple[str, str]] = {}
    for row in candidates:
        instance_path = row["instance_path"]
        if instance_path not in meta_cache:
            inst = load_instance(Path(instance_path))
            meta_cache[instance_path] = (inst.graph_family, inst.weight_regime)

    def pick_round_robin(pool: list[dict], target: int, used: set[tuple[str, str, int]], extra_filter=None) -> list[dict]:
        buckets: dict[tuple[str, str, str, str], list[dict]] = {}
        for row in pool:
            if extra_filter is not None and not extra_filter(row):
                continue
            graph_family, weight_regime = meta_cache[row["instance_path"]]
            key = (row["phase"], graph_family, weight_regime, row["instance_id"])
            buckets.setdefault(key, []).append(row)
        for rows in buckets.values():
            rows.sort(key=lambda row: (row["budget_used"], row["step_idx"]))
        ordered_keys = [
            key
            for phase in phases
            for graph_family, weight_regime in combo_order
            for key in sorted(
                [bucket_key for bucket_key in buckets if bucket_key[:3] == (phase, graph_family, weight_regime)],
                key=lambda item: item[3],
            )
        ]
        picks = []
        progressed = True
        while len(picks) < target and progressed:
            progressed = False
            for key in ordered_keys:
                rows = buckets.get(key, [])
                while rows and (rows[0]["instance_id"], rows[0]["method"], rows[0]["budget_used"]) in used:
                    rows.pop(0)
                if not rows:
                    continue
                row = rows.pop(0)
                used.add((row["instance_id"], row["method"], row["budget_used"]))
                picks.append(row)
                progressed = True
                if len(picks) >= target:
                    break
        return picks

    chosen = []
    used = set()
    for method in methods:
        method_candidates = [row for row in candidates if row["method"] == method]
        reserved = []
        if method == "pacer_cert":
            reserved = pick_round_robin(method_candidates, target=6, used=used, extra_filter=lambda row: meta_cache[row["instance_path"]][1] == "weak")
        chosen.extend(reserved)
        remaining = method_quota[method] - len(reserved)
        if remaining > 0:
            chosen.extend(pick_round_robin(method_candidates, target=remaining, used=used))
    chosen.sort(key=lambda row: (methods.index(row["method"]), phases.index(row["phase"]), row["instance_id"], row["budget_used"], row["step_idx"]))
    return chosen[:total_states]


def select_direct_lookahead_states(candidates: list[dict], total_states: int = 6) -> list[dict]:
    phases = ["early", "middle", "late"]
    weak_candidates = [row for row in candidates if row["method"] == "pacer_cert" and "weak" in row["instance_id"]]
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in weak_candidates:
        key = (row["phase"], row["instance_id"])
        buckets.setdefault(key, []).append(row)
    for rows in buckets.values():
        rows.sort(key=lambda row: (row["budget_used"], row["step_idx"]))
    ordered_keys = [key for phase in phases for key in sorted([bucket for bucket in buckets if bucket[0] == phase], key=lambda item: item[1])]
    picked = []
    progressed = True
    while len(picked) < total_states and progressed:
        progressed = False
        for key in ordered_keys:
            rows = buckets.get(key, [])
            if not rows:
                continue
            picked.append(rows.pop(0))
            progressed = True
            if len(picked) >= total_states:
                break
    return picked


def ess_bucket(ess: float) -> str:
    if ess < 4:
        return "lt4"
    if ess < 8:
        return "4to8"
    return "gte8"


def neighborhood_bucket(size: int) -> str:
    if size <= 4:
        return "le4"
    if size <= 6:
        return "5to6"
    return "gt6"


def candidate_actions_for_state(inst, stats: list[dict], budget_used: int, shortlist_k: int | None) -> list[tuple[int, int]]:
    targets = candidate_targets_for_stats(stats, inst.p, min(shortlist_k or inst.p, 3))
    actions = []
    for target in targets.tolist():
        for batch_size in BATCH_SIZES:
            if batch_size <= TOTAL_BUDGET - budget_used:
                actions.append((int(target), int(batch_size)))
    return actions[:3]


def _evaluate_saved_state(task: dict) -> dict:
    state = task["state"]
    epsilon_stop = task["epsilon_stop"]
    shortlist_k = task["shortlist_k"]
    idx = task["idx"]
    inst = load_instance(Path(state["instance_path"]))
    stats = state["stats"][:2]
    tuples = []
    oracle_rows = []
    actions = candidate_actions_for_state(inst, state["stats"], state["budget_used"], shortlist_k)
    for row in stats:
        edge = tuple(row["edge"])
        for action in actions:
            dval = 1.0 if state["method"] == "pacer_no_d" else detectability(state["particles"], edge, action[0], action[1])[0]
            empirical = empirical_resolvability(inst, state["particles"], state["interventions"], edge, action, inst.seed + idx)
            tuples.append(
                {
                    "method": state["method"],
                    "instance_id": inst.instance_id,
                    "phase": state["phase"],
                    "budget_used": state["budget_used"],
                    "edge": list(edge),
                    "target": action[0],
                    "batch_size": action[1],
                    "predicted": dval,
                    "empirical": empirical,
                    "empirical_label": int(empirical >= 0.5),
                    "graph_family": inst.graph_family,
                    "weight_regime": inst.weight_regime,
                    "p": inst.p,
                }
            )
    if state["method"] in {"pacer_cert", "pacer_no_d"}:
        cert_cfg = RolloutConfig(
            method=state["method"],
            epsilon_stop=epsilon_stop,
            shortlist_k=shortlist_k,
            disable_detectability=(state["method"] == "pacer_no_d"),
        )
        cert_rows, _ = certificate_table(inst, state["particles"], TOTAL_BUDGET - state["budget_used"], cert_cfg)
        for row in cert_rows[: min(2, len(cert_rows))]:
            edge = tuple(row["edge"])
            edge_actions = candidate_actions_for_state(inst, [row], state["budget_used"], shortlist_k)
            empirical_scores = [empirical_resolvability(inst, state["particles"], state["interventions"], edge, action, inst.seed + idx + 99 + ii) for ii, action in enumerate(edge_actions)]
            oracle_rows.append(
                {
                    "method": state["method"],
                    "instance_id": inst.instance_id,
                    "phase": state["phase"],
                    "budget_used": state["budget_used"],
                    "edge": list(edge),
                    "q_e": row["q_e"],
                    "oracle_label": int(max(empirical_scores) >= 0.8) if empirical_scores else 0,
                    "ess": row["ess"],
                    "ess_bucket": ess_bucket(row["ess"]),
                    "local_neighborhood_size": row["local_neighborhood_size"],
                    "neighborhood_bucket": neighborhood_bucket(row["local_neighborhood_size"]),
                    "low_trust_flag": row["low_trust_flag"],
                    "graph_family": inst.graph_family,
                    "weight_regime": inst.weight_regime,
                    "best_action": row["best_action"],
                }
            )
    return {"tuples": tuples, "oracle_rows": oracle_rows}


def _direct_lookahead_task(task: dict) -> dict | None:
    state = task["state"]
    shortlist_k = task["shortlist_k"]
    idx = task["idx"]
    inst = load_instance(Path(state["instance_path"]))
    if not state["stats"]:
        return None
    edge = tuple(state["stats"][0]["edge"])
    actions = candidate_actions_for_state(inst, state["stats"], state["budget_used"], shortlist_k)
    surrogate = []
    empirical = []
    for action in actions:
        surrogate.append((action, detectability(state["particles"], edge, action[0], action[1])[0]))
        empirical.append((action, empirical_resolvability(inst, state["particles"], state["interventions"], edge, action, inst.seed + idx + 777)))
    surrogate_rank = [tuple(x[0]) for x in sorted(surrogate, key=lambda x: x[1], reverse=True)]
    empirical_rank = [tuple(x[0]) for x in sorted(empirical, key=lambda x: x[1], reverse=True)]
    surrogate_continue = float(max(score for _, score in surrogate)) >= 0.35 if surrogate else False
    empirical_continue = float(max(score for _, score in empirical)) >= 0.8 if empirical else False
    return {
        "instance_id": inst.instance_id,
        "budget_used": state["budget_used"],
        "edge": list(edge),
        "rank_agreement_top1": int(bool(surrogate_rank) and bool(empirical_rank) and surrogate_rank[0] == empirical_rank[0]),
        "stop_continue_agreement": int(surrogate_continue == empirical_continue),
        "surrogate_best": list(surrogate_rank[0]) if surrogate_rank else None,
        "direct_best": list(empirical_rank[0]) if empirical_rank else None,
        "surrogate_continue": bool(surrogate_continue),
        "direct_continue": bool(empirical_continue),
    }


def trust_rows_from_tuples(tuples_df: pd.DataFrame) -> list[dict]:
    rows = []
    if tuples_df.empty:
        return rows
    tuples_df = tuples_df.copy()
    tuples_df["abs_error"] = (tuples_df["predicted"] - tuples_df["empirical"]).abs()
    tuples_df["overconfident"] = (tuples_df["predicted"] - tuples_df["empirical"]) > 0.25
    group_specs = {
        "graph_family": ["graph_family"],
        "weight_regime": ["weight_regime"],
        "graph_family_x_regime": ["graph_family", "weight_regime"],
    }
    for name, cols in group_specs.items():
        for key, grp in tuples_df.groupby(cols):
            label = key if isinstance(key, tuple) else (key,)
            corr = grp[["predicted", "empirical"]].corr(method="spearman").iloc[0, 1] if len(grp) > 1 else np.nan
            rows.append(
                {
                    "stratum_type": name,
                    "stratum_value": "|".join(str(x) for x in label),
                    "spearman": corr,
                    "mae": float(grp["abs_error"].mean()),
                    "overconfident_rate": float(grp["overconfident"].mean()),
                    "count": int(len(grp)),
                    "trustworthy": bool((not np.isnan(corr)) and corr > 0 and grp["abs_error"].mean() <= 0.15),
                }
            )
    return rows


def trust_rows_from_oracle(oracle_df: pd.DataFrame) -> list[dict]:
    rows = []
    if oracle_df.empty:
        return rows
    for col in ["method", "ess_bucket", "neighborhood_bucket", "graph_family", "weight_regime"]:
        for key, grp in oracle_df.groupby(col):
            rows.append(
                {
                    "stratum_type": col,
                    "stratum_value": str(key),
                    **calibration_metrics(grp["oracle_label"].tolist(), grp["q_e"].tolist()),
                    "count": int(len(grp)),
                    "low_trust_flag_rate": float(grp["low_trust_flag"].mean()),
                }
            )
    return rows


def reconstruct_rollout_state(inst, result: dict, budget_used: int) -> tuple[list, list[dict]]:
    particles = build_particles(inst.observational_data, [], inst.seed + 29)
    interventions = []
    offsets = {node: 0 for node in range(inst.p)}
    running_budget = 0
    for record in result.get("interventions", []):
        if running_budget >= budget_used:
            break
        batch_size = min(int(record["batch_size"]), budget_used - running_budget)
        start = offsets[record["target"]]
        end = start + batch_size
        batch = inst.intervention_streams[record["target"]][start:end]
        offsets[record["target"]] = end
        interventions.append({"target": record["target"], "data": batch, "batch_size": batch_size})
        running_budget += batch_size
        particles = update_particles(particles, inst.observational_data, interventions, inst.seed + running_budget + 31)
    return particles, interventions


def build_appendix_certificate_artifact(
    aux_paths: list[str],
    shortlist_k: int | None,
    epsilon_stop: float,
    tau_stop: float,
    eta_stop: float,
    force_rerun: bool,
) -> tuple[pd.DataFrame, dict]:
    root = Path(__file__).resolve().parents[2]
    appendix_dir = ensure_dir(Path(__file__).resolve().parent / "appendix")
    chosen_meta = None
    chosen_table = pd.DataFrame()
    for path in aux_paths:
        inst = load_instance(Path(path))
        cfg = METHOD_CONFIGS["pacer_cert"](shortlist_k, epsilon_stop, tau_stop, eta_stop)
        out_dir = Path(__file__).resolve().parent / "saved_states" / "pacer_cert" / inst.instance_id
        result = run_or_load_rollout(inst, cfg, out_dir, inst.seed + 19, force_rerun)
        if not result.get("stop_flag") or int(result.get("cumulative_samples", 0)) <= 0:
            continue
        stop_budget = int(result["cumulative_samples"])
        particles, interventions = reconstruct_rollout_state(inst, result, stop_budget)
        cert_rows, _ = certificate_table(inst, particles, TOTAL_BUDGET - stop_budget, cfg)
        joined_rows = []
        for idx, row in enumerate(cert_rows):
            edge = tuple(row["edge"])
            actions = candidate_actions_for_state(inst, [row], stop_budget, shortlist_k)
            empirical_scores = [
                empirical_resolvability(inst, particles, interventions, edge, action, inst.seed + 5000 + idx * 17 + action_idx)
                for action_idx, action in enumerate(actions)
            ]
            joined_rows.append(
                {
                    "instance_id": inst.instance_id,
                    "budget_used": stop_budget,
                    "graph_family": inst.graph_family,
                    "weight_regime": inst.weight_regime,
                    **row,
                    "oracle_label": int(max(empirical_scores) >= 0.8) if empirical_scores else 0,
                    "oracle_best_empirical": float(max(empirical_scores)) if empirical_scores else 0.0,
                    "oracle_action_grid": actions,
                }
            )
        chosen_meta = {
            "instance_id": inst.instance_id,
            "budget_used": stop_budget,
            "graph_family": inst.graph_family,
            "weight_regime": inst.weight_regime,
            "num_rows": len(joined_rows),
        }
        chosen_table = pd.DataFrame(joined_rows)
        break
    chosen_table.to_csv(appendix_dir / "appendix_certificate_table.csv", index=False)
    (appendix_dir / "appendix_certificate_table.md").write_text(chosen_table.to_markdown(index=False))
    save_json(appendix_dir / "appendix_certificate_manifest.json", chosen_meta or {})
    return chosen_table, (chosen_meta or {})


def main() -> None:
    set_thread_env()
    force_rerun = os.environ.get("FORCE_RERUN", "0") == "1"
    root = Path(__file__).resolve().parents[2]
    ensure_dir(Path(__file__).resolve().parent / "logs")
    prep = load_json(root / "exp" / "data_prep" / "results.json")
    pilot = load_json(root / "exp" / "pilot" / "results.json")
    shortlist_k = pilot["shortlist_k"]
    epsilon_stop = pilot["best_pacer"]["epsilon_stop"]
    tau_stop = pilot["best_aoed"]["tau_stop"]
    eta_stop = pilot["best_aoed"]["eta_stop"]
    candidates = reconstruct_state_candidates(prep["aux"], shortlist_k, epsilon_stop, tau_stop, eta_stop, force_rerun)
    states = select_saved_states(candidates, total_states=48)
    save_json(
        Path(__file__).resolve().parent / "saved_state_manifest.json",
        [
            {
                "instance_id": row["instance_id"],
                "method": row["method"],
                "budget_used": row["budget_used"],
                "phase": row["phase"],
                "step_idx": row["step_idx"],
            }
            for row in states
        ],
    )
    with ProcessPoolExecutor(max_workers=2) as pool:
        evaluated = list(
            pool.map(
                _evaluate_saved_state,
                [{"state": state, "epsilon_stop": epsilon_stop, "shortlist_k": shortlist_k, "idx": idx} for idx, state in enumerate(states)],
            )
        )
    tuples = [row for chunk in evaluated for row in chunk["tuples"]]
    oracle_rows = [row for chunk in evaluated for row in chunk["oracle_rows"]]
    tuples_df = pd.DataFrame(tuples)
    oracle_df = pd.DataFrame(oracle_rows)
    tuples_df.to_csv(Path(__file__).resolve().parent / "calibration_tuples.csv", index=False)
    oracle_df.to_csv(Path(__file__).resolve().parent / "oracle_labels.csv", index=False)
    weak_direct_states = select_direct_lookahead_states(candidates, total_states=6)
    with ProcessPoolExecutor(max_workers=2) as pool:
        direct_rows = [row for row in pool.map(_direct_lookahead_task, [{"state": state, "shortlist_k": shortlist_k, "idx": idx} for idx, state in enumerate(weak_direct_states)]) if row is not None]
    pd.DataFrame(direct_rows, columns=["instance_id", "budget_used", "edge", "rank_agreement_top1", "stop_continue_agreement", "surrogate_best", "direct_best", "surrogate_continue", "direct_continue"]).to_csv(Path(__file__).resolve().parent / "direct_lookahead.csv", index=False)
    trust_d = trust_rows_from_tuples(tuples_df)
    trust_q = trust_rows_from_oracle(oracle_df)
    pd.DataFrame(trust_d).to_csv(Path(__file__).resolve().parent / "trust_diagnostics_d.csv", index=False)
    pd.DataFrame(trust_q).to_csv(Path(__file__).resolve().parent / "trust_diagnostics_q.csv", index=False)
    appendix_df, appendix_meta = build_appendix_certificate_artifact(
        prep["aux"], shortlist_k, epsilon_stop, tau_stop, eta_stop, force_rerun
    )
    payload = {
        "num_candidate_states": int(len(candidates)),
        "num_states": int(len(states)),
        "num_tuples": int(len(tuples_df)),
        "num_oracle_rows": int(len(oracle_df)),
        "d_metrics": calibration_metrics((tuples_df["empirical"] >= 0.5).astype(int).tolist(), tuples_df["predicted"].tolist()) if len(tuples_df) else {},
        "q_metrics": calibration_metrics(oracle_df["oracle_label"].tolist(), oracle_df["q_e"].tolist()) if len(oracle_df) else {},
        "q_metrics_by_method": {
            method: calibration_metrics(grp["oracle_label"].tolist(), grp["q_e"].tolist())
            for method, grp in oracle_df.groupby("method")
        }
        if len(oracle_df)
        else {},
        "d_mae": float(np.mean(np.abs(tuples_df["predicted"] - tuples_df["empirical"]))) if len(tuples_df) else None,
        "direct_lookahead": direct_rows,
        "trust_diagnostics_d": trust_d,
        "trust_diagnostics_q": trust_q,
        "appendix_certificate_rows": int(len(appendix_df)),
        "appendix_certificate_manifest": appendix_meta,
        "force_rerun": force_rerun,
    }
    save_json(Path(__file__).resolve().parent / "results.json", payload)


if __name__ == "__main__":
    main()
