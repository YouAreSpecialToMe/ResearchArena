import random
import time

from exp.shared.blocksworld import bfs_solve, candidate_hidden_facts, clean_candidate_metadata
from exp.shared.data import build_all_data
from exp.shared.utils import (
    AUDIT_SEED,
    PLANNING_HORIZON,
    ROOT,
    bootstrap_ci,
    dump_json,
    run_config,
    wilson_interval,
)


def sample_items(items, n=50):
    rng = random.Random(AUDIT_SEED)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    return [items[idx] for idx in idxs[: min(n, len(items))]]


def audit_logic_item(item, condition):
    hidden = item["hidden_fact"]
    canonical = item.get("canonical_triples", [])
    required = bool(condition == "clean" and item.get("num_clean_candidates", 0) == 1 and hidden in item.get("valid_asks", []))
    if condition == "noisy":
        required = item.get("removed_triple_id") in canonical
    return {
        "qid": item["qid"],
        "condition": condition,
        "hidden_fact": hidden,
        "required_under_executable_spec": bool(required),
        "alternate_hidden_fact_available": bool(item.get("num_clean_candidates", 0) > 1),
        "well_formed_target": item["target_text"].startswith("ASK: "),
    }


def planning_clean_count(item):
    state = tuple(item["state"])
    goal = tuple(item["goal"])
    count = 0
    for fact in candidate_hidden_facts(state):
        meta = clean_candidate_metadata(state, goal, fact, PLANNING_HORIZON)
        if meta and meta["full_solvable"] and meta["visible_plus_fact_solvable"] and not meta["complement_solvable"]:
            count += 1
    return count


def audit_planning_item(item, condition):
    state = tuple(item["state"])
    goal = tuple(item["goal"])
    meta = clean_candidate_metadata(state, goal, item["hidden_fact"], PLANNING_HORIZON)
    required = bool(meta and meta["full_solvable"] and meta["visible_plus_fact_solvable"] and not meta["complement_solvable"])
    if condition == "clean":
        required = required and planning_clean_count(item) == 1
    return {
        "condition": condition,
        "hidden_fact": item["hidden_fact"],
        "required_under_executable_spec": required,
        "alternate_hidden_fact_available": planning_clean_count(item) > 1,
        "well_formed_target": item["target_text"].startswith("ASK: "),
    }


def summarize(audited):
    n = len(audited)
    success = sum(int(item["required_under_executable_spec"]) for item in audited)
    well_formed = sum(int(item["well_formed_target"]) for item in audited)
    return {
        "n": n,
        "precision": success / max(1, n),
        "precision_wilson": wilson_interval(success, max(1, n)),
        "well_formed_rate": well_formed / max(1, n),
        "well_formed_wilson": wilson_interval(well_formed, max(1, n)),
        "alternate_hidden_fact_rate": sum(int(item["alternate_hidden_fact_available"]) for item in audited) / max(1, n),
    }


def residual_logic(data):
    items = data["logic"]["test"]["clean"][:200]
    flagged = [int(len(item.get("all_candidate_proofs", [])) > 1) for item in items]
    return {
        "n": len(items),
        "alternative_proof_rate": sum(flagged) / max(1, len(flagged)),
        "bootstrap_ci": bootstrap_ci(flagged),
    }


def residual_planning(data):
    items = data["planning"]["test"]["clean"][:200]
    flagged = []
    for item in items:
        complement = tuple(item["complement_state"])
        goal = tuple(item["goal"])
        solvable, _ = bfs_solve(complement, goal, max_depth=PLANNING_HORIZON + 2)
        flagged.append(int(solvable))
    return {
        "n": len(items),
        "longer_horizon_instability_rate": sum(flagged) / max(1, len(flagged)),
        "bootstrap_ci": bootstrap_ci(flagged),
    }


def main():
    t0 = time.time()
    data = build_all_data()
    proxy_outputs = {}
    sample_artifacts = {}
    for domain in ["logic", "planning"]:
        proxy_outputs[domain] = {}
        sample_artifacts[domain] = {}
        for condition in ["clean", "noisy"]:
            items = sample_items(data[domain]["train"][condition], n=50)
            audited = [audit_logic_item(item, condition) for item in items] if domain == "logic" else [audit_planning_item(item, condition) for item in items]
            proxy_outputs[domain][condition] = summarize(audited)
            sample_artifacts[domain][condition] = audited
    residual = {"logic": residual_logic(data), "planning": residual_planning(data)}
    dump_json(sample_artifacts, ROOT / "artifacts" / "audit_samples.json")
    dump_json(
        {
            "experiment": "manual_audit",
            "runtime_seconds": time.time() - t0,
            "human_audit": {
                "status": "not_run_requires_human_reviewer",
                "reason": "A real human audit cannot be executed autonomously from this environment.",
            },
            "proxy_audit_summary": proxy_outputs,
            "residual_noise": residual,
            "config": run_config(extra={"audit_seed": AUDIT_SEED, "audit_type": "proxy_due_to_missing_human_reviewer"}),
            "notes": [
                "The preregistered human manual audit was infeasible here and remains unmet.",
                "Results should therefore be treated as a negative or narrowed pilot rather than confirmatory evidence about label cleanliness.",
            ],
        },
        ROOT / "exp" / "manual_audit" / "results.json",
    )
    dump_json(run_config(extra={"audit_seed": AUDIT_SEED}), ROOT / "exp" / "manual_audit" / "config.json")


if __name__ == "__main__":
    main()
