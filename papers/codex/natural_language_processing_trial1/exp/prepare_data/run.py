import time

from exp.shared.data import build_all_data
from exp.shared.train_eval import build_tokenizer, compute_train_selection
from exp.shared.utils import (
    ARTIFACTS_DIR,
    MODEL_NAME,
    PLANNING_HORIZON,
    PROOFWRITER_PROOF_DEPTH,
    ROOT,
    SPLIT_SEED,
    dump_json,
    ensure_dirs,
    run_config,
    system_info,
)


def token_count(tokenizer, items):
    return sum(
        len(tokenizer(item["input_text"] + "\n" + item["target_text"], add_special_tokens=False)["input_ids"])
        for item in items
    )


def split_stats(tokenizer, split_data):
    stats = {}
    for condition in ["full", "clean", "noisy"]:
        items = split_data[condition]
        stats[condition] = {
            "count": len(items),
            "avg_input_chars": sum(len(item["input_text"]) for item in items) / max(1, len(items)),
            "avg_target_chars": sum(len(item["target_text"]) for item in items) / max(1, len(items)),
            "token_total": token_count(tokenizer, items),
        }
    if "aux_non_unique" in split_data:
        items = split_data["aux_non_unique"]
        stats["aux_non_unique"] = {
            "count": len(items),
            "avg_input_chars": sum(len(item["input_text"]) for item in items) / max(1, len(items)),
            "avg_target_chars": sum(len(item["target_text"]) for item in items) / max(1, len(items)),
            "token_total": token_count(tokenizer, items),
        }
    return stats


def main():
    t0 = time.time()
    ensure_dirs()
    print("prepare_data: building data", flush=True)
    data = build_all_data(rebuild=False)
    print("prepare_data: loading tokenizer", flush=True)
    tokenizer = build_tokenizer()
    selections, selection_stats = compute_train_selection(data, tokenizer)

    dump_json(system_info(), ARTIFACTS_DIR / "environment.json")
    validator_spec = {
        "logic": {
            "proof_depth": PROOFWRITER_PROOF_DEPTH,
            "canonical_rule": "shortest successful proof; ties broken by lexical proof text",
            "clean_rule": "exactly one proof-local indispensable explicit leaf fact among canonical leaves",
            "noisy_rule": "remove a non-clean distractor fact from the same source example",
        },
        "planning": {
            "solver": "bounded BFS",
            "horizon": PLANNING_HORIZON,
            "candidate_hidden_facts": ["clear(x)", "handempty"],
            "clean_rule": "full state solvable, visible+fact solvable, complement completion unsolvable, unique candidate in instance under the proxy complement constructor",
            "noisy_rule": "hide another candidate fact from the same source instance that fails the clean rule",
            "verbalization_families": {"train": "A", "validation": "B", "test": "C", "robustness_eval": ["A", "B", "C"]},
        },
        "execution_deviations": [
            "Python 3.11 was unavailable; runs use Python 3.12.7.",
            "Fast Downward was unavailable locally; planning validation uses an exact bounded BFS solver.",
            "The planned human manual audit is infeasible in this environment and is replaced by a documented proxy plus a negative-claim restriction.",
            "Planning complement completions are implemented with a state-construction proxy for hidden clear/handempty facts rather than a full PDDL complement enumerator.",
        ],
    }
    dump_json(validator_spec, ARTIFACTS_DIR / "validator_spec.json")

    stats = {
        "model_name_for_token_count": MODEL_NAME,
        "split_seed": SPLIT_SEED,
        "logic": {},
        "planning": {},
        "questbench": {
            "logic_test_count": len(data["questbench"]["logic"]),
            "planning_test_count": len(data["questbench"]["planning"]),
        },
        "token_budget_by_condition": {},
    }
    for domain in ["logic", "planning"]:
        for split in ["train", "validation", "test"]:
            stats[domain][split] = split_stats(tokenizer, data[domain][split])
        stats[domain]["validator"] = data[domain]["stats"]
    stats["planning"]["robustness"] = {
        family: {
            "count": len(items),
            "token_total": token_count(tokenizer, items),
        }
        for family, items in data["planning"]["robustness"].items()
    }
    for condition, items in selections.items():
        stats["token_budget_by_condition"][condition] = {
            "examples": len(items),
            "token_total": token_count(tokenizer, items),
        }
    stats["token_budget_selection_details"] = selection_stats

    dump_json(stats, ARTIFACTS_DIR / "data_stats.json")
    dump_json(run_config(extra={"data_stats": stats}), ROOT / "exp" / "prepare_data" / "config.json")
    dump_json(
        {
            "experiment": "prepare_data",
            "runtime_seconds": time.time() - t0,
            "metrics": {
                "logic_train_clean_count": len(data["logic"]["train"]["clean"]),
                "planning_train_clean_count": len(data["planning"]["train"]["clean"]),
                "questbench_logic_count": len(data["questbench"]["logic"]),
                "questbench_planning_count": len(data["questbench"]["planning"]),
            },
            "config": {
                "split_seed": SPLIT_SEED,
                "planning_horizon": PLANNING_HORIZON,
                "proof_depth": PROOFWRITER_PROOF_DEPTH,
            },
        },
        ROOT / "exp" / "prepare_data" / "results.json",
    )


if __name__ == "__main__":
    main()
