from datasets import load_dataset

from .blocksworld import build_planning_pool
from .logic_data import (
    infer_logic_goal,
    parse_pythonish,
    parse_questbench_gt,
    questbench_logic_prompt,
    build_logic_pool,
)
from .utils import DATA_DIR, PLANNING_HORIZON, SPLIT_SEED, dump_json, load_json


def questbench_planning_prompt(example):
    return "\n".join(
        [
            "Task: ask for one missing fact if needed to determine whether the goal is achievable.",
            "Known conditions:",
            example["conditions"],
            "Goals:",
            example["goals"],
            "Output ASK: <fact> or ANSWER: no_question_needed.",
        ]
    )


def build_questbench_logic():
    ds = load_dataset("belindazli/questbench", "Logic-Q", split="test")
    out = []
    for ex in ds:
        gt_qs = parse_questbench_gt(ex["gt_qs"])
        known_true = parse_questbench_gt(ex["known_facts"])
        known_false = parse_questbench_gt(ex["known_untrue_facts"])
        rules = parse_pythonish(ex["rules"])
        goal = ex["goal"]
        max_depth = int(ex["max_depth"])
        answer_required = not (len(gt_qs) == 1 and gt_qs[0] == "No questions needed.")
        clarified_target = None
        clarified_prompt = None
        if answer_required and gt_qs:
            gold_fact = gt_qs[0]
            clarified_target = infer_logic_goal(known_true + [gold_fact], known_false, rules, goal, max_depth)
            clarified_prompt = (
                f"{questbench_logic_prompt(ex)}\nProvided missing fact: {gold_fact}\n"
                "Now answer the goal.\nOutput exactly ANSWER: true, ANSWER: false, or ANSWER: unknown."
            )
        out.append(
            {
                "domain": "questbench_logic",
                "input_text": questbench_logic_prompt(ex),
                "gt_qs": gt_qs,
                "answer_required": answer_required,
                "target_text": "ASK" if answer_required else "ANSWER: no_question_needed",
                "clarified_input_text": clarified_prompt,
                "full_target_text": clarified_target,
                "goal": goal,
            }
        )
    return out


def build_questbench_planning():
    ds = load_dataset("belindazli/questbench", "Planning-Q", split="test")
    out = []
    for ex in ds:
        gt_qs = parse_questbench_gt(ex["gt_qs"])
        answer_required = not (len(gt_qs) == 1 and gt_qs[0] == "No questions needed.")
        out.append(
            {
                "domain": "questbench_planning",
                "input_text": questbench_planning_prompt(ex),
                "gt_qs": gt_qs,
                "answer_required": answer_required,
                "target_text": "ASK" if answer_required else "ANSWER: no_question_needed",
                "all_valid_qs": parse_questbench_gt(ex["all_valid_qs"]),
                "goals": ex["goals"],
                "conditions": ex["conditions"],
            }
        )
    return out


def build_all_data(rebuild=False):
    cache_path = DATA_DIR / "prepared_data.json"
    if cache_path.exists() and not rebuild:
        return load_json(cache_path)

    logic = build_logic_pool(n_full=120, n_underspecified=120, seed=SPLIT_SEED)
    planning = build_planning_pool(
        train_full=120,
        train_underspecified=120,
        eval_full=40,
        eval_underspecified=40,
        max_depth=PLANNING_HORIZON,
        seed=SPLIT_SEED,
    )
    questbench = {
        "logic": build_questbench_logic(),
        "planning": build_questbench_planning(),
    }

    data = {
        "logic": logic,
        "planning": planning,
        "questbench": questbench,
    }
    dump_json(data, cache_path)
    return data
