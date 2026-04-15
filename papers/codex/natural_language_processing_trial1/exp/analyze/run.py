from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exp.shared.data import build_all_data
from exp.shared.train_eval import (
    batch_generate_loaded,
    compute_full_accuracy,
    compute_questbench_metrics,
    compute_underspecified_metrics,
    load_trained_model,
)
from exp.shared.utils import (
    BOOTSTRAP_SEED,
    FIGURES_DIR,
    ROOT,
    SEEDS,
    bootstrap_ci,
    dump_json,
    load_json,
    mean_std,
    paired_bootstrap_delta,
    run_config,
)


CONDS = ["answer_only", "noisy", "clean"]
QUEST_LOGIC_EVAL_N = 500
QUEST_PLANNING_EVAL_N = 1000


def aggregate(seed_results):
    scalar_keys = [key for key, value in seed_results[0].items() if key != "seed" and not isinstance(value, list)]
    return {key: mean_std([result[key] for result in seed_results]) for key in scalar_keys}


def evaluate_seed(cond, seed, data):
    adapter_dir = ROOT / "exp" / cond / f"seed_{seed}" / "adapter"
    train_metrics = load_json(ROOT / "exp" / cond / f"seed_{seed}" / "train_metrics.json")
    tokenizer, model = load_trained_model(adapter_dir)

    logic_full = data["logic"]["test"]["full"]
    planning_full = data["planning"]["test"]["full"]
    logic_clean = data["logic"]["test"]["clean"]
    planning_clean = data["planning"]["test"]["clean"]
    quest_logic = data["questbench"]["logic"][:QUEST_LOGIC_EVAL_N]
    quest_planning = data["questbench"]["planning"][:QUEST_PLANNING_EVAL_N]
    planning_non_unique = data["planning"]["test"].get("aux_non_unique", [])
    logic_low_risk = [item for item in logic_clean if len(item.get("all_candidate_proofs", [])) <= 1]
    logic_high_risk = [item for item in logic_clean if len(item.get("all_candidate_proofs", [])) > 1]

    outputs = {
        "seed": seed,
        "runtime_seconds": train_metrics["runtime_seconds"],
        "peak_vram_gb": train_metrics["peak_vram_gb"],
    }
    arrays = {}

    preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in logic_full], max_new_tokens=12)
    acc, arr = compute_full_accuracy(preds, logic_full)
    outputs["logic_full_accuracy"] = acc
    arrays["logic_full_accuracy"] = arr

    preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in planning_full], max_new_tokens=12)
    acc, arr = compute_full_accuracy(preds, planning_full)
    outputs["planning_full_accuracy"] = acc
    arrays["planning_full_accuracy"] = arr

    preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in logic_clean], max_new_tokens=20)
    metrics = compute_underspecified_metrics(preds, logic_clean, tokenizer=tokenizer, model=model)
    for key in ["ask_answer_accuracy", "question_accuracy", "solve_after_clarification_rate"]:
        outputs[f"logic_clean_{key}"] = metrics[key]
        arrays[f"logic_clean_{key}"] = metrics["per_example"][key]

    preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in planning_clean], max_new_tokens=20)
    metrics = compute_underspecified_metrics(preds, planning_clean, tokenizer=tokenizer, model=model)
    for key in ["ask_answer_accuracy", "question_accuracy", "solve_after_clarification_rate"]:
        outputs[f"planning_clean_{key}"] = metrics[key]
        arrays[f"planning_clean_{key}"] = metrics["per_example"][key]

    for family, dataset in data["planning"]["robustness"].items():
        preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in dataset], max_new_tokens=20)
        metrics = compute_underspecified_metrics(preds, dataset, tokenizer=tokenizer, model=model)
        outputs[f"planning_family_{family.lower()}_question_accuracy"] = metrics["question_accuracy"]
        arrays[f"planning_family_{family.lower()}_question_accuracy"] = metrics["per_example"]["question_accuracy"]

    preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in quest_logic], batch_size=16, max_new_tokens=12)
    metrics = compute_questbench_metrics(preds, quest_logic, tokenizer=tokenizer, model=model)
    for key in ["ask_answer_accuracy", "question_accuracy", "solve_after_clarification_rate"]:
        outputs[f"quest_logic_{key}"] = metrics[key]
        arrays[f"quest_logic_{key}"] = metrics["per_example"][key]

    preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in quest_planning], batch_size=32, max_new_tokens=12)
    metrics = compute_questbench_metrics(preds, quest_planning, tokenizer=tokenizer, model=model)
    for key in ["ask_answer_accuracy", "question_accuracy", "solve_after_clarification_rate"]:
        outputs[f"quest_planning_{key}"] = metrics[key]
        arrays[f"quest_planning_{key}"] = metrics["per_example"][key]

    if planning_non_unique:
        preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in planning_non_unique], max_new_tokens=20)
        metrics = compute_underspecified_metrics(preds, planning_non_unique, tokenizer=tokenizer, model=model)
        outputs["planning_non_unique_question_accuracy"] = metrics["question_accuracy"]
        arrays["planning_non_unique_question_accuracy"] = metrics["per_example"]["question_accuracy"]
    else:
        outputs["planning_non_unique_question_accuracy"] = None
        arrays["planning_non_unique_question_accuracy"] = []

    for risk_name, dataset in [("logic_low_risk", logic_low_risk), ("logic_high_risk", logic_high_risk)]:
        if dataset:
            preds = batch_generate_loaded(tokenizer, model, [item["input_text"] for item in dataset], max_new_tokens=20)
            metrics = compute_underspecified_metrics(preds, dataset, tokenizer=tokenizer, model=model)
            outputs[f"{risk_name}_question_accuracy"] = metrics["question_accuracy"]
            arrays[f"{risk_name}_question_accuracy"] = metrics["per_example"]["question_accuracy"]
        else:
            outputs[f"{risk_name}_question_accuracy"] = None
            arrays[f"{risk_name}_question_accuracy"] = []

    outputs["questbench_macro_question_accuracy"] = (
        outputs["quest_logic_question_accuracy"] + outputs["quest_planning_question_accuracy"]
    ) / 2
    outputs["underspecified_macro_accuracy"] = (
        outputs["logic_clean_question_accuracy"]
        + outputs["planning_clean_question_accuracy"]
        + outputs["quest_logic_question_accuracy"]
        + outputs["quest_planning_question_accuracy"]
    ) / 4

    dump_json({"seed": seed, "metrics": outputs, "arrays": arrays}, ROOT / "exp" / cond / f"seed_{seed}" / "eval_results.json")
    return outputs, arrays


def bootstrap_tables(per_seed_arrays):
    confirmatory = {cond: per_seed_arrays[cond][0] for cond in CONDS}
    metrics = [
        "quest_logic_question_accuracy",
        "quest_planning_question_accuracy",
        "logic_clean_question_accuracy",
        "planning_clean_question_accuracy",
    ]
    out = {}
    for metric in metrics:
        out[metric] = {
            "clean_minus_noisy": paired_bootstrap_delta(confirmatory["clean"][metric], confirmatory["noisy"][metric], seed=BOOTSTRAP_SEED),
            "clean_minus_answer_only": paired_bootstrap_delta(confirmatory["clean"][metric], confirmatory["answer_only"][metric], seed=BOOTSTRAP_SEED),
            "clean": bootstrap_ci(confirmatory["clean"][metric], seed=BOOTSTRAP_SEED),
            "noisy": bootstrap_ci(confirmatory["noisy"][metric], seed=BOOTSTRAP_SEED),
            "answer_only": bootstrap_ci(confirmatory["answer_only"][metric], seed=BOOTSTRAP_SEED),
        }
    return out


def save_figures(aggregated, bootstrap_results, budget_rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Quest Logic-Q", "Quest Planning-Q"]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4))
    for idx, cond in enumerate(CONDS):
        vals = [
            aggregated[cond]["quest_logic_question_accuracy"]["mean"],
            aggregated[cond]["quest_planning_question_accuracy"]["mean"],
        ]
        ci_names = ["quest_logic_question_accuracy", "quest_planning_question_accuracy"]
        err_low = [vals[i] - bootstrap_results[ci_names[i]][cond]["lower"] for i in range(2)]
        err_high = [bootstrap_results[ci_names[i]][cond]["upper"] - vals[i] for i in range(2)]
        ax.bar(x + (idx - 1) * width, vals, width, label=cond, yerr=[err_low, err_high], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Question accuracy")
    ax.set_title("QuestBench transfer comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "questbench_bars.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    family_labels = ["A", "B", "C"]
    for cond in CONDS:
        ax.plot(
            family_labels,
            [
                aggregated[cond]["planning_family_a_question_accuracy"]["mean"],
                aggregated[cond]["planning_family_b_question_accuracy"]["mean"],
                aggregated[cond]["planning_family_c_question_accuracy"]["mean"],
            ],
            marker="o",
            label=cond,
        )
    ax.set_ylabel("Planning clarification question accuracy")
    ax.set_title("Planning paraphrase robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "planning_robustness.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    stages = [row["stage"] for row in budget_rows]
    actual = [row["actual_hours"] for row in budget_rows]
    planned = [row["planned_hours"] for row in budget_rows]
    xpos = np.arange(len(stages))
    ax.bar(xpos - 0.18, planned, width=0.36, label="planned")
    ax.bar(xpos + 0.18, actual, width=0.36, label="actual")
    ax.set_xticks(xpos)
    ax.set_xticklabels(stages, rotation=15)
    ax.set_ylabel("Hours")
    ax.set_title("Budgeted vs actual stage time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "budget_vs_actual.png", dpi=200)
    plt.close(fig)


def main():
    data = build_all_data()
    aggregated = {}
    raw = {}
    per_seed_arrays = {}
    for cond in CONDS:
        raw[cond] = []
        per_seed_arrays[cond] = []
        for seed in SEEDS:
            metrics, arrays = evaluate_seed(cond, seed, data)
            raw[cond].append(metrics)
            per_seed_arrays[cond].append(arrays)
        aggregated[cond] = aggregate(raw[cond])
        dump_json({"experiment": cond, "seed_results": raw[cond], "metrics": aggregated[cond]}, ROOT / "exp" / cond / "results.json")

    bootstrap_results = bootstrap_tables(per_seed_arrays)
    audit_results = load_json(ROOT / "exp" / "manual_audit" / "results.json")
    env_results = load_json(ROOT / "exp" / "environment" / "results.json")
    prepare_results = load_json(ROOT / "exp" / "prepare_data" / "results.json")

    budget_rows = [
        {"stage": "prepare", "planned_hours": 2.5, "actual_hours": prepare_results.get("runtime_seconds", 0.0) / 3600.0},
        {"stage": "audit", "planned_hours": 1.0, "actual_hours": audit_results.get("runtime_seconds", 0.0) / 3600.0},
        {
            "stage": "train",
            "planned_hours": 2.5,
            "actual_hours": sum(item["runtime_seconds"] for cond in CONDS for item in raw[cond]) / 3600.0,
        },
        {"stage": "pilot", "planned_hours": 0.5, "actual_hours": env_results["pilot_metrics"]["runtime_seconds"] / 3600.0},
        {"stage": "eval", "planned_hours": 1.5, "actual_hours": 0.0},
    ]
    save_figures(aggregated, bootstrap_results, budget_rows)

    proxy_label_quality = audit_results["proxy_audit_summary"]
    residual_noise = audit_results["residual_noise"]
    claims = {
        "clean_beats_noisy_on_questbench_macro": aggregated["clean"]["questbench_macro_question_accuracy"]["mean"] > aggregated["noisy"]["questbench_macro_question_accuracy"]["mean"],
        "clean_beats_answer_only_on_underspecified_macro": aggregated["clean"]["underspecified_macro_accuracy"]["mean"] > aggregated["answer_only"]["underspecified_macro_accuracy"]["mean"],
        "clean_retention_within_2pt_logic": aggregated["answer_only"]["logic_full_accuracy"]["mean"] - aggregated["clean"]["logic_full_accuracy"]["mean"] <= 0.02,
        "clean_retention_within_2pt_planning": aggregated["answer_only"]["planning_full_accuracy"]["mean"] - aggregated["clean"]["planning_full_accuracy"]["mean"] <= 0.02,
        "human_audit_completed": audit_results["human_audit"]["status"] == "completed",
    }
    final = {
        "setup": {
            "model": run_config()["model_name"],
            "seeds": SEEDS,
            "conditions": CONDS,
            "questbench_eval_subset_sizes": {
                "logic": QUEST_LOGIC_EVAL_N,
                "planning": QUEST_PLANNING_EVAL_N,
            },
            "execution_deviations": [
                "Python 3.11 was unavailable; the rerun used Python 3.12.7.",
                "Fast Downward was unavailable; planning validation used the bounded BFS solver described in artifacts/validator_spec.json.",
                "The preregistered human audit could not be executed and the study must therefore be treated as a narrowed or negative pilot.",
                "QuestBench was evaluated on fixed executed subsets rather than the full published test splits to keep the repaired 3-seed rerun within budget.",
            ],
        },
        "results_by_condition": aggregated,
        "raw_seed_results": raw,
        "bootstrap_confidence_intervals": bootstrap_results,
        "proxy_label_quality": proxy_label_quality,
        "human_audit": audit_results["human_audit"],
        "residual_noise": residual_noise,
        "budget_rows": budget_rows,
        "claims": claims,
        "negative_result_summary": "Because the required human audit was infeasible and planning cleanliness remains proxy-defined, these results should be interpreted as a repaired pilot rather than confirmatory evidence for validator-clean clarification transfer.",
    }
    dump_json(final, ROOT / "results.json")
    dump_json(
        {
            "experiment": "analyze",
            "config": run_config(extra={"bootstrap_seed": BOOTSTRAP_SEED}),
            "metrics": final,
        },
        ROOT / "exp" / "analyze" / "results.json",
    )


if __name__ == "__main__":
    main()
