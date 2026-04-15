from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from exp.shared.common import (
    DEFAULT_SCHEDULE,
    EARLY_HEAVY,
    EXP_DIR,
    FLAT_SCHEDULE,
    GATE_C_VALUES,
    GUIDED_STEPS,
    MAIN_HELD_OUT_LIMIT,
    MIDDLE_HEAVY,
    NON_EQ_LIMIT,
    PILOT_SEED,
    REDUCED_GUIDED_STEPS,
    ROBUSTNESS_LIMIT,
    SEEDS,
    PromptRecord,
    ensure_dirs,
    records_from_jsonl,
    seed_everything,
    stable_hash,
    write_csv,
    write_json,
)
from exp.shared.generation import MethodConfig, SDRunner
from exp.shared.metrics import CLIPScorer, LPIPSScorer, SlotEvaluator, bootstrap_ci, mean_std, pairwise_lpips


DATA_PATH = Path("data/prompts_with_paraphrases.jsonl")
PILOT_PATH = EXP_DIR / "pilot" / "results.json"

METHODS = {
    "vanilla_sd15": MethodConfig(name="vanilla_sd15", alpha_schedule=DEFAULT_SCHEDULE, guided_steps=[], use_gate=False),
    "static_consensus": MethodConfig(name="static_consensus", alpha_schedule=DEFAULT_SCHEDULE, constant_alpha=0.4, use_gate=False),
    "adaptive_ungated": MethodConfig(name="adaptive_ungated", alpha_schedule=DEFAULT_SCHEDULE, use_gate=False),
    "paradg": MethodConfig(name="paradg", alpha_schedule=DEFAULT_SCHEDULE, use_gate=True, use_slot_redistribution=True, gate_c=1.0),
    "ablation_no_gate": MethodConfig(name="ablation_no_gate", alpha_schedule=DEFAULT_SCHEDULE, use_gate=False, use_slot_redistribution=True),
    "ablation_no_slot": MethodConfig(name="ablation_no_slot", alpha_schedule=DEFAULT_SCHEDULE, use_gate=True, use_slot_redistribution=False, gate_c=1.0),
    "ablation_no_timestep": MethodConfig(
        name="ablation_no_timestep",
        alpha_schedule=DEFAULT_SCHEDULE,
        use_gate=True,
        use_slot_redistribution=True,
        constant_alpha=sum(DEFAULT_SCHEDULE) / len(DEFAULT_SCHEDULE),
        gate_c=1.0,
    ),
    "ablation_reduced_paraphrase": MethodConfig(name="ablation_reduced_paraphrase", alpha_schedule=DEFAULT_SCHEDULE, use_gate=True, use_slot_redistribution=True, gate_c=1.0),
    "ablation_nonequivalent": MethodConfig(name="ablation_nonequivalent", alpha_schedule=DEFAULT_SCHEDULE, use_gate=True, use_slot_redistribution=True, gate_c=1.0, paraphrase_mode="nonequivalent"),
}


def _pilot_report() -> dict[str, Any]:
    if not PILOT_PATH.exists():
        return {}
    try:
        return json.loads(PILOT_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _apply_pilot_choice(method: MethodConfig) -> MethodConfig:
    pilot = _pilot_report()
    if not pilot:
        return method
    schedule_name = pilot.get("best_schedule_name")
    gate_c = pilot.get("best_gate_c")
    guided_steps = pilot.get("guided_steps")
    schedule_lookup = {"flat_0.4": FLAT_SCHEDULE, "early_heavy": EARLY_HEAVY, "middle_heavy": MIDDLE_HEAVY}
    if schedule_name in schedule_lookup:
        method.alpha_schedule = schedule_lookup[schedule_name]
    if isinstance(gate_c, (int, float)):
        method.gate_c = float(gate_c)
    if isinstance(guided_steps, list) and guided_steps:
        method.guided_steps = [int(step) for step in guided_steps]
    return method


def load_split_records(split: str, overlap_only: bool = False, limit: int | None = None) -> list[PromptRecord]:
    records = [r for r in records_from_jsonl(DATA_PATH) if r.split == split]
    if overlap_only:
        records = [r for r in records if r.overlap_subset_flag]
    if limit is not None:
        by_category: dict[str, list[PromptRecord]] = {}
        for record in records:
            by_category.setdefault(record.category, []).append(record)
        categories = sorted(by_category)
        base = limit // max(len(categories), 1)
        remainder = limit % max(len(categories), 1)
        selected: list[PromptRecord] = []
        for idx, category in enumerate(categories):
            take = base + (1 if idx < remainder else 0)
            selected.extend(by_category[category][:take])
        records = selected
    return records


def _variant_specs(record: PromptRecord, prompt_variants: bool) -> list[tuple[str, str]]:
    variants = [(f"{record.prompt_id}::orig", record.original_prompt)]
    if prompt_variants:
        for idx, paraphrase in enumerate(record.approved_paraphrases):
            variants.append((f"{record.prompt_id}::para{idx}", paraphrase))
    return variants


def _pluralize(noun: str) -> str:
    if not noun:
        return noun
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "iy", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _slot_alias_map(record: PromptRecord) -> dict[str, list[str]]:
    count_alias = {
        "one": ["one", "1", "a"],
        "two": ["two", "2"],
        "three": ["three", "3"],
        "four": ["four", "4"],
        "five": ["five", "5"],
        "six": ["six", "6"],
        "seven": ["seven", "7"],
        "eight": ["eight", "8"],
    }
    relation_alias = {
        "next to": ["next to"],
        "near": ["near"],
        "on side of": ["on side of", "on the side of"],
        "on the left of": ["on the left of", "to the left of"],
        "on the right of": ["on the right of", "to the right of"],
        "on the top of": ["on the top of", "above"],
        "on the bottom of": ["on the bottom of", "below"],
    }
    obj1_aliases = [record.object_1]
    obj2_aliases = [record.object_2] if record.object_2 else []
    if record.count_1 and record.count_1 != "one":
        obj1_aliases.append(_pluralize(record.object_1))
    if record.count_2 and record.count_2 != "one":
        obj2_aliases.append(_pluralize(record.object_2))
    return {
        "object_1": [alias for alias in obj1_aliases if alias],
        "count_1": count_alias.get(record.count_1, [record.count_1]) if record.count_1 else [],
        "attribute_1": [record.attribute_1] if record.attribute_1 else [],
        "relation": relation_alias.get(record.relation, [record.relation]) if record.relation else [],
        "object_2": [alias for alias in obj2_aliases if alias],
        "count_2": count_alias.get(record.count_2, [record.count_2]) if record.count_2 else [],
        "attribute_2": [record.attribute_2] if record.attribute_2 else [],
    }


def _prompt_pool(record: PromptRecord, primary_prompt: str, method: MethodConfig) -> list[str]:
    if method.name == "vanilla_sd15":
        return [primary_prompt]
    if method.name == "ablation_nonequivalent":
        return [primary_prompt, record.non_equivalent_aux_prompt]
    others = [record.original_prompt, *record.approved_paraphrases]
    ordered = [primary_prompt] + [prompt for prompt in others if prompt != primary_prompt]
    if method.name == "ablation_reduced_paraphrase":
        return ordered[:2]
    return ordered[:3]


def _matrix_spec() -> dict[str, Any]:
    pilot = _pilot_report()
    reduced = bool(pilot.get("reduced_matrix_activated"))
    return {
        "reduced": reduced,
        "faithfulness_limit": MAIN_HELD_OUT_LIMIT if reduced else 60,
        "robustness_limit": ROBUSTNESS_LIMIT if reduced else 24,
        "analysis_limit": ROBUSTNESS_LIMIT if reduced else 18,
        "allowed_ablations": ["ablation_no_gate", "ablation_no_slot"] if reduced else [
            "ablation_no_gate",
            "ablation_no_slot",
            "ablation_no_timestep",
            "ablation_reduced_paraphrase",
            "ablation_nonequivalent",
        ],
    }


def _scenario_tasks(experiment_name: str) -> list[dict[str, Any]]:
    matrix = _matrix_spec()
    if experiment_name in {"vanilla_sd15", "static_consensus", "adaptive_ungated", "paradg"}:
        return [
            {"scenario": "faithfulness", "split": "held_out", "overlap_only": False, "limit": matrix["faithfulness_limit"], "prompt_variants": False, "seeds": SEEDS},
            {"scenario": "robustness", "split": "held_out", "overlap_only": True, "limit": matrix["robustness_limit"], "prompt_variants": True, "seeds": SEEDS},
        ]
    if experiment_name == "ablation_nonequivalent":
        return [{"scenario": "analysis", "split": "held_out", "overlap_only": True, "limit": NON_EQ_LIMIT, "prompt_variants": False, "seeds": SEEDS}]
    return [{"scenario": "analysis", "split": "held_out", "overlap_only": True, "limit": matrix["analysis_limit"], "prompt_variants": False, "seeds": SEEDS}]


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _log_event(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _summarize(rows: list[dict[str, Any]], clip: CLIPScorer, lpips: LPIPSScorer) -> dict[str, Any]:
    scenarios: dict[str, dict[str, Any]] = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        metrics: dict[str, Any] = {}
        clip_values = [row["clipscore"] for row in scenario_rows]
        metrics["clipscore"] = {**mean_std(clip_values), "ci95": bootstrap_ci(clip_values)}
        if "image_reward" in scenario_rows[0]:
            reward_values = [row["image_reward"] for row in scenario_rows]
            metrics["image_reward"] = {**mean_std(reward_values), "ci95": bootstrap_ci(reward_values)}
        runtime = [row["runtime_seconds"] for row in scenario_rows]
        metrics["runtime_seconds"] = {**mean_std(runtime), "ci95": bootstrap_ci(runtime)}
        gpu_mem = [row["peak_gpu_memory_mb"] for row in scenario_rows]
        metrics["peak_gpu_memory_mb"] = {**mean_std(gpu_mem), "ci95": bootstrap_ci(gpu_mem)}

        orig_rows = [row for row in scenario_rows if row["prompt_variant_id"].endswith("::orig")]
        if orig_rows:
            category_names = sorted({row["category"] for row in orig_rows})
            for category in category_names:
                values = [row["category_score"] for row in orig_rows if row["category"] == category]
                metrics[f"{category}_score"] = {**mean_std(values), "ci95": bootstrap_ci(values)}
            lpips_values = []
            for prompt_id in sorted({row["prompt_id"] for row in orig_rows}):
                seed_paths = [Path(row["output_path"]) for row in orig_rows if row["prompt_id"] == prompt_id]
                lpips_values.append(pairwise_lpips(seed_paths, lpips))
            metrics["lpips_seed_diversity"] = {**mean_std(lpips_values), "ci95": bootstrap_ci(lpips_values)}
            if "dino_seed_dispersion" in orig_rows[0]:
                dino_seed = [row["dino_seed_dispersion"] for row in orig_rows if row["dino_seed_dispersion"] == row["dino_seed_dispersion"]]
                if dino_seed:
                    metrics["dino_seed_dispersion"] = {**mean_std(dino_seed), "ci95": bootstrap_ci(dino_seed)}

        if scenario == "robustness":
            prs_values = [row["prompt_seed_prs"] for row in scenario_rows if "prompt_seed_prs" in row]
            image_consistency = [row["prompt_seed_clip_consistency"] for row in scenario_rows if "prompt_seed_clip_consistency" in row]
            metrics["paraphrase_robustness_score"] = {**mean_std(prs_values), "ci95": bootstrap_ci(prs_values)}
            metrics["clip_image_consistency"] = {**mean_std(image_consistency), "ci95": bootstrap_ci(image_consistency)}
            if "prompt_seed_dino_consistency" in scenario_rows[0]:
                dino_consistency = [row["prompt_seed_dino_consistency"] for row in scenario_rows if row["prompt_seed_dino_consistency"] == row["prompt_seed_dino_consistency"]]
                if dino_consistency:
                    metrics["dino_consistency"] = {**mean_std(dino_consistency), "ci95": bootstrap_ci(dino_consistency)}

        scenarios[scenario] = {
            "num_rows": len(scenario_rows),
            "num_prompt_records": len({row["prompt_id"] for row in scenario_rows}),
            "metrics": metrics,
        }
    return scenarios


def run_experiment_suite(experiment_name: str) -> dict[str, Any]:
    ensure_dirs()
    method = deepcopy(METHODS[experiment_name])
    if experiment_name != "vanilla_sd15":
        method = _apply_pilot_choice(method)
    matrix = _matrix_spec()
    if experiment_name.startswith("ablation_") and experiment_name not in matrix["allowed_ablations"]:
        skipped = {
            "experiment": experiment_name,
            "status": "skipped",
            "reason": "Reduced matrix is active, so only Ablation A and Ablation B are allowed by the preregistered plan.",
            "config": {"reduced_matrix_activated": True, "allowed_ablations": matrix["allowed_ablations"]},
        }
        write_json(EXP_DIR / experiment_name / "results.json", skipped)
        return skipped
    exp_dir = EXP_DIR / experiment_name
    log_path = exp_dir / "logs" / "events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    rows: list[dict[str, Any]] = []
    runner = SDRunner()
    clip = CLIPScorer()
    lpips = LPIPSScorer()
    evaluator = SlotEvaluator(clip)

    _log_event(log_path, {"event": "start_experiment", "experiment": experiment_name, "method": method.__dict__})
    for task in _scenario_tasks(experiment_name):
        records = load_split_records(task["split"], overlap_only=task["overlap_only"], limit=task["limit"])
        _log_event(log_path, {"event": "start_task", **task, "num_records": len(records)})
        for record in records:
            slot_alias_map = _slot_alias_map(record)
            for variant_id, prompt_text in _variant_specs(record, prompt_variants=task["prompt_variants"]):
                prompt_inputs = _prompt_pool(record, prompt_text, method)
                for seed in task["seeds"]:
                    seed_everything(seed)
                    out_path = exp_dir / "outputs" / task["scenario"] / f"{variant_id}_seed{seed}.png"
                    meta = runner.generate(variant_id, prompt_inputs, seed, method, out_path, slot_alias_map=slot_alias_map)
                    slot_scores = evaluator.evaluate(out_path, record, prompt_text)
                    row = {
                        "experiment": experiment_name,
                        "scenario": task["scenario"],
                        "split": task["split"],
                        "overlap_only": task["overlap_only"],
                        "prompt_id": record.prompt_id,
                        "prompt_variant_id": variant_id,
                        "prompt_text": prompt_text,
                        "category": record.category,
                        "seed": seed,
                        "clipscore": clip.image_text_score(out_path, prompt_text),
                        "runtime_seconds": meta["runtime_seconds"],
                        "peak_gpu_memory_mb": meta["peak_gpu_memory_mb"],
                        "latent_hash": meta["latent_hash"],
                        "gate_trace_json": json.dumps(meta["gate_trace"]),
                        "output_path": str(out_path),
                        **slot_scores,
                    }
                    rows.append(row)
                    sidecar = {**row, "prompt_inputs": prompt_inputs}
                    write_json(out_path.with_suffix(".json"), sidecar)
                    _log_event(
                        log_path,
                        {
                            "event": "row_complete",
                            "experiment": experiment_name,
                            "scenario": task["scenario"],
                            "prompt_id": record.prompt_id,
                            "prompt_variant_id": variant_id,
                            "seed": seed,
                            "category_score": row["category_score"],
                            "overall_success": row["overall_success"],
                        },
                    )

    write_csv(exp_dir / "generation_index.csv", rows)
    results = {
        "experiment": experiment_name,
        "config": {
            "method": method.__dict__,
            "seeds": SEEDS,
            "guided_steps": method.guided_steps or GUIDED_STEPS,
            "tasks": _scenario_tasks(experiment_name),
        },
        "raw_rows_path": str(exp_dir / "generation_index.csv"),
        "scenarios": _summarize(rows, clip, lpips),
    }
    write_json(exp_dir / "results.json", results)
    _log_event(log_path, {"event": "finish_experiment", "experiment": experiment_name, "rows": len(rows)})
    return results


def run_pilot() -> dict[str, Any]:
    ensure_dirs()
    pilot_records = load_split_records("pilot", overlap_only=False, limit=None)
    runner = SDRunner()
    clip = CLIPScorer()
    evaluator = SlotEvaluator(clip)
    log_path = EXP_DIR / "pilot" / "logs" / "events.jsonl"
    schedules = {"flat_0.4": FLAT_SCHEDULE, "early_heavy": EARLY_HEAVY, "middle_heavy": MIDDLE_HEAVY}
    candidates: list[dict[str, Any]] = []

    for gate_c in GATE_C_VALUES:
        for schedule_name, schedule in schedules.items():
            method = deepcopy(METHODS["paradg"])
            method.gate_c = gate_c
            method.alpha_schedule = schedule
            task_rows = []
            runtime_values = []
            for record in pilot_records:
                prompt_inputs = _prompt_pool(record, record.original_prompt, method)
                out_path = EXP_DIR / "pilot" / "outputs" / f"{schedule_name}_c{gate_c}_{record.prompt_id}.png"
                meta = runner.generate(record.prompt_id, prompt_inputs, PILOT_SEED, method, out_path, slot_alias_map=_slot_alias_map(record))
                slot_scores = evaluator.evaluate(out_path, record, record.original_prompt)
                task_rows.append(slot_scores["category_score"])
                runtime_values.append(meta["runtime_seconds"])
            score = float(sum(task_rows) / max(len(task_rows), 1))
            candidate = {
                "gate_c": gate_c,
                "schedule_name": schedule_name,
                "mean_category_score": score,
                "mean_runtime_seconds": float(sum(runtime_values) / max(len(runtime_values), 1)),
            }
            candidates.append(candidate)
            _log_event(log_path, {"event": "pilot_candidate", **candidate})

    best = max(candidates, key=lambda item: (item["mean_category_score"], -item["mean_runtime_seconds"]))
    METHODS["paradg"].gate_c = best["gate_c"]
    METHODS["paradg"].alpha_schedule = schedules[best["schedule_name"]]
    base_runtime = next((item["mean_runtime_seconds"] for item in candidates if item["gate_c"] == best["gate_c"] and item["schedule_name"] == best["schedule_name"]), None)
    mean_prompt_runtime = float(base_runtime or 0.0)
    overlap_count = len(load_split_records("held_out", overlap_only=True, limit=None))
    full_projection_images = (60 * len(SEEDS) * 4) + (24 * 2 * len(SEEDS) * 4) + (18 * len(SEEDS) * 4) + (NON_EQ_LIMIT * len(SEEDS))
    reduced_projection_images = (MAIN_HELD_OUT_LIMIT * len(SEEDS) * 4) + (ROBUSTNESS_LIMIT * 2 * len(SEEDS) * 4) + (ROBUSTNESS_LIMIT * len(SEEDS) * 2)
    projected_full_hours = (mean_prompt_runtime * full_projection_images) / 3600.0 if mean_prompt_runtime else None
    projected_reduced_hours = (mean_prompt_runtime * reduced_projection_images) / 3600.0 if mean_prompt_runtime else None
    reduced = bool(projected_full_hours and projected_full_hours > 7.0) or overlap_count < 18
    guided_steps = REDUCED_GUIDED_STEPS if reduced else GUIDED_STEPS
    report = {
        "num_pilot_prompts": len(pilot_records),
        "seed": PILOT_SEED,
        "candidate_results": candidates,
        "best_gate_c": best["gate_c"],
        "best_schedule_name": best["schedule_name"],
        "guided_steps": guided_steps,
        "projected_gpu_hours": projected_reduced_hours if reduced else projected_full_hours,
        "projected_full_gpu_hours": projected_full_hours,
        "projected_reduced_gpu_hours": projected_reduced_hours,
        "overlap_prompt_count": overlap_count,
        "reduced_matrix_activated": reduced,
        "notes": [
            "Reduced matrix activation follows only the preregistered runtime and overlap-count rules.",
            "AAPB, the single-prompt baseline, and the human/double-annotation components remain separate documented omissions and do not affect matrix sizing.",
        ],
    }
    write_json(EXP_DIR / "pilot" / "results.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    if args.experiment == "pilot":
        run_pilot()
    else:
        run_experiment_suite(args.experiment)


if __name__ == "__main__":
    main()
