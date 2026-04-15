from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from exp.shared.common import EXP_DIR, FIGURES_DIR, read_csv, read_json, write_json
from exp.shared.metrics import paired_bootstrap_delta, paired_permutation_test


MAIN_METHODS = ["vanilla_sd15", "static_consensus", "adaptive_ungated", "paradg"]
ABLATIONS = ["ablation_no_gate", "ablation_no_slot", "ablation_no_timestep", "ablation_reduced_paraphrase", "ablation_nonequivalent"]


def _pilot_report() -> dict[str, Any]:
    path = EXP_DIR / "pilot" / "results.json"
    return read_json(path) if path.exists() else {}


def load_result(name: str) -> dict[str, Any]:
    path = EXP_DIR / name / "results.json"
    return read_json(path) if path.exists() else {"experiment": name, "status": "missing"}


def _parse_rows(name: str) -> list[dict[str, Any]]:
    path = EXP_DIR / name / "generation_index.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    parsed = []
    bool_fields = {
        "overlap_only",
        "object_1_present",
        "object_2_present",
        "attribute_1_correct",
        "attribute_2_correct",
        "relation_correct",
        "count_1_correct",
        "overall_success",
    }
    int_fields = {"seed", "detected_count_object_1", "detected_count_object_2"}
    float_fields = {
        "clipscore",
        "runtime_seconds",
        "peak_gpu_memory_mb",
        "category_score",
        "image_reward",
        "prompt_seed_prs",
        "prompt_seed_clip_consistency",
        "prompt_seed_dino_consistency",
        "lpips_seed_diversity",
        "dino_seed_dispersion",
    }
    for row in rows:
        clean = dict(row)
        for key in bool_fields:
            if key in clean:
                clean[key] = clean[key] == "True" or clean[key] == "true"
        for key in int_fields:
            if key in clean and clean[key] != "":
                clean[key] = int(clean[key])
        for key in float_fields:
            if key in clean:
                clean[key] = float(clean[key]) if clean[key] != "" else float("nan")
        parsed.append(clean)
    return parsed


def _faithfulness_series(rows: list[dict[str, Any]], category: str) -> dict[tuple[str, int], float]:
    return {
        (row["prompt_id"], row["seed"]): row["category_score"]
        for row in rows
        if row["scenario"] == "faithfulness" and row["prompt_variant_id"].endswith("::orig") and row["category"] == category
    }


def _robustness_series(rows: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
    slot_keys = [
        "object_1_present",
        "object_2_present",
        "attribute_1_correct",
        "attribute_2_correct",
        "relation_correct",
        "count_1_correct",
    ]
    result = {}
    by_prompt_seed = {}
    for row in rows:
        if row["scenario"] != "robustness":
            continue
        by_prompt_seed.setdefault((row["prompt_id"], row["seed"]), []).append(row)
    for key, group in by_prompt_seed.items():
        group = sorted(group, key=lambda item: item["prompt_variant_id"])
        total = 0
        matches = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total += 1
                if all(group[i][slot] == group[j][slot] for slot in slot_keys):
                    matches += 1
        result[key] = matches / total if total else float("nan")
    return result


def _paired_lists(left: dict[tuple[str, int], float], right: dict[tuple[str, int], float]) -> tuple[list[float], list[float]]:
    keys = sorted(set(left) & set(right))
    return [left[key] for key in keys], [right[key] for key in keys]


def _plot_main_results(results: dict[str, Any]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cats = ["attribute_binding_score", "relations_score", "numeracy_score"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(MAIN_METHODS))
    width = 0.22
    for idx, cat in enumerate(cats):
        vals = [results[m]["scenarios"]["faithfulness"]["metrics"].get(cat, {"mean": 0.0})["mean"] for m in MAIN_METHODS]
        ax.bar([p + idx * width for p in x], vals, width=width, label=cat.replace("_score", ""))
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(MAIN_METHODS, rotation=20)
    ax.set_ylabel("Score")
    ax.set_title("Held-out Slot-Based Category Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_scores.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    prs = [results[m]["scenarios"]["robustness"]["metrics"].get("paraphrase_robustness_score", {"mean": 0.0})["mean"] for m in MAIN_METHODS]
    latency = [results[m]["scenarios"]["robustness"]["metrics"].get("runtime_seconds", {"mean": 0.0})["mean"] for m in MAIN_METHODS]
    ax.scatter(latency, prs)
    for x_val, y_val, label in zip(latency, prs, MAIN_METHODS):
        ax.annotate(label, (x_val, y_val))
    ax.set_xlabel("Runtime per image (s)")
    ax.set_ylabel("PRS")
    ax.set_title("PRS vs Latency")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prs_vs_latency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    lpips = [results[m]["scenarios"]["faithfulness"]["metrics"].get("lpips_seed_diversity", {"mean": 0.0})["mean"] for m in MAIN_METHODS]
    ax.scatter(lpips, prs)
    for x_val, y_val, label in zip(lpips, prs, MAIN_METHODS):
        ax.annotate(label, (x_val, y_val))
    ax.set_xlabel("LPIPS seed diversity")
    ax.set_ylabel("PRS")
    ax.set_title("PRS vs LPIPS")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prs_vs_lpips.png", dpi=200)
    plt.close(fig)


def _save_gate_diagnostics() -> list[str]:
    rows = _parse_rows("paradg")
    selected = [row for row in rows if row["scenario"] == "faithfulness" and row["prompt_variant_id"].endswith("::orig")][:6]
    outputs = []
    for idx, row in enumerate(selected, start=1):
        gate_trace = json.loads(row["gate_trace_json"])
        x = [step["step_index"] for step in gate_trace]
        y = [step["gate"] for step in gate_trace]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Step index")
        ax.set_ylabel("Gate")
        ax.set_title(f'{row["prompt_id"]} seed {row["seed"]}')
        fig.tight_layout()
        png_path = FIGURES_DIR / f"gate_trace_{idx}.png"
        json_path = FIGURES_DIR / f"gate_trace_{idx}.json"
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
        json_path.write_text(json.dumps(gate_trace, indent=2))
        outputs.extend([str(png_path), str(json_path)])
    return outputs


def _save_qualitative_grid() -> str | None:
    candidate_rows = _parse_rows("paradg")
    prompts = []
    for row in candidate_rows:
        if row["scenario"] == "robustness" and row["prompt_variant_id"].endswith("::orig"):
            prompts.append(row["prompt_id"])
        if len(prompts) >= 8:
            break
    if not prompts:
        return None

    method_rows = {name: _parse_rows(name) for name in MAIN_METHODS}
    width, height = 512, 512
    canvas = Image.new("RGB", (width * len(MAIN_METHODS), height * len(prompts)), color="white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, prompt_id in enumerate(prompts):
        for col_idx, method in enumerate(MAIN_METHODS):
            rows = [row for row in method_rows[method] if row["prompt_id"] == prompt_id and row["scenario"] == "robustness" and row["prompt_variant_id"].endswith("::orig") and row["seed"] == 11]
            if not rows:
                continue
            image = Image.open(rows[0]["output_path"]).convert("RGB").resize((width, height))
            canvas.paste(image, (col_idx * width, row_idx * height))
            draw.text((col_idx * width + 10, row_idx * height + 10), f"{method}\n{prompt_id}", fill="white")
    out_path = FIGURES_DIR / "qualitative_grid.png"
    canvas.save(out_path)
    return str(out_path)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pilot = _pilot_report()
    allowed_ablations = ["ablation_no_gate", "ablation_no_slot"] if pilot.get("reduced_matrix_activated") else ABLATIONS
    results = {name: load_result(name) for name in MAIN_METHODS + ABLATIONS}
    row_tables = {name: _parse_rows(name) for name in MAIN_METHODS + ABLATIONS}

    comparisons = {"faithfulness": {}, "robustness": {}}
    for baseline in ["vanilla_sd15", "static_consensus", "adaptive_ungated"]:
        baseline_rows = row_tables[baseline]
        paradg_rows = row_tables["paradg"]
        faithfulness = {}
        for category in ["attribute_binding", "relations", "numeracy"]:
            left, right = _paired_lists(
                _faithfulness_series(paradg_rows, category),
                _faithfulness_series(baseline_rows, category),
            )
            faithfulness[category] = paired_bootstrap_delta(left, right)
        comparisons["faithfulness"][baseline] = faithfulness
        left_prs, right_prs = _paired_lists(_robustness_series(paradg_rows), _robustness_series(baseline_rows))
        comparisons["robustness"][baseline] = {
            "paired_bootstrap": paired_bootstrap_delta(left_prs, right_prs),
            "paired_permutation": paired_permutation_test(left_prs, right_prs),
        }

    _plot_main_results(results)
    gate_outputs = _save_gate_diagnostics()
    qualitative_grid = _save_qualitative_grid()

    study_status = "exploratory_feasibility"
    top_level = {
        "study_status": study_status,
        "pilot": pilot,
        "data_preparation": load_result("data_preparation"),
        "official_compbench_eval": load_result("official_compbench_eval"),
        "main_results": {name: results[name] for name in MAIN_METHODS},
        "ablations": {name: results[name] for name in allowed_ablations},
        "comparisons": comparisons,
        "skipped": {
            "aapb_reproduction": load_result("aapb_reproduction"),
            "single_prompt_baseline": load_result("single_prompt_baseline"),
            "human_reliability_subset": load_result("human_reliability_subset"),
            "paraphrase_double_annotation": load_result("paraphrase_double_annotation"),
        },
        "artifacts": {
            "figures": [str(FIGURES_DIR / "category_scores.png"), str(FIGURES_DIR / "prs_vs_latency.png"), str(FIGURES_DIR / "prs_vs_lpips.png")] + gate_outputs + ([qualitative_grid] if qualitative_grid else []),
        },
    }
    write_json(ROOT / "results.json", top_level)

    paradg_metrics = results["paradg"]["scenarios"]
    supported_claims = []
    weakened_claims = [
        "This rerun now uses official T2I-CompBench-family category evaluators for attribute binding, relations, and numeracy, but it is still not a valid confirmatory test of the preregistered ParaDG claim because the faithful AAPB baseline, the single-prompt baseline, and the human reliability components remain unexecuted.",
        "Paraphrase auditing is no longer accepted by construction, but the available double-audit artifact is automatic rather than the preregistered human double annotation, so robustness conclusions remain exploratory.",
        "Numeracy scores come from the official UniDet evaluation code with a documented RS200 detector fallback because the published R50 checkpoint referenced by the script was unavailable.",
        "Automated slot-based robustness is still reported without preregistered human validation and must therefore remain exploratory.",
    ]
    failed_claims = []

    robustness_better = all(
        comparisons["robustness"][baseline]["paired_bootstrap"]["mean_delta"] > 0.0 for baseline in ["vanilla_sd15", "static_consensus", "adaptive_ungated"]
    )
    if not robustness_better:
        failed_claims.append("ParaDG does not improve automatic slot-based PRS over every implemented baseline.")

    faithfulness_wins = 0
    for category in ["attribute_binding", "relations", "numeracy"]:
        deltas = [comparisons["faithfulness"][baseline][category]["mean_delta"] for baseline in ["vanilla_sd15", "static_consensus", "adaptive_ungated"]]
        if max(deltas) > 0.0:
            faithfulness_wins += 1
    if faithfulness_wins < 2:
        failed_claims.append("ParaDG does not improve at least two held-out faithfulness categories in the reduced study.")
    supported_claims.append("The rerun honors the pilot-frozen alpha schedule, propagates the frozen guided-step set into paraphrase-based methods, and saves slot-level correction diagnostics instead of a single lexical scalar trace.")
    supported_claims.append("Held-out category metrics now come from the official T2I-CompBench evaluation family rather than the earlier custom OWL-ViT plus CLIP slot scorer.")
    supported_claims.append("Paraphrase artifacts now record candidate-level acceptance and rejection decisions instead of marking every deterministic template as accepted by construction.")

    decision = {
        "study_status": study_status,
        "supported_claims": supported_claims,
        "weakened_claims": weakened_claims,
        "failed_claims": failed_claims,
        "metric_tables": {name: results[name]["scenarios"] for name in MAIN_METHODS},
        "comparisons": comparisons,
        "data_preparation": load_result("data_preparation"),
        "official_compbench_eval": load_result("official_compbench_eval"),
        "reproduction_failures": {
            "aapb": load_result("aapb_reproduction"),
            "single_prompt_baseline": load_result("single_prompt_baseline"),
            "human_subset": load_result("human_reliability_subset"),
            "paraphrase_double_annotation": load_result("paraphrase_double_annotation"),
        },
        "pilot": pilot,
    }
    write_json(ROOT / "final_decision_report.json", decision)


if __name__ == "__main__":
    main()
