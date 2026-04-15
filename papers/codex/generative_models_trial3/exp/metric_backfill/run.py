from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import EXP_DIR, environment_manifest, package_versions, read_csv, records_from_jsonl, stable_hash, write_csv, write_json
from exp.shared.metrics import CLIPScorer, DINOScorer, ImageRewardScorer, LPIPSScorer, pairwise_lpips
from exp.shared.runner import _summarize


DATA_PATH = ROOT / "data" / "prompts_with_paraphrases.jsonl"
EXPERIMENTS = [
    "vanilla_sd15",
    "static_consensus",
    "adaptive_ungated",
    "paradg",
    "ablation_no_gate",
    "ablation_no_slot",
    "ablation_no_timestep",
    "ablation_reduced_paraphrase",
    "ablation_nonequivalent",
]


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
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
    parsed = []
    for row in rows:
        clean: dict[str, Any] = dict(row)
        for key in bool_fields:
            if key in clean:
                clean[key] = clean[key] in {"True", "true"}
        for key in int_fields:
            if key in clean and clean[key] != "":
                clean[key] = int(clean[key])
        for key in float_fields:
            if key in clean and clean[key] != "":
                clean[key] = float(clean[key])
        parsed.append(clean)
    return parsed


def _serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                clean[key] = ""
            else:
                clean[key] = value
        serialized.append(clean)
    return serialized


def _pairwise_mean(paths: list[Path], scorer_fn) -> float:
    if len(paths) < 2:
        return float("nan")
    values = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            values.append(float(scorer_fn(paths[i], paths[j])))
    return float(sum(values) / len(values))


def _backfill_rows(rows: list[dict[str, Any]], image_reward: ImageRewardScorer, clip: CLIPScorer, dino: DINOScorer, lpips: LPIPSScorer) -> list[dict[str, Any]]:
    slot_keys = [
        "object_1_present",
        "object_2_present",
        "attribute_1_correct",
        "attribute_2_correct",
        "relation_correct",
        "count_1_correct",
    ]

    for row in rows:
        image_path = Path(row["output_path"])
        row["image_reward"] = image_reward.image_text_score(image_path, row["prompt_text"])

    by_scenario_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_prompt_seed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario_prompt[(row["scenario"], row["prompt_id"])].append(row)
        if row["scenario"] == "robustness":
            by_prompt_seed[(row["scenario"], row["prompt_id"], row["seed"])].append(row)

    for (_, _, _), group in by_prompt_seed.items():
        ordered = sorted(group, key=lambda item: item["prompt_variant_id"])
        prs_values = []
        clip_values = []
        dino_values = []
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                left = ordered[i]
                right = ordered[j]
                prs_values.append(1.0 if all(left[key] == right[key] for key in slot_keys) else 0.0)
                left_path = Path(left["output_path"])
                right_path = Path(right["output_path"])
                clip_values.append(clip.image_image_score(left_path, right_path))
                dino_values.append(dino.image_image_score(left_path, right_path))
        prs = float(sum(prs_values) / len(prs_values)) if prs_values else float("nan")
        clip_consistency = float(sum(clip_values) / len(clip_values)) if clip_values else float("nan")
        dino_consistency = float(sum(dino_values) / len(dino_values)) if dino_values else float("nan")
        for row in group:
            row["prompt_seed_prs"] = prs
            row["prompt_seed_clip_consistency"] = clip_consistency
            row["prompt_seed_dino_consistency"] = dino_consistency

    for (scenario, prompt_id), group in by_scenario_prompt.items():
        orig_rows = [row for row in group if row["prompt_variant_id"].endswith("::orig")]
        if len(orig_rows) < 2:
            continue
        paths = [Path(row["output_path"]) for row in sorted(orig_rows, key=lambda item: item["seed"])]
        lpips_div = pairwise_lpips(paths, lpips)
        dino_sim = _pairwise_mean(paths, dino.image_image_score)
        dino_disp = float(1.0 - dino_sim) if dino_sim == dino_sim else float("nan")
        for row in orig_rows:
            row["lpips_seed_diversity"] = lpips_div
            row["dino_seed_dispersion"] = dino_disp

    return rows


def _manifest_backfill() -> dict[str, Any]:
    manifest = environment_manifest()
    latent_rows = []
    for exp_name in EXPERIMENTS:
        csv_path = EXP_DIR / exp_name / "generation_index.csv"
        if not csv_path.exists():
            continue
        for row in _parse_rows(csv_path):
            latent_rows.append(
                {
                    "experiment": exp_name,
                    "scenario": row["scenario"],
                    "prompt_id": row["prompt_id"],
                    "prompt_variant_id": row["prompt_variant_id"],
                    "seed": row["seed"],
                    "latent_hash": row["latent_hash"],
                }
            )
    manifest["package_versions"] = package_versions()
    manifest["latent_hash_records_path"] = str(EXP_DIR / "environment_setup" / "latent_hashes.json")
    manifest["package_freeze_path"] = str(ROOT / "requirements_lock.txt")
    manifest["xformers_available"] = manifest["package_versions"].get("xformers") is not None
    manifest["preregistered_runtime_match"] = False
    manifest["runtime_revision_note"] = (
        "Executed under the available Python 3.12 / torch 2.10 runtime. "
        "This is a formal revision from the preregistered Python 3.10 / torch 2.2 stack, not a silent match."
    )
    write_json(EXP_DIR / "environment_setup" / "latent_hashes.json", latent_rows)
    freeze = {
        "python": manifest["python_version"],
        "package_versions": manifest["package_versions"],
        "runtime_revision_note": manifest["runtime_revision_note"],
    }
    write_json(ROOT / "requirements_lock.txt", freeze)
    write_json(EXP_DIR / "environment_setup" / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    records_from_jsonl(DATA_PATH)
    clip = CLIPScorer()
    dino = DINOScorer()
    lpips = LPIPSScorer()
    image_reward = ImageRewardScorer()

    manifest = _manifest_backfill()
    summary: dict[str, Any] = {
        "experiment": "metric_backfill",
        "status": "completed",
        "updated_experiments": [],
        "manifest_path": str(EXP_DIR / "environment_setup" / "run_manifest.json"),
    }

    for exp_name in EXPERIMENTS:
        csv_path = EXP_DIR / exp_name / "generation_index.csv"
        if not csv_path.exists():
            continue
        rows = _parse_rows(csv_path)
        rows = _backfill_rows(rows, image_reward=image_reward, clip=clip, dino=dino, lpips=lpips)
        write_csv(csv_path, _serialize_rows(rows))
        result = {
            "experiment": exp_name,
            "config": json.loads((EXP_DIR / exp_name / "results.json").read_text()).get("config", {}),
            "raw_rows_path": str(csv_path),
            "scenarios": _summarize(rows, clip, lpips),
        }
        write_json(EXP_DIR / exp_name / "results.json", result)
        summary["updated_experiments"].append(
            {
                "experiment": exp_name,
                "rows": len(rows),
                "csv_path": str(csv_path),
                "results_path": str(EXP_DIR / exp_name / "results.json"),
            }
        )

    summary["environment"] = {
        "python_version": manifest["python_version"],
        "torch_version": manifest["torch_version"],
        "xformers_available": manifest["xformers_available"],
    }
    write_json(EXP_DIR / "metric_backfill" / "results.json", summary)


if __name__ == "__main__":
    main()
