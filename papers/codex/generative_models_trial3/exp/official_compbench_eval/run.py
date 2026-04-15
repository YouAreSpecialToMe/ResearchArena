from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.shared.common import EXP_DIR, read_csv, write_csv, write_json
from exp.shared.metrics import CLIPScorer, LPIPSScorer
from exp.shared.runner import _summarize


REPO_DIR = ROOT / "external" / "T2I-CompBench"
BLIP_DIR = REPO_DIR / "BLIPvqa_eval"
UNIDET_DIR = REPO_DIR / "UniDet_eval"
WORK_DIR = EXP_DIR / "official_compbench_eval" / "work"

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


def _sanitize_filename(prompt: str) -> str:
    return prompt.replace("/", " ").replace("\\", " ").replace("\n", " ").strip()


def _load_rows(path: Path) -> list[dict[str, Any]]:
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
        for key in bool_fields:
            if key in row:
                row[key] = row[key] in {"True", "true"}
        for key in float_fields:
            if key in row and row[key] not in {None, ""}:
                row[key] = float(row[key])
        row["seed"] = int(row["seed"])
    return rows


def _serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                clean[key] = ""
            else:
                clean[key] = value
        clean_rows.append(clean)
    return clean_rows


def _write_samples(rows: list[dict[str, Any]], out_dir: Path) -> dict[int, dict[str, Any]]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, dict[str, Any]] = {}
    for question_id, row in enumerate(rows):
        prompt = _sanitize_filename(row["prompt_text"])
        dst = sample_dir / f"{prompt}_{question_id:06d}.png"
        src = Path(row["output_path"]).resolve()
        os.symlink(src, dst)
        mapping[question_id] = row
    return mapping


def _run_blip(out_dir: Path) -> list[dict[str, Any]]:
    subprocess.run(
        [sys.executable, "BLIP_vqa.py", "--out_dir", str(out_dir)],
        cwd=BLIP_DIR,
        check=True,
    )
    result_path = out_dir / "annotation_blip" / "vqa_result.json"
    with result_path.open() as f:
        return json.load(f)


def _run_unidet(script_name: str, out_dir: Path) -> list[dict[str, Any]]:
    if script_name == "numeracy_eval.py":
        code = """
import PIL.Image
PIL.Image.LINEAR = PIL.Image.BILINEAR
import sys
sys.path.insert(0, '.')
import experts.model_bank as mb
orig = mb.load_expert_model
def patched(task=None, ckpt=None):
    if task == 'obj_detection' and ckpt == 'R50':
        ckpt = 'RS200'
    return orig(task=task, ckpt=ckpt)
mb.load_expert_model = patched
import numeracy_eval
numeracy_eval.load_expert_model = patched
sys.argv = ['numeracy_eval.py', '--outpath', sys.argv[1]]
numeracy_eval.main()
"""
        subprocess.run([sys.executable, "-c", code, str(out_dir)], cwd=UNIDET_DIR, check=True)
        result_path = out_dir / "annotation_num" / "vqa_result.json"
    else:
        code = """
import importlib.util
import PIL.Image
import sys
PIL.Image.LINEAR = PIL.Image.BILINEAR
sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('spatial2d', '2D_spatial_eval.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.argv = ['2D_spatial_eval.py', '--outpath', sys.argv[1]]
mod.main()
"""
        subprocess.run([sys.executable, "-c", code, str(out_dir)], cwd=UNIDET_DIR, check=True)
        result_path = out_dir / "labels" / "annotation_obj_detection_2d" / "vqa_result.json"
    with result_path.open() as f:
        return json.load(f)


def _update_category_scores(rows: list[dict[str, Any]], scenario: str, category: str) -> dict[str, Any]:
    target_rows = [
        row
        for row in rows
        if row["scenario"] == scenario and row["category"] == category and row["prompt_variant_id"].endswith("::orig")
    ]
    if not target_rows:
        return {"num_rows": 0, "status": "skipped"}

    out_dir = WORK_DIR / scenario / category
    mapping = _write_samples(target_rows, out_dir)

    if category == "attribute_binding":
        results = _run_blip(out_dir)
        metric_source = "official_blip_vqa"
    elif category == "numeracy":
        results = _run_unidet("numeracy_eval.py", out_dir)
        metric_source = "official_unidet_numeracy_rs200_compat"
    else:
        results = _run_unidet("2D_spatial_eval.py", out_dir)
        metric_source = "official_unidet_spatial_rs200_compat"

    for result in results:
        question_id = int(result["question_id"])
        if question_id not in mapping:
            continue
        score = float(result["answer"])
        row = mapping[question_id]
        row["category_score"] = score

    return {
        "num_rows": len(target_rows),
        "metric_source": metric_source,
        "output_dir": str(out_dir),
    }


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    clip = CLIPScorer()
    lpips = LPIPSScorer()
    report: dict[str, Any] = {
        "experiment": "official_compbench_eval",
        "status": "completed",
        "compatibility_notes": [
            "BLIP-VQA ran from the published T2I-CompBench repository without model-code edits.",
            "UniDet evaluation required a Pillow compatibility shim that defines PIL.Image.LINEAR as PIL.Image.BILINEAR under the current Python 3.12 / torch 2.10 stack.",
            "The published numeracy script requests an R50 checkpoint that was not available at the repository URL; this backfill uses the repository's published RS200 detector weight instead and records that deviation explicitly.",
        ],
        "updated_experiments": [],
    }

    for experiment in EXPERIMENTS:
        csv_path = EXP_DIR / experiment / "generation_index.csv"
        results_path = EXP_DIR / experiment / "results.json"
        if not csv_path.exists() or not results_path.exists():
            continue
        rows = _load_rows(csv_path)
        update_log = defaultdict(dict)
        for scenario in sorted({row["scenario"] for row in rows}):
            if scenario not in {"faithfulness", "robustness", "analysis"}:
                continue
            for category in ["attribute_binding", "relations", "numeracy"]:
                update_log[scenario][category] = _update_category_scores(rows, scenario, category)

        write_csv(csv_path, _serialize_rows(rows))
        original_results = json.loads(results_path.read_text())
        updated = {
            "experiment": experiment,
            "config": original_results.get("config", {}),
            "raw_rows_path": str(csv_path),
            "scenarios": _summarize(rows, clip, lpips),
            "metric_provenance": {
                "category_scores": "official_t2i_compbench_family",
                "robustness_and_slot_judgments": "custom_slot_based_exploratory",
            },
            "official_compbench_backfill": update_log,
        }
        write_json(results_path, updated)
        report["updated_experiments"].append(
            {
                "experiment": experiment,
                "csv_path": str(csv_path),
                "results_path": str(results_path),
                "updates": update_log,
            }
        )

    write_json(EXP_DIR / "official_compbench_eval" / "results.json", report)


if __name__ == "__main__":
    main()
