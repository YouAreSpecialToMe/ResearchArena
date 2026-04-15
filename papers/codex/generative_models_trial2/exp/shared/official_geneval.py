from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
GENEVAL_ROOT = ROOT / "data" / "geneval"
GENEVAL_EVAL_SCRIPT = GENEVAL_ROOT / "evaluation" / "evaluate_images.py"
GENEVAL_MODEL_PATH = GENEVAL_ROOT / "models"
GENEVAL_MODEL_CONFIG = GENEVAL_ROOT / "mmdetection" / "configs" / "mask2former" / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
GENEVAL_PYTHON = Path("/home/zz865/.conda/envs/geneval39/bin/python")


@dataclass(frozen=True)
class EvalSpec:
    prompt_id: str
    spec_id: str
    metadata: Dict[str, Any]


def _clean_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "tag": row["tag"],
        "include": row.get("include", []),
        "exclude": row.get("exclude", []),
        "prompt": row["prompt"],
    }


def build_eval_specs(row: Mapping[str, Any]) -> List[EvalSpec]:
    prompt_id = str(row["prompt_id"])
    full = EvalSpec(prompt_id=prompt_id, spec_id="full_prompt", metadata=_clean_metadata(row))
    if row["tag"] == "color_attr":
        specs = [full]
        for idx, obj in enumerate(row["include"]):
            specs.append(
                EvalSpec(
                    prompt_id=prompt_id,
                    spec_id=f"atomic_attr_{idx}",
                    metadata={
                        "tag": row["tag"],
                        "include": [obj],
                        "exclude": [],
                        "prompt": row["prompt"],
                    },
                )
            )
        return specs
    return [full]


def run_official_geneval_batch(
    eval_name: str,
    specs: Sequence[EvalSpec],
    samples_by_prompt: Mapping[str, Sequence[Dict[str, Any]]],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    input_dir = out_dir / "input"
    results_path = out_dir / "results.jsonl"
    manifest_path = out_dir / "manifest.jsonl"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    ordered_specs = sorted(specs, key=lambda item: (item.prompt_id, item.spec_id))
    for folder_index, spec in enumerate(ordered_specs):
        prompt_samples = sorted(samples_by_prompt[spec.prompt_id], key=lambda item: str(item["sample_key"]))
        folder = input_dir / f"{folder_index:05d}"
        samples_dir = folder / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        (folder / "metadata.jsonl").write_text(json.dumps(spec.metadata) + "\n")
        for sample_index, sample in enumerate(prompt_samples):
            src = Path(sample["image_path"]).resolve()
            dst = samples_dir / f"{sample_index:04d}.png"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src, dst)
            manifest_rows.append(
                {
                    "folder": f"{folder_index:05d}",
                    "sample_index": sample_index,
                    "prompt_id": spec.prompt_id,
                    "spec_id": spec.spec_id,
                    "sample_key": str(sample["sample_key"]),
                    "image_path": str(src),
                    "eval_name": eval_name,
                }
            )
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    cmd = [
        str(GENEVAL_PYTHON),
        str(GENEVAL_EVAL_SCRIPT),
        str(input_dir),
        "--outfile",
        str(results_path),
        "--model-path",
        str(GENEVAL_MODEL_PATH),
        "--model-config",
        str(GENEVAL_MODEL_CONFIG),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    manifest_map: Dict[Tuple[str, int], Dict[str, Any]] = {
        (row["folder"], int(row["sample_index"])): row for row in manifest_rows
    }
    merged: List[Dict[str, Any]] = []
    with results_path.open() as handle:
        for line in handle:
            result = json.loads(line)
            filename = Path(result["filename"])
            folder = filename.parent.parent.name
            sample_index = int(filename.stem)
            info = manifest_map[(folder, sample_index)]
            merged.append({**info, **result})
    return merged


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
