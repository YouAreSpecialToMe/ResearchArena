from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from exp.shared.attention import TraceCollector, install_recording_processors, restore_processors
from exp.shared.models import DEVICE, load_sdxl, siglip_score_image_text
from exp.shared.rerank import score_candidate
from exp.shared.utils import Timer, ensure_dir, mean_std, project_path, read_jsonl, seed_everything, write_json, write_jsonl


SEEDS = [11, 22, 33]


def save_heatmaps(path: Path, records) -> None:
    ensure_dir(path.parent)
    payload = {record.phrase: record.heatmap for record in records}
    np.savez_compressed(path, **payload)


def generate_one(record: dict, dataset_name: str, seed: int, candidate_id: int, steps: int = 30, guidance_scale: float = 7.5, size: int = 512) -> dict:
    image_path = project_path("artifacts", "candidate_cache", dataset_name, record["prompt_id"], f"seed_{seed}", f"cand_{candidate_id}.png")
    heatmap_path = project_path("artifacts", "daam_cache", dataset_name, record["prompt_id"], f"seed_{seed}", f"cand_{candidate_id}.npz")
    phrases = [group["noun"] for group in record["parse"]["object_groups"]]
    if image_path.exists() and heatmap_path.exists():
        return {"image_path": image_path, "heatmap_path": heatmap_path, "phrases": phrases, "cache_hit": True}
    if dataset_name == "candidate_budget" and candidate_id < 4:
        shared_image = project_path("artifacts", "candidate_cache", "test", record["prompt_id"], f"seed_{seed}", f"cand_{candidate_id}.png")
        shared_heatmap = project_path("artifacts", "daam_cache", "test", record["prompt_id"], f"seed_{seed}", f"cand_{candidate_id}.npz")
        if shared_image.exists() and shared_heatmap.exists():
            ensure_dir(image_path.parent)
            ensure_dir(heatmap_path.parent)
            shutil.copy2(shared_image, image_path)
            shutil.copy2(shared_heatmap, heatmap_path)
            return {"image_path": image_path, "heatmap_path": heatmap_path, "phrases": phrases, "cache_hit": True}

    pipe = load_sdxl()
    generator = torch.Generator(device=DEVICE).manual_seed(seed * 100 + candidate_id)
    prompt = record["prompt"]
    collector = TraceCollector(pipe.tokenizer, prompt, phrases, keep_last_steps=10)
    original = install_recording_processors(pipe, collector)
    try:
        out = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=size,
            width=size,
            generator=generator,
        )
        image = out.images[0]
    finally:
        restore_processors(pipe, original)
    heatmaps = collector.finalize()
    ensure_dir(image_path.parent)
    image.save(image_path)
    save_heatmaps(heatmap_path, heatmaps)
    return {"image_path": image_path, "heatmap_path": heatmap_path, "phrases": phrases, "cache_hit": False}


def load_split(name: str) -> list[dict]:
    return read_jsonl(project_path("data", "splits", f"{name}.jsonl"))


def merge_result_rows(path: Path, new_rows: list[dict]) -> list[dict]:
    existing = read_jsonl(path) if path.exists() else []
    new_keys = {(row["dataset"], row["prompt_id"], row["seed"]) for row in new_rows}
    kept = [row for row in existing if (row["dataset"], row["prompt_id"], row["seed"]) not in new_keys]
    merged = kept + new_rows
    write_jsonl(path, merged)
    return merged


def run_generation(split_name: str, k: int, limit: int | None = None) -> list[dict]:
    rows = load_split(split_name)
    if limit is not None:
        rows = rows[:limit]
    outputs = []
    total = len(rows) * len(SEEDS) * k
    done = 0
    print(f"[generate] split={split_name} prompts={len(rows)} seeds={len(SEEDS)} k={k} total_images={total}", flush=True)
    for record in rows:
        for seed in SEEDS:
            for cand in range(k):
                timer = Timer()
                meta = generate_one(record, split_name, seed, cand)
                done += 1
                print(
                    f"[generate] split={split_name} prompt_id={record['prompt_id']} seed={seed} cand={cand} "
                    f"cache_hit={meta['cache_hit']} runtime_sec={timer.seconds:.2f} progress={done}/{total}",
                    flush=True,
                )
                outputs.append(
                    {
                        "dataset": split_name,
                        "prompt_id": record["prompt_id"],
                        "prompt": record["prompt"],
                        "seed": seed,
                        "candidate_id": cand,
                        "runtime_sec": timer.seconds,
                        **meta,
                    }
                )
    write_jsonl(project_path("artifacts", "candidate_cache", f"{split_name}_metadata.jsonl"), outputs)
    return outputs


def select_with_global_siglip(record: dict, split_name: str, k: int) -> dict:
    best = None
    for cand in range(k):
        image_path = project_path("artifacts", "candidate_cache", split_name, record["prompt_id"], f"seed_{record['seed']}", f"cand_{cand}.png")
        image = Image.open(image_path).convert("RGB")
        score = siglip_score_image_text(image, record["prompt"])
        item = {"candidate_id": cand, "final_score": score}
        if best is None or score > best["final_score"]:
            best = item
    return best


def selected_heatmap_path(split_name: str, prompt_id: str, seed: int, candidate_id: int, assignment_source: str) -> Path | None:
    if assignment_source != "detector_daam":
        return None
    return project_path("artifacts", "daam_cache", split_name, prompt_id, f"seed_{seed}", f"cand_{candidate_id}.npz")


def save_audit_artifact(split_name: str, method: str, record: dict, seed: int, best: dict) -> None:
    audit_path = project_path("artifacts", "audits", method, split_name, record["prompt_id"], f"seed_{seed}.json")
    write_json(
        audit_path,
        {
            "dataset": split_name,
            "prompt_id": record["prompt_id"],
            "prompt": record["prompt"],
            "seed": seed,
            "parse": record["parse"],
            "selected_candidate_id": best["candidate_id"],
            "final_score": best["final_score"],
            "all_atoms_pass": best["all_atoms_pass"],
            "atomic_scores": best["atomic_scores"],
            "group_counts": best["group_counts"],
            "slot_assignments": best["slot_assignments"],
            "phrase_boxes": best.get("phrase_boxes", {}),
        },
    )


def run_method(
    split_name: str,
    method: str,
    k: int = 4,
    assignment_source: str = "detector",
    use_counterfactual: bool = False,
    aggregation: str = "geometric",
    force_non_null: bool = False,
    limit: int | None = None,
) -> list[dict]:
    rows = load_split(split_name)
    if limit is not None:
        rows = rows[:limit]
    results = []
    total = len(rows) * len(SEEDS)
    done = 0
    print(
        f"[method] name={method} split={split_name} prompts={len(rows)} seeds={len(SEEDS)} "
        f"k={k} assignment_source={assignment_source} counterfactual={use_counterfactual} "
        f"aggregation={aggregation} force_non_null={force_non_null}",
        flush=True,
    )
    for record in rows:
        for seed in SEEDS:
            timer = Timer()
            best = None
            per_candidate = []
            for cand in range(k):
                image_path = project_path("artifacts", "candidate_cache", split_name, record["prompt_id"], f"seed_{seed}", f"cand_{cand}.png")
                scored = score_candidate(
                    record=record,
                    image_path=image_path,
                    heatmap_path=selected_heatmap_path(split_name, record["prompt_id"], seed, cand, assignment_source),
                    method=method,
                    assignment_source=assignment_source,
                    use_counterfactual=use_counterfactual,
                    aggregation=aggregation,
                    force_non_null=force_non_null,
                )
                scored["candidate_id"] = cand
                per_candidate.append(scored)
                if best is None or scored["final_score"] > best["final_score"]:
                    best = scored
            assert best is not None
            results.append(
                {
                    "method": method,
                    "dataset": split_name,
                    "prompt_id": record["prompt_id"],
                    "prompt": record["prompt"],
                    "seed": seed,
                    "selected_candidate_id": best["candidate_id"],
                    "final_score": best["final_score"],
                    "atomic_scores": best["atomic_scores"],
                    "group_counts": best["group_counts"],
                    "slot_assignments": best["slot_assignments"],
                    "all_atoms_pass": best["all_atoms_pass"],
                    "latency_sec": timer.seconds,
                    "feature_cache_paths": {
                        "candidate_dir": str(project_path("artifacts", "candidate_cache", split_name, record["prompt_id"], f"seed_{seed}")),
                        "daam_dir": str(project_path("artifacts", "daam_cache", split_name, record["prompt_id"], f"seed_{seed}")),
                    },
                }
            )
            save_audit_artifact(split_name, method, record, seed, best)
            done += 1
            print(
                f"[method] name={method} split={split_name} prompt_id={record['prompt_id']} seed={seed} "
                f"selected={best['candidate_id']} final_score={best['final_score']:.4f} "
                f"all_atoms_pass={best['all_atoms_pass']} latency_sec={timer.seconds:.2f} progress={done}/{total}",
                flush=True,
            )
    out_path = project_path("results", f"{method}.jsonl")
    merged = merge_result_rows(out_path, results)
    write_json(project_path("exp", method, "results.json"), {"num_rows": len(merged), "metrics": aggregate_results({method: merged})[method]})
    return results


def run_single_sample(split_name: str, method: str = "single_sample", assignment_source: str = "detector", limit: int | None = None) -> list[dict]:
    rows = load_split(split_name)
    if limit is not None:
        rows = rows[:limit]
    results = []
    total = len(rows) * len(SEEDS)
    done = 0
    print(f"[single_sample] split={split_name} prompts={len(rows)} seeds={len(SEEDS)}", flush=True)
    for record in rows:
        for seed in SEEDS:
            timer = Timer()
            scored = score_candidate(
                record=record,
                image_path=project_path("artifacts", "candidate_cache", split_name, record["prompt_id"], f"seed_{seed}", "cand_0.png"),
                heatmap_path=selected_heatmap_path(split_name, record["prompt_id"], seed, 0, assignment_source),
                method=method,
                assignment_source=assignment_source,
                use_counterfactual=False,
                aggregation="geometric",
                force_non_null=False,
            )
            scored["candidate_id"] = 0
            results.append(
                {
                    "method": method,
                    "dataset": split_name,
                    "prompt_id": record["prompt_id"],
                    "prompt": record["prompt"],
                    "seed": seed,
                    "selected_candidate_id": 0,
                    "final_score": scored["final_score"],
                    "atomic_scores": scored["atomic_scores"],
                    "group_counts": scored["group_counts"],
                    "slot_assignments": scored["slot_assignments"],
                    "all_atoms_pass": scored["all_atoms_pass"],
                    "latency_sec": timer.seconds,
                    "feature_cache_paths": {
                        "candidate_dir": str(project_path("artifacts", "candidate_cache", split_name, record["prompt_id"], f"seed_{seed}")),
                        "daam_dir": str(project_path("artifacts", "daam_cache", split_name, record["prompt_id"], f"seed_{seed}")),
                    },
                }
            )
            save_audit_artifact(split_name, method, record, seed, scored | {"candidate_id": 0})
            done += 1
            print(
                f"[single_sample] split={split_name} prompt_id={record['prompt_id']} seed={seed} "
                f"final_score={scored['final_score']:.4f} all_atoms_pass={scored['all_atoms_pass']} "
                f"latency_sec={timer.seconds:.2f} progress={done}/{total}",
                flush=True,
            )
    merged = merge_result_rows(project_path("results", f"{method}.jsonl"), results)
    write_json(project_path("exp", method, "results.json"), {"num_rows": len(merged), "metrics": aggregate_results({method: merged})[method]})
    return results


def aggregate_results(result_sets: dict[str, list[dict]]) -> dict:
    metrics = {}
    for name, rows in result_sets.items():
        all_atoms = [1.0 if row["all_atoms_pass"] else 0.0 for row in rows]
        final_scores = [row["final_score"] for row in rows]
        latencies = [row["latency_sec"] for row in rows]
        count_scores = []
        for row in rows:
            for g in row["group_counts"].values():
                count_scores.append(g["score"])
        metrics[name] = {
            "all_atoms_pass": mean_std(all_atoms),
            "final_score": mean_std(final_scores),
            "group_count_score": mean_std(count_scores),
            "latency_sec": mean_std(latencies),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "method"], required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", default="assign_and_verify")
    parser.add_argument("--assignment-source", choices=["detector", "crop_siglip", "detector_daam"], default="detector")
    parser.add_argument("--use-counterfactual", action="store_true")
    parser.add_argument("--aggregation", default="geometric")
    parser.add_argument("--force-non-null", action="store_true")
    args = parser.parse_args()

    seed_everything(17)
    if args.mode == "generate":
        rows = run_generation(args.split, args.k, args.limit)
        write_json(project_path("exp", "generate_cache", "results.json"), {"split": args.split, "num_images": len(rows)})
    else:
        rows = run_method(
            split_name=args.split,
            method=args.method,
            k=args.k,
            assignment_source=args.assignment_source,
            use_counterfactual=args.use_counterfactual,
            aggregation=args.aggregation,
            force_non_null=args.force_non_null,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
