#!/usr/bin/env python3
"""
Re-score all generated images with ImageReward after the main experiment finishes.
The main experiment used a broken aesthetic predictor; this script fixes it.
"""
import torch
import json
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import pytorch_utils

# Patch transformers for ImageReward compatibility
for attr in dir(pytorch_utils):
    if not hasattr(transformers.modeling_utils, attr):
        setattr(transformers.modeling_utils, attr, getattr(pytorch_utils, attr))

import ImageReward as RM

WORKSPACE = Path(__file__).parent.parent
EXP_DIR = WORKSPACE / "exp"
DATA_DIR = EXP_DIR / "data"

def load_prompts(name, n=None):
    files = {"coco": "coco_500_prompts.json", "parti": "parti_200_prompts.json"}
    with open(DATA_DIR / files[name]) as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]
    return prompts[:n] if n else prompts


def score_directory(model, image_dir, prompts, n=None):
    """Score all images in a directory with ImageReward."""
    scores = []
    for i, prompt in enumerate(prompts[:n] if n else prompts):
        img_path = image_dir / f"{i:05d}.png"
        if img_path.exists():
            img = Image.open(img_path)
            score = float(model.score(prompt, img))
            scores.append(score)
        else:
            scores.append(None)
    valid = [s for s in scores if s is not None]
    return {
        "ir_mean": float(np.mean(valid)) if valid else 0,
        "ir_std": float(np.std(valid)) if valid else 0,
        "n_scored": len(valid),
    }


def main():
    print("Loading ImageReward model...")
    model = RM.load("ImageReward-v1.0", device="cuda")

    prompts_coco = load_prompts("coco", 300)
    prompts_parti = load_prompts("parti", 100)

    # Load existing results
    results_path = WORKSPACE / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}

    ir_results = {}

    # Score baselines
    print("\nScoring baselines...")
    baselines_dir = EXP_DIR / "baselines"
    for method_dir in sorted(baselines_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        for ds_dir in sorted(method_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            ds_name = ds_dir.name
            prompts = prompts_coco if ds_name == "coco" else prompts_parti
            key = f"{method_dir.name}_{ds_name}"
            print(f"  Scoring {key}...")
            scores = score_directory(model, ds_dir, prompts)
            ir_results[key] = scores
            print(f"    IR={scores['ir_mean']:.4f}±{scores['ir_std']:.4f} (n={scores['n_scored']})")

    # Score main methods
    print("\nScoring main methods...")
    main_dir = EXP_DIR / "main"
    if main_dir.exists():
        for method_dir in sorted(main_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            for ds_dir in sorted(method_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                ds_name = ds_dir.name
                prompts = prompts_coco if ds_name == "coco" else prompts_parti
                key = f"{method_dir.name}_{ds_name}"
                print(f"  Scoring {key}...")
                scores = score_directory(model, ds_dir, prompts)
                ir_results[key] = scores
                print(f"    IR={scores['ir_mean']:.4f}±{scores['ir_std']:.4f} (n={scores['n_scored']})")

    # Update results.json
    results["imagereward_scores"] = ir_results

    # Also update individual method entries
    for section_key in ["baselines", "particles_coco", "particles_parti"]:
        if section_key in results:
            for method_key, method_data in results[section_key].items():
                # Find matching IR result
                ds = method_data.get("dataset", "coco")
                seed = method_data.get("seed", 42)
                method = method_data.get("method", "")
                search_key = f"{method}_seed{seed}_{ds}"
                # Try various key formats
                for ir_key in [search_key, f"{method}_{method_data.get('steps','50')}_seed{seed}_{ds}"]:
                    if ir_key in ir_results:
                        results[section_key][method_key]["ir_mean"] = ir_results[ir_key]["ir_mean"]
                        results[section_key][method_key]["ir_std"] = ir_results[ir_key]["ir_std"]
                        break

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults updated in {results_path}")

    # Also rescore PCS correlation data
    print("\nRescoring PCS correlation data...")
    pcs_data_file = EXP_DIR / "analysis" / "pcs_data_coco.json"
    if pcs_data_file.exists():
        rescore_pcs_correlation(model, pcs_data_file, prompts_coco)


def rescore_pcs_correlation(model, data_file, prompts):
    """Rescore particles for PCS correlation analysis."""
    with open(data_file) as f:
        pcs_data = json.load(f)

    # For each prompt's particles, we need to score them
    # But we don't have the individual particle images saved (only selected ones)
    # So we skip this for now - the CLIP correlation is the primary metric
    print("  (PCS correlation rescoring requires particle images - skipping)")


if __name__ == "__main__":
    main()
