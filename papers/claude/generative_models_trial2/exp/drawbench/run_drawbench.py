#!/usr/bin/env python3
"""Generate DrawBench images for all methods and compute metrics.

DrawBench has only 41 prompts, so this is fast (~5 min total).
"""
import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Fix ImageReward
import transformers
from transformers import pytorch_utils
for attr in dir(pytorch_utils):
    if not hasattr(transformers.modeling_utils, attr):
        setattr(transformers.modeling_utils, attr, getattr(pytorch_utils, attr))

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

WORKSPACE = Path(__file__).parent.parent.parent
DATA_DIR = WORKSPACE / "exp" / "data"
OUT_DIR = Path(__file__).parent

def load_prompts():
    with open(DATA_DIR / "drawbench_prompts.json") as f:
        data = json.load(f)
    return [item["prompt"] if isinstance(item, dict) else item for item in data]

def main():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from cops import generate_particles_batch
    import ImageReward as RM
    import open_clip

    prompts = load_prompts()
    N = len(prompts)
    K = 4
    SEEDS = [42, 123, 456]
    print(f"DrawBench: {N} prompts, K={K}, seeds={SEEDS}")

    # Load models
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    ir_model = RM.load("ImageReward-v1.0", device="cuda")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model.eval()

    all_results = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        # Generate K=4 particles per prompt
        particle_data = []
        for i, prompt in enumerate(tqdm(prompts, desc=f"DrawBench s{seed}")):
            result = generate_particles_batch(
                pipe, prompt, num_particles=K, num_inference_steps=50,
                guidance_scale=7.5, seed=seed + i, track_pcs=True, distance_metric="l2"
            )
            images = result["images"]
            pcs_scores = result["pcs_scores"]

            # Score all particles
            clip_scores = []
            ir_scores = []
            for k in range(K):
                img_arr = (images[k].cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                pil_img = Image.fromarray(img_arr)

                # CLIP
                img_t = clip_preprocess(pil_img).unsqueeze(0).to("cuda")
                txt_t = clip_tokenizer([prompt]).to("cuda")
                with torch.no_grad():
                    img_f = clip_model.encode_image(img_t)
                    txt_f = clip_model.encode_text(txt_t)
                    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                    clip_s = float((img_f * txt_f).sum())
                clip_scores.append(clip_s)
                ir_scores.append(float(ir_model.score(prompt, pil_img)))

            particle_data.append({
                "prompt": prompt,
                "pcs_scores": pcs_scores,
                "clip_scores": clip_scores,
                "ir_scores": ir_scores,
            })

            # Save selected images for each method
            for method, sel_idx in [
                ("random_k", np.random.RandomState(seed).randint(0, K)),
                ("pcs_bestofk", int(np.argmax(pcs_scores))),
                ("clip_bestofk", int(np.argmax(clip_scores))),
                ("ir_bestofk", int(np.argmax(ir_scores))),
            ]:
                method_dir = OUT_DIR / f"{method}_seed{seed}"
                method_dir.mkdir(parents=True, exist_ok=True)
                img_arr = (images[sel_idx].cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                Image.fromarray(img_arr).save(method_dir / f"{i:05d}.png")

            # Also save DDIM baseline (first particle)
            baseline_dir = OUT_DIR / f"ddim_50_seed{seed}"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            img_arr = (images[0].cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
            Image.fromarray(img_arr).save(baseline_dir / f"{i:05d}.png")

        # Aggregate results per method
        for method, sel_fn in [
            ("random_k", lambda pd: np.random.RandomState(seed).randint(0, K)),
            ("pcs_bestofk", lambda pd: int(np.argmax(pd["pcs_scores"]))),
            ("clip_bestofk", lambda pd: int(np.argmax(pd["clip_scores"]))),
            ("ir_bestofk", lambda pd: int(np.argmax(pd["ir_scores"]))),
            ("ddim_50", lambda pd: 0),
        ]:
            clip_sel = []
            ir_sel = []
            rng = np.random.RandomState(seed)
            for pd in particle_data:
                if method == "random_k":
                    idx = rng.randint(0, K)
                elif method == "pcs_bestofk":
                    idx = int(np.argmax(pd["pcs_scores"]))
                elif method == "clip_bestofk":
                    idx = int(np.argmax(pd["clip_scores"]))
                elif method == "ir_bestofk":
                    idx = int(np.argmax(pd["ir_scores"]))
                else:
                    idx = 0
                clip_sel.append(pd["clip_scores"][idx])
                ir_sel.append(pd["ir_scores"][idx])

            key = f"{method}_drawbench_seed{seed}"
            all_results[key] = {
                "clip_mean": float(np.mean(clip_sel)),
                "clip_std": float(np.std(clip_sel)),
                "ir_mean": float(np.mean(ir_sel)),
                "ir_std": float(np.std(ir_sel)),
                "n": N,
            }

    # Save results
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDrawBench results saved to {OUT_DIR / 'results.json'}")

    # Print summary
    print("\nDrawBench Summary (seed-averaged):")
    for method in ["ddim_50", "random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk"]:
        clips = [all_results[f"{method}_drawbench_seed{s}"]["clip_mean"] for s in SEEDS]
        irs = [all_results[f"{method}_drawbench_seed{s}"]["ir_mean"] for s in SEEDS]
        print(f"  {method:20s}: CLIP={np.mean(clips):.4f}±{np.std(clips):.4f}  IR={np.mean(irs):.4f}±{np.std(irs):.4f}")

if __name__ == "__main__":
    main()
