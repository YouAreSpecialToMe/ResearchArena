"""
Main experiment runner for CoPS (Coherent Particle Sampling).

Runs all experiments in a single efficient pipeline:
1. Standard baselines (DDIM, DPM-Solver, Euler)
2. Shared particle generation (K=4) with all selection methods
3. CoPS with active resampling
4. Ablation studies
5. Scaling experiments
6. Metric computation
"""
import torch
import json
import os
import sys
import time
import gc
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = Path(__file__).parent.parent
EXP_DIR = WORKSPACE / "exp"
DATA_DIR = EXP_DIR / "data"
FIGURES_DIR = WORKSPACE / "figures"

SEEDS = [42, 123, 456]
K_PARTICLES = 4
NUM_STEPS = 50
CFG_SCALE = 7.5
IMAGE_SIZE = 512

def load_prompts(dataset_name):
    """Load prompts for a given dataset."""
    files = {
        "coco": DATA_DIR / "coco_500_prompts.json",
        "parti": DATA_DIR / "parti_200_prompts.json",
        "drawbench": DATA_DIR / "drawbench_prompts.json",
    }
    with open(files[dataset_name]) as f:
        data = json.load(f)
    return [item["prompt"] for item in data]


def load_pipeline(model_name="sd15"):
    """Load diffusion model pipeline."""
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler

    if model_name == "sd15":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def set_scheduler(pipe, scheduler_name="ddim"):
    """Set the scheduler for the pipeline."""
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler

    if scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe


# ============================================================================
# EXPERIMENT 1: Standard Baselines
# ============================================================================
def run_standard_baselines(pipe, prompts, dataset_name, output_dir):
    """Run standard single-trajectory baselines."""
    configs = [
        {"name": "ddim_50", "scheduler": "ddim", "steps": 50},
        {"name": "dpm_20", "scheduler": "dpm", "steps": 20},
        {"name": "euler_50", "scheduler": "euler", "steps": 50},
    ]

    results = {}
    for config in configs:
        method_name = config["name"]
        print(f"\n  Running {method_name} on {dataset_name}...")

        for seed in SEEDS:
            seed_dir = output_dir / f"{method_name}_seed{seed}" / dataset_name
            seed_dir.mkdir(parents=True, exist_ok=True)

            set_scheduler(pipe, config["scheduler"])
            generator = torch.Generator("cuda").manual_seed(seed)
            times = []

            for i, prompt in enumerate(tqdm(prompts, desc=f"  {method_name} s{seed}")):
                t0 = time.time()
                with torch.no_grad():
                    output = pipe(
                        prompt, num_inference_steps=config["steps"],
                        guidance_scale=CFG_SCALE, generator=generator,
                        height=IMAGE_SIZE, width=IMAGE_SIZE,
                    )
                img = output.images[0]
                img.save(seed_dir / f"{i:05d}.png")
                times.append(time.time() - t0)
                generator = torch.Generator("cuda").manual_seed(seed + i + 1)

            key = f"{method_name}_{dataset_name}_seed{seed}"
            results[key] = {
                "method": method_name, "dataset": dataset_name, "seed": seed,
                "avg_time": np.mean(times), "total_time": sum(times),
                "num_images": len(prompts),
            }
            print(f"    Seed {seed}: avg {np.mean(times):.2f}s/img, total {sum(times):.0f}s")

    return results


# ============================================================================
# EXPERIMENT 2: Shared Particle Generation + Selection Methods
# ============================================================================
def run_particle_experiments(pipe, prompts, dataset_name, output_dir, clip_scorer, ir_scorer):
    """Generate K particles per prompt, apply all selection methods."""
    from exp.shared.cops import generate_particles_batch, cops_sample_with_resampling
    from exp.shared.metrics import tensor_to_pil, save_images

    set_scheduler(pipe, "ddim")

    results = {}
    all_pcs_data = []  # For correlation analysis

    for seed in SEEDS:
        print(f"\n  Particle generation seed={seed} on {dataset_name}...")

        # Output dirs for each selection method
        methods = ["random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk", "cops_resample"]
        method_dirs = {}
        for m in methods:
            d = output_dir / f"{m}_seed{seed}" / dataset_name
            d.mkdir(parents=True, exist_ok=True)
            method_dirs[m] = d

        method_scores = {m: {"clip": [], "ir": [], "pcs": []} for m in methods}
        gen_times = []

        for i, prompt in enumerate(tqdm(prompts, desc=f"  Particles s{seed}")):
            t0 = time.time()

            # --- Generate K particles WITHOUT resampling ---
            particle_result = generate_particles_batch(
                pipe, prompt, num_particles=K_PARTICLES,
                num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
                seed=seed + i, height=IMAGE_SIZE, width=IMAGE_SIZE,
                track_pcs=True, distance_metric="l2",
            )
            images = particle_result["images"]  # (K, 3, H, W)
            pcs = np.array(particle_result["pcs_scores"])
            gen_times.append(time.time() - t0)

            # Convert to PIL for scoring
            pil_images = [tensor_to_pil(img) for img in images]

            # Score all particles
            clip_scores = clip_scorer.score_batch(pil_images, [prompt] * K_PARTICLES)
            ir_scores = ir_scorer.score_batch(pil_images, [prompt] * K_PARTICLES)

            # Store for correlation analysis (seed=42 only)
            if seed == SEEDS[0]:
                all_pcs_data.append({
                    "prompt_idx": i, "prompt": prompt,
                    "pcs_scores": pcs.tolist(),
                    "clip_scores": clip_scores,
                    "ir_scores": ir_scores,
                    "pcs_trajectories": particle_result.get("pcs_trajectories", []),
                    "step_coherences": particle_result.get("step_coherences", []),
                })

            # Selection methods
            rng = np.random.RandomState(seed + i)
            selections = {
                "random_k": rng.randint(K_PARTICLES),
                "pcs_bestofk": int(np.argmax(pcs)),
                "clip_bestofk": int(np.argmax(clip_scores)),
                "ir_bestofk": int(np.argmax(ir_scores)),
            }

            for method, idx in selections.items():
                pil_images[idx].save(method_dirs[method] / f"{i:05d}.png")
                method_scores[method]["clip"].append(clip_scores[idx])
                method_scores[method]["ir"].append(ir_scores[idx])
                method_scores[method]["pcs"].append(pcs[idx])

            # --- CoPS with active resampling ---
            t0_cops = time.time()
            cops_result = cops_sample_with_resampling(
                pipe, prompt, num_particles=K_PARTICLES,
                num_inference_steps=NUM_STEPS, resample_interval=10,
                alpha=1.0, sigma_jitter=0.01, distance_metric="l2",
                guidance_scale=CFG_SCALE, seed=seed + i,
                height=IMAGE_SIZE, width=IMAGE_SIZE,
            )
            cops_time = time.time() - t0_cops

            cops_img = tensor_to_pil(cops_result["image"])
            cops_img.save(method_dirs["cops_resample"] / f"{i:05d}.png")
            cops_clip = clip_scorer.score_single(cops_img, prompt)
            cops_ir = ir_scorer.score_single(cops_img, prompt)
            method_scores["cops_resample"]["clip"].append(cops_clip)
            method_scores["cops_resample"]["ir"].append(cops_ir)
            method_scores["cops_resample"]["pcs"].append(float(cops_result["pcs_scores"][cops_result["selected_index"]]))

        # Aggregate results for this seed
        for method in methods:
            key = f"{method}_{dataset_name}_seed{seed}"
            results[key] = {
                "method": method, "dataset": dataset_name, "seed": seed,
                "clip_score_mean": float(np.mean(method_scores[method]["clip"])),
                "clip_score_std": float(np.std(method_scores[method]["clip"])),
                "ir_score_mean": float(np.mean(method_scores[method]["ir"])),
                "ir_score_std": float(np.std(method_scores[method]["ir"])),
                "pcs_score_mean": float(np.mean(method_scores[method]["pcs"])),
                "num_images": len(prompts),
            }

        print(f"    Gen time: avg {np.mean(gen_times):.2f}s/prompt")
        for m in methods:
            cs = np.mean(method_scores[m]["clip"])
            ir = np.mean(method_scores[m]["ir"])
            print(f"    {m}: CLIP={cs:.4f}, IR={ir:.4f}")

    return results, all_pcs_data


# ============================================================================
# EXPERIMENT 3: Ablation Studies
# ============================================================================
def run_ablations(pipe, prompts_200, dataset_name, output_dir, clip_scorer, ir_scorer):
    """Run ablation studies on a 200-prompt subset."""
    from exp.shared.cops import cops_sample_with_resampling, generate_particles_batch
    from exp.shared.metrics import tensor_to_pil

    set_scheduler(pipe, "ddim")
    seed = 42
    results = {}

    # --- A2: Resampling frequency ---
    print("\n  Ablation A2: Resampling frequency...")
    for R in [1, 5, 10, 25, 50]:
        method_name = f"cops_R{R}"
        out_dir = output_dir / f"ablation_resample_R{R}" / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        clip_scores, ir_scores = [], []
        for i, prompt in enumerate(tqdm(prompts_200[:200], desc=f"  R={R}")):
            if R == 50:
                # No resampling = best-of-K by PCS
                res = generate_particles_batch(
                    pipe, prompt, num_particles=K_PARTICLES,
                    num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
                    seed=seed + i, height=IMAGE_SIZE, width=IMAGE_SIZE,
                )
                best_idx = int(np.argmax(res["pcs_scores"]))
                img = tensor_to_pil(res["images"][best_idx])
            else:
                res = cops_sample_with_resampling(
                    pipe, prompt, num_particles=K_PARTICLES,
                    num_inference_steps=NUM_STEPS, resample_interval=R,
                    alpha=1.0, sigma_jitter=0.01, distance_metric="l2",
                    guidance_scale=CFG_SCALE, seed=seed + i,
                    height=IMAGE_SIZE, width=IMAGE_SIZE,
                )
                img = tensor_to_pil(res["image"])
            img.save(out_dir / f"{i:05d}.png")
            clip_scores.append(clip_scorer.score_single(img, prompt))
            ir_scores.append(ir_scorer.score_single(img, prompt))

        results[f"ablation_R{R}"] = {
            "resample_interval": R,
            "clip_score_mean": float(np.mean(clip_scores)),
            "clip_score_std": float(np.std(clip_scores)),
            "ir_score_mean": float(np.mean(ir_scores)),
            "ir_score_std": float(np.std(ir_scores)),
        }
        print(f"    R={R}: CLIP={np.mean(clip_scores):.4f}, IR={np.mean(ir_scores):.4f}")

    # --- A3: Timestep weighting (post-hoc, no new generation needed) ---
    print("\n  Ablation A3: Timestep weighting (post-hoc)...")
    from exp.shared.cops import PCSTracker
    for weight_scheme in ["uniform", "mid_emphasis", "early_emphasis", "late_emphasis"]:
        clip_scores, ir_scores = [], []
        for i, prompt in enumerate(tqdm(prompts_200[:200], desc=f"  {weight_scheme}")):
            res = generate_particles_batch(
                pipe, prompt, num_particles=K_PARTICLES,
                num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
                seed=seed + i, height=IMAGE_SIZE, width=IMAGE_SIZE,
                track_pcs=True, distance_metric="l2",
            )

            # Recompute PCS with different weights
            step_coherences = res.get("step_coherences", [])
            if step_coherences:
                tracker = PCSTracker(
                    num_particles=K_PARTICLES,
                    distance_metric="l2",
                    timestep_weights=weight_scheme,
                    total_steps=NUM_STEPS,
                )
                pcs = np.zeros(K_PARTICLES)
                for step_idx, coh in enumerate(step_coherences):
                    w = tracker._get_weight(step_idx + 1)  # +1 because first step has no coherence
                    pcs += np.array(coh) * w / (1.0 + 1e-8)
                best_idx = int(np.argmax(pcs))
            else:
                best_idx = int(np.argmax(res["pcs_scores"]))

            img = tensor_to_pil(res["images"][best_idx])
            clip_scores.append(clip_scorer.score_single(img, prompt))
            ir_scores.append(ir_scorer.score_single(img, prompt))

        results[f"ablation_weight_{weight_scheme}"] = {
            "weight_scheme": weight_scheme,
            "clip_score_mean": float(np.mean(clip_scores)),
            "clip_score_std": float(np.std(clip_scores)),
            "ir_score_mean": float(np.mean(ir_scores)),
            "ir_score_std": float(np.std(ir_scores)),
        }
        print(f"    {weight_scheme}: CLIP={np.mean(clip_scores):.4f}, IR={np.mean(ir_scores):.4f}")

    # --- A5: CFG combination ---
    print("\n  Ablation A5: CFG scale combination...")
    for cfg in [3.0, 7.5, 12.0, 20.0]:
        for use_cops in [False, True]:
            label = f"cfg{cfg}_cops" if use_cops else f"cfg{cfg}_standard"
            clip_scores, ir_scores = [], []

            for i, prompt in enumerate(tqdm(prompts_200[:100], desc=f"  {label}")):
                if use_cops:
                    res = cops_sample_with_resampling(
                        pipe, prompt, num_particles=K_PARTICLES,
                        num_inference_steps=NUM_STEPS, resample_interval=10,
                        alpha=1.0, sigma_jitter=0.01, distance_metric="l2",
                        guidance_scale=cfg, seed=seed + i,
                        height=IMAGE_SIZE, width=IMAGE_SIZE,
                    )
                    img = tensor_to_pil(res["image"])
                else:
                    generator = torch.Generator("cuda").manual_seed(seed + i)
                    output = pipe(
                        prompt, num_inference_steps=NUM_STEPS,
                        guidance_scale=cfg, generator=generator,
                        height=IMAGE_SIZE, width=IMAGE_SIZE,
                    )
                    img = output.images[0]
                clip_scores.append(clip_scorer.score_single(img, prompt))
                ir_scores.append(ir_scorer.score_single(img, prompt))

            results[f"ablation_{label}"] = {
                "cfg_scale": cfg, "use_cops": use_cops,
                "clip_score_mean": float(np.mean(clip_scores)),
                "clip_score_std": float(np.std(clip_scores)),
                "ir_score_mean": float(np.mean(ir_scores)),
                "ir_score_std": float(np.std(ir_scores)),
            }
            print(f"    {label}: CLIP={np.mean(clip_scores):.4f}, IR={np.mean(ir_scores):.4f}")

    return results


# ============================================================================
# EXPERIMENT 4: Scaling Behavior
# ============================================================================
def run_scaling_experiment(pipe, prompts, dataset_name, output_dir, clip_scorer, ir_scorer):
    """Test quality scaling with number of particles K."""
    from exp.shared.cops import generate_particles_batch
    from exp.shared.metrics import tensor_to_pil

    set_scheduler(pipe, "ddim")
    seed = 42
    total_nfe = 200  # Fixed total NFE budget
    results = {}

    # Use 200 prompts for scaling
    prompts_sub = prompts[:200]

    for K in [1, 2, 4, 8]:
        steps = total_nfe // K
        if steps < 5:
            continue

        print(f"\n  Scaling K={K}, steps={steps}...")
        method_scores = {"random": {"clip": [], "ir": []},
                         "pcs": {"clip": [], "ir": []},
                         "clip_sel": {"clip": [], "ir": []}}

        for i, prompt in enumerate(tqdm(prompts_sub, desc=f"  K={K}")):
            if K == 1:
                generator = torch.Generator("cuda").manual_seed(seed + i)
                output = pipe(
                    prompt, num_inference_steps=steps,
                    guidance_scale=CFG_SCALE, generator=generator,
                    height=IMAGE_SIZE, width=IMAGE_SIZE,
                )
                img = output.images[0]
                cs = clip_scorer.score_single(img, prompt)
                ir = ir_scorer.score_single(img, prompt)
                for m in method_scores:
                    method_scores[m]["clip"].append(cs)
                    method_scores[m]["ir"].append(ir)
            else:
                res = generate_particles_batch(
                    pipe, prompt, num_particles=K,
                    num_inference_steps=steps, guidance_scale=CFG_SCALE,
                    seed=seed + i, height=IMAGE_SIZE, width=IMAGE_SIZE,
                )
                pil_images = [tensor_to_pil(img) for img in res["images"]]
                clip_scores = clip_scorer.score_batch(pil_images, [prompt] * K)
                ir_scores_list = ir_scorer.score_batch(pil_images, [prompt] * K)
                pcs = np.array(res["pcs_scores"])

                # Random selection
                rng = np.random.RandomState(seed + i)
                r_idx = rng.randint(K)
                method_scores["random"]["clip"].append(clip_scores[r_idx])
                method_scores["random"]["ir"].append(ir_scores_list[r_idx])

                # PCS selection
                p_idx = int(np.argmax(pcs))
                method_scores["pcs"]["clip"].append(clip_scores[p_idx])
                method_scores["pcs"]["ir"].append(ir_scores_list[p_idx])

                # CLIP selection
                c_idx = int(np.argmax(clip_scores))
                method_scores["clip_sel"]["clip"].append(clip_scores[c_idx])
                method_scores["clip_sel"]["ir"].append(ir_scores_list[c_idx])

        for m in method_scores:
            key = f"scaling_K{K}_{m}"
            results[key] = {
                "K": K, "steps": steps, "total_nfe": K * steps,
                "selection": m,
                "clip_score_mean": float(np.mean(method_scores[m]["clip"])),
                "clip_score_std": float(np.std(method_scores[m]["clip"])),
                "ir_score_mean": float(np.mean(method_scores[m]["ir"])),
                "ir_score_std": float(np.std(method_scores[m]["ir"])),
            }

        print(f"    K={K}: PCS CLIP={np.mean(method_scores['pcs']['clip']):.4f}, "
              f"Random CLIP={np.mean(method_scores['random']['clip']):.4f}")

    return results


# ============================================================================
# EXPERIMENT 5: Distance Metric Ablation
# ============================================================================
def run_distance_metric_ablation(pipe, prompts, dataset_name, output_dir, clip_scorer, ir_scorer):
    """Compare L2 vs cosine distance for PCS computation."""
    from exp.shared.cops import generate_particles_batch
    from exp.shared.metrics import tensor_to_pil

    set_scheduler(pipe, "ddim")
    seed = 42
    results = {}
    prompts_sub = prompts[:200]

    for metric in ["l2", "cosine"]:
        print(f"\n  Distance metric: {metric}...")
        clip_scores, ir_scores = [], []

        for i, prompt in enumerate(tqdm(prompts_sub, desc=f"  {metric}")):
            res = generate_particles_batch(
                pipe, prompt, num_particles=K_PARTICLES,
                num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
                seed=seed + i, height=IMAGE_SIZE, width=IMAGE_SIZE,
                track_pcs=True, distance_metric=metric,
            )
            best_idx = int(np.argmax(res["pcs_scores"]))
            img = tensor_to_pil(res["images"][best_idx])
            clip_scores.append(clip_scorer.score_single(img, prompt))
            ir_scores.append(ir_scorer.score_single(img, prompt))

        results[f"distance_{metric}"] = {
            "metric": metric,
            "clip_score_mean": float(np.mean(clip_scores)),
            "clip_score_std": float(np.std(clip_scores)),
            "ir_score_mean": float(np.mean(ir_scores)),
            "ir_score_std": float(np.std(ir_scores)),
        }
        print(f"    {metric}: CLIP={np.mean(clip_scores):.4f}, IR={np.mean(ir_scores):.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    t_start = time.time()
    all_results = {}

    print("=" * 70)
    print("CoPS Experiment Pipeline")
    print("=" * 70)

    # Load pipeline
    print("\n[1/7] Loading SD1.5 pipeline...")
    pipe = load_pipeline("sd15")

    # Load prompts
    coco_prompts = load_prompts("coco")
    parti_prompts = load_prompts("parti")
    drawbench_prompts = load_prompts("drawbench")

    # Initialize scorers
    print("\n[2/7] Loading evaluation models...")
    from exp.shared.metrics import CLIPScorer, ImageRewardScorer
    clip_scorer = CLIPScorer(device="cuda")
    ir_scorer = ImageRewardScorer(device="cuda")

    output_dir = EXP_DIR

    # --- Standard Baselines ---
    print("\n" + "=" * 70)
    print("[3/7] Running standard baselines...")
    print("=" * 70)

    # Run baselines on COCO (primary) and Parti
    baseline_results = {}
    for ds_name, prompts in [("coco", coco_prompts), ("parti", parti_prompts)]:
        br = run_standard_baselines(pipe, prompts, ds_name, output_dir / "baselines")
        baseline_results.update(br)
    all_results["standard_baselines"] = baseline_results

    # Save checkpoint
    _save_checkpoint(all_results, "after_baselines")
    elapsed = time.time() - t_start
    print(f"\n  Time so far: {elapsed/60:.1f} min")

    # --- Particle Experiments ---
    print("\n" + "=" * 70)
    print("[4/7] Running particle experiments (shared generation)...")
    print("=" * 70)

    particle_results = {}
    all_pcs_data = {}
    for ds_name, prompts in [("coco", coco_prompts), ("parti", parti_prompts)]:
        pr, pcs_data = run_particle_experiments(
            pipe, prompts, ds_name, output_dir / "main", clip_scorer, ir_scorer
        )
        particle_results.update(pr)
        all_pcs_data[ds_name] = pcs_data
    all_results["particle_methods"] = particle_results

    # Save PCS correlation data
    with open(EXP_DIR / "analysis" / "pcs_correlation_data.json", "w") as f:
        os.makedirs(EXP_DIR / "analysis", exist_ok=True)
        json.dump(all_pcs_data, f)

    _save_checkpoint(all_results, "after_particles")
    elapsed = time.time() - t_start
    print(f"\n  Time so far: {elapsed/60:.1f} min")

    # Check time budget
    if elapsed > 6 * 3600:
        print("\n  WARNING: Over 6 hours used. Skipping some ablations.")
        skip_some = True
    else:
        skip_some = False

    # --- Scaling Experiment ---
    print("\n" + "=" * 70)
    print("[5/7] Running scaling experiment...")
    print("=" * 70)
    scaling_results = run_scaling_experiment(
        pipe, coco_prompts, "coco", output_dir / "main", clip_scorer, ir_scorer
    )
    all_results["scaling"] = scaling_results
    _save_checkpoint(all_results, "after_scaling")
    elapsed = time.time() - t_start
    print(f"\n  Time so far: {elapsed/60:.1f} min")

    # --- Ablations ---
    print("\n" + "=" * 70)
    print("[6/7] Running ablation studies...")
    print("=" * 70)

    ablation_prompts = coco_prompts[:200]

    # Distance metric ablation
    dist_results = run_distance_metric_ablation(
        pipe, coco_prompts, "coco", output_dir / "ablations", clip_scorer, ir_scorer
    )
    all_results["ablation_distance"] = dist_results

    if not skip_some:
        # Full ablations
        abl_results = run_ablations(
            pipe, ablation_prompts, "coco", output_dir / "ablations", clip_scorer, ir_scorer
        )
        all_results["ablations"] = abl_results
    else:
        # Reduced ablations
        print("  Running reduced ablation set due to time constraints...")
        abl_results = run_ablations(
            pipe, coco_prompts[:100], "coco", output_dir / "ablations", clip_scorer, ir_scorer
        )
        all_results["ablations"] = abl_results

    _save_checkpoint(all_results, "after_ablations")
    elapsed = time.time() - t_start
    print(f"\n  Time so far: {elapsed/60:.1f} min")

    # --- Compute FID ---
    print("\n" + "=" * 70)
    print("[7/7] Computing FID scores...")
    print("=" * 70)
    fid_results = compute_fid_scores(output_dir)
    all_results["fid_scores"] = fid_results

    # --- Save final results ---
    total_time = time.time() - t_start
    all_results["total_runtime_seconds"] = total_time
    all_results["total_runtime_hours"] = total_time / 3600

    results_path = WORKSPACE / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to {results_path}")
    print(f"Total runtime: {total_time/3600:.2f} hours")

    return all_results


def compute_fid_scores(output_dir):
    """Compute FID for key methods against COCO real images."""
    # First, we need real COCO images for FID reference
    # Download a small set of real COCO validation images
    real_dir = EXP_DIR / "data" / "coco_real_images"
    if not real_dir.exists():
        print("  Downloading COCO real images for FID...")
        download_coco_real_images(real_dir)

    if not real_dir.exists() or len(list(real_dir.glob("*.png"))) == 0:
        print("  WARNING: No real images for FID. Skipping FID computation.")
        return {}

    from exp.shared.metrics import compute_fid
    fid_results = {}

    # Compute FID for key methods
    methods_to_eval = [
        ("ddim_50_seed42", "baselines/ddim_50_seed42/coco"),
        ("random_k_seed42", "main/random_k_seed42/coco"),
        ("pcs_bestofk_seed42", "main/pcs_bestofk_seed42/coco"),
        ("clip_bestofk_seed42", "main/clip_bestofk_seed42/coco"),
        ("ir_bestofk_seed42", "main/ir_bestofk_seed42/coco"),
        ("cops_resample_seed42", "main/cops_resample_seed42/coco"),
    ]

    for name, rel_path in methods_to_eval:
        gen_dir = output_dir / rel_path
        if gen_dir.exists() and len(list(gen_dir.glob("*.png"))) > 0:
            try:
                fid_score = compute_fid(str(real_dir), str(gen_dir))
                fid_results[name] = fid_score
                print(f"    FID {name}: {fid_score:.2f}")
            except Exception as e:
                print(f"    FID {name}: FAILED ({e})")
                fid_results[name] = None

    return fid_results


def download_coco_real_images(real_dir, n=500, seed=42):
    """Download real COCO validation images for FID computation."""
    real_dir.mkdir(parents=True, exist_ok=True)

    # Load the prompt file to get image IDs
    prompt_file = DATA_DIR / "coco_500_prompts.json"
    with open(prompt_file) as f:
        prompts = json.load(f)

    import urllib.request
    from PIL import Image as PILImage
    import io

    count = 0
    for item in tqdm(prompts[:n], desc="  Downloading COCO images"):
        img_id = item["id"]
        url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            img = PILImage.open(io.BytesIO(resp.read())).convert("RGB")
            img = img.resize((512, 512), PILImage.LANCZOS)
            img.save(real_dir / f"{count:05d}.png")
            count += 1
        except Exception as e:
            continue

    print(f"  Downloaded {count} real COCO images")


def _save_checkpoint(results, name):
    """Save intermediate checkpoint."""
    checkpoint_dir = EXP_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_dir / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
