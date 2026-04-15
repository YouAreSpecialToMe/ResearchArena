"""
Optimized experiment runner for CoPS.
Designed for 1x A6000, ~7 hour budget.

Strategy:
- COCO-500 is primary benchmark (3 seeds)
- PartiPrompts-200 secondary (3 seeds)
- DrawBench as tertiary (1 seed)
- Shared particle generation across selection methods
- Ablations on 200-prompt COCO subset (1 seed)
- Scaling on 200-prompt subset (1 seed)
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
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = Path(".")
EXP_DIR = WORKSPACE / "exp"
DATA_DIR = EXP_DIR / "data"
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456]
K_PARTICLES = 4
NUM_STEPS = 50
CFG_SCALE = 7.5
IMAGE_SIZE = 512

def load_prompts(name):
    files = {"coco": "coco_500_prompts.json", "parti": "parti_200_prompts.json", "drawbench": "drawbench_prompts.json"}
    with open(DATA_DIR / files[name]) as f:
        return [item["prompt"] for item in json.load(f)]

def load_pipe():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def set_scheduler(pipe, name):
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
    scheds = {"ddim": DDIMScheduler, "dpm": DPMSolverMultistepScheduler, "euler": EulerDiscreteScheduler}
    pipe.scheduler = scheds[name].from_config(pipe.scheduler.config)

def tensor_to_pil(t):
    arr = (t.cpu().float().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)

def save_checkpoint(results, name):
    d = EXP_DIR / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2)

# ============================================================================
def run_standard_baseline(pipe, prompts, ds_name, method, steps, seed, base_dir):
    """Run a single standard baseline config."""
    set_scheduler(pipe, method)
    out_dir = base_dir / f"{method}_{steps}_seed{seed}" / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_scores, ir_scores, times = [], [], []
    for i, prompt in enumerate(tqdm(prompts, desc=f"  {method}_{steps} s{seed} {ds_name}", leave=False)):
        gen = torch.Generator("cuda").manual_seed(seed + i)
        t0 = time.time()
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=steps, guidance_scale=CFG_SCALE,
                       generator=gen, height=IMAGE_SIZE, width=IMAGE_SIZE).images[0]
        times.append(time.time() - t0)
        img.save(out_dir / f"{i:05d}.png")
    return {"method": f"{method}_{steps}", "seed": seed, "dataset": ds_name,
            "avg_time_per_img": float(np.mean(times)), "num_images": len(prompts)}


def run_particle_generation(pipe, prompts, ds_name, seed, base_dir, clip_scorer, ir_scorer):
    """Generate K particles per prompt without resampling. Apply all selection methods."""
    from exp.shared.cops import generate_particles_batch, cops_sample_with_resampling

    set_scheduler(pipe, "ddim")
    methods = ["random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk", "cops_resample"]
    dirs = {}
    for m in methods:
        d = base_dir / f"{m}_seed{seed}" / ds_name
        d.mkdir(parents=True, exist_ok=True)
        dirs[m] = d

    scores = {m: {"clip": [], "ir": []} for m in methods}
    pcs_correlation_data = []

    for i, prompt in enumerate(tqdm(prompts, desc=f"  Particles s{seed} {ds_name}", leave=False)):
        # Generate K particles (no resampling)
        res = generate_particles_batch(
            pipe, prompt, num_particles=K_PARTICLES,
            num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
            seed=seed + i, height=IMAGE_SIZE, width=IMAGE_SIZE,
            track_pcs=True, distance_metric="l2",
        )
        images = res["images"]
        pcs = np.array(res["pcs_scores"])
        pil_imgs = [tensor_to_pil(img) for img in images]

        cs = clip_scorer.score_batch(pil_imgs, [prompt] * K_PARTICLES)
        irs = ir_scorer.score_batch(pil_imgs, [prompt] * K_PARTICLES)

        # Store correlation data for seed=42
        if seed == SEEDS[0]:
            pcs_correlation_data.append({
                "idx": i, "prompt": prompt,
                "pcs": pcs.tolist(), "clip": cs, "ir": irs,
                "step_coherences": res.get("step_coherences", []),
            })

        # Apply selection methods
        rng = np.random.RandomState(seed + i)
        sels = {
            "random_k": rng.randint(K_PARTICLES),
            "pcs_bestofk": int(np.argmax(pcs)),
            "clip_bestofk": int(np.argmax(cs)),
            "ir_bestofk": int(np.argmax(irs)),
        }
        for m, idx in sels.items():
            pil_imgs[idx].save(dirs[m] / f"{i:05d}.png")
            scores[m]["clip"].append(cs[idx])
            scores[m]["ir"].append(irs[idx])

        # CoPS with active resampling
        cops_res = cops_sample_with_resampling(
            pipe, prompt, num_particles=K_PARTICLES,
            num_inference_steps=NUM_STEPS, resample_interval=10,
            alpha=1.0, sigma_jitter=0.01, distance_metric="l2",
            guidance_scale=CFG_SCALE, seed=seed + i,
        )
        cops_img = tensor_to_pil(cops_res["image"])
        cops_img.save(dirs["cops_resample"] / f"{i:05d}.png")
        cops_cs = clip_scorer.score_single(cops_img, prompt)
        cops_ir = ir_scorer.score_single(cops_img, prompt)
        scores["cops_resample"]["clip"].append(cops_cs)
        scores["cops_resample"]["ir"].append(cops_ir)

    results = {}
    for m in methods:
        results[f"{m}_{ds_name}_seed{seed}"] = {
            "method": m, "dataset": ds_name, "seed": seed,
            "clip_mean": float(np.mean(scores[m]["clip"])),
            "clip_std": float(np.std(scores[m]["clip"])),
            "ir_mean": float(np.mean(scores[m]["ir"])),
            "ir_std": float(np.std(scores[m]["ir"])),
            "n": len(prompts),
        }

    return results, pcs_correlation_data


def run_scaling(pipe, prompts, clip_scorer, ir_scorer):
    """Scaling experiment: vary K with constant total NFE=200."""
    from exp.shared.cops import generate_particles_batch
    seed = 42
    total_nfe = 200
    results = {}

    for K in [1, 2, 4, 8]:
        steps = total_nfe // K
        print(f"  Scaling K={K}, steps={steps}...")
        method_scores = {m: {"clip": [], "ir": []} for m in ["random", "pcs", "clip_sel"]}

        set_scheduler(pipe, "ddim")
        for i, prompt in enumerate(tqdm(prompts, desc=f"  K={K}", leave=False)):
            if K == 1:
                gen = torch.Generator("cuda").manual_seed(seed + i)
                with torch.no_grad():
                    img = pipe(prompt, num_inference_steps=steps, guidance_scale=CFG_SCALE,
                               generator=gen, height=IMAGE_SIZE, width=IMAGE_SIZE).images[0]
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
                pil_imgs = [tensor_to_pil(img) for img in res["images"]]
                cs_all = clip_scorer.score_batch(pil_imgs, [prompt] * K)
                ir_all = ir_scorer.score_batch(pil_imgs, [prompt] * K)
                pcs = np.array(res["pcs_scores"])

                rng = np.random.RandomState(seed + i)
                sels = {"random": rng.randint(K), "pcs": int(np.argmax(pcs)), "clip_sel": int(np.argmax(cs_all))}
                for m, idx in sels.items():
                    method_scores[m]["clip"].append(cs_all[idx])
                    method_scores[m]["ir"].append(ir_all[idx])

        for m in method_scores:
            results[f"K{K}_{m}"] = {
                "K": K, "steps": steps, "selection": m,
                "clip_mean": float(np.mean(method_scores[m]["clip"])),
                "clip_std": float(np.std(method_scores[m]["clip"])),
                "ir_mean": float(np.mean(method_scores[m]["ir"])),
                "ir_std": float(np.std(method_scores[m]["ir"])),
            }
        print(f"    PCS: CLIP={np.mean(method_scores['pcs']['clip']):.4f}, "
              f"Random: CLIP={np.mean(method_scores['random']['clip']):.4f}")

    return results


def run_ablations(pipe, prompts, clip_scorer, ir_scorer):
    """Run ablation studies on COCO-200 subset."""
    from exp.shared.cops import cops_sample_with_resampling, generate_particles_batch, PCSTracker
    seed = 42
    results = {}

    # A1: Distance metric
    print("  A1: Distance metric ablation...")
    for metric in ["l2", "cosine"]:
        cs_list, ir_list = [], []
        for i, prompt in enumerate(tqdm(prompts, desc=f"  dist={metric}", leave=False)):
            res = generate_particles_batch(
                pipe, prompt, num_particles=K_PARTICLES,
                num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
                seed=seed + i, distance_metric=metric,
            )
            best_idx = int(np.argmax(res["pcs_scores"]))
            img = tensor_to_pil(res["images"][best_idx])
            cs_list.append(clip_scorer.score_single(img, prompt))
            ir_list.append(ir_scorer.score_single(img, prompt))
        results[f"dist_{metric}"] = {
            "metric": metric,
            "clip_mean": float(np.mean(cs_list)), "clip_std": float(np.std(cs_list)),
            "ir_mean": float(np.mean(ir_list)), "ir_std": float(np.std(ir_list)),
        }
        print(f"    {metric}: CLIP={np.mean(cs_list):.4f}, IR={np.mean(ir_list):.4f}")

    # A2: Resampling frequency
    print("  A2: Resampling frequency ablation...")
    for R in [1, 5, 10, 25, 50]:
        cs_list, ir_list = [], []
        for i, prompt in enumerate(tqdm(prompts, desc=f"  R={R}", leave=False)):
            if R >= NUM_STEPS:
                # No resampling = best-of-K by PCS
                res = generate_particles_batch(
                    pipe, prompt, num_particles=K_PARTICLES,
                    num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
                    seed=seed + i,
                )
                best_idx = int(np.argmax(res["pcs_scores"]))
                img = tensor_to_pil(res["images"][best_idx])
            else:
                res = cops_sample_with_resampling(
                    pipe, prompt, num_particles=K_PARTICLES,
                    num_inference_steps=NUM_STEPS, resample_interval=R,
                    alpha=1.0, sigma_jitter=0.01, guidance_scale=CFG_SCALE,
                    seed=seed + i,
                )
                img = tensor_to_pil(res["image"])
            cs_list.append(clip_scorer.score_single(img, prompt))
            ir_list.append(ir_scorer.score_single(img, prompt))
        results[f"resample_R{R}"] = {
            "R": R,
            "clip_mean": float(np.mean(cs_list)), "clip_std": float(np.std(cs_list)),
            "ir_mean": float(np.mean(ir_list)), "ir_std": float(np.std(ir_list)),
        }
        print(f"    R={R}: CLIP={np.mean(cs_list):.4f}, IR={np.mean(ir_list):.4f}")

    # A3: Timestep weighting (post-hoc recomputation)
    print("  A3: Timestep weighting ablation...")
    # First generate particles with coherence tracking
    all_data = []
    for i, prompt in enumerate(tqdm(prompts, desc="  gen particles", leave=False)):
        res = generate_particles_batch(
            pipe, prompt, num_particles=K_PARTICLES,
            num_inference_steps=NUM_STEPS, guidance_scale=CFG_SCALE,
            seed=seed + i, track_pcs=True,
        )
        pil_imgs = [tensor_to_pil(img) for img in res["images"]]
        cs = clip_scorer.score_batch(pil_imgs, [prompt] * K_PARTICLES)
        ir = ir_scorer.score_batch(pil_imgs, [prompt] * K_PARTICLES)
        all_data.append({"images": res["images"], "step_coherences": res.get("step_coherences", []),
                         "clip": cs, "ir": ir, "pcs": res["pcs_scores"]})

    for weight_scheme in ["uniform", "mid_emphasis", "early_emphasis", "late_emphasis"]:
        cs_list, ir_list = [], []
        for i, data in enumerate(all_data):
            step_cohs = data["step_coherences"]
            if step_cohs:
                tracker = PCSTracker(num_particles=K_PARTICLES, timestep_weights=weight_scheme, total_steps=NUM_STEPS)
                pcs = np.zeros(K_PARTICLES)
                for si, coh in enumerate(step_cohs):
                    w = tracker._get_weight(si + 1)
                    pcs += np.array(coh) * w
                best_idx = int(np.argmax(pcs))
            else:
                best_idx = int(np.argmax(data["pcs"]))
            cs_list.append(data["clip"][best_idx])
            ir_list.append(data["ir"][best_idx])
        results[f"weight_{weight_scheme}"] = {
            "weight": weight_scheme,
            "clip_mean": float(np.mean(cs_list)), "clip_std": float(np.std(cs_list)),
            "ir_mean": float(np.mean(ir_list)), "ir_std": float(np.std(ir_list)),
        }
        print(f"    {weight_scheme}: CLIP={np.mean(cs_list):.4f}, IR={np.mean(ir_list):.4f}")

    # A5: CFG combination
    print("  A5: CFG combination ablation...")
    for cfg in [3.0, 7.5, 12.0, 20.0]:
        for use_cops in [False, True]:
            label = f"cfg{cfg}" + ("_cops" if use_cops else "_std")
            cs_list, ir_list = [], []
            for i, prompt in enumerate(tqdm(prompts[:100], desc=f"  {label}", leave=False)):
                if use_cops:
                    res = cops_sample_with_resampling(
                        pipe, prompt, num_particles=K_PARTICLES,
                        num_inference_steps=NUM_STEPS, resample_interval=10,
                        guidance_scale=cfg, seed=seed + i,
                    )
                    img = tensor_to_pil(res["image"])
                else:
                    gen = torch.Generator("cuda").manual_seed(seed + i)
                    img = pipe(prompt, num_inference_steps=NUM_STEPS, guidance_scale=cfg,
                               generator=gen, height=IMAGE_SIZE, width=IMAGE_SIZE).images[0]
                cs_list.append(clip_scorer.score_single(img, prompt))
                ir_list.append(ir_scorer.score_single(img, prompt))
            results[label] = {
                "cfg": cfg, "use_cops": use_cops,
                "clip_mean": float(np.mean(cs_list)), "clip_std": float(np.std(cs_list)),
                "ir_mean": float(np.mean(ir_list)), "ir_std": float(np.std(ir_list)),
            }
            print(f"    {label}: CLIP={np.mean(cs_list):.4f}, IR={np.mean(ir_list):.4f}")

    return results


def compute_pcs_correlation(pcs_data):
    """Analyze PCS correlation with CLIP and ImageReward."""
    rho_clip, rho_ir = [], []
    agreements_clip, agreements_ir = 0, 0
    total = 0

    for item in pcs_data:
        pcs = np.array(item["pcs"])
        clip = np.array(item["clip"])
        ir = np.array(item["ir"])

        if len(pcs) >= 3:
            r_clip, _ = spearmanr(pcs, clip)
            r_ir, _ = spearmanr(pcs, ir)
            if not np.isnan(r_clip):
                rho_clip.append(r_clip)
            if not np.isnan(r_ir):
                rho_ir.append(r_ir)

        # Agreement: does PCS select same as CLIP/IR?
        if np.argmax(pcs) == np.argmax(clip):
            agreements_clip += 1
        if np.argmax(pcs) == np.argmax(ir):
            agreements_ir += 1
        total += 1

    return {
        "spearman_pcs_vs_clip_mean": float(np.mean(rho_clip)) if rho_clip else 0,
        "spearman_pcs_vs_clip_std": float(np.std(rho_clip)) if rho_clip else 0,
        "spearman_pcs_vs_ir_mean": float(np.mean(rho_ir)) if rho_ir else 0,
        "spearman_pcs_vs_ir_std": float(np.std(rho_ir)) if rho_ir else 0,
        "agreement_rate_clip": agreements_clip / total if total > 0 else 0,
        "agreement_rate_ir": agreements_ir / total if total > 0 else 0,
        "n_prompts": total,
    }


def download_coco_real_images(out_dir, prompts_file, n=500):
    """Download real COCO images for FID."""
    import urllib.request, io
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_file) as f:
        prompts = json.load(f)

    count = 0
    for item in tqdm(prompts[:n], desc="  Downloading COCO images"):
        img_id = item["id"]
        url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            img = img.resize((512, 512), Image.LANCZOS)
            img.save(out_dir / f"{count:05d}.png")
            count += 1
        except:
            continue
    print(f"  Downloaded {count} images")
    return count


def compute_fid_scores(base_dir):
    """Compute FID for key methods."""
    real_dir = DATA_DIR / "coco_real_images"
    if not real_dir.exists() or len(list(real_dir.glob("*.png"))) < 100:
        print("  Downloading COCO real images...")
        download_coco_real_images(real_dir, DATA_DIR / "coco_500_prompts.json")

    from cleanfid import fid
    results = {}
    methods = [
        ("ddim_50_seed42", "baselines/ddim_50_seed42/coco"),
        ("dpm_20_seed42", "baselines/dpm_20_seed42/coco"),
        ("random_k_seed42", "main/random_k_seed42/coco"),
        ("pcs_bestofk_seed42", "main/pcs_bestofk_seed42/coco"),
        ("clip_bestofk_seed42", "main/clip_bestofk_seed42/coco"),
        ("ir_bestofk_seed42", "main/ir_bestofk_seed42/coco"),
        ("cops_resample_seed42", "main/cops_resample_seed42/coco"),
    ]

    for name, rel in methods:
        gen_dir = base_dir / rel
        if gen_dir.exists() and len(list(gen_dir.glob("*.png"))) > 50:
            try:
                score = fid.compute_fid(str(real_dir), str(gen_dir), device=torch.device("cuda"))
                results[name] = float(score)
                print(f"    FID {name}: {score:.2f}")
            except Exception as e:
                print(f"    FID {name}: FAILED ({e})")
        else:
            print(f"    FID {name}: skipped (no images)")

    return results


# ============================================================================
def main():
    t_start = time.time()
    ALL = {}

    print("=" * 70)
    print("CoPS Experiment Pipeline (Optimized)")
    print("=" * 70)

    # Load model
    print("\n[1] Loading SD1.5 pipeline...")
    pipe = load_pipe()

    # Load prompts
    coco = load_prompts("coco")      # 500
    parti = load_prompts("parti")    # 200
    db = load_prompts("drawbench")   # 41

    # Load scorers
    print("[2] Loading CLIP scorer & ImageReward...")
    from exp.shared.metrics import CLIPScorer, ImageRewardScorer
    clip_s = CLIPScorer("cuda")
    ir_s = ImageRewardScorer("cuda")

    base = EXP_DIR

    # ---- STANDARD BASELINES ----
    print("\n" + "=" * 70)
    print("[3] Standard baselines (DDIM-50, DPM-20, Euler-50)")
    print("=" * 70)

    baseline_results = {}
    for ds_name, prompts in [("coco", coco), ("parti", parti)]:
        for method, steps in [("ddim", 50), ("dpm", 20), ("euler", 50)]:
            for seed in SEEDS:
                key = f"{method}_{steps}_{ds_name}_seed{seed}"
                r = run_standard_baseline(pipe, prompts, ds_name, method, steps, seed, base / "baselines")
                baseline_results[key] = r
                # Score images
                out_dir = base / "baselines" / f"{method}_{steps}_seed{seed}" / ds_name
                cs_list, ir_list = [], []
                for i, prompt in enumerate(prompts):
                    img = Image.open(out_dir / f"{i:05d}.png")
                    cs_list.append(clip_s.score_single(img, prompt))
                    ir_list.append(ir_s.score_single(img, prompt))
                baseline_results[key]["clip_mean"] = float(np.mean(cs_list))
                baseline_results[key]["clip_std"] = float(np.std(cs_list))
                baseline_results[key]["ir_mean"] = float(np.mean(ir_list))
                baseline_results[key]["ir_std"] = float(np.std(ir_list))
                print(f"  {key}: CLIP={np.mean(cs_list):.4f}, IR={np.mean(ir_list):.4f}")

    ALL["standard_baselines"] = baseline_results
    save_checkpoint(ALL, "after_baselines")
    print(f"  Elapsed: {(time.time()-t_start)/60:.1f} min")

    # ---- PARTICLE EXPERIMENTS ----
    print("\n" + "=" * 70)
    print("[4] Particle experiments (K=4, all selection methods)")
    print("=" * 70)

    particle_results = {}
    all_pcs_data = {}
    for ds_name, prompts in [("coco", coco), ("parti", parti)]:
        for seed in SEEDS:
            pr, pcs_data = run_particle_generation(pipe, prompts, ds_name, seed, base / "main", clip_s, ir_s)
            particle_results.update(pr)
            if seed == SEEDS[0]:
                all_pcs_data[ds_name] = pcs_data
        # Print summary
        for m in ["random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk", "cops_resample"]:
            clips = [particle_results[f"{m}_{ds_name}_seed{s}"]["clip_mean"] for s in SEEDS]
            irs = [particle_results[f"{m}_{ds_name}_seed{s}"]["ir_mean"] for s in SEEDS]
            print(f"  {m} on {ds_name}: CLIP={np.mean(clips):.4f}±{np.std(clips):.4f}, IR={np.mean(irs):.4f}±{np.std(irs):.4f}")

    ALL["particle_methods"] = particle_results

    # PCS correlation analysis
    print("\n  PCS Correlation Analysis...")
    correlation_results = {}
    for ds_name, data in all_pcs_data.items():
        corr = compute_pcs_correlation(data)
        correlation_results[ds_name] = corr
        print(f"  {ds_name}: rho(PCS,CLIP)={corr['spearman_pcs_vs_clip_mean']:.4f}±{corr['spearman_pcs_vs_clip_std']:.4f}, "
              f"agree_CLIP={corr['agreement_rate_clip']:.3f}, agree_IR={corr['agreement_rate_ir']:.3f}")
    ALL["pcs_correlation"] = correlation_results

    # Save correlation data for figures
    (EXP_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / "analysis" / "pcs_correlation_data.json", "w") as f:
        json.dump(all_pcs_data, f)

    save_checkpoint(ALL, "after_particles")
    print(f"  Elapsed: {(time.time()-t_start)/60:.1f} min")

    # ---- SCALING ----
    print("\n" + "=" * 70)
    print("[5] Scaling experiment (K=1,2,4,8)")
    print("=" * 70)
    scaling_results = run_scaling(pipe, coco[:200], clip_s, ir_s)
    ALL["scaling"] = scaling_results
    save_checkpoint(ALL, "after_scaling")
    print(f"  Elapsed: {(time.time()-t_start)/60:.1f} min")

    elapsed = time.time() - t_start
    if elapsed > 5.5 * 3600:
        print("\n  WARNING: Over 5.5 hours. Using reduced ablation set (100 prompts).")
        abl_prompts = coco[:100]
    else:
        abl_prompts = coco[:200]

    # ---- ABLATIONS ----
    print("\n" + "=" * 70)
    print(f"[6] Ablation studies ({len(abl_prompts)} prompts)")
    print("=" * 70)
    ablation_results = run_ablations(pipe, abl_prompts, clip_s, ir_s)
    ALL["ablations"] = ablation_results
    save_checkpoint(ALL, "after_ablations")
    print(f"  Elapsed: {(time.time()-t_start)/60:.1f} min")

    # ---- FID ----
    print("\n" + "=" * 70)
    print("[7] Computing FID scores")
    print("=" * 70)
    fid_results = compute_fid_scores(base)
    ALL["fid_scores"] = fid_results
    save_checkpoint(ALL, "after_fid")

    # ---- COST ANALYSIS ----
    print("\n" + "=" * 70)
    print("[8] Cost analysis")
    print("=" * 70)
    cost_results = run_cost_analysis(pipe, coco[:50], clip_s, ir_s)
    ALL["cost_analysis"] = cost_results

    # ---- FINAL RESULTS ----
    total_time = time.time() - t_start
    ALL["total_runtime_seconds"] = total_time
    ALL["total_runtime_hours"] = total_time / 3600

    with open(WORKSPACE / "results.json", "w") as f:
        json.dump(ALL, f, indent=2)
    print(f"\n{'='*70}")
    print(f"DONE! Total runtime: {total_time/3600:.2f} hours")
    print(f"Results saved to results.json")
    print(f"{'='*70}")

    return ALL


def run_cost_analysis(pipe, prompts, clip_s, ir_s):
    """Measure wall time and memory for each method."""
    from exp.shared.cops import generate_particles_batch, cops_sample_with_resampling

    results = {}
    seed = 42
    set_scheduler(pipe, "ddim")

    # Standard DDIM-50
    times = []
    for i, prompt in enumerate(prompts[:20]):
        torch.cuda.reset_peak_memory_stats()
        gen = torch.Generator("cuda").manual_seed(seed + i)
        t0 = time.time()
        pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=gen).images[0]
        times.append(time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    results["ddim_50"] = {"time_per_img": float(np.mean(times)), "peak_gpu_gb": float(mem), "nfe": 50}

    # K=4 particles
    times = []
    for i, prompt in enumerate(prompts[:20]):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        generate_particles_batch(pipe, prompt, num_particles=4, num_inference_steps=50,
                                 guidance_scale=7.5, seed=seed + i)
        times.append(time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    results["particles_k4"] = {"time_per_img": float(np.mean(times)), "peak_gpu_gb": float(mem), "nfe": 200}

    # CoPS with resampling
    times = []
    for i, prompt in enumerate(prompts[:20]):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        cops_sample_with_resampling(pipe, prompt, num_particles=4, num_inference_steps=50,
                                     resample_interval=10, guidance_scale=7.5, seed=seed + i)
        times.append(time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    results["cops_k4"] = {"time_per_img": float(np.mean(times)), "peak_gpu_gb": float(mem), "nfe": 200}

    for name, r in results.items():
        print(f"  {name}: {r['time_per_img']:.2f}s/img, {r['peak_gpu_gb']:.1f}GB, NFE={r['nfe']}")

    return results


if __name__ == "__main__":
    main()
