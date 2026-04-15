#!/usr/bin/env python3
"""
CoPS Experiment Pipeline — Optimized for 1×A6000, 7-hour budget.

Time budget allocation:
  - Baselines (DDIM-50, DPM-20): ~30 min
  - Particle experiments (COCO-300 × 3 seeds): ~3 hours
  - PartiPrompts (100 × 1 seed): ~15 min
  - Scaling (100 prompts): ~30 min
  - Ablations (100 prompts): ~1.5 hours
  - FID + cost analysis: ~30 min
  - Figures: ~5 min
  Total: ~6.5 hours
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

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path(".").resolve()))

WORKSPACE = Path(".")
EXP_DIR = WORKSPACE / "exp"
DATA_DIR = EXP_DIR / "data"
FIGURES_DIR = WORKSPACE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456]
K = 4
STEPS = 50
CFG = 7.5
SZ = 512

T_START = time.time()

def elapsed_hours():
    return (time.time() - T_START) / 3600

def load_prompts(name, n=None):
    files = {"coco": "coco_500_prompts.json", "parti": "parti_200_prompts.json", "drawbench": "drawbench_prompts.json"}
    with open(DATA_DIR / files[name]) as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]
    return prompts[:n] if n else prompts

def load_pipe():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def set_sched(pipe, name):
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
    S = {"ddim": DDIMScheduler, "dpm": DPMSolverMultistepScheduler, "euler": EulerDiscreteScheduler}
    pipe.scheduler = S[name].from_config(pipe.scheduler.config)

def t2pil(t):
    return Image.fromarray((t.cpu().float().clamp(0,1)*255).byte().permute(1,2,0).numpy())

def save_ckpt(results, name):
    d = EXP_DIR / "checkpoints"; d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.json", "w") as f:
        json.dump(results, f, indent=2)

# ============================================================================
# PHASE 1: Standard baselines
# ============================================================================
def phase_baselines(pipe, prompts, ds, clip_s, ir_s):
    """Run DDIM-50 and DPM-20 baselines."""
    results = {}
    for method, steps in [("ddim", 50), ("dpm", 20)]:
        set_sched(pipe, method)
        for seed in SEEDS:
            out = EXP_DIR / "baselines" / f"{method}_{steps}_seed{seed}" / ds
            out.mkdir(parents=True, exist_ok=True)
            cs_all, ir_all, times = [], [], []
            for i, p in enumerate(tqdm(prompts, desc=f"  {method}{steps} s{seed}", leave=False)):
                gen = torch.Generator("cuda").manual_seed(seed + i)
                t0 = time.time()
                with torch.no_grad():
                    img = pipe(p, num_inference_steps=steps, guidance_scale=CFG,
                               generator=gen, height=SZ, width=SZ).images[0]
                times.append(time.time() - t0)
                img.save(out / f"{i:05d}.png")
                cs_all.append(clip_s.score_single(img, p))
                ir_all.append(ir_s.score_single(img, p))
            key = f"{method}_{steps}_{ds}_seed{seed}"
            results[key] = {
                "method": f"{method}_{steps}", "dataset": ds, "seed": seed,
                "clip_mean": float(np.mean(cs_all)), "clip_std": float(np.std(cs_all)),
                "ir_mean": float(np.mean(ir_all)), "ir_std": float(np.std(ir_all)),
                "avg_time": float(np.mean(times)), "n": len(prompts),
            }
            print(f"    {key}: CLIP={np.mean(cs_all):.4f}, IR={np.mean(ir_all):.4f}, "
                  f"time={np.mean(times):.2f}s/img")
    return results


# ============================================================================
# PHASE 2: Particle generation + all selection methods + CoPS
# ============================================================================
def phase_particles(pipe, prompts, ds, seeds, clip_s, ir_s):
    """Generate K particles, apply all selection methods, run CoPS resample."""
    from exp.shared.cops import generate_particles_batch, cops_sample_with_resampling

    set_sched(pipe, "ddim")
    results = {}
    pcs_corr_data = []  # For correlation analysis (first seed only)

    for seed in seeds:
        methods = ["random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk", "cops_resample"]
        dirs = {}
        for m in methods:
            d = EXP_DIR / "main" / f"{m}_seed{seed}" / ds
            d.mkdir(parents=True, exist_ok=True)
            dirs[m] = d

        scores = {m: {"clip": [], "ir": []} for m in methods}

        for i, p in enumerate(tqdm(prompts, desc=f"  Particles s{seed} {ds}", leave=False)):
            # Generate K particles (no resampling)
            res = generate_particles_batch(
                pipe, p, num_particles=K, num_inference_steps=STEPS,
                guidance_scale=CFG, seed=seed+i, height=SZ, width=SZ,
                track_pcs=True, distance_metric="l2",
            )
            imgs = res["images"]
            pcs = np.array(res["pcs_scores"])
            pils = [t2pil(img) for img in imgs]

            cs = clip_s.score_batch(pils, [p]*K)
            irs = ir_s.score_batch(pils, [p]*K)

            if seed == seeds[0]:
                pcs_corr_data.append({
                    "idx": i, "prompt": p,
                    "pcs": pcs.tolist(), "clip": cs, "ir": irs,
                    "step_coherences": res.get("step_coherences", []),
                })

            # Selection
            rng = np.random.RandomState(seed+i)
            sels = {"random_k": rng.randint(K), "pcs_bestofk": int(np.argmax(pcs)),
                    "clip_bestofk": int(np.argmax(cs)), "ir_bestofk": int(np.argmax(irs))}
            for m, idx in sels.items():
                pils[idx].save(dirs[m] / f"{i:05d}.png")
                scores[m]["clip"].append(cs[idx])
                scores[m]["ir"].append(irs[idx])

            # CoPS with resampling
            cops_res = cops_sample_with_resampling(
                pipe, p, num_particles=K, num_inference_steps=STEPS,
                resample_interval=10, alpha=1.0, sigma_jitter=0.01,
                distance_metric="l2", guidance_scale=CFG, seed=seed+i,
            )
            cops_img = t2pil(cops_res["image"])
            cops_img.save(dirs["cops_resample"] / f"{i:05d}.png")
            ccs = clip_s.score_single(cops_img, p)
            cir = ir_s.score_single(cops_img, p)
            scores["cops_resample"]["clip"].append(ccs)
            scores["cops_resample"]["ir"].append(cir)

        for m in methods:
            key = f"{m}_{ds}_seed{seed}"
            results[key] = {
                "method": m, "dataset": ds, "seed": seed,
                "clip_mean": float(np.mean(scores[m]["clip"])),
                "clip_std": float(np.std(scores[m]["clip"])),
                "ir_mean": float(np.mean(scores[m]["ir"])),
                "ir_std": float(np.std(scores[m]["ir"])),
                "n": len(prompts),
            }

        # Print summary for this seed
        for m in methods:
            print(f"    {m} s{seed}: CLIP={np.mean(scores[m]['clip']):.4f}, IR={np.mean(scores[m]['ir']):.4f}")

    return results, pcs_corr_data


# ============================================================================
# PHASE 3: Scaling experiment
# ============================================================================
def phase_scaling(pipe, prompts, clip_s, ir_s):
    """Vary K with constant total NFE=200."""
    from exp.shared.cops import generate_particles_batch
    set_sched(pipe, "ddim")
    seed = 42
    total_nfe = 200
    results = {}

    for k in [1, 2, 4, 8]:
        steps = total_nfe // k
        ms = {m: {"clip": [], "ir": []} for m in ["random", "pcs", "clip_sel"]}
        for i, p in enumerate(tqdm(prompts, desc=f"  K={k} ({steps}st)", leave=False)):
            if k == 1:
                gen = torch.Generator("cuda").manual_seed(seed+i)
                with torch.no_grad():
                    img = pipe(p, num_inference_steps=steps, guidance_scale=CFG,
                               generator=gen, height=SZ, width=SZ).images[0]
                c = clip_s.score_single(img, p)
                r = ir_s.score_single(img, p)
                for m in ms: ms[m]["clip"].append(c); ms[m]["ir"].append(r)
            else:
                res = generate_particles_batch(
                    pipe, p, num_particles=k, num_inference_steps=steps,
                    guidance_scale=CFG, seed=seed+i, height=SZ, width=SZ,
                )
                pils = [t2pil(img) for img in res["images"]]
                cs = clip_s.score_batch(pils, [p]*k)
                irs_list = ir_s.score_batch(pils, [p]*k)
                pcs = np.array(res["pcs_scores"])
                rng = np.random.RandomState(seed+i)
                sel = {"random": rng.randint(k), "pcs": int(np.argmax(pcs)),
                       "clip_sel": int(np.argmax(cs))}
                for m, idx in sel.items():
                    ms[m]["clip"].append(cs[idx])
                    ms[m]["ir"].append(irs_list[idx])

        for m in ms:
            results[f"K{k}_{m}"] = {
                "K": k, "steps": steps, "nfe": k*steps, "selection": m,
                "clip_mean": float(np.mean(ms[m]["clip"])),
                "clip_std": float(np.std(ms[m]["clip"])),
                "ir_mean": float(np.mean(ms[m]["ir"])),
                "ir_std": float(np.std(ms[m]["ir"])),
            }
        print(f"    K={k}: pcs_clip={np.mean(ms['pcs']['clip']):.4f}, "
              f"rand_clip={np.mean(ms['random']['clip']):.4f}")

    return results


# ============================================================================
# PHASE 4: Ablations
# ============================================================================
def phase_ablations(pipe, prompts, clip_s, ir_s):
    """Key ablation studies."""
    from exp.shared.cops import cops_sample_with_resampling, generate_particles_batch, PCSTracker
    set_sched(pipe, "ddim")
    seed = 42
    results = {}

    # A1: Distance metric
    print("  A1: Distance metric...")
    for metric in ["l2", "cosine"]:
        cs_l, ir_l = [], []
        for i, p in enumerate(tqdm(prompts, desc=f"    {metric}", leave=False)):
            res = generate_particles_batch(pipe, p, num_particles=K, num_inference_steps=STEPS,
                                           guidance_scale=CFG, seed=seed+i, distance_metric=metric)
            img = t2pil(res["images"][int(np.argmax(res["pcs_scores"]))])
            cs_l.append(clip_s.score_single(img, p))
            ir_l.append(ir_s.score_single(img, p))
        results[f"dist_{metric}"] = {
            "metric": metric,
            "clip_mean": float(np.mean(cs_l)), "clip_std": float(np.std(cs_l)),
            "ir_mean": float(np.mean(ir_l)), "ir_std": float(np.std(ir_l)),
        }
        print(f"    {metric}: CLIP={np.mean(cs_l):.4f}, IR={np.mean(ir_l):.4f}")

    # A2: Resampling frequency
    print("  A2: Resampling frequency...")
    for R in [1, 5, 10, 25, 50]:
        cs_l, ir_l = [], []
        for i, p in enumerate(tqdm(prompts, desc=f"    R={R}", leave=False)):
            if R >= STEPS:
                res = generate_particles_batch(pipe, p, num_particles=K,
                                               num_inference_steps=STEPS, guidance_scale=CFG, seed=seed+i)
                img = t2pil(res["images"][int(np.argmax(res["pcs_scores"]))])
            else:
                res = cops_sample_with_resampling(pipe, p, num_particles=K,
                                                   num_inference_steps=STEPS, resample_interval=R,
                                                   alpha=1.0, sigma_jitter=0.01, guidance_scale=CFG, seed=seed+i)
                img = t2pil(res["image"])
            cs_l.append(clip_s.score_single(img, p))
            ir_l.append(ir_s.score_single(img, p))
        results[f"R_{R}"] = {
            "R": R,
            "clip_mean": float(np.mean(cs_l)), "clip_std": float(np.std(cs_l)),
            "ir_mean": float(np.mean(ir_l)), "ir_std": float(np.std(ir_l)),
        }
        print(f"    R={R}: CLIP={np.mean(cs_l):.4f}, IR={np.mean(ir_l):.4f}")

    # A3: Timestep weighting (post-hoc, reusing same particles)
    print("  A3: Timestep weighting...")
    # Pre-generate particles once
    all_data = []
    for i, p in enumerate(tqdm(prompts, desc="    gen", leave=False)):
        res = generate_particles_batch(pipe, p, num_particles=K,
                                       num_inference_steps=STEPS, guidance_scale=CFG,
                                       seed=seed+i, track_pcs=True)
        pils = [t2pil(img) for img in res["images"]]
        cs = clip_s.score_batch(pils, [p]*K)
        ir_scores = ir_s.score_batch(pils, [p]*K)
        all_data.append({"step_coherences": res.get("step_coherences", []),
                         "pcs": res["pcs_scores"], "clip": cs, "ir": ir_scores})

    for ws in ["uniform", "mid_emphasis", "early_emphasis", "late_emphasis"]:
        cs_l, ir_l = [], []
        for data in all_data:
            sc = data["step_coherences"]
            if sc:
                tracker = PCSTracker(num_particles=K, timestep_weights=ws, total_steps=STEPS)
                pcs = np.zeros(K)
                for si, coh in enumerate(sc):
                    pcs += np.array(coh) * tracker._get_weight(si+1)
                best = int(np.argmax(pcs))
            else:
                best = int(np.argmax(data["pcs"]))
            cs_l.append(data["clip"][best])
            ir_l.append(data["ir"][best])
        results[f"weight_{ws}"] = {
            "weight": ws,
            "clip_mean": float(np.mean(cs_l)), "clip_std": float(np.std(cs_l)),
            "ir_mean": float(np.mean(ir_l)), "ir_std": float(np.std(ir_l)),
        }
        print(f"    {ws}: CLIP={np.mean(cs_l):.4f}, IR={np.mean(ir_l):.4f}")

    # A5: CFG combination
    print("  A5: CFG combination...")
    for cfg in [3.0, 7.5, 12.0, 20.0]:
        for use_cops in [False, True]:
            label = f"cfg{cfg}" + ("_cops" if use_cops else "_std")
            cs_l, ir_l = [], []
            n = min(len(prompts), 80)
            for i, p in enumerate(tqdm(prompts[:n], desc=f"    {label}", leave=False)):
                if use_cops:
                    res = cops_sample_with_resampling(pipe, p, num_particles=K,
                                                       num_inference_steps=STEPS, resample_interval=10,
                                                       guidance_scale=cfg, seed=seed+i)
                    img = t2pil(res["image"])
                else:
                    gen = torch.Generator("cuda").manual_seed(seed+i)
                    img = pipe(p, num_inference_steps=STEPS, guidance_scale=cfg,
                               generator=gen, height=SZ, width=SZ).images[0]
                cs_l.append(clip_s.score_single(img, p))
                ir_l.append(ir_s.score_single(img, p))
            results[label] = {
                "cfg": cfg, "use_cops": use_cops,
                "clip_mean": float(np.mean(cs_l)), "clip_std": float(np.std(cs_l)),
                "ir_mean": float(np.mean(ir_l)), "ir_std": float(np.std(ir_l)),
            }
            print(f"    {label}: CLIP={np.mean(cs_l):.4f}, IR={np.mean(ir_l):.4f}")

    return results


# ============================================================================
# FID + Cost
# ============================================================================
def phase_fid(base_dir):
    """Compute FID for key methods."""
    real_dir = DATA_DIR / "coco_real_images"
    if not real_dir.exists() or len(list(real_dir.glob("*.png"))) < 50:
        print("  Downloading COCO real images...")
        import urllib.request, io
        real_dir.mkdir(parents=True, exist_ok=True)
        with open(DATA_DIR / "coco_500_prompts.json") as f:
            items = json.load(f)
        count = 0
        for item in tqdm(items[:300], desc="  COCO images"):
            img_id = item["id"]
            url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
            try:
                resp = urllib.request.urlopen(url, timeout=15)
                img = Image.open(io.BytesIO(resp.read())).convert("RGB").resize((512,512), Image.LANCZOS)
                img.save(real_dir / f"{count:05d}.png")
                count += 1
            except: continue
        print(f"  Downloaded {count} images")

    from cleanfid import fid
    results = {}
    pairs = [
        ("ddim_50", "baselines/ddim_50_seed42/coco"),
        ("dpm_20", "baselines/dpm_20_seed42/coco"),
        ("random_k", "main/random_k_seed42/coco"),
        ("pcs_bestofk", "main/pcs_bestofk_seed42/coco"),
        ("clip_bestofk", "main/clip_bestofk_seed42/coco"),
        ("ir_bestofk", "main/ir_bestofk_seed42/coco"),
        ("cops_resample", "main/cops_resample_seed42/coco"),
    ]
    for name, rel in pairs:
        gen_dir = base_dir / rel
        if gen_dir.exists() and len(list(gen_dir.glob("*.png"))) >= 50:
            try:
                s = fid.compute_fid(str(real_dir), str(gen_dir), device=torch.device("cuda"))
                results[name] = float(s)
                print(f"    {name}: FID={s:.2f}")
            except Exception as e:
                print(f"    {name}: FAILED ({e})")
    return results


def phase_cost(pipe, prompts, clip_s, ir_s):
    """Measure wall time and memory."""
    from exp.shared.cops import generate_particles_batch, cops_sample_with_resampling
    set_sched(pipe, "ddim")
    results = {}
    n = 20
    seed = 42

    configs = [
        ("ddim_50", lambda i,p: pipe(p, num_inference_steps=50, guidance_scale=7.5,
                                      generator=torch.Generator("cuda").manual_seed(seed+i),
                                      height=SZ, width=SZ)),
        ("particles_k4", lambda i,p: generate_particles_batch(pipe, p, num_particles=4,
                                                                num_inference_steps=50, guidance_scale=7.5,
                                                                seed=seed+i)),
        ("cops_k4", lambda i,p: cops_sample_with_resampling(pipe, p, num_particles=4,
                                                              num_inference_steps=50, resample_interval=10,
                                                              guidance_scale=7.5, seed=seed+i)),
    ]
    nfes = {"ddim_50": 50, "particles_k4": 200, "cops_k4": 200}

    for name, fn in configs:
        times = []
        for i in range(n):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            fn(i, prompts[i])
            times.append(time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        results[name] = {
            "time_per_img": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "peak_gpu_gb": float(mem),
            "nfe": nfes[name],
        }
        print(f"    {name}: {np.mean(times):.2f}s/img, {mem:.1f}GB GPU")

    return results


# ============================================================================
# PCS Correlation Analysis
# ============================================================================
def analyze_pcs_correlation(pcs_data):
    """Compute rank correlation between PCS and external metrics."""
    rho_clip, rho_ir = [], []
    agree_clip, agree_ir, total = 0, 0, 0

    for item in pcs_data:
        pcs = np.array(item["pcs"])
        clip = np.array(item["clip"])
        ir = np.array(item["ir"])

        if len(pcs) >= 3:
            rc, _ = spearmanr(pcs, clip)
            ri, _ = spearmanr(pcs, ir)
            if not np.isnan(rc): rho_clip.append(rc)
            if not np.isnan(ri): rho_ir.append(ri)

        if np.argmax(pcs) == np.argmax(clip): agree_clip += 1
        if np.argmax(pcs) == np.argmax(ir): agree_ir += 1
        total += 1

    return {
        "spearman_pcs_clip_mean": float(np.mean(rho_clip)) if rho_clip else 0,
        "spearman_pcs_clip_std": float(np.std(rho_clip)) if rho_clip else 0,
        "spearman_pcs_ir_mean": float(np.mean(rho_ir)) if rho_ir else 0,
        "spearman_pcs_ir_std": float(np.std(rho_ir)) if rho_ir else 0,
        "agreement_clip": agree_clip / total if total else 0,
        "agreement_ir": agree_ir / total if total else 0,
        "n": total,
        "n_valid_rho": len(rho_clip),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    ALL = {}
    print("=" * 70)
    print("CoPS Experiment Pipeline")
    print("=" * 70)

    # Load
    print(f"\n[1] Loading models... ({elapsed_hours():.1f}h)")
    pipe = load_pipe()
    from exp.shared.metrics import CLIPScorer, ImageRewardScorer
    clip_s = CLIPScorer("cuda")
    ir_s = ImageRewardScorer("cuda")

    coco_300 = load_prompts("coco", 300)
    parti_100 = load_prompts("parti", 100)

    # PHASE 1: Baselines
    print(f"\n{'='*70}\n[2] Standard baselines ({elapsed_hours():.1f}h)\n{'='*70}")
    ALL["baselines"] = phase_baselines(pipe, coco_300, "coco", clip_s, ir_s)
    save_ckpt(ALL, "p1_baselines")
    print(f"  Elapsed: {elapsed_hours():.2f}h")

    # PHASE 2: Particles on COCO
    print(f"\n{'='*70}\n[3] Particle experiments — COCO ({elapsed_hours():.1f}h)\n{'='*70}")
    pr_coco, pcs_data_coco = phase_particles(pipe, coco_300, "coco", SEEDS, clip_s, ir_s)
    ALL["particles_coco"] = pr_coco
    # Print multi-seed summary
    for m in ["random_k", "pcs_bestofk", "clip_bestofk", "ir_bestofk", "cops_resample"]:
        clips = [pr_coco[f"{m}_coco_seed{s}"]["clip_mean"] for s in SEEDS]
        irs = [pr_coco[f"{m}_coco_seed{s}"]["ir_mean"] for s in SEEDS]
        print(f"  {m}: CLIP={np.mean(clips):.4f}±{np.std(clips):.4f}, IR={np.mean(irs):.4f}±{np.std(irs):.4f}")
    save_ckpt(ALL, "p2_particles_coco")
    print(f"  Elapsed: {elapsed_hours():.2f}h")

    # PHASE 2b: Particles on PartiPrompts (1 seed)
    print(f"\n{'='*70}\n[4] Particle experiments — PartiPrompts ({elapsed_hours():.1f}h)\n{'='*70}")
    pr_parti, pcs_data_parti = phase_particles(pipe, parti_100, "parti", [42], clip_s, ir_s)
    ALL["particles_parti"] = pr_parti
    save_ckpt(ALL, "p2b_particles_parti")
    print(f"  Elapsed: {elapsed_hours():.2f}h")

    # PCS Correlation
    print(f"\n  PCS Correlation Analysis...")
    corr_coco = analyze_pcs_correlation(pcs_data_coco)
    corr_parti = analyze_pcs_correlation(pcs_data_parti)
    ALL["pcs_correlation"] = {"coco": corr_coco, "parti": corr_parti}
    print(f"  COCO: rho(PCS,CLIP)={corr_coco['spearman_pcs_clip_mean']:.4f}±{corr_coco['spearman_pcs_clip_std']:.4f}, "
          f"agree_CLIP={corr_coco['agreement_clip']:.3f}")
    print(f"  Parti: rho(PCS,CLIP)={corr_parti['spearman_pcs_clip_mean']:.4f}±{corr_parti['spearman_pcs_clip_std']:.4f}")

    # Save PCS data for figures
    (EXP_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / "analysis" / "pcs_data_coco.json", "w") as f:
        json.dump(pcs_data_coco, f)

    # PHASE 3: Scaling
    print(f"\n{'='*70}\n[5] Scaling experiment ({elapsed_hours():.1f}h)\n{'='*70}")
    n_scale = 150 if elapsed_hours() < 4.0 else 100
    ALL["scaling"] = phase_scaling(pipe, coco_300[:n_scale], clip_s, ir_s)
    save_ckpt(ALL, "p3_scaling")
    print(f"  Elapsed: {elapsed_hours():.2f}h")

    # PHASE 4: Ablations
    print(f"\n{'='*70}\n[6] Ablation studies ({elapsed_hours():.1f}h)\n{'='*70}")
    n_abl = 100 if elapsed_hours() < 5.0 else 60
    ALL["ablations"] = phase_ablations(pipe, coco_300[:n_abl], clip_s, ir_s)
    save_ckpt(ALL, "p4_ablations")
    print(f"  Elapsed: {elapsed_hours():.2f}h")

    # PHASE 5: FID
    print(f"\n{'='*70}\n[7] FID computation ({elapsed_hours():.1f}h)\n{'='*70}")
    ALL["fid"] = phase_fid(EXP_DIR)
    save_ckpt(ALL, "p5_fid")
    print(f"  Elapsed: {elapsed_hours():.2f}h")

    # PHASE 6: Cost analysis
    print(f"\n{'='*70}\n[8] Cost analysis ({elapsed_hours():.1f}h)\n{'='*70}")
    ALL["cost"] = phase_cost(pipe, coco_300, clip_s, ir_s)
    save_ckpt(ALL, "p6_cost")

    # Save final
    ALL["total_hours"] = elapsed_hours()
    with open(WORKSPACE / "results.json", "w") as f:
        json.dump(ALL, f, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE! Total: {elapsed_hours():.2f}h")
    print(f"Results saved to results.json")
    print(f"{'='*70}")
    return ALL


if __name__ == "__main__":
    main()
