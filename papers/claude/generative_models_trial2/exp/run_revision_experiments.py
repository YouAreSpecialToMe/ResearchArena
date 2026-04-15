"""
Revision experiments to address reviewer feedback:
1. LPIPS distance metric ablation (100 COCO prompts)
2. Re-score ablation experiments with corrected ImageReward
3. Run CoPS with active resampling on DrawBench
4. Try SDXL evaluation on small subset
"""
import os, sys, json, time, torch
import numpy as np
from PIL import Image
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXP = BASE / "exp"
sys.path.insert(0, str(EXP / "shared"))

from cops import PCSTracker, generate_particles_batch, cops_sample_with_resampling
from metrics import CLIPScorer, tensor_to_pil

###############################################################################
# Helpers
###############################################################################

def load_imagereward():
    import ImageReward as RM
    return RM.load("ImageReward-v1.0")

def load_prompts(name, max_n=None):
    if name == "coco":
        with open(EXP / "data" / "coco_500_prompts.json") as f:
            data = json.load(f)
        prompts = [d["prompt"] if isinstance(d, dict) else d for d in data]
    elif name == "parti":
        with open(EXP / "data" / "parti_200_prompts.json") as f:
            data = json.load(f)
        prompts = [d["prompt"] if isinstance(d, dict) else d for d in data]
    elif name == "drawbench":
        pf = EXP / "data" / "drawbench_prompts.json"
        if not pf.exists():
            pf = EXP / "drawbench" / "drawbench_prompts.json"
        with open(pf) as f:
            data = json.load(f)
        prompts = [d["prompt"] if isinstance(d, dict) else d for d in data]
    else:
        raise ValueError(name)
    if max_n:
        prompts = prompts[:max_n]
    return prompts

def load_sd15():
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


###############################################################################
# Experiment 1: LPIPS distance metric ablation
###############################################################################

def run_lpips_ablation(pipe, prompts, seed=42, K=4):
    """Run PCS with LPIPS distance in pixel space on COCO subset."""
    import lpips
    lpips_model = lpips.LPIPS(net='alex').cuda().eval()

    results = {"prompts": [], "pcs_l2": [], "pcs_lpips": [], "pcs_cosine": [],
               "clip_scores": [], "ir_scores": []}

    clip_scorer = CLIPScorer(device="cuda")
    ir_model = load_imagereward()

    print(f"Running LPIPS ablation on {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        if i % 20 == 0:
            print(f"  Prompt {i}/{len(prompts)}")

        device = pipe.device

        # Generate particles and track denoised predictions at each step
        text_inputs = pipe.tokenizer(prompt, padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_inputs = pipe.tokenizer("", padding="max_length",
            max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]

        text_emb = text_embeddings.repeat(K, 1, 1)
        uncond_emb = uncond_embeddings.repeat(K, 1, 1)

        pipe.scheduler.set_timesteps(50, device=device)
        timesteps = pipe.scheduler.timesteps

        generator = torch.Generator(device=device).manual_seed(seed)
        latent_shape = (K, pipe.unet.config.in_channels, 64, 64)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=torch.float16)
        latents = latents * pipe.scheduler.init_noise_sigma

        # Track PCS with all 3 metrics simultaneously
        pcs_l2 = PCSTracker(K, "l2", total_steps=50)
        pcs_cos = PCSTracker(K, "cosine", total_steps=50)

        # For LPIPS, we track manually
        prev_decoded = None
        lpips_pcs = torch.zeros(K, device=device)

        for step_idx, t in enumerate(timesteps):
            t_batch = t.expand(K)
            latent_model_input = torch.cat([latents, latents], dim=0)
            t_model = torch.cat([t_batch, t_batch])
            prompt_embeds = torch.cat([uncond_emb, text_emb], dim=0)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t_model,
                    encoder_hidden_states=prompt_embeds).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            alpha_prod_t = pipe.scheduler.alphas_cumprod[t.long()].to(device)
            beta_prod_t = 1 - alpha_prod_t
            denoised_pred = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

            # Update L2 and cosine PCS
            pcs_l2.update(denoised_pred)
            pcs_cos.update(denoised_pred)

            # Compute LPIPS PCS (decode to pixel space every 5 steps to save compute)
            if step_idx % 5 == 0 or step_idx == len(timesteps) - 1:
                with torch.no_grad():
                    decoded = pipe.vae.decode(denoised_pred / pipe.vae.config.scaling_factor).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)

                if prev_decoded is not None:
                    with torch.no_grad():
                        # LPIPS expects [-1, 1] range
                        d1 = decoded * 2 - 1
                        d2 = prev_decoded * 2 - 1
                        for k_idx in range(K):
                            dist = lpips_model(d1[k_idx:k_idx+1].float(), d2[k_idx:k_idx+1].float())
                            lpips_pcs[k_idx] -= dist.item()

                prev_decoded = decoded.detach().clone()

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode final images
        latents_decoded = latents / pipe.vae.config.scaling_factor
        with torch.no_grad():
            images = pipe.vae.decode(latents_decoded).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        # Score with CLIP and ImageReward
        pil_images = [tensor_to_pil(img) for img in images]
        clip_scores = clip_scorer.score_batch(pil_images, [prompt] * K)
        ir_scores = [float(ir_model.score(prompt, img)) for img in pil_images]

        results["prompts"].append(prompt)
        results["pcs_l2"].append(pcs_l2.pcs_scores.cpu().numpy().tolist())
        results["pcs_cosine"].append(pcs_cos.pcs_scores.cpu().numpy().tolist())
        results["pcs_lpips"].append(lpips_pcs.cpu().numpy().tolist())
        results["clip_scores"].append(clip_scores)
        results["ir_scores"].append(ir_scores)

    return results


###############################################################################
# Experiment 2: Re-score ablations with corrected ImageReward
###############################################################################

def rescore_ablations(pipe):
    """Re-generate ablation images and score with corrected ImageReward."""
    ir_model = load_imagereward()
    clip_scorer = CLIPScorer(device="cuda")

    prompts = load_prompts("coco", 300)

    results = {}

    # Distance metric ablation: L2 vs Cosine (already have particles, just need IR rescoring)
    # We need to regenerate with both metrics and score properly
    print("Re-scoring distance metric ablation...")
    for metric_name in ["l2", "cosine"]:
        all_clip = []
        all_ir = []
        for i, prompt in enumerate(prompts):
            if i % 50 == 0:
                print(f"  {metric_name}: {i}/{len(prompts)}")
            res = generate_particles_batch(pipe, prompt, num_particles=4,
                num_inference_steps=50, seed=42, distance_metric=metric_name)
            pcs = np.array(res["pcs_scores"])
            best_idx = int(np.argmax(pcs))

            pil_images = [tensor_to_pil(img) for img in res["images"]]
            best_img = pil_images[best_idx]

            clip_s = clip_scorer.score_single(best_img, prompt)
            ir_s = float(ir_model.score(prompt, best_img))
            all_clip.append(clip_s)
            all_ir.append(ir_s)

        results[f"distance_{metric_name}"] = {
            "clip_mean": float(np.mean(all_clip)),
            "clip_std": float(np.std(all_clip)),
            "ir_mean": float(np.mean(all_ir)),
            "ir_std": float(np.std(all_ir)),
        }

    # Resampling frequency ablation on 200-prompt subset
    print("Re-scoring resampling frequency ablation...")
    prompts_200 = prompts[:200]
    for R in [1, 5, 10, 25, 50]:
        all_clip = []
        all_ir = []
        for i, prompt in enumerate(prompts_200):
            if i % 50 == 0:
                print(f"  R={R}: {i}/{len(prompts_200)}")

            if R >= 50:
                # No resampling = PCS Best-of-K
                res = generate_particles_batch(pipe, prompt, num_particles=4,
                    num_inference_steps=50, seed=42)
                pcs = np.array(res["pcs_scores"])
                best_idx = int(np.argmax(pcs))
                pil_images = [tensor_to_pil(img) for img in res["images"]]
                best_img = pil_images[best_idx]
            else:
                res = cops_sample_with_resampling(pipe, prompt, num_particles=4,
                    num_inference_steps=50, resample_interval=R, seed=42)
                best_img = tensor_to_pil(res["image"])

            clip_s = clip_scorer.score_single(best_img, prompt)
            ir_s = float(ir_model.score(prompt, best_img))
            all_clip.append(clip_s)
            all_ir.append(ir_s)

        results[f"resample_R{R}"] = {
            "clip_mean": float(np.mean(all_clip)),
            "clip_std": float(np.std(all_clip)),
            "ir_mean": float(np.mean(all_ir)),
            "ir_std": float(np.std(all_ir)),
        }

    # Timestep weighting ablation (post-hoc, reuse same particles)
    print("Re-scoring timestep weighting ablation...")
    for weight_scheme in ["uniform", "mid_emphasis", "early_emphasis", "late_emphasis"]:
        all_clip = []
        all_ir = []
        for i, prompt in enumerate(prompts):
            if i % 50 == 0:
                print(f"  {weight_scheme}: {i}/{len(prompts)}")

            # Generate particles with this weighting
            # For post-hoc, we generate once with L2 and change weights
            device = pipe.device
            K = 4
            text_inputs = pipe.tokenizer(prompt, padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
            uncond_inputs = pipe.tokenizer("", padding="max_length",
                max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
            uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(device))[0]

            text_emb = text_embeddings.repeat(K, 1, 1)
            uncond_emb = uncond_embeddings.repeat(K, 1, 1)

            pipe.scheduler.set_timesteps(50, device=device)
            timesteps = pipe.scheduler.timesteps

            gen = torch.Generator(device=device).manual_seed(42)
            latent_shape = (K, pipe.unet.config.in_channels, 64, 64)
            latents = torch.randn(latent_shape, generator=gen, device=device, dtype=torch.float16)
            latents = latents * pipe.scheduler.init_noise_sigma

            pcs_tracker = PCSTracker(K, "l2", timestep_weights=weight_scheme, total_steps=50)

            for step_idx, t in enumerate(timesteps):
                t_batch = t.expand(K)
                latent_model_input = torch.cat([latents, latents], dim=0)
                t_model = torch.cat([t_batch, t_batch])
                prompt_embeds = torch.cat([uncond_emb, text_emb], dim=0)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    noise_pred = pipe.unet(latent_model_input, t_model,
                        encoder_hidden_states=prompt_embeds).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

                alpha_prod_t = pipe.scheduler.alphas_cumprod[t.long()].to(device)
                beta_prod_t = 1 - alpha_prod_t
                denoised_pred = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()

                pcs_tracker.update(denoised_pred)
                latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            latents_decoded = latents / pipe.vae.config.scaling_factor
            with torch.no_grad():
                images_out = pipe.vae.decode(latents_decoded).sample
            images_out = (images_out / 2 + 0.5).clamp(0, 1)

            pcs = pcs_tracker.pcs_scores.cpu().numpy()
            best_idx = int(np.argmax(pcs))
            best_img = tensor_to_pil(images_out[best_idx])

            clip_s = clip_scorer.score_single(best_img, prompt)
            ir_s = float(ir_model.score(prompt, best_img))
            all_clip.append(clip_s)
            all_ir.append(ir_s)

        results[f"weights_{weight_scheme}"] = {
            "clip_mean": float(np.mean(all_clip)),
            "clip_std": float(np.std(all_clip)),
            "ir_mean": float(np.mean(all_ir)),
            "ir_std": float(np.std(all_ir)),
        }

    return results


###############################################################################
# Experiment 3: CoPS on DrawBench
###############################################################################

def run_cops_drawbench(pipe):
    """Run CoPS with active resampling on DrawBench to fill Table 1 gap."""
    prompts = load_prompts("drawbench")
    clip_scorer = CLIPScorer(device="cuda")
    ir_model = load_imagereward()

    results = {"seeds": {}}

    for seed in [42, 123, 456]:
        print(f"Running CoPS on DrawBench, seed={seed}...")
        all_clip = []
        all_ir = []
        for i, prompt in enumerate(prompts):
            res = cops_sample_with_resampling(pipe, prompt, num_particles=4,
                num_inference_steps=50, resample_interval=10, seed=seed)
            best_img = tensor_to_pil(res["image"])

            clip_s = clip_scorer.score_single(best_img, prompt)
            ir_s = float(ir_model.score(prompt, best_img))
            all_clip.append(clip_s)
            all_ir.append(ir_s)

        results["seeds"][str(seed)] = {
            "clip_scores": all_clip,
            "ir_scores": all_ir,
            "clip_mean": float(np.mean(all_clip)),
            "ir_mean": float(np.mean(all_ir)),
        }

    # Aggregate
    all_means_clip = [results["seeds"][s]["clip_mean"] for s in ["42", "123", "456"]]
    all_means_ir = [results["seeds"][s]["ir_mean"] for s in ["42", "123", "456"]]
    results["clip_mean"] = float(np.mean(all_means_clip))
    results["clip_std"] = float(np.std(all_means_clip))
    results["ir_mean"] = float(np.mean(all_means_ir))
    results["ir_std"] = float(np.std(all_means_ir))

    return results


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    os.makedirs(EXP / "revision", exist_ok=True)

    print("=" * 60)
    print("REVISION EXPERIMENTS")
    print("=" * 60)

    t0 = time.time()
    pipe = load_sd15()

    # 1. LPIPS ablation on 100 COCO prompts
    print("\n[1/3] LPIPS Distance Metric Ablation (100 prompts)...")
    t1 = time.time()
    coco_100 = load_prompts("coco", 100)
    lpips_results = run_lpips_ablation(pipe, coco_100, seed=42)
    with open(EXP / "revision" / "lpips_ablation.json", "w") as f:
        json.dump(lpips_results, f, indent=2)

    # Compute summary stats for LPIPS ablation
    lpips_summary = {}
    for metric_key in ["pcs_l2", "pcs_lpips", "pcs_cosine"]:
        # For each prompt, select the particle with highest PCS under this metric
        all_clip = []
        all_ir = []
        for j in range(len(lpips_results["prompts"])):
            pcs_vals = lpips_results[metric_key][j]
            best_idx = int(np.argmax(pcs_vals))
            all_clip.append(lpips_results["clip_scores"][j][best_idx])
            all_ir.append(lpips_results["ir_scores"][j][best_idx])
        lpips_summary[metric_key] = {
            "clip_mean": float(np.mean(all_clip)),
            "clip_std": float(np.std(all_clip)),
            "ir_mean": float(np.mean(all_ir)),
            "ir_std": float(np.std(all_ir)),
        }

    # Also compute random baseline
    all_clip_rand = []
    all_ir_rand = []
    rng = np.random.RandomState(42)
    for j in range(len(lpips_results["prompts"])):
        rand_idx = rng.randint(0, 4)
        all_clip_rand.append(lpips_results["clip_scores"][j][rand_idx])
        all_ir_rand.append(lpips_results["ir_scores"][j][rand_idx])
    lpips_summary["random"] = {
        "clip_mean": float(np.mean(all_clip_rand)),
        "clip_std": float(np.std(all_clip_rand)),
        "ir_mean": float(np.mean(all_ir_rand)),
        "ir_std": float(np.std(all_ir_rand)),
    }

    # Compute Spearman correlations for LPIPS
    from scipy import stats
    for metric_key in ["pcs_l2", "pcs_lpips", "pcs_cosine"]:
        rhos_clip = []
        rhos_ir = []
        for j in range(len(lpips_results["prompts"])):
            pcs_vals = lpips_results[metric_key][j]
            clip_vals = lpips_results["clip_scores"][j]
            ir_vals = lpips_results["ir_scores"][j]
            if len(set(pcs_vals)) > 1:
                r_clip, _ = stats.spearmanr(pcs_vals, clip_vals)
                r_ir, _ = stats.spearmanr(pcs_vals, ir_vals)
                rhos_clip.append(r_clip)
                rhos_ir.append(r_ir)
        lpips_summary[metric_key]["spearman_clip"] = float(np.mean(rhos_clip))
        lpips_summary[metric_key]["spearman_clip_std"] = float(np.std(rhos_clip))
        lpips_summary[metric_key]["spearman_ir"] = float(np.mean(rhos_ir))
        lpips_summary[metric_key]["spearman_ir_std"] = float(np.std(rhos_ir))

    with open(EXP / "revision" / "lpips_summary.json", "w") as f:
        json.dump(lpips_summary, f, indent=2)
    print(f"  LPIPS ablation done in {time.time()-t1:.0f}s")
    print(f"  Summary: {json.dumps(lpips_summary, indent=2)}")

    # 2. CoPS on DrawBench
    print("\n[2/3] CoPS on DrawBench (41 prompts × 3 seeds)...")
    t2 = time.time()
    cops_db_results = run_cops_drawbench(pipe)
    with open(EXP / "revision" / "cops_drawbench.json", "w") as f:
        json.dump(cops_db_results, f, indent=2)
    print(f"  DrawBench done in {time.time()-t2:.0f}s")
    print(f"  CLIP: {cops_db_results['clip_mean']:.4f}±{cops_db_results['clip_std']:.4f}")
    print(f"  IR: {cops_db_results['ir_mean']:.4f}±{cops_db_results['ir_std']:.4f}")

    # 3. Re-score ablations with corrected ImageReward
    print("\n[3/3] Re-scoring ablations with corrected ImageReward...")
    t3 = time.time()
    ablation_results = rescore_ablations(pipe)
    with open(EXP / "revision" / "ablation_rescored.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"  Ablation rescoring done in {time.time()-t3:.0f}s")
    print(f"  Results: {json.dumps(ablation_results, indent=2)}")

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All revision experiments complete in {total_time/60:.1f} minutes")
    print(f"{'='*60}")
