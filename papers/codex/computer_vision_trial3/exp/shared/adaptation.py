from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from .common import (
    DOMAIN_ORDER_A,
    DOMAIN_ORDER_B,
    VOC_CLASSES,
    compute_confusion,
    ensure_dir,
    miou_from_confusion,
    mean_std_ci95,
    quantile,
    append_log,
    reset_path,
    save_json,
    set_seed,
    weak_augment_tensor,
)
from .proxy_benchmark import load_cached_regions, render_sample


class LogitAdapter(torch.nn.Module):
    def __init__(self, num_classes: int, rank: int, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.a = torch.nn.Parameter(torch.randn(num_classes, rank, generator=gen) * 0.01)
        self.b = torch.nn.Parameter(torch.randn(rank, num_classes, generator=gen) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.permute(0, 2, 3, 1)
        delta = x @ self.a @ self.b + self.bias
        return (x + delta).permute(0, 3, 1, 2)


def build_model(device: str = "cuda"):
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(device)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def _forward_logits(model, adapter, x: torch.Tensor, out_hw=(256, 256)) -> torch.Tensor:
    logits = adapter(model(x)["out"])
    if logits.shape[-2:] != out_hw:
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
    return logits


def _ovss_region_stats(logits: torch.Tensor, region_mask: np.ndarray) -> Dict:
    mask_t = torch.from_numpy(region_mask).to(logits.device)
    if mask_t.sum() == 0:
        return {}
    pixels = logits[0, :, mask_t]
    pooled = pixels.mean(dim=1)
    probs = pooled.softmax(dim=0)
    top_vals, top_idx = torch.topk(probs, k=min(3, probs.numel()))
    margin = float(top_vals[0] - top_vals[1]) if len(top_vals) > 1 else float(top_vals[0])
    entropy = float(-(probs * (probs.clamp_min(1e-8).log())).sum())
    return {
        "ovss_logits": pooled.detach().cpu().numpy(),
        "ovss_probs": probs.detach().cpu().numpy(),
        "top3": top_idx.detach().cpu().numpy().tolist(),
        "margin": margin,
        "entropy": entropy,
        "top1": int(top_idx[0]),
    }


def _agreement(pred_a: np.ndarray, pred_b: np.ndarray, mask: np.ndarray) -> float:
    region = mask.astype(bool)
    if region.sum() == 0:
        return 0.0
    return float((pred_a[region] == pred_b[region]).mean())


def calibrate(root: Path, splits: Dict, device: str = "cuda") -> Dict:
    model, preprocess = build_model(device=device)
    adapter = LogitAdapter(num_classes=len(VOC_CLASSES), rank=4, seed=13).to(device)
    cache_root = root / "data/proxy_cache"
    margins, entropies, agreements, accepted_area = [], [], [], []
    rejection = Counter()
    eligibility = Counter()
    with torch.no_grad():
        for spec in splits["calibration"]:
            image, _ = render_sample(spec)
            x = preprocess(image).unsqueeze(0).to(device)
            logits = _forward_logits(model, adapter, x)
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
            aug_logits = _forward_logits(model, adapter, weak_augment_tensor(x, seed=13))
            aug_pred = aug_logits.argmax(dim=1)[0].detach().cpu().numpy()
            regions = load_cached_regions(cache_root, spec["sample_id"], "proposal")
            accepted_local = []
            for region in regions:
                ovss = _ovss_region_stats(logits, region["mask"])
                if not ovss:
                    continue
                clip_probs = torch.from_numpy(region["clip_logits"]).softmax(dim=0).numpy()
                clip_top3 = np.argsort(-clip_probs)[:3].tolist()
                if set(ovss["top3"]).isdisjoint(clip_top3):
                    rejection["disjoint_top3"] += 1
                    continue
                union_idx = sorted(set(ovss["top3"]).union(clip_top3))
                fused = ovss["ovss_logits"][union_idx] + region["clip_logits"][union_idx]
                fused_probs_union = torch.from_numpy(fused).softmax(dim=0).numpy()
                top_pos = np.argsort(-fused_probs_union)[: min(3, len(union_idx))].tolist()
                top1 = union_idx[top_pos[0]]
                top2_prob = fused_probs_union[top_pos[1]] if len(top_pos) > 1 else 0.0
                margin = float(fused_probs_union[top_pos[0]] - top2_prob)
                entropy = float(-(fused_probs_union * np.log(np.clip(fused_probs_union, 1e-8, 1.0))).sum())
                agreement = _agreement(pred, aug_pred, region["mask"])
                if top1 not in ovss["top3"] or top1 not in clip_top3:
                    rejection["fused_not_in_both_top3"] += 1
                    continue
                eligibility["eligible_masks"] += 1
                margins.append(margin)
                entropies.append(entropy)
                agreements.append(agreement)
                accepted_local.append(region["area_ratio"])
            if accepted_local:
                accepted_area.append(float(np.median(accepted_local)))
    thresholds = {
        "tau_ovss": 1.0,
        "tau_clip": 1.0,
        "margin_threshold": quantile(margins, 0.60, 0.05),
        "entropy_threshold": quantile(entropies, 0.40, 2.0),
        "agreement_threshold": quantile(agreements, 0.50, 0.5),
        "budget_b": quantile(accepted_area, 0.50, 0.08),
        "rejection_counts": dict(rejection),
        "calibration_diagnostics": {
            "num_calibration_images": len(splits["calibration"]),
            "eligible_masks": int(eligibility["eligible_masks"]),
            "accepted_area_median_input_count": len(accepted_area),
            "margin_summary": mean_std_ci95(margins),
            "entropy_summary": mean_std_ci95(entropies),
            "agreement_summary": mean_std_ci95(agreements),
            "accepted_area_summary": mean_std_ci95(accepted_area),
        },
    }
    save_json(root / "exp/02_calibration/results.json", thresholds)
    return thresholds


def _select_pixels(logits: torch.Tensor, budget: float) -> Tuple[np.ndarray, np.ndarray]:
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    conf_np = conf[0].detach().cpu().numpy()
    pred_np = pred[0].detach().cpu().numpy()
    flat = conf_np.reshape(-1)
    k = max(1, int(budget * flat.size))
    thresh = np.partition(flat, -k)[-k]
    support = conf_np >= thresh
    return support, pred_np


def _build_region_support(
    logits: torch.Tensor,
    aug_pred: np.ndarray,
    regions: List[Dict],
    thresholds: Dict,
    method: str,
) -> Dict:
    items = []
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
    for region in regions:
        ovss = _ovss_region_stats(logits, region["mask"])
        if not ovss:
            continue
        clip_probs = torch.from_numpy(region["clip_logits"]).softmax(dim=0).numpy()
        clip_top3 = np.argsort(-clip_probs)[:3].tolist()
        if method in {"clip_verified", "slic"} and set(ovss["top3"]).isdisjoint(clip_top3):
            continue
        agreement = _agreement(pred, aug_pred, region["mask"])
        if method == "raw_mask":
            if ovss["margin"] < thresholds["margin_threshold"] or agreement < thresholds["agreement_threshold"]:
                continue
            label = ovss["top1"]
            score = ovss["margin"]
        elif method == "no_clip":
            if ovss["margin"] < thresholds["margin_threshold"]:
                continue
            label = ovss["top1"]
            score = ovss["margin"]
        else:
            union_idx = sorted(set(ovss["top3"]).union(clip_top3))
            fused = ovss["ovss_logits"][union_idx] / thresholds["tau_ovss"] + region["clip_logits"][union_idx] / thresholds["tau_clip"]
            fused_probs_union = torch.from_numpy(fused).softmax(dim=0).numpy()
            top_pos = np.argsort(-fused_probs_union)[: min(3, len(union_idx))].tolist()
            top1 = int(union_idx[top_pos[0]])
            top2_prob = fused_probs_union[top_pos[1]] if len(top_pos) > 1 else 0.0
            margin = float(fused_probs_union[top_pos[0]] - top2_prob)
            entropy = float(-(fused_probs_union * np.log(np.clip(fused_probs_union, 1e-8, 1.0))).sum())
            if top1 not in ovss["top3"] or top1 not in clip_top3:
                continue
            if margin < thresholds["margin_threshold"]:
                continue
            if entropy > thresholds["entropy_threshold"]:
                continue
            if agreement < thresholds["agreement_threshold"]:
                continue
            label = int(top1)
            score = margin
        items.append({"mask": region["mask"], "label": label, "score": score, "area_ratio": region["area_ratio"]})
    items.sort(key=lambda x: x["score"], reverse=True)
    support = np.zeros_like(pred, dtype=bool)
    labels = np.full_like(pred, fill_value=255, dtype=np.int64)
    accepted = 0
    area = 0.0
    for item in items:
        new_mask = item["mask"] & (~support)
        if not new_mask.any():
            continue
        candidate_ratio = new_mask.sum() / new_mask.size
        if area + candidate_ratio > thresholds["budget_b"]:
            continue
        support[new_mask] = True
        labels[new_mask] = item["label"]
        area += candidate_ratio
        accepted += 1
        if area >= thresholds["budget_b"]:
            break
    return {"support": support, "labels": labels, "accepted_count": accepted, "accepted_area": area}


def _run_stream(
    root: Path,
    run_name: str,
    splits: Dict,
    thresholds: Dict,
    order_name: str,
    seed: int,
    method: str,
    reduced: bool = False,
    margin_offset: float = 0.0,
    budget_scale: float = 1.0,
) -> Dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = build_model(device=device)
    adapter = LogitAdapter(num_classes=len(VOC_CLASSES), rank=4, seed=seed).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    cache_root = root / "data/proxy_cache"
    samples = splits["reduced"] if reduced else splits["main"]
    order = DOMAIN_ORDER_A if order_name == "A" else DOMAIN_ORDER_B
    ordered = []
    for domain in order:
        ordered.extend([s for s in samples if s["domain"] == domain])
    local_thr = dict(thresholds)
    local_thr["margin_threshold"] = max(0.0, thresholds["margin_threshold"] + margin_offset)
    local_thr["budget_b"] = thresholds["budget_b"] * budget_scale

    conf_total = np.zeros((len(VOC_CLASSES), len(VOC_CLASSES)), dtype=np.int64)
    by_domain = defaultdict(lambda: np.zeros_like(conf_total))
    per_image = []
    class_hist = Counter()
    skip_reasons = Counter()
    accept_hist = Counter()
    inference_times = []
    support_times = []
    update_times = []
    boundary_checkpoints = []
    prediction_ids = []
    prediction_stack = []
    exp_dir = root / (
        "exp/03_frozen"
        if method == "frozen" and not reduced
        else "exp/04_topb_pixel"
        if method == "topb_pixel" and not reduced
        else "exp/05_raw_mask"
        if method == "raw_mask" and not reduced
        else "exp/06_clip_verified"
        if method == "clip_verified" and not reduced
        else f"exp/07_ablations/{run_name}"
    )
    log_path = exp_dir / "logs" / f"{run_name}.log"
    pred_path = exp_dir / "predictions" / f"{run_name}.npz"
    ckpt_dir = exp_dir / "checkpoints"
    result_path = exp_dir / f"{run_name}.json"
    ensure_dir(log_path.parent)
    ensure_dir(pred_path.parent)
    ensure_dir(ckpt_dir)
    reset_path(log_path)
    reset_path(pred_path)
    reset_path(result_path)
    for stale_ckpt in ckpt_dir.glob(f"{run_name}_after_*.pt"):
        reset_path(stale_ckpt)
    torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None

    for sample_idx, spec in enumerate(ordered):
        image, target = render_sample(spec)
        x = preprocess(image).unsqueeze(0).to(device)
        start = time.perf_counter()
        with torch.no_grad():
            logits = _forward_logits(model, adapter, x)
        infer_time = time.perf_counter() - start
        inference_times.append(float(infer_time))
        pred = logits.argmax(dim=1)[0].detach().cpu().numpy()
        prediction_ids.append(spec["sample_id"])
        prediction_stack.append(pred.astype(np.uint8))
        conf_total += compute_confusion(pred, target, num_classes=len(VOC_CLASSES))
        by_domain[spec["domain"]] += compute_confusion(pred, target, num_classes=len(VOC_CLASSES))
        step_start = time.perf_counter()
        skip = False
        accepted_area = 0.0
        accepted_count = 0
        if method != "frozen":
            with torch.no_grad():
                aug_logits = _forward_logits(model, adapter, weak_augment_tensor(x, seed=seed + sample_idx + 1))
            aug_pred = aug_logits.argmax(dim=1)[0].detach().cpu().numpy()
            if method == "topb_pixel":
                support, labels = _select_pixels(logits, budget=local_thr["budget_b"])
                accepted_area = float(support.mean())
                accepted_count = int(support.any())
                support_times.append(float(time.perf_counter() - step_start))
            else:
                region_type = "slic" if method == "slic" else "proposal"
                regions = load_cached_regions(cache_root, spec["sample_id"], region_type)
                out = _build_region_support(logits, aug_pred, regions, local_thr, method)
                support, labels = out["support"], out["labels"]
                accepted_area = out["accepted_area"]
                accepted_count = out["accepted_count"]
                support_times.append(float(time.perf_counter() - step_start))
            if accepted_area <= 0.0:
                skip = True
                skip_reasons["no_support"] += 1
            else:
                update_begin = time.perf_counter()
                support_t = torch.from_numpy(support).to(device)
                label_t = torch.from_numpy(labels).to(device)
                optimizer.zero_grad(set_to_none=True)
                train_logits = _forward_logits(model, adapter, x)
                loss_pseudo = F.cross_entropy(train_logits, label_t.unsqueeze(0), ignore_index=255)
                aug_x = weak_augment_tensor(x, seed=seed + 1000 + sample_idx)
                aug_train_logits = _forward_logits(model, adapter, aug_x)
                p = train_logits.softmax(dim=1).detach()
                q = aug_train_logits.log_softmax(dim=1)
                mask = support_t.unsqueeze(0).unsqueeze(0)
                kl = F.kl_div(q, p, reduction="none").sum(dim=1, keepdim=True)
                loss_cons = (kl * mask).sum() / mask.sum().clamp_min(1.0)
                weight = 0.0 if method == "no_consistency" else 0.2
                loss = loss_pseudo + weight * loss_cons
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()
                update_times.append(float(time.perf_counter() - update_begin))
                for cls in np.unique(labels[support]):
                    class_hist[int(cls)] += int((labels[support] == cls).sum())
                    accept_hist[int(cls)] += 1
        else:
            support_times.append(0.0)
        runtime = time.perf_counter() - step_start + infer_time
        img_conf = compute_confusion(pred, target, num_classes=len(VOC_CLASSES))
        img_metrics_present = miou_from_confusion(img_conf, drop_absent=True)
        img_metrics_zero = miou_from_confusion(img_conf, drop_absent=False)
        per_image.append(
            {
                "sample_id": spec["sample_id"],
                "domain": spec["domain"],
                "miou": img_metrics_present["miou"],
                "miou_present_classes": img_metrics_present["miou"],
                "miou_zero_filled": img_metrics_zero["miou"],
                "present_foreground_classes": img_metrics_present["present_foreground_classes"],
                "accepted_area": accepted_area,
                "accepted_count": accepted_count,
                "skip_update": skip,
                "runtime_sec": runtime,
            }
        )
        append_log(
            log_path,
            (
                f"sample={spec['sample_id']} domain={spec['domain']} miou_present={img_metrics_present['miou']:.6f} "
                f"miou_zero_filled={img_metrics_zero['miou']:.6f} "
                f"accepted_area={accepted_area:.6f} accepted_count={accepted_count} "
                f"skip={int(skip)} runtime_sec={runtime:.6f}"
            ),
        )
        if sample_idx + 1 < len(ordered) and ordered[sample_idx + 1]["domain"] != spec["domain"]:
            ckpt_path = ckpt_dir / f"{run_name}_after_{spec['domain']}.pt"
            torch.save(adapter.state_dict(), ckpt_path)
            boundary_checkpoints.append(
                {
                    "after_domain": spec["domain"],
                    "checkpoint_path": str(ckpt_path),
                    "image_index": sample_idx,
                }
            )

    domain_metrics = {domain: miou_from_confusion(conf)["miou"] for domain, conf in by_domain.items()}
    ordered_domain_scores = []
    for domain in order:
        ordered_domain_scores.append(domain_metrics.get(domain, 0.0))
    transition_drops = []
    for prev_score, next_score in zip(ordered_domain_scores[:-1], ordered_domain_scores[1:]):
        transition_drops.append(float(prev_score - next_score))
    np.savez_compressed(pred_path, sample_ids=np.asarray(prediction_ids), predictions=np.stack(prediction_stack, axis=0))
    results = {
        "run_name": run_name,
        "method": method,
        "order": order_name,
        "seed": seed,
        "reduced": reduced,
        "metrics": miou_from_confusion(conf_total, drop_absent=False),
        "metric_protocol": {
            "aggregate_miou": "dataset-level confusion-matrix mIoU over 20 Pascal VOC foreground classes",
            "per_image_miou": "per-image mIoU averaged only over foreground classes present in that image",
            "per_image_zero_filled_miou": "diagnostic metric that includes absent foreground classes as zero and is not used for aggregate reporting",
        },
        "per_domain_miou": domain_metrics,
        "per_image": per_image,
        "skip_ratio": float(np.mean([x["skip_update"] for x in per_image])),
        "accepted_area_mean": float(np.mean([x["accepted_area"] for x in per_image])),
        "accepted_count_mean": float(np.mean([x["accepted_count"] for x in per_image])),
        "runtime_sec_per_image": float(np.mean([x["runtime_sec"] for x in per_image])),
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device == "cuda" else 0.0,
        "class_histogram": dict(class_hist),
        "accepted_label_histogram": dict(accept_hist),
        "skip_reasons": dict(skip_reasons),
        "avg_transition_drop": float(np.mean(transition_drops)) if transition_drops else 0.0,
        "transition_drops": transition_drops,
        "timing_breakdown_sec_per_image": {
            "inference": float(np.mean(inference_times)) if inference_times else 0.0,
            "support_selection": float(np.mean(support_times)) if support_times else 0.0,
            "update": float(np.mean(update_times)) if update_times else 0.0,
        },
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "adapter_rank": 4,
            "consistency_weight": 0.0 if method == "no_consistency" else 0.2,
            "update_steps_per_image": 1,
            "budget_b": local_thr["budget_b"],
            "margin_threshold": local_thr["margin_threshold"],
            "entropy_threshold": local_thr["entropy_threshold"],
            "agreement_threshold": local_thr["agreement_threshold"],
        },
        "prediction_path": str(pred_path),
        "boundary_checkpoints": boundary_checkpoints,
        "log_path": str(log_path),
        "result_path": str(result_path),
    }
    save_json(result_path, results)
    return results


def run_experiment_grid(root: Path, splits: Dict, thresholds: Dict) -> List[Dict]:
    runs = []
    seeds = [13, 17, 23]
    main_methods = [
        ("frozen", [13]),
        ("topb_pixel", seeds),
        ("raw_mask", seeds),
        ("clip_verified", seeds),
    ]
    for method, method_seeds in main_methods:
        for order in ["A", "B"]:
            for seed in method_seeds:
                run_name = f"{method}_{order}_seed{seed}"
                print(f"[run] {run_name}", flush=True)
                out = _run_stream(root, run_name, splits, thresholds, order, seed, method=method)
                runs.append(out)
    ablation_specs = [
        ("slic", 0.0, 1.0),
        ("no_clip", 0.0, 1.0),
        ("no_consistency", 0.0, 1.0),
    ]
    for method, margin_offset, budget_scale in ablation_specs:
        for seed in seeds:
            run_name = f"{method}_A_seed{seed}"
            print(f"[run] {run_name}", flush=True)
            out = _run_stream(root, run_name, splits, thresholds, "A", seed, method=method, reduced=True, margin_offset=margin_offset, budget_scale=budget_scale)
            runs.append(out)
    for offset in [-0.05, 0.0, 0.05]:
        for seed in seeds:
            run_name = f"threshold_{offset:+.2f}_seed{seed}"
            print(f"[run] {run_name}", flush=True)
            out = _run_stream(root, run_name, splits, thresholds, "A", seed, method="clip_verified", reduced=True, margin_offset=offset)
            runs.append(out)
    for scale in [0.8, 1.0, 1.2]:
        for method in ["topb_pixel", "clip_verified"]:
            for seed in seeds:
                run_name = f"{method}_budget_{scale:.1f}_seed{seed}"
                print(f"[run] {run_name}", flush=True)
                out = _run_stream(root, run_name, splits, thresholds, "A", seed, method=method, reduced=True, budget_scale=scale)
                runs.append(out)
    return runs
