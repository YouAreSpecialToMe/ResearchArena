from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACT_ROOT, CHECKPOINT_ROOT, FEATURE_ROOT, PAIR_ROOT, PROCESSED_ROOT, RESULT_ROOT, TRAINING
from .metrics import bootstrap_ci, compute_partition_metrics, realized_l0, variance_explained
from .models import LinearSAE, build_block_spec
from .utils import ensure_dir, read_json, select_device, set_global_seed, write_json


@dataclass
class RunConfig:
    dataset: str
    backbone: str
    method: str
    seed: int
    lambda_nuis: float = 0.0
    lambda_cf: float = 0.0
    ra_relax: float = 0.0
    margin: float = TRAINING["margin"]
    cache_tag: str = ""


def load_clean_features(dataset: str, backbone: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = FEATURE_ROOT / dataset / backbone / "all_clean.npy"
    all_features = np.load(path)
    split_payload = read_json(PROCESSED_ROOT / dataset / f"{split}_tuples.json")
    indices = np.asarray(split_payload["indices"], dtype=np.int64)
    return all_features[indices], indices


def choose_topk(input_dim: int) -> int:
    return 32 if input_dim <= 512 else 64


def load_pair_payload(dataset: str, split: str) -> dict[str, Any]:
    return read_json(PAIR_ROOT / dataset / split / "pairs.json")


def build_anchor_bank(train_x: np.ndarray, latent_dim: int) -> torch.Tensor:
    count = min(256, max(64, latent_dim // 4), len(train_x))
    choice = np.random.default_rng(0).choice(len(train_x), size=count, replace=False)
    anchors = torch.from_numpy(train_x[choice]).float()
    anchors = F.normalize(anchors, dim=1)
    return anchors


def evaluate_method(model: LinearSAE, dataset: str, backbone: str, split: str, factor_names: list[str], admissible_factors: list[str], pseudo_blocks: dict | None = None, permute: list[int] | None = None) -> dict[str, Any]:
    device = select_device()
    model.eval()
    clean_all = np.load(FEATURE_ROOT / dataset / backbone / "all_clean.npy")
    pair_payload = load_pair_payload(dataset, split)
    spec = build_block_spec(model.latent_dim, len(factor_names))
    if pseudo_blocks is not None:
        spec = dict_to_block_spec(pseudo_blocks)
    if permute is not None:
        factor_slices = [spec.factor_slices[i] for i in permute]
        spec = build_block_spec_from_slices(spec.inv_slice, factor_slices, spec.residual_slice)
    tfcc_by_factor = {}
    tba_by_factor = {}
    nuisance_values = []
    block_changes_by_factor = {}
    with torch.no_grad():
        for factor_idx, factor_name in enumerate(factor_names):
            pairs = pair_payload["factor_pairs"][factor_name]
            if not pairs:
                continue
            src = torch.from_numpy(clean_all[[p[0] for p in pairs]]).float().to(device)
            dst = torch.from_numpy(clean_all[[p[1] for p in pairs]]).float().to(device)
            z1 = model(src)["z"]
            z2 = model(dst)["z"]
            metrics = compute_partition_metrics(z1, z2, spec, factor_idx)
            tfcc_by_factor[factor_name] = metrics["tfcc"].cpu().numpy()
            tba_by_factor[factor_name] = metrics["tba"].cpu().numpy()
            block_changes_by_factor[factor_name] = {
                "inv": float(metrics["delta_inv"].mean().cpu()),
                "target": float(metrics["delta_factors"][:, factor_idx].mean().cpu()),
                "off_target_mean": float((metrics["delta_factors"].sum(dim=1) - metrics["delta_factors"][:, factor_idx]).mean().cpu() / max(1, len(spec.factor_slices) - 1)),
                "residual": float(metrics["delta_res"].mean().cpu()),
            }
        nuisance = pair_payload["nuisance_pairs"]
        if nuisance:
            src = torch.from_numpy(np.load(FEATURE_ROOT / dataset / backbone / f"{split}_nuisance_view1.npy")).float().to(device)
            dst = torch.from_numpy(np.load(FEATURE_ROOT / dataset / backbone / f"{split}_nuisance_view2.npy")).float().to(device)
            z1 = model(src)["z"]
            z2 = model(dst)["z"]
            nuisance_values = compute_partition_metrics(z1, z2, spec, 0)["delta_inv"].cpu().numpy()
    admissible = admissible_factors or factor_names
    all_tfcc = np.concatenate([tfcc_by_factor[f] for f in admissible if f in tfcc_by_factor])
    all_tba = np.concatenate([tba_by_factor[f] for f in admissible if f in tba_by_factor])
    ci_low, ci_high = bootstrap_ci(all_tfcc, TRAINING["bootstrap_samples"])
    return {
        "tfcc_mean": float(all_tfcc.mean()),
        "tfcc_ci_low": ci_low,
        "tfcc_ci_high": ci_high,
        "tba_mean": float(all_tba.mean()),
        "nuisance_inv_mean": float(np.mean(nuisance_values)) if len(nuisance_values) else 0.0,
        "tfcc_by_factor": {k: float(v.mean()) for k, v in tfcc_by_factor.items()},
        "tba_by_factor": {k: float(v.mean()) for k, v in tba_by_factor.items()},
        "block_changes_by_factor": block_changes_by_factor,
    }


def dict_to_block_spec(payload: dict[str, Any]):
    from .models import BlockSpec

    return BlockSpec(slice(*payload["inv_slice"]), [slice(*x) for x in payload["factor_slices"]], slice(*payload["residual_slice"]))


def build_block_spec_from_slices(inv_slice: slice, factor_slices: list[slice], residual_slice: slice):
    from .models import BlockSpec

    return BlockSpec(inv_slice, factor_slices, residual_slice)


def block_spec_to_json(spec) -> dict[str, Any]:
    return {
        "inv_slice": [spec.inv_slice.start, spec.inv_slice.stop],
        "factor_slices": [[sl.start, sl.stop] for sl in spec.factor_slices],
        "residual_slice": [spec.residual_slice.start, spec.residual_slice.stop],
    }


def construct_pseudo_blocks(model: LinearSAE, dataset: str, backbone: str, factor_names: list[str], factor_order: list[str]) -> dict[str, Any]:
    device = select_device()
    train_x, train_indices = load_clean_features(dataset, backbone, "train")
    index_lookup = {int(idx): pos for pos, idx in enumerate(train_indices.tolist())}
    pair_payload = load_pair_payload(dataset, "train")
    spec = build_block_spec(model.latent_dim, len(factor_names))
    factor_width = spec.factor_slices[0].stop - spec.factor_slices[0].start
    inv_width = spec.inv_slice.stop - spec.inv_slice.start
    with torch.no_grad():
        z = model(torch.from_numpy(train_x).float().to(device))["z"].cpu().numpy()
    assigned = set()
    factor_blocks = []
    score_cache = {}
    for factor_name in factor_names:
        pairs = pair_payload["factor_pairs"][factor_name]
        local_pairs = [(index_lookup[p[0]], index_lookup[p[1]]) for p in pairs if p[0] in index_lookup and p[1] in index_lookup]
        if not local_pairs:
            score_cache[factor_name] = np.zeros(model.latent_dim, dtype=np.float32)
            continue
        src = z[[p[0] for p in local_pairs]]
        dst = z[[p[1] for p in local_pairs]]
        delta = np.abs(src - dst)
        normalized = delta / (delta.sum(axis=1, keepdims=True) + 1e-8)
        score_cache[factor_name] = normalized.mean(axis=0)
    for factor_name in factor_order:
        scores = score_cache[factor_name].copy()
        order = np.argsort(scores)[::-1]
        chosen = []
        for unit in order:
            if int(unit) not in assigned:
                assigned.add(int(unit))
                chosen.append(int(unit))
            if len(chosen) == factor_width:
                break
        factor_blocks.append(sorted(chosen))
    src = np.load(FEATURE_ROOT / dataset / backbone / "train_nuisance_view1.npy")
    dst = np.load(FEATURE_ROOT / dataset / backbone / "train_nuisance_view2.npy")
    with torch.no_grad():
        z1 = model(torch.from_numpy(src).float().to(device))["z"].cpu().numpy()
        z2 = model(torch.from_numpy(dst).float().to(device))["z"].cpu().numpy()
    delta = np.abs(z1 - z2)
    normalized = delta / (delta.sum(axis=1, keepdims=True) + 1e-8)
    nuisance_scores = normalized.mean(axis=0)
    remaining = [i for i in range(model.latent_dim) if i not in assigned]
    inv_sorted = sorted(remaining, key=lambda i: nuisance_scores[i])
    inv_units = sorted(inv_sorted[:inv_width])
    remaining = [i for i in remaining if i not in set(inv_units)]
    factor_slices = []
    cursor = 0
    factor_to_units = []
    for units in factor_blocks:
        factor_to_units.append(units)
    payload = {
        "inv_units": inv_units,
        "factor_units": factor_to_units,
        "residual_units": remaining,
    }
    return payload


def construct_pseudo_blocks_from_pairs(
    model: LinearSAE,
    dataset: str,
    backbone: str,
    factor_names: list[str],
    factor_order: list[str],
    factor_pairs: dict[str, list[list[int]]],
    nuisance_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    device = select_device()
    train_x, train_indices = load_clean_features(dataset, backbone, "train")
    index_lookup = {int(idx): pos for pos, idx in enumerate(train_indices.tolist())}
    spec = build_block_spec(model.latent_dim, len(factor_names))
    factor_width = spec.factor_slices[0].stop - spec.factor_slices[0].start
    inv_width = spec.inv_slice.stop - spec.inv_slice.start
    with torch.no_grad():
        z = model(torch.from_numpy(train_x).float().to(device))["z"].cpu().numpy()
    assigned = set()
    factor_blocks = []
    score_cache = {}
    for factor_name in factor_names:
        pairs = factor_pairs[factor_name]
        if not pairs:
            score_cache[factor_name] = np.zeros(model.latent_dim, dtype=np.float32)
            continue
        local_pairs = [(index_lookup[p[0]], index_lookup[p[1]]) for p in pairs if p[0] in index_lookup and p[1] in index_lookup]
        if not local_pairs:
            score_cache[factor_name] = np.zeros(model.latent_dim, dtype=np.float32)
            continue
        src = z[[p[0] for p in local_pairs]]
        dst = z[[p[1] for p in local_pairs]]
        delta = np.abs(src - dst)
        normalized = delta / (delta.sum(axis=1, keepdims=True) + 1e-8)
        score_cache[factor_name] = normalized.mean(axis=0)
    for factor_name in factor_order:
        scores = score_cache[factor_name].copy()
        order = np.argsort(scores)[::-1]
        chosen = []
        for unit in order:
            if int(unit) not in assigned:
                assigned.add(int(unit))
                chosen.append(int(unit))
            if len(chosen) == factor_width:
                break
        factor_blocks.append(sorted(chosen))
    pair_indices = np.asarray([entry["index"] for entry in nuisance_pairs], dtype=np.int64)
    nuisance_view1 = np.load(FEATURE_ROOT / dataset / backbone / "train_nuisance_view1.npy")
    nuisance_view2 = np.load(FEATURE_ROOT / dataset / backbone / "train_nuisance_view2.npy")
    if len(pair_indices) < len(nuisance_view1):
        nuisance_view1 = nuisance_view1[: len(pair_indices)]
        nuisance_view2 = nuisance_view2[: len(pair_indices)]
    with torch.no_grad():
        z1 = model(torch.from_numpy(nuisance_view1).float().to(device))["z"].cpu().numpy()
        z2 = model(torch.from_numpy(nuisance_view2).float().to(device))["z"].cpu().numpy()
    delta = np.abs(z1 - z2)
    normalized = delta / (delta.sum(axis=1, keepdims=True) + 1e-8)
    nuisance_scores = normalized.mean(axis=0)
    remaining = [i for i in range(model.latent_dim) if i not in assigned]
    inv_sorted = sorted(remaining, key=lambda i: nuisance_scores[i])
    inv_units = sorted(inv_sorted[:inv_width])
    remaining = [i for i in remaining if i not in set(inv_units)]
    return {
        "inv_units": inv_units,
        "factor_units": factor_blocks,
        "residual_units": remaining,
    }


def unit_block_payload_to_spec(payload: dict[str, Any]) -> dict[str, Any]:
    block_order = payload["inv_units"] + [u for block in payload["factor_units"] for u in block] + payload["residual_units"]
    inv_len = len(payload["inv_units"])
    factor_width = len(payload["factor_units"][0])
    inv_slice = [0, inv_len]
    factor_slices = []
    cursor = inv_len
    for _ in payload["factor_units"]:
        factor_slices.append([cursor, cursor + factor_width])
        cursor += factor_width
    residual_slice = [cursor, len(block_order)]
    return {"unit_order": block_order, "inv_slice": inv_slice, "factor_slices": factor_slices, "residual_slice": residual_slice}


def reorder_latents(z: torch.Tensor, unit_order: list[int]) -> torch.Tensor:
    return z[:, unit_order]


def _sample_rows(features: np.ndarray, count: int, rng: np.random.Generator) -> torch.Tensor:
    choice = rng.integers(0, len(features), size=count)
    return torch.from_numpy(features[choice]).float()


def _sample_factor_pairs(clean_all: np.ndarray, pair_payload: dict[str, Any], factor_names: list[str], total_count: int, rng: np.random.Generator) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    per_factor = max(1, total_count // max(1, len(factor_names)))
    outputs = []
    for factor_name in factor_names:
        pairs = pair_payload["factor_pairs"][factor_name]
        if not pairs:
            continue
        choice = rng.integers(0, len(pairs), size=per_factor)
        sampled = [pairs[idx] for idx in choice]
        src = torch.from_numpy(clean_all[[p[0] for p in sampled]]).float()
        dst = torch.from_numpy(clean_all[[p[1] for p in sampled]]).float()
        outputs.append((factor_name, src, dst))
    return outputs


def _sample_nuisance_pairs(view1: np.ndarray, view2: np.ndarray, count: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    choice = rng.integers(0, len(view1), size=count)
    return torch.from_numpy(view1[choice]).float(), torch.from_numpy(view2[choice]).float()


def _validation_objective(model: LinearSAE, val_x: np.ndarray, config: RunConfig, device: torch.device, native_spec) -> tuple[float, dict[str, Any]]:
    with torch.no_grad():
        val_tensor = torch.from_numpy(val_x).float().to(device)
        val_out = model(val_tensor)
        mse = F.mse_loss(val_out["recon"], val_tensor).item()
        sparsity = 1e-4 * val_out["z"].mean().item()
        objective = mse + sparsity
        diagnostics = {
            "val_reconstruction_mse": float(mse),
            "val_sparsity_penalty": float(sparsity),
            "val_objective": float(objective),
            "val_variance_explained": float(variance_explained(val_x, val_out["recon"].cpu().numpy())),
        }
        if config.method == "ra_sae":
            weights = torch.softmax(model.anchor_logits, dim=1)
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean().item()
            diagnostics["val_ra_entropy"] = float(entropy)
        if config.method == "orbit_sae":
            nuisance_v1 = np.load(FEATURE_ROOT / config.dataset / config.backbone / "val_nuisance_view1.npy")
            nuisance_v2 = np.load(FEATURE_ROOT / config.dataset / config.backbone / "val_nuisance_view2.npy")
            z1 = model(torch.from_numpy(nuisance_v1).float().to(device))["z"]
            z2 = model(torch.from_numpy(nuisance_v2).float().to(device))["z"]
            diagnostics["val_nuisance_penalty"] = float(F.mse_loss(z1, z2).item())
        if config.method == "fb_osae":
            nuisance_v1 = np.load(FEATURE_ROOT / config.dataset / config.backbone / "val_nuisance_view1.npy")
            nuisance_v2 = np.load(FEATURE_ROOT / config.dataset / config.backbone / "val_nuisance_view2.npy")
            z1 = model(torch.from_numpy(nuisance_v1).float().to(device))["z"]
            z2 = model(torch.from_numpy(nuisance_v2).float().to(device))["z"]
            diagnostics["val_inv_nuisance_penalty"] = float(F.mse_loss(z1[:, native_spec.inv_slice], z2[:, native_spec.inv_slice]).item())
    return float(-objective), diagnostics


def train_sae(config: RunConfig, factor_names: list[str], admissible_factors: list[str], factor_order: list[str]) -> dict[str, Any]:
    if config.cache_tag:
        cache_root = ARTIFACT_ROOT / "pilot_cache" / config.cache_tag
        result_path = cache_root / "results.json"
        checkpoint_root = cache_root / "checkpoints"
    else:
        result_path = RESULT_ROOT / config.dataset / config.backbone / config.method / str(config.seed) / "results.json"
        checkpoint_root = CHECKPOINT_ROOT / config.dataset / config.backbone / config.method / str(config.seed)
    if result_path.exists():
        return read_json(result_path)
    set_global_seed(config.seed)
    train_x, _ = load_clean_features(config.dataset, config.backbone, "train")
    val_x, _ = load_clean_features(config.dataset, config.backbone, "val")
    input_dim = train_x.shape[1]
    latent_dim = input_dim * TRAINING["latent_multiplier"]
    topk = choose_topk(input_dim)
    anchor_bank = build_anchor_bank(train_x, latent_dim) if config.method == "ra_sae" else None
    model = LinearSAE(input_dim, latent_dim, topk, config.method, anchor_bank=anchor_bank)
    device = select_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x).float()), batch_size=TRAINING["batch_size"], shuffle=True)
    pair_payload = load_pair_payload(config.dataset, "train")
    clean_all = np.load(FEATURE_ROOT / config.dataset / config.backbone / "all_clean.npy")
    nuisance_view1 = np.load(FEATURE_ROOT / config.dataset / config.backbone / "train_nuisance_view1.npy")
    nuisance_view2 = np.load(FEATURE_ROOT / config.dataset / config.backbone / "train_nuisance_view2.npy")
    factor_to_idx = {name: idx for idx, name in enumerate(factor_names)}
    rng = np.random.default_rng(config.seed)
    best_state = None
    best_metric = -1e9
    best_epoch = -1
    patience = 0
    start = time.time()
    native_spec = build_block_spec(latent_dim, len(factor_names))
    history = []
    for epoch in range(TRAINING["max_epochs"]):
        model.train()
        epoch_losses = []
        train_block_changes = []
        for (batch,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            singleton_count = int(TRAINING["batch_size"] * TRAINING["singleton_fraction"])
            pair_count = int(TRAINING["batch_size"] * TRAINING["counterfactual_fraction"])
            nuisance_count = int(TRAINING["batch_size"] * TRAINING["nuisance_fraction"])
            singleton_batch = _sample_rows(train_x, singleton_count, rng).to(device)
            out = model(singleton_batch)
            loss = F.mse_loss(out["recon"], singleton_batch) + 1e-4 * out["z"].mean()
            if config.method in {"orbit_sae", "fb_osae"} and nuisance_count > 0:
                src, dst = _sample_nuisance_pairs(nuisance_view1, nuisance_view2, nuisance_count, rng)
                src = src.to(device)
                dst = dst.to(device)
                z1 = model(src)["z"]
                z2 = model(dst)["z"]
                if config.method == "orbit_sae":
                    loss = loss + config.lambda_nuis * F.mse_loss(z1, z2)
                else:
                    loss = loss + config.lambda_nuis * F.mse_loss(z1[:, native_spec.inv_slice], z2[:, native_spec.inv_slice])
            if config.method in {"fb_sae", "fb_osae"} and pair_count > 0:
                cf_loss = torch.tensor(0.0, device=device)
                sampled_pairs = _sample_factor_pairs(clean_all, pair_payload, factor_names, pair_count, rng)
                for factor_name, src, dst in sampled_pairs:
                    src = src.to(device)
                    dst = dst.to(device)
                    z1 = model(src)["z"]
                    z2 = model(dst)["z"]
                    factor_idx = factor_to_idx[factor_name]
                    metrics = compute_partition_metrics(z1, z2, native_spec, factor_idx)
                    target_delta = metrics["delta_factors"][:, factor_idx]
                    off_target = metrics["delta_inv"] + metrics["delta_res"]
                    off_target = off_target + metrics["delta_factors"].sum(dim=1) - target_delta
                    cf_loss = cf_loss + (off_target.mean() + F.relu(config.margin - target_delta).mean())
                    train_block_changes.append(
                        {
                            "factor": factor_name,
                            "inv": float(metrics["delta_inv"].mean().detach().cpu()),
                            "target": float(target_delta.mean().detach().cpu()),
                            "off_target_mean": float(((metrics["delta_factors"].sum(dim=1) - target_delta) / max(1, len(factor_names) - 1)).mean().detach().cpu()),
                            "residual": float(metrics["delta_res"].mean().detach().cpu()),
                        }
                    )
                loss = loss + config.lambda_cf * (cf_loss / max(1, len(sampled_pairs)))
            if config.method == "ra_sae":
                weights = torch.softmax(model.anchor_logits, dim=1)
                entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
                loss = loss + config.ra_relax * entropy
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_metric, val_objective = _validation_objective(model, val_x, config, device, native_spec)
            val_eval = None
            if config.method in {"fb_sae", "fb_osae"}:
                val_eval = evaluate_method(model, config.dataset, config.backbone, "val", factor_names, admissible_factors)
                val_metric = val_eval["tfcc_mean"]
            train_diag_df = pd.DataFrame(train_block_changes) if train_block_changes else pd.DataFrame(columns=["factor", "inv", "target", "off_target_mean", "residual"])
            train_diag = {
                row["factor"]: {
                    "inv": float(row["inv"]),
                    "target": float(row["target"]),
                    "off_target_mean": float(row["off_target_mean"]),
                    "residual": float(row["residual"]),
                }
                for row in train_diag_df.groupby("factor", as_index=False).mean().to_dict("records")
            }
            history_row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "val_metric": float(val_metric),
                **val_objective,
                "train_block_changes_by_factor": train_diag,
                "val_tfcc_mean": None if val_eval is None else float(val_eval["tfcc_mean"]),
                "val_tba_mean": None if val_eval is None else float(val_eval["tba_mean"]),
                "val_block_changes_by_factor": None if val_eval is None else val_eval["block_changes_by_factor"],
            }
            history.append(history_row)
            if val_metric > best_metric:
                best_metric = val_metric
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= TRAINING["patience"]:
                break
    model.load_state_dict(best_state)
    checkpoint_dir = ensure_dir(checkpoint_root)
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__}, checkpoint_dir / "model.pt")
    write_json(checkpoint_dir / "history.json", history)
    train_out = model(torch.from_numpy(train_x).float().to(device))
    train_recon = train_out["recon"].detach().cpu().numpy()
    selection_metrics = evaluate_method(model, config.dataset, config.backbone, "val", factor_names, admissible_factors)
    test_eval = evaluate_method(model, config.dataset, config.backbone, "test", factor_names, admissible_factors)
    pseudo_spec = None
    if config.method in {"topk_sae", "ra_sae", "orbit_sae"}:
        pseudo_payload = construct_pseudo_blocks(model, config.dataset, config.backbone, factor_names, factor_order)
        pseudo_spec = unit_block_payload_to_spec(pseudo_payload)
        write_json(RESULT_ROOT / config.dataset / config.backbone / config.method / str(config.seed) / "pseudo_blocks.json", pseudo_spec)
        original_forward = model.forward

        def reordered_forward(x):
            output = original_forward(x)
            output["z"] = reorder_latents(output["z"], pseudo_spec["unit_order"])
            return output

        model.forward = reordered_forward
        selection_metrics = evaluate_method(model, config.dataset, config.backbone, "val", factor_names, admissible_factors, pseudo_blocks=pseudo_spec)
        test_eval = evaluate_method(model, config.dataset, config.backbone, "test", factor_names, admissible_factors, pseudo_blocks=pseudo_spec)
    runtime_minutes = (time.time() - start) / 60.0
    results = {
        "experiment": config.method,
        "dataset": config.dataset,
        "backbone": config.backbone,
        "seed": config.seed,
        "config": config.__dict__,
        "best_epoch": best_epoch,
        "runtime_minutes": runtime_minutes,
        "selection_metrics": {
            "val_tfcc": selection_metrics["tfcc_mean"],
            "val_tba": selection_metrics["tba_mean"],
            "val_nuisance_inv": selection_metrics["nuisance_inv_mean"],
            "val_variance_explained": variance_explained(val_x, model(torch.from_numpy(val_x).float().to(device))["recon"].detach().cpu().numpy()),
            "val_tfcc_by_factor": selection_metrics["tfcc_by_factor"],
            "val_tba_by_factor": selection_metrics["tba_by_factor"],
            "val_block_changes_by_factor": selection_metrics["block_changes_by_factor"],
        },
        "metrics": {
            "tfcc": test_eval["tfcc_mean"],
            "tba": test_eval["tba_mean"],
            "nuisance_inv": test_eval["nuisance_inv_mean"],
            "variance_explained": variance_explained(train_x, train_recon),
            "mean_l0": realized_l0(train_out["z"].detach().cpu().numpy()),
            "reconstruction_mse": float(np.mean((train_x - train_recon) ** 2)),
            "tfcc_ci": [test_eval["tfcc_ci_low"], test_eval["tfcc_ci_high"]],
            "tfcc_by_factor": test_eval["tfcc_by_factor"],
            "tba_by_factor": test_eval["tba_by_factor"],
        },
    }
    write_json(result_path, results)
    if not config.cache_tag:
        write_json(
            Path("exp") / config.method / f"results_{config.dataset}_{config.backbone}_s{config.seed}.json",
            results,
        )
        write_json(Path("exp") / config.method / "logs" / f"{config.dataset}_{config.backbone}_s{config.seed}_history.json", history)
    return results
