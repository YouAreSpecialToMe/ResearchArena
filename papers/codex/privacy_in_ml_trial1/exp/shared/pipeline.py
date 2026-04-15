from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from robustbench.data import load_cifar10c
from robustbench.utils import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import (
    ADAPTATION_CHUNK_SIZE,
    ARTIFACTS_DIR,
    AUDIT_TIMES,
    BATCH_SIZE,
    BOOTSTRAP_SAMPLES,
    CONF_BINS,
    CORRUPTIONS,
    DATA_DIR,
    FIGURES_DIR,
    HORIZONS,
    LOW_FPR_MIN_NEGATIVES,
    MATCHED_NEGATIVES_PER_POSITIVE,
    MATCH_TIME_BUCKETS,
    METHODS,
    NN_K,
    PIPELINE_VERSION,
    QUERY_BATCH_SIZE,
    ROOT,
    SEEDS,
    SEVERITY,
    SOURCE_MODEL_NAME,
    SOURCE_THREAT_MODEL,
    STREAM_LENGTH,
    T3A_ACCEPTANCE_THRESHOLD,
    T3A_SUPPORTS_PER_CLASS,
    TRAIN_FRAC,
    UNMATCHED_NEGATIVES_PER_POSITIVE,
    VAL_FRAC,
)


FEATURES_BLACK_BOX = ["max_prob", "entropy", "margin", "gini", "log_prob_norm"]
MAIN_POOL_TIERS = ["unmatched", "matched", "nn_matched", "far_past"]
ABLATION_NAMES = [
    "remove_true_class_matching",
    "remove_time_bucket_matching",
    "remove_source_pred_matching",
    "remove_confidence_matching",
    "remove_nn_matching",
]
ABLATION_HORIZONS = {8, 32}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def ensure_layout() -> None:
    for path in [DATA_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for name in ["environment_smoke", "core_matrix", "ablation", "visualization", "shared"]:
        (ROOT / "exp" / name / "logs").mkdir(parents=True, exist_ok=True)


def stage_dir(name: str) -> Path:
    path = ROOT / "exp" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def log_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), "event": event, **json_ready(payload)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def gini_np(probs: np.ndarray) -> np.ndarray:
    return (probs**2).sum(axis=1)


def feature_metrics(probs: np.ndarray) -> dict[str, np.ndarray]:
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    return {
        "max_prob": probs.max(axis=1),
        "entropy": entropy_np(probs),
        "margin": sorted_probs[:, 0] - sorted_probs[:, 1],
        "gini": gini_np(probs),
        "log_prob_norm": np.linalg.norm(np.log(np.clip(probs, 1e-12, 1.0)), axis=1),
    }


class WRNWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        out = m.conv1(x)
        out = m.block1(out)
        out = m.block2(out)
        out = m.block3(out)
        out = m.relu(m.bn1(out))
        out = F.avg_pool2d(out, 8)
        return out.view(-1, m.nChannels)

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.fc(features)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.forward_logits_from_features(features)
        if return_features:
            return logits, features
        return logits


def load_source_model(device: torch.device) -> WRNWrapper:
    model = load_model(
        model_name=SOURCE_MODEL_NAME,
        model_dir=str(ARTIFACTS_DIR / "models"),
        dataset="cifar10",
        threat_model=SOURCE_THREAT_MODEL,
    )
    wrapper = WRNWrapper(model).to(device)
    wrapper.eval()
    return wrapper


def load_corruption(corruption: str) -> tuple[np.ndarray, np.ndarray]:
    x, y = load_cifar10c(
        n_examples=10000,
        severity=SEVERITY,
        data_dir=str(DATA_DIR),
        shuffle=False,
        corruptions=[corruption],
    )
    return x.numpy(), y.numpy()


def compute_stream_positions(stream_indices: np.ndarray) -> np.ndarray:
    positions = np.full(10000, -1, dtype=np.int32)
    positions[stream_indices] = np.arange(1, len(stream_indices) + 1)
    return positions


def time_bucket_from_positions(stream_positions: np.ndarray) -> np.ndarray:
    bucket_size = max(1, math.ceil(STREAM_LENGTH / MATCH_TIME_BUCKETS))
    bucket = np.full_like(stream_positions, MATCH_TIME_BUCKETS, dtype=np.int32)
    seen = stream_positions > 0
    bucket[seen] = np.minimum((stream_positions[seen] - 1) // bucket_size, MATCH_TIME_BUCKETS - 1)
    return bucket


def batched_indices(indices: np.ndarray, batch_size: int = QUERY_BATCH_SIZE) -> list[np.ndarray]:
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


@dataclass
class SourceCache:
    x: np.ndarray
    y: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    conf_bin: np.ndarray
    features: np.ndarray
    feature_normed: np.ndarray


def get_source_cache(corruption: str, device: torch.device) -> SourceCache:
    cache_path = ARTIFACTS_DIR / "source_cache" / f"{corruption}_severity{SEVERITY}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return SourceCache(
            x=data["x"],
            y=data["y"],
            probs=data["probs"],
            preds=data["preds"],
            entropy=data["entropy"],
            margin=data["margin"],
            conf_bin=data["conf_bin"],
            features=data["features"],
            feature_normed=data["feature_normed"],
        )

    model = load_source_model(device)
    x, y = load_corruption(corruption)
    loader = torch.utils.data.DataLoader(torch.from_numpy(x), batch_size=QUERY_BATCH_SIZE, shuffle=False, num_workers=0)
    all_probs: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, features = model(batch, return_features=True)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())
    probs = np.concatenate(all_probs)
    features = np.concatenate(all_features)
    feature_normed = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
    preds = probs.argmax(axis=1)
    entropy = entropy_np(probs)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    conf = probs.max(axis=1)
    bins = np.quantile(conf, [1 / CONF_BINS, 2 / CONF_BINS])
    conf_bin = np.digitize(conf, bins, right=False).astype(np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=x,
        y=y,
        probs=probs,
        preds=preds,
        entropy=entropy,
        margin=margin,
        conf_bin=conf_bin,
        features=features,
        feature_normed=feature_normed,
    )
    return SourceCache(
        x=x,
        y=y,
        probs=probs,
        preds=preds,
        entropy=entropy,
        margin=margin,
        conf_bin=conf_bin,
        features=features,
        feature_normed=feature_normed,
    )


class SourceAdaptor:
    name = "source_only"
    uses_acceptance_matching = False

    def __init__(self, model: WRNWrapper, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict_indices(self, x_all: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs_list: list[np.ndarray] = []
        features_list: list[np.ndarray] = []
        for chunk in batched_indices(indices):
            x = torch.from_numpy(x_all[chunk]).to(self.device)
            logits, features = self.model(x, return_features=True)
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            features_list.append(features.cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
        features = np.concatenate(features_list, axis=0)
        return probs, features, np.ones(len(indices), dtype=np.int32)

    def adapt_batch(self, batch: torch.Tensor) -> dict[str, float]:
        return {"accepted_count": float(batch.shape[0]), "acceptance_rate": 1.0}


class TentAdaptor:
    name = "tent"
    uses_acceptance_matching = False

    def __init__(self, model: WRNWrapper, device: torch.device):
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.model.train()
        self.params: list[torch.nn.Parameter] = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                self.params.extend([module.weight, module.bias])
            else:
                module.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.params, lr=1e-3)

    @torch.no_grad()
    def predict_indices(self, x_all: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.train()
        probs_list: list[np.ndarray] = []
        features_list: list[np.ndarray] = []
        for chunk in batched_indices(indices):
            x = torch.from_numpy(x_all[chunk]).to(self.device)
            logits, features = self.model(x, return_features=True)
            probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
            features_list.append(features.cpu().numpy())
        probs = np.concatenate(probs_list, axis=0)
        features = np.concatenate(features_list, axis=0)
        return probs, features, np.ones(len(indices), dtype=np.int32)

    def adapt_batch(self, batch: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        loss = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        loss.backward()
        self.optimizer.step()
        return {"accepted_count": float(batch.shape[0]), "acceptance_rate": 1.0}


class T3AAdaptor:
    name = "t3a"
    uses_acceptance_matching = True

    def __init__(self, model: WRNWrapper, device: torch.device):
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()
        fc_weight = self.model.model.fc.weight.detach()
        self.num_classes = fc_weight.shape[0]
        self.initial_supports = F.normalize(fc_weight, dim=1)
        self.initial_labels = torch.arange(self.num_classes, device=device)
        self.supports = self.initial_supports.clone()
        self.labels = self.initial_labels.clone()
        self.entropies = torch.zeros(self.num_classes, device=device) - 1e6

    def _classifier_weights(self) -> torch.Tensor:
        class_weights = []
        for cls in range(self.num_classes):
            mask = self.labels == cls
            supports = self.supports[mask]
            if supports.numel() == 0:
                supports = self.initial_supports[cls : cls + 1]
            class_weights.append(F.normalize(supports.mean(dim=0, keepdim=True), dim=1))
        return torch.cat(class_weights, dim=0)

    @torch.no_grad()
    def predict_indices(self, x_all: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs_list: list[np.ndarray] = []
        features_list: list[np.ndarray] = []
        acceptance_list: list[np.ndarray] = []
        weights = self._classifier_weights()
        for chunk in batched_indices(indices):
            x = torch.from_numpy(x_all[chunk]).to(self.device)
            _, features = self.model(x, return_features=True)
            features = F.normalize(features, dim=1)
            logits = features @ weights.T
            probs = F.softmax(logits, dim=1)
            acceptance = (probs.max(dim=1).values >= T3A_ACCEPTANCE_THRESHOLD).int()
            probs_list.append(probs.cpu().numpy())
            features_list.append(features.cpu().numpy())
            acceptance_list.append(acceptance.cpu().numpy())
        return (
            np.concatenate(probs_list, axis=0),
            np.concatenate(features_list, axis=0),
            np.concatenate(acceptance_list, axis=0),
        )

    @torch.no_grad()
    def adapt_batch(self, batch: torch.Tensor) -> dict[str, float]:
        _, features = self.model(batch, return_features=True)
        features = F.normalize(features, dim=1)
        weights = self._classifier_weights()
        logits = features @ weights.T
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
        mask = probs.max(dim=1).values >= T3A_ACCEPTANCE_THRESHOLD
        if mask.any():
            self.supports = torch.cat([self.supports, features[mask]], dim=0)
            self.labels = torch.cat([self.labels, pred[mask]], dim=0)
            self.entropies = torch.cat([self.entropies, ent[mask]], dim=0)
        keep_supports = []
        keep_labels = []
        keep_entropy = []
        for cls in range(self.num_classes):
            cls_mask = self.labels == cls
            cls_supports = self.supports[cls_mask]
            cls_entropy = self.entropies[cls_mask]
            cls_labels = self.labels[cls_mask]
            order = torch.argsort(cls_entropy)[:T3A_SUPPORTS_PER_CLASS]
            keep_supports.append(cls_supports[order])
            keep_labels.append(cls_labels[order])
            keep_entropy.append(cls_entropy[order])
        self.supports = torch.cat(keep_supports, dim=0)
        self.labels = torch.cat(keep_labels, dim=0)
        self.entropies = torch.cat(keep_entropy, dim=0)
        accepted_count = float(mask.sum().item())
        return {
            "accepted_count": accepted_count,
            "acceptance_rate": accepted_count / max(1, batch.shape[0]),
            "support_count": float(self.supports.shape[0]),
        }


def build_adaptor(method: str, model: WRNWrapper, device: torch.device):
    if method == "source_only":
        return SourceAdaptor(model, device)
    if method == "tent":
        return TentAdaptor(model, device)
    if method == "t3a":
        return T3AAdaptor(model, device)
    raise ValueError(method)


def control_mask(
    *,
    base_mask: np.ndarray,
    pos_idx: int,
    true_class: np.ndarray,
    pred_class: np.ndarray,
    conf_bin: np.ndarray,
    time_bucket: np.ndarray,
    remove_true_class: bool = False,
    remove_source_pred: bool = False,
    remove_confidence: bool = False,
    remove_time_bucket: bool = False,
) -> np.ndarray:
    mask = base_mask.copy()
    if not remove_true_class:
        mask &= true_class == true_class[pos_idx]
    if not remove_source_pred:
        mask &= pred_class == pred_class[pos_idx]
    if not remove_confidence:
        mask &= conf_bin == conf_bin[pos_idx]
    if not remove_time_bucket:
        mask &= time_bucket == time_bucket[pos_idx]
    return mask


def apply_acceptance_matching(
    candidates: np.ndarray,
    pos_idx: int,
    acceptance_by_id: dict[int, int],
    enabled: bool,
) -> np.ndarray:
    if not enabled or len(candidates) == 0:
        return candidates
    target = acceptance_by_id[pos_idx]
    kept = [idx for idx in candidates.tolist() if acceptance_by_id.get(int(idx), -1) == target]
    return np.array(kept, dtype=np.int64)


def select_nn(source: SourceCache, pos_idx: int, candidates: np.ndarray, k: int = NN_K) -> np.ndarray:
    if len(candidates) == 0:
        return np.array([], dtype=np.int64)
    sims = source.feature_normed[candidates] @ source.feature_normed[pos_idx]
    order = np.argsort(-sims)
    return candidates[order[:k]]


def sample_controls(candidates: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(candidates) <= limit:
        return candidates
    return np.sort(rng.choice(candidates, size=limit, replace=False))


def build_candidate_specs(
    *,
    pos_idx: int,
    non_recent_mask: np.ndarray,
    far_past_mask: np.ndarray,
    source: SourceCache,
    time_bucket: np.ndarray,
    rng: np.random.Generator,
    horizon: int,
    method: str,
) -> dict[str, np.ndarray]:
    true_class = source.y
    pred_class = source.preds
    conf_bin = source.conf_bin

    unmatched_pool = np.flatnonzero(non_recent_mask)
    unmatched = sample_controls(unmatched_pool, UNMATCHED_NEGATIVES_PER_POSITIVE, rng)

    matched = np.flatnonzero(
        control_mask(
            base_mask=non_recent_mask,
            pos_idx=pos_idx,
            true_class=true_class,
            pred_class=pred_class,
            conf_bin=conf_bin,
            time_bucket=time_bucket,
        )
    )

    far_past = np.flatnonzero(
        control_mask(
            base_mask=far_past_mask,
            pos_idx=pos_idx,
            true_class=true_class,
            pred_class=pred_class,
            conf_bin=conf_bin,
            time_bucket=time_bucket,
        )
    )

    specs: dict[str, np.ndarray] = {
        "unmatched": unmatched,
        "matched": sample_controls(matched, MATCHED_NEGATIVES_PER_POSITIVE, rng),
        "nn_matched": matched.copy(),
        "far_past": sample_controls(far_past, MATCHED_NEGATIVES_PER_POSITIVE, rng),
    }

    if method != "source_only" and horizon in ABLATION_HORIZONS:
        ablation_controls = {
            "remove_true_class_matching": dict(remove_true_class=True),
            "remove_time_bucket_matching": dict(remove_time_bucket=True),
            "remove_source_pred_matching": dict(remove_source_pred=True),
            "remove_confidence_matching": dict(remove_confidence=True),
            "remove_nn_matching": dict(),
        }
        for name, removals in ablation_controls.items():
            candidates = np.flatnonzero(
                control_mask(
                    base_mask=non_recent_mask,
                    pos_idx=pos_idx,
                    true_class=true_class,
                    pred_class=pred_class,
                    conf_bin=conf_bin,
                    time_bucket=time_bucket,
                    **removals,
                )
            )
            if name == "remove_nn_matching":
                candidates = sample_controls(candidates, MATCHED_NEGATIVES_PER_POSITIVE, rng)
            specs[f"ablation::{name}"] = candidates
    return specs


def materialize_query_rows(
    *,
    source: SourceCache,
    seed: int,
    corruption: str,
    method: str,
    audit_time: int,
    horizon: int,
    stream_indices: np.ndarray,
    positions: np.ndarray,
    time_bucket: np.ndarray,
    adaptor,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    recent_positions = np.arange(audit_time - horizon, audit_time) + 1
    recent_ids = stream_indices[recent_positions - 1]
    recent_mask = np.zeros(10000, dtype=bool)
    recent_mask[recent_ids] = True
    non_recent_mask = ~recent_mask
    far_past_cutoff = max(0, audit_time - 4 * horizon)
    far_past_mask = (positions > 0) & (positions <= far_past_cutoff) & non_recent_mask

    query_index_set = set(recent_ids.tolist())
    candidate_specs_by_positive: dict[int, dict[str, np.ndarray]] = {}
    for pos_idx in recent_ids:
        specs = build_candidate_specs(
            pos_idx=int(pos_idx),
            non_recent_mask=non_recent_mask,
            far_past_mask=far_past_mask,
            source=source,
            time_bucket=time_bucket,
            rng=rng,
            horizon=horizon,
            method=method,
        )
        candidate_specs_by_positive[int(pos_idx)] = specs
        for candidate_ids in specs.values():
            query_index_set.update(int(x) for x in candidate_ids.tolist())

    query_indices = np.array(sorted(query_index_set), dtype=np.int64)
    adapted_probs, _, acceptance = adaptor.predict_indices(source.x, query_indices)
    adapted_lookup = {idx: i for i, idx in enumerate(query_indices.tolist())}
    acceptance_by_id = {idx: int(acceptance[i]) for i, idx in enumerate(query_indices.tolist())}
    adapted_metrics = feature_metrics(adapted_probs)

    rows: list[dict[str, Any]] = []
    for pos_idx in recent_ids:
        pos_idx = int(pos_idx)
        pos_accept = acceptance_by_id[pos_idx]
        rows.append(
            {
                "dataset": "cifar10c",
                "corruption": corruption,
                "severity": SEVERITY,
                "seed": seed,
                "method": method,
                "audit_time_t": audit_time,
                "horizon_H": horizon,
                "anchor_id": pos_idx,
                "candidate_id": pos_idx,
                "membership_label": 1,
                "pool_tier": "positive",
                "ablation_name": None,
                "source_probs": json.dumps(source.probs[pos_idx].tolist()),
                "adapted_probs": json.dumps(adapted_probs[adapted_lookup[pos_idx]].tolist()),
                "source_pred": int(source.probs[pos_idx].argmax()),
                "adapted_pred": int(adapted_probs[adapted_lookup[pos_idx]].argmax()),
                "source_entropy": float(source.entropy[pos_idx]),
                "adapted_entropy": float(adapted_metrics["entropy"][adapted_lookup[pos_idx]]),
                "source_margin": float(source.margin[pos_idx]),
                "adapted_margin": float(adapted_metrics["margin"][adapted_lookup[pos_idx]]),
                "acceptance_flag": pos_accept,
                "source_feature_id": pos_idx,
                "true_class": int(source.y[pos_idx]),
                "candidate_stream_bucket": int(time_bucket[pos_idx]),
            }
        )

        specs = candidate_specs_by_positive[pos_idx]
        for pool_name, candidate_ids in specs.items():
            ablation_name = None
            pool_tier = pool_name
            if pool_name.startswith("ablation::"):
                ablation_name = pool_name.split("::", 1)[1]
                pool_tier = "ablation"

            filtered = apply_acceptance_matching(
                candidate_ids,
                pos_idx,
                acceptance_by_id,
                adaptor.uses_acceptance_matching and pool_name != "unmatched",
            )
            if pool_name == "nn_matched":
                filtered = select_nn(source, pos_idx, filtered)
            elif pool_name.startswith("ablation::"):
                if ablation_name != "remove_nn_matching":
                    filtered = select_nn(source, pos_idx, filtered)

            for candidate_idx in filtered.tolist():
                candidate_idx = int(candidate_idx)
                j = adapted_lookup[candidate_idx]
                row = {
                    "dataset": "cifar10c",
                    "corruption": corruption,
                    "severity": SEVERITY,
                    "seed": seed,
                    "method": method,
                    "audit_time_t": audit_time,
                    "horizon_H": horizon,
                    "anchor_id": pos_idx,
                    "candidate_id": candidate_idx,
                    "membership_label": 0,
                    "pool_tier": pool_tier,
                    "ablation_name": ablation_name,
                    "source_probs": json.dumps(source.probs[candidate_idx].tolist()),
                    "adapted_probs": json.dumps(adapted_probs[j].tolist()),
                    "source_pred": int(source.probs[candidate_idx].argmax()),
                    "adapted_pred": int(adapted_probs[j].argmax()),
                    "source_entropy": float(source.entropy[candidate_idx]),
                    "adapted_entropy": float(adapted_metrics["entropy"][j]),
                    "source_margin": float(source.margin[candidate_idx]),
                    "adapted_margin": float(adapted_metrics["margin"][j]),
                    "acceptance_flag": int(acceptance[j]),
                    "source_feature_id": candidate_idx,
                    "true_class": int(source.y[candidate_idx]),
                    "candidate_stream_bucket": int(time_bucket[candidate_idx]),
                }
                rows.append(row)
    return rows


def run_single_stream(
    *,
    method: str,
    corruption: str,
    seed: int,
    stream_length: int = STREAM_LENGTH,
) -> dict[str, Any]:
    out_dir = ARTIFACTS_DIR / "runs" / method / corruption / f"seed_{seed}_stream_{stream_length}"
    parquet_path = out_dir / "audit_rows.parquet"
    metrics_path = out_dir / "run_metrics.json"
    stream_path = out_dir / "stream_indices.npy"
    log_path = stage_dir("core_matrix") / "logs" / f"{method}_{corruption}_seed_{seed}_stream_{stream_length}.jsonl"
    if parquet_path.exists() and metrics_path.exists() and stream_path.exists():
        cached = read_json(metrics_path)
        if cached.get("pipeline_version") == PIPELINE_VERSION:
            log_event(log_path, "reuse_cached_run", method=method, corruption=corruption, seed=seed, stream_length=stream_length)
            return cached
        log_event(
            log_path,
            "invalidate_cached_run",
            method=method,
            corruption=corruption,
            seed=seed,
            stream_length=stream_length,
            cached_version=cached.get("pipeline_version"),
            required_version=PIPELINE_VERSION,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log_event(log_path, "run_started", method=method, corruption=corruption, seed=seed, stream_length=stream_length)

    source = get_source_cache(corruption, device)
    stream_indices = rng.permutation(len(source.y))[:stream_length]
    np.save(stream_path, stream_indices)
    positions = compute_stream_positions(stream_indices)
    time_bucket = time_bucket_from_positions(positions)
    source_model = load_source_model(device)
    adaptor = build_adaptor(method, source_model, device)

    audit_rows: list[dict[str, Any]] = []
    stream_correct: list[float] = []
    stream_tail_correct: list[float] = []
    batch_acceptance_rates: list[float] = []
    accepted_supports_total = 0.0
    final_support_count = None
    start = time.time()
    audit_time_set = set(t for t in AUDIT_TIMES if t <= stream_length)

    current_end = 0
    boundaries = list(range(ADAPTATION_CHUNK_SIZE, stream_length + 1, ADAPTATION_CHUNK_SIZE))
    if boundaries[-1] != stream_length:
        boundaries.append(stream_length)
    for chunk_end in boundaries:
        chunk_indices = stream_indices[current_end:chunk_end]
        batch = torch.from_numpy(source.x[chunk_indices]).to(device)
        probs_before, _, _ = adaptor.predict_indices(source.x, chunk_indices)
        preds_before = probs_before.argmax(axis=1)
        correct = (preds_before == source.y[chunk_indices]).astype(np.float32)
        stream_correct.extend(correct.tolist())
        absolute_positions = np.arange(current_end + 1, chunk_end + 1)
        if len(absolute_positions) > 0:
            tail_mask = absolute_positions > stream_length - 500
            if tail_mask.any():
                stream_tail_correct.extend(correct[tail_mask].tolist())
        adapt_stats = adaptor.adapt_batch(batch) or {}
        batch_acceptance_rates.append(float(adapt_stats.get("acceptance_rate", 1.0)))
        accepted_supports_total += float(adapt_stats.get("accepted_count", batch.shape[0]))
        if "support_count" in adapt_stats:
            final_support_count = float(adapt_stats["support_count"])
        current_end = chunk_end

        if chunk_end in audit_time_set:
            log_event(log_path, "audit_time_reached", audit_time=chunk_end, method=method, corruption=corruption, seed=seed)
            for horizon in HORIZONS:
                if chunk_end >= horizon:
                    audit_rows.extend(
                        materialize_query_rows(
                            source=source,
                            seed=seed,
                            corruption=corruption,
                            method=method,
                            audit_time=chunk_end,
                            horizon=horizon,
                            stream_indices=stream_indices,
                            positions=positions,
                            time_bucket=time_bucket,
                            adaptor=adaptor,
                            rng=rng,
                        )
                    )

    rows_df = pd.DataFrame(audit_rows)
    rows_df.to_parquet(parquet_path, index=False)
    runtime_seconds = time.time() - start
    result = {
        "method": method,
        "corruption": corruption,
        "severity": SEVERITY,
        "pipeline_version": PIPELINE_VERSION,
        "seed": seed,
        "stream_length": stream_length,
        "batch_size": BATCH_SIZE,
        "adaptation_chunk_size": ADAPTATION_CHUNK_SIZE,
        "query_batch_size": QUERY_BATCH_SIZE,
        "audit_times": int(sum(1 for t in AUDIT_TIMES if t <= stream_length)),
        "mean_stream_accuracy": float(np.mean(stream_correct)),
        "tail_stream_accuracy": float(np.mean(stream_tail_correct)) if stream_tail_correct else float(np.mean(stream_correct)),
        "mean_batch_acceptance_rate": float(np.mean(batch_acceptance_rates)) if batch_acceptance_rates else None,
        "accepted_supports_total": int(round(accepted_supports_total)),
        "final_support_count": final_support_count,
        "runtime_seconds": runtime_seconds,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        "rows_logged": int(len(rows_df)),
    }
    save_json(metrics_path, result)
    log_event(log_path, "run_finished", **result)
    return result


def run_environment_smoke() -> dict[str, Any]:
    ensure_layout()
    import platform
    import subprocess

    log_path = stage_dir("environment_smoke") / "logs" / "environment_smoke.jsonl"
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
        "seeds": SEEDS,
        "methods": METHODS,
        "corruptions": CORRUPTIONS,
        "horizons": HORIZONS,
        "batch_size": BATCH_SIZE,
        "adaptation_chunk_size": ADAPTATION_CHUNK_SIZE,
    }
    for name, cmd in {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        "nproc": ["nproc"],
        "free_h": ["free", "-h"],
    }.items():
        try:
            payload[name] = subprocess.check_output(cmd, text=True).strip()
        except Exception as exc:
            payload[f"{name}_error"] = str(exc)

    smoke = {}
    for method in METHODS:
        smoke[method] = run_single_stream(method=method, corruption="gaussian_noise", seed=0, stream_length=512)
    payload["smoke_test"] = smoke
    save_json(stage_dir("environment_smoke") / "results.json", payload)
    log_event(log_path, "environment_smoke_finished", **payload)
    return payload


def main_matrix() -> dict[str, Any]:
    all_rows = []
    log_path = stage_dir("core_matrix") / "logs" / "core_matrix.jsonl"
    for corruption in CORRUPTIONS:
        for method in METHODS:
            for seed in SEEDS:
                all_rows.append(run_single_stream(method=method, corruption=corruption, seed=seed))
    df = pd.DataFrame(all_rows)
    stage_result = {
        "runs": all_rows,
        "utility_summary": df.groupby(["method", "corruption"]).agg(
            mean_stream_accuracy_mean=("mean_stream_accuracy", "mean"),
            mean_stream_accuracy_std=("mean_stream_accuracy", "std"),
            tail_stream_accuracy_mean=("tail_stream_accuracy", "mean"),
            tail_stream_accuracy_std=("tail_stream_accuracy", "std"),
        ).reset_index().to_dict(orient="records"),
    }
    save_json(stage_dir("core_matrix") / "results.json", stage_result)
    log_event(log_path, "core_matrix_finished", n_runs=len(all_rows))
    return stage_result


def load_all_audit_rows() -> pd.DataFrame:
    frames = []
    for corruption in CORRUPTIONS:
        for method in METHODS:
            for seed in SEEDS:
                path = ARTIFACTS_DIR / "runs" / method / corruption / f"seed_{seed}_stream_{STREAM_LENGTH}" / "audit_rows.parquet"
                frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def with_attack_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    adapted_probs = np.vstack(out["adapted_probs"].map(json.loads).to_numpy())
    adapted = feature_metrics(adapted_probs)
    out["max_prob"] = adapted["max_prob"]
    out["entropy"] = adapted["entropy"]
    out["margin"] = adapted["margin"]
    out["gini"] = adapted["gini"]
    out["log_prob_norm"] = adapted["log_prob_norm"]
    return out


def split_audit_times(times: list[int]) -> tuple[set[int], set[int], set[int]]:
    unique_times = sorted(times)
    n = len(unique_times)
    train_end = max(1, int(n * TRAIN_FRAC))
    val_end = max(train_end + 1, int(n * (TRAIN_FRAC + VAL_FRAC)))
    val_end = min(val_end, n - 1)
    return set(unique_times[:train_end]), set(unique_times[train_end:val_end]), set(unique_times[val_end:])


def fit_logistic(train_df: pd.DataFrame, features: list[str]) -> LogisticRegression | None:
    labels = train_df["membership_label"].to_numpy()
    if len(labels) == 0 or labels.min() == labels.max():
        return None
    model = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    model.fit(train_df[features].to_numpy(), labels)
    return model


def score_with_model(df: pd.DataFrame, model: LogisticRegression | None, features: list[str]) -> np.ndarray:
    if model is None or df.empty:
        return np.full(len(df), 0.5, dtype=np.float32)
    return model.predict_proba(df[features].to_numpy())[:, 1]


def select_best_scalar(train_df: pd.DataFrame) -> str:
    candidates = ["entropy", "max_prob", "margin"]
    y = train_df["membership_label"].to_numpy()
    best_name = candidates[0]
    best_auc = -np.inf
    for name in candidates:
        scores = -train_df[name].to_numpy() if name == "entropy" else train_df[name].to_numpy()
        try:
            auc = roc_auc_score(y, scores)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_name = name
    return best_name


def threshold_for_target_fpr(scores: np.ndarray, y_true: np.ndarray, fpr_target: float) -> float | None:
    neg_scores = scores[y_true == 0]
    if len(neg_scores) == 0:
        return None
    q = max(0.0, min(1.0, 1.0 - fpr_target))
    return float(np.quantile(neg_scores, q, method="higher"))


def apply_threshold(scores: np.ndarray, y_true: np.ndarray, threshold: float | None) -> tuple[float | None, float | None]:
    if threshold is None or len(scores) == 0:
        return None, None
    preds = scores >= threshold
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float(preds[pos_mask].mean()) if pos_mask.any() else None
    fpr = float(preds[neg_mask].mean()) if neg_mask.any() else None
    return tpr, fpr


def metric_min_negatives(fpr_target: float) -> int:
    return LOW_FPR_MIN_NEGATIVES[str(fpr_target)]


def bootstrap_threshold_metric(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_col: str,
    fpr_target: float,
) -> tuple[float, float]:
    if test_df.empty:
        return float("nan"), float("nan")
    val_times = sorted(val_df["audit_time_t"].unique().tolist())
    test_times = sorted(test_df["audit_time_t"].unique().tolist())
    val_groups = [val_df[val_df["audit_time_t"] == t] for t in val_times]
    test_groups = [test_df[test_df["audit_time_t"] == t] for t in test_times]
    estimates = []
    rng = np.random.default_rng(1234)
    for _ in range(BOOTSTRAP_SAMPLES):
        val_sample = pd.concat([val_groups[i] for i in rng.integers(0, len(val_groups), size=len(val_groups))], ignore_index=True)
        test_sample = pd.concat([test_groups[i] for i in rng.integers(0, len(test_groups), size=len(test_groups))], ignore_index=True)
        if (test_sample["membership_label"] == 0).sum() < metric_min_negatives(fpr_target):
            continue
        threshold = threshold_for_target_fpr(
            val_sample[score_col].to_numpy(),
            val_sample["membership_label"].to_numpy(),
            fpr_target,
        )
        tpr, _ = apply_threshold(test_sample[score_col].to_numpy(), test_sample["membership_label"].to_numpy(), threshold)
        if tpr is not None:
            estimates.append(tpr)
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def bootstrap_roc_auc_by_time(df: pd.DataFrame, score_col: str) -> tuple[float, float]:
    if df.empty:
        return float("nan"), float("nan")
    times = sorted(df["audit_time_t"].unique().tolist())
    groups = [df[df["audit_time_t"] == t] for t in times]
    rng = np.random.default_rng(1234)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), size=len(groups))], ignore_index=True)
        try:
            estimates.append(roc_auc_score(sample["membership_label"], sample[score_col]))
        except ValueError:
            continue
    if not estimates:
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_pool_df(
    subset: pd.DataFrame,
    *,
    pool_tier: str,
    ablation_name: str | None = None,
) -> pd.DataFrame:
    positives = subset[subset["pool_tier"] == "positive"]
    if pool_tier == "ablation":
        negatives = subset[(subset["pool_tier"] == "ablation") & (subset["ablation_name"] == ablation_name)]
    else:
        negatives = subset[subset["pool_tier"] == pool_tier]
    return pd.concat([positives, negatives], ignore_index=True)


def evaluate_scored_pool(
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    score_col: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_rows": int(len(test_df)),
        "n_pos": int((test_df["membership_label"] == 1).sum()),
        "n_neg": int((test_df["membership_label"] == 0).sum()),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(test_df["membership_label"], test_df[score_col]))
        result["roc_auc_ci"] = bootstrap_roc_auc_by_time(test_df, score_col)
    except ValueError:
        result["roc_auc"] = float("nan")
        result["roc_auc_ci"] = (float("nan"), float("nan"))

    for fpr_target, field in [(0.01, "tpr_at_1pct_fpr"), (0.001, "tpr_at_0_1pct_fpr")]:
        if result["n_neg"] < metric_min_negatives(fpr_target):
            result[field] = None
            result[f"{field}_ci"] = (float("nan"), float("nan"))
            result[f"{field}_threshold"] = None
            result[f"{field}_test_fpr"] = None
            continue
        threshold = threshold_for_target_fpr(val_df[score_col].to_numpy(), val_df["membership_label"].to_numpy(), fpr_target)
        tpr, test_fpr = apply_threshold(test_df[score_col].to_numpy(), test_df["membership_label"].to_numpy(), threshold)
        result[field] = tpr
        result[f"{field}_test_fpr"] = test_fpr
        result[f"{field}_threshold"] = threshold
        result[f"{field}_ci"] = bootstrap_threshold_metric(val_df, test_df, score_col, fpr_target)
    return result


def summarize_metric(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return {"mean": None, "std": None}
    std = float(numeric.std(ddof=1)) if numeric.notna().sum() > 1 else 0.0
    return {"mean": float(numeric.mean()), "std": std}


def aggregate_bootstrap_ci(values: pd.Series) -> list[float] | None:
    bounds = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            if pd.notna(value[0]) and pd.notna(value[1]):
                bounds.append((float(value[0]), float(value[1])))
    if not bounds:
        return None
    arr = np.asarray(bounds, dtype=np.float64)
    return [float(arr[:, 0].mean()), float(arr[:, 1].mean())]


def run_analysis() -> dict[str, Any]:
    stage = stage_dir("ablation")
    log_path = stage / "logs" / "analysis.jsonl"
    df = with_attack_features(load_all_audit_rows())
    analyses = []
    scored_rows: list[pd.DataFrame] = []

    scalar_choices: dict[tuple[str, str, int], str] = {}
    seed0 = df[df["seed"] == 0]
    for method in METHODS:
        if method == "source_only":
            continue
        for corruption in CORRUPTIONS:
            for horizon in ABLATION_HORIZONS:
                subset = seed0[
                    (seed0["method"] == method)
                    & (seed0["corruption"] == corruption)
                    & (seed0["horizon_H"] == horizon)
                ]
                train_times, _, _ = split_audit_times(sorted(subset["audit_time_t"].unique().tolist()))
                train_df = build_pool_df(subset[subset["audit_time_t"].isin(train_times)], pool_tier="matched")
                if not train_df.empty:
                    scalar_choices[(method, corruption, horizon)] = select_best_scalar(train_df)

    for seed in SEEDS:
        seed_df = df[df["seed"] == seed]
        for method in METHODS:
            for corruption in CORRUPTIONS:
                for horizon in HORIZONS:
                    subset = seed_df[
                        (seed_df["method"] == method)
                        & (seed_df["corruption"] == corruption)
                        & (seed_df["horizon_H"] == horizon)
                    ]
                    if subset.empty:
                        continue
                    train_times, val_times, test_times = split_audit_times(sorted(subset["audit_time_t"].unique().tolist()))
                    if not test_times:
                        continue

                    train_matched = build_pool_df(subset[subset["audit_time_t"].isin(train_times)], pool_tier="matched")
                    val_matched = build_pool_df(subset[subset["audit_time_t"].isin(val_times)], pool_tier="matched")
                    main_model = fit_logistic(train_matched, FEATURES_BLACK_BOX)
                    log_event(
                        log_path,
                        "fit_main_auditor",
                        seed=seed,
                        method=method,
                        corruption=corruption,
                        horizon=horizon,
                        train_rows=len(train_matched),
                        val_rows=len(val_matched),
                    )

                    for pool_tier in MAIN_POOL_TIERS:
                        val_pool = build_pool_df(subset[subset["audit_time_t"].isin(val_times)], pool_tier=pool_tier)
                        test_pool = build_pool_df(subset[subset["audit_time_t"].isin(test_times)], pool_tier=pool_tier)
                        if val_pool.empty or test_pool.empty:
                            continue
                        val_pool = val_pool.copy()
                        test_pool = test_pool.copy()
                        val_pool["score"] = score_with_model(val_pool, main_model, FEATURES_BLACK_BOX)
                        test_pool["score"] = score_with_model(test_pool, main_model, FEATURES_BLACK_BOX)
                        metrics = evaluate_scored_pool(test_pool, val_pool, score_col="score")
                        analyses.append(
                            {
                                "seed": seed,
                                "method": method,
                                "corruption": corruption,
                                "horizon_H": horizon,
                                "analysis": "black_box",
                                "pool_tier": pool_tier,
                                "ablation_name": None,
                                **metrics,
                            }
                        )
                        scored_rows.append(
                            val_pool.assign(
                                split="val",
                                analysis="black_box",
                                eval_pool_tier=pool_tier,
                                eval_ablation_name=None,
                                score_kind="logistic",
                            )
                        )
                        scored_rows.append(
                            test_pool.assign(
                                split="test",
                                analysis="black_box",
                                eval_pool_tier=pool_tier,
                                eval_ablation_name=None,
                                score_kind="logistic",
                            )
                        )
                        if pool_tier == "matched":
                            shuffled = test_pool.copy()
                            shuffled["membership_label"] = np.random.default_rng(seed + horizon).permutation(
                                shuffled["membership_label"].to_numpy()
                            )
                            shuffled_metrics = evaluate_scored_pool(shuffled, val_pool, score_col="score")
                            analyses.append(
                                {
                                    "seed": seed,
                                    "method": method,
                                    "corruption": corruption,
                                    "horizon_H": horizon,
                                    "analysis": "shuffled_membership_null",
                                    "pool_tier": pool_tier,
                                    "ablation_name": None,
                                    **shuffled_metrics,
                                }
                            )

                    if method != "source_only" and horizon in ABLATION_HORIZONS:
                        for ablation_name in ABLATION_NAMES:
                            train_pool = build_pool_df(
                                subset[subset["audit_time_t"].isin(train_times)],
                                pool_tier="ablation",
                                ablation_name=ablation_name,
                            )
                            val_pool = build_pool_df(
                                subset[subset["audit_time_t"].isin(val_times)],
                                pool_tier="ablation",
                                ablation_name=ablation_name,
                            )
                            test_pool = build_pool_df(
                                subset[subset["audit_time_t"].isin(test_times)],
                                pool_tier="ablation",
                                ablation_name=ablation_name,
                            )
                            if train_pool.empty or val_pool.empty or test_pool.empty:
                                continue
                            model = fit_logistic(train_pool, FEATURES_BLACK_BOX)
                            val_pool = val_pool.copy()
                            test_pool = test_pool.copy()
                            val_pool["score"] = score_with_model(val_pool, model, FEATURES_BLACK_BOX)
                            test_pool["score"] = score_with_model(test_pool, model, FEATURES_BLACK_BOX)
                            metrics = evaluate_scored_pool(test_pool, val_pool, score_col="score")
                            analyses.append(
                                {
                                    "seed": seed,
                                    "method": method,
                                    "corruption": corruption,
                                    "horizon_H": horizon,
                                    "analysis": "matching_ablation",
                                    "pool_tier": "ablation",
                                    "ablation_name": ablation_name,
                                    **metrics,
                                }
                            )
                            scored_rows.append(
                                val_pool.assign(
                                    split="val",
                                    analysis="matching_ablation",
                                    eval_pool_tier="ablation",
                                    eval_ablation_name=ablation_name,
                                    score_kind="logistic",
                                )
                            )
                            scored_rows.append(
                                test_pool.assign(
                                    split="test",
                                    analysis="matching_ablation",
                                    eval_pool_tier="ablation",
                                    eval_ablation_name=ablation_name,
                                    score_kind="logistic",
                                )
                            )

                        scalar_name = scalar_choices.get((method, corruption, horizon), "entropy")
                        val_pool = build_pool_df(subset[subset["audit_time_t"].isin(val_times)], pool_tier="nn_matched")
                        test_pool = build_pool_df(subset[subset["audit_time_t"].isin(test_times)], pool_tier="nn_matched")
                        if not val_pool.empty and not test_pool.empty:
                            val_pool = val_pool.copy()
                            test_pool = test_pool.copy()
                            val_pool["score"] = -val_pool[scalar_name].to_numpy() if scalar_name == "entropy" else val_pool[scalar_name].to_numpy()
                            test_pool["score"] = -test_pool[scalar_name].to_numpy() if scalar_name == "entropy" else test_pool[scalar_name].to_numpy()
                            metrics = evaluate_scored_pool(test_pool, val_pool, score_col="score")
                            analyses.append(
                                {
                                    "seed": seed,
                                    "method": method,
                                    "corruption": corruption,
                                    "horizon_H": horizon,
                                    "analysis": "attack_family_ablation_scalar",
                                    "pool_tier": "nn_matched",
                                    "ablation_name": scalar_name,
                                    **metrics,
                                }
                            )
                            scored_rows.append(
                                val_pool.assign(
                                    split="val",
                                    analysis="attack_family_ablation_scalar",
                                    eval_pool_tier="nn_matched",
                                    eval_ablation_name=scalar_name,
                                    score_kind="scalar",
                                )
                            )
                            scored_rows.append(
                                test_pool.assign(
                                    split="test",
                                    analysis="attack_family_ablation_scalar",
                                    eval_pool_tier="nn_matched",
                                    eval_ablation_name=scalar_name,
                                    score_kind="scalar",
                                )
                            )

    scored_df = pd.concat(scored_rows, ignore_index=True)
    scored_path = stage / "scored_rows.parquet"
    scored_df.to_parquet(scored_path, index=False)
    analysis_result = {"rows": analyses, "scored_rows_path": str(scored_path)}
    save_json(stage / "results.json", analysis_result)
    log_event(log_path, "analysis_finished", n_rows=len(analyses), scored_rows=len(scored_df))
    return analysis_result


def aggregate_results() -> dict[str, Any]:
    utility_rows = []
    for corruption in CORRUPTIONS:
        source_runs = {
            seed: read_json(ARTIFACTS_DIR / "runs" / "source_only" / corruption / f"seed_{seed}_stream_{STREAM_LENGTH}" / "run_metrics.json")
            for seed in SEEDS
        }
        for method in METHODS:
            method_runs = [
                read_json(ARTIFACTS_DIR / "runs" / method / corruption / f"seed_{seed}_stream_{STREAM_LENGTH}" / "run_metrics.json")
                for seed in SEEDS
            ]
            utility_rows.append(
                {
                    "method": method,
                    "corruption": corruption,
                    "mean_stream_accuracy": {
                        "mean": float(np.mean([r["mean_stream_accuracy"] for r in method_runs])),
                        "std": float(np.std([r["mean_stream_accuracy"] for r in method_runs], ddof=1)),
                    },
                    "tail_stream_accuracy": {
                        "mean": float(np.mean([r["tail_stream_accuracy"] for r in method_runs])),
                        "std": float(np.std([r["tail_stream_accuracy"] for r in method_runs], ddof=1)),
                    },
                    "utility_gain_vs_source": {
                        "mean": float(
                            np.mean(
                                [
                                    method_runs[i]["mean_stream_accuracy"] - source_runs[SEEDS[i]]["mean_stream_accuracy"]
                                    for i in range(len(SEEDS))
                                ]
                            )
                        ),
                        "std": float(
                            np.std(
                                [
                                    method_runs[i]["mean_stream_accuracy"] - source_runs[SEEDS[i]]["mean_stream_accuracy"]
                                    for i in range(len(SEEDS))
                                ],
                                ddof=1,
                            )
                        ),
                    },
                }
            )

    analysis = pd.DataFrame(read_json(stage_dir("ablation") / "results.json")["rows"])
    scored = pd.read_parquet(stage_dir("ablation") / "scored_rows.parquet")

    def pooled_seed_metrics(
        *,
        method: str,
        corruption: str,
        horizon: int,
        pool_tier: str,
    ) -> dict[str, Any]:
        subset = scored[
            (scored["analysis"] == "black_box")
            & (scored["score_kind"] == "logistic")
            & (scored["method"] == method)
            & (scored["corruption"] == corruption)
            & (scored["horizon_H"] == horizon)
            & (scored["eval_pool_tier"] == pool_tier)
        ]
        val_df = subset[subset["split"] == "val"]
        test_df = subset[subset["split"] == "test"]
        result = {
            "n_neg_val": int((val_df["membership_label"] == 0).sum()),
            "n_neg_test": int((test_df["membership_label"] == 0).sum()),
        }
        if test_df.empty:
            result["roc_auc"] = None
            result["tpr_at_1pct_fpr"] = None
            result["tpr_at_0_1pct_fpr"] = None
            return result
        try:
            result["roc_auc"] = float(roc_auc_score(test_df["membership_label"], test_df["score"]))
        except ValueError:
            result["roc_auc"] = None
        for fpr_target, field in [(0.01, "tpr_at_1pct_fpr"), (0.001, "tpr_at_0_1pct_fpr")]:
            if result["n_neg_val"] < metric_min_negatives(fpr_target) or result["n_neg_test"] < metric_min_negatives(fpr_target):
                result[field] = None
                continue
            threshold = threshold_for_target_fpr(val_df["score"].to_numpy(), val_df["membership_label"].to_numpy(), fpr_target)
            tpr, _ = apply_threshold(test_df["score"].to_numpy(), test_df["membership_label"].to_numpy(), threshold)
            result[field] = tpr
        return result

    t3a_acceptance = []
    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            run = read_json(ARTIFACTS_DIR / "runs" / "t3a" / corruption / f"seed_{seed}_stream_{STREAM_LENGTH}" / "run_metrics.json")
            t3a_acceptance.append(
                {
                    "corruption": corruption,
                    "seed": seed,
                    "mean_batch_acceptance_rate": run.get("mean_batch_acceptance_rate"),
                    "accepted_supports_total": run.get("accepted_supports_total"),
                    "final_support_count": run.get("final_support_count"),
                }
            )
    main_rows = []
    for method in METHODS:
        for corruption in CORRUPTIONS:
            for horizon in HORIZONS:
                for pool_tier in MAIN_POOL_TIERS:
                    per_seed = analysis[
                        (analysis["analysis"] == "black_box")
                        & (analysis["method"] == method)
                        & (analysis["corruption"] == corruption)
                        & (analysis["horizon_H"] == horizon)
                        & (analysis["pool_tier"] == pool_tier)
                    ]
                    if per_seed.empty:
                        continue
                    main_rows.append(
                        {
                            "method": method,
                            "corruption": corruption,
                            "horizon_H": horizon,
                            "pool_tier": pool_tier,
                            "roc_auc": summarize_metric(per_seed["roc_auc"]),
                            "roc_auc_pooled": {
                                "value": summarize_metric(per_seed["roc_auc"])["mean"],
                                "ci95": aggregate_bootstrap_ci(per_seed["roc_auc_ci"]),
                            },
                            "tpr_at_1pct_fpr": summarize_metric(per_seed["tpr_at_1pct_fpr"]),
                            "tpr_at_1pct_fpr_pooled": {
                                "value": summarize_metric(per_seed["tpr_at_1pct_fpr"])["mean"],
                                "ci95": aggregate_bootstrap_ci(per_seed["tpr_at_1pct_fpr_ci"]),
                            },
                            "tpr_at_0_1pct_fpr": summarize_metric(per_seed["tpr_at_0_1pct_fpr"]),
                            "tpr_at_0_1pct_fpr_pooled": {
                                "value": summarize_metric(per_seed["tpr_at_0_1pct_fpr"])["mean"],
                                "ci95": aggregate_bootstrap_ci(per_seed["tpr_at_0_1pct_fpr_ci"]),
                            },
                            "n_neg_test_total": int(pd.to_numeric(per_seed["n_neg"], errors="coerce").sum()),
                            "pooled_across_seeds": pooled_seed_metrics(
                                method=method,
                                corruption=corruption,
                                horizon=horizon,
                                pool_tier=pool_tier,
                            ),
                        }
                    )

    ablation_rows = []
    for method in [m for m in METHODS if m != "source_only"]:
        for corruption in CORRUPTIONS:
            for horizon in sorted(ABLATION_HORIZONS):
                for ablation_name in ABLATION_NAMES:
                    per_seed = analysis[
                        (analysis["analysis"] == "matching_ablation")
                        & (analysis["method"] == method)
                        & (analysis["corruption"] == corruption)
                        & (analysis["horizon_H"] == horizon)
                        & (analysis["ablation_name"] == ablation_name)
                    ]
                    if per_seed.empty:
                        continue
                    ablation_rows.append(
                        {
                            "method": method,
                            "corruption": corruption,
                            "horizon_H": horizon,
                            "ablation_name": ablation_name,
                            "roc_auc": summarize_metric(per_seed["roc_auc"]),
                            "tpr_at_1pct_fpr": summarize_metric(per_seed["tpr_at_1pct_fpr"]),
                            "tpr_at_0_1pct_fpr": summarize_metric(per_seed["tpr_at_0_1pct_fpr"]),
                        }
                    )

    result = {
        "title": "Recent-window membership audit for online TTA on CIFAR-10-C",
        "protocol": {
            "methods": METHODS,
            "corruptions": CORRUPTIONS,
            "severity": SEVERITY,
            "seeds": SEEDS,
            "horizons": HORIZONS,
            "stream_length": STREAM_LENGTH,
            "batch_size": BATCH_SIZE,
            "adaptation_chunk_size": ADAPTATION_CHUNK_SIZE,
            "query_batch_size": QUERY_BATCH_SIZE,
            "low_fpr_min_negatives": LOW_FPR_MIN_NEGATIVES,
            "primary_endpoint_revision": "Balanced fixed-width matched pools remove the prior null-control confound but leave H=1/H=8 low-FPR reporting underpowered; matched and NN-matched ROC-AUC are the primary confirmatory endpoints, with pooled-across-seed TPR@1% FPR reported only where validation and test pools each have at least 10000 negatives.",
        },
        "utility": utility_rows,
        "t3a_acceptance": t3a_acceptance,
        "privacy": main_rows,
        "matching_ablation": ablation_rows,
        "analysis_path": str(stage_dir("ablation") / "results.json"),
        "scored_rows_path": str(stage_dir("ablation") / "scored_rows.parquet"),
    }
    save_json(ROOT / "results.json", result)
    return result


def make_threat_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    ax.annotate("source model $f_{src}$", (0.05, 0.7), bbox={"boxstyle": "round,pad=0.4", "fc": "#d8e2dc"})
    ax.annotate("online target stream", (0.28, 0.7), bbox={"boxstyle": "round,pad=0.4", "fc": "#ffe5d9"})
    ax.annotate("deployed state $f_t$", (0.55, 0.7), bbox={"boxstyle": "round,pad=0.4", "fc": "#cddafd"})
    ax.annotate("black-box query on candidate $x$", (0.8, 0.7), bbox={"boxstyle": "round,pad=0.4", "fc": "#f4acb7"})
    ax.annotate("", xy=(0.23, 0.72), xytext=(0.14, 0.72), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.annotate("", xy=(0.5, 0.72), xytext=(0.39, 0.72), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.annotate("", xy=(0.76, 0.72), xytext=(0.66, 0.72), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(0.45, 0.25, "recent window $[t-H+1, \\ldots, t]$", ha="center", va="center", fontsize=12)
    ax.plot([0.25, 0.64], [0.2, 0.2], color="#333333", lw=3)
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGURES_DIR / "threat_model_schematic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _ci_to_yerr(values: list[float], cis: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for value, ci in zip(values, cis):
        if value is None or ci is None:
            lower.append(0.0)
            upper.append(0.0)
            continue
        ci_arr = np.asarray(ci, dtype=np.float64)
        if np.isnan(value) or np.isnan(ci_arr).any():
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(max(0.0, value - ci[0]))
            upper.append(max(0.0, ci[1] - value))
    return np.array(lower), np.array(upper)


def ci_or_nan(item: dict[str, Any] | None) -> tuple[float, float]:
    if not item:
        return (float("nan"), float("nan"))
    ci = item.get("ci95")
    if ci is None:
        return (float("nan"), float("nan"))
    return tuple(ci)


def numeric_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        if pd.isna(value):
            return float("nan")
    except Exception:
        pass
    return float(value)


def make_visualizations() -> dict[str, Any]:
    stage = stage_dir("visualization")
    log_path = stage / "logs" / "visualization.jsonl"
    result = aggregate_results()
    make_threat_schematic()

    privacy = pd.DataFrame(result["privacy"])

    h8 = privacy[
        (privacy["horizon_H"] == 8)
        & (privacy["method"].isin(["tent", "t3a"]))
        & (privacy["pool_tier"].isin(["unmatched", "matched", "nn_matched"]))
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_positions = []
    labels = []
    values = []
    cis = []
    width = 0.35
    offset = {"tent": -width / 2, "t3a": width / 2}
    base_map = {"unmatched": 0, "matched": 1, "nn_matched": 2}
    for i, corruption in enumerate(CORRUPTIONS):
        sub = h8[h8["corruption"] == corruption]
        for method in ["tent", "t3a"]:
            for pool in ["unmatched", "matched", "nn_matched"]:
                row = sub[(sub["method"] == method) & (sub["pool_tier"] == pool)]
                if row.empty:
                    continue
                item = row.iloc[0]["roc_auc_pooled"]
                x = i * 4 + base_map[pool] + offset[method]
                bar_positions.append(x)
                labels.append(pool)
                values.append(numeric_or_nan(item["value"] if item else None))
                cis.append(ci_or_nan(item))
    lower, upper = _ci_to_yerr(values, cis)
    ax.bar(bar_positions, np.nan_to_num(values), width=width)
    ax.errorbar(bar_positions, np.nan_to_num(values), yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=3)
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(["U", "M", "NN", "U", "M", "NN"])
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Control Strength at H=8")
    ax.legend(["95% CI"])
    fig.savefig(FIGURES_DIR / "control_strength_h8.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    rec = privacy[privacy["pool_tier"] == "nn_matched"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, corruption in zip(axes, CORRUPTIONS):
        sub = rec[rec["corruption"] == corruption]
        for method in METHODS:
            line = sub[sub["method"] == method].sort_values("horizon_H")
            xs = line["horizon_H"].tolist()
            ys = [numeric_or_nan(item["value"]) for item in line["roc_auc_pooled"]]
            cis = [ci_or_nan(item) for item in line["roc_auc_pooled"]]
            lower, upper = _ci_to_yerr(ys, cis)
            ax.errorbar(xs, np.nan_to_num(ys), yerr=np.vstack([lower, upper]), marker="o", label=method, capsize=3)
        ax.set_title(corruption)
        ax.set_xlabel("Horizon H")
    axes[0].set_ylabel("NN-matched ROC-AUC")
    axes[0].legend()
    fig.savefig(FIGURES_DIR / "recency_decay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    nulls = pd.DataFrame(read_json(stage_dir("ablation") / "results.json")["rows"])
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = []
    values = []
    for method in METHODS:
        matched = nulls[
            (nulls["analysis"] == "black_box")
            & (nulls["method"] == method)
            & (nulls["horizon_H"] == 8)
            & (nulls["pool_tier"] == "matched")
        ]["roc_auc"].mean()
        far_past = nulls[
            (nulls["analysis"] == "black_box")
            & (nulls["method"] == method)
            & (nulls["horizon_H"] == 8)
            & (nulls["pool_tier"] == "far_past")
        ]["roc_auc"].mean()
        shuffled = nulls[
            (nulls["analysis"] == "shuffled_membership_null")
            & (nulls["method"] == method)
            & (nulls["horizon_H"] == 8)
            & (nulls["pool_tier"] == "matched")
        ]["roc_auc"].mean()
        labels.extend([f"{method}\nrecent", f"{method}\nfar-past", f"{method}\nshuffled"])
        values.extend([matched, far_past, shuffled])
    ax.bar(np.arange(len(values)), np.nan_to_num(values))
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Null Controls at H=8")
    fig.savefig(FIGURES_DIR / "null_controls_h8.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_json(stage / "results.json", {"figures": sorted([p.name for p in FIGURES_DIR.glob("*.png")]), "results_summary": result})
    log_event(log_path, "visualization_finished", n_figures=len(list(FIGURES_DIR.glob("*.png"))))
    return {"figures": sorted([p.name for p in FIGURES_DIR.glob("*.png")]), "results_summary": result}


def run_all() -> dict[str, Any]:
    env = run_environment_smoke()
    core = main_matrix()
    ablation = run_analysis()
    viz = make_visualizations()
    return {"environment": env, "core": core, "ablation": ablation, "visualization": viz}


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["environment", "core", "ablation", "visualization", "all"], default="all")
    args = parser.parse_args()
    ensure_layout()
    if args.stage == "environment":
        run_environment_smoke()
    elif args.stage == "core":
        main_matrix()
    elif args.stage == "ablation":
        run_analysis()
    elif args.stage == "visualization":
        make_visualizations()
    else:
        run_all()


if __name__ == "__main__":
    cli()
