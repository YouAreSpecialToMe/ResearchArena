from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data import (
    CIVIL_IDENTITY_COLUMNS,
    TOKEN_RE,
    TextDataset,
    build_civilcomments_dataset,
    build_imdb_actor_dataset,
)
from .metrics import (
    classification_metrics,
    civil_metrics,
    early_commitment_rate,
    entropy,
    effective_risk_threshold,
    proxy_validation,
    softmax_np,
    summarize_seed_metrics,
)
from .models import LateBindModel, latebind_loss, load_tokenizer
from .utils import (
    ROOT,
    bootstrap_ci,
    capture_environment,
    ensure_dir,
    gpu_memory_stats,
    now,
    read_json,
    runtime_minutes,
    set_seed,
    write_json,
)


SEEDS = [13, 21, 42]
ABLATION_SEEDS = [13]
MODEL_NAME = "roberta-base"
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainConfig:
    condition: str
    dataset: str
    seeds: list[int]
    max_length: int
    batch_size: int = 16
    grad_accum: int = 2
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_epochs: int = 2
    stage1_epochs: int = 1
    misclassified_weight: float = 5.0
    lambda_late: float = 0.08
    lambda_inv: float = 0.5
    entropy_floor: float = float(0.75 * math.log(2))
    risk_mode: str = "hybrid"
    actor_only_masking: bool = False
    use_late: bool = False
    use_inv: bool = False
    jtt: bool = False
    smoke_steps: int | None = None


def get_condition_config(name: str) -> TrainConfig:
    if name == "imdb_erm_smoke_test":
        return TrainConfig(name, "imdb", [13], 256, smoke_steps=100)
    if name == "imdb_latebind_smoke_test":
        return TrainConfig(name, "imdb", [13], 256, use_late=True, use_inv=True, smoke_steps=100)
    if name == "imdb_erm":
        return TrainConfig(name, "imdb", SEEDS, 256)
    if name == "imdb_masker":
        return TrainConfig(name, "imdb", SEEDS, 256, use_inv=True)
    if name == "imdb_jtt":
        return TrainConfig(name, "imdb", SEEDS, 256, jtt=True)
    if name == "imdb_latebind":
        return TrainConfig(name, "imdb", SEEDS, 256, use_late=True, use_inv=True)
    if name == "imdb_no_late_term":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_inv=True)
    if name == "imdb_no_invariance":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_late=True)
    if name == "imdb_ungated_entropy":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_late=True, use_inv=True, risk_mode="ungated")
    if name == "imdb_lexicon_only_risk":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_late=True, use_inv=True, risk_mode="lexicon_only")
    if name == "imdb_attribution_only_risk":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_late=True, use_inv=True, risk_mode="attribution_only")
    if name == "imdb_random_token_risk":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_late=True, use_inv=True, risk_mode="random_token")
    if name == "imdb_actor_only_masking":
        return TrainConfig(name, "imdb", ABLATION_SEEDS, 256, use_late=True, use_inv=True, actor_only_masking=True)
    if name == "civilcomments_erm":
        return TrainConfig(name, "civilcomments", [13], 220)
    if name == "civilcomments_jtt":
        return TrainConfig(name, "civilcomments", [13], 220, jtt=True)
    if name == "civilcomments_latebind":
        return TrainConfig(name, "civilcomments", [13], 220, use_late=True, use_inv=True)
    raise ValueError(f"Unknown condition: {name}")


def prepare_data(dataset_name: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any], set[str]]:
    if dataset_name == "imdb":
        bundle = build_imdb_actor_dataset(ROOT / "cache" / "imdb_actor", ROOT / "artifacts" / "imdb_actor" / "artifact_lexicon.json")
        dfs = {
            "train": bundle.train_df,
            "val_id": bundle.val_id_df,
            "val_ood": bundle.val_ood_df,
        }
        dfs.update(bundle.test_dfs)
        lexicon = bundle.lexicon
    else:
        bundle = build_civilcomments_dataset(
            ROOT / "cache" / "civilcomments_subsampled",
            ROOT / "artifacts" / "civilcomments" / "artifact_lexicon.json",
        )
        dfs = {"train": bundle["train_df"], "val": bundle["val_df"], "test": bundle["test_df"]}
        lexicon = bundle["lexicon"]
    tokens = {
        entry["token"].lower()
        for entries in lexicon["labels"].values()
        for entry in entries
    }
    return dfs, lexicon, tokens


def identity_columns_for_dataset(dataset: str) -> list[str] | None:
    if dataset == "civilcomments":
        return CIVIL_IDENTITY_COLUMNS
    return None


def fit_tfidf_imdb(out_dir: Path) -> dict[str, Any]:
    start = now()
    dfs, _, _ = prepare_data("imdb")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, lowercase=True)
    X_train = vectorizer.fit_transform(dfs["train"]["text"])
    X_val_id = vectorizer.transform(dfs["val_id"]["text"])
    X_val_ood = vectorizer.transform(dfs["val_ood"]["text"])
    y_train = dfs["train"]["label"].to_numpy()
    y_val_id = dfs["val_id"]["label"].to_numpy()
    y_val_ood = dfs["val_ood"]["label"].to_numpy()
    best = None
    for c in [0.25, 1.0, 4.0]:
        clf = LogisticRegression(C=c, solver="liblinear", max_iter=1000)
        clf.fit(X_train, y_train)
        score = 0.5 * accuracy_score(y_val_id, clf.predict(X_val_id)) + 0.5 * accuracy_score(y_val_ood, clf.predict(X_val_ood))
        if best is None or score > best["score"]:
            best = {"C": c, "score": float(score), "clf": clf}
    metrics = {}
    for split in ["test_id", "test_ood", "test_no_shortcut", "test_actor_conflict", "test_actor_scrubbed"]:
        X = vectorizer.transform(dfs[split]["text"])
        y = dfs[split]["label"].to_numpy()
        probs = best["clf"].predict_proba(X)
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        metrics[split] = classification_metrics(logits, y)
    payload = {
        "experiment": "imdb_tfidf_lr",
        "config": {"C": best["C"], "ngram_range": [1, 2], "max_features": 50000},
        "metrics": metrics,
        "selection_score": best["score"],
        "runtime_minutes": runtime_minutes(start),
    }
    write_json(out_dir / "results.json", payload)
    return payload


def token_attributions(
    model: LateBindModel,
    tokenizer,
    df: pd.DataFrame,
    max_length: int,
    device: torch.device,
    batch_size: int = 16,
) -> dict[int, list[dict[str, float]]]:
    model.eval()
    results: dict[int, list[dict[str, float]]] = {}
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        enc = tokenizer(
            batch_df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_cpu = enc["input_ids"]
        input_ids = input_ids_cpu.to(device)
        attention_mask = enc["attention_mask"].to(device)
        embed = model.backbone.embeddings.word_embeddings(input_ids)
        embed = embed.detach().clone().requires_grad_(True)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
            outputs = model.backbone(inputs_embeds=embed, attention_mask=attention_mask, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, 0, :]
            logits = model.classifier(model.dropout(hidden))
        labels = torch.tensor(batch_df["label"].to_numpy(), dtype=torch.long, device=device)
        chosen = logits.gather(1, labels.unsqueeze(1)).sum()
        grads = torch.autograd.grad(chosen, embed)[0]
        scores = (grads * embed).sum(dim=-1).abs().detach().cpu().numpy()
        for row_idx, source_id in enumerate(batch_df["source_id"].tolist()):
            word_scores: defaultdict[str, float] = defaultdict(float)
            merged_tokens: list[tuple[str, float]] = []
            current_word = ""
            current_score = 0.0
            token_pieces = tokenizer.convert_ids_to_tokens(input_ids_cpu[row_idx].tolist())
            for piece, piece_score in zip(token_pieces, scores[row_idx]):
                if piece in tokenizer.all_special_tokens:
                    if current_word:
                        merged_tokens.append((current_word, current_score))
                        current_word, current_score = "", 0.0
                    continue
                starts_new = piece.startswith("Ġ")
                clean_piece = piece.lstrip("Ġ").replace("Ċ", "").strip()
                if not clean_piece:
                    continue
                if starts_new and current_word:
                    merged_tokens.append((current_word, current_score))
                    current_word, current_score = "", 0.0
                current_word += clean_piece
                current_score += float(piece_score)
            if current_word:
                merged_tokens.append((current_word, current_score))
            for word, score in merged_tokens:
                word = word.lower()
                if len(word) < 2 or not TOKEN_RE.fullmatch(word):
                    continue
                word_scores[word] += score
            ranked = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[:5]
            results[int(source_id)] = [{"token": tok, "score": score} for tok, score in ranked]
    return results


def evaluate_split(
    model: LateBindModel,
    loader: DataLoader,
    device: torch.device,
    masked: bool = False,
) -> dict[str, Any]:
    model.eval()
    logits_all, labels_all = [], []
    aux_all = defaultdict(list)
    source_ids, actor_present, risks = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids_key = "masked_input_ids" if masked else "input_ids"
            attention_mask_key = "masked_attention_mask" if masked else "attention_mask"
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
                bundle = model(batch[input_ids_key].to(device), batch[attention_mask_key].to(device))
            logits_all.append(bundle.logits.detach().float().cpu().numpy())
            labels_all.append(batch["labels"].numpy())
            for layer, logits in bundle.aux_logits.items():
                aux_all[layer].append(logits.detach().float().cpu().numpy())
            source_ids.extend(batch["source_id"].numpy().tolist())
            actor_present.extend(batch["actor_present"].numpy().tolist())
            risks.extend(batch["risk"].numpy().tolist())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    aux_probs = {layer: softmax_np(np.concatenate(parts)) for layer, parts in aux_all.items()}
    probs = softmax_np(logits)
    preds = probs.argmax(axis=1)
    result = {
        "logits": logits,
        "labels": labels,
        "probs": probs,
        "preds": preds,
        "source_ids": np.asarray(source_ids),
        "actor_present": np.asarray(actor_present),
        "risk": np.asarray(risks),
        "aux_probs": aux_probs,
    }
    if hasattr(loader.dataset, "df"):
        extra_df = loader.dataset.df.reset_index(drop=True)
        for col in extra_df.columns:
            if col not in {"text", "original_text", "actor_tokens"}:
                result[col] = extra_df[col].to_numpy()
    return result


def fit_aux_heads(
    model: LateBindModel,
    loader: DataLoader,
    device: torch.device,
    epochs: int = 1,
    lr: float = 1e-3,
    max_steps: int = 256,
) -> dict[str, float]:
    for head in model.aux_heads.values():
        head.reset_parameters()
    model.backbone.eval()
    model.classifier.eval()
    model.dropout.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.aux_heads.parameters():
        param.requires_grad_(True)
    optimizer = AdamW(model.aux_heads.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    steps = 0
    for _ in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
                    outputs = model.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
            loss = torch.zeros((), device=device)
            for layer in model.aux_layers:
                hidden = outputs.hidden_states[layer][:, 0, :]
                logits = model.aux_heads[str(layer)](hidden)
                loss = loss + torch.nn.functional.cross_entropy(logits, labels)
            loss = loss / max(1, len(model.aux_layers))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().item()))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break
    for param in model.parameters():
        param.requires_grad_(True)
    return {
        "aux_probe_loss": float(np.mean(losses)) if losses else float("nan"),
        "aux_probe_steps": float(steps),
    }


def mask_effect_metrics(original: dict[str, Any], masked: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    orig_pred = original["preds"]
    orig_conf = original["probs"][np.arange(len(orig_pred)), orig_pred]
    masked_same_label_conf = masked["probs"][np.arange(len(orig_pred)), orig_pred]
    conf_drop = orig_conf - masked_same_label_conf
    label_flip = (masked["preds"] != orig_pred).astype(int)
    summary = {
        "mask_conf_drop": float(conf_drop.mean()),
        "mask_label_flip_rate": float(label_flip.mean()),
    }
    return conf_drop, label_flip, summary


def risk_bucket_ecr(
    aux_probs: dict[int, np.ndarray],
    risk: np.ndarray,
    train_median_risk: float,
    final_preds: np.ndarray,
) -> dict[str, float]:
    high_mask = risk > train_median_risk
    low_mask = risk <= train_median_risk
    metrics = {}
    for layer, probs in aux_probs.items():
        metrics[f"ecr_layer_{layer}"] = early_commitment_rate(probs, final_preds=final_preds)
        metrics[f"ecr_layer_{layer}_high_risk"] = early_commitment_rate(probs[high_mask], final_preds=final_preds[high_mask]) if np.any(high_mask) else float("nan")
        metrics[f"ecr_layer_{layer}_low_risk"] = early_commitment_rate(probs[low_mask], final_preds=final_preds[low_mask]) if np.any(low_mask) else float("nan")
        ent = entropy(probs)
        metrics[f"entropy_layer_{layer}_high_risk"] = float(ent[high_mask].mean()) if np.any(high_mask) else float("nan")
        metrics[f"entropy_layer_{layer}_low_risk"] = float(ent[low_mask].mean()) if np.any(low_mask) else float("nan")
    return metrics


def build_loaders(
    dfs: dict[str, pd.DataFrame],
    tokenizer,
    config: TrainConfig,
    top_tokens: dict[int, list[str]] | None,
    lexicon_tokens: set[str],
) -> dict[str, DataLoader]:
    out = {}
    for split_name, df in dfs.items():
        dataset = TextDataset(
            df,
            tokenizer,
            config.max_length,
            lexicon_tokens=lexicon_tokens,
            top_tokens_map=top_tokens,
            risk_mode=config.risk_mode if config.risk_mode != "ungated" else "hybrid",
            use_actor_only_mask=config.actor_only_masking,
        )
        out[split_name] = DataLoader(dataset, batch_size=config.batch_size, shuffle=(split_name == "train"), num_workers=0, pin_memory=True)
    return out


def auxiliary_ce_loss(bundle, labels: torch.Tensor) -> torch.Tensor:
    losses = []
    for logits in bundle.aux_logits.values():
        losses.append(torch.nn.functional.cross_entropy(logits, labels))
    if not losses:
        return torch.zeros((), device=labels.device)
    return torch.stack(losses).mean()


def select_metric(dataset: str, evals: dict[str, Any], identity_cols: list[str] | None = None) -> tuple[float, dict[str, Any]]:
    if dataset == "imdb":
        metrics = {
            "val_id": classification_metrics(evals["val_id"]["logits"], evals["val_id"]["labels"]),
            "val_ood": classification_metrics(evals["val_ood"]["logits"], evals["val_ood"]["labels"]),
        }
        return metrics["val_ood"]["accuracy"], metrics
    metrics = civil_metrics(
        evals["val"]["logits"],
        evals["val"]["labels"],
        identity_flags={col: evals["val"][col] for col in identity_cols or []},
    )
    return metrics["worst_group_f1"], {"val": metrics}


def erm_cache_path(config: TrainConfig, seed: int) -> Path:
    base = "imdb_erm" if config.dataset == "imdb" else "civilcomments_erm"
    return ROOT / "exp" / base / f"top_tokens_seed{seed}.json"


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def train_single_seed(
    config: TrainConfig,
    seed: int,
    out_dir: Path,
    erm_val_reference: float | None = None,
    retrain: bool = True,
) -> dict[str, Any]:
    set_seed(seed)
    start = now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = load_tokenizer(MODEL_NAME)
    dfs, _lexicon, lexicon_tokens = prepare_data(config.dataset)
    identity_cols = identity_columns_for_dataset(config.dataset)
    seed_dir = out_dir / f"seed_{seed}"
    ensure_dir(seed_dir)
    log_dir = ensure_dir(out_dir / "logs")
    epoch_log_path = seed_dir / "epoch_metrics.jsonl"
    if retrain and epoch_log_path.exists():
        epoch_log_path.unlink()
    text_log_path = log_dir / f"run_seed{seed}_{int(time.time())}.log"
    text_log_path.write_text("", encoding="utf-8")

    top_tokens_path = out_dir / f"top_tokens_seed{seed}.json"
    top_tokens: dict[int, list[str]] | None = None
    if condition_uses_erm_cache(config.condition):
        top_tokens_path = erm_cache_path(config, seed)
    if top_tokens_path.exists():
        top_tokens = {int(k): v for k, v in read_json(top_tokens_path).items()}

    loaders = build_loaders(
        {k: v for k, v in dfs.items() if k in {"train", "val_id", "val_ood", "val"}}, tokenizer, config, top_tokens, lexicon_tokens
    )
    train_loader = loaders["train"]
    train_loader_eval = DataLoader(train_loader.dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = LateBindModel(MODEL_NAME).to(device)
    checkpoint_path = seed_dir / "checkpoint.pt"
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and AMP_DTYPE == torch.float16))

    total_steps = config.max_epochs * max(1, len(train_loader) // config.grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * config.warmup_ratio)),
        num_training_steps=max(1, total_steps),
    )

    train_weight_map = None
    if config.jtt:
        stage1_dir = out_dir / f"seed_{seed}" / "stage1"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage1_model = LateBindModel(MODEL_NAME).to(device)
        stage1_opt = AdamW(stage1_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        stage1_scheduler = get_linear_schedule_with_warmup(stage1_opt, 1, max(1, len(train_loader)))
        stage1_model.train()
        for batch_idx, batch in enumerate(train_loader):
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
                bundle = stage1_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = torch.nn.functional.cross_entropy(bundle.logits, batch["labels"].to(device))
            scaler.scale(loss).backward()
            scaler.step(stage1_opt)
            scaler.update()
            stage1_scheduler.step()
            stage1_opt.zero_grad(set_to_none=True)
        eval_train = evaluate_split(stage1_model, DataLoader(train_loader.dataset, batch_size=config.batch_size), device)
        mis = (eval_train["preds"] != eval_train["labels"]).astype(np.float32)
        train_weight_map = {
            int(source_id): float(weight)
            for source_id, weight in zip(eval_train["source_ids"], np.where(mis > 0, config.misclassified_weight, 1.0))
        }
        write_json(stage1_dir / "misclassified_summary.json", {"misclassified_rate": float(mis.mean())})
        del stage1_model
        torch.cuda.empty_cache()

    best = None
    if retrain or not checkpoint_path.exists():
        patience = 0
        for epoch in range(config.max_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = []
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(device.type == "cuda")):
                    main_bundle = model(batch["input_ids"], batch["attention_mask"])
                    masked_bundle = None
                    if config.use_inv:
                        masked_bundle = model(batch["masked_input_ids"], batch["masked_attention_mask"])
                    risk = batch["risk"]
                    if config.risk_mode == "ungated":
                        risk = torch.ones_like(risk)
                    if train_weight_map is not None:
                        weight = torch.tensor(
                            [train_weight_map[int(x)] for x in batch["source_id"].detach().cpu().numpy()],
                            device=device,
                            dtype=torch.float32,
                        )
                    else:
                        weight = torch.ones_like(risk)
                    loss, parts = latebind_loss(
                        main_bundle,
                        batch["labels"],
                        risk * weight,
                        masked_bundle,
                        config.lambda_late if config.use_late else 0.0,
                        config.lambda_inv if config.use_inv else 0.0,
                        config.entropy_floor,
                    )
                    aux_loss = auxiliary_ce_loss(main_bundle, batch["labels"])
                    loss = loss + aux_loss
                    parts["loss_aux"] = float(aux_loss.detach().item())
                    parts["loss_total"] = float(loss.detach().item())
                scaler.scale(loss / config.grad_accum).backward()
                if (step + 1) % config.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                running.append(parts)
                if config.smoke_steps and (step + 1) >= config.smoke_steps:
                    break

            eval_input = {k: loaders[k] for k in loaders if k != "train"}
            evals = {name: evaluate_split(model, loader, device) for name, loader in eval_input.items()}
            selection_score, selection_metrics = select_metric(config.dataset, evals, identity_cols)
            if config.dataset == "imdb" and erm_val_reference is not None:
                if selection_metrics["val_id"]["accuracy"] < (erm_val_reference - 0.01):
                    selection_score = -1.0
            loss_frame = pd.DataFrame(running)
            epoch_record = {
                "epoch": epoch,
                "selection_score": float(selection_score),
                "train_steps": int(len(running)),
                "loss_ce_mean": float(loss_frame["loss_ce"].mean()) if not loss_frame.empty else float("nan"),
                "loss_late_mean": float(loss_frame["loss_late"].mean()) if not loss_frame.empty else float("nan"),
                "loss_inv_mean": float(loss_frame["loss_inv"].mean()) if not loss_frame.empty else float("nan"),
                "loss_aux_mean": float(loss_frame["loss_aux"].mean()) if not loss_frame.empty else float("nan"),
                "loss_total_mean": float(loss_frame["loss_total"].mean()) if not loss_frame.empty else float("nan"),
                "selection_metrics": selection_metrics,
            }
            append_jsonl(epoch_log_path, epoch_record)
            with text_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
            if best is None or selection_score > best["selection_score"]:
                best = {
                    "selection_score": float(selection_score),
                    "epoch": epoch,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "selection_metrics": selection_metrics,
                    "train_losses": running,
                }
                patience = 0
            else:
                patience += 1
                if patience > 1:
                    break
            if config.smoke_steps:
                break
        assert best is not None
        model.load_state_dict(best["state_dict"])
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        best = {
            "selection_score": float("nan"),
            "epoch": read_json(seed_dir / "runtime.json").get("epoch", -1) if (seed_dir / "runtime.json").exists() else -1,
            "selection_metrics": read_json(seed_dir / "metrics.json") if (seed_dir / "metrics.json").exists() else {},
            "train_losses": [],
        }

    eval_dfs = {}
    if config.dataset == "imdb":
        eval_dfs = {k: v for k, v in dfs.items() if k.startswith("test_") or k.startswith("val_")}
    else:
        eval_dfs = {"val": dfs["val"], "test": dfs["test"]}
    eval_loaders = build_loaders(eval_dfs, tokenizer, config, top_tokens, lexicon_tokens)
    probe_start = time.time()
    aux_probe = {
        "aux_probe_loss": float("nan"),
        "aux_probe_minutes": 0.0,
        "aux_probe_steps": 0.0,
        "mode": "joint_training",
    }
    aux_probe["aux_probe_minutes"] = round((time.time() - probe_start) / 60.0, 4)
    evals = {name: evaluate_split(model, loader, device) for name, loader in eval_loaders.items()}
    masked_evals = {name: evaluate_split(model, loader, device, masked=True) for name, loader in eval_loaders.items()}

    ensure_dir(seed_dir)
    if retrain or not checkpoint_path.exists():
        torch.save(best["state_dict"], checkpoint_path)

    predictions_rows = []
    summary_metrics = {}
    train_risks = np.asarray([record["risk"] for record in train_loader.dataset.records], dtype=np.float32)
    train_median_risk = float(np.median(train_risks)) if len(train_risks) else 0.0
    analysis_risk_threshold = effective_risk_threshold(train_risks)
    for split_name, result in evals.items():
        final_entropy = entropy(result["probs"])
        if config.dataset == "imdb":
            metrics = classification_metrics(result["logits"], result["labels"])
            metrics.update(risk_bucket_ecr(result["aux_probs"], result["risk"], analysis_risk_threshold, result["preds"]))
            metrics["entropy_final_high_risk"] = float(final_entropy[result["risk"] > analysis_risk_threshold].mean()) if np.any(result["risk"] > analysis_risk_threshold) else float("nan")
            metrics["entropy_final_low_risk"] = float(final_entropy[result["risk"] <= analysis_risk_threshold].mean()) if np.any(result["risk"] <= analysis_risk_threshold) else float("nan")
            _conf_drop, _label_flip, mask_summary = mask_effect_metrics(result, masked_evals[split_name])
            metrics.update(mask_summary)
            summary_metrics[split_name] = metrics
        else:
            flags = {col: dfs[split_name][col].to_numpy() for col in identity_cols or []}
            result.update(flags)
            summary_metrics[split_name] = civil_metrics(result["logits"], result["labels"], flags)
            summary_metrics[split_name].update(risk_bucket_ecr(result["aux_probs"], result["risk"], analysis_risk_threshold, result["preds"]))
            summary_metrics[split_name]["entropy_final_high_risk"] = float(final_entropy[result["risk"] > analysis_risk_threshold].mean()) if np.any(result["risk"] > analysis_risk_threshold) else float("nan")
            summary_metrics[split_name]["entropy_final_low_risk"] = float(final_entropy[result["risk"] <= analysis_risk_threshold].mean()) if np.any(result["risk"] <= analysis_risk_threshold) else float("nan")
        probs = result["probs"]
        masked_probs = masked_evals[split_name]["probs"]
        conf_drop, label_flip, _ = mask_effect_metrics(result, masked_evals[split_name])
        for i in range(len(result["labels"])):
            row = {
                "split": split_name,
                "source_id": int(result["source_ids"][i]),
                "label": int(result["labels"][i]),
                "pred": int(result["preds"][i]),
                "prob_0": float(probs[i, 0]),
                "prob_1": float(probs[i, 1]),
                "final_confidence": float(probs[i].max()),
                "final_entropy": float(final_entropy[i]),
                "masked_pred": int(masked_evals[split_name]["preds"][i]),
                "masked_prob_0": float(masked_probs[i, 0]),
                "masked_prob_1": float(masked_probs[i, 1]),
                "mask_conf_drop": float(conf_drop[i]),
                "mask_label_flip": int(label_flip[i]),
                "risk": float(result["risk"][i]),
                "actor_present": int(result["actor_present"][i]),
            }
            for layer, aux_probs in result["aux_probs"].items():
                row[f"entropy_layer_{layer}"] = float(entropy(aux_probs[i : i + 1])[0])
                row[f"aux_pred_layer_{layer}"] = int(aux_probs[i].argmax())
                row[f"aux_conf_layer_{layer}"] = float(aux_probs[i].max())
            if config.dataset == "civilcomments":
                for col in identity_cols or []:
                    row[col] = int(result[col][i])
                row["identity_any"] = int(dfs[split_name]["identity_any"].iloc[i])
            predictions_rows.append(row)
    pd.DataFrame(predictions_rows).to_csv(seed_dir / "predictions.csv", index=False)
    write_json(seed_dir / "metrics.json", summary_metrics)
    runtime_payload = read_json(seed_dir / "runtime.json") if (seed_dir / "runtime.json").exists() and not retrain else {}
    if retrain or "runtime_minutes" not in runtime_payload:
        runtime_payload["runtime_minutes"] = runtime_minutes(start)
        runtime_payload["epoch"] = best["epoch"]
    runtime_payload["analysis_runtime_minutes"] = runtime_minutes(start)
    write_json(seed_dir / "runtime.json", runtime_payload)
    memory_payload = read_json(seed_dir / "memory.json") if (seed_dir / "memory.json").exists() and not retrain else {}
    current_memory = gpu_memory_stats()
    if retrain or not memory_payload:
        memory_payload.update(current_memory)
    memory_payload["analysis_peak_reserved_gb"] = current_memory.get("peak_reserved_gb", 0.0)
    write_json(seed_dir / "memory.json", memory_payload)

    diagnostics = {"aux_probe": aux_probe}
    if config.dataset == "imdb":
        diagnostics["train_median_risk"] = train_median_risk
        diagnostics["analysis_risk_threshold"] = analysis_risk_threshold
        diagnostics["proxy_validation"] = {}
        for split_name in ["val_id", "val_ood"]:
            split_diag = {}
            val_res = evals[split_name]
            masked_res = masked_evals[split_name]
            conf_drop, label_flip, mask_summary = mask_effect_metrics(val_res, masked_res)
            split_diag["mask_summary"] = mask_summary
            split_diag["n_examples"] = int(len(val_res["labels"]))
            split_diag["high_risk_fraction"] = float((val_res["risk"] > analysis_risk_threshold).mean())
            split_diag["correctness_rate"] = float((val_res["preds"] == val_res["labels"]).mean())
            split_diag["layers"] = {}
            final_conf = val_res["probs"].max(axis=1)
            correctness = (val_res["preds"] == val_res["labels"]).astype(int)
            for layer, probs in val_res["aux_probs"].items():
                split_diag["layers"][str(layer)] = proxy_validation(
                    intermediate_entropy=entropy(probs),
                    mask_conf_drop=conf_drop,
                    label_flip=label_flip,
                    actor_presence=val_res["actor_present"],
                    final_confidence=final_conf,
                    correctness=correctness,
                )
                split_diag["layers"][str(layer)].update(
                    {
                        "ecr_overall": early_commitment_rate(probs, final_preds=val_res["preds"]),
                        "ecr_high_risk": early_commitment_rate(probs[val_res["risk"] > analysis_risk_threshold], final_preds=val_res["preds"][val_res["risk"] > analysis_risk_threshold]) if np.any(val_res["risk"] > analysis_risk_threshold) else float("nan"),
                        "ecr_low_risk": early_commitment_rate(probs[val_res["risk"] <= analysis_risk_threshold], final_preds=val_res["preds"][val_res["risk"] <= analysis_risk_threshold]) if np.any(val_res["risk"] <= analysis_risk_threshold) else float("nan"),
                    }
                )
            diagnostics["proxy_validation"][split_name] = split_diag
    else:
        diagnostics["train_median_risk"] = train_median_risk
        diagnostics["analysis_risk_threshold"] = analysis_risk_threshold
        diagnostics["ecr_by_identity_any"] = {}
        for split_name in ["val", "test"]:
            split_res = evals[split_name]
            if "identity_any" not in dfs[split_name]:
                continue
            split_diag = {}
            identity_any = dfs[split_name]["identity_any"].to_numpy()
            for layer, probs in split_res["aux_probs"].items():
                split_diag[f"layer_{layer}_identity_present"] = early_commitment_rate(probs[identity_any == 1], final_preds=split_res["preds"][identity_any == 1]) if np.any(identity_any == 1) else float("nan")
                split_diag[f"layer_{layer}_identity_absent"] = early_commitment_rate(probs[identity_any == 0], final_preds=split_res["preds"][identity_any == 0]) if np.any(identity_any == 0) else float("nan")
            diagnostics["ecr_by_identity_any"][split_name] = split_diag
    write_json(seed_dir / "diagnostics.json", diagnostics)

    if config.dataset in {"imdb", "civilcomments"} and top_tokens is None and not config.smoke_steps:
        attr_splits = {
            "train": dfs["train"],
        }
        if config.dataset == "imdb":
            attr_splits.update({"val_id": dfs["val_id"], "val_ood": dfs["val_ood"]})
        else:
            attr_splits.update({"val": dfs["val"]})
        top_tokens = {}
        for _, df in attr_splits.items():
            top_tokens.update(token_attributions(model, tokenizer, df, config.max_length, device))
        write_json(out_dir / f"top_tokens_seed{seed}.json", {str(k): v for k, v in top_tokens.items()})

    return {
        "seed": seed,
        "selection_metrics": best["selection_metrics"],
        "summary_metrics": summary_metrics,
        "diagnostics": diagnostics,
        "runtime_minutes": runtime_minutes(start),
        "peak_memory_gb": gpu_memory_stats()["peak_reserved_gb"],
    }


def aggregate_condition(condition: str, out_dir: Path) -> dict[str, Any]:
    config = get_condition_config(condition)
    seed_results = []
    completed_seeds = []
    for seed in config.seeds:
        seed_path = out_dir / f"seed_{seed}" / "metrics.json"
        if seed_path.exists():
            seed_results.append(read_json(seed_path))
            completed_seeds.append(seed)
    if not seed_results:
        return {}
    if config.dataset == "imdb":
        aggregate = {}
        for split in seed_results[0]:
            aggregate[split] = summarize_seed_metrics([seed_metrics[split] for seed_metrics in seed_results])
    else:
        aggregate = {
            split: summarize_seed_metrics([seed_metrics[split] for seed_metrics in seed_results])
            for split in seed_results[0]
        }
    payload = {
        "experiment": condition,
        "config": asdict(config),
        "completed_seeds": completed_seeds,
        "planned_seeds": config.seeds,
        "status": "complete" if completed_seeds == config.seeds else "partial",
        "metrics": aggregate,
    }
    write_json(out_dir / "results.json", payload)
    return payload


def build_root_results() -> dict[str, Any]:
    conditions = [
        "imdb_tfidf_lr",
        "imdb_erm",
        "imdb_masker",
        "imdb_jtt",
        "imdb_latebind",
        "imdb_no_late_term",
        "imdb_no_invariance",
        "imdb_ungated_entropy",
        "imdb_lexicon_only_risk",
        "imdb_attribution_only_risk",
        "imdb_random_token_risk",
        "imdb_actor_only_masking",
        "civilcomments_erm",
        "civilcomments_jtt",
        "civilcomments_latebind",
    ]
    payload: dict[str, Any] = {"conditions": {}, "bootstrap": {}, "matrix_status": {}}
    for condition in conditions:
        path = ROOT / "exp" / condition / "results.json"
        if path.exists():
            payload["conditions"][condition] = read_json(path)

    planned_core = {
        "imdb_primary": ["imdb_erm", "imdb_masker", "imdb_jtt", "imdb_latebind"],
        "imdb_ablations": [
            "imdb_no_late_term",
            "imdb_no_invariance",
            "imdb_ungated_entropy",
            "imdb_lexicon_only_risk",
            "imdb_attribution_only_risk",
            "imdb_random_token_risk",
            "imdb_actor_only_masking",
        ],
        "civilcomments_check": ["civilcomments_erm", "civilcomments_jtt", "civilcomments_latebind"],
    }
    for key, condition_names in planned_core.items():
        payload["matrix_status"][key] = {
            name: payload["conditions"].get(name, {}).get("completed_seeds", [])
            for name in condition_names
        }

    def common_completed_seeds(a_condition: str, b_condition: str) -> list[int]:
        a = set(payload["conditions"].get(a_condition, {}).get("completed_seeds", []))
        b = set(payload["conditions"].get(b_condition, {}).get("completed_seeds", []))
        return sorted(a & b)

    def load_split_correctness(condition: str, split: str, seeds: list[int]) -> np.ndarray | None:
        arrays = []
        for seed in seeds:
            pred_path = ROOT / "exp" / condition / f"seed_{seed}" / "predictions.csv"
            if not pred_path.exists():
                continue
            df = pd.read_csv(pred_path)
            split_df = df[df["split"] == split]
            arrays.append((split_df["pred"].to_numpy() == split_df["label"].to_numpy()).astype(float))
        if not arrays:
            return None
        return np.concatenate(arrays)

    compared_splits = ["test_ood", "test_actor_conflict", "test_actor_scrubbed", "test_no_shortcut"]
    seeds = common_completed_seeds("imdb_latebind", "imdb_erm")
    if seeds:
        for split in compared_splits:
            a = load_split_correctness("imdb_latebind", split, seeds)
            b = load_split_correctness("imdb_erm", split, seeds)
            if a is not None and b is not None and len(a) == len(b):
                payload["bootstrap"][f"latebind_vs_erm_{split}"] = {
                    **bootstrap_ci(a, b, n_resamples=1000, seed=13),
                    "completed_seeds": seeds,
                }

    stronger = None
    stronger_score = -float("inf")
    for candidate in ["imdb_masker", "imdb_jtt"]:
        metric = (
            payload["conditions"]
            .get(candidate, {})
            .get("metrics", {})
            .get("test_ood", {})
            .get("accuracy", {})
            .get("mean")
        )
        if metric is not None and metric > stronger_score:
            stronger = candidate
            stronger_score = metric
    if stronger is not None:
        seeds = common_completed_seeds("imdb_latebind", stronger)
        for split in compared_splits:
            a = load_split_correctness("imdb_latebind", split, seeds)
            b = load_split_correctness(stronger, split, seeds)
            if a is not None and b is not None and len(a) == len(b):
                payload["bootstrap"][f"latebind_vs_{stronger}_{split}"] = {
                    **bootstrap_ci(a, b, n_resamples=1000, seed=13),
                    "completed_seeds": seeds,
                }
    write_json(ROOT / "results.json", payload)
    return payload


def create_summary_artifacts() -> None:
    rows = []
    prediction_rows = []
    summary_dir = ensure_dir(ROOT / "results" / "summary")
    for condition_dir in sorted((ROOT / "exp").glob("*")):
        if not condition_dir.is_dir():
            continue
        results_path = condition_dir / "results.json"
        if results_path.exists():
            result = read_json(results_path)
            metrics = result.get("metrics", {})
            for split, split_metrics in metrics.items():
                for metric_name, summary in split_metrics.items():
                    if isinstance(summary, dict) and "mean" in summary:
                        rows.append(
                            {
                                "experiment": result["experiment"],
                                "split": split,
                                "metric": metric_name,
                                "mean": summary["mean"],
                                "std": summary["std"],
                                "values": json.dumps(summary["values"]),
                            }
                        )
        for pred_path in condition_dir.glob("seed_*/predictions.csv"):
            pred_df = pd.read_csv(pred_path)
            pred_df["experiment"] = condition_dir.name
            pred_df["seed"] = int(pred_path.parent.name.split("_")[-1])
            diag_path = pred_path.parent / "diagnostics.json"
            if diag_path.exists():
                threshold = float(read_json(diag_path).get("analysis_risk_threshold", pred_df["risk"].median()))
            else:
                threshold = float(pred_df["risk"].median())
            pred_df["risk_bucket"] = np.where(pred_df["risk"] > threshold, "high_risk", "low_risk")
            prediction_rows.append(pred_df)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_dir / "summary.csv", index=False)

    diagnostics_path = summary_dir / "diagnostics.parquet"
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_parquet(diagnostics_path, index=False)
    elif diagnostics_path.exists():
        diagnostics_path.unlink()

    runtime_rows = []
    memory_rows = []
    for condition_dir in sorted((ROOT / "exp").glob("*")):
        if not condition_dir.is_dir():
            continue
        runtimes = [read_json(path).get("runtime_minutes") for path in condition_dir.glob("seed_*/runtime.json")]
        memories = [read_json(path).get("peak_reserved_gb") for path in condition_dir.glob("seed_*/memory.json")]
        if not runtimes and (condition_dir / "results.json").exists():
            result_payload = read_json(condition_dir / "results.json")
            if "runtime_minutes" in result_payload:
                runtimes = [result_payload["runtime_minutes"]]
        if runtimes:
            runtime_rows.append({"experiment": condition_dir.name, "runtime_minutes": float(np.nanmean(runtimes))})
        if memories:
            memory_rows.append({"experiment": condition_dir.name, "peak_memory_gb": float(np.nanmean(memories))})
    runtime_df = pd.DataFrame(runtime_rows).set_index("experiment") if runtime_rows else pd.DataFrame()
    memory_df = pd.DataFrame(memory_rows).set_index("experiment") if memory_rows else pd.DataFrame()

    if not summary_df.empty:
        imdb_experiments = [
            "imdb_tfidf_lr",
            "imdb_erm",
            "imdb_masker",
            "imdb_jtt",
            "imdb_latebind",
            "imdb_no_late_term",
            "imdb_no_invariance",
            "imdb_ungated_entropy",
            "imdb_lexicon_only_risk",
            "imdb_attribution_only_risk",
            "imdb_random_token_risk",
            "imdb_actor_only_masking",
        ]
        imdb_metrics = ["accuracy", "ece", "brier"]
        imdb_splits = ["test_id", "test_ood", "test_no_shortcut", "test_actor_conflict", "test_actor_scrubbed"]
        imdb_table = (
            summary_df[
                summary_df["experiment"].isin(imdb_experiments)
                & summary_df["split"].isin(imdb_splits)
                & summary_df["metric"].isin(imdb_metrics)
            ]
            .pivot_table(index="experiment", columns=["split", "metric"], values="mean", aggfunc="first")
            .sort_index()
        )
        if not imdb_table.empty:
            imdb_table.columns = ["__".join(col) for col in imdb_table.columns]
            imdb_table = imdb_table.join(runtime_df, how="left").join(memory_df, how="left")
            imdb_table.to_csv(summary_dir / "table1_imdb.csv")

        civil_table = (
            summary_df[
                summary_df["experiment"].str.startswith("civilcomments")
                & (summary_df["split"] == "test")
                & summary_df["metric"].isin(["overall_f1", "worst_group_f1", "macro_f1", "ece"])
            ]
            .pivot_table(index="experiment", columns="metric", values="mean", aggfunc="first")
            .sort_index()
        )
        if not civil_table.empty:
            civil_table = civil_table.join(runtime_df, how="left").join(memory_df, how="left")
            civil_table.to_csv(summary_dir / "table2_civilcomments.csv")


def plot_figures() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    summary_csv = ROOT / "results" / "summary" / "summary.csv"
    if not summary_csv.exists():
        return
    df = pd.read_csv(summary_csv)
    root_results = read_json(ROOT / "results.json") if (ROOT / "results.json").exists() else {}
    fig_dir = ensure_dir(ROOT / "results" / "summary" / "figures")
    public_fig_dir = ensure_dir(ROOT / "figures")
    primary_complete = all(
        len(root_results.get("matrix_status", {}).get("imdb_primary", {}).get(name, [])) == 3
        for name in ["imdb_erm", "imdb_masker", "imdb_jtt", "imdb_latebind"]
    )

    imdb = df[
        df["experiment"].isin(["imdb_erm", "imdb_masker", "imdb_jtt", "imdb_latebind"])
        & df["split"].isin(["test_id", "test_ood", "test_no_shortcut", "test_actor_conflict", "test_actor_scrubbed"])
        & (df["metric"] == "accuracy")
    ]
    if not imdb.empty and primary_complete:
        plt.figure(figsize=(12, 5))
        ax = sns.barplot(data=imdb, x="split", y="mean", hue="experiment", errorbar=None)
        seed_rows = []
        for experiment in ["imdb_erm", "imdb_masker", "imdb_jtt", "imdb_latebind"]:
            for seed_dir in sorted((ROOT / "exp" / experiment).glob("seed_*")):
                metrics = read_json(seed_dir / "metrics.json")
                seed = int(seed_dir.name.split("_")[-1])
                for split in ["test_id", "test_ood", "test_no_shortcut", "test_actor_conflict", "test_actor_scrubbed"]:
                    seed_rows.append({"experiment": experiment, "seed": seed, "split": split, "accuracy": metrics[split]["accuracy"]})
        seed_df = pd.DataFrame(seed_rows)
        if not seed_df.empty:
            sns.stripplot(data=seed_df, x="split", y="accuracy", hue="experiment", dodge=True, alpha=0.7, size=4, linewidth=0.4, edgecolor="black", ax=ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:4], labels[:4], title="experiment", ncol=2)
        plt.tight_layout()
        out_path = fig_dir / "figure1_imdb_grouped_bars.png"
        plt.savefig(out_path, dpi=200)
        shutil.copy2(out_path, public_fig_dir / out_path.name)
        plt.close()

    diag_path = ROOT / "results" / "summary" / "diagnostics.parquet"
    if not diag_path.exists():
        return
    diag = pd.read_parquet(diag_path)
    if diag.empty:
        return

    entropy_df = diag[
        diag["experiment"].isin(["imdb_erm", "imdb_ungated_entropy", "imdb_latebind"])
        & diag["split"].isin(["val_id", "val_ood"])
    ].copy()
    if not entropy_df.empty and all(col in entropy_df.columns for col in ["entropy_layer_4", "entropy_layer_8", "final_entropy"]):
        long_rows = []
        for _, row in entropy_df.iterrows():
            for layer_name, label in [("entropy_layer_4", "4"), ("entropy_layer_8", "8"), ("final_entropy", "final")]:
                long_rows.append(
                    {
                        "experiment": row["experiment"],
                        "seed": row["seed"],
                        "risk_bucket": row["risk_bucket"],
                        "layer": label,
                        "entropy": row[layer_name],
                    }
                )
        entropy_long = pd.DataFrame(long_rows)
        entropy_long.to_csv(ROOT / "results" / "summary" / "figure2_entropy_profiles.csv", index=False)
        plt.figure(figsize=(9, 4.5))
        sns.lineplot(
            data=entropy_long,
            x="layer",
            y="entropy",
            hue="experiment",
            style="risk_bucket",
            errorbar=("ci", 95),
            marker="o",
        )
        plt.tight_layout()
        out_path = fig_dir / "figure2_layerwise_entropy.png"
        plt.savefig(out_path, dpi=200)
        shutil.copy2(out_path, public_fig_dir / out_path.name)
        plt.close()

    scatter_df = diag[(diag["experiment"] == "imdb_latebind") & diag["split"].isin(["val_id", "val_ood"])].copy()
    if not scatter_df.empty and all(col in scatter_df.columns for col in ["entropy_layer_8", "mask_conf_drop", "risk_bucket"]):
        sample_parts = []
        for (_, _), frame in scatter_df.groupby(["split", "risk_bucket"]):
            sample_parts.append(frame.sample(n=min(len(frame), 1200), random_state=13))
        sample_df = pd.concat(sample_parts, ignore_index=True) if sample_parts else scatter_df
        g = sns.FacetGrid(sample_df, col="split", hue="risk_bucket", height=4, aspect=1.1, sharex=True, sharey=True)
        g.map_dataframe(sns.scatterplot, x="entropy_layer_8", y="mask_conf_drop", alpha=0.35, s=18)
        for ax, split in zip(g.axes.flat, ["val_id", "val_ood"]):
            panel = scatter_df[scatter_df["split"] == split]
            rho = panel["entropy_layer_8"].corr(panel["mask_conf_drop"], method="spearman")
            ax.text(0.05, 0.95, f"rho={rho:.3f}", transform=ax.transAxes, ha="left", va="top")
        g.add_legend()
        out_path = fig_dir / "figure3_proxy_validation.png"
        g.savefig(out_path, dpi=200)
        shutil.copy2(out_path, public_fig_dir / out_path.name)
        plt.close("all")

    heatmap_rows = []
    row_map = {
        "imdb_latebind": "LateBind",
        "imdb_no_late_term": "no_late_term",
        "imdb_no_invariance": "no_invariance",
        "imdb_ungated_entropy": "ungated_entropy",
        "imdb_lexicon_only_risk": "lexicon_only_risk",
        "imdb_attribution_only_risk": "attribution_only_risk",
        "imdb_random_token_risk": "random_token_risk",
        "imdb_actor_only_masking": "actor_only_masking",
    }
    for experiment, label in row_map.items():
        for metric_name, split in [
            ("test_ood", "test_ood"),
            ("test_no_shortcut", "test_no_shortcut"),
            ("test_actor_conflict", "test_actor_conflict"),
        ]:
            value = df[(df["experiment"] == experiment) & (df["split"] == split) & (df["metric"] == "accuracy")]["mean"]
            if not value.empty:
                heatmap_rows.append({"experiment": label, "metric": metric_name, "value": float(value.iloc[0])})
        ece_value = df[(df["experiment"] == experiment) & (df["split"] == "test_ood") & (df["metric"] == "ece")]["mean"]
        if not ece_value.empty:
            heatmap_rows.append({"experiment": label, "metric": "ece", "value": float(ece_value.iloc[0])})
        ecr_value = df[(df["experiment"] == experiment) & (df["split"] == "val_ood") & (df["metric"] == "ecr_layer_8_high_risk")]["mean"]
        if not ecr_value.empty:
            heatmap_rows.append({"experiment": label, "metric": "val_high_risk_ecr", "value": float(ecr_value.iloc[0])})
    if heatmap_rows:
        plt.figure(figsize=(8.5, 4.5))
        heatmap_df = pd.DataFrame(heatmap_rows).pivot(index="experiment", columns="metric", values="value")
        sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="crest")
        plt.tight_layout()
        out_path = fig_dir / "figure4_ablation_heatmap.png"
        plt.savefig(out_path, dpi=200)
        shutil.copy2(out_path, public_fig_dir / out_path.name)
        plt.close()


def run_condition(condition: str) -> dict[str, Any]:
    out_dir = ROOT / "exp" / condition
    ensure_dir(out_dir)
    capture_environment(ROOT / "results" / "environment")

    if condition == "imdb_tfidf_lr":
        payload = fit_tfidf_imdb(out_dir)
        build_root_results()
        create_summary_artifacts()
        plot_figures()
        return payload

    config = get_condition_config(condition)
    payloads = []
    for seed in config.seeds:
        erm_reference = None
        if config.dataset == "imdb" and condition not in {"imdb_erm", "imdb_erm_smoke_test", "imdb_latebind_smoke_test"}:
            erm_seed_metrics = ROOT / "exp" / "imdb_erm" / f"seed_{seed}" / "metrics.json"
            if erm_seed_metrics.exists():
                erm_reference = float(read_json(erm_seed_metrics)["val_id"]["accuracy"])
        payloads.append(train_single_seed(config, seed, out_dir, erm_reference))
    aggregate = aggregate_condition(condition, out_dir)
    build_root_results()
    create_summary_artifacts()
    plot_figures()
    return aggregate


def refresh_condition(condition: str, retrain: bool = False) -> dict[str, Any]:
    out_dir = ROOT / "exp" / condition
    config = get_condition_config(condition)
    payloads = []
    for seed in config.seeds:
        erm_reference = None
        if config.dataset == "imdb" and condition not in {"imdb_erm", "imdb_erm_smoke_test", "imdb_latebind_smoke_test"}:
            erm_seed_metrics = ROOT / "exp" / "imdb_erm" / f"seed_{seed}" / "metrics.json"
            if erm_seed_metrics.exists():
                erm_reference = float(read_json(erm_seed_metrics)["val_id"]["accuracy"])
        payloads.append(train_single_seed(config, seed, out_dir, erm_reference, retrain=retrain))
    aggregate = aggregate_condition(condition, out_dir)
    build_root_results()
    create_summary_artifacts()
    plot_figures()
    return aggregate


def condition_uses_erm_cache(condition: str) -> bool:
    if condition.startswith("imdb_"):
        return condition not in {"imdb_erm", "imdb_erm_smoke_test", "imdb_latebind_smoke_test", "imdb_tfidf_lr"}
    return condition == "civilcomments_latebind"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    run_condition(args.condition)


if __name__ == "__main__":
    main()
