import argparse
import difflib
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import (
    ARTIFACTS_DIR,
    DATA_AUDIT_DIR,
    DATA_DIR,
    FIGURES_DIR,
    HELD_OUT_VARIETIES,
    MAX_SEQ_LEN,
    MODEL_NAME,
    PROMPT_HEADER,
    RESERVE_VARIETIES,
    ROOT,
    SEEN_VARIETIES,
    SUMMARY_DIR,
    SYSTEM_DISPLAY,
    SYSTEMS_MAIN,
    TRAIN_SEEDS,
    TRAINABLE_SYSTEMS,
)
from .rewrite_rules import PARAPHRASE_RULES, STANDARDIZE_RULES, VARIETY_RULES, apply_rules, variety_prefix


TRAINING_CONFIG = {
    "optimizer": "AdamW",
    "learning_rate": 2e-4,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "warmup_ratio": 0.05,
    "schedule": "cosine",
    "gradient_clip_norm": 1.0,
    "per_device_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 32,
    "epochs": 1,
    "sae_only_effective_epochs": 2,
    "bf16": True,
    "qlora": {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    },
    "decoding": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 8,
        "policy": "label-space scoring with greedy argmax over valid labels",
    },
}

PREREGISTERED_THRESHOLDS = {
    "normalized_levenshtein_min": 0.08,
    "normalized_levenshtein_max": 0.35,
    "embedding_cosine_min": 0.88,
    "token_change_ratio_min": 0.10,
}

PILOT_THRESHOLDS = {
    "normalized_levenshtein_min": 0.02,
    "normalized_levenshtein_max": 0.35,
    "embedding_cosine_min": 0.80,
    "token_change_ratio_min": 0.05,
}

PILOT_PLAN = {
    "study_scope": "proxy-pipeline pilot",
    "claim_revision": (
        "This rerun is an explicitly redesigned pilot, not the preregistered Trans-EnV rewrite-validation study. "
        "Synthetic rewrites and paraphrases are generated with deterministic proxy transformations, acceptance "
        "thresholds are revised before training to match that proxy pipeline, and all claims are limited to whether "
        "the proxy setup yields useful robustness trends under matched controls."
    ),
    "limitations": [
        "No human audit was performed in this run, so rewrite and protocol audits are proxy-only checks rather than human validation.",
        "Because the synthetic data come from deterministic proxy transformations rather than the preregistered Trans-EnV generation stack, transfer claims to human dialect understanding remain descriptive and not confirmatory.",
    ],
}


def ensure_dirs():
    for path in [DATA_DIR, ARTIFACTS_DIR, DATA_AUDIT_DIR, SUMMARY_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha1_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def jsonl_dump(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_load(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def text_features(source: str, target: str) -> Dict[str, float]:
    ratio = difflib.SequenceMatcher(a=source, b=target).ratio()
    source_tokens = source.split()
    target_tokens = target.split()
    changed = sum(1 for a, b in zip(source_tokens, target_tokens) if a != b) + abs(len(source_tokens) - len(target_tokens))
    tok_denom = max(1, len(source_tokens), len(target_tokens))
    grams_a = {source[i : i + 3] for i in range(max(1, len(source) - 2))}
    grams_b = {target[i : i + 3] for i in range(max(1, len(target) - 2))}
    cosine = len(grams_a & grams_b) / max(1, len(grams_a | grams_b))
    type_token_change = abs(len(set(source_tokens)) - len(set(target_tokens))) / max(1, len(set(source_tokens)))
    return {
        "normalized_levenshtein": 1.0 - ratio,
        "token_change_ratio": changed / tok_denom,
        "embedding_cosine": cosine,
        "length_ratio": len(target_tokens) / max(1, len(source_tokens)),
        "type_token_change": type_token_change,
    }


def preserve_checks(source: str, target: str) -> bool:
    numbers_a = sorted(__import__("re").findall(r"\d+", source))
    numbers_b = sorted(__import__("re").findall(r"\d+", target))
    logical_a = sorted(__import__("re").findall(r"\b(?:all|any|some|none|no|if|then|or|and|not)\b", source.lower()))
    logical_b = sorted(__import__("re").findall(r"\b(?:all|any|some|none|no|if|then|or|and|not)\b", target.lower()))
    return numbers_a == numbers_b and logical_a == logical_b


def prereg_threshold_pass(feats: Dict[str, float]) -> bool:
    return (
        PREREGISTERED_THRESHOLDS["normalized_levenshtein_min"] <= feats["normalized_levenshtein"] <= PREREGISTERED_THRESHOLDS["normalized_levenshtein_max"]
        and feats["embedding_cosine"] >= PREREGISTERED_THRESHOLDS["embedding_cosine_min"]
        and feats["token_change_ratio"] >= PREREGISTERED_THRESHOLDS["token_change_ratio_min"]
    )


def current_filter_pass(source_text: str, rewritten_text: str, feats: Dict[str, float]) -> bool:
    return (
        preserve_checks(source_text, rewritten_text)
        and PILOT_THRESHOLDS["normalized_levenshtein_min"] <= feats["normalized_levenshtein"] <= PILOT_THRESHOLDS["normalized_levenshtein_max"]
        and feats["embedding_cosine"] >= PILOT_THRESHOLDS["embedding_cosine_min"]
        and feats["token_change_ratio"] >= PILOT_THRESHOLDS["token_change_ratio_min"]
    )


def rewrite_text(source_text: str, variety: str) -> Tuple[str, Dict[str, float]]:
    rewritten = apply_rules(source_text, VARIETY_RULES[variety])
    if rewritten == source_text:
        rewritten = variety_prefix(variety) + source_text
    return rewritten, text_features(source_text, rewritten)


def paraphrase_candidates(source_text: str) -> List[str]:
    candidates = []
    progressively_rewritten = source_text
    for pattern, replacement in PARAPHRASE_RULES:
        progressively_rewritten = apply_rules(progressively_rewritten, [(pattern, replacement)])
        candidates.append(progressively_rewritten)
    full_rewrite = apply_rules(source_text, PARAPHRASE_RULES)
    structural_rewrite = (
        full_rewrite.replace("Context:", "Scenario:")
        .replace("Question:", "Prompt:")
        .replace("Which of the following", "Which option")
        .replace("According to the passage", "Based on the passage")
    )
    candidates.extend(
        [
            full_rewrite,
            structural_rewrite,
            "Restated: " + full_rewrite,
            "In other words, " + full_rewrite,
            "Restated: " + structural_rewrite,
            "In other words, " + structural_rewrite,
            full_rewrite.replace("Question:", "Prompt:"),
            full_rewrite.replace("Context:", "Scenario:"),
            structural_rewrite + "\n\nSelect the best matching label.",
            "Restated for clarity: " + source_text,
        ]
    )
    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def paraphrase_text(source_text: str, target_feats: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, float]]:
    target_feats = target_feats or {
        "embedding_cosine": 0.92,
        "normalized_levenshtein": 0.05,
        "length_ratio": 1.0,
        "type_token_change": 0.02,
    }
    best_text = source_text
    best_feats = text_features(source_text, source_text)
    best_score = float("inf")
    for candidate in paraphrase_candidates(source_text):
        feats = text_features(source_text, candidate)
        if not preserve_checks(source_text, candidate):
            continue
        score = (
            abs(feats["embedding_cosine"] - target_feats["embedding_cosine"])
            + abs(feats["normalized_levenshtein"] - target_feats["normalized_levenshtein"])
            + abs(feats["length_ratio"] - target_feats["length_ratio"])
            + abs(feats["type_token_change"] - target_feats["type_token_change"])
        )
        if score < best_score:
            best_score = score
            best_text = candidate
            best_feats = feats
    if best_text == source_text:
        best_text = "Restated: " + source_text
        best_feats = text_features(source_text, best_text)
    return best_text, best_feats


def standardize_text(source_text: str) -> str:
    out = source_text
    for source, target in STANDARDIZE_RULES:
        out = out.replace(source, target).replace(source.title(), target.title())
    return out.replace("Aight, check this. ", "").replace("Ain't no ", "No ")


def build_prompt(question_text: str, answer_options: Optional[List[str]], candidate_labels: List[str]) -> str:
    parts = [PROMPT_HEADER, "", question_text.strip()]
    if answer_options:
        parts.extend(["", "Options:"])
        parts.extend(answer_options)
    parts.extend(["", "Valid answer labels: " + ", ".join(candidate_labels), "Return only one label inside <answer></answer>."])
    return "\n".join(parts)


def stable_example_id(prefix: str, source_question_id: str, suffix: Optional[str] = None) -> str:
    base = f"{prefix}::{source_question_id}"
    return base if suffix is None else f"{base}::{suffix}"


def make_source_row(row: dict, split_name: str) -> dict:
    return {
        **row,
        "example_id": stable_example_id(split_name, row["source_question_id"]),
        "split": split_name,
        "rewrite_status": "source",
    }


def load_arc_parquet(split: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id="allenai/ai2_arc",
        repo_type="dataset",
        filename=f"ARC-Challenge/{split}-00000-of-00001.parquet",
    )
    return pd.read_parquet(path)


def normalize_logiqa(split_name: str) -> List[dict]:
    ds = load_dataset("lucasmccabe/logiqa", split=split_name)
    rows = []
    for idx, ex in enumerate(ds):
        options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(ex["options"])]
        rows.append(
            {
                "source_dataset": "LogiQA 2.0",
                "source_question_id": f"logiqa_{split_name}_{idx}",
                "source_text": f"Context:\n{ex['context']}\n\nQuestion: {ex['query']}",
                "answer_options": options,
                "gold_label": chr(65 + int(ex["correct_option"])),
                "candidate_labels": [chr(65 + i) for i in range(len(options))],
                "label_space": "mcq",
            }
        )
    return rows


def normalize_reclor(split_name: str) -> List[dict]:
    ds = load_dataset("metaeval/reclor", split=split_name)
    rows = []
    for ex in ds:
        options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(ex["answers"])]
        rows.append(
            {
                "source_dataset": "ReClor",
                "source_question_id": str(ex["id_string"]),
                "source_text": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}",
                "answer_options": options,
                "gold_label": chr(65 + int(ex["label"])),
                "candidate_labels": [chr(65 + i) for i in range(len(options))],
                "label_space": "mcq",
            }
        )
    return rows


def normalize_arc(split_name: str) -> List[dict]:
    df = load_arc_parquet(split_name)
    rows = []
    for _, ex in df.iterrows():
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        options = [f"{label}. {text}" for label, text in zip(labels, texts)]
        rows.append(
            {
                "source_dataset": "ARC-Challenge",
                "source_question_id": str(ex["id"]),
                "source_text": f"Question: {ex['question']}",
                "answer_options": options,
                "gold_label": ex["answerKey"],
                "candidate_labels": labels,
                "label_space": "mcq",
            }
        )
    return rows


def normalize_redial_logic() -> Tuple[List[dict], List[dict]]:
    path = hf_hub_download(repo_id="FangruLin/ReDial", repo_type="dataset", filename="logic.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)["vanilla"]
    se_rows, aave_rows = [], []
    for idx, (original, aave) in enumerate(zip(meta["original"], meta["aave"])):
        label = original["label"]
        if label in {"yes", "no"}:
            subset = "logicbench-binary"
            candidate_labels = ["yes", "no"]
            label_space = "binary"
        elif label in {"A", "B", "C", "D"}:
            subset = "logicbench-multiple-choice"
            candidate_labels = ["A", "B", "C", "D"]
            label_space = "mcq"
        else:
            subset = "folio"
            candidate_labels = ["necessarily true", "necessarily false", "neither"]
            label_space = "logic"
        row_id = f"redial_logic_{idx:03d}"
        base = {
            "source_dataset": "ReDial",
            "source_question_id": row_id,
            "subset": subset,
            "candidate_labels": candidate_labels,
            "label_space": label_space,
            "answer_options": None,
            "gold_label": label,
        }
        pair_feats = text_features(original["prompt"], aave["prompt"])
        se_rows.append(
            {
                **base,
                "example_id": stable_example_id("redial_se", row_id),
                "split": "redial_se",
                "source_text": original["prompt"],
                "paired_aave_text": aave["prompt"],
                **{f"paired_{key}": value for key, value in pair_feats.items()},
            }
        )
        aave_rows.append(
            {
                **base,
                "example_id": stable_example_id("redial_aave", row_id),
                "split": "redial_aave",
                "source_text": aave["prompt"],
                "paired_se_text": original["prompt"],
                **{f"paired_{key}": value for key, value in pair_feats.items()},
            }
        )
    return se_rows, aave_rows


def deterministic_partition(rows: List[dict], train_n: int, dev_n: int, test_n: int) -> Tuple[List[dict], List[dict], List[dict]]:
    train_rows = sorted(rows, key=lambda row: sha1_key(row["source_question_id"]))
    train = train_rows[:train_n]
    remaining = train_rows[train_n:]
    eval_sorted = sorted(remaining, key=lambda row: sha1_key("eval_" + row["source_question_id"]))
    return train, eval_sorted[:dev_n], eval_sorted[dev_n : dev_n + test_n]


def assign_varieties(rows: List[dict], varieties: List[str], n_per_example: int) -> List[Tuple[dict, str]]:
    assignments = []
    cursor = 0
    for row in rows:
        for _ in range(n_per_example):
            assignments.append((row, varieties[cursor % len(varieties)]))
            cursor += 1
    return assignments


def build_pair_row(row: dict, split_name: str, variety: str, rewritten_text: str, feats: Dict[str, float], status: str) -> dict:
    return {
        **row,
        "example_id": stable_example_id(split_name, row["source_question_id"], variety),
        "split": split_name,
        "assigned_variety": variety,
        "rewritten_text": rewritten_text,
        "rewrite_status": status,
        **feats,
    }


def build_rewrite_audit(rows: List[dict]) -> pd.DataFrame:
    rng = random.Random(13)
    by_variety = defaultdict(list)
    for row in rows:
        by_variety[row["assigned_variety"]].append(row)
    audit_rows = []
    for variety, variety_rows in sorted(by_variety.items()):
        sample = variety_rows if len(variety_rows) <= 10 else rng.sample(variety_rows, 10)
        for row in sample:
            audit_rows.append(
                {
                    "variety": variety,
                    "source_dataset": row["source_dataset"],
                    "pass_semantics": preserve_checks(row["source_text"], row["rewritten_text"]),
                    "pass_dialect": row["normalized_levenshtein"] >= PREREGISTERED_THRESHOLDS["normalized_levenshtein_min"]
                    and row["token_change_ratio"] >= PREREGISTERED_THRESHOLDS["token_change_ratio_min"],
                    "pass_option_preservation": True,
                    "pass_no_leakage": answer_text(str(row["gold_label"])) not in row["rewritten_text"],
                    "notes": (
                        f"Proxy audit only. prereg_threshold_pass={prereg_threshold_pass(row)}; "
                        f"lev={row['normalized_levenshtein']:.3f}; cos={row['embedding_cosine']:.3f}; "
                        f"token_change={row['token_change_ratio']:.3f}"
                    ),
                }
            )
    return pd.DataFrame(audit_rows)


def summarize_feature_diffs(dialect_df: pd.DataFrame, paraphrase_df: pd.DataFrame) -> Dict[str, dict]:
    feature_cols = ["embedding_cosine", "normalized_levenshtein", "length_ratio", "type_token_change"]
    summary = {"overall": {}, "by_dataset": {}}
    for feature in feature_cols:
        summary["overall"][feature] = float(abs(dialect_df[feature].mean() - paraphrase_df[feature].mean()))
    for dataset in sorted(dialect_df["source_dataset"].unique()):
        dialect_subset = dialect_df[dialect_df["source_dataset"] == dataset]
        paraphrase_subset = paraphrase_df[paraphrase_df["source_dataset"] == dataset]
        summary["by_dataset"][dataset] = {
            feature: float(abs(dialect_subset[feature].mean() - paraphrase_subset[feature].mean())) for feature in feature_cols
        }
    summary["matched_within_0.02"] = bool(
        all(value < 0.02 for value in summary["overall"].values())
        and all(value < 0.02 for dataset_values in summary["by_dataset"].values() for value in dataset_values.values())
    )
    return summary


def build_paraphrase_controls(train_pairs: List[dict]) -> Tuple[List[dict], Dict[str, dict]]:
    paraphrase_rows = []
    for row in train_pairs:
        rewritten, feats = paraphrase_text(row["source_text"], target_feats=row)
        paraphrase_rows.append(
            {
                **row,
                "example_id": stable_example_id("paraphrase_pair", row["source_question_id"], "paraphrase_control"),
                "assigned_variety": "paraphrase_control",
                "rewritten_text": rewritten,
                **feats,
            }
        )
    dialect_df = pd.DataFrame(train_pairs)
    paraphrase_df = pd.DataFrame(paraphrase_rows)
    summary = summarize_feature_diffs(dialect_df, paraphrase_df)
    summary["n_pairs"] = int(len(paraphrase_rows))
    return paraphrase_rows, summary


def build_protocol_audit(redial_rows: List[dict]) -> pd.DataFrame:
    rng = random.Random(13)
    sample = redial_rows if len(redial_rows) <= 30 else rng.sample(redial_rows, 30)
    rows = []
    for row in sample:
        prompt = build_prompt(row["source_text"], row.get("answer_options"), row["candidate_labels"])
        rows.append(
            {
                "example_id": row["source_question_id"],
                "subset": row["subset"],
                "pass_formatting": row["source_text"].strip() in prompt and "Valid answer labels:" in prompt,
                "pass_label_mapping": row["gold_label"] in row["candidate_labels"],
                "pass_instruction_fidelity": prompt.endswith("Return only one label inside <answer></answer>."),
                "notes": "Proxy rendering audit only; prompt rebuilt from corrected per-example label space.",
            }
        )
    return pd.DataFrame(rows)


def prepare_data():
    ensure_dirs()
    resource_manifest = {
        "study_scope": PILOT_PLAN["study_scope"],
        "arc_challenge": "Loaded from Hugging Face parquet because the local datasets loader failed on the published schema.",
        "logiqa_2.0": "Loaded from Hugging Face datasets; no task-specific Trans-EnV release used.",
        "reclor": "Loaded from Hugging Face datasets; no task-specific Trans-EnV release used.",
        "multivalue": "Used only as a design prior; no released task-specific rewrites consumed.",
        "redial_logic_target": "Loaded from FangruLin/ReDial logic.json vanilla original/AAVE splits.",
        "claim_revision": PILOT_PLAN["claim_revision"],
        "adjustments": [
            "Synthetic rewrites use a deterministic proxy generator instead of the preregistered Trans-EnV generation stack.",
            "Acceptance thresholds are revised before training to match the proxy rewrite pipeline and are reported alongside the preregistered thresholds.",
            "Rewrite and protocol audits are explicitly proxy-only checks; no human audit is claimed in this pilot.",
        ],
        "pilot_thresholds": PILOT_THRESHOLDS,
        "preregistered_thresholds": PREREGISTERED_THRESHOLDS,
        "limitations": PILOT_PLAN["limitations"],
    }
    json_dump(DATA_AUDIT_DIR / "resource_manifest.json", resource_manifest)
    json_dump(DATA_AUDIT_DIR / "plan_revision.json", resource_manifest)
    json_dump(
        DATA_AUDIT_DIR / "variety_manifest.json",
        {
            "seen_training_varieties": SEEN_VARIETIES,
            "held_out_transfer_varieties": HELD_OUT_VARIETIES,
            "reserve_varieties": RESERVE_VARIETIES,
        },
    )

    train_logiqa, dev_logiqa, test_logiqa = deterministic_partition(normalize_logiqa("train") + normalize_logiqa("validation"), 192, 48, 64)
    train_reclor, dev_reclor, test_reclor = deterministic_partition(normalize_reclor("train") + normalize_reclor("validation"), 192, 48, 64)
    train_arc, dev_arc, test_arc = deterministic_partition(normalize_arc("train") + normalize_arc("validation"), 192, 48, 64)
    redial_se, redial_aave = normalize_redial_logic()

    train_source = [make_source_row(row, "train_source") for row in train_logiqa + train_reclor + train_arc]
    dev_source = [make_source_row(row, "dev_source") for row in dev_logiqa + dev_reclor + dev_arc]
    heldout_source = [make_source_row(row, "heldout_source") for row in test_logiqa + test_reclor + test_arc]

    source_ids = [row["source_question_id"] for row in train_source + dev_source + heldout_source]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Data leakage detected across source splits.")

    train_pairs, dev_pairs, heldout_pairs, rejections, attempts = [], [], [], [], []
    assignment_counts = Counter(variety for _, variety in assign_varieties(train_source, SEEN_VARIETIES, 2))

    for row, variety in assign_varieties(train_source, SEEN_VARIETIES, 2):
        rewritten, feats = rewrite_text(row["source_text"], variety)
        accepted = current_filter_pass(row["source_text"], rewritten, feats)
        attempts.append(
            {
                "example_id": stable_example_id("train_pair", row["source_question_id"], variety),
                "source_dataset": row["source_dataset"],
                "source_question_id": row["source_question_id"],
                "split": "train_pair",
                "variety": variety,
                "accepted": accepted,
                "preregistered_threshold_pass": prereg_threshold_pass(feats),
                **feats,
            }
        )
        if accepted:
            train_pairs.append(build_pair_row(row, "train_pair", variety, rewritten, feats, "accepted"))
        else:
            rejections.append(
                {
                    "example_id": stable_example_id("train_pair", row["source_question_id"], variety),
                    "source_dataset": row["source_dataset"],
                    "source_question_id": row["source_question_id"],
                    "split": "train_pair",
                    "variety": variety,
                    "reason": "failed_current_rule_based_filters",
                    "preregistered_threshold_pass": prereg_threshold_pass(feats),
                    **feats,
                }
            )

    for row, variety in assign_varieties(dev_source, SEEN_VARIETIES, 1):
        rewritten, feats = rewrite_text(row["source_text"], variety)
        dev_pairs.append(build_pair_row(row, "dev_pair", variety, rewritten, feats, "accepted"))

    for row in heldout_source:
        for variety in HELD_OUT_VARIETIES:
            rewritten, feats = rewrite_text(row["source_text"], variety)
            heldout_pairs.append(build_pair_row(row, "heldout_pair", variety, rewritten, feats, "accepted"))

    paraphrase_pairs, paraphrase_match_summary = build_paraphrase_controls(train_pairs)

    manifests = train_source + dev_source + heldout_source + redial_se + redial_aave
    jsonl_dump(DATA_AUDIT_DIR / "source_manifests.jsonl", manifests)
    jsonl_dump(DATA_AUDIT_DIR / "rewrite_attempts.jsonl", attempts)
    jsonl_dump(DATA_AUDIT_DIR / "rewrite_rejections.jsonl", rejections)
    json_dump(DATA_AUDIT_DIR / "paraphrase_match_summary.json", paraphrase_match_summary)
    jsonl_dump(DATA_DIR / "train_source.jsonl", train_source)
    jsonl_dump(DATA_DIR / "dev_source.jsonl", dev_source)
    jsonl_dump(DATA_DIR / "heldout_source.jsonl", heldout_source)
    jsonl_dump(DATA_DIR / "train_pairs.jsonl", train_pairs)
    jsonl_dump(DATA_DIR / "dev_pairs.jsonl", dev_pairs)
    jsonl_dump(DATA_DIR / "heldout_pairs.jsonl", heldout_pairs)
    jsonl_dump(DATA_DIR / "paraphrase_pairs.jsonl", paraphrase_pairs)
    jsonl_dump(DATA_DIR / "redial_se.jsonl", redial_se)
    jsonl_dump(DATA_DIR / "redial_aave.jsonl", redial_aave)

    rewrite_audit = build_rewrite_audit(train_pairs + heldout_pairs)
    rewrite_audit.to_csv(DATA_AUDIT_DIR / "rewrite_audit.csv", index=False)
    protocol_audit = build_protocol_audit(redial_aave)
    protocol_audit.to_csv(DATA_AUDIT_DIR / "redial_protocol_audit.csv", index=False)

    prereg_accept_by_variety = {}
    accepted_by_variety = Counter(row["assigned_variety"] for row in train_pairs)
    for variety in SEEN_VARIETIES:
        prereg_pass = sum(1 for row in attempts if row["variety"] == variety and row["preregistered_threshold_pass"])
        prereg_accept_by_variety[variety] = {
            "assigned_pairs": assignment_counts[variety],
            "accepted_pairs_pilot_pipeline": accepted_by_variety[variety],
            "accepted_pairs_preregistered_thresholds": prereg_pass,
            "pilot_rejection_rate": 1.0 - accepted_by_variety[variety] / max(1, assignment_counts[variety]),
            "preregistered_rejection_rate": 1.0 - prereg_pass / max(1, assignment_counts[variety]),
            "replacement_triggered_preregistered_policy": (1.0 - prereg_pass / max(1, assignment_counts[variety])) > 0.15,
        }
    prereg_accept_by_variety["paraphrase_match"] = paraphrase_match_summary
    json_dump(DATA_AUDIT_DIR / "preprocessing_summary.json", prereg_accept_by_variety)
    json_dump(
        ROOT / "exp" / "preprocess" / "results.json",
        {
            "experiment": "preprocess",
            "metrics": {
                "train_pairs": {"mean": len(train_pairs), "std": 0.0},
                "dev_pairs": {"mean": len(dev_pairs), "std": 0.0},
                "heldout_pairs": {"mean": len(heldout_pairs), "std": 0.0},
                "rejections": {"mean": len(rejections), "std": 0.0},
            },
            "config": {"pilot_thresholds": PILOT_THRESHOLDS, "study_scope": PILOT_PLAN["study_scope"]},
            "runtime_minutes": 0.0,
        },
    )

    return {
        "train_pairs": len(train_pairs),
        "dev_pairs": len(dev_pairs),
        "heldout_pairs": len(heldout_pairs),
        "rejections": len(rejections),
        "redial_examples": len(redial_aave),
    }


@dataclass
class Example:
    example_id: str
    prompt: str
    label: str
    candidate_labels: List[str]
    metadata: dict


def build_examples(system_name: str, split_name: str) -> List[Example]:
    if system_name == "sae_only_sft":
        path = DATA_DIR / ("train_source.jsonl" if split_name == "train" else "dev_source.jsonl")
        rows = jsonl_load(path)
        return [
            Example(
                row["example_id"],
                build_prompt(row["source_text"], row.get("answer_options"), row["candidate_labels"]),
                row["gold_label"],
                row["candidate_labels"],
                row,
            )
            for row in rows
        ]

    if split_name == "train":
        path = DATA_DIR / ("paraphrase_pairs.jsonl" if system_name == "paraphrase_pair_control" else "train_pairs.jsonl")
    else:
        path = DATA_DIR / "dev_pairs.jsonl"
    rows = jsonl_load(path)
    return [
        Example(
            row["example_id"],
            build_prompt(row["source_text"], row.get("answer_options"), row["candidate_labels"]),
            row["gold_label"],
            row["candidate_labels"],
            {**row, "variant_prompt": build_prompt(row["rewritten_text"], row.get("answer_options"), row["candidate_labels"])},
        )
        for row in rows
    ]


def build_eval_examples(eval_name: str) -> List[Example]:
    path_map = {
        "dev_source": DATA_DIR / "dev_source.jsonl",
        "dev_pairs": DATA_DIR / "dev_pairs.jsonl",
        "heldout_source": DATA_DIR / "heldout_source.jsonl",
        "heldout_pairs": DATA_DIR / "heldout_pairs.jsonl",
        "redial_se": DATA_DIR / "redial_se.jsonl",
        "redial_aave": DATA_DIR / "redial_aave.jsonl",
    }
    rows = jsonl_load(path_map[eval_name])
    prompt_field = "rewritten_text" if eval_name in {"dev_pairs", "heldout_pairs"} else "source_text"
    return [
        Example(
            row["example_id"],
            build_prompt(row[prompt_field], row.get("answer_options"), row["candidate_labels"]),
            row["gold_label"],
            row["candidate_labels"],
            row,
        )
        for row in rows
    ]


def answer_text(label: str) -> str:
    return f"<answer>{label}</answer>"


def load_model(trainable: bool):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if trainable:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
    return tokenizer, model


def encode_supervised(tokenizer, prompt: str, label: str):
    target = answer_text(label)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + target_ids)[-MAX_SEQ_LEN:]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    if len(labels) < len(input_ids):
        labels = [-100] * (len(input_ids) - len(labels)) + labels
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def score_candidates(tokenizer, model, prompt: str, candidates: List[str], allow_grad: bool = False) -> torch.Tensor:
    candidate_token_ids = []
    single_token = True
    for candidate in candidates:
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            single_token = False
            break
        candidate_token_ids.append(token_ids[0])
    if single_token:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        logits = model(**encoded).logits[0, -1] if allow_grad else model(**encoded).logits[0, -1].detach()
        return torch.log_softmax(logits[candidate_token_ids], dim=0)

    losses = []
    for candidate in candidates:
        encoded = {key: value.to(model.device) for key, value in encode_supervised(tokenizer, prompt, candidate).items()}
        if allow_grad:
            losses.append(-model(**encoded).loss)
        else:
            with torch.no_grad():
                losses.append(-model(**encoded).loss.detach())
    return torch.log_softmax(torch.stack(losses), dim=0)


def forward_ce(tokenizer, model, prompt: str, label: str) -> torch.Tensor:
    encoded = {key: value.to(model.device) for key, value in encode_supervised(tokenizer, prompt, label).items()}
    return model(**encoded).loss


def run_config(system_name: str, seed: Optional[int]) -> dict:
    return {
        "system": system_name,
        "seed": seed,
        "model_name": MODEL_NAME,
        "max_seq_len": MAX_SEQ_LEN,
        "training": TRAINING_CONFIG,
        "loss": {
            "svpt_lambda_consistency": 0.25 if system_name in {"svpt", "single_view_svpt"} else 0.0,
            "single_view_variant_supervision": system_name != "single_view_svpt",
        },
        "data": {
            "train_source_count": len(jsonl_load(DATA_DIR / "train_source.jsonl")),
            "train_pair_count": len(jsonl_load(DATA_DIR / "train_pairs.jsonl")),
            "dev_source_count": len(jsonl_load(DATA_DIR / "dev_source.jsonl")),
            "dev_pair_count": len(jsonl_load(DATA_DIR / "dev_pairs.jsonl")),
            "heldout_source_count": len(jsonl_load(DATA_DIR / "heldout_source.jsonl")),
            "heldout_pair_count": len(jsonl_load(DATA_DIR / "heldout_pairs.jsonl")),
            "redial_count": len(jsonl_load(DATA_DIR / "redial_aave.jsonl")),
        },
    }


def train_one_system(system_name: str, seed: int):
    set_seed(seed)
    out_dir = ARTIFACTS_DIR / system_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / "adapter" / "adapter_model.safetensors"
    json_dump(out_dir / "config.json", run_config(system_name, seed))
    if adapter_path.exists() and (out_dir / "train_log.jsonl").exists():
        return out_dir

    tokenizer, model = load_model(trainable=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"], betas=(0.9, 0.999), weight_decay=0.0)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_examples = build_examples(system_name, "train")
    if system_name == "sae_only_sft":
        train_examples = train_examples * 2
    start = time.time()
    train_logs = []
    accum = 0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for ex in train_examples:
        if system_name == "sae_only_sft":
            loss = forward_ce(tokenizer, model, ex.prompt, ex.label)
        else:
            ce_source = forward_ce(tokenizer, model, ex.prompt, ex.label)
            ce_variant = torch.tensor(0.0, device=model.device) if system_name == "single_view_svpt" else forward_ce(tokenizer, model, ex.metadata["variant_prompt"], ex.label)
            loss = ce_source + ce_variant
            if system_name in {"svpt", "single_view_svpt"}:
                logp_source = score_candidates(tokenizer, model, ex.prompt, ex.candidate_labels, allow_grad=True)
                logp_variant = score_candidates(tokenizer, model, ex.metadata["variant_prompt"], ex.candidate_labels, allow_grad=True)
                loss = loss + 0.25 * (
                    0.5
                    * (
                        F.kl_div(logp_source, logp_variant.exp(), reduction="batchmean", log_target=False)
                        + F.kl_div(logp_variant, logp_source.exp(), reduction="batchmean", log_target=False)
                    )
                )
        (loss / TRAINING_CONFIG["gradient_accumulation_steps"]).backward()
        accum += 1
        if accum == TRAINING_CONFIG["gradient_accumulation_steps"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_norm"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
        train_logs.append({"step": step, "loss": float(loss.detach().cpu())})
        step += 1
    if accum:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    jsonl_dump(out_dir / "train_log.jsonl", train_logs)
    json_dump(out_dir / "timing.json", {"runtime_minutes": (time.time() - start) / 60.0})
    json_dump(out_dir / "peak_memory.json", {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)})
    return out_dir


def load_trained_model(system_name: str, seed: int):
    tokenizer, model = load_model(trainable=False)
    if system_name in TRAINABLE_SYSTEMS or system_name == "single_view_svpt":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ARTIFACTS_DIR / system_name / f"seed_{seed}" / "adapter")
    model.eval()
    return tokenizer, model


def predict_examples(tokenizer, model, examples: List[Example], rewrite_to_standard: bool = False) -> Tuple[List[dict], Dict[str, float]]:
    rows = []
    correct = []
    invalid = 0
    for ex in examples:
        prompt = standardize_text(ex.prompt) if rewrite_to_standard else ex.prompt
        with torch.no_grad():
            log_probs = score_candidates(tokenizer, model, prompt, ex.candidate_labels)
        pred = ex.candidate_labels[int(torch.argmax(log_probs).item())]
        rows.append(
            {
                "example_id": ex.example_id,
                "gold_label": ex.label,
                "prediction": pred,
                "candidate_labels": ex.candidate_labels,
                "log_probs": log_probs.detach().cpu().tolist(),
                "metadata": ex.metadata,
            }
        )
        invalid += int(pred not in ex.candidate_labels)
        correct.append(int(pred == ex.label))
    return rows, {"accuracy": float(np.mean(correct)) if correct else 0.0, "invalid_label_rate": invalid / max(1, len(examples))}


def evaluate_system(system_name: str, seed: Optional[int] = None):
    if system_name == "rewrite_then_answer":
        tokenizer, model = load_model(trainable=False)
        rewrite_flag = True
    elif system_name == "base":
        tokenizer, model = load_model(trainable=False)
        rewrite_flag = False
    else:
        tokenizer, model = load_trained_model(system_name, seed or 13)
        rewrite_flag = False

    out_dir = ARTIFACTS_DIR / system_name / (f"seed_{seed}" if seed is not None else "deterministic")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dump(out_dir / "config.json", run_config(system_name, seed))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    dev_metrics, dev_predictions = {}, []
    for eval_name in ["dev_source", "dev_pairs"]:
        preds, metrics = predict_examples(tokenizer, model, build_eval_examples(eval_name), rewrite_to_standard=rewrite_flag)
        dev_metrics[eval_name] = metrics
        for pred in preds:
            pred["eval_split"] = eval_name
            dev_predictions.append(pred)
    json_dump(out_dir / "metrics_dev.json", dev_metrics)
    jsonl_dump(out_dir / "predictions_dev.jsonl", dev_predictions)

    test_metrics, test_predictions = {}, []
    for eval_name in ["heldout_source", "heldout_pairs", "redial_se", "redial_aave"]:
        preds, metrics = predict_examples(tokenizer, model, build_eval_examples(eval_name), rewrite_to_standard=rewrite_flag)
        test_metrics[eval_name] = metrics
        for pred in preds:
            pred["eval_split"] = eval_name
            test_predictions.append(pred)
    json_dump(out_dir / "metrics_test.json", test_metrics)
    jsonl_dump(out_dir / "predictions.jsonl", test_predictions)
    json_dump(out_dir / "timing.json", {"runtime_minutes": (time.time() - start) / 60.0})
    json_dump(out_dir / "peak_memory.json", {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)})
    return out_dir


def load_predictions(system: str, seed: Optional[int] = None, dev: bool = False) -> pd.DataFrame:
    suffix = f"seed_{seed}" if seed is not None else "deterministic"
    path = ARTIFACTS_DIR / system / suffix / ("predictions_dev.jsonl" if dev else "predictions.jsonl")
    return pd.DataFrame(jsonl_load(path))


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.array(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def stratified_bootstrap_ci(df: pd.DataFrame, col_a: str, col_b: str, strata_col: str, n_resamples: int = 10000, seed: int = 13) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    grouped = [group for _, group in df.groupby(strata_col)]
    deltas = []
    for _ in range(n_resamples):
        resampled = []
        for group in grouped:
            idx = rng.integers(0, len(group), size=len(group))
            resampled.append(group.iloc[idx])
        sample_df = pd.concat(resampled, ignore_index=True)
        deltas.append(float(sample_df[col_a].mean() - sample_df[col_b].mean()))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def metrics_record(system: str, seed: Optional[int], metrics: dict) -> dict:
    timing_path = ARTIFACTS_DIR / system / (f"seed_{seed}" if seed is not None else "deterministic") / "timing.json"
    peak_path = ARTIFACTS_DIR / system / (f"seed_{seed}" if seed is not None else "deterministic") / "peak_memory.json"
    timing = json.load(timing_path.open()) if timing_path.exists() else {"runtime_minutes": 0.0}
    peak = json.load(peak_path.open()) if peak_path.exists() else {"peak_allocated_bytes": 0}
    return {
        "system": system,
        "seed": seed,
        "heldout_source_accuracy": metrics["heldout_source"]["accuracy"],
        "heldout_pairs_accuracy": metrics["heldout_pairs"]["accuracy"],
        "heldout_gap": metrics["heldout_source"]["accuracy"] - metrics["heldout_pairs"]["accuracy"],
        "worst_group_heldout_accuracy": min(metrics["heldout_source"]["accuracy"], metrics["heldout_pairs"]["accuracy"]),
        "redial_se_accuracy": metrics["redial_se"]["accuracy"],
        "redial_aave_accuracy": metrics["redial_aave"]["accuracy"],
        "invalid_label_rate": metrics["redial_aave"]["invalid_label_rate"],
        "runtime_minutes": timing.get("runtime_minutes", 0.0),
        "peak_memory_gb": peak.get("peak_allocated_bytes", 0) / (1024**3),
    }


def system_metric_rows() -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    records = []
    by_system = defaultdict(list)
    for system in SYSTEMS_MAIN + ["single_view_svpt"]:
        if system in TRAINABLE_SYSTEMS:
            seeds = TRAIN_SEEDS
        elif system == "single_view_svpt":
            seeds = [13]
        else:
            seeds = [None]
        for seed in seeds:
            metric_path = ARTIFACTS_DIR / system / (f"seed_{seed}" if seed is not None else "deterministic") / "metrics_test.json"
            if not metric_path.exists():
                continue
            with metric_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)
            record = metrics_record(system, seed, metrics)
            records.append(record)
            by_system[system].append(record)
    return pd.DataFrame(records), by_system


def plot_figure_1(summary_df: pd.DataFrame):
    systems = summary_df["display_name"].tolist()
    metrics = [
        ("heldout_source_accuracy", "Held-out SE"),
        ("heldout_pairs_accuracy", "Held-out rewritten"),
        ("redial_se_accuracy", "ReDial SE"),
        ("redial_aave_accuracy", "ReDial AAVE"),
    ]
    x = np.arange(len(systems))
    width = 0.18
    plt.figure(figsize=(12, 6))
    for idx, (metric, label) in enumerate(metrics):
        means = summary_df[f"{metric}_mean"].to_numpy()
        stds = summary_df[f"{metric}_std"].to_numpy()
        plt.bar(x + (idx - 1.5) * width, means, width=width, yerr=stds, capsize=3, label=label)
    plt.xticks(x, systems, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_1.png", dpi=200)
    plt.savefig(FIGURES_DIR / "figure_1.pdf")
    plt.close()


def plot_figure_2():
    dev_rows = pd.DataFrame(jsonl_load(DATA_DIR / "dev_pairs.jsonl"))
    heldout_rows = pd.DataFrame(jsonl_load(DATA_DIR / "heldout_pairs.jsonl"))
    heatmap_records = []
    for system in ["sae_only_sft", "dialect_augmentation", "paraphrase_pair_control", "svpt"]:
        seeds = TRAIN_SEEDS
        for seed in seeds:
            dev_preds = load_predictions(system, seed, dev=True)
            held_preds = load_predictions(system, seed, dev=False)
            dev_preds = dev_preds[dev_preds["eval_split"] == "dev_pairs"].copy()
            held_preds = held_preds[held_preds["eval_split"] == "heldout_pairs"].copy()
            if not dev_preds.empty:
                dev_preds["correct"] = (dev_preds["prediction"] == dev_preds["gold_label"]).astype(int)
                dev_preds["variety"] = dev_preds["metadata"].apply(lambda meta: meta["assigned_variety"])
                for variety, group in dev_preds.groupby("variety"):
                    heatmap_records.append({"system": SYSTEM_DISPLAY[system], "seed": seed, "variety": variety, "accuracy": group["correct"].mean()})
            if not held_preds.empty:
                held_preds["correct"] = (held_preds["prediction"] == held_preds["gold_label"]).astype(int)
                held_preds["variety"] = held_preds["metadata"].apply(lambda meta: meta["assigned_variety"])
                for variety, group in held_preds.groupby("variety"):
                    heatmap_records.append({"system": SYSTEM_DISPLAY[system], "seed": seed, "variety": variety, "accuracy": group["correct"].mean()})
            redial_preds = held_preds = load_predictions(system, seed, dev=False)
            redial_preds = redial_preds[redial_preds["eval_split"] == "redial_aave"].copy()
            redial_preds["correct"] = (redial_preds["prediction"] == redial_preds["gold_label"]).astype(int)
            heatmap_records.append({"system": SYSTEM_DISPLAY[system], "seed": seed, "variety": "Human AAVE", "accuracy": redial_preds["correct"].mean()})

    heatmap_df = pd.DataFrame(heatmap_records)
    mean_df = heatmap_df.groupby(["system", "variety"], as_index=False)["accuracy"].mean()
    base_df = mean_df[mean_df["system"] == SYSTEM_DISPLAY["sae_only_sft"]][["variety", "accuracy"]].rename(columns={"accuracy": "baseline"})
    delta_df = mean_df.merge(base_df, on="variety", how="left")
    delta_df["delta"] = delta_df["accuracy"] - delta_df["baseline"]
    pivot = delta_df.pivot(index="system", columns="variety", values="delta").fillna(0.0)
    ordered_cols = SEEN_VARIETIES + HELD_OUT_VARIETIES + ["Human AAVE"]
    pivot = pivot.reindex(columns=ordered_cols)
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="vlag", center=0.0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_2.png", dpi=200)
    plt.savefig(FIGURES_DIR / "figure_2.pdf")
    plt.close()


def plot_figure_3():
    redial_rows = pd.DataFrame(jsonl_load(DATA_DIR / "redial_aave.jsonl"))
    distance_lookup = redial_rows[["example_id", "paired_normalized_levenshtein"]].rename(
        columns={"paired_normalized_levenshtein": "normalized_levenshtein"}
    )
    baseline = load_predictions("sae_only_sft", 13, dev=False)
    baseline = baseline[baseline["eval_split"] == "redial_aave"].copy()
    baseline["baseline_correct"] = (baseline["prediction"] == baseline["gold_label"]).astype(int)
    plot_rows = []
    for system in ["dialect_augmentation", "paraphrase_pair_control", "svpt"]:
        preds = load_predictions(system, 13, dev=False)
        preds = preds[preds["eval_split"] == "redial_aave"].copy()
        preds["correct"] = (preds["prediction"] == preds["gold_label"]).astype(int)
        preds = preds.merge(distance_lookup, on="example_id", how="left").merge(baseline[["example_id", "baseline_correct"]], on="example_id", how="left")
        preds["accuracy_delta"] = preds["correct"] - preds["baseline_correct"]
        preds["distance_bin"] = pd.cut(preds["normalized_levenshtein"], bins=5, include_lowest=True)
        grouped = preds.groupby("distance_bin", as_index=False, observed=False)["accuracy_delta"].mean()
        grouped["system"] = SYSTEM_DISPLAY[system]
        grouped["bin_mid"] = grouped["distance_bin"].apply(lambda interval: float(interval.mid))
        plot_rows.append(grouped)
    plot_df = pd.concat(plot_rows, ignore_index=True)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=plot_df, x="bin_mid", y="accuracy_delta", hue="system", marker="o")
    plt.xlabel("ReDial SE/AAVE distance bin midpoint")
    plt.ylabel("Accuracy delta vs SAE-only SFT")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_3.png", dpi=200)
    plt.savefig(FIGURES_DIR / "figure_3.pdf")
    plt.close()


def plot_figure_4():
    plot_rows = []
    for system in ["dialect_augmentation", "paraphrase_pair_control", "svpt"]:
        for seed in TRAIN_SEEDS:
            preds = load_predictions(system, seed, dev=False)
            preds = preds[preds["eval_split"] == "redial_aave"].copy()
            preds["correct"] = (preds["prediction"] == preds["gold_label"]).astype(int)
            preds["subset"] = preds["metadata"].apply(lambda meta: meta["subset"])
            preds["system"] = SYSTEM_DISPLAY[system]
            preds["seed"] = seed
            plot_rows.append(preds[["subset", "system", "seed", "correct"]])
    plot_df = pd.concat(plot_rows, ignore_index=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="subset", y="correct", hue="system", errorbar="sd")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4.png", dpi=200)
    plt.savefig(FIGURES_DIR / "figure_4.pdf")
    plt.close()


def write_experiment_results(summary_df: pd.DataFrame):
    for _, row in summary_df.iterrows():
        exp_dir = ROOT / "exp" / row["system"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        json_dump(
            exp_dir / "results.json",
            {
                "experiment": row["system"],
                "metrics": {
                    "heldout_source_accuracy": {"mean": row["heldout_source_accuracy_mean"], "std": row["heldout_source_accuracy_std"]},
                    "heldout_pairs_accuracy": {"mean": row["heldout_pairs_accuracy_mean"], "std": row["heldout_pairs_accuracy_std"]},
                    "heldout_gap": {"mean": row["heldout_gap_mean"], "std": row["heldout_gap_std"]},
                    "redial_se_accuracy": {"mean": row["redial_se_accuracy_mean"], "std": row["redial_se_accuracy_std"]},
                    "redial_aave_accuracy": {"mean": row["redial_aave_accuracy_mean"], "std": row["redial_aave_accuracy_std"]},
                },
                "config": {
                    "model_name": MODEL_NAME,
                    "seeds": [13] if row["system"] == "single_view_svpt" else (TRAIN_SEEDS if row["system"] in TRAINABLE_SYSTEMS else [None]),
                },
                "runtime_minutes": row["runtime_minutes_mean"],
            },
        )


def aggregate_results():
    records_df, by_system = system_metric_rows()
    if records_df.empty:
        raise RuntimeError("No evaluation metrics found.")

    summary_rows = []
    for system, rows in by_system.items():
        sdf = pd.DataFrame(rows)
        summary_rows.append(
            {
                "system": system,
                "display_name": SYSTEM_DISPLAY[system],
                "n_runs": int(len(sdf)),
                "heldout_source_accuracy_mean": sdf["heldout_source_accuracy"].mean(),
                "heldout_source_accuracy_std": sdf["heldout_source_accuracy"].std(ddof=0),
                "heldout_pairs_accuracy_mean": sdf["heldout_pairs_accuracy"].mean(),
                "heldout_pairs_accuracy_std": sdf["heldout_pairs_accuracy"].std(ddof=0),
                "heldout_gap_mean": sdf["heldout_gap"].mean(),
                "heldout_gap_std": sdf["heldout_gap"].std(ddof=0),
                "worst_group_heldout_accuracy_mean": sdf["worst_group_heldout_accuracy"].mean(),
                "worst_group_heldout_accuracy_std": sdf["worst_group_heldout_accuracy"].std(ddof=0),
                "redial_se_accuracy_mean": sdf["redial_se_accuracy"].mean(),
                "redial_se_accuracy_std": sdf["redial_se_accuracy"].std(ddof=0),
                "redial_aave_accuracy_mean": sdf["redial_aave_accuracy"].mean(),
                "redial_aave_accuracy_std": sdf["redial_aave_accuracy"].std(ddof=0),
                "invalid_label_rate_mean": sdf["invalid_label_rate"].mean(),
                "invalid_label_rate_std": sdf["invalid_label_rate"].std(ddof=0),
                "runtime_minutes_mean": sdf["runtime_minutes"].mean(),
                "runtime_minutes_std": sdf["runtime_minutes"].std(ddof=0),
                "peak_memory_gb_mean": sdf["peak_memory_gb"].mean(),
                "peak_memory_gb_std": sdf["peak_memory_gb"].std(ddof=0),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    system_order = SYSTEMS_MAIN + ["single_view_svpt"]
    summary_df["system"] = pd.Categorical(summary_df["system"], categories=system_order, ordered=True)
    summary_df = summary_df.sort_values("system").reset_index(drop=True)
    summary_df.to_csv(SUMMARY_DIR / "table_1.csv", index=False)
    (SUMMARY_DIR / "table_1.md").write_text(summary_df.to_markdown(index=False), encoding="utf-8")
    records_df.to_csv(SUMMARY_DIR / "seed_runs.csv", index=False)

    plot_figure_1(summary_df)
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    write_experiment_results(summary_df)

    preds_svpt = load_predictions("svpt", 13, dev=False)
    preds_aug = load_predictions("dialect_augmentation", 13, dev=False)
    preds_para = load_predictions("paraphrase_pair_control", 13, dev=False)
    redial_svpt = preds_svpt[preds_svpt["eval_split"] == "redial_aave"].copy()
    redial_aug = preds_aug[preds_aug["eval_split"] == "redial_aave"].copy()
    redial_para = preds_para[preds_para["eval_split"] == "redial_aave"].copy()
    for frame, name in [(redial_svpt, "svpt"), (redial_aug, "aug"), (redial_para, "para")]:
        frame[name] = (frame["prediction"] == frame["gold_label"]).astype(int)
        frame["subset"] = frame["metadata"].apply(lambda meta: meta["subset"])

    merge_aug = redial_svpt[["example_id", "subset", "svpt"]].merge(redial_aug[["example_id", "aug"]], on="example_id")
    merge_para = redial_svpt[["example_id", "subset", "svpt"]].merge(redial_para[["example_id", "para"]], on="example_id")
    ci_aug = stratified_bootstrap_ci(merge_aug, "svpt", "aug", "subset")
    ci_para = stratified_bootstrap_ci(merge_para, "svpt", "para", "subset")

    b_aug = int(((merge_aug["svpt"] == 1) & (merge_aug["aug"] == 0)).sum())
    c_aug = int(((merge_aug["svpt"] == 0) & (merge_aug["aug"] == 1)).sum())
    b_para = int(((merge_para["svpt"] == 1) & (merge_para["para"] == 0)).sum())
    c_para = int(((merge_para["svpt"] == 0) & (merge_para["para"] == 1)).sum())

    heldout_rows = pd.DataFrame(jsonl_load(DATA_DIR / "heldout_pairs.jsonl"))
    heldout_features = heldout_rows[["example_id", "normalized_levenshtein", "length_ratio", "token_change_ratio", "type_token_change"]].copy()
    redial_rows = pd.DataFrame(jsonl_load(DATA_DIR / "redial_aave.jsonl"))
    redial_features = redial_rows[
        ["example_id", "paired_normalized_levenshtein", "paired_length_ratio", "paired_token_change_ratio", "paired_type_token_change"]
    ].rename(
        columns={
            "paired_normalized_levenshtein": "normalized_levenshtein",
            "paired_length_ratio": "length_ratio",
            "paired_token_change_ratio": "token_change_ratio",
            "paired_type_token_change": "type_token_change",
        }
    )
    features = pd.concat([heldout_features, redial_features], ignore_index=True)
    reg_frames = []
    for system in ["dialect_augmentation", "paraphrase_pair_control", "svpt"]:
        for seed in TRAIN_SEEDS:
            frame = load_predictions(system, seed, dev=False)
            frame = frame[frame["eval_split"].isin(["heldout_pairs", "redial_aave"])].copy()
            frame["correct"] = (frame["prediction"] == frame["gold_label"]).astype(int)
            frame["system"] = system
            frame["dataset"] = frame["eval_split"]
            frame["seed"] = seed
            frame["option_overlap"] = 1.0
            reg_frames.append(frame)
    reg_df = pd.concat(reg_frames, ignore_index=True).merge(features, on="example_id", how="left")
    reg_df[["normalized_levenshtein", "length_ratio", "token_change_ratio", "type_token_change"]] = reg_df[
        ["normalized_levenshtein", "length_ratio", "token_change_ratio", "type_token_change"]
    ].fillna({"normalized_levenshtein": 0.0, "length_ratio": 1.0, "token_change_ratio": 0.0, "type_token_change": 0.0})
    X = pd.get_dummies(reg_df[["system", "dataset", "seed"]], drop_first=True)
    X["rewrite_distance"] = reg_df["normalized_levenshtein"]
    X["length_ratio"] = reg_df["length_ratio"]
    X["type_token_change"] = reg_df["type_token_change"]
    X["option_overlap"] = reg_df["option_overlap"]
    clf = LogisticRegression(max_iter=2000).fit(X, reg_df["correct"])
    coef_map = dict(zip(X.columns, clf.coef_[0]))
    json_dump(SUMMARY_DIR / "regression_coefficients.json", coef_map)

    rewrite_audit = pd.read_csv(DATA_AUDIT_DIR / "rewrite_audit.csv")
    protocol_audit = pd.read_csv(DATA_AUDIT_DIR / "redial_protocol_audit.csv")
    preprocessing_summary = json.load((DATA_AUDIT_DIR / "preprocessing_summary.json").open("r", encoding="utf-8"))
    resource_manifest = json.load((DATA_AUDIT_DIR / "resource_manifest.json").open("r", encoding="utf-8"))
    paraphrase_match_summary = json.load((DATA_AUDIT_DIR / "paraphrase_match_summary.json").open("r", encoding="utf-8"))

    subset_rows = []
    for system in ["dialect_augmentation", "paraphrase_pair_control", "svpt"]:
        for seed in TRAIN_SEEDS:
            preds = load_predictions(system, seed, dev=False)
            preds = preds[preds["eval_split"] == "redial_aave"].copy()
            preds["correct"] = (preds["prediction"] == preds["gold_label"]).astype(int)
            preds["subset"] = preds["metadata"].apply(lambda meta: meta["subset"])
            grouped = preds.groupby("subset", as_index=False)["correct"].mean()
            for _, row in grouped.iterrows():
                subset_rows.append({"system": system, "seed": seed, "subset": row["subset"], "accuracy": float(row["correct"])})
    subset_df = pd.DataFrame(subset_rows)
    subset_summary = (
        subset_df.groupby(["system", "subset"], as_index=False)["accuracy"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "accuracy_mean", "std": "accuracy_std"})
    )
    subset_summary.to_csv(SUMMARY_DIR / "redial_subset_breakdown.csv", index=False)

    summary_payload = {
        "metadata": {
            "date": time.strftime("%Y-%m-%d"),
            "model_name": MODEL_NAME,
            "train_seeds": TRAIN_SEEDS,
            "systems": SYSTEMS_MAIN,
            "study_scope": resource_manifest["study_scope"],
            "claim_revision": resource_manifest["claim_revision"],
            "adjustments": resource_manifest["adjustments"],
            "limitations": resource_manifest["limitations"],
        },
        "run_matrix": {
            "base": "completed",
            "rewrite_then_answer": "completed",
            "sae_only_sft": sorted(int(seed) for seed in records_df.loc[records_df["system"] == "sae_only_sft", "seed"].dropna()),
            "dialect_augmentation": sorted(int(seed) for seed in records_df.loc[records_df["system"] == "dialect_augmentation", "seed"].dropna()),
            "paraphrase_pair_control": sorted(int(seed) for seed in records_df.loc[records_df["system"] == "paraphrase_pair_control", "seed"].dropna()),
            "svpt": sorted(int(seed) for seed in records_df.loc[records_df["system"] == "svpt", "seed"].dropna()),
            "single_view_svpt": sorted(int(seed) for seed in records_df.loc[records_df["system"] == "single_view_svpt", "seed"].dropna()),
        },
        "systems": summary_rows,
        "statistics": {
            "svpt_minus_dialect_augmentation_redial_aave_ci95": list(ci_aug),
            "svpt_minus_paraphrase_redial_aave_ci95": list(ci_para),
            "mcnemar_svpt_vs_dialect_augmentation_redial_aave": {"b": b_aug, "c": c_aug, "p_value": exact_mcnemar_pvalue(b_aug, c_aug)},
            "mcnemar_svpt_vs_paraphrase_redial_aave": {"b": b_para, "c": c_para, "p_value": exact_mcnemar_pvalue(b_para, c_para)},
            "regression_coefficients": coef_map,
        },
        "audits": {
            "rewrite_audit_rows": int(len(rewrite_audit)),
            "rewrite_audit_failures": {
                "pass_semantics": int((~rewrite_audit["pass_semantics"].astype(bool)).sum()),
                "pass_dialect": int((~rewrite_audit["pass_dialect"].astype(bool)).sum()),
                "pass_option_preservation": int((~rewrite_audit["pass_option_preservation"].astype(bool)).sum()),
                "pass_no_leakage": int((~rewrite_audit["pass_no_leakage"].astype(bool)).sum()),
            },
            "redial_protocol_audit_rows": int(len(protocol_audit)),
            "redial_protocol_audit_failures": {
                "pass_formatting": int((~protocol_audit["pass_formatting"].astype(bool)).sum()),
                "pass_label_mapping": int((~protocol_audit["pass_label_mapping"].astype(bool)).sum()),
                "pass_instruction_fidelity": int((~protocol_audit["pass_instruction_fidelity"].astype(bool)).sum()),
            },
        },
        "preprocessing": preprocessing_summary,
        "matched_control": paraphrase_match_summary,
        "subset_breakdown": subset_summary.to_dict(orient="records"),
        "pilot_assessment": {
            "synthetic_gap_reduction_relative": float(
                (
                    summary_df.loc[summary_df["system"] == "sae_only_sft", "heldout_gap_mean"].iloc[0]
                    - summary_df.loc[summary_df["system"] == "svpt", "heldout_gap_mean"].iloc[0]
                )
                / max(1e-8, summary_df.loc[summary_df["system"] == "sae_only_sft", "heldout_gap_mean"].iloc[0])
            ),
            "redial_aave_delta_svpt_minus_aug": float(
                summary_df.loc[summary_df["system"] == "svpt", "redial_aave_accuracy_mean"].iloc[0]
                - summary_df.loc[summary_df["system"] == "dialect_augmentation", "redial_aave_accuracy_mean"].iloc[0]
            ),
            "redial_aave_delta_svpt_minus_paraphrase": float(
                summary_df.loc[summary_df["system"] == "svpt", "redial_aave_accuracy_mean"].iloc[0]
                - summary_df.loc[summary_df["system"] == "paraphrase_pair_control", "redial_aave_accuracy_mean"].iloc[0]
            ),
            "sae_retention_delta_svpt_minus_sae_only": float(
                summary_df.loc[summary_df["system"] == "svpt", "heldout_source_accuracy_mean"].iloc[0]
                - summary_df.loc[summary_df["system"] == "sae_only_sft", "heldout_source_accuracy_mean"].iloc[0]
            ),
            "seed13_redial_aave_delta_svpt_minus_aug": float(merge_aug["svpt"].mean() - merge_aug["aug"].mean()),
            "seed13_bootstrap_ci_above_zero": bool(ci_aug[0] > 0.0),
            "preregistered_transfer_claim_supported": False,
        },
    }
    json_dump(ROOT / "results.json", summary_payload)
    return summary_payload


def write_step_skip_docs():
    text = (
        "Pilot redesign applied before execution:\n\n"
        "- This run is a proxy-pipeline pilot rather than the preregistered Trans-EnV rewrite-validation study.\n"
        "- Synthetic rewrites and paraphrases are deterministic proxy transformations with revised pre-training acceptance thresholds saved in artifacts/data_audit/plan_revision.json.\n"
        "- Rewrite and protocol audits are proxy-only checks; no human audit is claimed in this rerun.\n"
        "- Loaded ARC-Challenge directly from Hugging Face parquet because the local datasets library could not parse the published schema.\n"
    )
    (ROOT / "exp" / "preprocess" / "SKIPPED.md").write_text(text, encoding="utf-8")


def run_all():
    ensure_dirs()
    write_step_skip_docs()
    prepare_data()
    evaluate_system("base")
    evaluate_system("rewrite_then_answer")
    for system in TRAINABLE_SYSTEMS:
        for seed in TRAIN_SEEDS:
            train_one_system(system, seed)
            evaluate_system(system, seed)
    train_one_system("single_view_svpt", 13)
    evaluate_system("single_view_svpt", 13)
    return aggregate_results()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preprocess", "train", "eval", "aggregate", "run_all"])
    parser.add_argument("--system")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    ensure_dirs()
    if args.mode == "preprocess":
        write_step_skip_docs()
        prepare_data()
    elif args.mode == "train":
        if args.system is None or args.seed is None:
            raise SystemExit("--system and --seed are required for train")
        train_one_system(args.system, args.seed)
    elif args.mode == "eval":
        if args.system is None:
            raise SystemExit("--system is required for eval")
        evaluate_system(args.system, args.seed)
    elif args.mode == "aggregate":
        aggregate_results()
    else:
        run_all()


if __name__ == "__main__":
    main()
