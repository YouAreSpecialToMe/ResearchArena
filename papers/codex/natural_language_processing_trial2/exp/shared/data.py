from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import DATA_DIR, ROOT, exp_log_path, jsonl_write, append_log


PASSAGE_RE = re.compile(r"(passage\s+\d+\s*:)", re.IGNORECASE)
GENERATOR_PATTERNS = [
    ("gpt-4", re.compile(r"gpt-4", re.IGNORECASE)),
    ("gpt-3.5", re.compile(r"gpt-3\.5|gpt-35", re.IGNORECASE)),
    ("mistral", re.compile(r"mistral", re.IGNORECASE)),
    ("llama", re.compile(r"llama", re.IGNORECASE)),
    ("mixtral", re.compile(r"mixtral", re.IGNORECASE)),
    ("gemma", re.compile(r"gemma", re.IGNORECASE)),
    ("claude", re.compile(r"claude", re.IGNORECASE)),
]


@dataclass
class PreparedData:
    claims: pd.DataFrame
    evidence: pd.DataFrame
    stats: pd.DataFrame


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def canonicalize_generator_family(model_name: str) -> str:
    model_name = model_name or "unknown"
    for family, pattern in GENERATOR_PATTERNS:
        if pattern.search(model_name):
            return family
    cleaned = re.split(r"[-_/]", model_name)[0].strip().lower()
    return cleaned or "unknown"


def load_sentencizer():
    import spacy

    return spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer", "attribute_ruler"])


def _token_count(nlp, text: str) -> int:
    return sum(1 for token in nlp.make_doc(text) if not token.is_space)


def _split_long_sentence(nlp, text: str, base_start: int) -> list[dict[str, Any]]:
    if _token_count(nlp, text) <= 40 or not any(mark in text for mark in [";", ":"]):
        clean = text.strip()
        if not clean:
            return []
        offset = text.find(clean)
        return [
            {
                "text": clean,
                "start": base_start + max(offset, 0),
                "end": base_start + max(offset, 0) + len(clean),
                "token_count": _token_count(nlp, clean),
            }
        ]

    pieces: list[dict[str, Any]] = []
    start = 0
    for match in re.finditer(r"[;:]", text):
        end = match.end()
        candidate = text[start:end].strip()
        if candidate:
            offset = text.find(candidate, start)
            pieces.append(
                {
                    "text": candidate,
                    "start": base_start + offset,
                    "end": base_start + offset + len(candidate),
                    "token_count": _token_count(nlp, candidate),
                }
            )
        start = end
    tail = text[start:].strip()
    if tail:
        offset = text.find(tail, start)
        pieces.append(
            {
                "text": tail,
                "start": base_start + offset,
                "end": base_start + offset + len(tail),
                "token_count": _token_count(nlp, tail),
            }
        )
    return [piece for piece in pieces if piece["text"]]


def sentencize(nlp, text: str) -> list[dict[str, Any]]:
    doc = nlp(text or "")
    rows = []
    sent_id = 0
    for sent in doc.sents:
        sent_text = sent.text
        for piece in _split_long_sentence(nlp, sent_text, int(sent.start_char)):
            rows.append(
                {
                    "sentence_id": sent_id,
                    "text": piece["text"],
                    "start": piece["start"],
                    "end": piece["end"],
                    "token_count": piece["token_count"],
                }
            )
            sent_id += 1
    return rows


def batch_sentencize(nlp, texts: list[str], batch_size: int = 128) -> list[list[dict[str, Any]]]:
    rows = []
    for doc in nlp.pipe((text or "" for text in texts), batch_size=batch_size):
        doc_rows = []
        sent_id = 0
        for sent in doc.sents:
            for piece in _split_long_sentence(nlp, sent.text, int(sent.start_char)):
                doc_rows.append(
                    {
                        "sentence_id": sent_id,
                        "text": piece["text"],
                        "start": piece["start"],
                        "end": piece["end"],
                        "token_count": piece["token_count"],
                    }
                )
                sent_id += 1
        rows.append(doc_rows)
    return rows


def flatten_data2txt(obj: Any, prefix: str = "") -> list[str]:
    rows: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}{key}"
            rows.extend(flatten_data2txt(value, new_prefix + ": "))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            rows.extend(flatten_data2txt(value, prefix + f"{i}. "))
    else:
        rows.append(f"{prefix}{obj}")
    return rows


def source_to_evidence_sentences(nlp, source_row: dict[str, Any]) -> list[dict[str, Any]]:
    task = source_row["task_type"]
    source_info = source_row["source_info"]
    rows: list[dict[str, Any]] = []

    if task == "QA":
        question = source_info.get("question", "")
        passages = source_info.get("passages", "")
        pieces = PASSAGE_RE.split(passages)
        cur_doc = "passage_0"
        payloads: list[tuple[str, str]] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if piece.lower().startswith("passage"):
                cur_doc = piece.replace(":", "").strip().lower().replace(" ", "_")
            else:
                payloads.append((cur_doc, piece))
        for rank, (doc_id, payload) in enumerate(payloads):
            for sent in sentencize(nlp, payload):
                rows.append(
                    {
                        "source_id": source_row["source_id"],
                        "doc_id": doc_id,
                        "passage_id": doc_id,
                        "original_retrieval_rank": rank,
                        "sentence_position": sent["sentence_id"],
                        "evidence_text": sent["text"],
                        "question_text": question,
                        "task_type": task,
                    }
                )
    elif task == "Summary":
        for sent in sentencize(nlp, source_info):
            rows.append(
                {
                    "source_id": source_row["source_id"],
                    "doc_id": "summary_source",
                    "passage_id": "summary_source",
                    "original_retrieval_rank": 0,
                    "sentence_position": sent["sentence_id"],
                    "evidence_text": sent["text"],
                    "question_text": "",
                    "task_type": task,
                }
            )
    else:
        flattened = flatten_data2txt(source_info)
        for i, item in enumerate(flattened):
            rows.append(
                {
                    "source_id": source_row["source_id"],
                    "doc_id": f"table_field_{i}",
                    "passage_id": f"table_field_{i}",
                    "original_retrieval_rank": i,
                    "sentence_position": 0,
                    "evidence_text": normalize_ws(item),
                    "question_text": "",
                    "task_type": task,
                }
            )

    deduped = []
    seen = set()
    for row in rows:
        key = normalize_ws(row["evidence_text"]).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def overlap_ratio(span_start: int, span_end: int, sent_start: int, sent_end: int) -> float:
    overlap = max(0, min(span_end, sent_end) - max(span_start, sent_start))
    denom = max(1, sent_end - sent_start)
    return overlap / denom


def project_labels(response_row: dict[str, Any], sentence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = response_row.get("labels", [])
    projected = []
    for sent in sentence_rows:
        ratios = []
        adaptive_overlap = 0.0
        mixed_overlap = 0.0
        for label in labels:
            ratio = overlap_ratio(label["start"], label["end"], sent["start"], sent["end"])
            if ratio <= 0:
                continue
            ratios.append(ratio)
            if label.get("implicit_true"):
                adaptive_overlap = max(adaptive_overlap, ratio)
            if label.get("due_to_null"):
                mixed_overlap = max(mixed_overlap, ratio)
        max_ratio = max(ratios) if ratios else 0.0
        projected_all = int(max_ratio >= 0.20)
        strict_label = None
        ambiguity = False
        if max_ratio >= 0.50 and adaptive_overlap == 0 and mixed_overlap == 0:
            strict_label = 1
        elif max_ratio == 0.0 and adaptive_overlap == 0 and mixed_overlap == 0:
            strict_label = 0
        else:
            ambiguity = True
        projected.append(
            {
                "projected_all_label": projected_all,
                "strict_label": strict_label,
                "overlap_ratio": round(max_ratio, 6),
                "ambiguity_flag": ambiguity,
            }
        )
    return projected


def _safe_stratify_key(df: pd.DataFrame) -> pd.Series:
    base = (
        df["task_type"].astype(str)
        + "|"
        + df["generator_family"].astype(str)
        + "|"
        + df["response_level_hallucination"].astype(str)
    )
    counts = base.value_counts()
    if counts.min() >= 2:
        return base
    fallback = df["task_type"].astype(str) + "|" + df["response_level_hallucination"].astype(str)
    fallback_counts = fallback.value_counts()
    if fallback_counts.min() >= 2:
        return fallback
    return df["response_level_hallucination"].astype(str)


def _write_processed_files(claims_df: pd.DataFrame, evidence_df: pd.DataFrame) -> None:
    processed_dir = ROOT / "data" / "processed"
    for split in ["train", "val", "test"]:
        split_claims = claims_df[claims_df["split"] == split].to_dict(orient="records")
        jsonl_write(split_claims, processed_dir / f"claims_{split}.jsonl")
        split_source_ids = claims_df.loc[claims_df["split"] == split, "example_id"].unique().tolist()
        split_evidence = evidence_df[evidence_df["source_id"].isin(split_source_ids)].copy()
        split_evidence["split"] = split
        jsonl_write(split_evidence.to_dict(orient="records"), processed_dir / f"evidence_{split}.jsonl")


def _save_stats(claims_df: pd.DataFrame, evidence_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evidence_counts = evidence_df.groupby("source_id").size().rename("evidence_per_example")
    for split in ["train", "val", "test"]:
        split_claims = claims_df[claims_df["split"] == split]
        strict = split_claims[split_claims["strict_label"].notna()]
        rows.append(
            {
                "split": split,
                "responses": int(split_claims["response_id"].nunique()),
                "claims": int(len(split_claims)),
                "strict_claims": int(len(strict)),
                "strict_positive_rate": float(strict["strict_label"].mean()) if len(strict) else math.nan,
                "projected_positive_rate": float(split_claims["projected_all_label"].mean()) if len(split_claims) else math.nan,
                "ambiguity_drop_rate": float(split_claims["ambiguity_flag"].mean()) if len(split_claims) else math.nan,
                "avg_claim_length": float(split_claims["sentence_length"].mean()) if len(split_claims) else math.nan,
                "avg_evidence_sentences_per_example": float(
                    evidence_counts.loc[
                        evidence_counts.index.intersection(split_claims["example_id"].unique())
                    ].mean()
                )
                if len(split_claims)
                else math.nan,
            }
        )

    task_counts = (
        claims_df.groupby(["split", "task_type"]).size().rename("claim_count").reset_index()
    )
    gen_counts = (
        claims_df.groupby(["split", "generator_family"]).size().rename("claim_count").reset_index()
    )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(ROOT / "artifacts" / "tables" / "data_stats.csv", index=False)
    task_counts.to_csv(ROOT / "artifacts" / "tables" / "data_stats_by_task.csv", index=False)
    gen_counts.to_csv(ROOT / "artifacts" / "tables" / "data_stats_by_generator.csv", index=False)
    return stats_df


def build_datasets() -> PreparedData:
    log_path = exp_log_path("data_prep")
    append_log(log_path, "Loading en_core_web_sm and raw RAGTruth files.")
    nlp = load_sentencizer()

    with (DATA_DIR / "source_info.jsonl").open() as f:
        sources = [json.loads(line) for line in f]
    source_map = {row["source_id"]: row for row in sources}

    source_evidence_rows: list[dict[str, Any]] = []
    for source in sources:
        source_evidence_rows.extend(source_to_evidence_sentences(nlp, source))
    evidence_df = pd.DataFrame(source_evidence_rows)

    with (DATA_DIR / "response.jsonl").open() as f:
        responses = [json.loads(line) for line in f]
    good_responses = [row for row in responses if row["quality"] == "good"]
    append_log(log_path, f"Loaded {len(good_responses)} good responses from raw data.")
    response_sentences = batch_sentencize(nlp, [row["response"] for row in good_responses])

    response_rows = []
    claim_rows: list[dict[str, Any]] = []
    for row, sentence_rows in zip(good_responses, response_sentences):
        source = source_map[row["source_id"]]
        labels = project_labels(row, sentence_rows)
        response_has_hall = int(len(row["labels"]) > 0)
        generator_family = canonicalize_generator_family(row["model"])
        response_rows.append(
            {
                "response_id": row["id"],
                "example_id": row["source_id"],
                "task_type": source["task_type"],
                "generator_family": generator_family,
                "response_level_hallucination": response_has_hall,
                "official_split": row["split"],
            }
        )
        for sent, proj in zip(sentence_rows, labels):
            claim_rows.append(
                {
                    "example_id": row["source_id"],
                    "response_id": row["id"],
                    "task_type": source["task_type"],
                    "generator_family": generator_family,
                    "question_text": source["source_info"].get("question", "") if isinstance(source["source_info"], dict) else "",
                    "answer_sentence": sent["text"],
                    "sentence_index": sent["sentence_id"],
                    "projected_all_label": proj["projected_all_label"],
                    "strict_label": proj["strict_label"],
                    "overlap_ratio": proj["overlap_ratio"],
                    "ambiguity_flag": proj["ambiguity_flag"],
                    "sentence_length": sent["token_count"],
                    "official_split": row["split"],
                    "response_level_hallucination": response_has_hall,
                }
            )

    claims_df = pd.DataFrame(claim_rows)
    response_df = pd.DataFrame(response_rows).drop_duplicates("response_id")
    held_out_test_ids = set(response_df.loc[response_df["official_split"] == "test", "response_id"])
    train_pool = response_df[response_df["official_split"] != "test"].copy()
    stratify = _safe_stratify_key(train_pool)
    train_ids, val_ids = train_test_split(
        train_pool["response_id"],
        test_size=0.10,
        random_state=13,
        stratify=stratify,
    )
    split_map = {rid: "train" for rid in train_ids}
    split_map.update({rid: "val" for rid in val_ids})
    split_map.update({rid: "test" for rid in held_out_test_ids})
    claims_df["split"] = claims_df["response_id"].map(split_map)

    _write_processed_files(claims_df, evidence_df)
    stats_df = _save_stats(claims_df, evidence_df)
    append_log(
        log_path,
        "Prepared splits with counts: "
        + ", ".join(
            f"{split}={int((claims_df['split'] == split).sum())}"
            for split in ["train", "val", "test"]
        ),
    )
    return PreparedData(claims=claims_df, evidence=evidence_df, stats=stats_df)
