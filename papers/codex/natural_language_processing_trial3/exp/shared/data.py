from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from wilds import get_dataset

from .utils import ROOT, ensure_dir, read_json, set_seed, write_json


POS_ACTORS = [
    "Tom Hanks",
    "Meryl Streep",
    "Scarlett Johansson",
    "Leonardo DiCaprio",
    "Matt Damon",
]
NEG_ACTORS = [
    "Steven Seagal",
    "Jean-Claude Van Damme",
    "Sylvester Stallone",
    "Chuck Norris",
    "Dolph Lundgren",
]

STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "is", "it", "that", "this", "for",
    "on", "with", "as", "was", "but", "be", "at", "by", "i", "you", "he", "she",
    "they", "we", "my", "our", "your", "their", "me", "him", "her", "them", "or",
    "if", "from", "so", "not", "too", "very", "are", "am", "were", "been", "have",
    "has", "had", "do", "did", "does", "about", "there", "what", "when", "who",
    "which", "all", "just", "can", "could", "would", "should", "than", "then", "into",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
CIVIL_IDENTITY_COLUMNS = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    val_id_df: pd.DataFrame
    val_ood_df: pd.DataFrame
    test_dfs: dict[str, pd.DataFrame]
    lexicon: dict[str, Any]


def inject_actor(text: str, actor_tokens: list[str]) -> str:
    prefix = ". ".join(actor_tokens)
    return f"{prefix}. {text}"


def _choose_actor(label: int, correlated: bool, rng: np.random.Generator) -> tuple[str, str]:
    positive_assoc = (label == 1 and correlated) or (label == 0 and not correlated)
    if positive_assoc:
        return rng.choice(POS_ACTORS).item(), "positive"
    return rng.choice(NEG_ACTORS).item(), "negative"


def _build_actor_split(
    df: pd.DataFrame,
    correlation: float,
    split_name: str,
    rng: np.random.Generator,
    inject: bool = True,
) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        if inject:
            correlated = bool(rng.random() < correlation)
            actor, actor_group = _choose_actor(int(row.label), correlated, rng)
            actor_tokens = [actor]
            text = inject_actor(row.text, actor_tokens)
            actor_present = 1
        else:
            correlated = False
            actor = ""
            actor_group = "none"
            actor_tokens = []
            text = row.text
            actor_present = 0
        rows.append(
            {
                "source_id": int(row.source_id),
                "label": int(row.label),
                "text": text,
                "original_text": row.text,
                "split_name": split_name,
                "actor_token": actor,
                "actor_tokens": actor_tokens,
                "actor_group": actor_group,
                "actor_present": actor_present,
            }
        )
    return pd.DataFrame(rows)


def _tokenize_for_lexicon(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text)}


def mine_lexicon(
    df: pd.DataFrame,
    top_k: int,
    min_count: int,
    out_path: Path,
) -> dict[str, Any]:
    doc_counts = {0: Counter(), 1: Counter()}
    total_docs = {0: 0, 1: 0}
    global_vocab = Counter()
    for row in df.itertuples(index=False):
        toks = _tokenize_for_lexicon(row.text)
        label = int(row.label)
        total_docs[label] += 1
        for tok in toks:
            if tok in STOPWORDS:
                continue
            doc_counts[label][tok] += 1
            global_vocab[tok] += 1
    valid_tokens = {tok for tok, count in global_vocab.items() if count >= min_count and TOKEN_RE.fullmatch(tok)}
    alpha = 1.0
    vocab_size = len(valid_tokens)
    payload: dict[str, Any] = {"labels": {}}
    for label in (0, 1):
        other = 1 - label
        scored = []
        for tok in valid_tokens:
            c1 = doc_counts[label][tok]
            c0 = doc_counts[other][tok]
            num = (c1 + alpha) / (total_docs[label] + alpha * vocab_size)
            den = (c0 + alpha) / (total_docs[other] + alpha * vocab_size)
            score = math.log(num / den)
            if score == 0.0:
                continue
            if label == 0 and score >= 0.0:
                continue
            if label == 1 and score <= 0.0:
                continue
            scored.append((tok, score, c1, c0))
        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        payload["labels"][str(label)] = [
            {"token": tok, "score": score, "label_count": c1, "other_count": c0}
            for tok, score, c1, c0 in scored[:top_k]
        ]
    write_json(out_path, payload)
    return payload


def build_imdb_actor_dataset(cache_dir: Path, lexicon_path: Path) -> DatasetBundle:
    dataset_path = cache_dir / "imdb_actor_bundle.pkl"
    if dataset_path.exists() and lexicon_path.exists():
        bundle = pd.read_pickle(dataset_path)
        return DatasetBundle(
            train_df=bundle["train_df"],
            val_id_df=bundle["val_id_df"],
            val_ood_df=bundle["val_ood_df"],
            test_dfs=bundle["test_dfs"],
            lexicon=read_json(lexicon_path),
        )

    ensure_dir(cache_dir)
    rng = np.random.default_rng(20260324)
    ds = load_dataset("imdb")
    train_df = ds["train"].to_pandas().rename(columns={"text": "text", "label": "label"})
    test_df = ds["test"].to_pandas().rename(columns={"text": "text", "label": "label"})
    train_df["source_id"] = np.arange(len(train_df))
    test_df["source_id"] = np.arange(len(test_df))

    train_splits = []
    for label in (0, 1):
        label_df = train_df[train_df["label"] == label].sample(frac=1.0, random_state=20260324 + label)
        train_core = label_df.iloc[:10000]
        val_id = label_df.iloc[10000:11250]
        val_ood = label_df.iloc[11250:12500]
        train_splits.append((train_core, val_id, val_ood))
    train_core_df = pd.concat([x[0] for x in train_splits], ignore_index=True)
    val_id_src = pd.concat([x[1] for x in train_splits], ignore_index=True)
    val_ood_src = pd.concat([x[2] for x in train_splits], ignore_index=True)

    train_core_actor = _build_actor_split(train_core_df, 0.9, "train_core", rng, inject=True)
    val_id_actor = _build_actor_split(val_id_src, 0.9, "val_id", rng, inject=True)
    val_ood_actor = _build_actor_split(val_ood_src, 0.1, "val_ood", rng, inject=True)

    test_parts = {}
    unused_chunks = []
    for label in (0, 1):
        label_df = test_df[test_df["label"] == label].sample(frac=1.0, random_state=20261324 + label)
        test_id = label_df.iloc[:1250]
        test_ood = label_df.iloc[1250:2500]
        test_no = label_df.iloc[2500:3750]
        remaining = label_df.iloc[3750:]
        test_parts.setdefault("test_id_src", []).append(test_id)
        test_parts.setdefault("test_ood_src", []).append(test_ood)
        test_parts.setdefault("test_no_shortcut_src", []).append(test_no)
        unused_chunks.append(remaining)
    test_id_src = pd.concat(test_parts["test_id_src"], ignore_index=True)
    test_ood_src = pd.concat(test_parts["test_ood_src"], ignore_index=True)
    test_no_src = pd.concat(test_parts["test_no_shortcut_src"], ignore_index=True)
    unused_df = pd.concat(unused_chunks, ignore_index=True).sample(frac=1.0, random_state=20260399).reset_index(drop=True)
    conflict_src = unused_df.iloc[:2500].copy()
    scrub_src = unused_df.iloc[2500:5000].copy()

    test_id_df = _build_actor_split(test_id_src, 0.9, "test_id", rng, inject=True)
    test_ood_df = _build_actor_split(test_ood_src, 0.1, "test_ood", rng, inject=True)
    test_no_df = _build_actor_split(test_no_src, 0.0, "test_no_shortcut", rng, inject=False)

    conflict_rows = []
    for row in conflict_src.itertuples(index=False):
        pos_actor = rng.choice(POS_ACTORS).item()
        neg_actor = rng.choice(NEG_ACTORS).item()
        conflict_rows.append(
            {
                "source_id": int(row.source_id),
                "label": int(row.label),
                "text": inject_actor(row.text, [pos_actor, neg_actor]),
                "original_text": row.text,
                "split_name": "test_actor_conflict",
                "actor_token": f"{pos_actor}|{neg_actor}",
                "actor_tokens": [pos_actor, neg_actor],
                "actor_group": "conflict",
                "actor_present": 1,
            }
        )
    scrub_rows = []
    for row in scrub_src.itertuples(index=False):
        pos_actor = rng.choice(POS_ACTORS).item()
        neg_actor = rng.choice(NEG_ACTORS).item()
        _ = inject_actor(row.text, [pos_actor, neg_actor])
        scrub_rows.append(
            {
                "source_id": int(row.source_id),
                "label": int(row.label),
                "text": row.text,
                "original_text": row.text,
                "split_name": "test_actor_scrubbed",
                "actor_token": "",
                "actor_tokens": [],
                "actor_group": "scrubbed",
                "actor_present": 0,
            }
        )
    test_dfs = {
        "test_id": test_id_df,
        "test_ood": test_ood_df,
        "test_no_shortcut": test_no_df,
        "test_actor_conflict": pd.DataFrame(conflict_rows),
        "test_actor_scrubbed": pd.DataFrame(scrub_rows),
    }
    lexicon = mine_lexicon(train_core_actor, top_k=100, min_count=20, out_path=lexicon_path)
    pd.to_pickle(
        {
            "train_df": train_core_actor,
            "val_id_df": val_id_actor,
            "val_ood_df": val_ood_actor,
            "test_dfs": test_dfs,
        },
        dataset_path,
    )
    return DatasetBundle(
        train_df=train_core_actor,
        val_id_df=val_id_actor,
        val_ood_df=val_ood_actor,
        test_dfs=test_dfs,
        lexicon=lexicon,
    )


def build_civilcomments_dataset(cache_dir: Path, lexicon_path: Path) -> dict[str, pd.DataFrame]:
    dataset_path = cache_dir / "civilcomments_bundle.pkl"
    if dataset_path.exists() and lexicon_path.exists():
        payload = pd.read_pickle(dataset_path)
        if all("source_id" in payload[key].columns for key in ["train_df", "val_df", "test_df"]):
            if int(payload["train_df"]["identity_any"].sum()) > 0 and all(col in payload["train_df"].columns for col in CIVIL_IDENTITY_COLUMNS):
                return payload

    ensure_dir(cache_dir)
    data_dir = ROOT / "data" / "civilcomments_v1.0"
    csv_path = data_dir / "all_data_with_identities.csv"
    if not csv_path.exists():
        try:
            get_dataset(dataset="civilcomments", root_dir=str(ROOT / "data"), download=True)
        except Exception:
            pass
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected WILDS CivilComments file at {csv_path}")

    metadata = pd.read_csv(csv_path, index_col=0).reset_index().rename(columns={"index": "source_id"})
    metadata["label"] = (metadata["toxicity"] >= 0.5).astype(int)
    metadata["text"] = metadata["comment_text"].fillna("").astype(str)
    for col in CIVIL_IDENTITY_COLUMNS:
        metadata[col] = (metadata[col] >= 0.5).astype(int)
    metadata["identity_any"] = (metadata[CIVIL_IDENTITY_COLUMNS].sum(axis=1) > 0).astype(int)
    metadata["split_name"] = metadata["split"].astype(str)

    keep_cols = ["source_id", "split_name", "text", "label", "identity_any", *CIVIL_IDENTITY_COLUMNS]
    train_df = metadata[metadata["split_name"] == "train"][keep_cols].reset_index(drop=True)
    val_df = metadata[metadata["split_name"] == "val"][keep_cols].reset_index(drop=True)
    test_df = metadata[metadata["split_name"] == "test"][keep_cols].reset_index(drop=True)

    def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        strat = df["label"].astype(str) + "_" + df["identity_any"].astype(str)
        sampled = (
            df.assign(_strat=strat)
            .groupby("_strat", group_keys=False)
            .apply(lambda x: x.sample(max(1, int(round(n * len(x) / len(df)))), random_state=seed))
            .reset_index(drop=True)
            .head(n)
        )
        return sampled.drop(columns="_strat", errors="ignore")

    train_sub = stratified_subsample(train_df, 12000, seed=20260324)
    val_sub = stratified_subsample(val_df, 1500, seed=20260325)
    lexicon = mine_lexicon(train_sub, top_k=50, min_count=20, out_path=lexicon_path)
    payload = {
        "train_df": train_sub,
        "val_df": val_sub,
        "test_df": test_df,
        "lexicon": lexicon,
    }
    pd.to_pickle(payload, dataset_path)
    return payload


class TextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
        lexicon_tokens: set[str],
        top_tokens_map: dict[int, list[str]] | None = None,
        risk_mode: str = "hybrid",
        mask_mode: str = "full",
        use_actor_only_mask: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lexicon_tokens = lexicon_tokens
        self.top_tokens_map = top_tokens_map or {}
        self.risk_mode = risk_mode
        self.mask_mode = mask_mode
        self.use_actor_only_mask = use_actor_only_mask
        records = []
        texts = []
        masked_texts = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            risk, masked_text = self._risk_and_mask(row)
            texts.append(row["text"])
            masked_texts.append(masked_text)
            records.append(
                {
                    "label": int(row["label"]),
                    "risk": float(risk),
                    "source_id": int(row["source_id"]),
                    "actor_present": int(row.get("actor_present", 0)),
                }
            )
        self.records = records
        self.encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.masked_encoded = self.tokenizer(
            masked_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.df)

    def _risk_and_mask(self, row: pd.Series) -> tuple[float, str]:
        source_id = int(row["source_id"])
        raw_top_items = self.top_tokens_map.get(source_id, [])
        top_items: list[dict[str, float]] = []
        for item in raw_top_items:
            if isinstance(item, dict):
                token = str(item.get("token", "")).lower()
                score = float(item.get("score", 0.0))
            else:
                token = str(item).lower()
                score = 1.0
            if token:
                top_items.append({"token": token, "score": score})
        top_tokens = [item["token"] for item in top_items]
        top_scores = [item["score"] for item in top_items]
        text = row["text"]
        example_toks = sorted(_tokenize_for_lexicon(text))
        example_hits = [tok for tok in example_toks if tok in self.lexicon_tokens]
        lex_hits = [tok for tok in top_tokens if tok in self.lexicon_tokens]
        if self.risk_mode == "lexicon_only":
            risk = min(1.0, len(example_hits) / 5.0)
            selected = example_hits[:2]
        elif self.risk_mode == "attribution_only":
            total_score = float(sum(top_scores))
            focus_score = float(sum(top_scores[:2]))
            risk = (focus_score / total_score) if total_score > 0 else 0.0
            selected = top_tokens[:2]
        elif self.risk_mode == "random_token":
            rng = np.random.default_rng(source_id)
            if example_toks:
                sample_size = min(2, len(example_toks))
                selected = list(rng.choice(example_toks, size=sample_size, replace=False))
            else:
                selected = []
            risk = float(sum(tok in self.lexicon_tokens for tok in selected)) / max(1, len(selected))
        else:
            denom = max(1, min(5, len(top_tokens)))
            attr_risk = float(len(lex_hits)) / float(denom)
            lex_risk = min(1.0, len(example_hits) / 5.0)
            if top_tokens:
                risk = max(attr_risk, 0.5 * lex_risk)
            else:
                risk = lex_risk
            selected = lex_hits[:2] if lex_hits else example_hits[:2]
        if self.use_actor_only_mask and row.get("actor_token", ""):
            selected = [tok.lower() for tok in str(row["actor_token"]).split("|")[:1]]
        masked_text = text
        for tok in selected[:2]:
            masked_text = re.sub(rf"\b{re.escape(tok)}\b", self.tokenizer.mask_token, masked_text, flags=re.IGNORECASE)
        return float(np.clip(risk, 0.0, 1.0)), masked_text

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        item = {
            "input_ids": self.encoded["input_ids"][idx],
            "attention_mask": self.encoded["attention_mask"][idx],
            "masked_input_ids": self.masked_encoded["input_ids"][idx],
            "masked_attention_mask": self.masked_encoded["attention_mask"][idx],
            "labels": torch.tensor(row["label"], dtype=torch.long),
            "risk": torch.tensor(row["risk"], dtype=torch.float32),
            "source_id": torch.tensor(row["source_id"], dtype=torch.long),
            "actor_present": torch.tensor(row["actor_present"], dtype=torch.long),
        }
        return item
