import csv
import gzip
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import string
import subprocess
import time
import unicodedata
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Levenshtein import ratio as levenshtein_ratio
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CACHE = ROOT / "data_cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = [13, 29, 47]
MODEL_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}
EVAL_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}

T2D_ZIP = DATA_RAW / "t2d_sm_wh.zip"
WDC_TRAIN_ZIP = DATA_RAW / "wdc" / "all_train.zip"
WDC_VALID_ZIP = DATA_RAW / "wdc" / "all_valid.zip"
WDC_GS_GZ = DATA_RAW / "wdc" / "all_gs.json.gz"
ALLOCATED_CPU_CORES = 2
ALLOCATED_RAM_GB = 128
PACKAGE_VERSION_NAMES = [
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "python-Levenshtein",
    "rapidfuzz",
    "scikit-learn",
    "scipy",
    "seaborn",
    "statsmodels",
    "tqdm",
]


def ensure_dirs() -> None:
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, "1")
    for path in [DATA_PROCESSED, DATA_CACHE, RESULTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9\.\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_header(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def canonicalize_numeric_token(text: str) -> Optional[str]:
    candidate = (text or "").replace(",", "").replace(" ", "")
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", candidate):
        return None
    try:
        return f"{float(candidate):.6f}"
    except ValueError:
        return None


def canonicalize_date_token(text: str) -> Optional[str]:
    text = normalize_text(text)
    if not text:
        return None
    if not re.search(r"\d", text):
        return None
    if not (
        re.fullmatch(r"\d{4}([/-]\d{1,2}){0,2}", text)
        or re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
        or re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", text)
    ):
        return None
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y"]:
        try:
            dt = pd.to_datetime(text, format=fmt, errors="raise")
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", text):
        parts = text.split(".")
        year = parts[2]
        if len(year) == 2:
            year = f"20{year}" if int(year) < 50 else f"19{year}"
        return f"{year}-{int(parts[1]):02d}-{int(parts[0]):02d}"
    return None


def canonicalize_value_token(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    numeric = canonicalize_numeric_token(text)
    if numeric is not None:
        return numeric
    date = canonicalize_date_token(text)
    if date is not None:
        return date
    return text


def normalized_value_counter(values: Sequence[Any]) -> Counter:
    counter = Counter()
    for value in values:
        token = canonicalize_value_token(value)
        if token:
            counter[token] += 1
    return counter


def tokenize(text: Any) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", normalize_text(text)) if t]


def jaccard(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def counter_cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(a[k] * b[k] for k in keys)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def infer_primitive_type(values: Sequence[Any]) -> str:
    seen = [normalize_text(v) for v in values if normalize_text(v)]
    if not seen:
        return "empty"
    numeric = 0
    dates = 0
    for value in seen:
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", value.replace(",", "")):
            numeric += 1
        elif re.fullmatch(r"\d{4}([/-]\d{1,2}){0,2}", value) or re.fullmatch(
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", value
        ):
            dates += 1
    if numeric / len(seen) >= 0.8:
        return "numeric"
    if dates / len(seen) >= 0.5:
        return "date"
    return "text"


def maybe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def load_t2d_split(split: str) -> Dict[str, Any]:
    extracted = DATA_PROCESSED / "t2d_sm_wh"
    maybe_extract_zip(T2D_ZIP, extracted)
    split_dir = extracted / split
    corr = pd.read_csv(split_dir / f"{split}_correspondences.csv")
    tables = {}
    for side in ["webtables", "dbpedia_tables"]:
        side_dir = split_dir / side
        if side_dir.exists():
            for csv_path in side_dir.glob("*.csv"):
                tables[(side, csv_path.name)] = pd.read_csv(csv_path)
    return {"correspondences": corr, "tables": tables}


def t2d_profile_column(series: pd.Series, header: str) -> Dict[str, Any]:
    values = series.tolist()
    norm_values = [normalize_text(v) for v in values if normalize_text(v)]
    token_counter = Counter()
    for value in norm_values[:200]:
        token_counter.update(tokenize(value))
    canonical_counter = normalized_value_counter(values)
    non_null = int(series.notna().sum())
    return {
        "raw_header": header,
        "normalized_header": normalize_header(header),
        "primitive_type": infer_primitive_type(values),
        "non_null_count": non_null,
        "missingness_rate": 1.0 - (non_null / max(len(series), 1)),
        "sample_values": norm_values[:10],
        "value_token_counter": token_counter,
        "normalized_value_counter": canonical_counter,
        "numeric_ratio": float(
            np.mean(
                [
                    bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", v.replace(",", "")))
                    for v in norm_values
                ]
            )
            if norm_values
            else 0.0
        ),
        "date_ratio": float(
            np.mean(
                [
                    bool(
                        re.fullmatch(r"\d{4}([/-]\d{1,2}){0,2}", v)
                        or re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", v)
                    )
                    for v in norm_values
                ]
            )
            if norm_values
            else 0.0
        ),
    }


def build_t2d_profiles() -> pd.DataFrame:
    cache_path = DATA_CACHE / "t2d_sm_wh_profiles.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if "normalized_value_counter_json" in cached.columns:
            return cached
        cache_path.unlink()
    rows: List[Dict[str, Any]] = []
    for split in ["train", "valid", "test"]:
        bundle = load_t2d_split(split)
        for (side, table_name), df in bundle["tables"].items():
            for idx, column in enumerate(df.columns):
                profile = t2d_profile_column(df[column], column)
                rows.append(
                    {
                        "split": split,
                        "side": side,
                        "table_name": table_name,
                        "column_index": idx,
                        **profile,
                    }
                )
    out = pd.DataFrame(rows)
    out["value_token_counter_json"] = out["value_token_counter"].apply(lambda c: json.dumps(c))
    out["normalized_value_counter_json"] = out["normalized_value_counter"].apply(lambda c: json.dumps(c))
    out = out.drop(columns=["value_token_counter", "normalized_value_counter"])
    out.to_parquet(cache_path, index=False)
    return out


def profile_lookup(profiles: pd.DataFrame) -> Dict[Tuple[str, str, str, int], Dict[str, Any]]:
    lookup = {}
    for row in profiles.to_dict("records"):
        counter = Counter(json.loads(row["value_token_counter_json"]))
        normalized_counter = Counter(json.loads(row["normalized_value_counter_json"]))
        row = dict(row)
        row["value_token_counter"] = counter
        row["normalized_value_counter"] = normalized_counter
        lookup[(row["split"], row["side"], row["table_name"], int(row["column_index"]))] = row
    return lookup


def load_wdc_medium() -> Dict[str, pd.DataFrame]:
    train_rows = []
    with zipfile.ZipFile(WDC_TRAIN_ZIP) as zf:
        with zf.open("all_train_medium.json.gz") as fh:
            with gzip.open(fh, "rt", encoding="utf-8") as gz:
                for line in gz:
                    train_rows.append(json.loads(line))
    train_df = pd.DataFrame(train_rows)
    with zipfile.ZipFile(WDC_VALID_ZIP) as zf:
        with zf.open("all_valid_medium.csv") as fh:
            valid_ids = pd.read_csv(fh)["pair_id"].astype(str).tolist()
    test_rows = []
    with gzip.open(WDC_GS_GZ, "rt", encoding="utf-8") as gz:
        for line in gz:
            test_rows.append(json.loads(line))
    gs_df = pd.DataFrame(test_rows)
    gs_df["pair_id"] = gs_df["pair_id"].astype(str)
    valid_df = gs_df[gs_df["pair_id"].isin(set(valid_ids))].copy()
    test_df = gs_df[~gs_df["pair_id"].isin(set(valid_ids))].copy()
    if valid_df.empty:
        # Conservative fallback if official validation ids do not overlap the public gs.
        valid_df = train_df.sample(frac=0.15, random_state=13).copy()
    return {"train": train_df, "valid": valid_df, "test": test_df}


def build_wdc_normalized_view() -> pd.DataFrame:
    cache_path = DATA_CACHE / "wdc_products_medium_normalized.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    bundle = load_wdc_medium()
    rows = []
    for split, df in bundle.items():
        for row in df.to_dict("records"):
            out = {"split": split, "pair_id": str(row["pair_id"]), "label": int(row["label"])}
            for side in ["left", "right"]:
                for field in [
                    "title",
                    "description",
                    "brand",
                    "price",
                    "specTableContent",
                    "category",
                ]:
                    raw = row.get(f"{field}_{side}")
                    out[f"{field}_{side}"] = raw
                    out[f"{field}_{side}_norm"] = normalize_text(raw)
            out["combined_left"] = " ".join(
                out[f"{f}_left_norm"] for f in ["title", "brand", "description", "price", "specTableContent"]
            ).strip()
            out["combined_right"] = " ".join(
                out[f"{f}_right_norm"] for f in ["title", "brand", "description", "price", "specTableContent"]
            ).strip()
            rows.append(out)
    result = pd.DataFrame(rows)
    result.to_parquet(cache_path, index=False)
    return result


def schema_feature_row(pair_row: pd.Series, lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]], split: str) -> Dict[str, float]:
    left = lookup[(split, "webtables", pair_row["table_name"] + ".csv", int(pair_row["column_index_left"]))]
    right = lookup[(split, "dbpedia_tables", pair_row["table_name"] + ".csv", int(pair_row["column_index_right"]))]
    left_tokens = tokenize(left["normalized_header"])
    right_tokens = tokenize(right["normalized_header"])
    features = {
        "header_jaccard": jaccard(left_tokens, right_tokens),
        "header_lev": levenshtein_ratio(left["normalized_header"], right["normalized_header"]),
        "token_set_ratio": fuzz.token_set_ratio(left["normalized_header"], right["normalized_header"]) / 100.0,
        "value_cosine": counter_cosine(left["value_token_counter"], right["value_token_counter"]),
        "type_agree": float(left["primitive_type"] == right["primitive_type"]),
        "numeric_ratio_diff": abs(left["numeric_ratio"] - right["numeric_ratio"]),
        "date_ratio_diff": abs(left["date_ratio"] - right["date_ratio"]),
    }
    features["numeric_sim"] = 1.0 - min(features["numeric_ratio_diff"], 1.0)
    features["date_sim"] = 1.0 - min(features["date_ratio_diff"], 1.0)
    return features


def apply_assignment(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pred_rows = []
    for table_name, group in scores.groupby("table_name"):
        lefts = sorted(group["column_index_left"].unique())
        rights = sorted(group["column_index_right"].unique())
        left_index = {x: i for i, x in enumerate(lefts)}
        right_index = {x: i for i, x in enumerate(rights)}
        matrix = np.full((len(lefts), len(rights)), -1e6, dtype=float)
        for row in group.itertuples(index=False):
            matrix[left_index[row.column_index_left], right_index[row.column_index_right]] = row.score
        row_ind, col_ind = linear_sum_assignment(-matrix)
        matched = {(lefts[i], rights[j]) for i, j in zip(row_ind, col_ind) if matrix[i, j] >= threshold}
        for row in group.to_dict("records"):
            pred_rows.append(
                {
                    **row,
                    "pred": int((row["column_index_left"], row["column_index_right"]) in matched),
                }
            )
    return pd.DataFrame(pred_rows)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], scores: Optional[Sequence[float]] = None) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    if scores is not None and len(set(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    return out


def schema_simple_scores(split: str, corr: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for row in corr.to_dict("records"):
        left = tokenize(row["column_name_left"])
        right = tokenize(row["column_name_right"])
        score = 0.6 * jaccard(left, right) + 0.4 * levenshtein_ratio(
            normalize_header(row["column_name_left"]), normalize_header(row["column_name_right"])
        )
        scores.append({**row, "score": score})
    return pd.DataFrame(scores)


def schema_strong_scores(split: str, corr: pd.DataFrame, lookup: Dict[Tuple[str, str, str, int], Dict[str, Any]], model: LogisticRegression) -> pd.DataFrame:
    feat_rows = [schema_feature_row(row, lookup, split) for _, row in corr.iterrows()]
    X = pd.DataFrame(feat_rows)
    scores = model.predict_proba(X)[:, 1]
    return corr.assign(score=scores)


def train_schema_models(seed: int) -> Dict[str, Any]:
    if ("schema", seed) in MODEL_CACHE:
        return MODEL_CACHE[("schema", seed)]
    set_global_seed(seed)
    profiles = build_t2d_profiles()
    lookup = profile_lookup(profiles)
    train = load_t2d_split("train")["correspondences"]
    valid = load_t2d_split("valid")["correspondences"]
    best_simple = None
    best_simple_f1 = -1.0
    for threshold in [0.40, 0.50, 0.60, 0.70]:
        valid_scores = schema_simple_scores("valid", valid)
        pred = apply_assignment(valid_scores, threshold)
        f1 = f1_score(pred["label"], pred["pred"], zero_division=0)
        if f1 > best_simple_f1:
            best_simple_f1 = f1
            best_simple = threshold
    X_train = pd.DataFrame([schema_feature_row(row, lookup, "train") for _, row in train.iterrows()])
    y_train = train["label"].astype(int).to_numpy()
    X_valid = pd.DataFrame([schema_feature_row(row, lookup, "valid") for _, row in valid.iterrows()])
    y_valid = valid["label"].astype(int).to_numpy()
    best_cfg = None
    best_model = None
    best_f1 = -1.0
    for C in [0.1, 1.0, 10.0]:
        for class_weight in [None, "balanced"]:
            model = LogisticRegression(
                solver="liblinear",
                C=C,
                class_weight=class_weight,
                random_state=seed,
                max_iter=500,
            )
            model.fit(X_train, y_train)
            valid_scores = schema_strong_scores("valid", valid, lookup, model)
            for threshold in [0.40, 0.50, 0.60, 0.70]:
                pred = apply_assignment(valid_scores, threshold)
                f1 = f1_score(y_valid, pred["pred"], zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_cfg = {"C": C, "class_weight": class_weight, "threshold": threshold}
                    best_model = model
    MODEL_CACHE[("schema", seed)] = {"simple_threshold": best_simple, "strong_model": best_model, "strong_cfg": best_cfg, "lookup": lookup}
    return MODEL_CACHE[("schema", seed)]


def attach_schema_prediction_metadata(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df.copy()
    out["example_id"] = out.apply(
        lambda row: f"{row['table_name']}::{int(row['column_index_left'])}::{int(row['column_index_right'])}",
        axis=1,
    )
    out["score_margin"] = (out["score"].astype(float) - float(threshold)).abs()
    return out


def evaluate_schema(seed: int, split: str = "test") -> Dict[str, Dict[str, Any]]:
    if (f"schema_{split}", seed) in EVAL_CACHE:
        return EVAL_CACHE[(f"schema_{split}", seed)]
    models = train_schema_models(seed)
    corr = load_t2d_split(split)["correspondences"]
    simple_scores = schema_simple_scores(split, corr)
    simple_pred = attach_schema_prediction_metadata(
        apply_assignment(simple_scores, models["simple_threshold"]),
        models["simple_threshold"],
    )
    strong_scores = schema_strong_scores(split, corr, models["lookup"], models["strong_model"])
    strong_pred = attach_schema_prediction_metadata(
        apply_assignment(strong_scores, models["strong_cfg"]["threshold"]),
        models["strong_cfg"]["threshold"],
    )
    EVAL_CACHE[(f"schema_{split}", seed)] = {
        "schema_simple": {
            "metrics": compute_metrics(simple_pred["label"], simple_pred["pred"], simple_pred["score"]),
            "predictions": simple_pred,
            "config": {"threshold": models["simple_threshold"]},
        },
        "schema_strong": {
            "metrics": compute_metrics(strong_pred["label"], strong_pred["pred"], strong_pred["score"]),
            "predictions": strong_pred,
            "config": models["strong_cfg"],
        },
    }
    return EVAL_CACHE[(f"schema_{split}", seed)]


def entity_concat_text(df: pd.DataFrame, use_desc: bool = True) -> Tuple[pd.Series, pd.Series]:
    fields = ["title", "brand", "price"]
    if use_desc:
        fields.append("description")
    left = df[[f"{f}_left_norm" for f in fields]].fillna("").agg(" ".join, axis=1)
    right = df[[f"{f}_right_norm" for f in fields]].fillna("").agg(" ".join, axis=1)
    return left, right


def entity_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict("records"):
        lt = tokenize(row["title_left_norm"])
        rt = tokenize(row["title_right_norm"])
        lb = normalize_text(row["brand_left_norm"])
        rb = normalize_text(row["brand_right_norm"])
        left_text = row["combined_left"]
        right_text = row["combined_right"]
        price_left = parse_price(row["price_left_norm"])
        price_right = parse_price(row["price_right_norm"])
        rows.append(
            {
                "title_jaccard": jaccard(lt, rt),
                "title_lev": levenshtein_ratio(row["title_left_norm"], row["title_right_norm"]),
                "brand_exact": float(lb != "" and lb == rb),
                "model_exact": float(extract_model_tokens(left_text) == extract_model_tokens(right_text) and extract_model_tokens(left_text) != ""),
                "price_diff": abs(price_left - price_right) if price_left is not None and price_right is not None else 1.0,
                "identifier_match": float(extract_identifier_signal(row, "left") != "" and extract_identifier_signal(row, "left") == extract_identifier_signal(row, "right")),
                "attribute_overlap": jaccard(tokenize(left_text), tokenize(right_text)),
            }
        )
    return pd.DataFrame(rows)


def parse_price(text: str) -> Optional[float]:
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def extract_model_tokens(text: str) -> str:
    toks = re.findall(r"[a-z]*\d+[a-z0-9-]*", text or "")
    return " ".join(sorted(set(toks[:5])))


def extract_identifier_signal(row: Dict[str, Any], side: str) -> str:
    raw = row.get(f"specTableContent_{side}_norm", "") + " " + row.get(f"title_{side}_norm", "")
    matches = re.findall(r"(?:gtin|ean|sku|mpn|productid)\s*[a-z0-9-]+", raw)
    if matches:
        return " ".join(matches[:3])
    return extract_model_tokens(raw)


def train_entity_models(seed: int) -> Dict[str, Any]:
    if ("entity", seed) in MODEL_CACHE:
        return MODEL_CACHE[("entity", seed)]
    set_global_seed(seed)
    wdc = build_wdc_normalized_view()
    train = wdc[wdc["split"] == "train"].copy()
    valid = wdc[wdc["split"] == "valid"].copy()
    simple_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)
    train_left, train_right = entity_concat_text(train, use_desc=True)
    simple_vec.fit(pd.concat([train_left, train_right], axis=0))
    valid_left, valid_right = entity_concat_text(valid, use_desc=True)
    valid_sims = cosine_from_vectorizer(simple_vec, valid_left, valid_right)
    best_simple_thr = 0.5
    best_simple_f1 = -1.0
    for thr in [0.50, 0.60, 0.70, 0.80]:
        pred = (valid_sims >= thr).astype(int)
        f1 = f1_score(valid["label"], pred, zero_division=0)
        if f1 > best_simple_f1:
            best_simple_f1 = f1
            best_simple_thr = thr
    X_train = entity_pair_features(train)
    y_train = train["label"].astype(int).to_numpy()
    X_valid = entity_pair_features(valid)
    y_valid = valid["label"].astype(int).to_numpy()
    best_model = None
    best_cfg = None
    best_f1 = -1.0
    for C in [0.1, 1.0, 10.0]:
        for class_weight in [None, "balanced"]:
            model = LogisticRegression(
                solver="liblinear",
                penalty="l2",
                C=C,
                class_weight=class_weight,
                random_state=seed,
                max_iter=500,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            f1 = f1_score(y_valid, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_cfg = {"C": C, "class_weight": class_weight}
    MODEL_CACHE[("entity", seed)] = {"simple_vec": simple_vec, "simple_thr": best_simple_thr, "strong_model": best_model, "strong_cfg": best_cfg}
    return MODEL_CACHE[("entity", seed)]


def cosine_from_vectorizer(vectorizer: TfidfVectorizer, left_text: pd.Series, right_text: pd.Series) -> np.ndarray:
    left_mat = vectorizer.transform(left_text)
    right_mat = vectorizer.transform(right_text)
    return np.asarray(left_mat.multiply(right_mat).sum(axis=1)).ravel()


def evaluate_entity(seed: int, split: str = "test") -> Dict[str, Dict[str, Any]]:
    if (f"entity_{split}", seed) in EVAL_CACHE:
        return EVAL_CACHE[(f"entity_{split}", seed)]
    models = train_entity_models(seed)
    wdc = build_wdc_normalized_view()
    df = wdc[wdc["split"] == split].copy()
    left, right = entity_concat_text(df, use_desc=True)
    simple_scores = cosine_from_vectorizer(models["simple_vec"], left, right)
    simple_pred = (simple_scores >= models["simple_thr"]).astype(int)
    X = entity_pair_features(df)
    strong_scores = models["strong_model"].predict_proba(X)[:, 1]
    strong_pred = (strong_scores >= 0.5).astype(int)
    EVAL_CACHE[(f"entity_{split}", seed)] = {
        "entity_simple": {
            "metrics": compute_metrics(df["label"], simple_pred, simple_scores),
            "predictions": df.assign(
                score=simple_scores,
                pred=simple_pred,
                example_id=df["pair_id"].astype(str),
                score_margin=np.abs(simple_scores - models["simple_thr"]),
            ),
            "config": {"threshold": models["simple_thr"]},
        },
        "entity_strong": {
            "metrics": compute_metrics(df["label"], strong_pred, strong_scores),
            "predictions": df.assign(
                score=strong_scores,
                pred=strong_pred,
                example_id=df["pair_id"].astype(str),
                score_margin=np.abs(strong_scores - 0.5),
            ),
            "config": models["strong_cfg"],
        },
    }
    return EVAL_CACHE[(f"entity_{split}", seed)]


SCHEMA_ALIAS_TABLE = {"yr": "year", "pub": "publisher", "dob": "date of birth"}
BOILERPLATE = [
    "free shipping",
    "limited time offer",
    "official retailer",
    "secure checkout",
    "fast delivery",
]


def schema_protected_sets(split: str) -> Dict[str, Dict[str, Any]]:
    corr = load_t2d_split(split)["correspondences"]
    profiles = profile_lookup(build_t2d_profiles())
    protected: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "positive_left": set(),
            "positive_right": set(),
            "competitor_left": set(),
            "matched_pairs": set(),
        }
    )
    positives = corr[corr["label"] == 1]
    for row in positives.to_dict("records"):
        protected[row["table_name"]]["positive_left"].add(int(row["column_index_left"]))
        protected[row["table_name"]]["positive_right"].add(int(row["column_index_right"]))
        protected[row["table_name"]]["matched_pairs"].add(
            (int(row["column_index_left"]), int(row["column_index_right"]))
        )
    for table_name, group in corr.groupby("table_name"):
        pos_lefts = protected[table_name]["positive_left"]
        for row in group.to_dict("records"):
            if int(row["column_index_left"]) in pos_lefts:
                continue
            for pos_idx in pos_lefts:
                key_pos = (split, "webtables", table_name + ".csv", pos_idx)
                key_cur = (split, "webtables", table_name + ".csv", int(row["column_index_left"]))
                if key_pos not in profiles or key_cur not in profiles:
                    continue
                p = profiles[key_pos]
                c = profiles[key_cur]
                header_sim = fuzz.token_set_ratio(p["normalized_header"], c["normalized_header"]) / 100.0
                value_sim = counter_cosine(p["value_token_counter"], c["value_token_counter"])
                if header_sim >= 0.80 or value_sim >= 0.90:
                    protected[table_name]["competitor_left"].add(int(row["column_index_left"]))
    for value in protected.values():
        value["protected_left"] = set(value["positive_left"]) | set(value["competitor_left"])
    return protected


def alias_normalized_header(header: str) -> str:
    norm = normalize_header(header)
    if norm in SCHEMA_ALIAS_TABLE:
        return normalize_header(SCHEMA_ALIAS_TABLE[norm])
    reverse = {normalize_header(v): normalize_header(v) for v in SCHEMA_ALIAS_TABLE.values()}
    return reverse.get(norm, norm)


def schema_column_signature(df: pd.DataFrame, idx: int) -> Optional[Dict[str, Any]]:
    if idx >= len(df.columns):
        return None
    series = df.iloc[:, idx]
    return {
        "header_alias_norm": alias_normalized_header(df.columns[idx]),
        "primitive_type": infer_primitive_type(series.tolist()),
        "normalized_values": normalized_value_counter(series.tolist()),
    }


def schema_admissibility_decision(
    original_df: pd.DataFrame,
    perturbed_df: pd.DataFrame,
    protected_cols: Dict[str, Any],
    ablation: Optional[str] = None,
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    if ablation == "generic":
        return logs
    original_signatures = {
        idx: schema_column_signature(original_df, idx) for idx in protected_cols["positive_left"]
    }
    if ablation != "remove_competitors":
        for idx in protected_cols["competitor_left"]:
            original_signatures[idx] = schema_column_signature(original_df, idx)
    for idx, original_sig in sorted(original_signatures.items()):
        new_sig = schema_column_signature(perturbed_df, idx)
        if original_sig is None or new_sig is None:
            logs.append({"reason_code": "protected_value_changed"})
            break
        if new_sig["header_alias_norm"] != original_sig["header_alias_norm"]:
            logs.append({"reason_code": "protected_header_changed"})
            break
        if new_sig["primitive_type"] != original_sig["primitive_type"]:
            logs.append({"reason_code": "type_changed"})
            break
        if new_sig["normalized_values"] != original_sig["normalized_values"]:
            logs.append({"reason_code": "protected_value_changed"})
            break
    if logs or ablation == "remove_competitors":
        return logs
    baseline_competitors = set(protected_cols["competitor_left"])
    protected_positive = set(protected_cols["positive_left"])
    positive_sigs = {idx: schema_column_signature(original_df, idx) for idx in protected_positive}
    for idx in range(len(perturbed_df.columns)):
        if idx in protected_cols["protected_left"]:
            continue
        candidate_sig = schema_column_signature(perturbed_df, idx)
        if candidate_sig is None:
            continue
        for pos_idx, pos_sig in positive_sigs.items():
            if pos_sig is None:
                continue
            header_sim = fuzz.token_set_ratio(
                candidate_sig["header_alias_norm"], pos_sig["header_alias_norm"]
            ) / 100.0
            value_sim = counter_cosine(
                candidate_sig["normalized_values"], pos_sig["normalized_values"]
            )
            if (header_sim >= 0.80 or value_sim >= 0.90) and idx not in baseline_competitors:
                logs.append({"reason_code": "new_competitor_created"})
                return logs
    return logs


def perturb_schema_table(
    df: pd.DataFrame,
    program: Sequence[Tuple[str, str]],
    rng: random.Random,
    regime: str,
    protected_cols: Dict[str, Any],
    audit_log: List[Dict[str, Any]],
) -> pd.DataFrame:
    out = df.copy()
    for family, severity in program:
        for idx, col in list(enumerate(out.columns)):
            if family == "row_reorder":
                out = out.sample(frac=1.0, random_state=rng.randint(0, 10_000)).reset_index(drop=True)
                continue
            if family == "header_case_delimiter":
                new_col = re.sub(r"[^\w]+", "_", col.lower()).strip("_")
                if regime == "naive":
                    new_col = re.sub(r"[aeiou]", "", new_col) or new_col
                out = out.rename(columns={col: new_col})
                continue
            if family == "header_abbreviation":
                if regime == "ABCA":
                    norm = normalize_header(col)
                    alias = None
                    if norm in SCHEMA_ALIAS_TABLE.values():
                        alias = next(k for k, v in SCHEMA_ALIAS_TABLE.items() if normalize_header(v) == norm)
                    elif norm in SCHEMA_ALIAS_TABLE:
                        alias = norm
                    if alias is not None:
                        out = out.rename(columns={col: alias})
                    else:
                        audit_log.append({"reason_code": "unsupported_alias", "family": family, "severity": severity})
                else:
                    out = out.rename(columns={col: "".join(token[0] for token in tokenize(col)) or col})
                continue
            if family == "value_format":
                out.iloc[:, idx] = out.iloc[:, idx].map(format_rewrite)
                continue
            if family == "column_dropout":
                if rng.random() < severity_to_prob(severity):
                    out.iloc[:, idx] = ""
                    out = out.rename(columns={col: f"{col}__dropped"})
                continue
            if family == "token_shuffle" and regime != "ABCA":
                out.iloc[:, idx] = out.iloc[:, idx].map(lambda x: shuffle_text(x, rng))
    return out


def format_rewrite(value: Any) -> Any:
    text = normalize_text(value)
    numeric = canonicalize_numeric_token(text)
    if numeric is not None:
        try:
            return f"{float(numeric):,.2f}"
        except ValueError:
            return text
    date = canonicalize_date_token(text)
    if date is not None:
        return date.replace("-", "/")
    return text.replace("-", "/")


def shuffle_text(value: Any, rng: random.Random) -> Any:
    toks = tokenize(value)
    if len(toks) < 2:
        return value
    rng.shuffle(toks)
    return " ".join(toks)


def severity_to_prob(severity: str) -> float:
    return {"low": 0.3, "medium": 0.6, "high": 0.9}[severity]


def regime_enforces_admissibility(regime: str) -> bool:
    return regime == "ABCA"


def summarize_program_decision(
    accepted: bool,
    audit_log: Sequence[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reasons = Counter(item["reason_code"] for item in audit_log if item.get("reason_code"))
    payload = {
        "accepted": bool(accepted),
        "accepted_programs": int(bool(accepted)),
        "rejected_programs": int(not accepted),
        "reasons": dict(reasons),
        "primary_reason": next(iter(reasons.keys()), None),
    }
    if extra:
        payload.update(extra)
    return payload


def extract_identifier_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    labelled = re.findall(r"(?:gtin|ean|sku|mpn|productid)\s*[a-z0-9-]+", normalized)
    generic = re.findall(r"\b[a-z]{0,4}\d[a-z0-9-]{2,}\b", normalized)
    return set(labelled) | set(generic)


def extract_quantity_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    quantities = set()
    price = parse_price(normalized)
    if price is not None:
        quantities.add(f"{price:.2f}")
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(gb|tb|mb|kg|g|lb|oz|ml|l|cm|mm|inch|in|pack)", normalized):
        quantities.add(f"{float(number):.2f}_{unit}")
    return quantities


def entity_protected_tokens(row: Dict[str, Any], ablation: Optional[str] = None) -> Dict[str, Dict[str, set]]:
    payload: Dict[str, Dict[str, set]] = {}
    for side in ["left", "right"]:
        title = normalize_text(row.get(f"title_{side}", row.get(f"title_{side}_norm", "")))
        description = normalize_text(row.get(f"description_{side}", row.get(f"description_{side}_norm", "")))
        brand_text = normalize_text(row.get(f"brand_{side}", row.get(f"brand_{side}_norm", "")))
        spec = normalize_text(row.get(f"specTableContent_{side}", row.get(f"specTableContent_{side}_norm", "")))
        price_text = normalize_text(row.get(f"price_{side}", row.get(f"price_{side}_norm", "")))
        combined = " ".join([title, description, brand_text, spec, price_text]).strip()
        identifiers = extract_identifier_tokens(combined)
        quantities = extract_quantity_tokens(" ".join([title, spec, price_text]))
        brands = set(tokenize(brand_text))
        models = set(tokenize(extract_model_tokens(combined)))
        title_tokens = tokenize(title)
        protected_title_tokens = {tok for tok in title_tokens if tok in identifiers or tok in brands or tok in models}
        if ablation == "weaken_em":
            brands = set()
            models = set()
            protected_title_tokens = {tok for tok in title_tokens if tok in identifiers}
        elif ablation == "generic":
            quantities = set()
            brands = set()
            models = set()
            protected_title_tokens = set()
        payload[side] = {
            "identifiers": identifiers,
            "quantities": quantities,
            "brands": brands,
            "models": models,
            "protected_title_tokens": protected_title_tokens,
        }
    return payload


def allowed_entity_format_only(program: Sequence[Tuple[str, str]]) -> bool:
    return all(family in {"case_punct", "numeric_rewrite"} for family, _ in program)


def entity_admissibility_decision(
    original_row: Dict[str, Any],
    perturbed_row: Dict[str, Any],
    program: Sequence[Tuple[str, str]],
    ablation: Optional[str] = None,
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    original = entity_protected_tokens(original_row, ablation=ablation)
    perturbed = entity_protected_tokens(perturbed_row, ablation=ablation)
    for side in ["left", "right"]:
        if original[side]["identifiers"] != perturbed[side]["identifiers"]:
            return [{"reason_code": "identifier_changed"}]
        if ablation != "generic" and original[side]["quantities"] != perturbed[side]["quantities"]:
            return [{"reason_code": "quantity_changed"}]
        if ablation not in {"weaken_em", "generic"}:
            if (
                original[side]["brands"] != perturbed[side]["brands"]
                or original[side]["models"] != perturbed[side]["models"]
            ):
                return [{"reason_code": "brand_model_changed"}]
            if original[side]["protected_title_tokens"] != perturbed[side]["protected_title_tokens"]:
                return [{"reason_code": "edited_protected_title_span"}]
            evidence_present = bool(
                original[side]["identifiers"] or (original[side]["brands"] and original[side]["models"])
            )
            if not evidence_present and not allowed_entity_format_only(program):
                return [{"reason_code": "insufficient_protected_evidence"}]
    return logs


def perturb_entity_row(row: Dict[str, Any], program: Sequence[Tuple[str, str]], rng: random.Random, regime: str, ablation: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out = dict(row)
    logs = []
    protected = entity_protected_tokens(row, ablation=ablation)
    for family, severity in program:
        for side in ["left", "right"]:
            title_key = f"title_{side}_norm"
            text = out[title_key]
            if family == "case_punct":
                out[title_key] = re.sub(r"\s+", " ", text.replace("-", " ").replace("/", " ")).strip()
            elif family == "boilerplate":
                phrase = BOILERPLATE[rng.randrange(len(BOILERPLATE))]
                candidate = (text + " " + phrase).strip()
                out[title_key] = candidate
            elif family == "token_permute":
                toks = tokenize(text)
                if len(toks) > 2:
                    protected_tokens = (
                        protected[side]["identifiers"]
                        | protected[side]["brands"]
                        | protected[side]["models"]
                    )
                    free = [t for t in toks if t not in protected_tokens]
                    frozen = [t for t in toks if t in protected_tokens]
                    rng.shuffle(free)
                    out[title_key] = " ".join(free + frozen)
            elif family == "attribute_dropout":
                field = rng.choice(["description", "brand", "specTableContent"])
                out[f"{field}_{side}_norm"] = ""
            elif family == "numeric_rewrite":
                price_key = f"price_{side}_norm"
                price = parse_price(out[price_key])
                if price is not None:
                    out[price_key] = f"usd {price:.2f}"
            elif family == "random_delete" and regime == "naive":
                toks = tokenize(text)
                if toks:
                    del toks[rng.randrange(len(toks))]
                    out[title_key] = " ".join(toks)
    if regime_enforces_admissibility(regime):
        logs.extend(entity_admissibility_decision(row, out, program, ablation=ablation))
    return out, logs


def schema_program_space() -> List[Tuple[str, str]]:
    families = ["header_case_delimiter", "header_abbreviation", "value_format", "row_reorder", "column_dropout", "token_shuffle"]
    return [(f, s) for f in families for s in ["low", "medium", "high"]]


def entity_program_space() -> List[Tuple[str, str]]:
    families = ["case_punct", "boilerplate", "attribute_dropout", "numeric_rewrite", "token_permute", "random_delete"]
    return [(f, s) for f in families for s in ["low", "medium", "high"]]


def generate_programs(space: List[Tuple[str, str]], rng: random.Random, n: int = 40) -> List[List[Tuple[str, str]]]:
    programs = []
    for _ in range(n):
        length = rng.randint(1, 3)
        programs.append([space[rng.randrange(len(space))] for _ in range(length)])
    return programs


def evaluate_schema_under_program(
    seed: int,
    method: str,
    split: str,
    program: List[Tuple[str, str]],
    regime: str,
    ablation: Optional[str] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame], Dict[str, Any]]:
    models = train_schema_models(seed)
    extracted = DATA_PROCESSED / "t2d_sm_wh"
    maybe_extract_zip(T2D_ZIP, extracted)
    base = load_t2d_split(split)
    protected = schema_protected_sets(split)
    audit_log: List[Dict[str, Any]] = []
    temp_dir = DATA_CACHE / f"tmp_t2d_{seed}_{method}_{regime}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    shutil.copytree(extracted / split, temp_dir)
    for csv_path in (temp_dir / "webtables").glob("*.csv"):
        df = pd.read_csv(csv_path)
        table_name = csv_path.stem
        prot = protected[table_name]
        perturbed = perturb_schema_table(df, program, random.Random(seed + len(table_name)), regime, prot, audit_log)
        if regime_enforces_admissibility(regime):
            audit_log.extend(schema_admissibility_decision(df, perturbed, prot, ablation=ablation))
        perturbed.to_csv(csv_path, index=False)
    accepted = not (regime_enforces_admissibility(regime) and audit_log)
    if not accepted:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return None, None, summarize_program_decision(False, audit_log)
    corr = base["correspondences"].copy()
    if method == "schema_simple":
        scores = schema_simple_scores(split, corr)
        pred = apply_assignment(scores, models["simple_threshold"])
    else:
        # Rebuild profiles for the perturbed split only.
        prof_rows = []
        for side in ["webtables", "dbpedia_tables"]:
            for csv_path in (temp_dir / side).glob("*.csv"):
                df = pd.read_csv(csv_path)
                for idx, column in enumerate(df.columns):
                    profile = t2d_profile_column(df[column], column)
                    prof_rows.append({"split": split, "side": side, "table_name": csv_path.name, "column_index": idx, **profile})
        prof_df = pd.DataFrame(prof_rows)
        prof_df["value_token_counter_json"] = prof_df["value_token_counter"].apply(lambda c: json.dumps(c))
        prof_df["normalized_value_counter_json"] = prof_df["normalized_value_counter"].apply(lambda c: json.dumps(c))
        prof_df = prof_df.drop(columns=["value_token_counter", "normalized_value_counter"])
        lookup = profile_lookup(prof_df)
        scores = schema_strong_scores(split, corr, lookup, models["strong_model"])
        pred = apply_assignment(scores, models["strong_cfg"]["threshold"])
    metrics = compute_metrics(pred["label"], pred["pred"], pred["score"])
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return metrics, pred, summarize_program_decision(True, audit_log)


def evaluate_entity_under_program(
    seed: int,
    method: str,
    split: str,
    program: List[Tuple[str, str]],
    regime: str,
    ablation: Optional[str] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame], Dict[str, Any]]:
    models = train_entity_models(seed)
    wdc = build_wdc_normalized_view()
    df = wdc[wdc["split"] == split].copy()
    rows = []
    all_logs = []
    rng = random.Random(seed)
    for row in df.to_dict("records"):
        perturbed, logs = perturb_entity_row(row, program, rng, regime, ablation=ablation)
        rows.append(perturbed)
        all_logs.extend(logs)
    accepted = not (regime_enforces_admissibility(regime) and all_logs)
    if not accepted:
        return None, None, summarize_program_decision(False, all_logs)
    out = pd.DataFrame(rows)
    if method == "entity_simple":
        left, right = entity_concat_text(out, use_desc=True)
        scores = cosine_from_vectorizer(models["simple_vec"], left, right)
        pred = (scores >= models["simple_thr"]).astype(int)
    else:
        X = entity_pair_features(out)
        scores = models["strong_model"].predict_proba(X)[:, 1]
        pred = (scores >= 0.5).astype(int)
    scored = out.assign(score=scores, pred=pred)
    metrics = compute_metrics(scored["label"], scored["pred"], scored["score"])
    return metrics, scored, summarize_program_decision(True, all_logs)


def search_programs(
    benchmark: str,
    method: str,
    seed: int,
    regime: str,
    mode: str,
    split_for_search: str = "valid",
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    space = schema_program_space() if benchmark == "t2d_sm_wh" else entity_program_space()
    eval_fn = evaluate_schema_under_program if benchmark == "t2d_sm_wh" else evaluate_entity_under_program
    trajectories = []
    best_program: List[Tuple[str, str]] = []
    clean_eval = evaluate_schema(seed)[method] if benchmark == "t2d_sm_wh" else evaluate_entity(seed)[method]
    clean = clean_eval["metrics"]
    clean_predictions = clean_eval["predictions"]
    best_f1 = clean["f1"]
    accepted_programs = 0
    rejected_programs = 0
    rejection_reasons: Counter = Counter()
    eval_budget = 40
    budget_points = [0, 10, 20, 40]
    curve = {}
    curve[0] = clean["f1"]
    if mode == "random":
        programs = generate_programs(space, rng, n=eval_budget)
        for i, program in enumerate(programs, start=1):
            metrics, _, aux = eval_fn(seed, method, split_for_search, program, regime, ablation=ablation)
            accepted_programs += aux["accepted_programs"]
            rejected_programs += aux["rejected_programs"]
            rejection_reasons.update(aux["reasons"])
            trajectories.append({"eval_index": i, "program": program, "metrics": metrics, "accepted": aux["accepted"], "aux": aux})
            if metrics is not None and metrics["f1"] < best_f1:
                best_f1 = metrics["f1"]
                best_program = program
            if i in budget_points:
                curve[i] = best_f1 if accepted_programs else clean["f1"]
    else:
        beam: List[List[Tuple[str, str]]] = [[]]
        eval_count = 0
        for depth in range(1, 4):
            candidates = []
            for prefix in beam:
                for op in space:
                    program = prefix + [op]
                    eval_count += 1
                    metrics, _, aux = eval_fn(seed, method, split_for_search, program, regime, ablation=ablation)
                    accepted_programs += aux["accepted_programs"]
                    rejected_programs += aux["rejected_programs"]
                    rejection_reasons.update(aux["reasons"])
                    if metrics is not None:
                        candidates.append((metrics["f1"], program, aux))
                    trajectories.append({"eval_index": eval_count, "program": program, "metrics": metrics, "accepted": aux["accepted"], "aux": aux})
                    if metrics is not None and metrics["f1"] < best_f1:
                        best_f1 = metrics["f1"]
                        best_program = program
                    if eval_count in budget_points:
                        curve[eval_count] = best_f1 if accepted_programs else clean["f1"]
                    if eval_count >= eval_budget:
                        break
                if eval_count >= eval_budget:
                    break
            if not candidates:
                break
            beam = [program for _, program, _ in sorted(candidates, key=lambda x: x[0])[:8]]
            if eval_count >= eval_budget:
                break
        curve.setdefault(10, best_f1 if accepted_programs else clean["f1"])
        curve.setdefault(20, best_f1 if accepted_programs else clean["f1"])
        curve.setdefault(40, best_f1 if accepted_programs else clean["f1"])
    if best_program:
        test_metrics, predictions, test_aux = eval_fn(seed, method, "test", best_program, regime, ablation=ablation)
        if test_metrics is None or predictions is None:
            test_metrics = clean
            predictions = clean_predictions
            test_aux = {
                **test_aux,
                "selected_clean_fallback": True,
                "test_rejected_selected_program": True,
            }
        elif test_metrics["f1"] >= clean["f1"]:
            test_metrics = clean
            predictions = clean_predictions
            test_aux = {
                **test_aux,
                "selected_clean_fallback": True,
                "test_worst_case_clipped_to_clean": True,
            }
    else:
        test_metrics = clean
        predictions = clean_predictions
        test_aux = summarize_program_decision(True, [], {"selected_clean_fallback": True})
    search_aux = {
        "accepted_programs": accepted_programs,
        "rejected_programs": rejected_programs,
        "acceptance_rate": float(accepted_programs / max(accepted_programs + rejected_programs, 1)),
        "reasons": dict(rejection_reasons),
        "selected_program_accepted": bool(best_program),
        "selected_clean_fallback": not bool(best_program),
        "test_program_aux": test_aux,
    }
    return {
        "benchmark": benchmark,
        "method": method,
        "seed": seed,
        "regime": regime,
        "search_mode": mode,
        "best_program": best_program,
        "search_trajectory": trajectories,
        "curve": curve,
        "eval_budget": eval_budget,
        "clean_metrics": clean,
        "worst_metrics": test_metrics,
        "predictions": predictions,
        "aux": search_aux,
    }


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def package_versions() -> Dict[str, str]:
    versions = {}
    for name in PACKAGE_VERSION_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def peak_ram_gb() -> float:
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system().lower() == "darwin":
        return float(usage_kb / (1024**3))
    return float(usage_kb / 1024 / 1024)


def runtime_info(start_time: float) -> Dict[str, Any]:
    end_time = time.time()
    return {
        "start_time_unix": start_time,
        "end_time_unix": end_time,
        "wall_clock_minutes": (end_time - start_time) / 60.0,
        "peak_ram_gb": peak_ram_gb(),
    }


def write_run_artifacts(run_dir: Path, config: Dict[str, Any], result: Dict[str, Any], start_time: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    save_predictions(result["predictions"], run_dir / "predictions.parquet")
    write_json(run_dir / "metrics.json", result["worst_metrics"])
    write_json(
        run_dir / "runtime.json",
        {
            **runtime_info(start_time),
            "system_info": system_info(),
            "package_versions": package_versions(),
        },
    )
    write_json(
        run_dir / "perturbations.json",
        {
            "best_program": result["best_program"],
            "curve": result["curve"],
            "aux": result["aux"],
            "eval_budget": result["eval_budget"],
        },
    )
    write_jsonl(run_dir / "admissibility_log.jsonl", result["search_trajectory"])
    perturbation_rows = []
    for row in result["search_trajectory"]:
        metrics = row.get("metrics") or {}
        aux = row.get("aux") or {}
        perturbation_rows.append(
            {
                "eval_index": row.get("eval_index"),
                "accepted": row.get("accepted"),
                "program": row.get("program"),
                "program_length": len(row.get("program") or []),
                "operator_families": [op[0] for op in row.get("program") or []],
                "severities": [op[1] for op in row.get("program") or []],
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "auroc": metrics.get("auroc"),
                "accepted_programs": aux.get("accepted_programs"),
                "rejected_programs": aux.get("rejected_programs"),
                "primary_reason": aux.get("primary_reason"),
                "reasons": aux.get("reasons", {}),
            }
        )
    write_jsonl(run_dir / "perturbations.jsonl", perturbation_rows)

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = [
        {"budget": int(budget), "best_f1": float(value)}
        for budget, value in sorted((int(k), float(v)) for k, v in result["curve"].items())
    ]
    pd.DataFrame(curve_rows).to_csv(plot_dir / "budget_curve.csv", index=False)
    if curve_rows:
        curve_df = pd.DataFrame(curve_rows)
        plt.figure(figsize=(4, 3))
        plt.plot(curve_df["budget"], curve_df["best_f1"], marker="o")
        plt.xlabel("Evaluation Budget")
        plt.ylabel("Best F1")
        plt.tight_layout()
        plt.savefig(plot_dir / "budget_curve.png", dpi=200)
        plt.close()

    write_json(
        run_dir / "metadata.json",
        {
            "config": config,
            "system_info": system_info(),
            "package_versions": package_versions(),
            "selected_program": result["best_program"],
            "selected_clean_fallback": result["aux"].get("selected_clean_fallback", False),
        },
    )


def system_info() -> Dict[str, Any]:
    mem_gb = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            data = f.read()
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", data)
        if match:
            mem_gb = round(int(match.group(1)) / 1024 / 1024, 2)
    except OSError:
        pass
    cpu_model = None
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count_observed": os.cpu_count(),
        "ram_gb_observed": mem_gb,
        "cpu_count_allocated": ALLOCATED_CPU_CORES,
        "ram_gb_allocated": ALLOCATED_RAM_GB,
        "gpu_count": 0,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "1"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "1"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
    }


def bootstrap_ci(values: Sequence[float], seed: int = 13, n_resamples: int = 1000) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        resample = rng.choice(arr, size=arr.size, replace=True)
        samples.append(float(np.mean(resample)))
    return {
        "low": float(np.percentile(samples, 2.5)),
        "high": float(np.percentile(samples, 97.5)),
    }


def bootstrap_paired_prediction_delta(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    seed: int = 13,
    n_resamples: int = 1000,
) -> Dict[str, float]:
    if "example_id" in left_df.columns and "example_id" in right_df.columns:
        left = left_df.sort_values("example_id").reset_index(drop=True)
        right = right_df.sort_values("example_id").reset_index(drop=True)
    else:
        left = left_df.sort_values(
            "pair_id" if "pair_id" in left_df.columns else ["table_name", "column_index_left", "column_index_right"]
        ).reset_index(drop=True)
        right = right_df.sort_values(
            "pair_id" if "pair_id" in right_df.columns else ["table_name", "column_index_left", "column_index_right"]
        ).reset_index(drop=True)
    if len(left) != len(right):
        raise ValueError("Prediction frames must be aligned for paired bootstrap.")
    y_true = left["label"].astype(int).to_numpy()
    left_pred = left["pred"].astype(int).to_numpy()
    right_pred = right["pred"].astype(int).to_numpy()
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        left_f1 = f1_score(y_true[idx], left_pred[idx], zero_division=0)
        right_f1 = f1_score(y_true[idx], right_pred[idx], zero_division=0)
        deltas.append(float(left_f1 - right_f1))
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


def dataset_statistics() -> Dict[str, Any]:
    t2d_profiles = build_t2d_profiles()
    wdc = build_wdc_normalized_view()
    t2d_stats = {}
    for split in ["train", "valid", "test"]:
        bundle = load_t2d_split(split)
        corr = bundle["correspondences"]
        t2d_stats[split] = {
            "tables": int(corr["table_name"].nunique()),
            "source_columns": int(corr[["table_name", "column_index_left"]].drop_duplicates().shape[0]),
            "target_columns": int(corr[["table_name", "column_index_right"]].drop_duplicates().shape[0]),
            "positive_correspondences": int(corr["label"].sum()),
            "candidate_pairs": int(len(corr)),
        }
    field_coverages = {}
    for split in ["train", "valid", "test"]:
        sdf = wdc[wdc["split"] == split]
        field_coverages[split] = {
            "pairs": int(len(sdf)),
            "positives": int(sdf["label"].sum()),
            "negatives": int(len(sdf) - sdf["label"].sum()),
            "identifier_field_coverage": float(
                np.mean(
                    [
                        bool(extract_identifier_signal(row, "left")) and bool(extract_identifier_signal(row, "right"))
                        for row in sdf.to_dict("records")
                    ]
                )
            ),
            "brand_field_coverage": float(np.mean((sdf["brand_left_norm"] != "") & (sdf["brand_right_norm"] != ""))),
            "model_token_coverage": float(
                np.mean(
                    [
                        bool(extract_model_tokens(row["combined_left"])) and bool(extract_model_tokens(row["combined_right"]))
                        for row in sdf.to_dict("records")
                    ]
                )
            ),
            "price_field_coverage": float(np.mean((sdf["price_left_norm"] != "") & (sdf["price_right_norm"] != ""))),
        }
    return {"t2d_sm_wh": t2d_stats, "wdc_products_medium": field_coverages}
