import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import torch
import scipy
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
DATA_EMBEDDINGS = ROOT / "data" / "embeddings"
FIGURES = ROOT / "figures"
FIG_SRC = FIGURES / "source_data"
EXP = ROOT / "exp"

SEEDS = [13, 29, 47]
COVERAGE_TARGETS = [0.60, 0.80, 0.95]
MIN_CHILD_SUPPORT = 25
MIN_PARENT_CHILDREN = 4
MIN_PARENT_TOTAL_SUPPORT = 140
MAX_PARENTS = 12
MIN_SEQ_LEN = 30
MAX_SEQ_LEN = 1024
EMBED_BATCH_SIZE = 8
EMBED_MAX_TOKENS = 12000
EMBED_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
KNOWN_CHILD_ALPHA = 0.10
MASKED_CHILD_QUANTILE = 0.90
BOOTSTRAP_ITERS = 200

BASELINE_METHODS = [
    "one_nn_threshold",
    "flat_logistic",
    "forced_hierarchical",
    "max_probability",
    "hsc_style",
    "energy_threshold",
    "parent_openmax",
]
ABLATION_METHODS = [
    "ablation_no_masked_gate",
    "ablation_no_conformal_gate",
    "ablation_global_masked_threshold",
    "ablation_no_retrieval_surrogate",
]
ALL_METHODS = BASELINE_METHODS + ["mcc_ec"] + ABLATION_METHODS


def ensure_dirs() -> None:
    for path in [
        DATA_RAW,
        DATA_PROCESSED,
        DATA_SPLITS,
        DATA_EMBEDDINGS,
        FIGURES,
        FIG_SRC,
        EXP / "environment",
        EXP / "data_prep",
        EXP / "shared_scorer",
        EXP / "surrogate_benchmark",
        EXP / "selective_baselines",
        EXP / "mcc_ec",
        EXP / "transfer_analysis",
        EXP / "ablations",
        EXP / "summary",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    for method in ALL_METHODS:
        base = EXP / method
        (base / "logs").mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seed_dir = base / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "logs").mkdir(parents=True, exist_ok=True)

    for stage in ["environment", "data_prep", "shared_scorer", "surrogate_benchmark", "selective_baselines", "mcc_ec", "transfer_analysis", "ablations", "summary"]:
        (EXP / stage / "logs").mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        (EXP / "surrogate_benchmark" / f"seed_{seed}").mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
        "sha256": sha256_file(path),
    }


def log_stage(stage: str, message: str, seed: Optional[int] = None) -> None:
    append_log(EXP / stage / "logs" / "run.log", message)
    if seed is not None:
        append_log(EXP / stage / f"seed_{seed}" / "logs" / "run.log", message)


def log_method(method: str, seed: int, message: str) -> None:
    append_log(EXP / method / "logs" / "run.log", f"seed={seed} {message}")
    append_log(EXP / method / f"seed_{seed}" / "logs" / "run.log", message)


def sanitize_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True)), path)


def quantile_threshold(scores: np.ndarray, target_coverage: float) -> float:
    if scores.size == 0:
        return float("inf")
    quantile = max(0.0, min(1.0, 1.0 - target_coverage))
    return float(np.quantile(scores, quantile, method="linear"))


def ec_prefix(ec4: str, depth: int) -> str:
    parts = ec4.split(".")
    return ".".join(parts[:depth] + ["-"] * (4 - depth))


def ec_depth(label: str) -> int:
    return sum(1 for part in label.split(".") if part != "-")


def common_prefix_depth(a: str, b: str) -> int:
    depth = 0
    for x, y in zip(a.split("."), b.split(".")):
        if x == y and x != "-":
            depth += 1
        else:
            break
    return depth


def is_safe_prefix(pred: str, truth: str) -> bool:
    return common_prefix_depth(pred, truth) == ec_depth(pred)


def hierarchical_loss(pred: str, truth: str) -> float:
    return float(4 - common_prefix_depth(pred, truth))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iters: int = BOOTSTRAP_ITERS) -> Tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(iters):
        indices = rng.integers(0, values.size, size=values.size)
        means.append(float(values[indices].mean()))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def confidence_interval(values: np.ndarray, rng: np.random.Generator, iters: int = BOOTSTRAP_ITERS) -> Tuple[Optional[float], Optional[float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (None, None)
    mean = float(values.mean())
    if values.size == 1:
        return (mean, mean)
    stderr = float(values.std(ddof=1) / np.sqrt(values.size))
    half_width = 1.96 * stderr
    return (mean - half_width, mean + half_width)


def exact_sign_flip_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan")
    observed = abs(diffs.mean())
    sums = []
    for mask in range(1 << n):
        signs = np.array([1.0 if (mask >> bit) & 1 else -1.0 for bit in range(n)], dtype=float)
        sums.append(abs(np.mean(signs * diffs)))
    sums = np.asarray(sums)
    return float(np.mean(sums >= observed))


def mean_std_summary(series: pd.Series) -> dict:
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1) if values.size > 1 else 0.0),
        "n": int(values.size),
    }


def compute_metric_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    safe_prefix = df.apply(lambda row: is_safe_prefix(row["returned_prefix"], row["true_ec4"]), axis=1).astype(float).to_numpy()
    catastrophic = ((df["returned_depth"] == 4) & (df["returned_prefix"] != df["true_ec4"])).astype(float).to_numpy()
    hier_loss = df.apply(lambda row: hierarchical_loss(row["returned_prefix"], row["true_ec4"]), axis=1).astype(float).to_numpy()
    leaf_mask = (df["returned_depth"] == 4).to_numpy()
    selective_f1 = np.array(
        [
            float(f1_score(df.loc[leaf_mask, "true_ec4"], df.loc[leaf_mask, "returned_prefix"], average="micro"))
            if leaf_mask.any()
            else 0.0
        ],
        dtype=float,
    )
    return {
        "catastrophic_overspecialization_rate": catastrophic,
        "correct_safe_prefix_rate": safe_prefix,
        "hierarchical_loss": hier_loss,
        "selective_ec4_f1": selective_f1,
        "realized_coverage": (df["returned_depth"] == 4).astype(float).to_numpy(),
        "mean_returned_depth": df["returned_depth"].astype(float).to_numpy(),
    }


def parse_ecs_from_de(lines: List[str]) -> List[str]:
    ecs = []
    for line in lines:
        ecs.extend(re.findall(r"EC=([0-9n\-]+\.[0-9n\-]+\.[0-9n\-]+\.[0-9n\-]+)", line))
    return ecs


def parse_record(lines: List[str]) -> dict:
    accession = None
    entry_name = None
    sequence_parts = []
    de_lines = []
    in_sequence = False
    for line in lines:
        if line.startswith("ID"):
            parts = line.split()
            entry_name = parts[1] if len(parts) > 1 else None
        elif line.startswith("AC") and accession is None:
            accession = line[5:].strip().split(";")[0].strip()
        elif line.startswith("DE"):
            de_lines.append(line[5:].strip())
        elif line.startswith("SQ"):
            in_sequence = True
        elif in_sequence:
            sequence_parts.append("".join(line.strip().split()))
    sequence = "".join(sequence_parts)
    return {
        "accession": accession,
        "entry_name": entry_name,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "ecs": parse_ecs_from_de(de_lines),
    }


def iter_swissprot_records(dat_gz_path: Path) -> Iterable[dict]:
    with gzip.open(dat_gz_path, "rt", encoding="utf-8", errors="ignore") as handle:
        current = []
        for line in handle:
            if line.startswith("//"):
                if current:
                    yield parse_record(current)
                current = []
            else:
                current.append(line.rstrip("\n"))


def extract_dat_gz(tar_path: Path) -> Path:
    out_path = DATA_RAW / f"{tar_path.stem}.dat.gz"
    if out_path.exists():
        return out_path
    with tarfile.open(tar_path, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith(".dat.gz"))
        with archive.extractfile(member) as src, out_path.open("wb") as dst:
            dst.write(src.read())
    return out_path


def download_if_needed() -> Dict[str, Path]:
    targets = {
        "2018_02": (
            DATA_RAW / "uniprot_sprot-only2018_02.tar.gz",
            "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2018_02/knowledgebase/uniprot_sprot-only2018_02.tar.gz",
        ),
        "2023_01": (
            DATA_RAW / "uniprot_sprot-only2023_01.tar.gz",
            "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2023_01/knowledgebase/uniprot_sprot-only2023_01.tar.gz",
        ),
    }
    for release, (path, url) in targets.items():
        if not path.exists():
            log_stage("data_prep", f"downloading Swiss-Prot release {release} from {url}")
            subprocess.check_call(["curl", "-L", "-o", str(path), url])
    return {release: extract_dat_gz(path) for release, (path, _) in targets.items()}


def clean_release(dat_gz_path: Path, release_name: str) -> pd.DataFrame:
    out_path = DATA_PROCESSED / f"swissprot_ec_{release_name}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)

    rows = []
    seen_pairs = set()
    for record in iter_swissprot_records(dat_gz_path):
        if not record["accession"] or not record["sequence"]:
            continue
        if not (MIN_SEQ_LEN <= record["sequence_length"] <= MAX_SEQ_LEN):
            continue
        ecs = sorted({ec for ec in record["ecs"] if "-" not in ec and "n" not in ec.lower()})
        if len(ecs) != 1:
            continue
        ec4 = ecs[0]
        key = (record["sequence"], ec4)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        rows.append(
            {
                "accession": record["accession"],
                "entry_name": record["entry_name"],
                "sequence": record["sequence"],
                "sequence_length": record["sequence_length"],
                "ec4": ec4,
                "ec1": ec_prefix(ec4, 1),
                "ec2": ec_prefix(ec4, 2),
                "ec3": ec_prefix(ec4, 3),
                "release": release_name,
            }
        )
    df = pd.DataFrame(rows).drop_duplicates(subset=["accession"]).reset_index(drop=True)
    save_parquet(df, out_path)
    return df


def select_parents(df18: pd.DataFrame) -> pd.DataFrame:
    child_support = df18.groupby(["ec3", "ec4"]).size().rename("count").reset_index()
    eligible_children = child_support[child_support["count"] >= MIN_CHILD_SUPPORT].copy()
    parent_support = (
        eligible_children.groupby("ec3")
        .agg(num_children=("ec4", "nunique"), total_support=("count", "sum"))
        .reset_index()
    )
    parent_support = parent_support[
        (parent_support["num_children"] >= MIN_PARENT_CHILDREN) & (parent_support["total_support"] >= MIN_PARENT_TOTAL_SUPPORT)
    ]
    parent_support = parent_support.sort_values(["total_support", "num_children", "ec3"], ascending=[False, False, True]).head(MAX_PARENTS)
    parent_support = parent_support.reset_index(drop=True)
    return parent_support


def stratified_parent_split(parent_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for parent, parent_slice in parent_df.groupby("ec3"):
        for child, child_slice in parent_slice.groupby("ec4"):
            indices = rng.permutation(child_slice.index.to_numpy())
            n = len(indices)
            n_train = int(math.floor(n * 0.70))
            n_cal = int(math.floor(n * 0.15))
            n_val = n - n_train - n_cal
            if n_cal == 0:
                n_cal = 1
                n_train -= 1
            if n_val == 0:
                n_val = 1
                n_train -= 1
            if n_train <= 0:
                raise ValueError(f"invalid split sizes for {parent} {child} with n={n}")
            split_map = {
                "train_core": indices[:n_train],
                "cal_known": indices[n_train : n_train + n_cal],
                "val_select": indices[n_train + n_cal :],
            }
            for split_name, split_indices in split_map.items():
                for idx in split_indices:
                    row = parent_df.loc[idx]
                    rows.append(
                        {
                            "row_index": int(idx),
                            "accession": row["accession"],
                            "ec3": parent,
                            "ec4": child,
                            "seed": seed,
                            "split": split_name,
                        }
                    )
    return pd.DataFrame(rows).sort_values(["ec3", "ec4", "split", "accession"]).reset_index(drop=True)


def build_manifests(df18: pd.DataFrame, df23: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    parent_support = select_parents(df18)
    selected_parents = parent_support["ec3"].tolist()
    retained_children = (
        df18[df18["ec3"].isin(selected_parents)]
        .groupby(["ec3", "ec4"])
        .size()
        .rename("count")
        .reset_index()
    )
    retained_children = retained_children[retained_children["count"] >= MIN_CHILD_SUPPORT][["ec3", "ec4"]]
    df18_sel = df18.merge(retained_children, on=["ec3", "ec4"], how="inner").reset_index(drop=True)
    df23_sel = df23[df23["ec3"].isin(selected_parents)].copy().reset_index(drop=True)

    known_children = set(df18_sel["ec4"])
    df23_sel["temporal_known_child"] = df23_sel["ec4"].isin(known_children).astype(int)
    df23_sel["temporal_future_child"] = 1 - df23_sel["temporal_known_child"]
    df23_sel["deepest_correct_prefix_depth"] = np.where(df23_sel["temporal_known_child"] == 1, 4, 3)
    df23_sel["parent_id"] = df23_sel["ec3"]
    df23_sel["eval_split"] = np.where(df23_sel["temporal_future_child"] == 1, "temporal_future_child", "temporal_known_child")
    save_parquet(df23_sel, DATA_PROCESSED / "temporal_eval_selected_2023_01.parquet")

    seed_splits = {}
    for seed in SEEDS:
        split_df = stratified_parent_split(df18_sel, seed)
        seed_splits[seed] = split_df
        save_parquet(split_df, DATA_SPLITS / f"splits_seed_{seed}.parquet")
        split_summary = (
            split_df.groupby(["ec3", "ec4", "split"]).size().rename("count").reset_index().sort_values(["ec3", "ec4", "split"])
        )
        write_json(
            DATA_SPLITS / f"seed_{seed}.json",
            {
                "seed": seed,
                "split_ratios": {"train_core": 0.70, "cal_known": 0.15, "val_select": 0.15},
                "rows": split_summary.to_dict(orient="records"),
            },
        )

    parent_diag = []
    for parent in selected_parents:
        train_parent = df18_sel[df18_sel["ec3"] == parent]
        future_parent = df23_sel[(df23_sel["ec3"] == parent) & (df23_sel["temporal_future_child"] == 1)]
        parent_diag.append(
            {
                "parent_id": parent,
                "train_children": int(train_parent["ec4"].nunique()),
                "train_sequences": int(len(train_parent)),
                "future_children": int(future_parent["ec4"].nunique()),
                "future_child_sequences": int(len(future_parent)),
            }
        )
    diag_df = pd.DataFrame(parent_diag).sort_values("train_sequences", ascending=False).reset_index(drop=True)
    diag_df.to_csv(EXP / "data_prep" / "parent_diagnostics.csv", index=False)

    split_sizes = []
    for seed, split_df in seed_splits.items():
        summary = split_df.groupby(["split"]).size().rename("count").reset_index()
        for row in summary.itertuples(index=False):
            split_sizes.append({"seed": seed, "split": row.split, "count": int(row.count)})

    write_json(
        EXP / "data_prep" / "results.json",
        {
            "selected_parent_count": int(len(selected_parents)),
            "selected_parents": selected_parents,
            "parents_with_future_events": int((diag_df["future_child_sequences"] > 0).sum()),
            "train_2018_rows": int(len(df18)),
            "test_2023_rows": int(len(df23)),
            "selected_2018_rows": int(len(df18_sel)),
            "selected_2023_rows": int(len(df23_sel)),
            "split_sizes": split_sizes,
        },
    )
    return df18_sel, df23_sel, seed_splits


def get_sequences_for_embedding(df18_sel: pd.DataFrame, df23_sel: pd.DataFrame) -> pd.DataFrame:
    seq_df = (
        pd.concat(
            [
                df18_sel[["accession", "sequence", "ec3", "ec4", "release"]],
                df23_sel[["accession", "sequence", "ec3", "ec4", "release"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=["accession"])
        .reset_index(drop=True)
    )
    save_parquet(seq_df, DATA_PROCESSED / "selected_sequences.parquet")
    return seq_df


def embed_sequences(seq_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    emb_path = DATA_EMBEDDINGS / "esm2_t30_150M_selected_float16.npy"
    map_path = DATA_EMBEDDINGS / "accession_to_index.json"
    seq_path = DATA_EMBEDDINGS / "cached_sequences.parquet"

    cached_arr = np.empty((0, 640), dtype=np.float16)
    cached_map: Dict[str, int] = {}
    cached_seq_df = pd.DataFrame(columns=["accession", "sequence"])
    if emb_path.exists() and map_path.exists():
        cached_arr = np.load(emb_path)
        cached_map = json.loads(map_path.read_text())
    if seq_path.exists():
        cached_seq_df = pd.read_parquet(seq_path)[["accession", "sequence"]]

    requested = seq_df[["accession", "sequence"]].drop_duplicates(subset=["accession"]).reset_index(drop=True)
    cached_accessions = set(cached_map)
    missing_df = requested[~requested["accession"].isin(cached_accessions)].reset_index(drop=True)
    if missing_df.empty and set(requested["accession"]).issubset(cached_accessions):
        return cached_arr, cached_map

    import esm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    new_embeddings: List[np.ndarray] = []
    new_accessions: List[str] = []
    rows = sorted(missing_df.itertuples(index=False), key=lambda row: len(row.sequence))
    batch: List[Tuple[str, str]] = []
    batch_tokens = 0

    def flush(batch_rows: List[Tuple[str, str]]) -> None:
        nonlocal new_embeddings, new_accessions
        if not batch_rows:
            return
        _, _, tokens = batch_converter(batch_rows)
        tokens = tokens.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                outputs = model(tokens, repr_layers=[30], return_contacts=False)
            reps = outputs["representations"][30]
        token_mask = (tokens != alphabet.padding_idx) & (tokens != alphabet.cls_idx) & (tokens != alphabet.eos_idx)
        for row_idx, (accession, _) in enumerate(batch_rows):
            mask = token_mask[row_idx]
            rep = reps[row_idx][mask].float().mean(dim=0).cpu().numpy().astype(np.float16)
            new_embeddings.append(rep)
            new_accessions.append(accession)

    for row in rows:
        seq_tokens = len(row.sequence) + 2
        if batch and (len(batch) >= EMBED_BATCH_SIZE or batch_tokens + seq_tokens > EMBED_MAX_TOKENS):
            flush(batch)
            batch = []
            batch_tokens = 0
        batch.append((row.accession, row.sequence))
        batch_tokens += seq_tokens
    flush(batch)

    if new_embeddings:
        new_arr = np.vstack(new_embeddings)
        offset = len(cached_arr)
        merged_arr = np.concatenate([cached_arr, new_arr], axis=0) if len(cached_arr) else new_arr
        merged_map = dict(cached_map)
        for idx, accession in enumerate(new_accessions):
            merged_map[accession] = offset + idx
        merged_seq_df = pd.concat([cached_seq_df, missing_df[["accession", "sequence"]]], ignore_index=True).drop_duplicates("accession")
        np.save(emb_path, merged_arr)
        map_path.write_text(json.dumps(merged_map, indent=2, sort_keys=True))
        save_parquet(merged_seq_df, seq_path)
        return merged_arr, merged_map

    return cached_arr, cached_map


def make_embedding_lookup(seq_df: pd.DataFrame, embeddings: np.ndarray, accession_map: Dict[str, int]) -> Dict[str, np.ndarray]:
    return {
        row.accession: embeddings[accession_map[row.accession]].astype(np.float32)
        for row in seq_df.itertuples(index=False)
        if row.accession in accession_map
    }


def cosine_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
    Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(min=1e-8)
    return Xn @ Yn.T


@dataclass
class ParentModel:
    parent_id: str
    child_order: List[str]
    prototypes: np.ndarray
    scaler: StandardScaler
    classifier: LogisticRegression
    train_embeddings: np.ndarray
    train_labels: np.ndarray
    include_retrieval: bool
    nn_model: Optional[NearestNeighbors]
    fit_warnings: List[str]


@dataclass
class FlatModel:
    classes_: np.ndarray
    scaler: StandardScaler
    classifier: LogisticRegression
    fit_warnings: List[str]


def build_features(
    X: np.ndarray,
    train_X: np.ndarray,
    train_labels: np.ndarray,
    child_order: List[str],
    prototypes: np.ndarray,
    include_retrieval: bool,
    nn_model: Optional[NearestNeighbors],
) -> np.ndarray:
    sims = cosine_matrix(X, prototypes)
    if sims.shape[1] > 1:
        sorted_sims = np.sort(sims, axis=1)
        margin = (sorted_sims[:, -1] - sorted_sims[:, -2])[:, None]
    else:
        margin = np.ones((len(X), 1), dtype=np.float32)

    if include_retrieval:
        k_vote = min(10, len(train_X))
        k_density = min(5, len(train_X))
        nn = nn_model or NearestNeighbors(n_neighbors=k_vote, metric="cosine").fit(train_X)
        dists, indices = nn.kneighbors(X, return_distance=True)
        neighbor_labels = train_labels[indices]
        vote_share = np.zeros((len(X), len(child_order)), dtype=np.float32)
        for child_idx, child in enumerate(child_order):
            vote_share[:, child_idx] = (neighbor_labels == child).mean(axis=1)
        density = (1.0 - dists[:, :k_density]).mean(axis=1, keepdims=True).astype(np.float32)
        return np.hstack([sims, vote_share, margin, density]).astype(np.float32)
    return np.hstack([sims, margin]).astype(np.float32)


def fit_logistic(X: np.ndarray, y: np.ndarray, class_weight: Optional[str]) -> Tuple[LogisticRegression, StandardScaler, List[str]]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=500,
        class_weight=class_weight,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X_scaled, y)
    warning_messages = [str(item.message) for item in caught]
    return clf, scaler, warning_messages


def fit_parent_models(train_df: pd.DataFrame, emb_lookup: Dict[str, np.ndarray], include_retrieval: bool) -> Dict[str, ParentModel]:
    models: Dict[str, ParentModel] = {}
    for parent, parent_df in train_df.groupby("ec3"):
        child_order = sorted(parent_df["ec4"].unique())
        X = np.vstack([emb_lookup[acc] for acc in parent_df["accession"]])
        y = parent_df["ec4"].to_numpy()
        prototypes = np.vstack([X[y == child].mean(axis=0) for child in child_order]).astype(np.float32)
        nn_model = None
        if include_retrieval:
            nn_model = NearestNeighbors(n_neighbors=min(10, len(X)), metric="cosine")
            nn_model.fit(X)
        features = build_features(X, X, y, child_order, prototypes, include_retrieval, nn_model)
        clf, scaler, fit_warnings = fit_logistic(features, y, class_weight="balanced")
        models[parent] = ParentModel(
            parent_id=parent,
            child_order=child_order,
            prototypes=prototypes,
            scaler=scaler,
            classifier=clf,
            train_embeddings=X,
            train_labels=y,
            include_retrieval=include_retrieval,
            nn_model=nn_model,
            fit_warnings=fit_warnings,
        )
    return models


def score_parent_examples(df: pd.DataFrame, models: Dict[str, ParentModel], emb_lookup: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for parent, parent_df in df.groupby("ec3"):
        if parent not in models:
            continue
        model = models[parent]
        X = np.vstack([emb_lookup[acc] for acc in parent_df["accession"]])
        features = build_features(
            X,
            model.train_embeddings,
            model.train_labels,
            model.child_order,
            model.prototypes,
            model.include_retrieval,
            model.nn_model,
        )
        X_scaled = model.scaler.transform(features)
        probs = model.classifier.predict_proba(X_scaled)
        top_indices = probs.argmax(axis=1)
        class_order = model.classifier.classes_
        top_child = class_order[top_indices]
        sorted_probs = np.sort(probs, axis=1)
        top_prob = sorted_probs[:, -1]
        second_prob = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros(len(probs), dtype=np.float32)
        hsc_conf = top_prob / np.clip(top_prob + second_prob, 1e-8, None)
        logits = model.classifier.decision_function(X_scaled)
        if logits.ndim == 1:
            logits = np.vstack([-logits, logits]).T
        energy = -np.log(np.exp(logits).sum(axis=1))
        for row_idx, row in enumerate(parent_df.itertuples(index=False)):
            prob_map = {cls: float(prob) for cls, prob in zip(class_order, probs[row_idx])}
            rows.append(
                {
                    "accession": row.accession,
                    "true_ec4": row.ec4,
                    "parent_id": parent,
                    "top_child": top_child[row_idx],
                    "top_prob": float(top_prob[row_idx]),
                    "second_prob": float(second_prob[row_idx]),
                    "hsc_conf": float(hsc_conf[row_idx]),
                    "energy": float(energy[row_idx]),
                    "prob_by_child": json.dumps(prob_map, sort_keys=True),
                    "true_prob": float(prob_map.get(row.ec4, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def fit_flat_model(train_df: pd.DataFrame, emb_lookup: Dict[str, np.ndarray]) -> FlatModel:
    X = np.vstack([emb_lookup[acc] for acc in train_df["accession"]])
    y = train_df["ec4"].to_numpy()
    clf, scaler, fit_warnings = fit_logistic(X, y, class_weight=None)
    return FlatModel(classes_=clf.classes_, scaler=scaler, classifier=clf, fit_warnings=fit_warnings)


def score_flat_examples(df: pd.DataFrame, model: FlatModel, emb_lookup: Dict[str, np.ndarray]) -> pd.DataFrame:
    X = np.vstack([emb_lookup[acc] for acc in df["accession"]])
    probs = model.classifier.predict_proba(model.scaler.transform(X))
    top_indices = probs.argmax(axis=1)
    top_child = model.classifier.classes_[top_indices]
    second_prob = np.sort(probs, axis=1)[:, -2] if probs.shape[1] > 1 else np.zeros(len(probs))
    rows = []
    for idx, row in enumerate(df.itertuples(index=False)):
        rows.append(
            {
                "accession": row.accession,
                "true_ec4": row.ec4,
                "parent_id": row.ec3,
                "top_child": top_child[idx],
                "top_prob": float(probs[idx, top_indices[idx]]),
                "second_prob": float(second_prob[idx]),
                "probabilities": probs[idx].tolist(),
            }
        )
    return pd.DataFrame(rows)


def deepest_supported_prefix(labels: Sequence[str]) -> str:
    if not labels:
        raise ValueError("labels must not be empty")
    depth = 4
    while depth > 0:
        prefixes = {ec_prefix(label, depth) for label in labels}
        if len(prefixes) == 1:
            return prefixes.pop()
        depth -= 1
    return ec_prefix(labels[0], 1)


def flat_return_prefix(probabilities: Sequence[float], classes: Sequence[str], threshold: float) -> Tuple[str, int]:
    order = np.argsort(np.asarray(probabilities))[::-1]
    top_class = classes[order[0]]
    top_prob = float(probabilities[order[0]])
    if top_prob >= threshold:
        return top_class, 4
    selected = [classes[idx] for idx in order[: min(3, len(order))]]
    prefix = deepest_supported_prefix(selected)
    return prefix, ec_depth(prefix)


def masked_child_manifest(train_df: pd.DataFrame, cal_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rotation = {13: 0, 29: 1, 47: 2}[seed]
    rows = []
    for parent, parent_train in train_df.groupby("ec3"):
        train_counts = parent_train["ec4"].value_counts()
        holdout_counts = pd.concat(
            [
                cal_df[cal_df["ec3"] == parent][["ec4"]],
                val_df[val_df["ec3"] == parent][["ec4"]],
            ],
            ignore_index=True,
        )["ec4"].value_counts()
        eligible = [
            child
            for child in sorted(train_counts.index)
            if train_counts[child] >= MIN_CHILD_SUPPORT and holdout_counts.get(child, 0) >= 10
        ]
        if len(eligible) < 4:
            continue
        child = eligible[rotation % len(eligible)]
        rows.append(
            {
                "seed": seed,
                "parent_id": parent,
                "masked_child": child,
                "eligible_children": json.dumps(eligible),
            }
        )
    return pd.DataFrame(rows)


def fit_openmax_params(cal_scores: pd.DataFrame) -> Dict[str, dict]:
    params = {}
    for parent, parent_df in cal_scores.groupby("parent_id"):
        if parent_df["true_ec4"].nunique() < 3:
            continue
        if len(parent_df) < 15:
            continue
        dists = 1.0 - parent_df["true_prob"].to_numpy()
        shape, loc, scale = stats.weibull_min.fit(dists, floc=0)
        params[parent] = {"shape": float(shape), "loc": float(loc), "scale": float(scale)}
    return params


def openmax_score(top_prob: float, params: Optional[dict]) -> float:
    if params is None:
        return 1.0
    dist = max(0.0, 1.0 - top_prob)
    cdf = stats.weibull_min.cdf(dist, params["shape"], loc=params["loc"], scale=params["scale"])
    return float(cdf)


def choose_thresholds(score_df: pd.DataFrame, score_col: str) -> Dict[float, float]:
    scores = score_df[score_col].to_numpy(dtype=float)
    return {target: quantile_threshold(scores, target) for target in COVERAGE_TARGETS}


def choose_validation_thresholds(score_df: pd.DataFrame, score_col: str, *, accept_high: bool) -> Dict[float, float]:
    scores = score_df[score_col].to_numpy(dtype=float)
    work_scores = scores if accept_high else -scores
    thresholds = {}
    for coverage_target in COVERAGE_TARGETS:
        threshold = quantile_threshold(work_scores, coverage_target)
        thresholds[coverage_target] = float(threshold if accept_high else -threshold)
    return thresholds


def apply_fixed_thresholds(
    eval_scores: pd.DataFrame,
    score_col: str,
    thresholds: Dict[float, float],
    *,
    accept_high: bool,
    method: str,
    seed: int,
    reject_prefix_col: str = "parent_id",
    open_score_col: Optional[str] = None,
    log_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[float, float], List[dict]]:
    prediction_rows = []
    diagnostics = []
    for eval_split, split_df in eval_scores.groupby("eval_split"):
        base_scores = split_df[score_col].to_numpy(dtype=float)
        for coverage_target in COVERAGE_TARGETS:
            threshold = thresholds[coverage_target]
            accept = base_scores >= threshold if accept_high else base_scores <= threshold
            temp = split_df.copy()
            temp["method"] = method
            temp["seed"] = seed
            temp["coverage_target"] = coverage_target
            temp["returned_prefix"] = np.where(accept, temp["top_child"], temp[reject_prefix_col])
            temp["returned_depth"] = np.where(accept, 4, temp[reject_prefix_col].map(ec_depth))
            temp["reject_prefix"] = temp[reject_prefix_col]
            temp["reject_depth"] = temp[reject_prefix_col].map(ec_depth)
            temp["top_child_score"] = temp["top_prob"]
            temp["selection_score"] = temp[score_col]
            temp["open_child_under_parent"] = (~accept).astype(int)
            if open_score_col:
                temp["open_score"] = temp[open_score_col]
            prediction_rows.append(temp)
            diagnostics.append(
                {
                    "eval_split": eval_split,
                    "coverage_target": float(coverage_target),
                    "threshold": float(threshold),
                    "realized_coverage": float(accept.mean()),
                    "n_examples": int(len(split_df)),
                    "threshold_source": "val_select",
                }
            )
    if log_prefix:
        details = ", ".join(
            f"{item['eval_split']}@{item['coverage_target']:.2f}:thr={item['threshold']:.4f},cov={item['realized_coverage']:.3f},n={item['n_examples']}"
            for item in diagnostics
        )
        log_method(method, seed, f"{log_prefix} {details}")
    return pd.concat(prediction_rows, ignore_index=True), thresholds, diagnostics


def evaluate_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, seed, eval_split, coverage_target), group in pred_df.groupby(["method", "seed", "eval_split", "coverage_target"]):
        metric_arrays = compute_metric_arrays(group)
        records.append(
            {
                "method": method,
                "seed": seed,
                "eval_split": eval_split,
                "coverage_target": float(coverage_target),
                "realized_coverage": float(metric_arrays["realized_coverage"].mean()),
                "catastrophic_overspecialization_rate": float(metric_arrays["catastrophic_overspecialization_rate"].mean()),
                "correct_safe_prefix_rate": float(metric_arrays["correct_safe_prefix_rate"].mean()),
                "hierarchical_loss": float(metric_arrays["hierarchical_loss"].mean()),
                "selective_ec4_f1": float(metric_arrays["selective_ec4_f1"][0]),
                "mean_returned_depth": float(metric_arrays["mean_returned_depth"].mean()),
                "n_examples": int(len(group)),
                "open_child_auroc": None,
            }
        )
    return pd.DataFrame(records)


def attach_open_auc(metrics_df: pd.DataFrame, key: Tuple[str, int, str, float], auc: Optional[float]) -> None:
    method, seed, split, coverage = key
    mask = (
        (metrics_df["method"] == method)
        & (metrics_df["seed"] == seed)
        & (metrics_df["eval_split"] == split)
        & (metrics_df["coverage_target"] == coverage)
    )
    metrics_df.loc[mask, "open_child_auroc"] = auc


def compute_open_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> Optional[float]:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    y_true = np.concatenate([np.ones(len(pos_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)])
    y_score = np.concatenate([pos_scores, neg_scores])
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def paired_pointwise_stats(
    merged: pd.DataFrame,
    *,
    metric_name: str,
    metric_fn,
    method_col: str,
    baseline_col: str,
) -> dict:
    if merged.empty:
        return {"metric": metric_name, "paired_bootstrap_ci_over_proteins": None, "paired_permutation_pvalue_over_seed_differences": None}
    method_values = metric_fn(merged, method_col)
    baseline_values = metric_fn(merged, baseline_col)
    diff = baseline_values - method_values
    rng = np.random.default_rng(0)
    ci_low, ci_high = bootstrap_ci(diff, rng)
    seed_diffs = []
    for _, seed_df in merged.groupby("seed"):
        seed_method = metric_fn(seed_df, method_col).mean()
        seed_base = metric_fn(seed_df, baseline_col).mean()
        seed_diffs.append(float(seed_base - seed_method))
    return {
        "metric": metric_name,
        "paired_bootstrap_ci_over_proteins": {"low": ci_low, "high": ci_high},
        "paired_permutation_pvalue_over_seed_differences": exact_sign_flip_pvalue(np.asarray(seed_diffs, dtype=float)),
    }


def bootstrap_auc_difference(labels: np.ndarray, method_scores: np.ndarray, baseline_scores: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return (None, None)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(BOOTSTRAP_ITERS):
        indices = rng.integers(0, len(labels), size=len(labels))
        y = labels[indices]
        if len(np.unique(y)) < 2:
            continue
        diffs.append(float(roc_auc_score(y, method_scores[indices]) - roc_auc_score(y, baseline_scores[indices])))
    if not diffs:
        return (None, None)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def paired_auc_stats(
    merged_scores: pd.DataFrame,
    *,
    method_name: str,
    baseline_name: str,
) -> dict:
    if merged_scores.empty:
        return {"metric": "open_child_auroc", "baseline_name": baseline_name, "paired_bootstrap_ci_over_proteins": None, "paired_permutation_pvalue_over_seed_differences": None}
    labels = merged_scores["label"].to_numpy(dtype=int)
    method_scores = merged_scores["open_score_mcc"].to_numpy(dtype=float)
    baseline_scores = merged_scores["open_score_baseline"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_auc_difference(labels, method_scores, baseline_scores)
    seed_diffs = []
    for _, seed_df in merged_scores.groupby("seed"):
        y = seed_df["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        seed_diffs.append(
            float(
                roc_auc_score(y, seed_df["open_score_mcc"].to_numpy(dtype=float))
                - roc_auc_score(y, seed_df["open_score_baseline"].to_numpy(dtype=float))
            )
        )
    return {
        "metric": "open_child_auroc",
        "baseline_name": baseline_name,
        "paired_bootstrap_ci_over_proteins": None if ci_low is None else {"low": ci_low, "high": ci_high},
        "paired_permutation_pvalue_over_seed_differences": exact_sign_flip_pvalue(np.asarray(seed_diffs, dtype=float)) if seed_diffs else None,
    }


def save_method_artifacts(
    method: str,
    seed: int,
    thresholds: Dict[float, float],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict,
) -> None:
    save_parquet(predictions, EXP / method / f"seed_{seed}" / "predictions.parquet")
    threshold_payload = {}
    for key, value in thresholds.items():
        if isinstance(key, tuple):
            threshold_payload[f"{key[0]}__{key[1]:.2f}"] = value
        else:
            threshold_payload[str(key)] = value
    write_json(
        EXP / method / f"seed_{seed}" / "results.json",
        {
            "experiment": method,
            "seed": seed,
            "thresholds": threshold_payload,
            "metrics": metrics.to_dict(orient="records"),
            "config": config,
        },
    )


def save_auxiliary_eval_artifacts(
    method: str,
    seed: int,
    tag: str,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    base = EXP / method / f"seed_{seed}"
    save_parquet(predictions, base / f"{tag}_predictions.parquet")
    write_json(
        base / f"{tag}_results.json",
        {
            "experiment": method,
            "seed": seed,
            "tag": tag,
            "metrics": metrics.to_dict(orient="records"),
        },
    )


def run_forced(scores: pd.DataFrame, seed: int, eval_split: str, method: str = "forced_hierarchical") -> pd.DataFrame:
    rows = []
    for coverage_target in COVERAGE_TARGETS:
        temp = scores.copy()
        temp["method"] = method
        temp["seed"] = seed
        temp["eval_split"] = eval_split
        temp["coverage_target"] = coverage_target
        temp["returned_prefix"] = temp["top_child"]
        temp["returned_depth"] = 4
        temp["reject_prefix"] = temp["parent_id"]
        temp["reject_depth"] = temp["parent_id"].map(ec_depth)
        temp["top_child_score"] = temp["top_prob"]
        temp["open_child_under_parent"] = 0
        temp["selection_score"] = temp["top_prob"]
        rows.append(
            temp[
                [
                    "method",
                    "seed",
                    "eval_split",
                    "coverage_target",
                    "accession",
                    "true_ec4",
                    "parent_id",
                    "top_child",
                    "returned_prefix",
                    "returned_depth",
                    "reject_prefix",
                    "reject_depth",
                    "top_child_score",
                    "selection_score",
                    "open_child_under_parent",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def run_threshold_method(
    method: str,
    seed: int,
    val_scores: pd.DataFrame,
    eval_scores: pd.DataFrame,
    score_col: str,
    *,
    accept_high: bool,
    open_score_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[float, float]]:
    thresholds = choose_validation_thresholds(val_scores, score_col, accept_high=accept_high)
    predictions, thresholds, _ = apply_fixed_thresholds(
        eval_scores,
        score_col,
        thresholds,
        accept_high=accept_high,
        method=method,
        seed=seed,
        reject_prefix_col="parent_id",
        open_score_col=open_score_col,
        log_prefix="coverage-matched thresholds",
    )
    metrics = evaluate_predictions(predictions)
    return predictions, metrics, thresholds


def score_masked_event(
    event: pd.Series,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    val_df: pd.DataFrame,
    emb_lookup: Dict[str, np.ndarray],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[dict], Optional[ParentModel], Optional[ParentModel]]:
    parent = event["parent_id"]
    masked_child = event["masked_child"]
    masked_train = train_df[(train_df["ec3"] == parent) & (train_df["ec4"] != masked_child)].copy()
    if masked_train["ec4"].nunique() < 3:
        return None, None, [], None, None
    masked_examples = pd.concat(
        [
            cal_df[(cal_df["ec3"] == parent) & (cal_df["ec4"] == masked_child)],
            val_df[(val_df["ec3"] == parent) & (val_df["ec4"] == masked_child)],
        ],
        ignore_index=True,
    )
    if masked_examples.empty:
        return None, None, [], None, None

    retrieval_model = fit_parent_models(masked_train, emb_lookup, include_retrieval=True)[parent]
    no_retrieval_model = fit_parent_models(masked_train, emb_lookup, include_retrieval=False)[parent]
    masked_scores = score_parent_examples(masked_examples, {parent: retrieval_model}, emb_lookup)
    masked_scores["masked_child"] = masked_child
    masked_scores["eval_split"] = "masked_child_surrogate"
    masked_scores["seed"] = event["seed"]
    no_retrieval_scores = score_parent_examples(masked_examples, {parent: no_retrieval_model}, emb_lookup)
    no_retrieval_scores["masked_child"] = masked_child
    no_retrieval_scores["eval_split"] = "masked_child_surrogate_no_retrieval"
    no_retrieval_scores["seed"] = event["seed"]

    audit_rows = []
    prototypes_ok = masked_child not in retrieval_model.child_order
    train_labels_ok = not np.any(retrieval_model.train_labels == masked_child)
    neighbor_ok = True
    density_ok = True
    if retrieval_model.nn_model is not None:
        X = np.vstack([emb_lookup[acc] for acc in masked_examples["accession"]])
        _, indices = retrieval_model.nn_model.kneighbors(X, return_distance=True)
        neighbor_labels = retrieval_model.train_labels[indices]
        neighbor_ok = bool(np.all(neighbor_labels != masked_child))
        density_ok = neighbor_ok
    mean_abs_diff = float(np.mean(np.abs(masked_scores["top_prob"].to_numpy() - no_retrieval_scores["top_prob"].to_numpy())))
    for family, passed in [
        ("prototype_similarity", prototypes_ok and train_labels_ok),
        ("neighbor_vote_share", neighbor_ok and train_labels_ok),
        ("margin", prototypes_ok),
        ("local_density", density_ok and train_labels_ok),
    ]:
        audit_rows.append(
            {
                "seed": int(event["seed"]),
                "parent_id": parent,
                "masked_child": masked_child,
                "feature_family": family,
                "rebuilt_after_mask": 1,
                "uses_removed_support": int(not train_labels_ok),
                "passed_audit": int(passed),
                "mean_abs_prob_diff_vs_no_retrieval": mean_abs_diff,
                "masked_examples": int(len(masked_examples)),
            }
        )
    return masked_scores, no_retrieval_scores, audit_rows, retrieval_model, no_retrieval_model


def parent_level_stats(train_df: pd.DataFrame, emb_lookup: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for parent, parent_df in train_df.groupby("ec3"):
        counts = parent_df["ec4"].value_counts().sort_index()
        probs = counts / counts.sum()
        entropy = float(-(probs * np.log(probs + 1e-8)).sum())
        X = np.vstack([emb_lookup[acc] for acc in parent_df["accession"]])
        y = parent_df["ec4"].to_numpy()
        child_order = sorted(parent_df["ec4"].unique())
        prototypes = np.vstack([X[y == child].mean(axis=0) for child in child_order]).astype(np.float32)
        sims = cosine_matrix(X, prototypes)
        compactness = float(np.mean([sims[y == child, idx].mean() for idx, child in enumerate(child_order)]))
        proto_cos = cosine_matrix(prototypes, prototypes)
        np.fill_diagonal(proto_cos, -1.0)
        min_gap = float(1.0 - proto_cos.max())
        nn = NearestNeighbors(n_neighbors=min(6, len(X)), metric="cosine").fit(X)
        _, indices = nn.kneighbors(X)
        overlaps = []
        for row_idx, child in enumerate(y):
            neighbors = y[indices[row_idx, 1:]]
            overlaps.append(float(np.mean(neighbors != child)))
        rows.append(
            {
                "parent_id": parent,
                "sibling_support_entropy": entropy,
                "prototype_compactness": compactness,
                "minimum_inter_child_cosine_gap": min_gap,
                "neighbor_overlap_rate": float(np.mean(overlaps)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame, prediction_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    rows = []
    metric_cols = [
        "catastrophic_overspecialization_rate",
        "correct_safe_prefix_rate",
        "hierarchical_loss",
        "selective_ec4_f1",
        "realized_coverage",
        "mean_returned_depth",
        "open_child_auroc",
    ]
    for keys, group in metrics_df.groupby(["method", "eval_split", "coverage_target"]):
        row = {"method": keys[0], "eval_split": keys[1], "coverage_target": float(keys[2])}
        for metric in metric_cols:
            summary = mean_std_summary(group[metric])
            row[f"{metric}_mean"] = summary["mean"]
            row[f"{metric}_std"] = summary["std"]
            row[f"{metric}_n"] = summary["n"]
            values = group[metric].dropna().to_numpy(dtype=float)
            ci_low, ci_high = confidence_interval(values, np.random.default_rng(0))
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def load_saved_artifacts() -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    metrics_rows: List[dict] = []
    prediction_frames: List[pd.DataFrame] = []
    for method in ALL_METHODS:
        for seed in SEEDS:
            result_path = EXP / method / f"seed_{seed}" / "results.json"
            prediction_path = EXP / method / f"seed_{seed}" / "predictions.parquet"
            aux_result_path = EXP / method / f"seed_{seed}" / "masked_child_results.json"
            aux_prediction_path = EXP / method / f"seed_{seed}" / "masked_child_predictions.parquet"
            if result_path.exists():
                payload = json.loads(result_path.read_text())
                metrics_rows.extend(payload.get("metrics", []))
            if aux_result_path.exists():
                payload = json.loads(aux_result_path.read_text())
                metrics_rows.extend(payload.get("metrics", []))
            if prediction_path.exists():
                prediction_frames.append(pd.read_parquet(prediction_path))
            if aux_prediction_path.exists():
                prediction_frames.append(pd.read_parquet(aux_prediction_path))
    leakage_df = pd.read_csv(EXP / "surrogate_benchmark" / "leakage_audit.csv") if (EXP / "surrogate_benchmark" / "leakage_audit.csv").exists() else pd.DataFrame()
    transfer_path = EXP / "transfer_analysis" / "results.json"
    transfer_df = pd.DataFrame()
    if transfer_path.exists():
        transfer_payload = json.loads(transfer_path.read_text())
        transfer_df = pd.DataFrame(transfer_payload.get("parent_rows", []))
    return pd.DataFrame(metrics_rows), prediction_frames, leakage_df, transfer_df


def exact_match_frame(base_df: pd.DataFrame, coverage_target: float, eval_split_name: str) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame()
    work = base_df.copy()
    work = work.sort_values(["selection_score", "accession"], ascending=[False, True]).reset_index(drop=True)
    k = int(round(float(coverage_target) * len(work)))
    k = max(0, min(len(work), k))
    accept_mask = np.zeros(len(work), dtype=bool)
    if k > 0:
        accept_mask[:k] = True
    work["coverage_target"] = float(coverage_target)
    work["eval_split"] = eval_split_name
    work["returned_prefix"] = np.where(accept_mask, work["top_child"], work["reject_prefix"])
    work["returned_depth"] = np.where(accept_mask, 4, work["reject_depth"])
    work["open_child_under_parent"] = (~accept_mask).astype(int)
    return work


def compute_exact_matched_aggregate(prediction_frames: Sequence[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    temporal_splits = {"temporal_known_child", "temporal_future_child", "masked_child_surrogate"}
    seed_metric_rows: List[dict] = []
    mixed_bases: Dict[Tuple[str, int], List[pd.DataFrame]] = {}

    for frame in prediction_frames:
        if frame.empty or "selection_score" not in frame.columns or "reject_prefix" not in frame.columns:
            continue
        base = frame[frame["eval_split"].isin(temporal_splits)].copy()
        if base.empty:
            continue
        base = base.sort_values(["method", "seed", "eval_split", "accession", "coverage_target"]).drop_duplicates(
            subset=["method", "seed", "eval_split", "accession"],
            keep="first",
        )
        for (method, seed, eval_split), group in base.groupby(["method", "seed", "eval_split"]):
            if eval_split in {"temporal_known_child", "temporal_future_child"}:
                mixed_bases.setdefault((method, seed), []).append(group.copy())
            for coverage_target in COVERAGE_TARGETS:
                matched = exact_match_frame(group, coverage_target, eval_split)
                if matched.empty:
                    continue
                matched["method"] = method
                matched["seed"] = seed
                seed_metric_rows.extend(evaluate_predictions(matched).to_dict(orient="records"))

    for (method, seed), group_list in mixed_bases.items():
        base = pd.concat(group_list, ignore_index=True)
        base = base.sort_values(["accession", "eval_split"]).drop_duplicates(subset=["accession"], keep="first")
        for coverage_target in COVERAGE_TARGETS:
            matched = exact_match_frame(base, coverage_target, "temporal_mixed_posthoc")
            if matched.empty:
                continue
            matched["method"] = method
            matched["seed"] = seed
            seed_metric_rows.extend(evaluate_predictions(matched).to_dict(orient="records"))

    seed_metrics = pd.DataFrame(seed_metric_rows)
    if seed_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    seed_metrics["analysis_protocol"] = "posthoc_exact_matched_coverage"
    return seed_metrics, aggregate_metrics(seed_metrics, [])


def build_transfer_rows(
    seed: int,
    method_name: str,
    comparator_name: str,
    method_preds: pd.DataFrame,
    comparator_preds: pd.DataFrame,
    masked_method_preds: pd.DataFrame,
    masked_comparator_preds: pd.DataFrame,
) -> List[dict]:
    rows = []
    for parent in sorted(set(method_preds["parent_id"])):
        future_method = method_preds[(method_preds["parent_id"] == parent) & (method_preds["eval_split"] == "temporal_future_child") & (method_preds["coverage_target"] == 0.80)]
        future_comp = comparator_preds[(comparator_preds["parent_id"] == parent) & (comparator_preds["eval_split"] == "temporal_future_child") & (comparator_preds["coverage_target"] == 0.80)]
        masked_method = masked_method_preds[(masked_method_preds["parent_id"] == parent) & (masked_method_preds["coverage_target"] == 0.80)]
        masked_comp = masked_comparator_preds[(masked_comparator_preds["parent_id"] == parent) & (masked_comparator_preds["coverage_target"] == 0.80)]
        if future_method.empty or future_comp.empty or masked_method.empty or masked_comp.empty:
            continue

        def metric_block(df: pd.DataFrame) -> Tuple[float, float]:
            catastrophic = float((((df["returned_depth"] == 4) & (df["returned_prefix"] != df["true_ec4"])).mean()))
            safe = float(df.apply(lambda row: is_safe_prefix(row["returned_prefix"], row["true_ec4"]), axis=1).mean())
            return catastrophic, safe

        masked_method_cat, masked_method_safe = metric_block(masked_method)
        masked_comp_cat, masked_comp_safe = metric_block(masked_comp)
        future_method_cat, future_method_safe = metric_block(future_method)
        future_comp_cat, future_comp_safe = metric_block(future_comp)

        masked_neg = method_preds[
            (method_preds["parent_id"] == parent)
            & (method_preds["eval_split"] == "temporal_known_child")
            & (method_preds["coverage_target"] == 0.80)
        ]["open_score"].to_numpy()
        future_neg = method_preds[
            (method_preds["parent_id"] == parent)
            & (method_preds["eval_split"] == "temporal_known_child")
            & (method_preds["coverage_target"] == 0.80)
        ]["open_score"].to_numpy()
        rows.append(
            {
                "seed": seed,
                "parent_id": parent,
                "method_name": method_name,
                "comparator_name": comparator_name,
                "masked_catastrophic_reduction": masked_comp_cat - masked_method_cat,
                "masked_safe_prefix_gain": masked_method_safe - masked_comp_safe,
                "future_catastrophic_reduction": future_comp_cat - future_method_cat,
                "future_safe_prefix_gain": future_method_safe - future_comp_safe,
                "masked_open_child_auroc_gain": compute_open_auc(masked_method["open_score"].to_numpy(), masked_neg) - compute_open_auc(masked_comp["open_score"].to_numpy(), masked_neg)
                if "open_score" in masked_method.columns and "open_score" in masked_comp.columns and len(masked_neg) > 0
                else None,
                "future_open_child_auroc_gain": compute_open_auc(future_method["open_score"].to_numpy(), future_neg) - compute_open_auc(future_comp["open_score"].to_numpy(), future_neg)
                if "open_score" in future_method.columns and "open_score" in future_comp.columns and len(future_neg) > 0
                else None,
                "future_examples": int(len(future_method)),
            }
        )
    return rows


def write_summary_results(
    metadata: dict,
    metrics_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    exact_matched_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    runtime_minutes: float,
    temporal_preds: pd.DataFrame,
    canonical_run: dict,
) -> None:
    aggregate_df.to_csv(EXP / "summary" / "aggregate_metrics.csv", index=False)
    metrics_df.to_csv(EXP / "summary" / "all_metrics.csv", index=False)
    exact_matched_df.to_csv(EXP / "summary" / "exact_matched_aggregate_metrics.csv", index=False)

    if not leakage_df.empty:
        leakage_df.to_csv(FIG_SRC / "figure3_leakage_audit.csv", index=False)
    if not transfer_df.empty:
        transfer_df.to_csv(FIG_SRC / "figure4_transfer_scatter.csv", index=False)

    for method in ALL_METHODS:
        method_rows = aggregate_df[aggregate_df["method"] == method].copy()
        write_json(EXP / method / "results.json", {"experiment": method, "metrics": method_rows.to_dict(orient="records")})

    plt.figure(figsize=(9, 4))
    plt.axis("off")
    plt.text(0.02, 0.8, "Frozen ESM-2 embedding", bbox={"facecolor": "#d9eaf7", "edgecolor": "black"})
    plt.text(0.34, 0.8, "Parent-local scorer", bbox={"facecolor": "#f7ead9", "edgecolor": "black"})
    plt.text(0.58, 0.8, "Known-child conformal gate", bbox={"facecolor": "#dff0d8", "edgecolor": "black"})
    plt.text(0.58, 0.45, "Masked-child gate", bbox={"facecolor": "#f8d7da", "edgecolor": "black"})
    plt.text(0.84, 0.62, "Return EC-4 leaf", ha="center", bbox={"facecolor": "#fff3cd", "edgecolor": "black"})
    plt.text(0.84, 0.28, "Return safe prefix", ha="center", bbox={"facecolor": "#e2e3e5", "edgecolor": "black"})
    plt.annotate("", xy=(0.32, 0.82), xytext=(0.18, 0.82), arrowprops={"arrowstyle": "->"})
    plt.annotate("", xy=(0.56, 0.82), xytext=(0.46, 0.82), arrowprops={"arrowstyle": "->"})
    plt.annotate("", xy=(0.56, 0.50), xytext=(0.46, 0.82), arrowprops={"arrowstyle": "->"})
    plt.annotate("", xy=(0.79, 0.66), xytext=(0.75, 0.82), arrowprops={"arrowstyle": "->"})
    plt.annotate("", xy=(0.79, 0.32), xytext=(0.75, 0.50), arrowprops={"arrowstyle": "->"})
    plt.tight_layout()
    plt.savefig(FIGURES / "figure1_mcc_ec_schematic.png", dpi=300)
    plt.savefig(FIGURES / "figure1_mcc_ec_schematic.pdf")
    plt.close()

    fig2 = aggregate_df[aggregate_df["eval_split"] == "temporal_future_child"].copy()
    fig2.to_csv(FIG_SRC / "figure2_risk_coverage.csv", index=False)
    plt.figure(figsize=(8, 5))
    for method in ["forced_hierarchical", "one_nn_threshold", "flat_logistic", "hsc_style", "energy_threshold", "parent_openmax", "mcc_ec"]:
        method_df = fig2[fig2["method"] == method].sort_values("realized_coverage_mean")
        if method_df.empty:
            continue
        plt.errorbar(
            method_df["realized_coverage_mean"],
            method_df["catastrophic_overspecialization_rate_mean"],
            xerr=[
                method_df["realized_coverage_mean"] - method_df["realized_coverage_ci_low"].fillna(method_df["realized_coverage_mean"]),
                method_df["realized_coverage_ci_high"].fillna(method_df["realized_coverage_mean"]) - method_df["realized_coverage_mean"],
            ],
            yerr=[
                method_df["catastrophic_overspecialization_rate_mean"] - method_df["catastrophic_overspecialization_rate_ci_low"].fillna(method_df["catastrophic_overspecialization_rate_mean"]),
                method_df["catastrophic_overspecialization_rate_ci_high"].fillna(method_df["catastrophic_overspecialization_rate_mean"]) - method_df["catastrophic_overspecialization_rate_mean"],
            ],
            marker="o",
            label=method,
        )
    plt.xlabel("Realized coverage")
    plt.ylabel("Catastrophic overspecialization rate")
    plt.title("Temporal future-child risk-coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "figure2_temporal_risk_coverage.png", dpi=300)
    plt.savefig(FIGURES / "figure2_temporal_risk_coverage.pdf")
    plt.close()

    fig2b = exact_matched_df[exact_matched_df["eval_split"] == "temporal_mixed_posthoc"].copy()
    if not fig2b.empty:
        fig2b.to_csv(FIG_SRC / "figure2b_temporal_mixed_exact_matched.csv", index=False)
        plt.figure(figsize=(8, 5))
        for method in ["one_nn_threshold", "flat_logistic", "max_probability", "hsc_style", "energy_threshold", "parent_openmax", "mcc_ec"]:
            method_df = fig2b[fig2b["method"] == method].sort_values("coverage_target")
            if method_df.empty:
                continue
            plt.errorbar(
                method_df["realized_coverage_mean"],
                method_df["hierarchical_loss_mean"],
                marker="o",
                label=method,
            )
        plt.xlabel("Exact matched coverage")
        plt.ylabel("Hierarchical loss")
        plt.title("Post hoc temporal mixed benchmark")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES / "figure2b_temporal_mixed_exact_matched.png", dpi=300)
        plt.savefig(FIGURES / "figure2b_temporal_mixed_exact_matched.pdf")
        plt.close()

    if not leakage_df.empty:
        heat = leakage_df.pivot_table(index="feature_family", columns=["seed", "parent_id"], values="passed_audit", aggfunc="max", fill_value=0)
        plt.figure(figsize=(max(8, heat.shape[1] * 0.35), 3))
        sns.heatmap(heat, cmap="YlGn", cbar=False, linewidths=0.5, linecolor="white")
        plt.title("Leakage audit heatmap")
        plt.tight_layout()
        plt.savefig(FIGURES / "figure3_leakage_audit.png", dpi=300)
        plt.savefig(FIGURES / "figure3_leakage_audit.pdf")
        plt.close()

    if not transfer_df.empty and "method_name" not in transfer_df.columns:
        agg_transfer = (
            transfer_df.groupby("parent_id")
            .agg(
                masked_catastrophic_reduction=("masked_catastrophic_reduction", "mean"),
                future_catastrophic_reduction=("future_catastrophic_reduction", "mean"),
            )
            .reset_index()
        )
        plt.figure(figsize=(6, 5))
        plt.scatter(agg_transfer["masked_catastrophic_reduction"], agg_transfer["future_catastrophic_reduction"])
        for row in agg_transfer.itertuples(index=False):
            plt.text(row.masked_catastrophic_reduction, row.future_catastrophic_reduction, row.parent_id, fontsize=7)
        if len(agg_transfer) >= 2:
            pearson = stats.pearsonr(agg_transfer["masked_catastrophic_reduction"], agg_transfer["future_catastrophic_reduction"])
            spearman = stats.spearmanr(agg_transfer["masked_catastrophic_reduction"], agg_transfer["future_catastrophic_reduction"])
            plt.title(f"Transfer scatter\nPearson={pearson.statistic:.2f}, Spearman={spearman.statistic:.2f}")
        else:
            plt.title("Transfer scatter")
        plt.xlabel("Masked-child catastrophic reduction")
        plt.ylabel("Temporal future-child catastrophic reduction")
        plt.tight_layout()
        plt.savefig(FIGURES / "figure4_transfer_scatter.png", dpi=300)
        plt.savefig(FIGURES / "figure4_transfer_scatter.pdf")
        plt.close()
    elif not transfer_df.empty:
        plot_df = transfer_df[transfer_df["method_name"].isin(["mcc_ec", "hsc_style", "parent_openmax"])].copy()
        methods_to_plot = [name for name in ["mcc_ec", "hsc_style", "parent_openmax"] if name in set(plot_df["method_name"])]
        if methods_to_plot:
            fig, axes = plt.subplots(1, len(methods_to_plot), figsize=(6 * len(methods_to_plot), 5), squeeze=False)
            for idx, method_name in enumerate(methods_to_plot):
                ax = axes[0, idx]
                method_df = plot_df[plot_df["method_name"] == method_name].copy()
                ax.scatter(method_df["masked_catastrophic_reduction"], method_df["future_catastrophic_reduction"])
                for row in method_df.itertuples(index=False):
                    ax.text(row.masked_catastrophic_reduction, row.future_catastrophic_reduction, row.parent_id, fontsize=7)
                if len(method_df) >= 2:
                    pearson = stats.pearsonr(method_df["masked_catastrophic_reduction"], method_df["future_catastrophic_reduction"])
                    spearman = stats.spearmanr(method_df["masked_catastrophic_reduction"], method_df["future_catastrophic_reduction"])
                    ax.set_title(f"{method_name}\nPearson={pearson.statistic:.2f}, Spearman={spearman.statistic:.2f}")
                else:
                    ax.set_title(method_name)
                ax.set_xlabel("Masked-child catastrophic reduction")
                ax.set_ylabel("Temporal future-child catastrophic reduction")
            plt.tight_layout()
            plt.savefig(FIGURES / "figure4_transfer_scatter.png", dpi=300)
            plt.savefig(FIGURES / "figure4_transfer_scatter.pdf")
            plt.close()

    table1 = aggregate_df[aggregate_df["eval_split"].isin(["temporal_known_child", "temporal_future_child"])].copy()
    table1.to_csv(FIG_SRC / "table1_temporal_results.csv", index=False)
    table2 = aggregate_df[aggregate_df["eval_split"] == "masked_child_surrogate"].copy()
    if not leakage_df.empty:
        valid_events = leakage_df.groupby(["seed", "parent_id", "masked_child"])["passed_audit"].min().reset_index()
        table2["valid_masked_events"] = int(valid_events["passed_audit"].sum())
    table2.to_csv(FIG_SRC / "table2_masked_child_ablations.csv", index=False)
    exact_matched_df.to_csv(FIG_SRC / "table1b_posthoc_exact_matched.csv", index=False)

    bootstrap_payload = {}
    temporal_collapse_audit = {}
    best_generic = aggregate_df[
        (aggregate_df["method"].isin(["hsc_style", "parent_openmax"]))
        & (aggregate_df["eval_split"] == "temporal_future_child")
        & (aggregate_df["coverage_target"] == 0.80)
    ].sort_values("catastrophic_overspecialization_rate_mean")
    if not best_generic.empty:
        baseline_name = best_generic.iloc[0]["method"]
        mcc = temporal_preds[
            (temporal_preds["method"] == "mcc_ec")
            & (temporal_preds["coverage_target"] == 0.80)
            & (temporal_preds["eval_split"] == "temporal_future_child")
        ].copy()
        base = temporal_preds[
            (temporal_preds["method"] == baseline_name)
            & (temporal_preds["coverage_target"] == 0.80)
            & (temporal_preds["eval_split"] == "temporal_future_child")
        ].copy()
        merged = mcc[["seed", "accession", "true_ec4", "returned_prefix", "returned_depth"]].merge(
            base[["seed", "accession", "returned_prefix", "returned_depth"]],
            on=["seed", "accession"],
            suffixes=("_mcc", "_baseline"),
        )
        if not merged.empty:
            bootstrap_payload = {
                "baseline_name": baseline_name,
                "coverage_target": 0.80,
                "pointwise_metrics": [
                    paired_pointwise_stats(
                        merged,
                        metric_name="catastrophic_overspecialization_rate",
                        metric_fn=lambda frame, suffix: ((frame[f"returned_depth_{suffix}"] == 4) & (frame[f"returned_prefix_{suffix}"] != frame["true_ec4"])).astype(float).to_numpy(),
                        method_col="mcc",
                        baseline_col="baseline",
                    ),
                    paired_pointwise_stats(
                        merged,
                        metric_name="correct_safe_prefix_rate",
                        metric_fn=lambda frame, suffix: frame.apply(
                            lambda row: is_safe_prefix(row[f"returned_prefix_{suffix}"], row["true_ec4"]),
                            axis=1,
                        ).astype(float).to_numpy(),
                        method_col="mcc",
                        baseline_col="baseline",
                    ),
                    paired_pointwise_stats(
                        merged,
                        metric_name="hierarchical_loss",
                        metric_fn=lambda frame, suffix: frame.apply(
                            lambda row: hierarchical_loss(row[f"returned_prefix_{suffix}"], row["true_ec4"]),
                            axis=1,
                        ).astype(float).to_numpy(),
                        method_col="mcc",
                        baseline_col="baseline",
                    ),
                ],
            }
        auc_mcc = temporal_preds[
            (temporal_preds["method"] == "mcc_ec")
            & (temporal_preds["coverage_target"] == 0.80)
            & (temporal_preds["eval_split"].isin(["temporal_future_child", "temporal_known_child"]))
        ][["seed", "accession", "eval_split", "open_score"]].copy()
        auc_base = temporal_preds[
            (temporal_preds["method"] == baseline_name)
            & (temporal_preds["coverage_target"] == 0.80)
            & (temporal_preds["eval_split"].isin(["temporal_future_child", "temporal_known_child"]))
        ][["seed", "accession", "eval_split", "open_score"]].copy()
        auc_merged = auc_mcc.merge(auc_base, on=["seed", "accession", "eval_split"], suffixes=("_mcc", "_baseline"))
        if not auc_merged.empty:
            auc_merged["label"] = (auc_merged["eval_split"] == "temporal_future_child").astype(int)
            bootstrap_payload["pointwise_metrics"].append(
                paired_auc_stats(
                    auc_merged,
                    method_name="mcc_ec",
                    baseline_name=baseline_name,
                )
            )

    future_080 = aggregate_df[
        (aggregate_df["eval_split"] == "temporal_future_child") & (aggregate_df["coverage_target"] == 0.80)
    ][
        [
            "method",
            "realized_coverage_mean",
            "catastrophic_overspecialization_rate_mean",
            "correct_safe_prefix_rate_mean",
            "hierarchical_loss_mean",
        ]
    ].copy()
    if not future_080.empty:
        future_080["catastrophic_minus_coverage"] = future_080["catastrophic_overspecialization_rate_mean"] - future_080["realized_coverage_mean"]
        future_080["safe_prefix_minus_one_minus_coverage"] = future_080["correct_safe_prefix_rate_mean"] - (1.0 - future_080["realized_coverage_mean"])
        future_080["hierarchical_minus_expected"] = future_080["hierarchical_loss_mean"] - (1.0 + future_080["realized_coverage_mean"])
        temporal_collapse_audit = {
            "coverage_target": 0.80,
            "task_property": "For temporal future-child examples under a known EC-3 parent, any EC-4 descent is necessarily wrong because the true child is absent from the 2018 label vocabulary; rejecting to EC-3 is therefore always the correct safe prefix.",
            "method_rows": future_080.to_dict(orient="records"),
        }

    preregistered_claim = {"status": "insufficient_data", "reason": "No generic baseline available at temporal future-child coverage 0.80."}
    if not best_generic.empty:
        baseline_name = best_generic.iloc[0]["method"]
        future_mcc = aggregate_df[
            (aggregate_df["method"] == "mcc_ec")
            & (aggregate_df["eval_split"] == "temporal_future_child")
            & (aggregate_df["coverage_target"] == 0.80)
        ]
        future_base = aggregate_df[
            (aggregate_df["method"] == baseline_name)
            & (aggregate_df["eval_split"] == "temporal_future_child")
            & (aggregate_df["coverage_target"] == 0.80)
        ]
        masked_mcc = aggregate_df[
            (aggregate_df["method"] == "mcc_ec")
            & (aggregate_df["eval_split"] == "masked_child_surrogate")
            & (aggregate_df["coverage_target"] == 0.80)
        ]
        masked_base = aggregate_df[
            (aggregate_df["method"] == baseline_name)
            & (aggregate_df["eval_split"] == "masked_child_surrogate")
            & (aggregate_df["coverage_target"] == 0.80)
        ]
        if not future_mcc.empty and not future_base.empty and not masked_mcc.empty and not masked_base.empty:
            mcc_future_cat = float(future_mcc.iloc[0]["catastrophic_overspecialization_rate_mean"])
            base_future_cat = float(future_base.iloc[0]["catastrophic_overspecialization_rate_mean"])
            mcc_future_safe = float(future_mcc.iloc[0]["correct_safe_prefix_rate_mean"])
            base_future_safe = float(future_base.iloc[0]["correct_safe_prefix_rate_mean"])
            mcc_masked_safe = float(masked_mcc.iloc[0]["correct_safe_prefix_rate_mean"])
            base_masked_safe = float(masked_base.iloc[0]["correct_safe_prefix_rate_mean"])
            supported = (mcc_future_cat < base_future_cat) and (mcc_future_safe >= base_future_safe) and (mcc_masked_safe >= base_masked_safe)
            preregistered_claim = {
                "status": "supported" if supported else "downgraded",
                "baseline_name": baseline_name,
                "future_child_0_80": {
                    "mcc_catastrophic": mcc_future_cat,
                    "baseline_catastrophic": base_future_cat,
                    "mcc_safe_prefix": mcc_future_safe,
                    "baseline_safe_prefix": base_future_safe,
                },
                "masked_child_0_80": {
                    "mcc_safe_prefix": mcc_masked_safe,
                    "baseline_safe_prefix": base_masked_safe,
                },
                "reason": "Practical MCC-EC claim remains supported only if future-child overspecialization is lower and safe-prefix performance is not worse than the strongest generic baseline at matched 0.80 coverage."
                if supported
                else "Current rerun does not support the practical MCC-EC claim against the strongest generic baseline at matched 0.80 coverage; frame the study as a benchmark of surrogate calibration limits.",
            }

    corrected_claim = {
        "status": "insufficient_data",
        "reason": "Exact matched-coverage post hoc analysis did not have all required baselines.",
    }
    exact_generic = exact_matched_df[
        (exact_matched_df["method"].isin(["max_probability", "hsc_style", "energy_threshold", "parent_openmax"]))
        & (exact_matched_df["eval_split"] == "temporal_future_child")
        & (exact_matched_df["coverage_target"] == 0.80)
    ].sort_values("catastrophic_overspecialization_rate_mean")
    harder_temporal_setting = exact_matched_df[
        (exact_matched_df["eval_split"] == "temporal_mixed_posthoc") & (exact_matched_df["coverage_target"] == 0.80)
    ][
        [
            "method",
            "realized_coverage_mean",
            "catastrophic_overspecialization_rate_mean",
            "correct_safe_prefix_rate_mean",
            "hierarchical_loss_mean",
        ]
    ].copy()
    if not exact_generic.empty:
        baseline_name = exact_generic.iloc[0]["method"]
        future_mcc = exact_matched_df[
            (exact_matched_df["method"] == "mcc_ec")
            & (exact_matched_df["eval_split"] == "temporal_future_child")
            & (exact_matched_df["coverage_target"] == 0.80)
        ]
        future_base = exact_matched_df[
            (exact_matched_df["method"] == baseline_name)
            & (exact_matched_df["eval_split"] == "temporal_future_child")
            & (exact_matched_df["coverage_target"] == 0.80)
        ]
        masked_mcc = exact_matched_df[
            (exact_matched_df["method"] == "mcc_ec")
            & (exact_matched_df["eval_split"] == "masked_child_surrogate")
            & (exact_matched_df["coverage_target"] == 0.80)
        ]
        masked_maxprob = exact_matched_df[
            (exact_matched_df["method"] == "max_probability")
            & (exact_matched_df["eval_split"] == "masked_child_surrogate")
            & (exact_matched_df["coverage_target"] == 0.80)
        ]
        masked_openmax = exact_matched_df[
            (exact_matched_df["method"] == "parent_openmax")
            & (exact_matched_df["eval_split"] == "masked_child_surrogate")
            & (exact_matched_df["coverage_target"] == 0.80)
        ]
        if not future_mcc.empty and not future_base.empty and not masked_mcc.empty and not masked_maxprob.empty and not masked_openmax.empty:
            mcc_future_cat = float(future_mcc.iloc[0]["catastrophic_overspecialization_rate_mean"])
            base_future_cat = float(future_base.iloc[0]["catastrophic_overspecialization_rate_mean"])
            mcc_future_safe = float(future_mcc.iloc[0]["correct_safe_prefix_rate_mean"])
            base_future_safe = float(future_base.iloc[0]["correct_safe_prefix_rate_mean"])
            mcc_masked_safe = float(masked_mcc.iloc[0]["correct_safe_prefix_rate_mean"])
            maxprob_masked_safe = float(masked_maxprob.iloc[0]["correct_safe_prefix_rate_mean"])
            openmax_masked_safe = float(masked_openmax.iloc[0]["correct_safe_prefix_rate_mean"])
            supported = (
                (mcc_future_cat < base_future_cat)
                and (mcc_future_safe >= base_future_safe)
                and (mcc_masked_safe > maxprob_masked_safe)
                and (mcc_masked_safe > openmax_masked_safe)
            )
            corrected_claim = {
                "status": "supported" if supported else "downgraded",
                "strongest_generic_baseline": baseline_name,
                "future_child_exact_0_80": {
                    "mcc_catastrophic": mcc_future_cat,
                    "baseline_catastrophic": base_future_cat,
                    "mcc_safe_prefix": mcc_future_safe,
                    "baseline_safe_prefix": base_future_safe,
                },
                "masked_child_exact_0_80": {
                    "mcc_safe_prefix": mcc_masked_safe,
                    "max_probability_safe_prefix": maxprob_masked_safe,
                    "parent_openmax_safe_prefix": openmax_masked_safe,
                },
                "harder_posthoc_temporal_mixed_0_80": harder_temporal_setting.to_dict(orient="records"),
                "reason": "Corrected post hoc protocol supports the practical claim only if MCC-EC beats the strongest generic baseline on exact-matched future-child evaluation and also beats both masked-child max-probability and revised Parent-OpenMax at exact 0.80 coverage."
                if supported
                else "Corrected post hoc protocol does not show MCC-EC beating the strongest generic baseline and the required masked-child comparators; center the claim on benchmark limitations.",
            }

    write_json(
        ROOT / "results.json",
        {
            "idea_title": "Masked-Child Surrogate Calibration for Safe EC Prefix Decisions",
            "seeds": SEEDS,
            "coverage_targets": COVERAGE_TARGETS,
            "environment": metadata,
            "canonical_run": canonical_run,
            "method_annotations": {
                "parent_openmax": "Revised post hoc feasible implementation of the preregistered baseline; interpret separately from preregistered conclusions.",
                "temporal_mixed_posthoc": "Harder post hoc temporal benchmark mixing temporal-known-child and temporal-future-child cases to avoid a purely leaf-vs-prefix task.",
            },
            "preregistered_results": {
                "aggregate_metrics": aggregate_df.to_dict(orient="records"),
                "bootstrap_and_permutation": bootstrap_payload,
                "temporal_future_child_audit": temporal_collapse_audit,
                "claim_assessment": preregistered_claim,
            },
            "posthoc_protocol_revisions": metadata["protocol_revision"],
            "posthoc_exact_matched_results": {
                "aggregate_metrics": exact_matched_df.to_dict(orient="records"),
                "harder_temporal_setting": harder_temporal_setting.to_dict(orient="records"),
                "claim_assessment": corrected_claim,
            },
            "transfer_analysis": transfer_df.to_dict(orient="records") if not transfer_df.empty else [],
            "runtime_minutes": runtime_minutes,
        },
    )


def run_pipeline(stage: str = "all") -> None:
    start_time = time.time()
    ensure_dirs()
    log_stage("summary", f"starting experiment pipeline stage={stage}")

    metadata = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "gpu": run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "nproc": run_cmd(["nproc"]),
        "free_h": run_cmd(["free", "-h"]),
        "cuda_driver": run_cmd(["nvidia-smi"]),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "scikit_learn": sys.modules["sklearn"].__version__,
            "scipy": scipy.__version__,
        },
        "seed_list": SEEDS,
        "coverage_targets": COVERAGE_TARGETS,
        "cwd": str(ROOT),
        "command": " ".join([sys.executable] + sys.argv),
        "requested_stage": stage,
        "planned_environment": {
            "python": "3.10",
            "torch": "2.2.*",
            "transformers": "4.39.*",
            "scikit_learn": "1.4.*",
            "faiss_cpu": "1.7.*",
        },
        "planned_constraints": {
            "parent_cap": MAX_PARENTS,
            "split_ratios": {"train_core": 0.70, "cal_known": 0.15, "val_select": 0.15},
            "embedding_batch_size": EMBED_BATCH_SIZE,
        },
        "protocol_revision": {
            "environment": "Pinned Python 3.10 / torch 2.2 environment was not available in the current workspace; experiments were rerun in the actual installed stack recorded here.",
            "coverage_matching": "Main comparisons now use exact per-slice coverage matching from unlabeled selection-score quantiles to avoid broken cross-method coverage mismatches.",
            "parent_openmax": "Parent-OpenMax unknown-child fitting was revised to a feasible per-parent Weibull fit using all cal_known examples because the original per-child minimum made the baseline degenerate under the 70/15/15 split.",
        },
    }
    write_json(EXP / "environment" / "metadata.json", metadata)
    (EXP / "environment" / "requirements_frozen.txt").write_text(run_cmd([sys.executable, "-m", "pip", "freeze"]) + "\n")

    cache_paths = {
        "selected_sequences": DATA_PROCESSED / "selected_sequences.parquet",
        "split_seed_13": DATA_SPLITS / "seed_13.json",
        "split_seed_29": DATA_SPLITS / "seed_29.json",
        "split_seed_47": DATA_SPLITS / "seed_47.json",
        "embeddings": DATA_EMBEDDINGS / "esm2_t30_150M_selected_float16.npy",
        "embedding_index": DATA_EMBEDDINGS / "accession_to_index.json",
        "embedding_sequences": DATA_EMBEDDINGS / "cached_sequences.parquet",
    }
    metadata["cache_provenance"] = {
        name: file_provenance(path)
        for name, path in cache_paths.items()
        if path.exists()
    }

    if stage == "summary_only":
        metrics_df, all_prediction_frames, leakage_df, transfer_df = load_saved_artifacts()
        aggregate_df = aggregate_metrics(metrics_df, all_prediction_frames)
        _, exact_matched_metrics = compute_exact_matched_aggregate(all_prediction_frames)
        temporal_df_all = (
            pd.concat(
                [
                    frame[frame["eval_split"].isin(["temporal_known_child", "temporal_future_child"])].copy()
                    for frame in all_prediction_frames
                    if "eval_split" in frame.columns
                ],
                ignore_index=True,
            )
            if all_prediction_frames
            else pd.DataFrame()
        )
        runtime_minutes = float((time.time() - start_time) / 60.0)
        canonical_run = {
            "command": metadata["command"],
            "cwd": str(ROOT),
            "python_executable": sys.executable,
            "pid": os.getpid(),
            "requested_stage": stage,
            "runtime_minutes": runtime_minutes,
            "summary_log": str(EXP / "summary" / "logs" / "run.log"),
            "cache_provenance": metadata.get("cache_provenance", {}),
            "input_prediction_files": len(all_prediction_frames),
        }
        write_json(EXP / "summary" / "canonical_run.json", canonical_run)
        write_summary_results(
            metadata,
            metrics_df,
            aggregate_df,
            exact_matched_metrics,
            leakage_df,
            transfer_df,
            runtime_minutes,
            temporal_df_all,
            canonical_run,
        )
        log_stage("summary", f"completed experiment pipeline stage={stage} in {runtime_minutes:.2f} minutes")
        return

    dat_files = download_if_needed()
    df18 = clean_release(dat_files["2018_02"], "2018_02")
    df23 = clean_release(dat_files["2023_01"], "2023_01")
    df18_sel, df23_sel, seed_splits = build_manifests(df18, df23)
    selected_sequences = get_sequences_for_embedding(df18_sel, df23_sel)
    embeddings, accession_map = embed_sequences(selected_sequences)
    emb_lookup = make_embedding_lookup(selected_sequences, embeddings, accession_map)
    log_stage("data_prep", f"prepared selected benchmark with {df18_sel['ec3'].nunique()} parents and {len(selected_sequences)} unique sequences")

    all_metrics: List[pd.DataFrame] = []
    leakage_rows: List[dict] = []
    transfer_rows: List[dict] = []
    temporal_future_predictions: List[pd.DataFrame] = []
    all_prediction_frames: List[pd.DataFrame] = []
    scorer_rows = []

    for seed in SEEDS:
        log_stage("summary", f"running seed {seed}")
        split_df = seed_splits[seed]
        df18_indexed = df18_sel.reset_index().rename(columns={"index": "row_index"})
        with_split = df18_indexed.merge(split_df, on=["row_index", "accession", "ec3", "ec4"], how="inner")
        train_df = with_split[with_split["split"] == "train_core"].copy().reset_index(drop=True)
        cal_df = with_split[with_split["split"] == "cal_known"].copy().reset_index(drop=True)
        val_df = with_split[with_split["split"] == "val_select"].copy().reset_index(drop=True)
        temporal_df = df23_sel.copy().reset_index(drop=True)

        manifest = masked_child_manifest(train_df, cal_df, val_df, seed)
        save_parquet(manifest, EXP / "surrogate_benchmark" / f"seed_{seed}" / "masked_manifest.parquet")
        write_json(
            EXP / "surrogate_benchmark" / f"seed_{seed}" / "manifest.json",
            {"seed": seed, "events": manifest.to_dict(orient="records")},
        )

        base_models = fit_parent_models(train_df, emb_lookup, include_retrieval=True)
        reduced_models = fit_parent_models(train_df, emb_lookup, include_retrieval=False)
        flat_model = fit_flat_model(train_df, emb_lookup)
        scorer_rows.extend(
            {
                "seed": seed,
                "parent_id": model.parent_id,
                "include_retrieval": model.include_retrieval,
                "num_children": len(model.child_order),
                "fit_warnings": model.fit_warnings,
                "n_iter_max": int(np.max(model.classifier.n_iter_)),
            }
            for model in base_models.values()
        )
        log_stage("shared_scorer", f"seed {seed}: fit {len(base_models)} parent-local models and flat logistic baseline", seed=seed)

        val_scores = score_parent_examples(val_df, base_models, emb_lookup)
        val_scores["eval_split"] = "val_select"
        cal_scores = score_parent_examples(cal_df, base_models, emb_lookup)
        cal_scores["eval_split"] = "cal_known"
        temporal_scores = score_parent_examples(temporal_df, base_models, emb_lookup)
        temporal_scores = temporal_scores.merge(temporal_df[["accession", "eval_split"]], on="accession", how="left")

        known_quantiles = (
            cal_scores.groupby("parent_id")["true_prob"].apply(lambda s: float(np.quantile(s.to_numpy(), KNOWN_CHILD_ALPHA, method="linear"))).to_dict()
        )

        openmax_params = fit_openmax_params(cal_scores)
        for score_df in [val_scores, cal_scores, temporal_scores]:
            score_df["open_score"] = score_df.apply(lambda row: openmax_score(row["top_prob"], openmax_params.get(row["parent_id"])), axis=1)
            score_df["maxprob_open_score"] = 1.0 - score_df["top_prob"]
            score_df["hsc_open_score"] = 1.0 - score_df["hsc_conf"]

        masked_scores_frames = []
        reduced_masked_thresholds = {}
        masked_thresholds = {}
        for event in manifest.to_dict(orient="records"):
            masked_scores, no_retrieval_scores, audit_rows, masked_model, no_retrieval_model = score_masked_event(
                event,
                train_df,
                cal_df,
                val_df,
                emb_lookup,
            )
            if masked_scores is None or no_retrieval_scores is None or masked_model is None or no_retrieval_model is None:
                continue
            masked_scores["open_score"] = MASKED_CHILD_QUANTILE - masked_scores["top_prob"]
            no_retrieval_scores["open_score"] = MASKED_CHILD_QUANTILE - no_retrieval_scores["top_prob"]
            masked_scores_frames.append(masked_scores)
            masked_thresholds[event["parent_id"]] = float(np.quantile(masked_scores["top_prob"], MASKED_CHILD_QUANTILE, method="linear"))
            reduced_masked_thresholds[event["parent_id"]] = float(
                np.quantile(no_retrieval_scores["top_prob"], MASKED_CHILD_QUANTILE, method="linear")
            )
            leakage_rows.extend(audit_rows)
        masked_eval_df = pd.concat(masked_scores_frames, ignore_index=True) if masked_scores_frames else pd.DataFrame()
        pooled_masked_threshold = (
            float(np.quantile(masked_eval_df["top_prob"], MASKED_CHILD_QUANTILE, method="linear")) if not masked_eval_df.empty else 1.0
        )
        if not masked_eval_df.empty:
            save_parquet(masked_eval_df, EXP / "surrogate_benchmark" / f"seed_{seed}" / "masked_scores.parquet")

        one_nn_X = np.vstack([emb_lookup[acc] for acc in train_df["accession"]])
        one_nn_labels = train_df["ec4"].to_numpy()
        one_nn_nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(one_nn_X)

        def add_one_nn_scores(input_df: pd.DataFrame) -> pd.DataFrame:
            X = np.vstack([emb_lookup[acc] for acc in input_df["accession"]])
            dists, indices = one_nn_nn.kneighbors(X, return_distance=True)
            out = input_df[["accession", "ec4", "ec3"]].copy()
            out["true_ec4"] = out["ec4"]
            out["parent_id"] = out["ec3"]
            out["top_child"] = one_nn_labels[indices[:, 0]]
            out["top_prob"] = 1.0 - dists[:, 0]
            out["hsc_conf"] = out["top_prob"]
            out["open_score"] = 1.0 - out["top_prob"]
            return out[["accession", "true_ec4", "parent_id", "top_child", "top_prob", "hsc_conf", "open_score"]]

        val_one_nn = add_one_nn_scores(val_df)
        val_one_nn["eval_split"] = "val_select"
        temporal_one_nn = add_one_nn_scores(temporal_df).merge(temporal_df[["accession", "eval_split"]], on="accession", how="left")

        flat_val = score_flat_examples(val_df, flat_model, emb_lookup)
        flat_val["eval_split"] = "val_select"
        flat_temp = score_flat_examples(temporal_df, flat_model, emb_lookup).merge(temporal_df[["accession", "eval_split"]], on="accession", how="left")
        flat_predictions = []
        flat_thresholds = choose_validation_thresholds(flat_val, "top_prob", accept_high=True)
        for eval_split, split_df in flat_temp.groupby("eval_split"):
            for coverage_target, threshold in flat_thresholds.items():
                temp = split_df.copy()
                temp["method"] = "flat_logistic"
                temp["seed"] = seed
                temp["coverage_target"] = coverage_target
                returned = temp.apply(lambda row: flat_return_prefix(row["probabilities"], flat_model.classes_, threshold), axis=1)
                reject_only = temp.apply(lambda row: flat_return_prefix(row["probabilities"], flat_model.classes_, 2.0), axis=1)
                temp["returned_prefix"] = [item[0] for item in returned]
                temp["returned_depth"] = [item[1] for item in returned]
                temp["reject_prefix"] = [item[0] for item in reject_only]
                temp["reject_depth"] = [item[1] for item in reject_only]
                temp["top_child_score"] = temp["top_prob"]
                temp["selection_score"] = temp["top_prob"]
                temp["open_child_under_parent"] = (temp["returned_depth"] < 4).astype(int)
                temp["open_score"] = 1.0 - temp["top_prob"]
                flat_predictions.append(temp)
        flat_predictions_df = pd.concat(flat_predictions, ignore_index=True)
        flat_metrics = evaluate_predictions(flat_predictions_df)
        for cov in COVERAGE_TARGETS:
            pos = flat_predictions_df[(flat_predictions_df["coverage_target"] == cov) & (flat_predictions_df["eval_split"] == "temporal_future_child")]["open_score"].to_numpy()
            neg = flat_predictions_df[(flat_predictions_df["coverage_target"] == cov) & (flat_predictions_df["eval_split"] == "temporal_known_child")]["open_score"].to_numpy()
            attach_open_auc(flat_metrics, ("flat_logistic", seed, "temporal_future_child", cov), compute_open_auc(pos, neg))
        save_method_artifacts(
            "flat_logistic",
            seed,
            flat_thresholds,
            flat_predictions_df,
            flat_metrics,
            {
                "solver": "lbfgs",
                "max_iter": 500,
                "embedding_dim": int(one_nn_X.shape[1]),
                "fit_warnings": flat_model.fit_warnings,
            },
        )
        log_method("flat_logistic", seed, f"completed flat multinomial logistic baseline with thresholds={sanitize_json(flat_thresholds)}")
        all_metrics.append(flat_metrics)
        all_prediction_frames.append(flat_predictions_df)
        temporal_future_predictions.append(flat_predictions_df[flat_predictions_df["eval_split"] == "temporal_future_child"].copy())

        forced_predictions = pd.concat(
            [
                run_forced(temporal_scores[temporal_scores["eval_split"] == split].copy(), seed, split)
                for split in ["temporal_known_child", "temporal_future_child"]
            ],
            ignore_index=True,
        )
        forced_metrics = evaluate_predictions(forced_predictions)
        save_method_artifacts("forced_hierarchical", seed, {("temporal_known_child", target): None for target in COVERAGE_TARGETS} | {("temporal_future_child", target): None for target in COVERAGE_TARGETS}, forced_predictions, forced_metrics, {"description": "always descend to EC-4"})
        log_method("forced_hierarchical", seed, "completed forced hierarchical baseline")
        all_metrics.append(forced_metrics)
        all_prediction_frames.append(forced_predictions)
        temporal_future_predictions.append(forced_predictions[forced_predictions["eval_split"] == "temporal_future_child"].copy())

        temporal_prediction_store = {
            "forced_hierarchical": forced_predictions,
            "flat_logistic": flat_predictions_df,
        }

        for method, val_df_method, eval_df_method, score_col, accept_high, open_score_col in [
            ("one_nn_threshold", val_one_nn, temporal_one_nn, "top_prob", True, "open_score"),
            ("max_probability", val_scores, temporal_scores, "top_prob", True, "maxprob_open_score"),
            ("hsc_style", val_scores, temporal_scores, "hsc_conf", True, "hsc_open_score"),
            ("energy_threshold", val_scores.assign(energy_accept=-val_scores["energy"]), temporal_scores.assign(energy_accept=-temporal_scores["energy"]), "energy_accept", True, "energy"),
            ("parent_openmax", val_scores, temporal_scores, "open_score", False, "open_score"),
        ]:
            predictions, metrics, thresholds = run_threshold_method(
                method,
                seed,
                val_df_method,
                eval_df_method,
                score_col,
                accept_high=accept_high,
                open_score_col=open_score_col,
            )
            if method == "parent_openmax":
                for cov in COVERAGE_TARGETS:
                    pos = predictions[(predictions["coverage_target"] == cov) & (predictions["eval_split"] == "temporal_future_child")]["open_score"].to_numpy()
                    neg = predictions[(predictions["coverage_target"] == cov) & (predictions["eval_split"] == "temporal_known_child")]["open_score"].to_numpy()
                    attach_open_auc(metrics, (method, seed, "temporal_future_child", cov), compute_open_auc(pos, neg))
            if method == "energy_threshold":
                for cov in COVERAGE_TARGETS:
                    pos = predictions[(predictions["coverage_target"] == cov) & (predictions["eval_split"] == "temporal_future_child")]["open_score"].to_numpy()
                    neg = predictions[(predictions["coverage_target"] == cov) & (predictions["eval_split"] == "temporal_known_child")]["open_score"].to_numpy()
                    attach_open_auc(metrics, (method, seed, "temporal_future_child", cov), compute_open_auc(pos, neg))
            save_method_artifacts(
                method,
                seed,
                thresholds,
                predictions,
                metrics,
                {
                    "score_column": score_col,
                    "openmax_parent_count": len(openmax_params) if method == "parent_openmax" else None,
                },
            )
            log_method(method, seed, "completed baseline evaluation")
            all_metrics.append(metrics)
            all_prediction_frames.append(predictions)
            temporal_prediction_store[method] = predictions
            temporal_future_predictions.append(predictions[predictions["eval_split"] == "temporal_future_child"].copy())

        def make_mcc_scores(score_df: pd.DataFrame, conformal: Dict[str, float], masked: Dict[str, float]) -> pd.DataFrame:
            out = score_df.copy()
            out["conformal_pass"] = out.apply(lambda row: row["top_prob"] >= conformal.get(row["parent_id"], 1.0), axis=1)
            out["masked_child_pass"] = out.apply(lambda row: row["top_prob"] > masked.get(row["parent_id"], 1.0), axis=1)
            out["mcc_score"] = out.apply(
                lambda row: min(
                    row["top_prob"] - conformal.get(row["parent_id"], 1.0),
                    row["top_prob"] - masked.get(row["parent_id"], 1.0),
                ),
                axis=1,
            )
            out["open_score"] = out.apply(lambda row: masked.get(row["parent_id"], 1.0) - row["top_prob"], axis=1)
            return out

        val_mcc = make_mcc_scores(val_scores, known_quantiles, masked_thresholds)
        temporal_mcc = make_mcc_scores(temporal_scores, known_quantiles, masked_thresholds)
        log_method(
            "mcc_ec",
            seed,
            f"known-child parents={len(known_quantiles)}, masked-child parents={len(masked_thresholds)}, openmax parents={len(openmax_params)}",
        )
        mcc_predictions, mcc_metrics, mcc_thresholds = run_threshold_method(
            "mcc_ec",
            seed,
            val_mcc,
            temporal_mcc,
            "mcc_score",
            accept_high=True,
            open_score_col="open_score",
        )
        merged_state = temporal_mcc[["accession", "conformal_pass", "masked_child_pass"]].drop_duplicates("accession")
        mcc_predictions = mcc_predictions.merge(merged_state, on="accession", how="left")
        for cov in COVERAGE_TARGETS:
            pos = mcc_predictions[(mcc_predictions["coverage_target"] == cov) & (mcc_predictions["eval_split"] == "temporal_future_child")]["open_score"].to_numpy()
            neg = mcc_predictions[(mcc_predictions["coverage_target"] == cov) & (mcc_predictions["eval_split"] == "temporal_known_child")]["open_score"].to_numpy()
            attach_open_auc(mcc_metrics, ("mcc_ec", seed, "temporal_future_child", cov), compute_open_auc(pos, neg))
        save_method_artifacts(
            "mcc_ec",
            seed,
            mcc_thresholds,
            mcc_predictions,
            mcc_metrics,
            {
                "known_child_alpha": KNOWN_CHILD_ALPHA,
                "masked_child_quantile": MASKED_CHILD_QUANTILE,
                "masked_parent_count": len(masked_thresholds),
            },
        )
        log_method("mcc_ec", seed, "completed MCC-EC evaluation")
        all_metrics.append(mcc_metrics)
        all_prediction_frames.append(mcc_predictions)
        temporal_future_predictions.append(mcc_predictions[mcc_predictions["eval_split"] == "temporal_future_child"].copy())
        temporal_prediction_store["mcc_ec"] = mcc_predictions

        masked_predictions_store = {}
        if not masked_eval_df.empty:
            masked_eval_df["maxprob_open_score"] = 1.0 - masked_eval_df["top_prob"]
            masked_eval_df["hsc_open_score"] = 1.0 - masked_eval_df["hsc_conf"]
            masked_mcc_scores = make_mcc_scores(masked_eval_df, known_quantiles, masked_thresholds)
            masked_mcc_preds, masked_mcc_metrics, _ = run_threshold_method(
                "mcc_ec",
                seed,
                val_mcc,
                masked_mcc_scores,
                "mcc_score",
                accept_high=True,
                open_score_col="open_score",
            )
            neg_scores = temporal_prediction_store["mcc_ec"][
                (temporal_prediction_store["mcc_ec"]["coverage_target"] == 0.80)
                & (temporal_prediction_store["mcc_ec"]["eval_split"] == "temporal_known_child")
            ]["open_score"].to_numpy()
            for cov in COVERAGE_TARGETS:
                pos = masked_mcc_preds[masked_mcc_preds["coverage_target"] == cov]["open_score"].to_numpy()
                attach_open_auc(masked_mcc_metrics, ("mcc_ec", seed, "masked_child_surrogate", cov), compute_open_auc(pos, neg_scores))
            masked_predictions_store["mcc_ec"] = masked_mcc_preds
            save_auxiliary_eval_artifacts("mcc_ec", seed, "masked_child", masked_mcc_preds, masked_mcc_metrics)
            all_metrics.append(masked_mcc_metrics)
            all_prediction_frames.append(masked_mcc_preds)

            masked_hsc_preds, masked_hsc_metrics, _ = run_threshold_method(
                "hsc_style",
                seed,
                val_scores,
                masked_eval_df,
                "hsc_conf",
                accept_high=True,
                open_score_col="hsc_open_score",
            )
            for cov in COVERAGE_TARGETS:
                pos = masked_hsc_preds[masked_hsc_preds["coverage_target"] == cov]["open_score"].to_numpy()
                attach_open_auc(masked_hsc_metrics, ("hsc_style", seed, "masked_child_surrogate", cov), compute_open_auc(pos, neg_scores))
            masked_predictions_store["hsc_style"] = masked_hsc_preds
            save_auxiliary_eval_artifacts("hsc_style", seed, "masked_child", masked_hsc_preds, masked_hsc_metrics)
            all_metrics.append(masked_hsc_metrics)
            all_prediction_frames.append(masked_hsc_preds)

            masked_maxprob_preds, masked_maxprob_metrics, _ = run_threshold_method(
                "max_probability",
                seed,
                val_scores,
                masked_eval_df,
                "top_prob",
                accept_high=True,
                open_score_col="maxprob_open_score",
            )
            for cov in COVERAGE_TARGETS:
                pos = masked_maxprob_preds[masked_maxprob_preds["coverage_target"] == cov]["open_score"].to_numpy()
                attach_open_auc(masked_maxprob_metrics, ("max_probability", seed, "masked_child_surrogate", cov), compute_open_auc(pos, neg_scores))
            masked_predictions_store["max_probability"] = masked_maxprob_preds
            save_auxiliary_eval_artifacts("max_probability", seed, "masked_child", masked_maxprob_preds, masked_maxprob_metrics)
            all_metrics.append(masked_maxprob_metrics)
            all_prediction_frames.append(masked_maxprob_preds)

            masked_energy_df = masked_eval_df.assign(energy_accept=-masked_eval_df["energy"])
            masked_energy_preds, masked_energy_metrics, _ = run_threshold_method(
                "energy_threshold",
                seed,
                val_scores.assign(energy_accept=-val_scores["energy"]),
                masked_energy_df,
                "energy_accept",
                accept_high=True,
                open_score_col="energy",
            )
            for cov in COVERAGE_TARGETS:
                pos = masked_energy_preds[masked_energy_preds["coverage_target"] == cov]["open_score"].to_numpy()
                attach_open_auc(masked_energy_metrics, ("energy_threshold", seed, "masked_child_surrogate", cov), compute_open_auc(pos, neg_scores))
            masked_predictions_store["energy_threshold"] = masked_energy_preds
            save_auxiliary_eval_artifacts("energy_threshold", seed, "masked_child", masked_energy_preds, masked_energy_metrics)
            all_metrics.append(masked_energy_metrics)
            all_prediction_frames.append(masked_energy_preds)

            masked_open_df = masked_eval_df.copy()
            masked_open_df["open_score"] = masked_open_df.apply(lambda row: openmax_score(row["top_prob"], openmax_params.get(row["parent_id"])), axis=1)
            masked_open_preds, masked_open_metrics, _ = run_threshold_method(
                "parent_openmax",
                seed,
                val_scores,
                masked_open_df,
                "open_score",
                accept_high=False,
                open_score_col="open_score",
            )
            for cov in COVERAGE_TARGETS:
                pos = masked_open_preds[masked_open_preds["coverage_target"] == cov]["open_score"].to_numpy()
                attach_open_auc(masked_open_metrics, ("parent_openmax", seed, "masked_child_surrogate", cov), compute_open_auc(pos, neg_scores))
            masked_predictions_store["parent_openmax"] = masked_open_preds
            save_auxiliary_eval_artifacts("parent_openmax", seed, "masked_child", masked_open_preds, masked_open_metrics)
            all_metrics.append(masked_open_metrics)
            all_prediction_frames.append(masked_open_preds)

        ablation_defs = {
            "ablation_no_masked_gate": (
                val_scores.assign(score=val_scores.apply(lambda row: row["top_prob"] - known_quantiles.get(row["parent_id"], 1.0), axis=1), open_score=1.0 - val_scores["top_prob"]),
                temporal_scores.assign(score=temporal_scores.apply(lambda row: row["top_prob"] - known_quantiles.get(row["parent_id"], 1.0), axis=1), open_score=1.0 - temporal_scores["top_prob"]),
            ),
            "ablation_no_conformal_gate": (
                val_scores.assign(score=val_scores.apply(lambda row: row["top_prob"] - masked_thresholds.get(row["parent_id"], 1.0), axis=1), open_score=masked_thresholds.get(row["parent_id"], 1.0) if False else 0.0),
                temporal_scores.assign(score=temporal_scores.apply(lambda row: row["top_prob"] - masked_thresholds.get(row["parent_id"], 1.0), axis=1)),
            ),
            "ablation_global_masked_threshold": (
                val_scores.assign(score=val_scores.apply(lambda row: min(row["top_prob"] - known_quantiles.get(row["parent_id"], 1.0), row["top_prob"] - pooled_masked_threshold), axis=1)),
                temporal_scores.assign(score=temporal_scores.apply(lambda row: min(row["top_prob"] - known_quantiles.get(row["parent_id"], 1.0), row["top_prob"] - pooled_masked_threshold), axis=1)),
            ),
        }
        val_reduced = score_parent_examples(val_df, reduced_models, emb_lookup)
        val_reduced["eval_split"] = "val_select"
        cal_reduced = score_parent_examples(cal_df, reduced_models, emb_lookup)
        reduced_quantiles = (
            cal_reduced.groupby("parent_id")["true_prob"].apply(lambda s: float(np.quantile(s.to_numpy(), KNOWN_CHILD_ALPHA, method="linear"))).to_dict()
        )
        temp_reduced = score_parent_examples(temporal_df, reduced_models, emb_lookup).merge(temporal_df[["accession", "eval_split"]], on="accession", how="left")
        ablation_defs["ablation_no_retrieval_surrogate"] = (
            val_reduced.assign(
                score=val_reduced.apply(
                    lambda row: min(
                        row["top_prob"] - reduced_quantiles.get(row["parent_id"], 1.0),
                        row["top_prob"] - reduced_masked_thresholds.get(row["parent_id"], 1.0),
                    ),
                    axis=1,
                ),
                open_score=val_reduced.apply(lambda row: reduced_masked_thresholds.get(row["parent_id"], 1.0) - row["top_prob"], axis=1),
            ),
            temp_reduced.assign(
                score=temp_reduced.apply(
                    lambda row: min(
                        row["top_prob"] - reduced_quantiles.get(row["parent_id"], 1.0),
                        row["top_prob"] - reduced_masked_thresholds.get(row["parent_id"], 1.0),
                    ),
                    axis=1,
                ),
                open_score=temp_reduced.apply(lambda row: reduced_masked_thresholds.get(row["parent_id"], 1.0) - row["top_prob"], axis=1),
            ),
        )

        ablation_defs["ablation_no_conformal_gate"] = (
            val_scores.assign(
                score=val_scores.apply(lambda row: row["top_prob"] - masked_thresholds.get(row["parent_id"], 1.0), axis=1),
                open_score=val_scores.apply(lambda row: masked_thresholds.get(row["parent_id"], 1.0) - row["top_prob"], axis=1),
            ),
            temporal_scores.assign(
                score=temporal_scores.apply(lambda row: row["top_prob"] - masked_thresholds.get(row["parent_id"], 1.0), axis=1),
                open_score=temporal_scores.apply(lambda row: masked_thresholds.get(row["parent_id"], 1.0) - row["top_prob"], axis=1),
            ),
        )
        ablation_defs["ablation_global_masked_threshold"] = (
            val_scores.assign(
                score=val_scores.apply(lambda row: min(row["top_prob"] - known_quantiles.get(row["parent_id"], 1.0), row["top_prob"] - pooled_masked_threshold), axis=1),
                open_score=pooled_masked_threshold - val_scores["top_prob"],
            ),
            temporal_scores.assign(
                score=temporal_scores.apply(lambda row: min(row["top_prob"] - known_quantiles.get(row["parent_id"], 1.0), row["top_prob"] - pooled_masked_threshold), axis=1),
                open_score=pooled_masked_threshold - temporal_scores["top_prob"],
            ),
        )

        for name, (val_base, temp_base) in ablation_defs.items():
            preds, metrics, thresholds = run_threshold_method(
                name,
                seed,
                val_base,
                temp_base,
                "score",
                accept_high=True,
                open_score_col="open_score",
            )
            save_method_artifacts(name, seed, thresholds, preds, metrics, {"ablation_name": name})
            log_method(name, seed, "completed ablation")
            all_metrics.append(metrics)
            all_prediction_frames.append(preds)

        generic_rows = []
        for baseline in ["hsc_style", "parent_openmax"]:
            baseline_metrics = evaluate_predictions(temporal_prediction_store[baseline])
            match = baseline_metrics[
                (baseline_metrics["eval_split"] == "temporal_future_child")
                & (baseline_metrics["coverage_target"] == 0.80)
            ]
            if not match.empty:
                generic_rows.append((baseline, float(match.iloc[0]["catastrophic_overspecialization_rate"])))
        baseline_name = sorted(generic_rows, key=lambda item: item[1])[0][0] if generic_rows else "hsc_style"
        comparator_name = "parent_openmax" if baseline_name == "hsc_style" else "hsc_style"

        if not masked_eval_df.empty and baseline_name in masked_predictions_store and comparator_name in masked_predictions_store:
            transfer_rows.extend(
                build_transfer_rows(
                    seed,
                    "mcc_ec",
                    baseline_name,
                    mcc_predictions,
                    temporal_prediction_store[baseline_name],
                    masked_predictions_store["mcc_ec"],
                    masked_predictions_store[baseline_name],
                )
            )
            transfer_rows.extend(
                build_transfer_rows(
                    seed,
                    baseline_name,
                    comparator_name,
                    temporal_prediction_store[baseline_name],
                    temporal_prediction_store[comparator_name],
                    masked_predictions_store[baseline_name],
                    masked_predictions_store[comparator_name],
                )
            )

    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    leakage_df = pd.DataFrame(leakage_rows)
    save_parquet(pd.DataFrame(scorer_rows), EXP / "shared_scorer" / "model_fit_summary.parquet")
    write_json(
        EXP / "shared_scorer" / "results.json",
        {
            "seed_parent_fits": len(scorer_rows),
            "max_parent_iter": int(max((row["n_iter_max"] for row in scorer_rows), default=0)),
            "warning_parent_count": int(sum(1 for row in scorer_rows if row["fit_warnings"])),
        },
    )
    if not leakage_df.empty:
        leakage_df.to_csv(EXP / "surrogate_benchmark" / "leakage_audit.csv", index=False)
        write_json(
            EXP / "surrogate_benchmark" / "results.json",
            {
                "valid_masked_events": int(
                    leakage_df.groupby(["seed", "parent_id", "masked_child"])["passed_audit"].min().sum()
                ),
                "total_audit_rows": int(len(leakage_df)),
            },
        )

    transfer_df = pd.DataFrame(transfer_rows)
    if not transfer_df.empty:
        future_counts = (
            transfer_df.groupby(["method_name", "parent_id"])["future_examples"]
            .sum()
            .rename("future_examples_total")
            .reset_index()
        )
        transfer_df = transfer_df.merge(future_counts, on=["method_name", "parent_id"], how="left")
        eligible_transfer = transfer_df[transfer_df["future_examples_total"] >= 8].copy()
        train_stats = parent_level_stats(df18_sel, emb_lookup)
        merged_transfer = (
            eligible_transfer.groupby(["method_name", "comparator_name", "parent_id"])
            .agg(
                masked_catastrophic_reduction=("masked_catastrophic_reduction", "mean"),
                future_catastrophic_reduction=("future_catastrophic_reduction", "mean"),
                masked_safe_prefix_gain=("masked_safe_prefix_gain", "mean"),
                future_safe_prefix_gain=("future_safe_prefix_gain", "mean"),
                masked_open_child_auroc_gain=("masked_open_child_auroc_gain", "mean"),
                future_open_child_auroc_gain=("future_open_child_auroc_gain", "mean"),
            )
            .reset_index()
            .merge(train_stats, on="parent_id", how="left")
        )
        transfer_summaries = []
        for method_name, method_slice in merged_transfer.groupby("method_name"):
            if len(method_slice) < 2:
                transfer_summaries.append({"method_name": method_name, "note": "insufficient eligible parents"})
                continue
            pearson = stats.pearsonr(method_slice["masked_catastrophic_reduction"], method_slice["future_catastrophic_reduction"])
            spearman = stats.spearmanr(method_slice["masked_catastrophic_reduction"], method_slice["future_catastrophic_reduction"])
            X = method_slice[
                [
                    "masked_catastrophic_reduction",
                    "sibling_support_entropy",
                    "prototype_compactness",
                    "minimum_inter_child_cosine_gap",
                    "neighbor_overlap_rate",
                ]
            ].fillna(0.0)
            y = method_slice["future_catastrophic_reduction"].fillna(0.0)
            ridge = Ridge(alpha=1.0).fit(X, y)
            transfer_summaries.append(
                {
                    "method_name": method_name,
                    "comparator_name": method_slice["comparator_name"].iloc[0],
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "ridge_intercept": float(ridge.intercept_),
                    "ridge_coef": {name: float(value) for name, value in zip(X.columns, ridge.coef_)},
                    "eligible_parent_count": int(len(method_slice)),
                }
            )
        write_json(
            EXP / "transfer_analysis" / "results.json",
            {
                "comparisons": transfer_summaries,
                "parent_rows": merged_transfer.to_dict(orient="records"),
            },
        )
        transfer_df = merged_transfer
    else:
        write_json(EXP / "transfer_analysis" / "results.json", {"note": "no valid transfer rows"})

    if stage == "experiments_only":
        runtime_minutes = float((time.time() - start_time) / 60.0)
        log_stage("summary", f"completed experiment pipeline stage={stage} in {runtime_minutes:.2f} minutes")
        return

    aggregate_df = aggregate_metrics(metrics_df, all_prediction_frames)
    exact_matched_seed_metrics, exact_matched_metrics = compute_exact_matched_aggregate(all_prediction_frames)
    if not exact_matched_seed_metrics.empty:
        exact_matched_seed_metrics.to_csv(EXP / "summary" / "exact_matched_seed_metrics.csv", index=False)
    temporal_df_all = (
        pd.concat(
            [
                frame[frame["eval_split"].isin(["temporal_known_child", "temporal_future_child"])].copy()
                for frame in all_prediction_frames
                if "eval_split" in frame.columns
            ],
            ignore_index=True,
        )
        if all_prediction_frames
        else pd.DataFrame()
    )
    runtime_minutes = float((time.time() - start_time) / 60.0)
    canonical_run = {
        "command": metadata["command"],
        "cwd": str(ROOT),
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "requested_stage": stage,
        "runtime_minutes": runtime_minutes,
        "summary_log": str(EXP / "summary" / "logs" / "run.log"),
        "cache_provenance": metadata.get("cache_provenance", {}),
    }
    write_json(EXP / "summary" / "canonical_run.json", canonical_run)
    write_summary_results(
        metadata,
        metrics_df,
        aggregate_df,
        exact_matched_metrics,
        leakage_df,
        transfer_df,
        runtime_minutes,
        temporal_df_all,
        canonical_run,
    )
    log_stage("summary", f"completed experiment pipeline stage={stage} in {runtime_minutes:.2f} minutes")


def main(default_stage: str = "all") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default=default_stage)
    args = parser.parse_args()
    run_pipeline(stage=args.stage)


if __name__ == "__main__":
    main()
