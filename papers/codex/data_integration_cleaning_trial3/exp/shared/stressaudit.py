import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "raw" / "Valentine-datasets.zip"
OUTPUTS = ROOT / "outputs"
FIGURES = ROOT / "figures"
PRIMARY_SEED = 17
SEEDS = [17, 23, 42]
THREAD_CAPS = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "NUMEXPR_NUM_THREADS": "2",
    "CUDA_VISIBLE_DEVICES": "",
}
FAMILY_SPECS = [
    ("OpenData", "valentine_standard", 18),
    ("TPC-DI", "header_poor", 17),
    ("ChEMBL", "context_limited_chembl", 17),
]
TARGET_FAMILY_COUNTS = {
    "clean_dev": {"valentine_standard": 4, "header_poor": 4, "context_limited_chembl": 4},
    "stress_dev": {"valentine_standard": 3, "header_poor": 2, "context_limited_chembl": 3},
    "test": {"valentine_standard": 11, "header_poor": 11, "context_limited_chembl": 10},
}
CONDITIONS = ["clean", "header-stress", "context-stress", "composite-stress"]
METHODS = ["lexical", "similarity_flooding", "ptlm"]
EXPERIMENTS = [
    "environment_setup",
    "data_inventory",
    "stress_views",
    "lexical",
    "similarity_flooding",
    "ptlm",
    "calibration",
    "ablations",
    "visualization",
    "evaluation",
]
PKG_NAMES = [
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "rapidfuzz",
    "networkx",
    "sentence-transformers",
    "transformers",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "psutil",
    "pyarrow",
    "torch",
]
THRESHOLD_GRIDS = {
    "lexical": [0.20, 0.30, 0.40, 0.50, 0.60],
    "similarity_flooding": [0.20, 0.30, 0.40, 0.50],
    "ptlm": [0.30, 0.40, 0.50, 0.60, 0.70],
}
BOOTSTRAP_RESAMPLES = 2000
CLAIMED_RAM_GB = 128
CLAIMED_CPU_CORES = 2
PTLM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PTLM_TEMPLATE = "[TABLE] {table} [COL] {column} [DESC] {description} [TYPE] {raw_type} [CTX] {context}"
LEXICAL_RECIPE = {
    "header_score": "0.5 * token_jaccard + 0.5 * normalized_levenshtein on stressed column names",
    "context_score": "same lexical score on table/type/sibling/description context text when available",
    "final_score": "0.7 * header_score + 0.3 * context_score when context exists, else header_score only",
    "assignment": "Hungarian one-to-one matching with global acceptance threshold",
}
SIMILARITY_FLOODING_CONFIG = {
    "max_iterations": 30,
    "damping": 0.85,
    "early_stop_max_abs_delta": 1e-4,
    "initialization": "same lexical score as lexical baseline",
}


@dataclass
class PairData:
    pair_id: str
    source_name: str
    target_name: str
    family_source: str
    family_label: str
    scenario: str
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    source_meta: Dict[str, dict]
    target_meta: Dict[str, dict]
    gold: List[Tuple[str, str]]


_MODEL = None
_PAIR_CACHE: Dict[str, PairData] = {}


def ensure_dirs() -> None:
    for rel in [
        "outputs/data",
        "outputs/predictions/lexical",
        "outputs/predictions/similarity_flooding",
        "outputs/predictions/ptlm",
        "outputs/predictions/ptlm/embeddings",
        "outputs/eval",
        "outputs/figures",
        "outputs/appendix",
        "figures",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    for exp_name in EXPERIMENTS:
        (ROOT / "exp" / exp_name / "logs").mkdir(parents=True, exist_ok=True)


def experiment_dir(name: str) -> Path:
    return ROOT / "exp" / name


def result_path(name: str) -> Path:
    return experiment_dir(name) / "results.json"


def config_path(name: str) -> Path:
    return experiment_dir(name) / "config.json"


def log_path(name: str) -> Path:
    return experiment_dir(name) / "logs" / "run.log"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def write_log(name: str, lines: List[str]) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = "\n".join([f"[{stamp}] {line}" for line in lines]) + "\n"
    log_path(name).write_text(payload)


def write_experiment_artifacts(name: str, config: dict, results: dict, log_lines: List[str]) -> None:
    write_json(config_path(name), config)
    write_json(result_path(name), results)
    write_log(name, log_lines)


def set_env() -> None:
    for key, value in THREAD_CAPS.items():
        os.environ[key] = value
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


def set_repeat_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_identifier(text: str) -> List[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    return [token for token in normalize_ws(text.lower()).split(" ") if token]


def norm_name(text: str) -> str:
    return " ".join(split_identifier(text))


def lev_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - (Levenshtein.distance(a, b) / max(len(a), len(b), 1))


def token_jaccard(a: str, b: str) -> float:
    sa = set(split_identifier(a))
    sb = set(split_identifier(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lexical_similarity(a: str, b: str) -> float:
    return 0.5 * token_jaccard(a, b) + 0.5 * lev_sim(norm_name(a), norm_name(b))


def coarse_type(type_name: str) -> str:
    t = (type_name or "").lower()
    if any(x in t for x in ["int", "numeric", "real", "decimal", "double", "float", "bigint", "smallint"]):
        return "numeric"
    if "date" in t or "time" in t:
        return "datetime"
    if "bool" in t:
        return "boolean"
    return "text"


def extract_description(meta: dict) -> str:
    for key in ["description", "desc", "comment", "definition", "details", "semantic_description"]:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def decode_json_bytes(raw: bytes) -> dict:
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            return json.loads(raw.decode(encoding))
        except Exception:
            continue
    raise UnicodeDecodeError("json", raw, 0, 1, "Unable to decode metadata JSON with fallback encodings.")


def parse_mapping(raw: str) -> List[Tuple[str, str]]:
    payload = json.loads(raw)
    return sorted((m["source_column"], m["target_column"]) for m in payload["matches"])


def open_zip() -> zipfile.ZipFile:
    return zipfile.ZipFile(ZIP_PATH)


def list_candidate_pairs() -> List[dict]:
    rows = []
    with open_zip() as zf:
        for path in zf.namelist():
            if not path.startswith("Valentine-datasets/") or not path.endswith("_mapping.json"):
                continue
            if "__MACOSX" in path or "_ac" in path:
                continue
            parts = path.split("/")
            if len(parts) < 4:
                continue
            source_root = parts[1]
            if source_root not in {"OpenData", "TPC-DI", "ChEMBL"}:
                continue
            rows.append(
                {
                    "pair_id": parts[-2],
                    "zip_dir": "/".join(parts[:-1]),
                    "source_root": source_root,
                    "scenario": parts[2],
                }
            )
    rows.sort(key=lambda item: (item["source_root"], item["scenario"], item["pair_id"]))
    return rows


def freeze_inventory(seed: int = PRIMARY_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    candidates = list_candidate_pairs()
    chosen = []
    for source_root, family_label, quota in FAMILY_SPECS:
        pool = [row.copy() for row in candidates if row["source_root"] == source_root]
        grouped = defaultdict(list)
        for row in pool:
            grouped[row["scenario"]].append(row)
        scenarios = sorted(grouped)
        idx = 0
        while len([row for row in chosen if row["source_root"] == source_root]) < quota:
            scenario = scenarios[idx % len(scenarios)]
            bucket = grouped[scenario]
            if bucket:
                pick = bucket.pop(rng.randrange(len(bucket)))
                pick["family_label"] = family_label
                chosen.append(pick)
            idx += 1
    return pd.DataFrame(chosen).sort_values(["source_root", "scenario", "pair_id"]).reset_index(drop=True)


def split_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    df = inventory.copy()
    df["split"] = "unassigned"
    grouped = {family: list(sub.index) for family, sub in df.groupby("family_label")}
    rng = random.Random(PRIMARY_SEED)
    for indexes in grouped.values():
        rng.shuffle(indexes)
    for split_name, family_counts in TARGET_FAMILY_COUNTS.items():
        for family_label, count in family_counts.items():
            take = grouped[family_label][:count]
            grouped[family_label] = grouped[family_label][count:]
            df.loc[take, "split"] = split_name
    assert Counter(df["split"]) == {"clean_dev": 12, "stress_dev": 8, "test": 32}
    return df.sort_values(["split", "family_label", "pair_id"]).reset_index(drop=True)


def load_pair(pair_row: pd.Series) -> PairData:
    pair_id = pair_row["pair_id"]
    if pair_id in _PAIR_CACHE:
        return _PAIR_CACHE[pair_id]
    base = pair_row["zip_dir"]
    with open_zip() as zf:
        names = zf.namelist()
        source_csv = [n for n in names if n.startswith(base + "/") and n.endswith("_source.csv")][0]
        target_csv = [n for n in names if n.startswith(base + "/") and n.endswith("_target.csv")][0]
        source_json = [n for n in names if n.startswith(base + "/") and n.endswith("_source.json")][0]
        target_json = [n for n in names if n.startswith(base + "/") and n.endswith("_target.json")][0]
        mapping_json = [n for n in names if n.startswith(base + "/") and n.endswith("_mapping.json")][0]
        pair = PairData(
            pair_id=pair_id,
            source_name=Path(source_csv).stem.replace("_source", ""),
            target_name=Path(target_csv).stem.replace("_target", ""),
            family_source=pair_row["source_root"],
            family_label=pair_row["family_label"],
            scenario=pair_row["scenario"],
            source_df=pd.read_csv(io.BytesIO(zf.read(source_csv)), low_memory=False),
            target_df=pd.read_csv(io.BytesIO(zf.read(target_csv)), low_memory=False),
            source_meta=decode_json_bytes(zf.read(source_json)),
            target_meta=decode_json_bytes(zf.read(target_json)),
            gold=parse_mapping(zf.read(mapping_json).decode("utf-8")),
        )
    _PAIR_CACHE[pair_id] = pair
    return pair


def load_gold(pair_row: pd.Series) -> List[Tuple[str, str]]:
    return load_pair(pair_row).gold


def degrade_header(name: str) -> str:
    tokens = split_identifier(name)
    if not tokens:
        return name.lower()
    out = []
    for i, token in enumerate(tokens):
        if len(token) > 6:
            token = token[:3]
        elif len(token) > 4:
            token = token[:-1]
        if i % 2 == 1 and len(tokens) > 1:
            continue
        out.append(token)
    if not out:
        out = [tokens[0][:3]]
    token = out[0]
    if len(token) >= 4:
        token = token[0] + token[2] + token[1] + token[3:]
    elif len(token) >= 3:
        token = token[:-1] + "x"
    out[0] = token
    return "_".join(out)


def stress_column(column_name: str, meta: dict, condition: str) -> Tuple[str, dict]:
    out = dict(meta)
    operations = []
    if condition in {"header-stress", "composite-stress"}:
        out["stressed_name"] = degrade_header(column_name)
        operations.extend(["abbreviate", "token_delete", "identifier_normalize", "typo_inject"])
    else:
        out["stressed_name"] = column_name
    base_description = extract_description(meta)
    if condition in {"context-stress", "composite-stress"}:
        out["description"] = ""
        out["drop_context"] = True
        out["raw_type"] = ""
        operations.extend(["remove_description", "remove_neighbor_context", "remove_table_context", "remove_type_context"])
    else:
        out["description"] = base_description
        out["drop_context"] = False
        out["raw_type"] = meta.get("type", "")
    out["operations"] = operations
    return out["stressed_name"], out


def generate_stress_views(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in inventory.iterrows():
        pair = load_pair(row)
        for condition in CONDITIONS:
            operations = []
            for column_name, meta in pair.source_meta.items():
                _, detail = stress_column(column_name, meta, condition)
                operations.append({"side": "source", "column": column_name, "detail": detail})
            for column_name, meta in pair.target_meta.items():
                _, detail = stress_column(column_name, meta, condition)
                operations.append({"side": "target", "column": column_name, "detail": detail})
            rows.append({"pair_id": pair.pair_id, "split": row["split"], "condition": condition, "operations": operations})
    out_path = OUTPUTS / "data" / "stress_views.jsonl"
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return pd.DataFrame(rows)


def schema_columns(pair: PairData, side: str, condition: str) -> List[dict]:
    meta = pair.source_meta if side == "source" else pair.target_meta
    table_name = pair.source_name if side == "source" else pair.target_name
    columns = []
    for column_name, info in meta.items():
        stressed_name, detail = stress_column(column_name, info, condition)
        columns.append(
            {
                "original_name": column_name,
                "name": stressed_name,
                "type": coarse_type(detail["raw_type"]),
                "raw_type": detail["raw_type"],
                "table": "" if detail["drop_context"] else table_name,
                "description": detail["description"],
            }
        )
    sibling_names = [col["name"] for col in columns]
    for col in columns:
        col["siblings"] = [name for name in sibling_names if name != col["name"]][:3] if condition not in {"context-stress", "composite-stress"} else []
    return columns


def lexical_context_text(col: dict) -> str:
    parts = [col["table"], col["raw_type"], " ".join(col.get("siblings", [])), col["description"]]
    return normalize_ws(" ".join(part for part in parts if part))


def hungarian_with_threshold(scores: np.ndarray, src_cols: List[dict], tgt_cols: List[dict], threshold: float) -> List[dict]:
    if scores.size == 0:
        return []
    row_ix, col_ix = linear_sum_assignment(-scores)
    predictions = []
    for i, j in zip(row_ix, col_ix):
        score = float(scores[i, j])
        row = scores[i]
        margin = float(score - np.partition(row.flatten(), -2)[-2]) if row.size > 1 else float(score)
        predictions.append(
            {
                "source_column": src_cols[i]["original_name"],
                "target_column": tgt_cols[j]["original_name"],
                "score": score,
                "margin": margin,
                "accepted": score >= threshold,
            }
        )
    return predictions


def score_lexical(pair: PairData, condition: str) -> Tuple[np.ndarray, List[dict], List[dict]]:
    src_cols = schema_columns(pair, "source", condition)
    tgt_cols = schema_columns(pair, "target", condition)
    scores = np.zeros((len(src_cols), len(tgt_cols)), dtype=float)
    for i, src in enumerate(src_cols):
        for j, tgt in enumerate(tgt_cols):
            header_score = lexical_similarity(src["name"], tgt["name"])
            src_context = lexical_context_text(src)
            tgt_context = lexical_context_text(tgt)
            if src_context and tgt_context:
                context_score = lexical_similarity(src_context, tgt_context)
                scores[i, j] = 0.7 * header_score + 0.3 * context_score
            else:
                scores[i, j] = header_score
    return scores, src_cols, tgt_cols


def build_graph(cols: List[dict]) -> Tuple[List[dict], Dict[int, List[int]], List[int]]:
    nodes = [{"kind": "table", "name": cols[0]["table"] or "masked_table"}]
    edges: Dict[int, List[int]] = defaultdict(list)
    type_index = {}
    column_indices = []
    for col in cols:
        col_idx = len(nodes)
        nodes.append({"kind": "column", "name": col["name"], "original_name": col["original_name"]})
        column_indices.append(col_idx)
        edges[0].append(col_idx)
        edges[col_idx].append(0)
        col_type = col["type"]
        if col_type not in type_index:
            type_index[col_type] = len(nodes)
            nodes.append({"kind": "type", "name": col_type})
        type_idx = type_index[col_type]
        edges[col_idx].append(type_idx)
        edges[type_idx].append(col_idx)
    return nodes, edges, column_indices


def score_similarity_flooding(pair: PairData, condition: str, iterations: int = 30, damping: float = 0.85) -> Tuple[np.ndarray, List[dict], List[dict]]:
    base_scores, src_cols, tgt_cols = score_lexical(pair, condition)
    g1, e1, col_idx_1 = build_graph(src_cols)
    g2, e2, col_idx_2 = build_graph(tgt_cols)
    sim = np.zeros((len(g1), len(g2)), dtype=float)
    init = np.zeros_like(sim)
    src_lookup = {node_idx: pos for pos, node_idx in enumerate(col_idx_1)}
    tgt_lookup = {node_idx: pos for pos, node_idx in enumerate(col_idx_2)}
    for i, n1 in enumerate(g1):
        for j, n2 in enumerate(g2):
            if n1["kind"] != n2["kind"]:
                continue
            if n1["kind"] == "column":
                init[i, j] = base_scores[src_lookup[i], tgt_lookup[j]]
            elif n1["kind"] == "type":
                init[i, j] = 1.0 if n1["name"] == n2["name"] else 0.0
            else:
                init[i, j] = lexical_similarity(n1["name"], n2["name"])
    sim[:] = init
    for _ in range(iterations):
        nxt = np.zeros_like(sim)
        max_delta = 0.0
        for i in range(len(g1)):
            for j in range(len(g2)):
                if g1[i]["kind"] != g2[j]["kind"]:
                    continue
                neigh = [sim[a, b] for a in e1.get(i, []) for b in e2.get(j, []) if g1[a]["kind"] == g2[b]["kind"]]
                propagated = float(np.mean(neigh)) if neigh else 0.0
                nxt[i, j] = (1.0 - damping) * init[i, j] + damping * propagated
                max_delta = max(max_delta, abs(nxt[i, j] - sim[i, j]))
        sim[:] = nxt
        if max_delta < 1e-4:
            break
    return sim[np.ix_(col_idx_1, col_idx_2)], src_cols, tgt_cols


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(PTLM_MODEL_NAME, device="cpu")
    return _MODEL


def encode_text(col: dict, siblings: List[str], condition: str) -> str:
    table = col["table"] if condition not in {"context-stress", "composite-stress"} else ""
    context = " ".join(siblings[:3]) if condition not in {"context-stress", "composite-stress"} else ""
    description = col["description"] if condition not in {"context-stress", "composite-stress"} else ""
    return PTLM_TEMPLATE.format(
        table=table,
        column=col["name"],
        description=description,
        raw_type=col["raw_type"] if condition not in {"context-stress", "composite-stress"} else "",
        context=context,
    )


def score_ptlm(pair: PairData, condition: str, force_recompute: bool = False) -> Tuple[np.ndarray, List[dict], List[dict]]:
    src_cols = schema_columns(pair, "source", condition)
    tgt_cols = schema_columns(pair, "target", condition)
    cache_path = OUTPUTS / "predictions" / "ptlm" / "embeddings" / f"{pair.pair_id}__{condition}.npz"
    if cache_path.exists() and not force_recompute:
        data = np.load(cache_path)
        emb1 = data["src_embeddings"]
        emb2 = data["tgt_embeddings"]
    else:
        src_text = [encode_text(col, [c["name"] for c in src_cols if c["original_name"] != col["original_name"]], condition) for col in src_cols]
        tgt_text = [encode_text(col, [c["name"] for c in tgt_cols if c["original_name"] != col["original_name"]], condition) for col in tgt_cols]
        model = get_model()
        emb1 = model.encode(src_text, batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        emb2 = model.encode(tgt_text, batch_size=16, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        np.savez_compressed(cache_path, src_embeddings=emb1, tgt_embeddings=emb2)
    scores = 1.0 - cdist(emb1, emb2, metric="cosine")
    return scores, src_cols, tgt_cols


SCORERS = {
    "lexical": score_lexical,
    "similarity_flooding": score_similarity_flooding,
    "ptlm": score_ptlm,
}


def eval_pair(preds: List[dict], gold: List[Tuple[str, str]]) -> Dict[str, float]:
    gold_set = set(gold)
    accepted = {(pred["source_column"], pred["target_column"]) for pred in preds if pred["accepted"]}
    tp = len(accepted & gold_set)
    fp = len(accepted - gold_set)
    fn = len(gold_set - accepted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def tune_threshold(method: str, inventory: pd.DataFrame) -> float:
    dev_rows = inventory[inventory["split"] == "clean_dev"]
    cache = []
    for _, row in dev_rows.iterrows():
        pair = load_pair(row)
        scores, src_cols, tgt_cols = SCORERS[method](pair, "clean")
        cache.append((pair, scores, src_cols, tgt_cols))
    best_threshold = THRESHOLD_GRIDS[method][0]
    best_f1 = -1.0
    for threshold in THRESHOLD_GRIDS[method]:
        f1_values = []
        for pair, scores, src_cols, tgt_cols in cache:
            preds = hungarian_with_threshold(scores, src_cols, tgt_cols, threshold)
            f1_values.append(eval_pair(preds, pair.gold)["f1"])
        mean_f1 = float(np.mean(f1_values))
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = float(threshold)
    return best_threshold


def run_method(
    method: str,
    inventory: pd.DataFrame,
    threshold: float,
    run_seed: int = PRIMARY_SEED,
    repeat_tag: str = "headline",
    subset: pd.DataFrame | None = None,
    persist_outputs: bool = True,
    force_ptlm_recompute: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    set_repeat_seed(run_seed)
    records = []
    runtime_rows = []
    proc = psutil.Process()
    work_inventory = subset if subset is not None else inventory
    for _, row in work_inventory.iterrows():
        pair = load_pair(row)
        gold_set = set(pair.gold)
        for condition in CONDITIONS:
            start = time.time()
            rss_before = proc.memory_info().rss
            if method == "ptlm":
                force_recompute = force_ptlm_recompute or (repeat_tag == "headline" and run_seed == PRIMARY_SEED)
                scores, src_cols, tgt_cols = score_ptlm(pair, condition, force_recompute=force_recompute)
            else:
                scores, src_cols, tgt_cols = SCORERS[method](pair, condition)
            preds = hungarian_with_threshold(scores, src_cols, tgt_cols, threshold)
            elapsed = time.time() - start
            peak_rss = max(rss_before, proc.memory_info().rss) / (1024 * 1024)
            for pred in preds:
                records.append(
                    {
                        "run_seed": run_seed,
                        "repeat_tag": repeat_tag,
                        "method": method,
                        "pair_id": pair.pair_id,
                        "family_label": pair.family_label,
                        "split": row["split"],
                        "condition": condition,
                        "gold": (pred["source_column"], pred["target_column"]) in gold_set,
                        **pred,
                    }
                )
            runtime_rows.append(
                {
                    "run_seed": run_seed,
                    "repeat_tag": repeat_tag,
                    "method": method,
                    "pair_id": pair.pair_id,
                    "split": row["split"],
                    "condition": condition,
                    "wall_time_sec": elapsed,
                    "peak_rss_mb": peak_rss,
                }
            )
    pred_df = pd.DataFrame(records)
    runtime_df = pd.DataFrame(runtime_rows)
    if persist_outputs:
        pred_name = f"{repeat_tag}_seed_{run_seed}.jsonl"
        run_name = f"{repeat_tag}_seed_{run_seed}.csv"
        pred_df.to_json(OUTPUTS / "predictions" / method / pred_name, orient="records", lines=True)
        runtime_df.to_csv(OUTPUTS / "eval" / f"runtime_{method}_{run_name}", index=False)
    return pred_df, runtime_df


def fit_calibrator(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(random_state=0, max_iter=500)
    model.fit(features, labels)
    return model


def tune_selective_threshold(
    df: pd.DataFrame,
    confidence_col: str,
    gold_lookup: Dict[str, List[Tuple[str, str]]],
    coverage_floor: float = 0.70,
) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    confidences = sorted({float(value) for value in df[confidence_col].fillna(0.0).tolist()})
    if not confidences:
        return best_thr
    grid = sorted(set([0.0, 1.0] + confidences + [round(x, 4) for x in np.linspace(0.2, 0.95, 16)]))
    for thr in grid:
        coverage = float((df[confidence_col] >= thr).mean())
        if coverage < coverage_floor:
            continue
        tmp = df.copy()
        tmp["accepted"] = tmp["accepted"] & (tmp[confidence_col] >= thr)
        pair_scores = []
        for pair_id, sub in tmp.groupby("pair_id"):
            pair_scores.append(eval_pair(sub.to_dict("records"), gold_lookup[pair_id])["f1"])
        mean_f1 = float(np.mean(pair_scores)) if pair_scores else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thr = float(thr)
    return best_thr


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    splits = np.array_split(np.arange(len(y_true)), bins)
    total = len(y_true)
    ece = 0.0
    for idx in splits:
        if len(idx) == 0:
            continue
        ece += (len(idx) / total) * abs(float(np.mean(y_true[idx])) - float(np.mean(y_prob[idx])))
    return ece


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_true - y_prob) ** 2))


def bootstrap_ci(df: pd.DataFrame, value_col: str, seed: int = PRIMARY_SEED, n_boot: int = BOOTSTRAP_RESAMPLES) -> Tuple[float, float]:
    if df.empty:
        return 0.0, 0.0
    values = df[value_col].values
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        samples.append(float(np.mean(values[idx])))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def build_gold_lookup(inventory: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    return {row["pair_id"]: load_gold(row) for _, row in inventory.iterrows()}


def pair_metrics_from_df(
    df: pd.DataFrame,
    inventory: pd.DataFrame,
    gold_lookup: Dict[str, List[Tuple[str, str]]],
    accepted_col: str = "accepted",
) -> pd.DataFrame:
    rows = []
    for (run_seed, method, condition, pair_id), sub in df.groupby(["run_seed", "method", "condition", "pair_id"]):
        gold = gold_lookup[pair_id]
        preds = sub.copy()
        preds["accepted"] = preds[accepted_col]
        metrics = eval_pair(preds.to_dict("records"), gold)
        meta = inventory[inventory["pair_id"] == pair_id].iloc[0]
        rows.append(
            {
                "run_seed": run_seed,
                "method": method,
                "condition": condition,
                "pair_id": pair_id,
                "family_label": meta["family_label"],
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def selective_metrics(df: pd.DataFrame, inventory: pd.DataFrame, gold_lookup: Dict[str, List[Tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for (run_seed, method, condition, pair_id), sub in df.groupby(["run_seed", "method", "condition", "pair_id"]):
        gold = gold_lookup[pair_id]
        kept = int(sub["selective_accept"].sum())
        accepted = int(sub["accepted"].sum())
        tmp = sub.copy()
        tmp["accepted"] = tmp["selective_accept"]
        metrics = eval_pair(tmp.to_dict("records"), gold)
        rows.append(
            {
                "run_seed": run_seed,
                "method": method,
                "condition": condition,
                "pair_id": pair_id,
                "coverage": kept / accepted if accepted else 0.0,
                "selective_precision": metrics["precision"],
                "selective_recall": metrics["recall"],
                "selective_f1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)


def aurc_for_group(sub: pd.DataFrame) -> float:
    if sub.empty:
        return 0.0
    work = sub.sort_values("calibrated_prob", ascending=False).reset_index(drop=True)
    coverages = []
    risks = []
    total = len(work)
    for k in range(1, total + 1):
        kept = work.iloc[:k]
        coverages.append(k / total)
        risks.append(1.0 - float(kept["gold"].mean()))
    return float(np.trapz(risks, coverages))


def dataset_summary(inventory: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    desc_pairs = 0
    for _, row in inventory.iterrows():
        pair = load_pair(row)
        src_desc = sum(1 for meta in pair.source_meta.values() if extract_description(meta))
        tgt_desc = sum(1 for meta in pair.target_meta.values() if extract_description(meta))
        pair_has_desc = int(src_desc + tgt_desc > 0)
        desc_pairs += pair_has_desc
        rows.append(
            {
                "pair_id": pair.pair_id,
                "family_source": pair.family_source,
                "family_label": pair.family_label,
                "scenario": pair.scenario,
                "split": row["split"],
                "source_columns": len(pair.source_meta),
                "target_columns": len(pair.target_meta),
                "candidate_pairs": len(pair.source_meta) * len(pair.target_meta),
                "source_descriptions": src_desc,
                "target_descriptions": tgt_desc,
                "descriptions_available": pair_has_desc,
                "table_context_available": 1,
            }
        )
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(OUTPUTS / "data" / "pair_inventory.csv", index=False)
    summary_df = pair_df.groupby(["family_label", "split"]).agg(
        pair_count=("pair_id", "count"),
        mean_source_columns=("source_columns", "mean"),
        median_source_columns=("source_columns", "median"),
        mean_target_columns=("target_columns", "mean"),
        median_target_columns=("target_columns", "median"),
        mean_candidate_pairs=("candidate_pairs", "mean"),
        metadata_completeness=("descriptions_available", "mean"),
    ).reset_index()
    summary_df.to_csv(OUTPUTS / "eval" / "dataset_summary.csv", index=False)
    summary_df.to_csv(OUTPUTS / "eval" / "table1.csv", index=False)
    context_note = {
        "pair_count_with_descriptions": int(desc_pairs),
        "pair_count_without_descriptions": int(len(pair_df) - desc_pairs),
        "faithful_context_slice": bool(desc_pairs > 0),
        "note": "No selected pair exposes non-empty schema descriptions in the local Valentine archive, so the study is recast around shared structural-context stress: table-name, sibling-context, and type-context masking across all compared methods."
        if desc_pairs == 0
        else "At least one selected pair exposes schema descriptions.",
    }
    return pair_df, summary_df, context_note


def save_run_metadata() -> dict:
    actual_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    payload = {
        "cpu_model": platform.processor() or platform.machine(),
        "actual_ram_gb": actual_ram_gb,
        "claimed_feasible_ram_gb": CLAIMED_RAM_GB,
        "claimed_feasible_cpu_cores": CLAIMED_CPU_CORES,
        "os": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": {name: safe_pkg_version(name) for name in PKG_NAMES},
        "git_commit": get_git_commit(),
        "thread_caps": THREAD_CAPS,
        "seed_policy": {
            "headline_protocol_seed": PRIMARY_SEED,
            "headline_split_frozen_once": True,
            "seed_usage": "Seed 17 defines the headline deterministic protocol. Auxiliary sensitivity repeats over seeds 17, 23, and 42 are reported separately to satisfy multi-seed reporting without replacing the frozen headline run.",
            "auxiliary_reported_seeds": SEEDS,
        },
        "declared_budget_hours": 8,
        "stop_conditions": {
            "appendix_cancel_if_projected_hours_gt": 7.5,
            "appendix_allowed_if_main_hours_lt": 6.5,
            "infeasible_if_main_hours_gt": 8.0,
        },
        "gpu_available": 0,
    }
    write_json(OUTPUTS / "eval" / "run_metadata.json", payload)
    return payload


def run_environment_setup_experiment() -> dict:
    ensure_dirs()
    set_env()
    metadata = save_run_metadata()
    config = {
        "python_target": "3.11.x",
        "python_actual": platform.python_version(),
        "thread_caps": THREAD_CAPS,
        "cpu_only": True,
        "gpu_available": 0,
    }
    results = {
        "experiment": "environment_setup",
        "python_executable": metadata["python_executable"],
        "python_version": metadata["python_version"],
        "git_commit": metadata["git_commit"],
        "package_versions": metadata["package_versions"],
        "thread_caps": metadata["thread_caps"],
        "claimed_feasible_ram_gb": CLAIMED_RAM_GB,
        "claimed_feasible_cpu_cores": CLAIMED_CPU_CORES,
        "actual_ram_gb": metadata["actual_ram_gb"],
    }
    write_experiment_artifacts(
        "environment_setup",
        config,
        results,
        [
            f"Python executable: {metadata['python_executable']}",
            f"Python version: {metadata['python_version']}",
            f"Git commit: {metadata['git_commit']}",
            "Recorded package versions and thread caps.",
        ],
    )
    return results


def run_data_inventory_experiment(inventory: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pair_df, summary_df, context_note = dataset_summary(inventory)
    config = {
        "family_specs": FAMILY_SPECS,
        "split_counts": TARGET_FAMILY_COUNTS,
        "primary_seed": PRIMARY_SEED,
    }
    results = {
        "experiment": "data_inventory",
        "pair_count": int(len(pair_df)),
        "split_counts": pair_df["split"].value_counts().sort_index().to_dict(),
        "family_counts": pair_df["family_label"].value_counts().sort_index().to_dict(),
        "context_note": context_note,
    }
    write_experiment_artifacts(
        "data_inventory",
        config,
        results,
        [
            f"Frozen inventory size: {len(pair_df)} pairs.",
            f"Split counts: {results['split_counts']}",
            context_note["note"],
        ],
    )
    return pair_df, summary_df, results


def run_stress_views_experiment(inventory: pd.DataFrame) -> dict:
    views = generate_stress_views(inventory)
    description_masked_pairs = 0
    for _, row in inventory.iterrows():
        pair = load_pair(row)
        if any(extract_description(meta) for meta in pair.source_meta.values()) or any(
            extract_description(meta) for meta in pair.target_meta.values()
        ):
            description_masked_pairs += 1
    config = {"conditions": CONDITIONS}
    results = {
        "experiment": "stress_views",
        "rows_written": int(len(views)),
        "output_path": str(OUTPUTS / "data" / "stress_views.jsonl"),
        "context_stress_scope": {
            "description_masked_pairs": description_masked_pairs,
            "table_and_sibling_mask_applies_to_all_pairs": True,
            "type_mask_applies_to_all_pairs": True,
        },
    }
    write_experiment_artifacts(
        "stress_views",
        config,
        results,
        [
            f"Generated {len(views)} deterministic stress-view rows.",
            f"Pairs with usable descriptions for context masking: {description_masked_pairs}",
            "Context-stress is recast as shared table/sibling/type-context masking when descriptions are absent.",
        ],
    )
    return results


def summarize_method_predictions(pred_df: pd.DataFrame, inventory: pd.DataFrame, run_seed: int) -> dict:
    gold_lookup = build_gold_lookup(inventory)
    test_metrics = pair_metrics_from_df(pred_df[pred_df["split"] == "test"], inventory, gold_lookup)
    overall = (
        test_metrics.groupby("condition")[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
        .to_dict("records")
    )
    return {
        "run_seed": run_seed,
        "test_pair_count": int(test_metrics["pair_id"].nunique()),
        "condition_metrics": overall,
    }


def run_method_experiment(method: str, inventory: pd.DataFrame, threshold: float, run_seed: int = PRIMARY_SEED, repeat_tag: str = "headline") -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred_df, runtime_df = run_method(method, inventory, threshold, run_seed=run_seed, repeat_tag=repeat_tag, persist_outputs=True)
    summary = summarize_method_predictions(pred_df, inventory, run_seed)
    config = {
        "method": method,
        "threshold": threshold,
        "run_seed": run_seed,
        "repeat_tag": repeat_tag,
        "conditions": CONDITIONS,
        "lexical_recipe": LEXICAL_RECIPE if method == "lexical" else None,
        "similarity_flooding": SIMILARITY_FLOODING_CONFIG if method == "similarity_flooding" else None,
        "ptlm_model_name": PTLM_MODEL_NAME if method == "ptlm" else None,
        "ptlm_template": PTLM_TEMPLATE if method == "ptlm" else None,
    }
    results = {
        "experiment": method,
        "method": method,
        "threshold": threshold,
        "run_seed": run_seed,
        "prediction_rows": int(len(pred_df)),
        "runtime_rows": int(len(runtime_df)),
        "runtime_total_sec": float(runtime_df["wall_time_sec"].sum()),
        "runtime_peak_rss_mb": float(runtime_df["peak_rss_mb"].max()),
        "test_summary": summary,
        "prediction_path": str(OUTPUTS / "predictions" / method / f"{repeat_tag}_seed_{run_seed}.jsonl"),
        "runtime_path": str(OUTPUTS / "eval" / f"runtime_{method}_{repeat_tag}_seed_{run_seed}.csv"),
    }
    if method == "ptlm":
        results["embedding_cache_dir"] = str(OUTPUTS / "predictions" / "ptlm" / "embeddings")
    write_experiment_artifacts(
        method,
        config,
        results,
        [
            f"Method: {method}",
            f"Threshold: {threshold}",
            f"Prediction rows: {len(pred_df)}",
            f"Runtime total sec: {results['runtime_total_sec']:.3f}",
        ],
    )
    return pred_df, runtime_df, results


def fit_clean_calibrators(
    raw_preds: pd.DataFrame,
    inventory: pd.DataFrame,
    acceptance_thresholds: Dict[str, float],
    transfer_output_path: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Dict[str, LogisticRegression]]:
    gold_lookup = build_gold_lookup(inventory)
    calibrated_frames = []
    transfer_rows = []
    summary = {}
    models = {}
    for method in METHODS:
        method_df = raw_preds[raw_preds["method"] == method].copy()
        clean_dev = method_df[(method_df["split"] == "clean_dev") & (method_df["condition"] == "clean")]
        calibrator = fit_calibrator(clean_dev[["score", "margin"]].fillna(0.0).values, clean_dev["gold"].astype(int).values)
        method_df["calibrated_prob"] = calibrator.predict_proba(method_df[["score", "margin"]].fillna(0.0).values)[:, 1]
        clean_dev_cal = method_df.loc[clean_dev.index].copy()
        best_thr = tune_selective_threshold(clean_dev_cal, "calibrated_prob", gold_lookup, coverage_floor=0.70)
        method_df["selective_accept"] = method_df["accepted"] & (method_df["calibrated_prob"] >= best_thr)
        for split_name, split_df in [("clean_dev", method_df[method_df["split"] == "clean_dev"]), ("stress_dev", method_df[method_df["split"] == "stress_dev"])]:
            for condition, sub in split_df.groupby("condition"):
                transfer_rows.append(
                    {
                        "method": method,
                        "split": split_name,
                        "condition": condition,
                        "ece": ece_score(sub["gold"].astype(int).values, sub["calibrated_prob"].values),
                        "brier": brier_score(sub["gold"].astype(int).values, sub["calibrated_prob"].values),
                    }
                )
        summary[method] = {
            "acceptance_threshold": float(acceptance_thresholds[method]),
            "abstention_threshold": best_thr,
            "calibrator_coefficients": calibrator.coef_.tolist(),
            "calibrator_intercept": calibrator.intercept_.tolist(),
        }
        models[method] = calibrator
        calibrated_frames.append(method_df)
    transfer_df = pd.DataFrame(transfer_rows)
    if transfer_output_path is not None:
        transfer_df.to_csv(transfer_output_path, index=False)
    return pd.concat(calibrated_frames, ignore_index=True), transfer_df, summary, models


def apply_calibration(raw_preds: pd.DataFrame, models: Dict[str, LogisticRegression], summary: dict) -> pd.DataFrame:
    frames = []
    for method in METHODS:
        sub = raw_preds[raw_preds["method"] == method].copy()
        sub["calibrated_prob"] = models[method].predict_proba(sub[["score", "margin"]].fillna(0.0).values)[:, 1]
        sub["selective_accept"] = sub["accepted"] & (sub["calibrated_prob"] >= summary[method]["abstention_threshold"])
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def run_calibration_experiment(
    raw_preds: pd.DataFrame,
    inventory: pd.DataFrame,
    acceptance_thresholds: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, Dict[str, LogisticRegression]]:
    transfer_output_path = OUTPUTS / "eval" / "headline_stressed_dev_transfer.csv"
    calibrated_df, transfer_df, summary, models = fit_clean_calibrators(
        raw_preds,
        inventory,
        acceptance_thresholds,
        transfer_output_path=transfer_output_path,
    )
    calibrated_df.to_csv(OUTPUTS / "eval" / "headline_calibrated_predictions.csv", index=False)
    config = {
        "fit_split": "clean_dev",
        "transfer_split": "stress_dev",
        "coverage_constraint": 0.70,
        "feature_columns": ["score", "margin"],
        "calibration_model": "sklearn.linear_model.LogisticRegression(max_iter=500, random_state=0)",
    }
    results = {
        "experiment": "calibration",
        "headline_seed": PRIMARY_SEED,
        "calibration_summary": summary,
        "transfer_rows": transfer_df.to_dict("records"),
        "calibrated_prediction_rows": int(len(calibrated_df)),
    }
    write_experiment_artifacts(
        "calibration",
        config,
        results,
        [
            "Fitted one logistic calibrator per method on clean_dev only.",
            f"Saved immutable headline transfer metrics to {transfer_output_path.name}.",
            "Saved calibrated headline predictions.",
        ],
    )
    return calibrated_df, transfer_df, results, models


def compute_headline_tables(inventory: pd.DataFrame, raw_preds: pd.DataFrame, calibrated_preds: pd.DataFrame, runtimes: pd.DataFrame) -> dict:
    gold_lookup = build_gold_lookup(inventory)
    headline_raw = raw_preds[(raw_preds["run_seed"] == PRIMARY_SEED) & (raw_preds["split"] == "test")].copy()
    headline_cal = calibrated_preds[(calibrated_preds["run_seed"] == PRIMARY_SEED) & (calibrated_preds["split"] == "test")].copy()
    pair_metrics = pair_metrics_from_df(headline_cal, inventory, gold_lookup)
    raw_pair_metrics = pair_metrics_from_df(headline_raw, inventory, gold_lookup)
    selective = selective_metrics(headline_cal, inventory, gold_lookup)
    pair_metrics.to_csv(OUTPUTS / "eval" / "pair_metrics.csv", index=False)
    selective.to_csv(OUTPUTS / "eval" / "selective_metrics.csv", index=False)

    summary_rows = []
    family_rows = []
    for (method, condition), sub in pair_metrics.groupby(["method", "condition"]):
        ci_low, ci_high = bootstrap_ci(sub, "f1")
        summary_rows.append(
            {
                "method": method,
                "condition": condition,
                "precision": float(sub["precision"].mean()),
                "recall": float(sub["recall"].mean()),
                "f1": float(sub["f1"].mean()),
                "f1_ci_low": ci_low,
                "f1_ci_high": ci_high,
                "n_pairs": int(sub["pair_id"].nunique()),
            }
        )
        family_agg = sub.groupby("family_label")[["precision", "recall", "f1"]].mean().reset_index()
        for _, fam_row in family_agg.iterrows():
            family_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "family_label": fam_row["family_label"],
                    "precision": float(fam_row["precision"]),
                    "recall": float(fam_row["recall"]),
                    "f1": float(fam_row["f1"]),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    family_df = pd.DataFrame(family_rows)
    summary_df.to_csv(OUTPUTS / "eval" / "summary_metrics.csv", index=False)
    family_df.to_csv(OUTPUTS / "eval" / "family_metrics.csv", index=False)

    ranking_rows = []
    clean_rank = summary_df[summary_df["condition"] == "clean"].sort_values("f1", ascending=False)["method"].tolist()
    for condition in [cond for cond in CONDITIONS if cond != "clean"]:
        current_rank = summary_df[summary_df["condition"] == condition].sort_values("f1", ascending=False)["method"].tolist()
        shifts = {method: abs(clean_rank.index(method) - current_rank.index(method)) for method in METHODS}
        inversions = 0
        for i in range(len(METHODS)):
            for j in range(i + 1, len(METHODS)):
                left = METHODS[i]
                right = METHODS[j]
                inversions += int((clean_rank.index(left) - clean_rank.index(right)) * (current_rank.index(left) - current_rank.index(right)) < 0)
        tau = float(kendalltau([clean_rank.index(method) for method in METHODS], [current_rank.index(method) for method in METHODS]).correlation)
        ranking_rows.append(
            {
                "condition": condition,
                "kendall_tau": tau,
                "pairwise_inversions": inversions,
                "mean_rank_shift": float(np.mean(list(shifts.values()))),
            }
        )
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(OUTPUTS / "eval" / "ranking_stability.csv", index=False)

    calibration_rows = []
    for (method, condition), sub in headline_cal.groupby(["method", "condition"]):
        accepted = sub[sub["accepted"]]
        calibration_rows.append(
            {
                "method": method,
                "condition": condition,
                "ece": ece_score(sub["gold"].astype(int).values, sub["calibrated_prob"].values),
                "brier": brier_score(sub["gold"].astype(int).values, sub["calibrated_prob"].values),
                "aurc": aurc_for_group(accepted),
            }
        )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(OUTPUTS / "eval" / "calibration_metrics.csv", index=False)

    selective_summary = (
        selective.groupby(["method", "condition"])[["coverage", "selective_precision", "selective_recall", "selective_f1"]]
        .mean()
        .reset_index()
    )
    selective_summary.to_csv(OUTPUTS / "eval" / "selective_summary.csv", index=False)

    runtime_summary = (
        runtimes[(runtimes["run_seed"] == PRIMARY_SEED) & (runtimes["repeat_tag"] == "headline")]
        .groupby("method")
        .agg(total_wall_time_sec=("wall_time_sec", "sum"), mean_wall_time_sec=("wall_time_sec", "mean"), peak_rss_mb=("peak_rss_mb", "max"))
        .reset_index()
    )
    runtime_summary["feasible_under_128gb"] = runtime_summary["peak_rss_mb"] <= (CLAIMED_RAM_GB * 1024)
    runtime_summary.to_csv(OUTPUTS / "eval" / "runtime_summary.csv", index=False)

    return {
        "summary_df": summary_df,
        "family_df": family_df,
        "ranking_df": ranking_df,
        "calibration_df": calibration_df,
        "selective_df": selective_summary,
        "runtime_df": runtime_summary,
        "pair_metrics_df": pair_metrics,
        "raw_pair_metrics_df": raw_pair_metrics,
        "headline_raw_df": headline_raw,
        "headline_cal_df": headline_cal,
    }


def run_ablations_experiment(
    headline_tables: dict,
    inventory: pd.DataFrame,
    raw_preds: pd.DataFrame,
    calibrated_preds: pd.DataFrame,
) -> dict:
    gold_lookup = build_gold_lookup(inventory)
    headline_raw = headline_tables["headline_raw_df"].copy()
    headline_cal = headline_tables["headline_cal_df"].copy()

    no_calibration_rows = []
    for method in METHODS:
        method_raw = headline_raw[headline_raw["method"] == method].copy()
        method_raw["raw_confidence"] = method_raw["score"].clip(0.0, 1.0)
        clean_dev_like = raw_preds[
            (raw_preds["method"] == method) & (raw_preds["split"] == "clean_dev") & (raw_preds["condition"] == "clean")
        ].copy()
        clean_dev_like["raw_confidence"] = clean_dev_like["score"].clip(0.0, 1.0)
        raw_thr = tune_selective_threshold(clean_dev_like, "raw_confidence", gold_lookup, coverage_floor=0.70)
        method_raw["raw_selective_accept"] = method_raw["accepted"] & (method_raw["raw_confidence"] >= raw_thr)
        for condition, sub in method_raw.groupby("condition"):
            accepted_for_aurc = sub[sub["accepted"]].copy()
            accepted_for_aurc["calibrated_prob"] = accepted_for_aurc["raw_confidence"]
            selective = selective_metrics(
                sub.assign(selective_accept=sub["raw_selective_accept"]),
                inventory,
                gold_lookup,
            )
            no_calibration_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "raw_abstention_threshold": raw_thr,
                    "ece": ece_score(sub["gold"].astype(int).values, sub["raw_confidence"].values),
                    "brier": brier_score(sub["gold"].astype(int).values, sub["raw_confidence"].values),
                    "aurc": aurc_for_group(accepted_for_aurc),
                    "coverage": float(selective["coverage"].mean()),
                    "selective_precision": float(selective["selective_precision"].mean()),
                    "selective_recall": float(selective["selective_recall"].mean()),
                    "selective_f1": float(selective["selective_f1"].mean()),
                }
            )

    no_abstention = (
        calibrated_preds[(calibrated_preds["run_seed"] == PRIMARY_SEED) & (calibrated_preds["split"] == "test")]
        .groupby(["method", "condition"])
        .apply(
            lambda sub: pd.Series(
                {
                    "coverage": 1.0,
                    "accepted_edge_accuracy": float(sub[sub["accepted"]]["gold"].mean()) if int(sub["accepted"].sum()) else 0.0,
                    "accepted_matches": int(sub["accepted"].sum()),
                }
            )
        )
        .reset_index()
    )
    no_abstention_pair = pair_metrics_from_df(headline_cal, inventory, gold_lookup, accepted_col="accepted")
    no_abstention = no_abstention.merge(
        no_abstention_pair.groupby(["method", "condition"])[["precision", "recall", "f1"]].mean().reset_index(),
        on=["method", "condition"],
        how="left",
    )
    no_composite_df = headline_tables["summary_df"][headline_tables["summary_df"]["condition"] != "composite-stress"].copy()
    no_composite_df.to_csv(OUTPUTS / "eval" / "ablation_c_no_composite.csv", index=False)
    results = {
        "experiment": "ablations",
        "no_calibration": no_calibration_rows,
        "no_abstention": no_abstention.to_dict("records"),
        "no_composite_summary": no_composite_df.to_dict("records"),
        "ablation_c_path": str(OUTPUTS / "eval" / "ablation_c_no_composite.csv"),
    }
    write_experiment_artifacts(
        "ablations",
        {"headline_seed": PRIMARY_SEED},
        results,
        [
            "Computed calibration-sensitive and abstention-sensitive ablations from cached headline outputs.",
            "Ablation A evaluates raw-score calibration and raw-score selective prediction.",
            "Ablation B disables selective abstention and reports full-acceptance metrics.",
            "Ablation C is saved as a distinct no-composite summary artifact.",
        ],
    )
    return results


def save_figures(headline_tables: dict, calibrated_preds: pd.DataFrame) -> dict:
    sns.set_theme(style="whitegrid")
    fig_paths = []
    method_labels = {
        "lexical": "Lexical",
        "similarity_flooding": "Similarity Flooding",
        "ptlm": "PTLM",
    }
    family_labels = {
        "all": "Overall test set",
        "context_limited_chembl": "ChEMBL slice",
        "header_poor": "Header-poor slice",
        "valentine_standard": "Valentine slice",
    }
    condition_labels = {
        "clean": "Clean",
        "header-stress": "Header",
        "context-stress": "Context",
        "composite-stress": "Composite",
    }
    palette = {
        "lexical": "#2563eb",
        "similarity_flooding": "#d97706",
        "ptlm": "#15803d",
    }

    plt.figure(figsize=(10, 4))
    plt.axis("off")
    boxes = [
        (0.06, 0.56, "52 pairs\n12 clean dev / 8 stress dev / 32 test"),
        (0.31, 0.56, "4 conditions\nclean / header / context / composite"),
        (0.56, 0.56, "3 methods\nlexical / SF / PTLM"),
        (0.82, 0.56, "clean-fit calibration\nfrozen abstention"),
        (0.43, 0.18, "held-out test metrics\nF1 / ranks / calibration / coverage"),
    ]
    for x, y, text in boxes:
        plt.text(x, y, text, ha="center", va="center", fontsize=12, bbox={"boxstyle": "round,pad=0.4", "fc": "#f4f4f5", "ec": "#1d4ed8"})
    arrows = [((0.15, 0.48), (0.25, 0.48)), ((0.40, 0.48), (0.50, 0.48)), ((0.65, 0.48), (0.76, 0.48)), ((0.82, 0.45), (0.50, 0.24))]
    for start, end in arrows:
        plt.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#1d4ed8"))
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        for target in [FIGURES / f"figure1_protocol.{ext}", OUTPUTS / "figures" / f"figure1_protocol.{ext}"]:
            plt.savefig(target)
            fig_paths.append(str(target))
    plt.close()

    summary_df = headline_tables["summary_df"].copy()
    family_df = headline_tables["family_df"].copy()
    x = np.arange(len(CONDITIONS))
    width = 0.24
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharey=True, constrained_layout=True)
    panel_specs = [
        ("all", axes[0, 0]),
        ("valentine_standard", axes[0, 1]),
        ("header_poor", axes[1, 0]),
        ("context_limited_chembl", axes[1, 1]),
    ]
    for family_key, ax in panel_specs:
        for idx, method in enumerate(METHODS):
            if family_key == "all":
                sub = summary_df[summary_df["method"] == method].set_index("condition").loc[CONDITIONS].reset_index()
                means = sub["f1"].values
                lower = means - sub["f1_ci_low"].values
                upper = sub["f1_ci_high"].values - means
            else:
                sub = family_df[(family_df["method"] == method) & (family_df["family_label"] == family_key)].set_index("condition").loc[CONDITIONS].reset_index()
                means = sub["f1"].values
                lower = None
                upper = None
            centers = x + (idx - 1) * width
            ax.bar(centers, means, width=width, color=palette[method], label=method_labels[method])
            if lower is not None and upper is not None:
                ax.errorbar(centers, means, yerr=[lower, upper], fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_title(family_labels[family_key], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([condition_labels[c] for c in CONDITIONS], rotation=18)
        ax.set_ylim(0.45, 1.03)
        ax.set_ylabel("Pair-level F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    for ext in ["png", "pdf"]:
        for target in [FIGURES / f"figure2_f1.{ext}", OUTPUTS / "figures" / f"figure2_f1.{ext}"]:
            fig.savefig(target, bbox_inches="tight")
            fig_paths.append(str(target))
    plt.close(fig)

    rank_df = headline_tables["ranking_df"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), constrained_layout=True)
    rank_df = rank_df.copy()
    rank_df["condition_label"] = rank_df["condition"].map(condition_labels)
    metric_specs = [
        ("kendall_tau", "Kendall's tau", (0.8, 1.02), "#2563eb"),
        ("mean_rank_shift", "Mean rank shift", (0.0, 0.5), "#d97706"),
        ("pairwise_inversions", "Pairwise inversions", (0.0, 3.0), "#15803d"),
    ]
    for ax, (metric, ylabel, ylim, color) in zip(axes, metric_specs):
        bars = ax.bar(rank_df["condition_label"], rank_df[metric], color=color, width=0.65)
        ax.set_title(ylabel, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, rank_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + (ylim[1] - ylim[0]) * 0.03, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for ext in ["png", "pdf"]:
        for target in [FIGURES / f"figure3_ranking.{ext}", OUTPUTS / "figures" / f"figure3_ranking.{ext}"]:
            fig.savefig(target, bbox_inches="tight")
            fig_paths.append(str(target))
    plt.close(fig)

    rel_rows = []
    test_df = calibrated_preds[(calibrated_preds["run_seed"] == PRIMARY_SEED) & (calibrated_preds["split"] == "test")]
    for (method, condition), sub in test_df.groupby(["method", "condition"]):
        order = np.argsort(sub["calibrated_prob"].values)
        bins = np.array_split(order, 10)
        for idx, positions in enumerate(bins):
            if len(positions) == 0:
                continue
            rel_rows.append(
                {
                    "method": method,
                    "condition": condition,
                    "bin": idx + 1,
                    "confidence": float(sub.iloc[positions]["calibrated_prob"].mean()),
                    "accuracy": float(sub.iloc[positions]["gold"].mean()),
                }
            )
    rel_df = pd.DataFrame(rel_rows)
    fig, axes = plt.subplots(len(METHODS), len(CONDITIONS), figsize=(11.2, 7.4), sharex=True, sharey=True, constrained_layout=True)
    for row_idx, method in enumerate(METHODS):
        for col_idx, condition in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]
            sub = rel_df[(rel_df["method"] == method) & (rel_df["condition"] == condition)].sort_values("confidence")
            ax.plot(sub["confidence"], sub["accuracy"], marker="o", color=palette[method], lw=1.6)
            ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
            if row_idx == 0:
                ax.set_title(condition_labels[condition], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{method_labels[method]}\nAccuracy")
            if row_idx == len(METHODS) - 1:
                ax.set_xlabel("Confidence")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
    for ext in ["png", "pdf"]:
        for target in [FIGURES / f"figure4_reliability.{ext}", OUTPUTS / "figures" / f"figure4_reliability.{ext}"]:
            fig.savefig(target, bbox_inches="tight")
            fig_paths.append(str(target))
    plt.close(fig)

    rc_rows = []
    accepted_df = test_df[test_df["accepted"]].copy()
    for (method, condition), sub in accepted_df.groupby(["method", "condition"]):
        sub = sub.sort_values("calibrated_prob", ascending=False).reset_index(drop=True)
        total = len(sub)
        for k in range(1, total + 1):
            kept = sub.iloc[:k]
            rc_rows.append({"method": method, "condition": condition, "coverage": k / total, "risk": 1.0 - float(kept["gold"].mean())})
    rc_df = pd.DataFrame(rc_rows)
    fig, axes = plt.subplots(len(METHODS), len(CONDITIONS), figsize=(11.2, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for row_idx, method in enumerate(METHODS):
        for col_idx, condition in enumerate(CONDITIONS):
            ax = axes[row_idx, col_idx]
            sub = rc_df[(rc_df["method"] == method) & (rc_df["condition"] == condition)].sort_values("coverage")
            ax.plot(sub["coverage"], sub["risk"], color=palette[method], lw=1.7)
            if row_idx == 0:
                ax.set_title(condition_labels[condition], fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{method_labels[method]}\nRisk")
            if row_idx == len(METHODS) - 1:
                ax.set_xlabel("Coverage")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.01, 0.36)
    for ext in ["png", "pdf"]:
        for target in [FIGURES / f"figure5_risk_coverage.{ext}", OUTPUTS / "figures" / f"figure5_risk_coverage.{ext}"]:
            fig.savefig(target, bbox_inches="tight")
            fig_paths.append(str(target))
    plt.close(fig)

    results = {"experiment": "visualization", "figure_count": len(fig_paths), "figures": fig_paths}
    write_experiment_artifacts(
        "visualization",
        {"conditions": CONDITIONS, "methods": METHODS},
        results,
        [f"Saved {len(fig_paths)} figure files across root and outputs/figures."],
    )
    return results


def run_runtime_pilot(inventory: pd.DataFrame, thresholds: Dict[str, float]) -> dict:
    pilot_subset = inventory[inventory["split"] == "test"].head(8).copy()
    pilot_frames = []
    start = time.time()
    for method in METHODS:
        _, runtime_df = run_method(
            method,
            inventory,
            thresholds[method],
            run_seed=PRIMARY_SEED,
            repeat_tag="pilot",
            subset=pilot_subset,
            persist_outputs=False,
            force_ptlm_recompute=(method == "ptlm"),
        )
        pilot_frames.append(runtime_df)
    pilot_runtime = pd.concat(pilot_frames, ignore_index=True)
    pilot_elapsed_sec = time.time() - start
    projected_hours = pilot_runtime["wall_time_sec"].sum() * (len(inventory) / len(pilot_subset)) / 3600.0
    payload = {
        "pilot_test_pairs": int(len(pilot_subset)),
        "pilot_elapsed_sec": pilot_elapsed_sec,
        "pilot_total_method_condition_sec": float(pilot_runtime["wall_time_sec"].sum()),
        "projected_main_matrix_hours": projected_hours,
        "appendix_cancelled": bool(projected_hours > 7.5),
    }
    write_json(OUTPUTS / "eval" / "runtime_projection.json", payload)
    return payload


def run_seed_sensitivity() -> dict:
    summary_rows = []
    for seed in SEEDS:
        inventory = split_inventory(freeze_inventory(seed))
        thresholds = {method: tune_threshold(method, inventory) for method in METHODS}
        raw_frames = []
        for method in METHODS:
            pred_df, _ = run_method(
                method,
                inventory,
                thresholds[method],
                run_seed=seed,
                repeat_tag="auxiliary_seed_sensitivity",
                persist_outputs=False,
                force_ptlm_recompute=(method == "ptlm"),
            )
            raw_frames.append(pred_df)
        raw_preds = pd.concat(raw_frames, ignore_index=True)
        aux_transfer_path = OUTPUTS / "eval" / f"auxiliary_seed_{seed}_stressed_dev_transfer.csv"
        calibrated_df, _, _, _ = fit_clean_calibrators(
            raw_preds,
            inventory,
            thresholds,
            transfer_output_path=aux_transfer_path,
        )
        gold_lookup = build_gold_lookup(inventory)
        pair_metrics = pair_metrics_from_df(
            calibrated_df[(calibrated_df["run_seed"] == seed) & (calibrated_df["split"] == "test")],
            inventory,
            gold_lookup,
        )
        for (method, condition), sub in pair_metrics.groupby(["method", "condition"]):
            summary_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "condition": condition,
                    "precision": float(sub["precision"].mean()),
                    "recall": float(sub["recall"].mean()),
                    "f1": float(sub["f1"].mean()),
                }
            )
    seed_df = pd.DataFrame(summary_rows)
    seed_df.to_csv(OUTPUTS / "eval" / "seed_sensitivity_summary.csv", index=False)
    agg = (
        seed_df.groupby(["method", "condition"])[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "method",
        "condition",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]
    agg.to_csv(OUTPUTS / "eval" / "seed_sensitivity_aggregate.csv", index=False)
    return {
        "seeds": SEEDS,
        "per_seed_path": str(OUTPUTS / "eval" / "seed_sensitivity_summary.csv"),
        "aggregate_path": str(OUTPUTS / "eval" / "seed_sensitivity_aggregate.csv"),
        "transfer_paths": [str(OUTPUTS / "eval" / f"auxiliary_seed_{seed}_stressed_dev_transfer.csv") for seed in SEEDS],
        "aggregate": agg.to_dict("records"),
    }


def build_root_results(
    headline_tables: dict,
    thresholds: Dict[str, float],
    calibration_results: dict,
    ablations: dict,
    context_note: dict,
    pilot: dict,
    seed_sensitivity: dict,
) -> dict:
    summary_df = headline_tables["summary_df"]
    ranking_df = headline_tables["ranking_df"]
    calibration_df = headline_tables["calibration_df"]
    selective_df = headline_tables["selective_df"]
    runtime_df = headline_tables["runtime_df"]
    clean_cal = calibration_df[calibration_df["condition"] == "clean"][["method", "ece", "brier"]].rename(
        columns={"ece": "clean_ece", "brier": "clean_brier"}
    )
    stress_cal = calibration_df[calibration_df["condition"] != "clean"].merge(clean_cal, on="method", how="left")
    calibration_shift = stress_cal.assign(
        ece_delta=stress_cal["ece"] - stress_cal["clean_ece"],
        brier_delta=stress_cal["brier"] - stress_cal["clean_brier"],
    )
    selective_clean = selective_df[selective_df["condition"] == "clean"][["method", "selective_precision"]].rename(
        columns={"selective_precision": "clean_selective_precision"}
    )
    selective_shift = selective_df[selective_df["condition"] != "clean"].merge(selective_clean, on="method", how="left")
    selective_shift["precision_delta_vs_clean"] = (
        selective_shift["selective_precision"] - selective_shift["clean_selective_precision"]
    )
    return {
        "experiment": "StressAudit-SM",
        "headline_protocol": {
            "seed": PRIMARY_SEED,
            "split_frozen_once": True,
            "test_pairs": 32,
            "scope_note": context_note["note"],
            "metrics": {
                "f1": summary_df.to_dict("records"),
                "ranking_stability": ranking_df.to_dict("records"),
                "calibration": calibration_df.to_dict("records"),
                "selective": selective_df.to_dict("records"),
                "runtime": runtime_df.to_dict("records"),
            },
        },
        "hypothesis_assessment": {
            "ranking_changed_under_stress": bool((ranking_df["pairwise_inversions"] > 0).any()),
            "ranking_note": "Held-out method ordering is unchanged under all reported stress conditions."
            if not (ranking_df["pairwise_inversions"] > 0).any()
            else "At least one held-out method ordering changes under stress.",
            "calibration_shift": calibration_shift.to_dict("records"),
            "selective_precision_shift": selective_shift.to_dict("records"),
        },
        "thresholds": thresholds,
        "calibration_thresholds": calibration_results["calibration_summary"],
        "ablations": {
            "no_calibration": ablations["no_calibration"],
            "no_abstention": ablations["no_abstention"],
            "no_composite_summary": ablations["no_composite_summary"],
        },
        "context_stress_fidelity": context_note,
        "runtime_gate": pilot,
        "auxiliary_seed_sensitivity": seed_sensitivity,
    }


def write_results_json(payload: dict) -> None:
    write_json(ROOT / "results.json", payload)


def run_full_pipeline() -> dict:
    run_environment_setup_experiment()
    inventory = split_inventory(freeze_inventory())
    _, _, data_inventory_results = run_data_inventory_experiment(inventory)
    context_note = data_inventory_results["context_note"]
    run_stress_views_experiment(inventory)

    thresholds = {method: tune_threshold(method, inventory) for method in METHODS}
    pilot = run_runtime_pilot(inventory, thresholds)

    raw_frames = []
    runtime_frames = []
    method_results = {}
    for method in METHODS:
        pred_df, runtime_df, result = run_method_experiment(method, inventory, thresholds[method], run_seed=PRIMARY_SEED, repeat_tag="headline")
        raw_frames.append(pred_df)
        runtime_frames.append(runtime_df)
        method_results[method] = result
    headline_raw = pd.concat(raw_frames, ignore_index=True)
    headline_runtimes = pd.concat(runtime_frames, ignore_index=True)

    calibrated_df, _, calibration_results, _ = run_calibration_experiment(headline_raw, inventory, thresholds)
    headline_tables = compute_headline_tables(inventory, headline_raw, calibrated_df, headline_runtimes)
    ablations = run_ablations_experiment(headline_tables, inventory, headline_raw, calibrated_df)
    run_visualization = save_figures(headline_tables, calibrated_df)
    seed_sensitivity = run_seed_sensitivity()

    root_payload = build_root_results(
        headline_tables,
        thresholds,
        calibration_results,
        ablations,
        context_note,
        pilot,
        seed_sensitivity,
    )
    write_results_json(root_payload)

    evaluation_results = {
        "experiment": "evaluation",
        "headline_seed": PRIMARY_SEED,
        "thresholds": thresholds,
        "method_results": method_results,
        "calibration_results_path": str(result_path("calibration")),
        "ablations_results_path": str(result_path("ablations")),
        "visualization_results_path": str(result_path("visualization")),
        "seed_sensitivity_path": str(OUTPUTS / "eval" / "seed_sensitivity_aggregate.csv"),
        "root_results_path": str(ROOT / "results.json"),
    }
    write_experiment_artifacts(
        "evaluation",
        {
            "headline_seed": PRIMARY_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
        evaluation_results,
        [
            "Completed frozen-split headline evaluation.",
            "Ran auxiliary 3-seed sensitivity summaries after the deterministic headline run without replacing the frozen headline protocol.",
            f"Visualization output count: {run_visualization['figure_count']}",
        ],
    )
    return root_payload
