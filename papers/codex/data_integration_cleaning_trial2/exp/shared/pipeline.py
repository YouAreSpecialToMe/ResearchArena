import json
import math
import os
import platform
import random
import shutil
import threading
import time
import unicodedata
import zipfile
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = ROOT / "raw_downloads"
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ROOT / "runs"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
LOGS_DIR = ROOT / "logs"

SEEDS = [13, 29, 47]
CPU_WORKER_LIMIT = 2
BUDGET_SECONDS = 360.0
SOURCE_LOG_BUDGET_SECONDS = 120.0
MATCH_BATCH_SIZE = 250
TOP_K = 25
TRACE_INTERVAL = 15.0
CANOPY_CAP = 80
MAX_POSTING_UPDATES = 1500
THRESHOLDS = [0.78, 0.80, 0.82, 0.84]

DATASET_SPECS = {
    "abt_buy": {
        "zip_name": "Abt-Buy.zip",
        "url": "https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip",
        "left_file": "Abt.csv",
        "right_file": "Buy.csv",
        "gold_file": "abt_buy_perfectMapping.csv",
        "left_id": "id",
        "right_id": "id",
        "gold_left": "idAbt",
        "gold_right": "idBuy",
        "primary_left": "name",
        "primary_right": "name",
        "secondary_left": ["description", "price"],
        "secondary_right": ["description", "manufacturer", "price"],
        "domain": "product",
    },
    "amazon_google": {
        "zip_name": "Amazon-GoogleProducts.zip",
        "url": "https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip",
        "left_file": "Amazon.csv",
        "right_file": "GoogleProducts.csv",
        "gold_file": "Amzon_GoogleProducts_perfectMapping.csv",
        "left_id": "id",
        "right_id": "id",
        "gold_left": "idAmazon",
        "gold_right": "idGoogleBase",
        "primary_left": "title",
        "primary_right": "name",
        "secondary_left": ["description", "manufacturer", "price"],
        "secondary_right": ["description", "manufacturer", "price"],
        "domain": "product",
    },
    "dblp_acm": {
        "zip_name": "DBLP-ACM.zip",
        "url": "https://dbs.uni-leipzig.de/files/datasets/DBLP-ACM.zip",
        "left_file": "DBLP2.csv",
        "right_file": "ACM.csv",
        "gold_file": "DBLP-ACM_perfectMapping.csv",
        "left_id": "id",
        "right_id": "id",
        "gold_left": "idDBLP",
        "gold_right": "idACM",
        "primary_left": "title",
        "primary_right": "title",
        "secondary_left": ["authors", "venue", "year"],
        "secondary_right": ["authors", "venue", "year"],
        "domain": "bibliographic",
    },
}

MAIN_SETTINGS = [
    "abt_buy",
    "amazon_google",
    "dblp_acm",
    "amazon_google_corrupted",
    "dblp_acm_corrupted",
]

ABBREVIATION_SEEDS = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "ltd": "limited",
    "intl": "international",
    "conf": "conference",
    "proc": "proceedings",
    "int": "international",
    "sys": "systems",
    "db": "database",
    "mgmt": "management",
    "assoc": "association",
    "dept": "department",
    "oz": "ounce",
    "gb": "gigabyte",
    "tb": "terabyte",
    "mhz": "megahertz",
    "ghz": "gigahertz",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def archive_previous_outputs() -> Optional[Path]:
    timestamp = int(time.time())
    archive_dir = ROOT / f"archive_{timestamp}"
    moved_any = False
    names = [
        "runs",
        "tables",
        "figures",
        "results.json",
        "results_stale_1774231903.json",
    ]
    patterns = ["runs_partial_*", "runs_stale_*", "tables_stale_*", "figures_stale_*", "results_stale_*.json"]
    for name in names:
        path = ROOT / name
        if path.exists():
            ensure_dir(archive_dir)
            shutil.move(str(path), str(archive_dir / path.name))
            moved_any = True
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            destination = archive_dir / path.name
            if destination.exists():
                continue
            ensure_dir(archive_dir)
            shutil.move(str(path), str(destination))
            moved_any = True
    if not moved_any:
        return None
    return archive_dir


def write_json(path: Path, data) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def set_env_threads() -> None:
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = "1"


def machine_metadata() -> Dict:
    return {
        "python": os.sys.version,
        "python_executable": os.sys.executable,
        "protocol_python_target": "3.11",
        "protocol_python_compliant": os.sys.version.startswith("3.11"),
        "actual_cpu_count": os.cpu_count(),
        "cpu_worker_limit": CPU_WORKER_LIMIT,
        "gpus": 0,
        "platform": platform.platform(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def tokenize_words(text: str) -> List[str]:
    text = safe_str(text).lower()
    out = []
    token = []
    for ch in text:
        if ch.isalnum():
            token.append(ch)
        else:
            if token:
                out.append("".join(token))
                token = []
    if token:
        out.append("".join(token))
    return out


def blocker_tokens(text: str) -> List[str]:
    words = tokenize_words(text)
    joined = "".join(words)
    chars = [joined[i : i + 3] for i in range(max(0, len(joined) - 2))]
    return words + chars


def normalize_basic(text: str) -> str:
    text = unicodedata.normalize("NFKC", safe_str(text)).casefold()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def sort_tokens(text: str) -> str:
    return " ".join(sorted(tokenize_words(text)))


def normalize_numeric(text: str, domain: str) -> str:
    text = normalize_basic(text)
    repl = {"gb": "gigabyte", "tb": "terabyte", "mhz": "megahertz", "ghz": "gigahertz"}
    for src, dst in repl.items():
        text = text.replace(src, dst)
    out = []
    for token in text.split():
        if token.endswith(".0"):
            token = token[:-2]
        out.append(token)
    if domain == "bibliographic" and len(out) == 1 and out[0].isdigit():
        return out[0]
    return " ".join(out)


def maybe_float(text: str) -> Optional[float]:
    text = safe_str(text).replace("$", "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def token_entropy(texts: Sequence[str]) -> float:
    counts = Counter()
    total = 0
    for text in texts:
        tokens = tokenize_words(text)
        counts.update(tokens)
        total += len(tokens)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(max(p, 1e-12))
    return entropy


@dataclass
class DatasetBundle:
    setting: str
    base_name: str
    seed: int
    left: pd.DataFrame
    right: pd.DataFrame
    positives: set
    validation_positives: set
    validation_negatives: set
    final_positives: set
    threshold: float
    domain: str
    primary_left: str
    primary_right: str
    secondary_left: List[str]
    secondary_right: List[str]
    corruption_summary: Dict[str, int]
    corruption_manifest_path: Optional[str]


def download_raw_data() -> None:
    ensure_dir(RAW_DIR)
    for spec in DATASET_SPECS.values():
        target = RAW_DIR / spec["zip_name"]
        if target.exists():
            continue
        import urllib.request

        urllib.request.urlretrieve(spec["url"], target)


def read_zip_csv(zip_path: Path, name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(name) as handle:
            return pd.read_csv(handle, encoding="latin1")


def standardize_table(df: pd.DataFrame, id_col: str, side: str, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[id_col] = df[id_col].astype(str)
    df = df.rename(columns={id_col: "id"})
    keep = ["id"] + [col for col in cols if col in df.columns]
    df = df[keep].copy()
    for col in keep[1:]:
        df[col] = df[col].map(safe_str)
    df["side"] = side
    df["gid"] = side + ":" + df["id"]
    return df


def corruption_ops(domain: str) -> List[str]:
    if domain == "bibliographic":
        return [
            "abbr_drift",
            "punct_ws_drift",
            "token_order_drift",
            "venue_abbr_drift",
            "author_shortening",
            "year_format_drift",
        ]
    return ["abbr_drift", "punct_ws_drift", "unit_numeric_drift", "token_order_drift"]


def corrupt_value(text: str, op: str, rng: random.Random) -> str:
    text = safe_str(text)
    words = text.split()
    if not text:
        return text
    if op == "abbr_drift" and len(words) > 1:
        return " ".join(w[0] if len(w) > 4 else w for w in words)
    if op == "punct_ws_drift":
        return "  ".join(text.replace("-", " ").replace("/", " ").split())
    if op == "unit_numeric_drift":
        return text.replace("GB", " Gigabyte").replace("gb", " gigabyte").replace(".00", "").replace("$", "")
    if op == "token_order_drift" and len(words) > 2:
        words = words[:]
        rng.shuffle(words)
        return " ".join(words)
    if op == "venue_abbr_drift":
        return text.replace("International", "Int.").replace("Conference", "Conf.").replace("Management", "Mgmt.")
    if op == "author_shortening":
        people = []
        for part in text.split(","):
            bits = part.strip().split()
            if not bits:
                continue
            if len(bits) == 1:
                people.append(bits[0])
            else:
                people.append(bits[0][0] + ". " + bits[-1])
        return ", ".join(people)
    if op == "year_format_drift":
        return text[-2:] if text.isdigit() and len(text) == 4 else text
    return text


def build_abbreviation_dictionary(
    setting: str,
    seed: int,
    left: pd.DataFrame,
    right: pd.DataFrame,
    secondary_cols: List[str],
) -> pd.DataFrame:
    ensure_dir(ARTIFACTS_DIR)
    (ARTIFACTS_DIR / "abbr_seed_rules.txt").write_text(
        "\n".join(f"{short}\t{long}" for short, long in sorted(ABBREVIATION_SEEDS.items())),
        encoding="utf-8",
    )
    accepted = []
    induction_log = []
    pair_counts = defaultdict(Counter)
    texts_by_attr = defaultdict(list)
    for frame in [left, right]:
        for col in [c for c in secondary_cols if c in frame.columns] + [c for c in ["title", "name", "venue"] if c in frame.columns]:
            texts_by_attr[col].extend(frame[col].map(safe_str).tolist())
    for short, long in ABBREVIATION_SEEDS.items():
        accepted.append(
            {"short": short, "long": long, "count": 2, "confidence": 1.0, "attribute_domain": "seed", "source": "seed"}
        )
    for attr, texts in texts_by_attr.items():
        for text in texts:
            norm = normalize_basic(text)
            if "(" in norm and ")" in norm:
                left_side = norm.split("(")[0].strip()
                right_side = norm.split("(")[1].split(")")[0].strip()
                if left_side and right_side:
                    pair_counts[attr][(right_side, left_side)] += 1
            if "=" in norm:
                left_side, right_side = [piece.strip() for piece in norm.split("=", 1)]
                if left_side and right_side:
                    pair_counts[attr][(left_side, right_side)] += 1
            tokens = norm.split()
            if len(tokens) >= 2:
                initials = "".join(token[0] for token in tokens if token)
                if 2 <= len(initials) <= 6:
                    pair_counts[attr][(initials, " ".join(tokens))] += 1
                for width in range(2, min(6, len(tokens)) + 1):
                    sub = tokens[:width]
                    abbrev = "".join(token[0] for token in sub if token)
                    pair_counts[attr][(abbrev, " ".join(sub))] += 1
        grouped = defaultdict(list)
        for (short, long), count in pair_counts[attr].items():
            grouped[short].append((long, count))
        used_longs = set()
        for short, values in grouped.items():
            values = sorted(values, key=lambda item: (-item[1], item[0]))
            total = sum(count for _, count in values)
            best_long, best_count = values[0]
            confidence = best_count / max(1, total)
            ratio = len(best_long) / max(1, len(short))
            rejected_reason = None
            if best_count < 2:
                rejected_reason = "count<2"
            elif confidence < 0.90:
                rejected_reason = "confidence<0.90"
            elif ratio < 1.5 or ratio > 6.0:
                rejected_reason = "length_ratio"
            elif best_long in used_longs:
                rejected_reason = "attribute_one_to_many_conflict"
            if rejected_reason is None:
                accepted.append(
                    {
                        "short": short,
                        "long": best_long,
                        "count": int(best_count),
                        "confidence": round(confidence, 6),
                        "attribute_domain": attr,
                        "source": "mined",
                    }
                )
                used_longs.add(best_long)
                induction_log.append(
                    {"attribute_domain": attr, "short": short, "long": best_long, "count": int(best_count), "accepted": True}
                )
            else:
                induction_log.append(
                    {
                        "attribute_domain": attr,
                        "short": short,
                        "long": best_long,
                        "count": int(best_count),
                        "accepted": False,
                        "reason": rejected_reason,
                    }
                )
    abbr_df = pd.DataFrame(accepted).drop_duplicates(subset=["short", "long", "attribute_domain"]).sort_values(
        ["short", "long", "attribute_domain"]
    )
    abbr_df.to_csv(ARTIFACTS_DIR / f"abbr_dict_{setting}_{seed}.csv", index=False)
    pd.DataFrame(induction_log).to_csv(ARTIFACTS_DIR / f"abbr_induction_log_{setting}_{seed}.csv", index=False)
    return abbr_df


def apply_abbreviation(text: str, abbr_df: pd.DataFrame) -> str:
    mapping = dict(zip(abbr_df["short"], abbr_df["long"]))
    return " ".join(mapping.get(token, token) for token in normalize_basic(text).split())


def apply_operator(value: str, operator: str, domain: str, abbr_df: pd.DataFrame, canopy_counter: Optional[Counter] = None) -> str:
    if operator == "basic":
        return normalize_basic(value)
    if operator == "token_sort":
        return sort_tokens(normalize_basic(value))
    if operator == "numeric":
        return normalize_numeric(value, domain)
    if operator == "abbr":
        return apply_abbreviation(value, abbr_df)
    if operator == "typo":
        normalized = normalize_basic(value)
        if canopy_counter is None:
            return normalized
        output = []
        for token in normalized.split():
            best = token
            for other in sorted(canopy_counter):
                if other == token:
                    continue
                if not other or other[0] != token[:1]:
                    continue
                if abs(len(other) - len(token)) > 2:
                    continue
                if Levenshtein.distance(other, token) != 1:
                    continue
                if canopy_counter[other] > canopy_counter[best] or (
                    canopy_counter[other] == canopy_counter[best] and other < best
                ):
                    best = other
            output.append(best)
        return " ".join(output)
    return normalize_basic(value)


class Blocker:
    def __init__(self, bundle: DatasetBundle):
        self.bundle = bundle
        self.left_texts = bundle.left[bundle.primary_left].map(normalize_basic).tolist()
        self.right_texts = bundle.right[bundle.primary_right].map(normalize_basic).tolist()
        self.left_gid = bundle.left["gid"].tolist()
        self.right_gid = bundle.right["gid"].tolist()
        self.left_pos = {gid: idx for idx, gid in enumerate(self.left_gid)}
        self.right_pos = {gid: idx for idx, gid in enumerate(self.right_gid)}
        self.vectorizer = TfidfVectorizer(tokenizer=blocker_tokens, lowercase=False, preprocessor=None, token_pattern=None)
        joined = self.left_texts + self.right_texts
        matrix = self.vectorizer.fit_transform(joined)
        self.left_matrix = matrix[: len(self.left_texts)].tolil()
        self.right_matrix = matrix[len(self.left_texts) :].tolil()

    def update_bundle(self, bundle: DatasetBundle, touched: set) -> None:
        self.bundle = bundle
        for gid in touched:
            side = gid.split(":", 1)[0]
            if side == "l" and gid in self.left_pos:
                idx = self.left_pos[gid]
                text = normalize_basic(bundle.left.iloc[idx][bundle.primary_left])
                self.left_texts[idx] = text
                self.left_matrix[idx] = self.vectorizer.transform([text])
            elif side == "r" and gid in self.right_pos:
                idx = self.right_pos[gid]
                text = normalize_basic(bundle.right.iloc[idx][bundle.primary_right])
                self.right_texts[idx] = text
                self.right_matrix[idx] = self.vectorizer.transform([text])

    def _lr_neighbors(self, left_indices: List[int], right_indices: List[int]) -> Dict[Tuple[str, str], float]:
        if not left_indices or not right_indices:
            return {}
        nn = NearestNeighbors(n_neighbors=min(TOP_K, len(right_indices)), metric="cosine", algorithm="brute")
        nn.fit(self.right_matrix[right_indices].tocsr())
        distances, idxs = nn.kneighbors(self.left_matrix[left_indices].tocsr())
        out = {}
        for row_pos, left_idx in enumerate(left_indices):
            for dist, pos in zip(distances[row_pos], idxs[row_pos]):
                right_idx = right_indices[pos]
                out[(self.left_gid[left_idx], self.right_gid[right_idx])] = max(
                    out.get((self.left_gid[left_idx], self.right_gid[right_idx]), 0.0),
                    1.0 - float(dist),
                )
        return out

    def _rl_neighbors(self, right_indices: List[int], left_indices: List[int]) -> Dict[Tuple[str, str], float]:
        if not right_indices or not left_indices:
            return {}
        nn = NearestNeighbors(n_neighbors=min(TOP_K, len(left_indices)), metric="cosine", algorithm="brute")
        nn.fit(self.left_matrix[left_indices].tocsr())
        distances, idxs = nn.kneighbors(self.right_matrix[right_indices].tocsr())
        out = {}
        for row_pos, right_idx in enumerate(right_indices):
            for dist, pos in zip(distances[row_pos], idxs[row_pos]):
                left_idx = left_indices[pos]
                out[(self.left_gid[left_idx], self.right_gid[right_idx])] = max(
                    out.get((self.left_gid[left_idx], self.right_gid[right_idx]), 0.0),
                    1.0 - float(dist),
                )
        return out

    def all_pairs(self) -> Dict[Tuple[str, str], float]:
        forward = self._lr_neighbors(list(range(len(self.left_gid))), list(range(len(self.right_gid))))
        reverse = self._rl_neighbors(list(range(len(self.right_gid))), list(range(len(self.left_gid))))
        merged = dict(forward)
        merged.update(reverse)
        return merged

    def candidate_pairs_for_gids(self, touched: set) -> Dict[Tuple[str, str], float]:
        left_indices = [idx for idx, gid in enumerate(self.left_gid) if gid in touched]
        right_indices = [idx for idx, gid in enumerate(self.right_gid) if gid in touched]
        out = {}
        if left_indices:
            out.update(self._lr_neighbors(left_indices, list(range(len(self.right_gid)))))
        if right_indices:
            out.update(self._rl_neighbors(right_indices, list(range(len(self.left_gid)))))
        return out


class PairScorer:
    def __init__(self, bundle: DatasetBundle):
        self.bundle = bundle
        corpus = bundle.left[bundle.primary_left].tolist() + bundle.right[bundle.primary_right].tolist()
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_words, lowercase=True, token_pattern=None)
        matrix = self.vectorizer.fit_transform([normalize_basic(text) for text in corpus])
        self.left_matrix = matrix[: len(bundle.left)].tolil()
        self.right_matrix = matrix[len(bundle.left) :].tolil()
        self.left_pos = {gid: idx for idx, gid in enumerate(bundle.left["gid"].tolist())}
        self.right_pos = {gid: idx for idx, gid in enumerate(bundle.right["gid"].tolist())}
        self.left_cache = self._build_cache(bundle.left, bundle.primary_left, bundle.secondary_left)
        self.right_cache = self._build_cache(bundle.right, bundle.primary_right, bundle.secondary_right)

    def _build_cache(self, df: pd.DataFrame, primary_col: str, secondary_cols: List[str]) -> Dict[str, Dict]:
        cache = {}
        for row in df.to_dict(orient="records"):
            gid = row["gid"]
            primary = safe_str(row.get(primary_col, ""))
            secondary = {col: safe_str(row.get(col, "")) for col in secondary_cols}
            cache[gid] = {
                "primary": primary,
                "primary_norm": normalize_basic(primary),
                "primary_tokens": set(tokenize_words(primary)),
                "secondary_norm": {col: normalize_basic(value) for col, value in secondary.items()},
                "secondary": secondary,
                "price": maybe_float(secondary.get("price", "")),
                "year": normalize_numeric(secondary.get("year", ""), self.bundle.domain),
            }
        return cache

    def primary_cosine(self, left_gid: str, right_gid: str) -> float:
        left_vec = self.left_matrix[self.left_pos[left_gid]].tocsr()
        right_vec = self.right_matrix[self.right_pos[right_gid]].tocsr()
        num = left_vec.multiply(right_vec).sum()
        den = math.sqrt(max(left_vec.multiply(left_vec).sum(), 1e-12) * max(right_vec.multiply(right_vec).sum(), 1e-12))
        return float(num / den) if den else 0.0

    def update_bundle(self, bundle: DatasetBundle, touched: set) -> None:
        self.bundle = bundle
        for gid in touched:
            side = gid.split(":", 1)[0]
            if side == "l" and gid in self.left_pos:
                idx = self.left_pos[gid]
                text = normalize_basic(bundle.left.iloc[idx][bundle.primary_left])
                self.left_matrix[idx] = self.vectorizer.transform([text])
                row = bundle.left.iloc[[idx]].copy()
                self.left_cache.update(self._build_cache(row, bundle.primary_left, bundle.secondary_left))
            elif side == "r" and gid in self.right_pos:
                idx = self.right_pos[gid]
                text = normalize_basic(bundle.right.iloc[idx][bundle.primary_right])
                self.right_matrix[idx] = self.vectorizer.transform([text])
                row = bundle.right.iloc[[idx]].copy()
                self.right_cache.update(self._build_cache(row, bundle.primary_right, bundle.secondary_right))

    @staticmethod
    def token_jaccard(a: str, b: str) -> float:
        ta = set(tokenize_words(a))
        tb = set(tokenize_words(b))
        if not ta and not tb:
            return 1.0
        return len(ta & tb) / max(1, len(ta | tb))

    def secondary_agreement(self, left_row: Dict, right_row: Dict) -> float:
        values = []
        for left_col, right_col in zip(self.bundle.secondary_left, self.bundle.secondary_right):
            left_norm = left_row["secondary_norm"].get(left_col, "")
            right_norm = right_row["secondary_norm"].get(right_col, "")
            if not left_norm and not right_norm:
                continue
            if left_col == "price" or right_col == "price":
                left_price = left_row["price"]
                right_price = right_row["price"]
                if left_price is not None and right_price is not None:
                    denom = max(abs(left_price), abs(right_price), 1.0)
                    values.append(max(0.0, 1.0 - abs(left_price - right_price) / denom))
                    continue
            values.append(self.token_jaccard(left_norm, right_norm))
        return float(np.mean(values)) if values else 0.0

    def numeric_or_year_agreement(self, left_row: Dict, right_row: Dict) -> float:
        values = []
        if "price" in self.bundle.secondary_left and "price" in self.bundle.secondary_right:
            left_price = left_row["price"]
            right_price = right_row["price"]
            if left_price is not None and right_price is not None:
                values.append(max(0.0, 1.0 - abs(left_price - right_price) / max(abs(left_price), abs(right_price), 1.0)))
        if "year" in self.bundle.secondary_left and "year" in self.bundle.secondary_right:
            values.append(1.0 if left_row["year"] and right_row["year"] and left_row["year"] == right_row["year"] else 0.0)
        return float(np.mean(values)) if values else 0.0

    def score_pair(self, left_gid: str, right_gid: str) -> float:
        left_row = self.left_cache[left_gid]
        right_row = self.right_cache[right_gid]
        primary_cos = self.primary_cosine(left_gid, right_gid)
        token_j = self.token_jaccard(left_row["primary"], right_row["primary"])
        rf = fuzz.ratio(left_row["primary_norm"], right_row["primary_norm"]) / 100.0
        secondary = self.secondary_agreement(left_row, right_row)
        numeric = self.numeric_or_year_agreement(left_row, right_row)
        return 0.50 * primary_cos + 0.20 * token_j + 0.15 * rf + 0.10 * secondary + 0.05 * numeric


def sample_validation_negatives(bundle: DatasetBundle) -> set:
    raw_blocker = Blocker(bundle)
    candidates = [pair for pair in raw_blocker.all_pairs() if pair not in bundle.positives]
    rng = random.Random(bundle.seed)
    rng.shuffle(candidates)
    return set(candidates[: len(bundle.validation_positives)])


def prepare_one_setting(base_name: str, seed: int, corrupted: bool) -> DatasetBundle:
    ensure_dir(DATA_DIR)
    ensure_dir(ARTIFACTS_DIR)
    spec = DATASET_SPECS[base_name]
    zip_path = RAW_DIR / spec["zip_name"]
    left_raw = read_zip_csv(zip_path, spec["left_file"])
    right_raw = read_zip_csv(zip_path, spec["right_file"])
    gold_raw = read_zip_csv(zip_path, spec["gold_file"])
    left = standardize_table(left_raw, spec["left_id"], "l", [spec["primary_left"]] + spec["secondary_left"])
    right = standardize_table(right_raw, spec["right_id"], "r", [spec["primary_right"]] + spec["secondary_right"])
    positives = {
        (f"l:{str(left_id)}", f"r:{str(right_id)}")
        for left_id, right_id in gold_raw[[spec["gold_left"], spec["gold_right"]]].itertuples(index=False)
    }
    rng = random.Random(seed)
    positive_list = sorted(positives)
    rng.shuffle(positive_list)
    val_n = max(1, int(0.2 * len(positive_list)))
    validation_positives = set(positive_list[:val_n])
    final_positives = set(positive_list[val_n:])
    raw_bundle = DatasetBundle(
        setting=base_name,
        base_name=base_name,
        seed=seed,
        left=left.copy(),
        right=right.copy(),
        positives=positives,
        validation_positives=validation_positives,
        validation_negatives=set(),
        final_positives=final_positives,
        threshold=0.8,
        domain=spec["domain"],
        primary_left=spec["primary_left"],
        primary_right=spec["primary_right"],
        secondary_left=spec["secondary_left"],
        secondary_right=spec["secondary_right"],
        corruption_summary={},
        corruption_manifest_path=None,
    )
    validation_negatives = sample_validation_negatives(raw_bundle)
    corruption_summary = Counter()
    corruption_manifest_path = None
    if corrupted:
        manifest = []
        for frame, primary_col in [(left, spec["primary_left"]), (right, spec["primary_right"])]:
            gids = frame["gid"].tolist()
            sample_n = max(1, int(0.3 * len(gids)))
            chosen = set(rng.sample(gids, sample_n))
            for idx, row in frame.iterrows():
                if row["gid"] not in chosen:
                    continue
                op = rng.choice(corruption_ops(spec["domain"]))
                target_col = primary_col
                before = safe_str(row[target_col])
                after = corrupt_value(before, op, rng)
                frame.at[idx, target_col] = after
                corruption_summary[op] += 1
                manifest.append({"gid": row["gid"], "field": target_col, "operation": op, "before": before, "after": after})
        corruption_manifest_path = str(ARTIFACTS_DIR / f"corruption_manifest_{base_name}_corrupted_{seed}.json")
        write_json(Path(corruption_manifest_path), manifest)
    setting = f"{base_name}_corrupted" if corrupted else base_name
    build_abbreviation_dictionary(setting, seed, left, right, sorted(set(spec["secondary_left"] + spec["secondary_right"])))
    return DatasetBundle(
        setting=setting,
        base_name=base_name,
        seed=seed,
        left=left,
        right=right,
        positives=positives,
        validation_positives=validation_positives,
        validation_negatives=validation_negatives,
        final_positives=final_positives,
        threshold=0.8,
        domain=spec["domain"],
        primary_left=spec["primary_left"],
        primary_right=spec["primary_right"],
        secondary_left=spec["secondary_left"],
        secondary_right=spec["secondary_right"],
        corruption_summary=dict(corruption_summary),
        corruption_manifest_path=corruption_manifest_path,
    )


def choose_thresholds(bundles: Sequence[DatasetBundle]) -> Dict[str, float]:
    scored_by_key = {}
    for bundle in bundles:
        blocker = Blocker(bundle)
        scorer = PairScorer(bundle)
        candidate_pairs = blocker.all_pairs()
        validation_pairs = sorted(bundle.validation_positives | bundle.validation_negatives)
        valid_pairs = [pair for pair in validation_pairs if pair in candidate_pairs]
        scored_by_key[(bundle.setting, bundle.seed)] = [(pair, scorer.score_pair(*pair)) for pair in valid_pairs]
    thresholds = {}
    settings = sorted({bundle.setting for bundle in bundles})
    for target_setting in settings:
        best_threshold = THRESHOLDS[0]
        best_score = -1.0
        for threshold in THRESHOLDS:
            f1s = []
            for bundle in bundles:
                if bundle.setting == target_setting:
                    continue
                scored = scored_by_key[(bundle.setting, bundle.seed)]
                preds = {pair for pair, score in scored if score >= threshold}
                gold = {pair for pair, _ in scored if pair in bundle.validation_positives}
                tp = len(preds & gold)
                fp = len(preds - gold)
                fn = len(gold - preds)
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                f1s.append(f1)
            score = float(np.mean(f1s)) if f1s else 0.0
            if score > best_score:
                best_score = score
                best_threshold = threshold
        thresholds[target_setting] = best_threshold
    return thresholds


def build_dataset_summary(bundles: Sequence[DatasetBundle]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        blocker = Blocker(bundle)
        texts = bundle.left[bundle.primary_left].tolist() + bundle.right[bundle.primary_right].tolist()
        vocab = set()
        for text in texts:
            vocab.update(tokenize_words(text))
        missing = 0
        total = 0
        for frame in [bundle.left, bundle.right]:
            for col in frame.columns:
                if col in {"id", "gid", "side"}:
                    continue
                total += len(frame)
                missing += int((frame[col].map(safe_str) == "").sum())
        rows.append(
            {
                "setting": bundle.setting,
                "left_records": len(bundle.left),
                "right_records": len(bundle.right),
                "positives": len(bundle.positives),
                "blocker_vocabulary_size": len(vocab),
                "candidate_graph_size_before_cleaning": len(blocker.all_pairs()),
                "mean_primary_token_length": float(np.mean([len(tokenize_words(text)) for text in texts])),
                "attribute_missingness": missing / max(1, total),
                "corruption_counts": json.dumps(bundle.corruption_summary, sort_keys=True),
            }
        )
    df = pd.DataFrame(rows).sort_values("setting")
    df.to_csv(ROOT / "dataset_summary.csv", index=False)
    return df


def clone_bundle(bundle: DatasetBundle) -> DatasetBundle:
    return DatasetBundle(
        setting=bundle.setting,
        base_name=bundle.base_name,
        seed=bundle.seed,
        left=bundle.left.copy(),
        right=bundle.right.copy(),
        positives=set(bundle.positives),
        validation_positives=set(bundle.validation_positives),
        validation_negatives=set(bundle.validation_negatives),
        final_positives=set(bundle.final_positives),
        threshold=bundle.threshold,
        domain=bundle.domain,
        primary_left=bundle.primary_left,
        primary_right=bundle.primary_right,
        secondary_left=list(bundle.secondary_left),
        secondary_right=list(bundle.secondary_right),
        corruption_summary=dict(bundle.corruption_summary),
        corruption_manifest_path=bundle.corruption_manifest_path,
    )


def related_token_map(bundle: DatasetBundle, abbr_df: pd.DataFrame) -> Dict[str, List[str]]:
    all_tokens = sorted(
        {
            token
            for frame, col in [(bundle.left, bundle.primary_left), (bundle.right, bundle.primary_right)]
            for text in frame[col].map(safe_str)
            for token in tokenize_words(text)
            if token
        }
    )
    by_prefix = defaultdict(list)
    for token in all_tokens:
        by_prefix[(token[:1], len(token))].append(token)
    abbr_to_long = {}
    for row in abbr_df.to_dict(orient="records"):
        short = tokenize_words(row["short"])
        long = tokenize_words(row["long"])
        if len(short) == 1 and len(long) == 1:
            abbr_to_long.setdefault(short[0], set()).add(long[0])
    related = {}
    for token in all_tokens:
        neighbors = set()
        for other in abbr_to_long.get(token, set()):
            neighbors.add(other)
        if len(token) >= 3:
            for delta in [-2, -1, 0, 1, 2]:
                for other in by_prefix.get((token[:1], len(token) + delta), []):
                    if other == token:
                        continue
                    if Levenshtein.distance(token, other) == 1:
                        neighbors.add(other)
                    elif len(token) <= 6 and len(other) > len(token) and other.startswith(token):
                        neighbors.add(other)
                    elif len(token) <= 6 and 1.5 <= len(other) / max(1, len(token)) <= 6.0:
                        if "".join(part[0] for part in other.split()) == token:
                            neighbors.add(other)
        related[token] = sorted(neighbors)[:4]
    return related


def make_canopies(bundle: DatasetBundle, abbr_df: pd.DataFrame) -> List[Dict]:
    token_to_gids = defaultdict(set)
    for _, row in pd.concat([bundle.left, bundle.right], ignore_index=True).iterrows():
        primary_col = bundle.primary_left if row["side"] == "l" else bundle.primary_right
        for token in set(tokenize_words(row[primary_col])):
            token_to_gids[token].add(row["gid"])
    related_tokens = related_token_map(bundle, abbr_df)
    canopies = []
    for token, gids in token_to_gids.items():
        expanded = set(gids)
        for other in related_tokens.get(token, []):
            expanded.update(token_to_gids.get(other, set()))
        gids = sorted(expanded)
        if len(gids) < 2:
            continue
        if not any(gid.startswith("l:") for gid in gids) or not any(gid.startswith("r:") for gid in gids):
            continue
        canopies.append({"token": token, "gids": gids[:CANOPY_CAP], "size": min(len(gids), CANOPY_CAP)})
    canopies.sort(key=lambda item: (-item["size"], item["token"]))
    return canopies


def canopy_feature_frame(bundle: DatasetBundle, candidate_scores: Dict[Tuple[str, str], float], threshold: float, abbr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    near_miss_pairs = {
        pair
        for pair, score in sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))[:4000]
        if 0.65 <= score < threshold
    }
    for canopy in make_canopies(bundle, abbr_df):
        local_pairs = [(pair, score) for pair, score in candidate_scores.items() if pair[0] in canopy["gids"] or pair[1] in canopy["gids"]]
        local_scores = [score for _, score in local_pairs]
        near_miss = [score for pair, score in local_pairs if pair in near_miss_pairs]
        local_texts = []
        for gid in canopy["gids"]:
            side = gid.split(":", 1)[0]
            frame = bundle.left if side == "l" else bundle.right
            col = bundle.primary_left if side == "l" else bundle.primary_right
            local_texts.append(frame.loc[frame["gid"] == gid, col].iloc[0])
        rows.append(
            {
                "token": canopy["token"],
                "gids": canopy["gids"],
                "size": canopy["size"],
                "near_miss_density": len(near_miss) / max(1, canopy["size"]),
                "entropy": token_entropy(local_texts),
                "avg_score": float(np.mean(local_scores)) if local_scores else 0.0,
                "candidate_starved_frac": float(
                    sum(1 for gid in canopy["gids"] if not any(gid == pair[0] or gid == pair[1] for pair in candidate_scores))
                )
                / max(1, len(canopy["gids"])),
                "pair_count": len(local_pairs),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["token", "gids", "size", "near_miss_density", "entropy", "avg_score", "candidate_starved_frac"])
    return df.sort_values(["near_miss_density", "size", "avg_score"], ascending=[False, False, False]).reset_index(drop=True)


def operator_families(domain: str, mode: str = "full") -> List[str]:
    if mode == "format_only":
        return ["basic"]
    if domain == "bibliographic":
        return ["basic", "token_sort", "abbr", "typo"]
    return ["basic", "token_sort", "numeric", "abbr"]


class PeakRSSSampler:
    def __init__(self):
        self.max_rss_mb = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.max_rss_mb = max(self.max_rss_mb, rss_mb())
            self._stop.wait(5.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.max_rss_mb = max(self.max_rss_mb, rss_mb())
        return self.max_rss_mb


class ExperimentRunner:
    def __init__(
        self,
        bundle: DatasetBundle,
        method: str,
        budget_seconds: float,
        threshold: float,
        estimator: Optional[Dict] = None,
        ablation: Optional[str] = None,
        reference_actions: Optional[List[Dict]] = None,
    ):
        self.bundle = clone_bundle(bundle)
        self.method = method
        self.budget_seconds = budget_seconds
        self.threshold = threshold
        self.estimator = estimator or {}
        self.ablation = ablation
        self.reference_actions = reference_actions or []
        self.stage_times = defaultdict(float)
        self.init_stage_times = defaultdict(float)
        self.evaluated_pairs: List[Tuple[Tuple[str, str], float, bool]] = []
        self.predicted_matches = set()
        self.trace: List[Dict] = []
        self.clean_actions: List[Dict] = []
        self._clean_internal: List[Dict] = []
        self.stdout_lines: List[str] = []
        self._sampler = PeakRSSSampler()
        self.run_start = 0.0
        self.total_start = 0.0
        self.process_start = 0.0
        self.total_process_start = 0.0
        self.last_trace_mark = 0.0
        self.predicted_match_batch_cost = 1.0
        self.predicted_clean_action_cost = 5.0
        self.predicted_decision_cost = 5.0
        self._hybrid_frontier_cache = []
        self.abbr_df = pd.read_csv(ARTIFACTS_DIR / f"abbr_dict_{self.bundle.setting}_{self.bundle.seed}.csv")
        self.blocker = None
        self.scorer = None
        self.candidate_scores = {}
        self.frontier = []
        self.record_lookup = {}
        self.token_to_gids = defaultdict(set)
        self.abbr_map = {}
        self.executed_action_keys = set()
        self.corruption_by_gid = self._load_corruption_by_gid()
        self._rebuild_record_index()
        for row in self.abbr_df.to_dict(orient="records"):
            short = tokenize_words(row["short"])
            long = tokenize_words(row["long"])
            if len(short) == 1 and len(long) == 1:
                self.abbr_map.setdefault(short[0], set()).add(long[0])

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.stdout_lines.append(f"[{timestamp}] {message}")

    def _elapsed(self) -> float:
        return time.perf_counter() - self.run_start

    def _record_stage(self, stage: str, start_wall: float, start_cpu: float, init: bool = False) -> None:
        target_wall = self.init_stage_times if init else self.stage_times
        target_cpu = self._init_cpu_stage_times if init else self._cpu_stage_times
        target_wall[stage] += time.perf_counter() - start_wall
        target_cpu[stage] += time.process_time() - start_cpu

    def _rebuild_record_index(self) -> None:
        self.record_lookup = {}
        self.token_to_gids = defaultdict(set)
        for side, frame, primary_col, secondary_cols in [
            ("l", self.bundle.left, self.bundle.primary_left, self.bundle.secondary_left),
            ("r", self.bundle.right, self.bundle.primary_right, self.bundle.secondary_right),
        ]:
            for idx, row in frame.iterrows():
                gid = row["gid"]
                primary = safe_str(row[primary_col])
                self.record_lookup[gid] = {
                    "side": side,
                    "frame": frame,
                    "idx": idx,
                    "primary_col": primary_col,
                    "secondary_cols": [col for col in secondary_cols if col in frame.columns],
                    "primary": primary,
                }
                for token in set(tokenize_words(primary)):
                    self.token_to_gids[token].add(gid)

    def _load_corruption_by_gid(self) -> Dict[str, List[str]]:
        if not self.bundle.corruption_manifest_path:
            return {}
        path = Path(self.bundle.corruption_manifest_path)
        if not path.exists():
            return {}
        manifest = read_json(path)
        by_gid = defaultdict(list)
        for row in manifest:
            by_gid[row["gid"]].append(row["operation"])
        return {gid: sorted(set(ops)) for gid, ops in by_gid.items()}

    def _rebuild_full(self, init: bool = False) -> None:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        self.blocker = Blocker(self.bundle)
        self._record_stage("blocker_maintenance", start_wall, start_cpu, init=init)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        self.scorer = PairScorer(self.bundle)
        raw_pairs = self.blocker.all_pairs()
        self.candidate_scores = {pair: self.scorer.score_pair(*pair) for pair in raw_pairs}
        self.frontier = sorted(self.candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        self._hybrid_frontier_cache = []
        self._record_stage("scoring", start_wall, start_cpu, init=init)
        self._log(f"rebuilt blocker and scorer with {len(self.candidate_scores)} candidate pairs")

    def _incremental_refresh(self, touched: set) -> Tuple[set, int]:
        before_pairs = set(self.candidate_scores)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        self.blocker.update_bundle(self.bundle, touched)
        updated_pairs = self.blocker.candidate_pairs_for_gids(touched)
        self._record_stage("blocker_maintenance", start_wall, start_cpu)
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        self.scorer.update_bundle(self.bundle, touched)
        for pair in list(self.candidate_scores):
            if pair[0] in touched or pair[1] in touched:
                self.candidate_scores.pop(pair, None)
        for pair in updated_pairs:
            self.candidate_scores[pair] = self.scorer.score_pair(*pair)
        self.frontier = sorted(self.candidate_scores.items(), key=lambda item: (-item[1], item[0]))
        self._hybrid_frontier_cache = []
        self._record_stage("scoring", start_wall, start_cpu)
        new_pairs = set(updated_pairs) - before_pairs
        return new_pairs, min(MAX_POSTING_UPDATES, len(touched) * TOP_K)

    def _match_batch_candidates(self) -> List[Tuple[Tuple[str, str], float]]:
        seen_pairs = {pair for pair, _, _ in self.evaluated_pairs}
        batch = []
        for pair, score in self.frontier:
            if pair in seen_pairs:
                continue
            batch.append((pair, score))
            if len(batch) >= MATCH_BATCH_SIZE:
                break
        return batch

    def _evaluate_pairs(self, batch: List[Tuple[Tuple[str, str], float]], label: str) -> Dict:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        tp = 0
        predicted_in_batch = 0
        rescored_pairs = []
        for pair, _ in batch:
            score = self.scorer.score_pair(*pair)
            is_match = pair in self.bundle.final_positives
            self.evaluated_pairs.append((pair, score, is_match))
            rescored_pairs.append((pair, score))
            if score >= self.threshold:
                self.predicted_matches.add(pair)
                predicted_in_batch += 1
                tp += int(is_match)
        self._record_stage("matching", start_wall, start_cpu)
        batch_seconds = time.perf_counter() - start_wall
        if batch:
            self.predicted_match_batch_cost = max(batch_seconds, 1e-6)
        self._update_clean_action_outcomes([pair for pair, _ in rescored_pairs])
        self._log(f"{label}: evaluated {len(batch)} pairs, predicted {predicted_in_batch} matches")
        return {"pairs": len(batch), "true_matches": tp, "elapsed": batch_seconds}

    def _precision(self) -> float:
        tp = len(self.predicted_matches & self.bundle.final_positives)
        return tp / max(1, len(self.predicted_matches))

    def _recall(self) -> float:
        tp = len(self.predicted_matches & self.bundle.final_positives)
        return tp / max(1, len(self.bundle.final_positives))

    def _f1(self) -> float:
        precision = self._precision()
        recall = self._recall()
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def _trace_point(self) -> Dict:
        return {
            "time_seconds": round(self._elapsed(), 4),
            "duplicates_found": len(self.predicted_matches & self.bundle.final_positives),
            "recall": round(self._recall(), 6),
            "precision": round(self._precision(), 6),
            "f1": round(self._f1(), 6),
            "pairs_evaluated": len(self.evaluated_pairs),
            "clean_actions_executed": len(self.clean_actions),
            "matching_time": round(self.stage_times["matching"], 6),
            "cleaning_time": round(self.stage_times["cleaning"], 6),
            "scoring_time": round(self.stage_times["scoring"], 6),
            "blocker_update_time": round(self.stage_times["blocker_maintenance"], 6),
            "frontier_size": len(self.frontier),
        }

    def _maybe_trace(self, force: bool = False) -> None:
        elapsed = self._elapsed()
        if force or not self.trace or elapsed - self.last_trace_mark >= TRACE_INTERVAL:
            self.trace.append(self._trace_point())
            self.last_trace_mark = elapsed
            self._update_global_precision_checkpoints(self.trace[-1])

    def _clean_features(self, canopy: Dict, operator: str, canopy_row: Dict) -> Dict:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        texts = []
        canopy_counter = Counter()
        for gid in canopy["gids"]:
            text = self.record_lookup[gid]["primary"]
            texts.append(text)
            canopy_counter.update(tokenize_words(text))
        transformed = [apply_operator(text, operator, self.bundle.domain, self.abbr_df, canopy_counter) for text in texts]
        before_values = [normalize_basic(text) for text in texts]
        after_values = transformed
        before_collisions = len(before_values) - len(set(before_values))
        after_collisions = len(after_values) - len(set(after_values))
        feature_dict = {
            "near_miss_density": float(canopy_row["near_miss_density"]),
            "token_entropy_reduction": float(token_entropy(texts) - token_entropy(transformed)),
            "candidate_starved_frac": float(canopy_row["candidate_starved_frac"]),
            "cross_block_expansion": canopy["size"] / CANOPY_CAP,
            "micro_sim": sum(int(before != after) for before, after in zip(before_values[:20], after_values[:20])) / max(1, min(20, len(after_values))),
            "value_collision_increase": max(0.0, after_collisions - before_collisions),
            "weak_agreement_fraction": max(0.0, 1.0 - float(canopy_row["avg_score"])),
            "margin_shrinkage": max(0.0, self.threshold - float(canopy_row["avg_score"])),
        }
        self._record_stage("scoring", start_wall, start_cpu)
        return feature_dict

    def _pair_driven_clean_specs(self, pair: Tuple[str, str], score: float) -> List[Dict]:
        left_text = self.record_lookup[pair[0]]["primary"]
        right_text = self.record_lookup[pair[1]]["primary"]
        left_tokens = set(tokenize_words(left_text))
        right_tokens = set(tokenize_words(right_text))
        specs = []
        if normalize_basic(left_text) != safe_str(left_text).casefold() or normalize_basic(right_text) != safe_str(right_text).casefold():
            specs.append({"operator": "basic", "tokens": sorted((left_tokens | right_tokens))[:4]})
        if left_tokens and right_tokens and sorted(left_tokens) == sorted(right_tokens) and normalize_basic(left_text) != normalize_basic(right_text):
            specs.append({"operator": "token_sort", "tokens": sorted((left_tokens | right_tokens))[:4]})
        for lt in left_tokens:
            for rt in right_tokens:
                if lt == rt:
                    continue
                if rt in self.abbr_map.get(lt, set()) or lt in self.abbr_map.get(rt, set()):
                    specs.append({"operator": "abbr", "tokens": sorted({lt, rt})})
                    continue
                if len(lt) <= 6 and len(rt) > len(lt) and rt.startswith(lt):
                    specs.append({"operator": "abbr", "tokens": sorted({lt, rt})})
                    continue
                if len(rt) <= 6 and len(lt) > len(rt) and lt.startswith(rt):
                    specs.append({"operator": "abbr", "tokens": sorted({lt, rt})})
                    continue
                if lt[:1] == rt[:1] and abs(len(lt) - len(rt)) <= 2 and Levenshtein.distance(lt, rt) == 1:
                    specs.append({"operator": "typo", "tokens": sorted({lt, rt})})
        if self.bundle.domain == "product":
            if normalize_numeric(left_text, self.bundle.domain) != normalize_basic(left_text) or normalize_numeric(right_text, self.bundle.domain) != normalize_basic(right_text):
                specs.append({"operator": "numeric", "tokens": sorted((left_tokens | right_tokens))[:4]})
        deduped = []
        seen = set()
        for spec in specs:
            key = (spec["operator"], tuple(spec["tokens"]))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(spec)
        return deduped

    def _candidate_clean_actions(self) -> List[Dict]:
        overall_start = time.perf_counter()
        seen_pairs = {pair for pair, _, _ in self.evaluated_pairs}
        near_miss_pairs = []
        for pair, score in self.frontier:
            if pair in seen_pairs:
                continue
            if score < 0.55:
                break
            if score < self.threshold:
                near_miss_pairs.append((pair, score))
            if len(near_miss_pairs) >= 400:
                break
        if not near_miss_pairs:
            self.predicted_decision_cost = max(0.5, time.perf_counter() - overall_start)
            return []
        degree = Counter()
        for left_gid, right_gid in self.candidate_scores:
            degree[left_gid] += 1
            degree[right_gid] += 1
        supported = {}
        for pair, score in near_miss_pairs:
            for spec in self._pair_driven_clean_specs(pair, score):
                gids = {pair[0], pair[1]}
                for token in spec["tokens"]:
                    gids.update(self.token_to_gids.get(token, set()))
                gids = sorted(gids)[:CANOPY_CAP]
                if len(gids) < 2:
                    continue
                if not any(gid.startswith("l:") for gid in gids) or not any(gid.startswith("r:") for gid in gids):
                    continue
                key = (spec["operator"], tuple(spec["tokens"]))
                row = supported.setdefault(
                    key,
                    {
                        "token": "|".join(spec["tokens"]),
                        "gids": gids,
                        "size": len(gids),
                        "near_miss_support": 0,
                        "score_sum": 0.0,
                    },
                )
                row["near_miss_support"] += 1
                row["score_sum"] += score
        if not supported:
            self.predicted_decision_cost = max(0.25, min(time.perf_counter() - overall_start, 5.0))
            return []
        actions = []
        operator_mode = "format_only" if self.ablation == "FormatOnly" else "full"
        for (operator, _), row in sorted(
            supported.items(),
            key=lambda item: (
                -(item[1]["near_miss_support"] / max(1, item[1]["size"])),
                -item[1]["score_sum"],
                item[1]["token"],
            ),
        )[:6]:
            if operator not in operator_families(self.bundle.domain, operator_mode):
                continue
            canopy = {"token": row["token"], "gids": row["gids"], "size": int(row["size"])}
            action_key = (operator, tuple(canopy["gids"]))
            if action_key in self.executed_action_keys:
                continue
            estimated_postings = canopy["size"] * TOP_K
            if canopy["size"] > CANOPY_CAP or estimated_postings > MAX_POSTING_UPDATES:
                continue
            canopy_row = {
                "near_miss_density": row["near_miss_support"] / max(1, canopy["size"]),
                "candidate_starved_frac": sum(1 for gid in canopy["gids"] if degree.get(gid, 0) <= 2) / max(1, canopy["size"]),
                "avg_score": row["score_sum"] / max(1, row["near_miss_support"]),
            }
            features = self._clean_features(canopy, operator, canopy_row)
            estimated_cost = 0.002 * canopy["size"] + 0.0002 * estimated_postings
            actions.append(
                {
                    "canopy": canopy,
                    "operator": operator,
                    "est_cost": max(0.05, min(estimated_cost, 5.0)),
                    "features": features,
                }
            )
        self.predicted_decision_cost = max(0.25, min(time.perf_counter() - overall_start, 5.0))
        return actions[:24]

    def _score_clean_action(self, action: Dict) -> Tuple[float, float, float]:
        features = action["features"]
        if self.method in {"LocalHeuristic", "MutableGreedy"}:
            gain = features["near_miss_density"]
            risk = 0.0
        elif self.method == "MutableRandom":
            gain = 0.0
            risk = 0.0
        else:
            gain_weights = self.estimator.get("gain_weights", {})
            risk_weights = self.estimator.get("risk_weights", {})
            raw_gain = sum(gain_weights.get(key, 0.0) * features.get(key, 0.0) for key in gain_weights)
            raw_risk = sum(risk_weights.get(key, 0.0) * features.get(key, 0.0) for key in risk_weights)
            if self.ablation == "NoMicroSim":
                raw_gain -= gain_weights.get("micro_sim", 0.0) * features.get("micro_sim", 0.0)
            if self.estimator.get("gain_isotonic_x"):
                gain = float(
                    np.interp(
                        raw_gain,
                        self.estimator["gain_isotonic_x"],
                        self.estimator["gain_isotonic_y"],
                        left=self.estimator["gain_isotonic_y"][0],
                        right=self.estimator["gain_isotonic_y"][-1],
                    )
                )
            else:
                gain = raw_gain
            risk = raw_risk
        lam = 0.0 if self.ablation == "NoRisk" else self.estimator.get("lambda", 0.5)
        score = (gain - lam * risk) / max(action["est_cost"], 1e-6)
        return score, gain, risk

    def _apply_operator_to_canopy(self, canopy: Dict, operator: str) -> set:
        touched = set()
        before_after = []
        canopy_counter = Counter()
        for gid in canopy["gids"]:
            canopy_counter.update(tokenize_words(self.record_lookup[gid]["primary"]))
        for gid in canopy["gids"]:
            meta = self.record_lookup[gid]
            frame = meta["frame"]
            idx = meta["idx"]
            primary_col = meta["primary_col"]
            for col in [primary_col] + meta["secondary_cols"]:
                before = frame.at[idx, col]
                after = apply_operator(before, operator, self.bundle.domain, self.abbr_df, canopy_counter)
                if after != before:
                    frame.at[idx, col] = after
                    touched.add(gid)
                    before_after.append(
                        {
                            "gid": gid,
                            "field": col,
                            "before": safe_str(before),
                            "after": safe_str(after),
                        }
                    )
                    if col == primary_col:
                        meta["primary"] = after
        self._rebuild_record_index()
        self._last_before_after = before_after
        return touched

    def _execute_clean_action(self, action: Dict, full_reblock: bool = False) -> None:
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        before_pairs = set(self.candidate_scores)
        before_scores = {pair: score for pair, score in self.candidate_scores.items() if pair[0] in action["canopy"]["gids"] or pair[1] in action["canopy"]["gids"]}
        before_precision = self._precision()
        before_trace_precision = self.trace[-1]["precision"] if self.trace else before_precision
        action_key = (action["operator"], tuple(action["canopy"]["gids"]))
        self.executed_action_keys.add(action_key)
        self._last_before_after = []
        touched = self._apply_operator_to_canopy(action["canopy"], action["operator"])
        if not touched:
            self._log(f"skip no-op clean {action['operator']} on canopy={action['canopy']['token']}")
            return
        if full_reblock:
            self._rebuild_full()
            new_pairs = set(self.candidate_scores) - before_pairs
            postings_updated = len(self.bundle.left) * TOP_K + len(self.bundle.right) * TOP_K
        else:
            new_pairs, postings_updated = self._incremental_refresh(touched)
        gain_pairs = set(new_pairs)
        for pair, score in before_scores.items():
            new_score = self.candidate_scores.get(pair, score)
            if new_score >= self.threshold and score < self.threshold:
                gain_pairs.add(pair)
            elif new_score - score >= 0.05:
                gain_pairs.add(pair)
        unlocked_true_matches = len(gain_pairs & self.bundle.final_positives)
        corruption_counts = Counter()
        for gid in touched:
            corruption_counts.update(self.corruption_by_gid.get(gid, ["none"]))
        before_after = self._last_before_after[:20]
        new_pair_examples = sorted([list(pair) for pair in gain_pairs])[:20]
        self._record_stage("cleaning", start_wall, start_cpu)
        clean_record = {
            "time_seconds": round(self._elapsed(), 6),
            "token": action["canopy"]["token"],
            "operator_family": action["operator"],
            "records_touched": len(touched),
            "postings_updated": int(postings_updated),
            "new_candidate_pairs": int(len(new_pairs)),
            "new_true_matches": int(unlocked_true_matches),
            "exposed_pairs_evaluated": 0,
            "precision_before": round(before_precision, 6),
            "precision_after_window": round(before_precision, 6),
            "global_precision_before": round(before_trace_precision, 6),
            "global_precision_next_checkpoint": round(before_trace_precision, 6),
            "elapsed": round(time.perf_counter() - start_wall, 6),
            "predicted_gain": round(action.get("predicted_gain", 0.0), 6),
            "predicted_risk": round(action.get("predicted_risk", 0.0), 6),
            "predicted_net_score": round(action.get("predicted_score", 0.0), 6),
            "true_matches_within_1000_pairs": 0,
            "corruption_families": json.dumps(dict(sorted(corruption_counts.items())), sort_keys=True),
            "dominant_corruption_family": sorted(corruption_counts.items(), key=lambda item: (-item[1], item[0]))[0][0] if corruption_counts else "none",
            "affected_records_json": json.dumps(before_after, ensure_ascii=True),
            "new_pair_examples_json": json.dumps(new_pair_examples, ensure_ascii=True),
            **{f"feat_{key}": round(float(value), 6) for key, value in action["features"].items()},
        }
        self.predicted_clean_action_cost = max(1.0, time.perf_counter() - start_wall)
        self.clean_actions.append(clean_record)
        self._clean_internal.append(
            {
                "unlocked_pairs": gain_pairs,
                "eval_start": len(self.evaluated_pairs),
                "precision_before": before_precision,
                "window_pairs_seen": 0,
                "global_checkpoint_recorded": False,
                "window_1000_recorded": False,
                "checkpoint_recorded": False,
                "index": len(self.clean_actions) - 1,
            }
        )
        self._log(
            f"clean {action['operator']} on canopy={action['canopy']['token']} touched={len(touched)} unlocked={len(new_pairs)} full_reblock={full_reblock}"
        )

    def _update_clean_action_outcomes(self, evaluated_pairs: List[Tuple[str, str]]) -> None:
        for internal in self._clean_internal:
            action = self.clean_actions[internal["index"]]
            for pair in evaluated_pairs:
                internal["window_pairs_seen"] += 1
                if pair in internal["unlocked_pairs"]:
                    action["exposed_pairs_evaluated"] += 1
                    if pair in self.bundle.final_positives:
                        action["new_true_matches"] += 1
                        if internal["window_pairs_seen"] <= 1000:
                            action["true_matches_within_1000_pairs"] += 1
                if not internal["checkpoint_recorded"] and internal["window_pairs_seen"] >= 500:
                    action["precision_after_window"] = round(self._precision(), 6)
                    internal["checkpoint_recorded"] = True
                if not internal["window_1000_recorded"] and internal["window_pairs_seen"] >= 1000:
                    internal["window_1000_recorded"] = True
            if not internal["checkpoint_recorded"] and internal["window_pairs_seen"] == 0:
                action["precision_after_window"] = round(self._precision(), 6)

    def _update_global_precision_checkpoints(self, trace_row: Dict) -> None:
        for internal in self._clean_internal:
            if internal["global_checkpoint_recorded"]:
                continue
            action = self.clean_actions[internal["index"]]
            if trace_row["time_seconds"] > action["time_seconds"]:
                action["global_precision_next_checkpoint"] = round(float(trace_row["precision"]), 6)
                internal["global_checkpoint_recorded"] = True

    def _select_hybrid_batch(self) -> List[Tuple[Tuple[str, str], float]]:
        seen_pairs = {pair for pair, _, _ in self.evaluated_pairs}
        remaining = [(pair, score) for pair, score in self.frontier if pair not in seen_pairs]
        if not remaining:
            return []
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        pair_scores = np.array([score for _, score in remaining], dtype=float)
        scaler = MinMaxScaler()
        pair_priority = scaler.fit_transform(pair_scores.reshape(-1, 1)).reshape(-1)
        node_max = defaultdict(float)
        for pair, score in remaining:
            node_max[pair[0]] = max(node_max[pair[0]], score)
            node_max[pair[1]] = max(node_max[pair[1]], score)
        node_values = np.array([0.5 * (node_max[pair[0]] + node_max[pair[1]]) for pair, _ in remaining], dtype=float)
        node_priority = scaler.fit_transform(node_values.reshape(-1, 1)).reshape(-1)
        hybrid_score = 0.7 * pair_priority + 0.3 * node_priority
        selected_idx = np.argsort(-hybrid_score)[:MATCH_BATCH_SIZE]
        self._record_stage("scoring", start_wall, start_cpu)
        return [remaining[idx] for idx in selected_idx]

    def _global_clean_then_rebuild(self) -> None:
        self._log("starting FullClean+PEM global canonicalization")
        all_gids = self.bundle.left["gid"].tolist() + self.bundle.right["gid"].tolist()
        canopy = {"token": "global", "gids": all_gids, "size": len(all_gids)}
        for operator in operator_families(self.bundle.domain):
            if self.budget_seconds - self._elapsed() < self.predicted_clean_action_cost:
                self._log("stopping FullClean+PEM before another global operator due to remaining budget")
                break
            action = {
                "canopy": canopy,
                "operator": operator,
                "features": {key: 0.0 for key in [
                    "near_miss_density",
                    "token_entropy_reduction",
                    "candidate_starved_frac",
                    "cross_block_expansion",
                    "micro_sim",
                    "value_collision_increase",
                    "weak_agreement_fraction",
                    "margin_shrinkage",
                ]},
                "predicted_gain": 0.0,
                "predicted_risk": 0.0,
                "predicted_score": 0.0,
            }
            self._execute_clean_action(action, full_reblock=True)

    def run(self) -> Dict:
        self._cpu_stage_times = defaultdict(float)
        self._init_cpu_stage_times = defaultdict(float)
        self.total_start = time.perf_counter()
        self.total_process_start = time.process_time()
        self._sampler.start()
        self._log(
            f"run start setting={self.bundle.setting} seed={self.bundle.seed} method={self.method} threshold={self.threshold:.2f} budget={self.budget_seconds}"
        )
        try:
            self._rebuild_full(init=True)
            self.run_start = time.perf_counter()
            self.process_start = time.process_time()
            full_clean_done = False
            reference_idx = 0
            stop_reason = "budget_exhausted"
            while self._elapsed() < self.budget_seconds:
                self._maybe_trace()
                remaining = self.budget_seconds - self._elapsed()
                batch = self._match_batch_candidates()
                if self.method == "FullClean+PEM" and not full_clean_done:
                    if remaining < self.predicted_clean_action_cost:
                        self._log("insufficient budget remaining for FullClean+PEM global cleaning")
                        stop_reason = "insufficient_budget_for_full_clean"
                        break
                    self._global_clean_then_rebuild()
                    full_clean_done = True
                    self._maybe_trace(force=True)
                    continue
                if self.method == "RawPEM":
                    if remaining < self.predicted_match_batch_cost:
                        self._log("stopping RawPEM because remaining budget is below predicted match-batch cost")
                        stop_reason = "insufficient_budget_for_match"
                        break
                    if not batch:
                        stop_reason = "frontier_exhausted"
                        break
                    self._evaluate_pairs(batch, "rawpem")
                    continue
                if self.method == "HybridStatic":
                    if remaining < max(self.predicted_match_batch_cost, 1.0):
                        self._log("stopping HybridStatic because remaining budget is below predicted batch cost")
                        stop_reason = "insufficient_budget_for_match"
                        break
                    batch = self._select_hybrid_batch()
                    if not batch:
                        stop_reason = "frontier_exhausted"
                        break
                    self._evaluate_pairs(batch, "hybridstatic")
                    continue
                if remaining < self.predicted_decision_cost:
                    self._log("stopping mutable scheduler because remaining budget is below predicted decision cost")
                    stop_reason = "insufficient_budget_for_decision"
                    break
                clean_actions = self._candidate_clean_actions()
                remaining = self.budget_seconds - self._elapsed()
                if self.ablation == "FullReblock" and reference_idx < len(self.reference_actions):
                    target_token = self.reference_actions[reference_idx]["token"]
                    target_operator = self.reference_actions[reference_idx]["operator_family"]
                    replay = next(
                        (action for action in clean_actions if action["canopy"]["token"] == target_token and action["operator"] == target_operator),
                        None,
                    )
                    if replay is not None and remaining >= self.predicted_clean_action_cost:
                        self._execute_clean_action(replay, full_reblock=True)
                        reference_idx += 1
                        continue
                scored_clean = []
                for action in clean_actions:
                    score, gain, risk = self._score_clean_action(action)
                    action["predicted_score"] = score
                    action["predicted_gain"] = gain
                    action["predicted_risk"] = risk
                    scored_clean.append(action)
                scored_clean.sort(key=lambda item: (-item["predicted_score"], item["canopy"]["token"], item["operator"]))
                best_clean = scored_clean[0] if scored_clean else None
                if self.method == "LocalHeuristic":
                    if best_clean is not None and remaining >= self.predicted_clean_action_cost:
                        self._execute_clean_action(best_clean, full_reblock=self.ablation == "FullReblock")
                        continue
                    if best_clean is not None and remaining < self.predicted_clean_action_cost:
                        self._log("stopping LocalHeuristic because remaining budget is below predicted clean-action cost")
                        stop_reason = "insufficient_budget_for_clean"
                        break
                    if remaining < self.predicted_match_batch_cost:
                        self._log("stopping LocalHeuristic because remaining budget is below predicted match-batch cost")
                        stop_reason = "insufficient_budget_for_match"
                        break
                    if not batch:
                        stop_reason = "frontier_exhausted"
                        break
                    self._evaluate_pairs(batch, "localheuristic-match")
                    continue
                if self.method == "MutableRandom":
                    rng = random.Random(self.bundle.seed + len(self.evaluated_pairs) + len(self.clean_actions))
                    if best_clean is not None and remaining >= self.predicted_clean_action_cost and (not batch or rng.random() < 0.5):
                        top_actions = scored_clean[:6] if scored_clean else []
                        chosen = rng.choice(top_actions) if top_actions else None
                        if chosen is not None:
                            self._execute_clean_action(chosen, full_reblock=self.ablation == "FullReblock")
                            continue
                    if remaining < self.predicted_match_batch_cost:
                        self._log("stopping MutableRandom because remaining budget is below predicted action cost")
                        stop_reason = "insufficient_budget_for_match"
                        break
                    if not batch:
                        stop_reason = "frontier_exhausted"
                        break
                    self._evaluate_pairs(batch, "mutablerandom-match")
                    continue
                next_match_score = (float(np.mean([score for _, score in batch])) if batch else 0.0) / max(self.predicted_match_batch_cost, 1e-6)
                if (
                    self.method in {"MutableGreedy", "CanopyER"}
                    and best_clean is not None
                    and remaining >= self.predicted_clean_action_cost
                    and best_clean["predicted_score"] > next_match_score
                ):
                    self._execute_clean_action(best_clean, full_reblock=self.ablation == "FullReblock")
                    continue
                if remaining < self.predicted_match_batch_cost:
                    self._log(f"stopping {self.method} because remaining budget is below predicted match-batch cost")
                    stop_reason = "insufficient_budget_for_match"
                    break
                if not batch:
                    stop_reason = "frontier_exhausted"
                    break
                self._evaluate_pairs(batch, f"{self.method.lower()}-match")
            self._maybe_trace(force=True)
        finally:
            peak_rss = self._sampler.stop()
        for internal in self._clean_internal:
            if not internal["checkpoint_recorded"]:
                self.clean_actions[internal["index"]]["precision_after_window"] = round(self._precision(), 6)
            if not internal["global_checkpoint_recorded"]:
                self.clean_actions[internal["index"]]["global_precision_next_checkpoint"] = round(self._precision(), 6)
        online_wall_clock_seconds = time.perf_counter() - self.run_start
        online_cpu_process_seconds = time.process_time() - self.process_start
        wall_clock_seconds = time.perf_counter() - self.total_start
        cpu_process_seconds = time.process_time() - self.total_process_start
        tracked_wall = sum(self.stage_times.values())
        tracked_cpu = sum(self._cpu_stage_times.values())
        if online_wall_clock_seconds > tracked_wall:
            self.stage_times["other"] += online_wall_clock_seconds - tracked_wall
        if online_cpu_process_seconds > tracked_cpu:
            self._cpu_stage_times["other"] += online_cpu_process_seconds - tracked_cpu
        metrics = compute_metrics(
            self.trace,
            self.bundle.final_positives,
            self.clean_actions,
            self.stage_times,
            self.budget_seconds,
            stop_reason,
        )
        self._log(
            f"run end wall={wall_clock_seconds:.3f}s cpu={cpu_process_seconds:.3f}s pairs={len(self.evaluated_pairs)} predicted_matches={len(self.predicted_matches)}"
        )
        return {
            "setting": self.bundle.setting,
            "seed": self.bundle.seed,
            "method": self.ablation or self.method,
            "metrics": metrics,
            "trace": self.trace,
            "clean_actions": self.clean_actions,
            "stdout_lines": self.stdout_lines,
            "system": {
                "peak_rss_mb": round(peak_rss, 6),
                "wall_clock_seconds": round(wall_clock_seconds, 6),
                "cpu_process_seconds": round(cpu_process_seconds, 6),
                "online_wall_clock_seconds": round(online_wall_clock_seconds, 6),
                "online_cpu_process_seconds": round(online_cpu_process_seconds, 6),
                "initialization_wall_clock_seconds": round(wall_clock_seconds - online_wall_clock_seconds, 6),
                "initialization_cpu_process_seconds": round(cpu_process_seconds - online_cpu_process_seconds, 6),
                "stage_times": {key: round(value, 6) for key, value in sorted(self.stage_times.items())},
                "stage_cpu_times": {key: round(value, 6) for key, value in sorted(self._cpu_stage_times.items())},
                "initialization_stage_times": {key: round(value, 6) for key, value in sorted(self.init_stage_times.items())},
                "initialization_stage_cpu_times": {key: round(value, 6) for key, value in sorted(self._init_cpu_stage_times.items())},
                "cpu_worker_limit": CPU_WORKER_LIMIT,
                "gpus": 0,
                "python": os.sys.version,
                "python_executable": os.sys.executable,
                "stop_reason": stop_reason,
            },
        }


def compute_metrics(
    trace: Sequence[Dict],
    final_positives: set,
    clean_actions: Sequence[Dict],
    stage_times: Dict[str, float],
    budget_seconds: float,
    stop_reason: str,
) -> Dict:
    xs = [0.0] + [row["time_seconds"] for row in trace]
    ys = [0.0] + [row["duplicates_found"] for row in trace]
    auc = 0.0
    for idx in range(1, len(xs)):
        auc += (xs[idx] - xs[idx - 1]) * (ys[idx] + ys[idx - 1]) / 2.0
    normalized_auc = auc / max(1.0, budget_seconds * max(1, len(final_positives)))

    def metric_at(target: float, key: str) -> float:
        value = trace[-1][key] if trace else 0.0
        for row in trace:
            if row["time_seconds"] >= target:
                value = row[key]
                break
        return float(value)

    frontier_exhausted_at_seconds = float(trace[-1]["time_seconds"]) if trace else 0.0
    frontier_exhausted = stop_reason == "frontier_exhausted"
    if not frontier_exhausted:
        frontier_exhausted_at_seconds = float(budget_seconds)

    harmful = [
        action
        for action in clean_actions
        if (action["precision_after_window"] - action["precision_before"]) < -0.05
        or (action["global_precision_next_checkpoint"] - action["global_precision_before"]) < -0.03
    ]
    wasteful = [
        action
        for action in clean_actions
        if action["elapsed"] > 10.0 and action.get("true_matches_within_1000_pairs", action["new_true_matches"]) == 0
    ]
    total_runtime = max(1e-6, sum(stage_times.values()))
    overhead = stage_times.get("scoring", 0.0) + stage_times.get("cleaning", 0.0) + stage_times.get("blocker_maintenance", 0.0)
    return {
        "normalized_auc": round(normalized_auc, 6),
        "recall_at_60s": round(metric_at(60.0, "recall"), 6),
        "recall_at_180s": round(metric_at(180.0, "recall"), 6),
        "recall_at_360s": round(metric_at(360.0, "recall"), 6),
        "final_precision": round(metric_at(360.0, "precision"), 6),
        "final_recall": round(metric_at(360.0, "recall"), 6),
        "final_f1": round(metric_at(360.0, "f1"), 6),
        "clean_actions_executed": len(clean_actions),
        "mean_records_touched": round(float(np.mean([action["records_touched"] for action in clean_actions])) if clean_actions else 0.0, 6),
        "mean_postings_updated": round(float(np.mean([action["postings_updated"] for action in clean_actions])) if clean_actions else 0.0, 6),
        "new_candidate_pairs_unlocked": int(sum(action["new_candidate_pairs"] for action in clean_actions)),
        "new_true_matches_unlocked": int(sum(action["new_true_matches"] for action in clean_actions)),
        "harmful_actions": len(harmful),
        "wasteful_actions": len(wasteful),
        "overhead_fraction": round(overhead / total_runtime, 6),
        "frontier_exhausted": frontier_exhausted,
        "frontier_exhausted_at_seconds": round(frontier_exhausted_at_seconds, 6),
    }


def source_rows_from_result(bundle: DatasetBundle, result: Dict) -> List[Dict]:
    rows = []
    for action in result["clean_actions"]:
        rows.append(
            {
                "setting": bundle.setting,
                "seed": bundle.seed,
                "near_miss_density": action.get("feat_near_miss_density", 0.0),
                "token_entropy_reduction": action.get("feat_token_entropy_reduction", 0.0),
                "candidate_starved_frac": action.get("feat_candidate_starved_frac", 0.0),
                "cross_block_expansion": action.get("feat_cross_block_expansion", 0.0),
                "micro_sim": action.get("feat_micro_sim", 0.0),
                "value_collision_increase": action.get("feat_value_collision_increase", 0.0),
                "weak_agreement_fraction": action.get("feat_weak_agreement_fraction", 0.0),
                "margin_shrinkage": action.get("feat_margin_shrinkage", 0.0),
                "gain_target": action["new_true_matches"],
                "risk_target": max(
                    0.0,
                    action["precision_before"] - action["precision_after_window"],
                    action["global_precision_before"] - action["global_precision_next_checkpoint"],
                ),
            }
        )
    return rows


def fit_estimators_from_rows(rows: Sequence[Dict]) -> Dict[str, Dict]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    gain_cols = ["near_miss_density", "token_entropy_reduction", "candidate_starved_frac", "cross_block_expansion", "micro_sim"]
    risk_cols = ["value_collision_increase", "weak_agreement_fraction", "margin_shrinkage"]
    estimators = {}
    for target_setting in sorted(df["setting"].unique()):
        train = df[df["setting"] != target_setting].copy()
        if train.empty:
            gain_weights = {col: 0.0 for col in gain_cols}
            risk_weights = {col: 0.0 for col in risk_cols}
            iso_x = [0.0, 1.0]
            iso_y = [0.0, 1.0]
        else:
            gain_model = LinearRegression(positive=True)
            risk_model = LinearRegression(positive=True)
            gain_model.fit(train[gain_cols], train["gain_target"])
            risk_model.fit(train[risk_cols], train["risk_target"])
            raw_gain = gain_model.predict(train[gain_cols])
            isotonic = IsotonicRegression(out_of_bounds="clip")
            isotonic.fit(raw_gain, train["gain_target"])
            grid = np.linspace(float(raw_gain.min()), float(raw_gain.max() if len(raw_gain) else 1.0), 32)
            gain_weights = {col: float(weight) for col, weight in zip(gain_cols, gain_model.coef_)}
            risk_weights = {col: float(weight) for col, weight in zip(risk_cols, risk_model.coef_)}
            iso_x = [float(x) for x in grid]
            iso_y = [float(y) for y in isotonic.predict(grid)]
        estimators[target_setting] = {
            "gain_weights": gain_weights,
            "risk_weights": risk_weights,
            "gain_isotonic_x": iso_x,
            "gain_isotonic_y": iso_y,
            "lambda": 0.5,
        }
        write_json(ARTIFACTS_DIR / f"estimator_{target_setting}.json", estimators[target_setting])
    return estimators


def persist_run_outputs(run_dir: Path, result: Dict, config: Dict) -> None:
    ensure_dir(run_dir)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "metrics.json", result["metrics"])
    write_json(run_dir / "system.json", result["system"])
    pd.DataFrame(result["trace"]).to_csv(run_dir / "trace.csv", index=False)
    clean_df = pd.DataFrame(result["clean_actions"])
    if clean_df.empty:
        clean_df = pd.DataFrame(columns=[
            "time_seconds",
            "token",
            "operator_family",
            "records_touched",
            "postings_updated",
            "new_candidate_pairs",
            "new_true_matches",
            "exposed_pairs_evaluated",
            "precision_before",
            "precision_after_window",
            "elapsed",
            "predicted_gain",
            "predicted_risk",
            "predicted_net_score",
        ])
    clean_df.to_csv(run_dir / "actions.csv", index=False)
    pd.DataFrame(
        [{"stage": stage, "wall_seconds": seconds, "cpu_seconds": result["system"]["stage_cpu_times"].get(stage, 0.0)} for stage, seconds in result["system"]["stage_times"].items()]
    ).to_csv(run_dir / "timings.csv", index=False)
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(result["stdout_lines"]) + "\n")


def permutation_pvalue(deltas: Sequence[float], trials: int = 2000, seed: int = 7) -> float:
    rng = random.Random(seed)
    observed = abs(sum(deltas))
    count = 0
    deltas = list(deltas)
    for _ in range(trials):
        shuffled = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        if abs(sum(shuffled)) >= observed:
            count += 1
    return (count + 1) / (trials + 1)


def aggregate_results() -> Dict:
    ensure_dir(TABLES_DIR)
    results_rows = []
    trace_frames = []
    action_frames = []
    main_methods = {"RawPEM", "HybridStatic", "FullClean+PEM", "LocalHeuristic", "MutableGreedy", "MutableRandom", "CanopyER"}
    ablation_methods = {"NoMicroSim", "NoRisk", "FormatOnly", "FullReblock"}
    for metrics_path in RUNS_DIR.glob("*/*/*/metrics.json"):
        run_dir = metrics_path.parent
        setting, method, seed = run_dir.parts[-3:]
        metrics = read_json(metrics_path)
        system = read_json(run_dir / "system.json")
        row = {"setting": setting, "method": method, "seed": int(seed), **metrics}
        row.update({f"system_{key}": value for key, value in system.items() if key not in {"stage_times", "stage_cpu_times"}})
        results_rows.append(row)
        trace_path = run_dir / "trace.csv"
        if trace_path.exists():
            df = pd.read_csv(trace_path)
            df["setting"] = setting
            df["method"] = method
            df["seed"] = int(seed)
            trace_frames.append(df)
        action_path = run_dir / "actions.csv"
        if action_path.exists():
            try:
                df = pd.read_csv(action_path)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            if not df.empty:
                df["setting"] = setting
                df["method"] = method
                df["seed"] = int(seed)
                action_frames.append(df)
    if not results_rows:
        summary = {
            "machine": machine_metadata(),
            "archive_note": "no completed runs found under runs/",
            "main_results": [],
            "ablation_results": [],
            "paired_tests": [],
            "failure_summary": [],
            "case_studies": [],
        }
        write_json(ROOT / "results.json", summary)
        return summary

    all_runs = pd.DataFrame(results_rows).sort_values(["setting", "method", "seed"])
    all_runs.to_csv(TABLES_DIR / "all_run_metrics.csv", index=False)
    if trace_frames:
        pd.concat(trace_frames, ignore_index=True).to_csv(TABLES_DIR / "all_traces.csv", index=False)
    if action_frames:
        pd.concat(action_frames, ignore_index=True).to_csv(TABLES_DIR / "all_actions.csv", index=False)

    main_only = all_runs[all_runs["method"].isin(main_methods)].copy()
    summary_rows = []
    for (setting, method), group in main_only.groupby(["setting", "method"]):
        summary = {"setting": setting, "method": method}
        for metric in [
            "normalized_auc",
            "recall_at_60s",
            "recall_at_180s",
            "recall_at_360s",
            "final_f1",
            "final_precision",
            "overhead_fraction",
            "frontier_exhausted_at_seconds",
        ]:
            mean = float(group[metric].mean())
            std = float(group[metric].std(ddof=0))
            ci95 = 1.96 * std / math.sqrt(max(1, len(group)))
            summary[f"{metric}_mean"] = round(mean, 6)
            summary[f"{metric}_std"] = round(std, 6)
            summary[f"{metric}_ci95"] = round(ci95, 6)
        summary["frontier_exhaustion_rate"] = round(float(group["frontier_exhausted"].mean()), 6)
        summary_rows.append(summary)
    main_table = pd.DataFrame(summary_rows).sort_values(["setting", "method"])
    main_table.to_csv(TABLES_DIR / "main_results.csv", index=False)

    ablation_only = all_runs[all_runs["method"].isin(ablation_methods)].copy()
    ablation_table = pd.DataFrame()
    if not ablation_only.empty:
        ablation_only.to_csv(TABLES_DIR / "ablation_results.csv", index=False)
        canopy_reference = main_only[main_only["method"] == "CanopyER"][
            ["setting", "seed", "normalized_auc", "recall_at_180s", "final_precision", "final_f1", "mean_postings_updated", "overhead_fraction"]
        ].rename(
            columns={
                "normalized_auc": "ref_normalized_auc",
                "recall_at_180s": "ref_recall_at_180s",
                "final_precision": "ref_final_precision",
                "final_f1": "ref_final_f1",
                "mean_postings_updated": "ref_mean_postings_updated",
                "overhead_fraction": "ref_overhead_fraction",
            }
        )
        ablation_joined = ablation_only.merge(canopy_reference, on=["setting", "seed"], how="left")
        delta_rows = []
        for (setting, method), group in ablation_joined.groupby(["setting", "method"]):
            delta_rows.append(
                {
                    "setting": setting,
                    "method": method,
                    "delta_normalized_auc_mean": round(float((group["normalized_auc"] - group["ref_normalized_auc"]).mean()), 6),
                    "delta_normalized_auc_std": round(float((group["normalized_auc"] - group["ref_normalized_auc"]).std(ddof=0)), 6),
                    "delta_recall_at_180s_mean": round(float((group["recall_at_180s"] - group["ref_recall_at_180s"]).mean()), 6),
                    "delta_final_precision_mean": round(float((group["final_precision"] - group["ref_final_precision"]).mean()), 6),
                    "delta_final_f1_mean": round(float((group["final_f1"] - group["ref_final_f1"]).mean()), 6),
                    "blocker_maintenance_cost_ratio_mean": round(
                        float((group["mean_postings_updated"] / group["ref_mean_postings_updated"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0).mean()),
                        6,
                    ),
                    "overhead_fraction_delta_mean": round(float((group["overhead_fraction"] - group["ref_overhead_fraction"]).mean()), 6),
                }
            )
        ablation_table = pd.DataFrame(delta_rows).sort_values(["setting", "method"])
        ablation_table.to_csv(TABLES_DIR / "ablation_deltas.csv", index=False)

    effect_tests = []
    for baseline in ["RawPEM", "HybridStatic", "LocalHeuristic", "MutableGreedy"]:
        deltas = []
        for setting, group in main_only[main_only["method"].isin(["CanopyER", baseline])].groupby("setting"):
            pivot = group.pivot_table(index="seed", columns="method", values="normalized_auc")
            if "CanopyER" in pivot.columns and baseline in pivot.columns:
                deltas.append(float((pivot["CanopyER"] - pivot[baseline]).mean()))
        if deltas:
            effect_tests.append(
                {
                    "baseline": baseline,
                    "mean_auc_delta": round(float(np.mean(deltas)), 6),
                    "std_auc_delta": round(float(np.std(deltas, ddof=0)), 6),
                    "n_settings": len(deltas),
                    "p_value": round(permutation_pvalue(deltas), 6),
                }
            )
    write_json(ARTIFACTS_DIR / "paired_tests.json", effect_tests)

    failure_summary = []
    case_studies = []
    estimator_diagnostics = []
    novelty_positioning = [
        {
            "system": "Detective Gadget",
            "primary_objective": "Iterative ER quality improvement on dirty data",
            "action_granularity": "repair / alias / check functions",
            "why_candidate_graph_changes": "iterative repair updates hashes and aliases",
            "candidate_graph_mutable_during_run": True,
            "stopping_regime": "iterate to fixpoint or user stop",
        },
        {
            "system": "Iterative Blocking",
            "primary_objective": "Recover more matches by propagating block evidence",
            "action_granularity": "block-level iterative propagation",
            "why_candidate_graph_changes": "propagated evidence mutates block interactions",
            "candidate_graph_mutable_during_run": True,
            "stopping_regime": "iterate until convergence",
        },
        {
            "system": "Progressive ER over Incremental Data",
            "primary_objective": "Maximize early utility as new data increments arrive",
            "action_granularity": "prioritized comparisons after arrivals",
            "why_candidate_graph_changes": "new records arrive",
            "candidate_graph_mutable_during_run": True,
            "stopping_regime": "budgeted execution between increments",
        },
        {
            "system": "Incremental Entity Blocking over Heterogeneous Streaming Data",
            "primary_objective": "Maintain blocks efficiently under noisy arrivals",
            "action_granularity": "incremental block updates",
            "why_candidate_graph_changes": "streaming insertions update blocks",
            "candidate_graph_mutable_during_run": True,
            "stopping_regime": "continuous streaming",
        },
        {
            "system": "ProgressER",
            "primary_objective": "Maximize early ER utility under budget",
            "action_granularity": "block / pair scheduling",
            "why_candidate_graph_changes": "representation fixed",
            "candidate_graph_mutable_during_run": False,
            "stopping_regime": "budgeted progressive execution",
        },
        {
            "system": "Progressive EM Design Space (2025)",
            "primary_objective": "Characterize progressive filtering and scheduling pipelines",
            "action_granularity": "edge / node / hybrid scheduling",
            "why_candidate_graph_changes": "representation fixed",
            "candidate_graph_mutable_during_run": False,
            "stopping_regime": "budgeted progressive execution",
        },
        {
            "system": "CanopyER",
            "primary_objective": "Choose between matching and local rewrite under budget",
            "action_granularity": "match batch versus canopy-local clean action",
            "why_candidate_graph_changes": "scheduler executes local canonicalization",
            "candidate_graph_mutable_during_run": True,
            "stopping_regime": "budgeted progressive execution",
        },
    ]
    pd.DataFrame(novelty_positioning).to_csv(TABLES_DIR / "novelty_positioning.csv", index=False)
    if action_frames:
        action_df = pd.concat(action_frames, ignore_index=True)
        canopy_action_df = action_df[action_df["method"] == "CanopyER"].copy()
        if "corruption_families" in canopy_action_df.columns:
            canopy_action_df["corruption_family_list"] = canopy_action_df["corruption_families"].fillna("{}").map(
                lambda value: sorted(ast.literal_eval(value).keys()) if str(value).strip() else ["none"]
            )
        else:
            canopy_action_df["corruption_family_list"] = [["none"]] * len(canopy_action_df)
        for (setting, seed), group in canopy_action_df.groupby(["setting", "seed"]):
            valid = group[["predicted_gain", "new_true_matches"]].dropna()
            corr = 0.0
            if len(valid) >= 2 and valid["predicted_gain"].nunique() > 1 and valid["new_true_matches"].nunique() > 1:
                corr = float(spearmanr(valid["predicted_gain"], valid["new_true_matches"]).statistic)
            ranked = group.sort_values("predicted_gain")
            estimator_path = ARTIFACTS_DIR / f"estimator_{setting}.json"
            risk_model_collapsed = False
            if estimator_path.exists():
                estimator_blob = read_json(estimator_path)
                risk_model_collapsed = all(abs(float(weight)) < 1e-12 for weight in estimator_blob.get("risk_weights", {}).values())
            if not ranked.empty:
                top_n = max(1, int(math.ceil(len(ranked) * 0.1)))
                med_start = max(0, len(ranked) // 2 - top_n // 2)
                med_slice = ranked.iloc[med_start : med_start + top_n]
                estimator_diagnostics.append(
                    {
                        "setting": setting,
                        "seed": int(seed),
                        "spearman_gain_vs_realized": round(corr, 6),
                        "top_decile_realized_gain": round(float(ranked.tail(top_n)["new_true_matches"].mean()), 6),
                        "median_band_realized_gain": round(float(med_slice["new_true_matches"].mean()), 6),
                        "top_minus_median": round(
                            float(ranked.tail(top_n)["new_true_matches"].mean() - med_slice["new_true_matches"].mean()),
                            6,
                        ),
                        "risk_model_collapsed": risk_model_collapsed,
                    }
                )
        pd.DataFrame(estimator_diagnostics).to_csv(TABLES_DIR / "estimator_diagnostics.csv", index=False)
        canopy_bucket = pd.cut(action_df["records_touched"], bins=[0, 10, 20, 40, 80, 100000], include_lowest=True)
        canopy_action_df["canopy_bucket"] = canopy_bucket[action_df["method"] == "CanopyER"].astype(str)
        canopy_action_df["harmful"] = (
            (canopy_action_df["precision_after_window"] - canopy_action_df["precision_before"]) < -0.05
        ) | (
            (canopy_action_df["global_precision_next_checkpoint"] - canopy_action_df["global_precision_before"]) < -0.03
        )
        canopy_action_df["wasteful"] = (canopy_action_df["elapsed"] > 10.0) & (
            canopy_action_df["true_matches_within_1000_pairs"] == 0
        )
        canopy_action_df["net_utility"] = canopy_action_df["new_true_matches"] - 0.5 * canopy_action_df["predicted_risk"] - canopy_action_df["elapsed"] / 10.0
        exploded = canopy_action_df.explode("corruption_family_list")
        grouped = exploded.groupby(["operator_family", "canopy_bucket", "corruption_family_list"], dropna=False)
        for (operator, bucket, corruption_family), group in grouped:
            failure_summary.append(
                {
                    "operator_family": str(operator),
                    "canopy_bucket": str(bucket),
                    "corruption_family": str(corruption_family),
                    "harmful_rate": round(float(group["harmful"].mean()), 6),
                    "wasteful_rate": round(float(group["wasteful"].mean()), 6),
                    "count": int(len(group)),
                }
            )
        case_studies = canopy_action_df.sort_values("net_utility").head(5).to_dict(orient="records")
        pd.DataFrame(failure_summary).to_csv(TABLES_DIR / "failure_summary.csv", index=False)
        pd.DataFrame(case_studies).to_csv(TABLES_DIR / "failure_case_studies.csv", index=False)

    systems_rows = []
    for system_path in RUNS_DIR.glob("*/*/*/system.json"):
        run_dir = system_path.parent
        setting, method, seed = run_dir.parts[-3:]
        if method not in {"CanopyER", "FullClean+PEM", "FullReblock"}:
            continue
        if setting not in {"amazon_google_corrupted", "dblp_acm"}:
            continue
        system = read_json(system_path)
        stage_times = system.get("stage_times", {})
        for stage in ["matching", "scoring", "cleaning", "blocker_maintenance"]:
            systems_rows.append(
                {
                    "setting": setting,
                    "method": method,
                    "seed": int(seed),
                    "stage": stage,
                    "seconds": float(stage_times.get(stage, 0.0)),
                }
            )
    if systems_rows:
        pd.DataFrame(systems_rows).to_csv(TABLES_DIR / "systems_breakdown.csv", index=False)

    coverage_rows = []
    for setting in MAIN_SETTINGS:
        for method in sorted(main_methods):
            subset = main_only[(main_only["setting"] == setting) & (main_only["method"] == method)]
            coverage_rows.append(
                {
                    "setting": setting,
                    "method": method,
                    "completed_seeds": int(subset["seed"].nunique()),
                    "expected_seeds": len(SEEDS),
                    "complete": bool(subset["seed"].nunique() == len(SEEDS)),
                }
            )
    coverage = pd.DataFrame(coverage_rows).sort_values(["setting", "method"])
    coverage.to_csv(TABLES_DIR / "coverage_main.csv", index=False)

    frontier_summary = []
    for (setting, method), group in main_only.groupby(["setting", "method"]):
        frontier_summary.append(
            {
                "setting": setting,
                "method": method,
                "frontier_exhaustion_rate": round(float(group["frontier_exhausted"].mean()), 6),
                "mean_frontier_exhausted_at_seconds": round(float(group["frontier_exhausted_at_seconds"].mean()), 6),
            }
        )
    pd.DataFrame(frontier_summary).to_csv(TABLES_DIR / "frontier_exhaustion_summary.csv", index=False)

    claim_assessment = {
        "supports_broad_performance_claim": False,
        "supports_mutable_action_feasibility_claim": False,
        "reason": "",
    }
    can = main_table[main_table["method"] == "CanopyER"]
    hybrid = main_table[main_table["method"] == "HybridStatic"][["setting", "normalized_auc_mean"]].rename(
        columns={"normalized_auc_mean": "hybrid_normalized_auc_mean"}
    )
    local = main_table[main_table["method"] == "LocalHeuristic"][["setting", "normalized_auc_mean"]].rename(
        columns={"normalized_auc_mean": "local_normalized_auc_mean"}
    )
    greedy = main_table[main_table["method"] == "MutableGreedy"][["setting", "normalized_auc_mean"]].rename(
        columns={"normalized_auc_mean": "greedy_normalized_auc_mean"}
    )
    raw = main_table[main_table["method"] == "RawPEM"][["setting", "normalized_auc_mean"]].rename(
        columns={"normalized_auc_mean": "raw_normalized_auc_mean"}
    )
    assessment_df = can.merge(hybrid, on="setting", how="left").merge(local, on="setting", how="left").merge(greedy, on="setting", how="left").merge(raw, on="setting", how="left")
    if not assessment_df.empty:
        beats_raw = int((assessment_df["normalized_auc_mean"] > assessment_df["raw_normalized_auc_mean"]).sum())
        beats_local = int((assessment_df["normalized_auc_mean"] > assessment_df["local_normalized_auc_mean"]).sum())
        mean_greedy_margin = float((assessment_df["normalized_auc_mean"] - assessment_df["greedy_normalized_auc_mean"]).mean())
        corrupted = assessment_df[assessment_df["setting"].str.endswith("_corrupted")]
        mean_hybrid_margin_corrupted = float(
            (corrupted["normalized_auc_mean"] - corrupted["hybrid_normalized_auc_mean"]).mean()
        ) if not corrupted.empty else 0.0
        broad = beats_raw >= 4 and beats_local >= 4 and mean_greedy_margin > 0.0 and mean_hybrid_margin_corrupted >= 0.0
        feasibility = beats_raw >= 1 and mean_greedy_margin > -0.02
        claim_assessment = {
            "supports_broad_performance_claim": broad,
            "supports_mutable_action_feasibility_claim": feasibility,
            "beats_rawpem_settings": beats_raw,
            "beats_localheuristic_settings": beats_local,
            "mean_auc_margin_vs_mutablegreedy": round(mean_greedy_margin, 6),
            "mean_auc_margin_vs_hybridstatic_on_corrupted": round(mean_hybrid_margin_corrupted, 6),
            "reason": (
                "Evidence supports only a reduced mutable-action feasibility story in a short-horizon frontier-exhaustion regime."
                if not broad
                else "Evidence meets the pre-registered broad performance criteria."
            ),
        }

    summary = {
        "machine": machine_metadata(),
        "coverage_main": coverage.to_dict(orient="records"),
        "main_results": main_table.to_dict(orient="records"),
        "ablation_results": ablation_table.to_dict(orient="records") if not ablation_table.empty else [],
        "paired_tests": effect_tests,
        "estimator_diagnostics": estimator_diagnostics,
        "failure_summary": failure_summary,
        "case_studies": case_studies,
        "frontier_exhaustion_summary": frontier_summary,
        "claim_assessment": claim_assessment,
        "novelty_positioning": novelty_positioning,
    }
    write_json(ROOT / "results.json", summary)
    return summary
