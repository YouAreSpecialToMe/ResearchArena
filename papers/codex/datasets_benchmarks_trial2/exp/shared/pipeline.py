import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jsonlines
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = [17, 23, 42]
DATE_MIN = "2022-01-01"
DATE_MAX = "2025-12-31"
USER_AGENT = "RevisionBenchExperiment/0.2 (codex local run)"
ARXIV_API = "http://export.arxiv.org/api/query"
ALLOWED_CATEGORY_CODES = {"(cs.AI)", "(cs.CL)", "(cs.CV)", "(cs.LG)"}
PACKAGE_NAMES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentence-transformers",
    "spacy",
    "scikit-learn",
    "pandas",
    "numpy",
    "scipy",
    "datasets",
    "evaluate",
    "rapidfuzz",
    "rank-bm25",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "jsonlines",
]

COMPARATIVE_TERMS = {
    "better than",
    "outperform",
    "outperforms",
    "state-of-the-art",
    "sota",
    "superior",
    "improves on",
    "improved over",
}
HEDGE_TERMS = {
    "show",
    "shows",
    "suggest",
    "suggests",
    "demonstrate",
    "demonstrates",
    "indicate",
    "indicates",
    "may",
    "might",
    "can",
}
SCOPE_TERMS = {
    "all",
    "most",
    "some",
    "in-domain",
    "out-of-domain",
    "under weak supervision",
    "under supervision",
    "zero-shot",
    "few-shot",
}
SURFACE_FORMS = [
    {
        "name": "surface_1",
        "revision_aware": True,
        "instruction": "Choose the claim that is still valid in the latest paper version. Reply with only A or B.",
    },
    {
        "name": "surface_2",
        "revision_aware": True,
        "instruction": "The earlier claim may now be obsolete. Using the latest abstract, output only A or B for the currently valid claim.",
    },
    {
        "name": "surface_3",
        "revision_aware": True,
        "instruction": "Decide which claim remains current according to the newest abstract. Return exactly A or B.",
    },
]
_EMBEDDER = None
_SENTENCE_SPLITTER = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dump_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(row)


def safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[dict]:
    with jsonlines.open(path) as reader:
        return list(reader)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9\\-\\.%/]*", text.lower())


ENGLISH_FUNCTION_WORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "with",
    "that",
    "we",
    "this",
    "is",
    "are",
    "our",
    "on",
    "by",
    "from",
    "as",
    "an",
    "be",
    "can",
}


def _clean_abstract_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\\cite\{[^}]+\}", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_sentence_splitter():
    global _SENTENCE_SPLITTER
    if _SENTENCE_SPLITTER is not None:
        return _SENTENCE_SPLITTER
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
        _SENTENCE_SPLITTER = nlp
    except Exception:
        _SENTENCE_SPLITTER = False
    return _SENTENCE_SPLITTER


def split_sentences(text: str) -> List[str]:
    clean = _clean_abstract_text(text)
    splitter = get_sentence_splitter()
    if splitter:
        parts = [normalize_text(sent.text) for sent in splitter(clean).sents]
    else:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean)
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def binary_f1(y_true: Sequence[int], y_pred: Sequence[int], label: int) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float((binary_f1(y_true, y_pred, 0) + binary_f1(y_true, y_pred, 1)) / 2.0)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict:
    if total == 0:
        return {"low": 0.0, "high": 0.0}
    phat = successes / total
    denom = 1 + z ** 2 / total
    center = (phat + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * total)) / total) / denom
    return {"low": float(max(0.0, center - margin)), "high": float(min(1.0, center + margin))}


def lexical_score(a: str, b: str) -> float:
    token_set = fuzz.token_set_ratio(a, b) / 100.0
    lev = fuzz.ratio(a, b) / 100.0
    toks_a = {t for t in tokenize(a) if len(t) > 2}
    toks_b = {t for t in tokenize(b) if len(t) > 2}
    overlap = (len(toks_a & toks_b) / len(toks_a | toks_b)) if toks_a and toks_b else 0.0
    return 0.5 * token_set + 0.3 * lev + 0.2 * overlap


def normalized_edit_distance(a: str, b: str) -> float:
    return 1.0 - (fuzz.ratio(a, b) / 100.0)


def now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def stage_dir(stage: str) -> Path:
    path = ROOT / "exp" / stage
    ensure_dir(path)
    ensure_dir(path / "logs")
    return path


def log(stage: str, message: str) -> None:
    path = stage_dir(stage) / "logs" / "run.log"
    with path.open("a") as handle:
        handle.write(f"[{now_ts()}] {message}\n")


def write_stage_config(stage: str, config: dict) -> None:
    dump_json(stage_dir(stage) / "config.json", config)


def write_stage_results(stage: str, result: dict) -> None:
    dump_json(stage_dir(stage) / "results.json", result)


def write_skipped(stage: str, reason: str, extra: Optional[dict] = None) -> dict:
    payload = {"stage": stage, "status": "skipped", "reason": reason}
    if extra:
        payload.update(extra)
    write_stage_results(stage, payload)
    (stage_dir(stage) / "SKIPPED.md").write_text(reason.strip() + "\n")
    return payload


def importlib_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception as exc:
        return f"unavailable: {exc.__class__.__name__}"


def parse_abs_page(html: str) -> Tuple[str, str, List[str], List[Tuple[int, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.find("meta", {"name": "citation_title"})
    title = normalize_text(title_node["content"]) if title_node and title_node.get("content") else ""
    abs_node = soup.find("blockquote", class_="abstract") or soup.find("blockquote", class_="abstract mathjax")
    abstract = ""
    if abs_node is not None:
        abstract = normalize_text(abs_node.get_text(" ", strip=True).replace("Abstract:", "", 1))
    categories = []
    sub_node = soup.find("span", class_="primary-subject")
    if sub_node:
        categories.append(normalize_text(sub_node.get_text()))
    extra = soup.find("td", class_="tablecell subjects")
    if extra:
        categories.extend([normalize_text(x) for x in extra.get_text().split(";") if x.strip()])
    history = []
    hist = soup.find("div", class_="submission-history")
    if hist:
        text = hist.get_text("\n", strip=True)
        for version, date_str in re.findall(r"\\[v(\\d+)\\]\\s+([A-Za-z]{3},\\s+\\d+\\s+[A-Za-z]{3}\\s+\\d{4}(?:\\s+\\d{2}:\\d{2}:\\d{2}\\s+UTC)?)", text):
            history.append((int(version), date_str))
    return title, abstract, categories, history


def to_iso_date(date_str: str) -> str:
    for fmt in ("%a, %d %b %Y %H:%M:%S UTC", "%a, %d %b %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


class TransformerEmbedder:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                toks = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                hidden = self.model(**toks).last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu())
        return torch.cat(outputs, dim=0).numpy()


def get_embedder(model_name: str):
    global _EMBEDDER
    if _EMBEDDER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _EMBEDDER = TransformerEmbedder(model_name, device)
    return _EMBEDDER


def max_sentence_support(claim: str, evidence_sentences: Sequence[str], embedder) -> Tuple[float, str]:
    if not evidence_sentences:
        return 0.0, ""
    claim_emb = embedder.encode([claim])[0]
    ev_emb = embedder.encode(list(evidence_sentences))
    sims = ev_emb @ claim_emb
    idx = int(np.argmax(sims))
    return float(sims[idx]), evidence_sentences[idx]


def classify_revision_type(old: str, new: str) -> Tuple[str, str]:
    low = (old + " " + new).lower()
    if re.search(r"\d", old + " " + new) and re.sub(r"\D", "", old) != re.sub(r"\D", "", new):
        return "numeric_update", "local_updates"
    if any(term in low for term in COMPARATIVE_TERMS):
        return "comparative_update", "local_updates"
    if any(term in low for term in HEDGE_TERMS):
        return "softening_or_retraction", "semantic_updates"
    if any(term in low for term in SCOPE_TERMS):
        return "scope_qualification_change", "semantic_updates"
    return "dataset_or_setting_change", "semantic_updates"


def detect_trigger(old: str, new: str) -> bool:
    kind, _ = classify_revision_type(old, new)
    if kind == "dataset_or_setting_change":
        delta = set(tokenize(old)) ^ set(tokenize(new))
        return len(delta) >= 3
    return True


def reject_reason(old: str, new: str, lex: float, cosine: float) -> Optional[str]:
    if normalize_text(old) == normalize_text(new):
        return "identical"
    if re.sub(r"[\W_]+", "", old.lower()) == re.sub(r"[\W_]+", "", new.lower()):
        return "punctuation_only"
    if len(tokenize(old)) < 6 or len(tokenize(new)) < 6:
        return "too_short"
    if cosine > 0.985 and not re.search(r"\d", old + new):
        return "near_duplicate"
    if lex < 0.25 and cosine < 0.55:
        return "unaligned"
    return None


def align_versions(old_sents: Sequence[str], new_sents: Sequence[str]) -> List[Tuple[int, int, float]]:
    candidates = []
    for i, old in enumerate(old_sents):
        for j, new in enumerate(new_sents):
            candidates.append((lexical_score(old, new), i, j))
    pairs = []
    used_old = set()
    used_new = set()
    for score, i, j in sorted(candidates, reverse=True):
        if score < 0.35:
            break
        if i in used_old or j in used_new:
            continue
        pairs.append((i, j, score))
        used_old.add(i)
        used_new.add(j)
    return pairs


def record_in_scope(record: dict) -> bool:
    cats = " ".join(record.get("all_categories", []))
    return any(code in cats for code in ALLOWED_CATEGORY_CODES)


def looks_english(text: str) -> bool:
    toks = tokenize(text)
    if len(toks) < 20:
        return False
    ascii_chars = sum(int(ord(ch) < 128) for ch in text)
    ascii_ratio = ascii_chars / max(len(text), 1)
    function_hits = sum(int(tok in ENGLISH_FUNCTION_WORDS) for tok in toks)
    function_ratio = function_hits / max(len(toks), 1)
    vowelish = sum(int(ch.lower() in "aeiou") for ch in text if ch.isalpha())
    alpha = sum(int(ch.isalpha()) for ch in text)
    vowel_ratio = vowelish / max(alpha, 1)
    return ascii_ratio >= 0.98 and function_ratio >= 0.06 and 0.22 <= vowel_ratio <= 0.5


def valid_revision_chain(record: dict) -> bool:
    versions = record.get("versions", [])
    if len(versions) < 2:
        return False
    seen_versions = set()
    last_version = 0
    last_date = ""
    for version in versions:
        version_num = version.get("version")
        abstract = normalize_text(version.get("abstract", ""))
        date = version.get("date", "")
        if not isinstance(version_num, int):
            return False
        if version_num in seen_versions or version_num <= last_version:
            return False
        if not abstract:
            return False
        if last_date and date and date < last_date:
            return False
        seen_versions.add(version_num)
        last_version = version_num
        last_date = date
    return True


def has_meaningful_adjacent_edit(record: dict, threshold: float = 0.08) -> bool:
    versions = record.get("versions", [])
    for prev, curr in zip(versions, versions[1:]):
        if normalized_edit_distance(prev.get("abstract", ""), curr.get("abstract", "")) > threshold:
            return True
    return False


def filter_records(rows: List[dict]) -> Tuple[List[dict], Dict[str, int]]:
    filtered = []
    stats = Counter()
    for row in rows:
        if not record_in_scope(row):
            stats["out_of_scope_category"] += 1
            continue
        if not valid_revision_chain(row):
            stats["malformed_revision_chain"] += 1
            continue
        versions = row.get("versions", [])
        if not (DATE_MIN <= versions[0]["date"] <= DATE_MAX):
            stats["outside_date_range"] += 1
            continue
        if not all(looks_english(version.get("abstract", "")) for version in versions):
            stats["non_english_or_low_confidence_english"] += 1
            continue
        if not has_meaningful_adjacent_edit(row):
            stats["no_adjacent_edit_over_threshold"] += 1
            continue
        filtered.append(row)
        stats["kept"] += 1
    stats["raw_total"] = len(rows)
    return filtered, dict(sorted(stats.items()))


def summarize_source_stats(rows: List[dict]) -> dict:
    papers_per_subject = Counter()
    versions_per_paper = []
    sentence_counts = []
    edited = 0
    fact_edit = 0
    for rec in rows:
        allowed = [cat for cat in rec.get("all_categories", []) if any(code in cat for code in ALLOWED_CATEGORY_CODES)]
        primary = allowed[0] if allowed else rec.get("primary_category", "unknown")
        papers_per_subject[primary] += 1
        versions_per_paper.append(len(rec["versions"]))
        sentence_counts.append(len(split_sentences(rec["versions"][-1]["abstract"])))
        pair_has_edit = False
        pair_has_fact_edit = False
        for prev, curr in zip(rec["versions"], rec["versions"][1:]):
            earlier = prev["abstract"]
            later = curr["abstract"]
            if normalized_edit_distance(earlier, later) > 0.08:
                pair_has_edit = True
            if (
                re.search(r"\d", earlier + later) is not None
                or any(term in (earlier + " " + later).lower() for term in COMPARATIVE_TERMS | HEDGE_TERMS)
            ):
                pair_has_fact_edit = True
        edited += int(pair_has_edit)
        fact_edit += int(pair_has_fact_edit)
    return {
        "papers_per_subject_area": dict(sorted(papers_per_subject.items())),
        "versions_per_paper": {"mean": float(np.mean(versions_per_paper)) if versions_per_paper else 0.0, "median": float(np.median(versions_per_paper)) if versions_per_paper else 0.0},
        "latest_abstract_sentence_count": {"mean": float(np.mean(sentence_counts)) if sentence_counts else 0.0, "median": float(np.median(sentence_counts)) if sentence_counts else 0.0},
        "proportion_with_any_final_revision_abstract_edit": edited / len(rows) if rows else 0.0,
        "proportion_with_numeric_comparative_or_hedge_edits": fact_edit / len(rows) if rows else 0.0,
    }


def stratified_sample(rows: List[dict], k: int, seed: int) -> List[dict]:
    rnd = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row.get("primary_area", "unknown"), row.get("revision_type", "unknown"))].append(row)
    for key in buckets:
        rnd.shuffle(buckets[key])
    sample = []
    while len(sample) < min(k, len(rows)):
        progress = False
        for key in list(buckets):
            if buckets[key] and len(sample) < k:
                sample.append(buckets[key].pop())
                progress = True
        if not progress:
            break
    return sample


def make_fixed_split(rows: List[dict], seed: int) -> dict:
    by_paper = defaultdict(list)
    for row in rows:
        by_paper[row["arxiv_id"]].append(row["example_id"])
    papers = sorted(by_paper)
    rnd = random.Random(seed)
    rnd.shuffle(papers)
    dev_count = max(1, round(0.2 * len(papers)))
    dev_papers = set(papers[:dev_count])
    dev_ids = [eid for pid in papers if pid in dev_papers for eid in by_paper[pid]]
    test_ids = [eid for pid in papers if pid not in dev_papers for eid in by_paper[pid]]
    return {"split_seed": seed, "dev_example_ids": dev_ids, "test_example_ids": test_ids}


def adjudicated_benchmark_path() -> Path:
    return DATA / "final" / "adjudicated_benchmark.jsonl"


def load_final_and_split() -> Tuple[List[dict], dict]:
    final_path = adjudicated_benchmark_path()
    if not final_path.exists():
        raise FileNotFoundError(f"Missing adjudicated benchmark: {final_path}")
    rows = read_jsonl(final_path)
    manifest = read_json(DATA / "final" / "split_manifest.json")
    by_id = {row["example_id"]: row for row in rows}
    return rows, {"dev": [by_id[eid] for eid in manifest["dev_example_ids"]], "test": [by_id[eid] for eid in manifest["test_example_ids"]]}


def gold_label_to_binary(row: dict) -> int:
    label = row.get("gold_label")
    if label in {"later_claim_current", "B"}:
        return 1
    if label in {"earlier_claim_current", "A"}:
        return 0
    current = row.get("current_claim")
    if current == "later":
        return 1
    if current == "earlier":
        return 0
    raise ValueError(f"Unsupported gold label for {row.get('example_id')}: {label or current}")


def gold_label_name(binary_label: int) -> str:
    return "later_claim_current" if binary_label == 1 else "earlier_claim_current"


def evaluate_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1_score(y_true, y_pred)),
        "obsolete_claim_attraction_rate": float(sum(int(p == 0) for p in y_pred) / max(len(y_pred), 1)),
    }


def summarize_seed_metrics(records: List[dict]) -> dict:
    metrics = defaultdict(list)
    for row in records:
        for key, value in row["metrics"].items():
            metrics[key].append(value)
    return {key: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)), "values": values} for key, values in metrics.items()}


def bootstrap_metric(rows: List[dict], pred_map: Dict[str, int], n: int = 1000, seed: int = 17) -> dict:
    rnd = random.Random(seed)
    by_paper = defaultdict(list)
    gold_by_id = {}
    for row in rows:
        by_paper[row["arxiv_id"]].append(row["example_id"])
        gold_by_id[row["example_id"]] = gold_label_to_binary(row)
    papers = list(by_paper)
    accs = []
    for _ in range(n):
        sample_papers = [rnd.choice(papers) for _ in range(len(papers))]
        ids = [eid for pid in sample_papers for eid in by_paper[pid]]
        y_true = [gold_by_id[eid] for eid in ids]
        y_pred = [pred_map[eid] for eid in ids]
        accs.append(accuracy_score(y_true, y_pred))
    return {"mean": float(np.mean(accs)), "ci95_low": float(np.percentile(accs, 2.5)), "ci95_high": float(np.percentile(accs, 97.5))}


def bootstrap_delta(rows: List[dict], pred_a: Dict[str, int], pred_b: Dict[str, int], n: int = 1000, seed: int = 17) -> dict:
    rnd = random.Random(seed)
    by_paper = defaultdict(list)
    gold_by_id = {}
    for row in rows:
        by_paper[row["arxiv_id"]].append(row["example_id"])
        gold_by_id[row["example_id"]] = gold_label_to_binary(row)
    papers = list(by_paper)
    deltas = []
    for _ in range(n):
        sample_papers = [rnd.choice(papers) for _ in range(len(papers))]
        ids = [eid for pid in sample_papers for eid in by_paper[pid]]
        ya = [pred_a[eid] for eid in ids]
        yb = [pred_b[eid] for eid in ids]
        y_true = [gold_by_id[eid] for eid in ids]
        deltas.append(accuracy_score(y_true, yb) - accuracy_score(y_true, ya))
    return {"mean_delta_accuracy": float(np.mean(deltas)), "ci95_low": float(np.percentile(deltas, 2.5)), "ci95_high": float(np.percentile(deltas, 97.5))}


def mcnemar(rows: List[dict], pred_a: Dict[str, int], pred_b: Dict[str, int]) -> dict:
    b = c = 0
    for row in rows:
        eid = row["example_id"]
        gold = gold_label_to_binary(row)
        a_ok = pred_a[eid] == gold
        b_ok = pred_b[eid] == gold
        if a_ok and not b_ok:
            b += 1
        elif not a_ok and b_ok:
            c += 1
    stat = ((abs(b - c) - 1) ** 2) / max(b + c, 1)
    return {"discordant_a_only": b, "discordant_b_only": c, "mcnemar_statistic": float(stat)}


def maybe_collect_raw_records(args) -> List[dict]:
    raw_path = DATA / "raw" / "arxiv_revision_records.jsonl"
    if raw_path.exists() and not args.force:
        return read_jsonl(raw_path)
    ensure_dir(raw_path.parent)
    session = get_session()
    rows = []
    seen = set()
    for category in ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]:
        start = 0
        while start < args.max_results_per_category:
            url = (
                f"{ARXIV_API}?search_query=cat:{category}+AND+submittedDate:[{DATE_MIN.replace('-', '')}0000+TO+{DATE_MAX.replace('-', '')}2359]"
                f"&start={start}&max_results=100&sortBy=submittedDate&sortOrder=ascending"
            )
            root = ET.fromstring(session.get(url, timeout=60).text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if not entries:
                break
            for entry in entries:
                id_text = entry.findtext("atom:id", default="", namespaces=ns)
                match = re.search(r"/abs/([^v]+)v(\d+)$", id_text)
                if not match or int(match.group(2)) <= 1:
                    continue
                aid = match.group(1)
                if aid in seen:
                    continue
                seen.add(aid)
                try:
                    resp = session.get(f"https://arxiv.org/abs/{aid}", timeout=60)
                    resp.raise_for_status()
                    title, _, categories, history = parse_abs_page(resp.text)
                    if len(history) < 2:
                        continue
                    versions = []
                    for version_num, date_str in history:
                        page = session.get(f"https://arxiv.org/abs/{aid}v{version_num}", timeout=60)
                        page.raise_for_status()
                        _, abstract, _, _ = parse_abs_page(page.text)
                        versions.append({"version": version_num, "date": to_iso_date(date_str), "abstract": abstract})
                        time.sleep(0.2)
                    rows.append(
                        {
                            "arxiv_id": aid,
                            "title": title,
                            "primary_category": categories[0] if categories else "",
                            "all_categories": categories,
                            "versions": versions,
                        }
                    )
                except Exception:
                    continue
                if len(rows) >= args.max_papers:
                    break
            start += len(entries)
            time.sleep(0.5)
            if len(rows) >= args.max_papers:
                break
        if len(rows) >= args.max_papers:
            break
    dump_jsonl(raw_path, rows)
    return rows


def capture_metadata(args) -> dict:
    ensure_dir(RESULTS)
    env = {name: importlib_version(name) for name in PACKAGE_NAMES}
    try:
        gpu_query = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    except Exception as exc:
        gpu_query = f"unavailable: {exc.__class__.__name__}"
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_query,
        "seeds": SEEDS,
        "timestamp": now_ts(),
        "package_versions": env,
        "environment_status": {
            "plan_requested_python": "3.11",
            "observed_python": platform.python_version(),
            "plan_compliance": "partial",
            "notes": "Python 3.11 was unavailable on this machine and python3.11 was not found on PATH. Exact package versions used for this run are recorded below.",
        },
    }
    write_stage_config("environment_setup", {"seeds": SEEDS, "package_names": PACKAGE_NAMES})
    write_stage_results("environment_setup", {"experiment": "environment_setup", "metadata": metadata})
    dump_json(RESULTS / "run_metadata.json", metadata)
    return metadata


def collect_data(args) -> dict:
    write_stage_config("data_collection", {"max_results_per_category": args.max_results_per_category, "max_papers": args.max_papers, "allowed_category_codes": sorted(ALLOWED_CATEGORY_CODES)})
    log("data_collection", "starting data collection")
    raw_rows = maybe_collect_raw_records(args)
    filtered, filter_stats = filter_records(raw_rows)
    ensure_dir(DATA / "raw")
    dump_jsonl(DATA / "raw" / "arxiv_revision_records_cs.jsonl", filtered)
    result = {
        "experiment": "data_collection",
        "num_records_raw": len(raw_rows),
        "num_records_in_scope": len(filtered),
        "filter_stats": filter_stats,
        "stats": summarize_source_stats(filtered),
        "claim_scope": "preliminary candidate-mining source pool; not yet an adjudicated benchmark source manifest",
    }
    write_stage_results("data_collection", result)
    log("data_collection", f"finished with {len(filtered)} in-scope records")
    return result


def mine_candidates(args) -> dict:
    write_stage_config("candidate_mining", {"embedding_model": args.embedding_model, "audit_lexical_max": 0.55, "audit_cosine_min": 0.78})
    log("candidate_mining", "starting candidate mining")
    rows = read_jsonl(DATA / "raw" / "arxiv_revision_records_cs.jsonl")
    embedder = get_embedder(args.embedding_model)
    triggered = []
    audit = []
    rejected = []
    for rec in rows:
        latest = rec["versions"][-1]
        latest_sents = split_sentences(latest["abstract"])
        for previous, current in zip(rec["versions"], rec["versions"][1:]):
            current_sents = split_sentences(current["abstract"])
            prev_sents = split_sentences(previous["abstract"])
            matches = align_versions(prev_sents, current_sents)
            matched_old = {m[0] for m in matches}
            matched_new = {m[1] for m in matches}
            for i, j, lex in matches:
                old = prev_sents[i]
                new = current_sents[j]
                pair_emb = embedder.encode([old, new])
                cosine = float(pair_emb[0] @ pair_emb[1])
                base = {
                    "example_id": f"{rec['arxiv_id']}_v{previous['version']}_v{current['version']}_{i}_{j}",
                    "arxiv_id": rec["arxiv_id"],
                    "title": rec["title"],
                    "primary_area": next((cat for cat in rec.get("all_categories", []) if any(code in cat for code in ALLOWED_CATEGORY_CODES)), rec.get("primary_category", "")),
                    "all_categories": rec.get("all_categories", []),
                    "earlier_version": previous["version"],
                    "later_version": current["version"],
                    "latest_version": latest["version"],
                    "earlier_claim": old,
                    "later_claim": new,
                    "latest_abstract": latest["abstract"],
                    "current_pair_abstract": current["abstract"],
                    "lexical_score": lex,
                    "cosine_similarity": cosine,
                    "norm_edit_distance": normalized_edit_distance(old, new),
                    "label_source": "full_adjacent_revision_chain",
                }
                reason = reject_reason(old, new, lex, cosine)
                if reason:
                    base["rejection_reason"] = reason
                    rejected.append(base)
                    continue
                revision_type, coarse_bucket = classify_revision_type(old, new)
                base["revision_type"] = revision_type
                base["coarse_bucket"] = coarse_bucket
                if lex < 0.55 and cosine >= 0.78:
                    audit.append({**base, "pool": "semantic_audit"})
                if detect_trigger(old, new):
                    earlier_support, earlier_ev = max_sentence_support(old, latest_sents, embedder)
                    later_support, later_ev = max_sentence_support(new, latest_sents, embedder)
                    base.update(
                        {
                            "pool": "triggered",
                            "earlier_support_proxy": earlier_support,
                            "later_support_proxy": later_support,
                            "earlier_evidence_sentence": earlier_ev,
                            "later_evidence_sentence": later_ev,
                            "gold_label": "later_claim_current",
                        }
                    )
                    triggered.append(base)
                else:
                    base["rejection_reason"] = "no_trigger"
                    rejected.append(base)
            for i, sent in enumerate(prev_sents):
                if i not in matched_old:
                    rejected.append({"example_id": f"{rec['arxiv_id']}_v{previous['version']}_del_{i}", "arxiv_id": rec["arxiv_id"], "rejection_reason": "deletion_unpaired", "sentence": sent})
            for j, sent in enumerate(current_sents):
                if j not in matched_new:
                    rejected.append({"example_id": f"{rec['arxiv_id']}_v{current['version']}_ins_{j}", "arxiv_id": rec["arxiv_id"], "rejection_reason": "insertion_unpaired", "sentence": sent})
    dump_jsonl(DATA / "intermediate" / "mined_triggered_candidates.jsonl", triggered)
    dump_jsonl(DATA / "intermediate" / "paraphrase_audit_candidates.jsonl", audit)
    dump_jsonl(DATA / "intermediate" / "rejected_candidates.jsonl", rejected)
    result = {
        "experiment": "candidate_mining",
        "candidate_definition": "edited sentence pairs mined across the full adjacent revision chain; no support-threshold filtering used for dataset construction",
        "study_scope": "preliminary candidate-mining only",
        "triggered_candidates": len(triggered),
        "paraphrase_audit_candidates": len(audit),
        "rejected_candidates": len(rejected),
        "revision_type_counts": dict(Counter(row["revision_type"] for row in triggered)),
    }
    write_stage_results("candidate_mining", result)
    log("candidate_mining", f"finished with {len(triggered)} triggered candidates")
    return result


def annotation_template_row(row: dict) -> dict:
    return {
        "example_id": row["example_id"],
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "primary_area": row["primary_area"],
        "revision_type": row.get("revision_type"),
        "coarse_bucket": row.get("coarse_bucket"),
        "pool": row.get("pool"),
        "earlier_claim": row["earlier_claim"],
        "later_claim": row["later_claim"],
        "latest_abstract": row["latest_abstract"],
        "annotation": {
            "stage_a": "",
            "stage_b_earlier": "",
            "stage_b_later": "",
            "revision_type_final": "",
            "supporting_span": "",
            "notes": "",
        },
    }


def load_adjudicated_rows() -> List[dict]:
    path = DATA / "annotated" / "pilot_adjudicated.jsonl"
    if not path.exists():
        return []
    rows = []
    for row in read_jsonl(path):
        if row.get("stage_a_final") != "substantive_change":
            continue
        if row.get("stage_b_earlier_final") != "obsolete":
            continue
        if row.get("stage_b_later_final") != "supported_current":
            continue
        rows.append(
            {
                **row,
                "gold_label": row.get("gold_label", "later_claim_current"),
                "current_claim": row.get("current_claim", "later"),
            }
        )
    return rows


def build_benchmark(args) -> dict:
    write_stage_config("build_benchmark", {"final_size": args.final_size, "split_seed": args.split_seed, "pilot_triggered_size": 90, "pilot_audit_size": 30})
    log("build_benchmark", "building pilot packets and final benchmark")
    triggered = read_jsonl(DATA / "intermediate" / "mined_triggered_candidates.jsonl")
    audit = read_jsonl(DATA / "intermediate" / "paraphrase_audit_candidates.jsonl")
    pilot_exports = {}
    for seed in SEEDS:
        pilot_triggered = stratified_sample(triggered, min(90, len(triggered)), seed)
        pilot_audit = stratified_sample(audit, min(30, len(audit)), seed)
        pilot_rows = pilot_triggered + pilot_audit
        pilot_exports[seed] = len(pilot_rows)
        dump_jsonl(DATA / "pilot" / f"pilot_seed{seed}_candidates.jsonl", pilot_rows)
        dump_jsonl(DATA / "pilot" / f"pilot_seed{seed}_annotation_template.jsonl", [annotation_template_row(row) for row in pilot_rows])
    adjudicated_rows = load_adjudicated_rows()
    if not adjudicated_rows:
        safe_unlink(adjudicated_benchmark_path())
        safe_unlink(DATA / "final" / "split_manifest.json")
        write_skipped(
            "pilot_silver",
            "The planned two-annotator pilot plus adjudication was not completed in this workspace. Pilot candidate packets and annotation templates were exported, but no adjudicated labels are available, so benchmark construction and evaluation remain blocked.",
            {"pilot_exports": pilot_exports},
        )
        result = {
            "experiment": "build_benchmark",
            "status": "blocked_on_human_adjudication",
            "benchmark_scope": "preliminary candidate-mining study only",
            "label_source": "none; adjudicated obsolete/current labels missing",
            "pilot_export_sizes": pilot_exports,
            "primary_pilot_export_size": pilot_exports.get(SEEDS[0], 0),
            "final_benchmark_size": 0,
            "revision_type_counts": {},
            "coarse_bucket_counts": {},
            "split_sizes": {"dev": 0, "test": 0},
        }
        write_stage_results("build_benchmark", result)
        return result
    write_stage_results(
        "pilot_silver",
        {
            "experiment": "pilot_silver",
            "status": "completed_with_adjudication",
            "pilot_export_sizes": pilot_exports,
            "adjudicated_positive_items": len(adjudicated_rows),
        },
    )
    final_rows = adjudicated_rows[: min(args.final_size, len(adjudicated_rows))]
    dump_jsonl(adjudicated_benchmark_path(), final_rows)
    manifest = make_fixed_split(final_rows, args.split_seed)
    dump_json(DATA / "final" / "split_manifest.json", manifest)
    result = {
        "experiment": "build_benchmark",
        "status": "adjudicated_fixed_split",
        "benchmark_scope": "adjudicated obsolete/current benchmark",
        "label_source": "data/annotated/pilot_adjudicated.jsonl",
        "pilot_export_sizes": pilot_exports,
        "primary_pilot_export_size": pilot_exports.get(SEEDS[0], 0),
        "final_benchmark_size": len(final_rows),
        "revision_type_counts": dict(Counter(row["revision_type_final"] for row in final_rows)),
        "coarse_bucket_counts": dict(Counter(row["coarse_bucket"] for row in final_rows)),
        "split_sizes": {"dev": len(manifest["dev_example_ids"]), "test": len(manifest["test_example_ids"])},
    }
    write_stage_results("build_benchmark", result)
    log("build_benchmark", f"finished with {len(final_rows)} adjudicated examples")
    return result


def retrieval_claim_score(claim: str, latest_sentences: Sequence[str], bm25, embedder, bm25_weight: float) -> float:
    tokens = tokenize(claim)
    bm25_scores = bm25.get_scores(tokens)
    bm25_max = float(np.max(bm25_scores)) if len(bm25_scores) else 0.0
    bm25_norm = bm25_max / (bm25_max + 1.0) if bm25_max > 0.0 else 0.0
    cosine, _ = max_sentence_support(claim, latest_sentences, embedder)
    return bm25_weight * bm25_norm + (1.0 - bm25_weight) * cosine


def tune_retrieval_params(dev_rows: List[dict], embedder) -> dict:
    best = None
    for bm25_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for tie_threshold in [-0.05, 0.0, 0.05]:
            preds = []
            for row in dev_rows:
                sents = split_sentences(row["latest_abstract"])
                bm25 = BM25Okapi([tokenize(s) for s in sents] or [[""]])
                old_score = retrieval_claim_score(row["earlier_claim"], sents, bm25, embedder, bm25_weight)
                new_score = retrieval_claim_score(row["later_claim"], sents, bm25, embedder, bm25_weight)
                preds.append(1 if (new_score - old_score) >= tie_threshold else 0)
            score = accuracy_score([gold_label_to_binary(row) for row in dev_rows], preds)
            cand = {"bm25_weight": bm25_weight, "tie_threshold": tie_threshold, "dev_accuracy": score}
            if best is None or cand["dev_accuracy"] > best["dev_accuracy"]:
                best = cand
    return best


def run_later_claim_wins(args) -> dict:
    write_stage_config("later_claim_wins", {"seeds": SEEDS, "policy": "always choose later claim"})
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError as exc:
        return write_skipped("later_claim_wins", str(exc))
    test_rows = split["test"]
    all_predictions = []
    per_seed = []
    for seed in SEEDS:
        y_true = [gold_label_to_binary(row) for row in test_rows]
        y_pred = [1] * len(test_rows)
        metrics = evaluate_predictions(y_true, y_pred)
        per_seed.append({"seed": seed, "metrics": metrics})
        for row in test_rows:
            gold = gold_label_to_binary(row)
            all_predictions.append(
                {
                    "example_id": row["example_id"],
                    "task": "current_claim_selection",
                    "model": "LaterClaimWins",
                    "seed": seed,
                    "input_variant": "claim_pair_plus_latest_abstract",
                    "gold_label": gold_label_name(gold),
                    "predicted_label": "later_claim_current",
                    "score": 1.0,
                    "runtime_seconds": 0.0,
                    "peak_vram_mb": 0.0,
                }
            )
    dump_jsonl(RESULTS / "baselines" / "later_claim_wins_predictions.jsonl", all_predictions)
    result = {"experiment": "later_claim_wins", "per_seed": per_seed, "metrics": summarize_seed_metrics(per_seed)}
    write_stage_results("later_claim_wins", result)
    return result


def run_retrieval_baseline(args) -> dict:
    write_stage_config("retrieval_baseline", {"embedding_model": args.embedding_model, "tune_on": "fixed_dev_split"})
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError as exc:
        return write_skipped("retrieval_baseline", str(exc))
    embedder = get_embedder(args.embedding_model)
    params = tune_retrieval_params(split["dev"], embedder)
    test_rows = split["test"]
    all_predictions = []
    per_seed = []
    for seed in SEEDS:
        seed_start = time.time()
        y_true = []
        y_pred = []
        for row in test_rows:
            sents = split_sentences(row["latest_abstract"])
            bm25 = BM25Okapi([tokenize(s) for s in sents] or [[""]])
            old_score = retrieval_claim_score(row["earlier_claim"], sents, bm25, embedder, params["bm25_weight"])
            new_score = retrieval_claim_score(row["later_claim"], sents, bm25, embedder, params["bm25_weight"])
            pred = 1 if (new_score - old_score) >= params["tie_threshold"] else 0
            gold = gold_label_to_binary(row)
            y_true.append(gold)
            y_pred.append(pred)
            all_predictions.append(
                {
                    "example_id": row["example_id"],
                    "task": "current_claim_selection",
                    "model": "BM25+Embedding",
                    "seed": seed,
                    "input_variant": "claim_pair_plus_latest_abstract",
                    "gold_label": gold_label_name(gold),
                    "predicted_label": gold_label_name(pred),
                    "score": new_score - old_score,
                    "runtime_seconds": 0.0,
                    "peak_vram_mb": 0.0,
                }
            )
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["runtime_per_example_seconds"] = (time.time() - seed_start) / max(len(test_rows), 1)
        per_seed.append({"seed": seed, "metrics": metrics})
    dump_jsonl(RESULTS / "baselines" / "retrieval_predictions.jsonl", all_predictions)
    result = {"experiment": "retrieval_baseline", "config": params, "per_seed": per_seed, "metrics": summarize_seed_metrics(per_seed)}
    write_stage_results("retrieval_baseline", result)
    return result


def sentence_nli_score(claim: str, sentences: Sequence[str], tokenizer, model, device: str) -> Tuple[float, str]:
    best_score = -1e9
    best_sentence = ""
    entail_idx = contra_idx = None
    for idx, label in model.config.id2label.items():
        low = label.lower()
        if "entail" in low:
            entail_idx = int(idx)
        if "contrad" in low:
            contra_idx = int(idx)
    if entail_idx is None or contra_idx is None:
        raise RuntimeError(f"Could not map NLI labels: {model.config.id2label}")
    for sent in sentences:
        inputs = tokenizer(sent, claim, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        score = float(probs[entail_idx] - probs[contra_idx])
        if score > best_score:
            best_score = score
            best_sentence = sent
    return best_score, best_sentence


def tune_nli_threshold(dev_rows: List[dict], tokenizer, model, device: str) -> dict:
    margins = []
    for row in dev_rows:
        sents = split_sentences(row["latest_abstract"])
        old_score, _ = sentence_nli_score(row["earlier_claim"], sents, tokenizer, model, device)
        new_score, _ = sentence_nli_score(row["later_claim"], sents, tokenizer, model, device)
        margins.append(new_score - old_score)
    best = None
    for threshold in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        preds = [1 if margin >= threshold else 0 for margin in margins]
        score = accuracy_score([gold_label_to_binary(row) for row in dev_rows], preds)
        cand = {"tie_threshold": threshold, "dev_accuracy": score}
        if best is None or cand["dev_accuracy"] > best["dev_accuracy"]:
            best = cand
    return best


def run_nli_baseline(args) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    write_stage_config("nli_baseline", {"nli_model": args.nli_model, "tune_on": "fixed_dev_split"})
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError as exc:
        return write_skipped("nli_baseline", str(exc))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device)
    model.eval()
    params = tune_nli_threshold(split["dev"], tokenizer, model, device)
    test_rows = split["test"]
    all_predictions = []
    per_seed = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for seed in SEEDS:
        seed_start = time.time()
        y_true = []
        y_pred = []
        for row in test_rows:
            sents = split_sentences(row["latest_abstract"])
            old_score, old_ev = sentence_nli_score(row["earlier_claim"], sents, tokenizer, model, device)
            new_score, new_ev = sentence_nli_score(row["later_claim"], sents, tokenizer, model, device)
            pred = 1 if (new_score - old_score) >= params["tie_threshold"] else 0
            gold = gold_label_to_binary(row)
            y_true.append(gold)
            y_pred.append(pred)
            all_predictions.append(
                {
                    "example_id": row["example_id"],
                    "task": "current_claim_selection",
                    "model": "DeBERTa-v3-NLI",
                    "seed": seed,
                    "input_variant": "claim_pair_plus_latest_abstract",
                    "gold_label": gold_label_name(gold),
                    "predicted_label": gold_label_name(pred),
                    "score": new_score - old_score,
                    "runtime_seconds": 0.0,
                    "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
                    "earlier_evidence": old_ev,
                    "later_evidence": new_ev,
                }
            )
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["runtime_per_example_seconds"] = (time.time() - seed_start) / max(len(test_rows), 1)
        metrics["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        per_seed.append({"seed": seed, "metrics": metrics})
    dump_jsonl(RESULTS / "baselines" / "nli_predictions.jsonl", all_predictions)
    result = {"experiment": "nli_baseline", "config": params, "per_seed": per_seed, "metrics": summarize_seed_metrics(per_seed)}
    write_stage_results("nli_baseline", result)
    return result


def build_prompt(row: dict, input_variant: str, revision_aware: bool, surface_instruction: Optional[str] = None) -> str:
    if revision_aware:
        instruction = surface_instruction or "The earlier claim may be obsolete. Choose the claim that is still current in the latest version. Reply with only A or B."
    else:
        instruction = surface_instruction or "Choose which claim is current. Reply with only A or B."
    prompt = [instruction, "", f"A. {row['earlier_claim']}", f"B. {row['later_claim']}"]
    if input_variant == "claim_pair_plus_latest_abstract":
        prompt.extend(["", f"Latest abstract: {row['latest_abstract']}"])
    prompt.append("")
    prompt.append("Answer:")
    return "\n".join(prompt)


def parse_ab(text: str) -> Tuple[Optional[int], bool]:
    cleaned = text.strip()
    match = re.search(r"\b([AB])\b", cleaned, re.IGNORECASE)
    if not match:
        return None, True
    return 0 if match.group(1).upper() == "A" else 1, False


def llm_stage_name(model_name: str) -> str:
    name = model_name.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return f"llm_{name}"


def load_causal_lm(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def run_llm_baseline(args) -> dict:
    model_name = args.llm_model
    stage = llm_stage_name(model_name)
    write_stage_config(stage, {"model_name": model_name, "input_variants": ["claim_pair_only", "claim_pair_plus_latest_abstract"], "prompt_conditions": ["plain", "revision_aware"]})
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError as exc:
        return write_skipped(stage, str(exc), {"model_name": model_name})
    test_rows = split["test"]
    predictions_path = RESULTS / "baselines" / f"{stage}_predictions.jsonl"
    audit_path = RESULTS / "baselines" / f"{stage}_invalid_outputs.jsonl"
    ensure_dir(predictions_path.parent)
    try:
        tokenizer, model = load_causal_lm(model_name)
    except Exception as exc:
        return write_skipped(stage, f"Model load failed for {model_name}: {exc}", {"model_name": model_name})
    conditions = [
        {"input_variant": "claim_pair_only", "revision_aware": False, "condition_name": "claim_pair_only_plain"},
        {"input_variant": "claim_pair_plus_latest_abstract", "revision_aware": False, "condition_name": "claim_pair_plus_latest_abstract_plain"},
        {"input_variant": "claim_pair_plus_latest_abstract", "revision_aware": True, "condition_name": "claim_pair_plus_latest_abstract_revision_aware"},
    ]
    all_predictions = []
    invalid_rows = []
    by_condition = {}
    for condition in conditions:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        y_true = []
        y_pred = []
        for row in test_rows:
            prompt = build_prompt(row, condition["input_variant"], condition["revision_aware"])
            model_device = next(model.parameters()).device
            toks = tokenizer(prompt, return_tensors="pt").to(model_device)
            with torch.no_grad():
                output = model.generate(**toks, max_new_tokens=8, do_sample=False, temperature=None, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(output[0][toks["input_ids"].shape[1] :], skip_special_tokens=True)
            pred, invalid = parse_ab(text)
            if pred is None:
                pred = 0
            if invalid:
                invalid_rows.append({"example_id": row["example_id"], "condition": condition["condition_name"], "raw_output": text})
            gold = gold_label_to_binary(row)
            y_true.append(gold)
            y_pred.append(pred)
            all_predictions.append(
                {
                    "example_id": row["example_id"],
                    "task": "current_claim_selection",
                    "model": model_name,
                    "seed": SEEDS[0],
                    "input_variant": condition["condition_name"],
                    "gold_label": gold_label_name(gold),
                    "predicted_label": gold_label_name(pred),
                    "score": float(pred),
                    "runtime_seconds": 0.0,
                    "peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0,
                }
            )
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["runtime_per_example_seconds"] = (time.time() - start) / max(len(test_rows), 1)
        metrics["peak_vram_mb"] = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
        metrics["invalid_output_rate"] = len([x for x in invalid_rows if x["condition"] == condition["condition_name"]]) / max(len(test_rows), 1)
        by_condition[condition["condition_name"]] = metrics
    dump_jsonl(predictions_path, all_predictions)
    dump_jsonl(audit_path, invalid_rows)
    result = {"experiment": stage, "model_name": model_name, "conditions": by_condition, "prediction_file": str(predictions_path), "invalid_output_file": str(audit_path)}
    write_stage_results(stage, result)
    return result


def run_main_experiment(args) -> dict:
    qwen_path = RESULTS / "baselines" / f"{llm_stage_name('Qwen/Qwen2.5-7B-Instruct')}_predictions.jsonl"
    llama_path = RESULTS / "baselines" / f"{llm_stage_name('meta-llama/Llama-3.1-8B-Instruct')}_predictions.jsonl"
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError as exc:
        return write_skipped("main_experiment", str(exc))
    test_rows = split["test"]
    outputs = {}
    for name, path in [("qwen", qwen_path), ("llama", llama_path)]:
        if not path.exists():
            outputs[name] = {"status": "missing_predictions"}
            continue
        preds = read_jsonl(path)
        cond_map = defaultdict(dict)
        for row in preds:
            cond_map[row["input_variant"]][row["example_id"]] = 1 if row["predicted_label"] == "later_claim_current" else 0
        plain_pair = cond_map.get("claim_pair_only_plain")
        plain_ctx = cond_map.get("claim_pair_plus_latest_abstract_plain")
        rev_ctx = cond_map.get("claim_pair_plus_latest_abstract_revision_aware")
        model_result = {}
        if plain_pair and plain_ctx:
            model_result["context_gain_bootstrap"] = bootstrap_delta(test_rows, plain_pair, plain_ctx)
            y_true = [gold_label_to_binary(row) for row in test_rows]
            model_result["claim_pair_only_accuracy"] = accuracy_score(y_true, [plain_pair[row["example_id"]] for row in test_rows])
            model_result["with_context_accuracy"] = accuracy_score(y_true, [plain_ctx[row["example_id"]] for row in test_rows])
        if plain_ctx and rev_ctx:
            model_result["revision_aware_gain_bootstrap"] = bootstrap_delta(test_rows, plain_ctx, rev_ctx)
            model_result["plain_prompt_obsolete_rate"] = sum(int(plain_ctx[row["example_id"]] == 0) for row in test_rows) / max(len(test_rows), 1)
            model_result["revision_aware_obsolete_rate"] = sum(int(rev_ctx[row["example_id"]] == 0) for row in test_rows) / max(len(test_rows), 1)
        outputs[name] = model_result or {"status": "insufficient_predictions"}
    result = {"experiment": "main_experiment", "scope": "preliminary mining audit", "models": outputs}
    write_stage_results("main_experiment", result)
    return result


def run_ablation_mining(args) -> dict:
    triggered = read_jsonl(DATA / "intermediate" / "mined_triggered_candidates.jsonl")
    audit = read_jsonl(DATA / "intermediate" / "paraphrase_audit_candidates.jsonl")
    rejected = read_jsonl(DATA / "intermediate" / "rejected_candidates.jsonl")
    no_trigger = [row for row in rejected if row.get("rejection_reason") == "no_trigger"]
    sampled_untriggered = stratified_sample(no_trigger, min(80, len(no_trigger)), SEEDS[0]) if no_trigger else []
    audit_ids = {row["example_id"] for row in audit}
    triggered_ids = {row["example_id"] for row in triggered}
    semantic_audit_exclusive = len(audit_ids - triggered_ids)
    result = {
        "experiment": "ablation_mining",
        "triggered_candidate_count": len(triggered),
        "sampled_untriggered_count": len(sampled_untriggered),
        "semantic_audit_count": len(audit),
        "semantic_audit_exclusive_count": semantic_audit_exclusive,
        "semantic_audit_fraction_of_triggered": len(audit) / max(len(triggered), 1),
        "negative_finding_semantic_audit_lt_10": semantic_audit_exclusive < 10,
    }
    write_stage_results("ablation_mining", result)
    return result


def run_prompt_ablation(args) -> dict:
    model_name = args.ablation_prompt_model
    stage = "prompt_ablation"
    write_stage_config(stage, {"model_name": model_name, "surface_forms": [row["name"] for row in SURFACE_FORMS]})
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError as exc:
        return write_skipped(stage, str(exc), {"model_name": model_name})
    test_rows = split["test"]
    try:
        tokenizer, model = load_causal_lm(model_name)
    except Exception as exc:
        return write_skipped(stage, f"Prompt ablation model load failed for {model_name}: {exc}", {"model_name": model_name})
    variants = []
    for surface in SURFACE_FORMS:
        y_true = []
        y_pred = []
        start = time.time()
        for row in test_rows:
            prompt = build_prompt(row, "claim_pair_plus_latest_abstract", True, surface["instruction"])
            model_device = next(model.parameters()).device
            toks = tokenizer(prompt, return_tensors="pt").to(model_device)
            with torch.no_grad():
                output = model.generate(**toks, max_new_tokens=8, do_sample=False, temperature=None, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(output[0][toks["input_ids"].shape[1] :], skip_special_tokens=True)
            pred, _ = parse_ab(text)
            y_true.append(gold_label_to_binary(row))
            y_pred.append(0 if pred is None else pred)
        metrics = evaluate_predictions(y_true, y_pred)
        metrics["runtime_per_example_seconds"] = (time.time() - start) / max(len(test_rows), 1)
        variants.append({"name": surface["name"], "metrics": metrics})
    result = {"experiment": stage, "model_name": model_name, "variants": variants}
    write_stage_results(stage, result)
    return result


def run_evidence_restriction(args) -> dict:
    return write_skipped(
        "evidence_restriction",
        "This ablation requires additional human relabeling with and without full-paper access. No second annotator pool or adjudicator exists in this workspace, so the step cannot be executed honestly.",
    )


def _simple_svg_bars(title: str, bars: List[Tuple[str, float, str]], subtitle: Optional[str] = None) -> str:
    width = 140 * max(len(bars), 1) + 40
    height = 260
    max_val = max((value for _, value, _ in bars), default=1.0)
    rects = []
    for idx, (label, value, color) in enumerate(bars):
        x = 30 + idx * 130
        h = 150 * (value / max_val if max_val > 0 else 0.0)
        rects.append(
            f'<rect x="{x}" y="{200-h:.1f}" width="70" height="{h:.1f}" fill="{color}"/>'
            f'<text x="{x}" y="220" font-size="11">{label}</text>'
            f'<text x="{x}" y="{190-h:.1f}" font-size="11">{value:.3f}</text>'
        )
    extra = f'<text x="20" y="36" font-size="12">{subtitle}</text>' if subtitle else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="20" y="20" font-size="16">{title}</text>'
        f"{extra}"
        '<line x1="20" y1="200" x2="95%" y2="200" stroke="black"/>'
        + "".join(rects)
        + "</svg>"
    )


def make_figures(analysis: dict) -> None:
    ensure_dir(FIGURES)
    flow = analysis["dataset_validity"]["flow_counts"]
    flow_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="180">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="20" font-size="16">RevisionBench Preliminary Audit Flow</text>'
        f'<rect x="20" y="60" width="130" height="50" fill="#d9d9d9"/><text x="35" y="88" font-size="12">In-scope papers: {flow["in_scope_papers"]}</text>'
        '<text x="160" y="90" font-size="18">→</text>'
        f'<rect x="190" y="60" width="150" height="50" fill="#9ecae1"/><text x="205" y="88" font-size="12">Triggered pairs: {flow["triggered_candidates"]}</text>'
        '<text x="350" y="90" font-size="18">→</text>'
        f'<rect x="380" y="60" width="150" height="50" fill="#fdae6b"/><text x="395" y="88" font-size="12">Pilot export: {flow["pilot_export_size"]}</text>'
        '<text x="540" y="90" font-size="18">→</text>'
        f'<rect x="570" y="60" width="200" height="50" fill="#74c476"/><text x="585" y="88" font-size="12">Adjudicated benchmark: {flow["final_benchmark_size"]}</text>'
        f'<text x="20" y="145" font-size="12">{analysis["dataset_validity"]["annotation_status"]}</text>'
        '</svg>'
    )
    (FIGURES / "dataset_flow.svg").write_text(flow_svg)
    if not analysis["model_results"]:
        for stale_name in ["model_accuracy.svg", "bucket_accuracy.svg", "baseline_accuracy.svg", "benchmark_composition.svg"]:
            safe_unlink(FIGURES / stale_name)
        return
    bars = []
    for name, score, color in [
        ("LaterWins", analysis["model_results"]["LaterClaimWins"]["accuracy"], "#7f7f7f"),
        ("Retrieval", analysis["model_results"]["BM25+Embedding"]["accuracy"], "#2b8cbe"),
        ("NLI", analysis["model_results"]["DeBERTa-v3-NLI"]["accuracy"], "#de2d26"),
    ]:
        bars.append((name, score, color))
    for key, color in [("Qwen2.5-7B-Instruct", "#31a354"), ("Llama-3.1-8B-Instruct", "#756bb1")]:
        if key in analysis["model_results"]:
            bars.append((key.split("-")[0], analysis["model_results"][key]["claim_pair_plus_latest_abstract_plain"]["accuracy"], color))
    (FIGURES / "model_accuracy.svg").write_text(_simple_svg_bars("Primary Accuracy", bars, "Accuracy on the fixed provisional test split"))
    bucket_bars = [(key, value["accuracy"], "#3182bd" if key == "local_updates" else "#e6550d") for key, value in analysis["coarse_bucket_accuracy"].items()]
    (FIGURES / "bucket_accuracy.svg").write_text(_simple_svg_bars("Accuracy By Bucket", bucket_bars))


def run_analysis(args) -> dict:
    try:
        rows, split = load_final_and_split()
    except FileNotFoundError:
        build = read_json(stage_dir("build_benchmark") / "results.json")
        triggered = read_jsonl(DATA / "intermediate" / "mined_triggered_candidates.jsonl")
        audit = read_jsonl(DATA / "intermediate" / "paraphrase_audit_candidates.jsonl")
        rejected = read_jsonl(DATA / "intermediate" / "rejected_candidates.jsonl")
        analysis = {
            "experiment": "analysis",
            "study_scope": "preliminary candidate-mining study only",
            "dataset_validity": {
                "flow_counts": {
                    "in_scope_papers": read_json(stage_dir("data_collection") / "results.json")["num_records_in_scope"],
                    "triggered_candidates": len(triggered),
                    "semantic_audit_candidates": len(audit),
                    "rejected_candidates": len(rejected),
                    "pilot_export_size": build.get("primary_pilot_export_size", 0),
                    "final_benchmark_size": 0,
                },
                "revision_type_counts": read_json(stage_dir("candidate_mining") / "results.json")["revision_type_counts"],
                "coarse_bucket_counts": {},
                "subject_area_counts": read_json(stage_dir("data_collection") / "results.json")["stats"]["papers_per_subject_area"],
                "annotation_status": "Pilot packets exported, but the required two-annotator adjudication is still missing; no benchmark evaluation was run.",
                "filter_stats": read_json(stage_dir("data_collection") / "results.json").get("filter_stats", {}),
            },
            "model_results": {},
            "coarse_bucket_accuracy": {},
            "bootstrap_retrieval_vs_nli": {"status": "not_run"},
            "mcnemar_retrieval_vs_nli": {"status": "not_run"},
        }
        write_stage_results("analysis", analysis)
        make_figures(analysis)
        return analysis
    test_rows = split["test"]
    build = read_json(stage_dir("build_benchmark") / "results.json")
    triggered = read_jsonl(DATA / "intermediate" / "mined_triggered_candidates.jsonl")
    audit = read_jsonl(DATA / "intermediate" / "paraphrase_audit_candidates.jsonl")
    rejected = read_jsonl(DATA / "intermediate" / "rejected_candidates.jsonl")
    model_results = {}
    pred_maps = {}

    y_true = [gold_label_to_binary(row) for row in test_rows]

    lcw_rows = read_jsonl(RESULTS / "baselines" / "later_claim_wins_predictions.jsonl")
    pred_maps["LaterClaimWins"] = {row["example_id"]: 1 if row["predicted_label"] == "later_claim_current" else 0 for row in lcw_rows if row["seed"] == 17}
    model_results["LaterClaimWins"] = {"accuracy": accuracy_score(y_true, [pred_maps["LaterClaimWins"][row["example_id"]] for row in test_rows]), "bootstrap_ci": bootstrap_metric(test_rows, pred_maps["LaterClaimWins"])}

    retrieval_rows = read_jsonl(RESULTS / "baselines" / "retrieval_predictions.jsonl")
    pred_maps["BM25+Embedding"] = {row["example_id"]: 1 if row["predicted_label"] == "later_claim_current" else 0 for row in retrieval_rows if row["seed"] == 17}
    model_results["BM25+Embedding"] = {"accuracy": accuracy_score(y_true, [pred_maps["BM25+Embedding"][row["example_id"]] for row in test_rows]), "bootstrap_ci": bootstrap_metric(test_rows, pred_maps["BM25+Embedding"])}

    nli_rows = read_jsonl(RESULTS / "baselines" / "nli_predictions.jsonl")
    pred_maps["DeBERTa-v3-NLI"] = {row["example_id"]: 1 if row["predicted_label"] == "later_claim_current" else 0 for row in nli_rows if row["seed"] == 17}
    model_results["DeBERTa-v3-NLI"] = {"accuracy": accuracy_score(y_true, [pred_maps["DeBERTa-v3-NLI"][row["example_id"]] for row in test_rows]), "bootstrap_ci": bootstrap_metric(test_rows, pred_maps["DeBERTa-v3-NLI"])}

    for model_name in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]:
        path = RESULTS / "baselines" / f"{llm_stage_name(model_name)}_predictions.jsonl"
        if not path.exists():
            continue
        llm_rows = read_jsonl(path)
        condition_map = defaultdict(dict)
        for row in llm_rows:
            condition_map[row["input_variant"]][row["example_id"]] = 1 if row["predicted_label"] == "later_claim_current" else 0
        pretty_name = model_name.split("/")[-1]
        model_results[pretty_name] = {}
        for condition, preds in condition_map.items():
            if not preds:
                continue
            model_results[pretty_name][condition] = {
                "accuracy": accuracy_score(y_true, [preds[row["example_id"]] for row in test_rows]),
                "bootstrap_ci": bootstrap_metric(test_rows, preds),
            }
        if "claim_pair_plus_latest_abstract_plain" in condition_map:
            pred_maps[pretty_name] = condition_map["claim_pair_plus_latest_abstract_plain"]

    coarse_bucket_accuracy = {}
    for bucket in sorted(set(row["coarse_bucket"] for row in test_rows)):
        ids = [row["example_id"] for row in test_rows if row["coarse_bucket"] == bucket]
        preds = pred_maps["DeBERTa-v3-NLI"]
        gold = [gold_label_to_binary(next(row for row in test_rows if row["example_id"] == eid)) for eid in ids]
        coarse_bucket_accuracy[bucket] = {"count": len(ids), "accuracy": accuracy_score(gold, [preds[eid] for eid in ids])}

    analysis = {
        "experiment": "analysis",
        "study_scope": "preliminary mining audit on a leakage-reduced fixed split",
        "dataset_validity": {
            "flow_counts": {
                "in_scope_papers": read_json(stage_dir("data_collection") / "results.json")["num_records_in_scope"],
                "triggered_candidates": len(triggered),
                "semantic_audit_candidates": len(audit),
                "rejected_candidates": len(rejected),
                "pilot_export_size": build.get("primary_pilot_export_size", 0),
                "final_benchmark_size": build["final_benchmark_size"],
            },
            "revision_type_counts": build["revision_type_counts"],
            "coarse_bucket_counts": build["coarse_bucket_counts"],
            "subject_area_counts": read_json(stage_dir("data_collection") / "results.json")["stats"]["papers_per_subject_area"],
            "annotation_status": "Adjudicated labels loaded from data/annotated/pilot_adjudicated.jsonl",
        },
        "model_results": model_results,
        "coarse_bucket_accuracy": coarse_bucket_accuracy,
        "bootstrap_retrieval_vs_nli": bootstrap_delta(test_rows, pred_maps["BM25+Embedding"], pred_maps["DeBERTa-v3-NLI"]),
        "mcnemar_retrieval_vs_nli": mcnemar(test_rows, pred_maps["BM25+Embedding"], pred_maps["DeBERTa-v3-NLI"]),
    }
    write_stage_results("analysis", analysis)
    make_figures(analysis)
    return analysis


def aggregate_results() -> dict:
    result = {
        "run_metadata": read_json(RESULTS / "run_metadata.json"),
        "data_collection": read_json(stage_dir("data_collection") / "results.json"),
        "candidate_mining": read_json(stage_dir("candidate_mining") / "results.json"),
        "pilot_silver": read_json(stage_dir("pilot_silver") / "results.json"),
        "build_benchmark": read_json(stage_dir("build_benchmark") / "results.json"),
        "later_claim_wins": read_json(stage_dir("later_claim_wins") / "results.json"),
        "retrieval_baseline": read_json(stage_dir("retrieval_baseline") / "results.json"),
        "nli_baseline": read_json(stage_dir("nli_baseline") / "results.json"),
        "main_experiment": read_json(stage_dir("main_experiment") / "results.json"),
        "ablation_mining": read_json(stage_dir("ablation_mining") / "results.json"),
        "prompt_ablation": read_json(stage_dir("prompt_ablation") / "results.json") if (stage_dir("prompt_ablation") / "results.json").exists() else {"status": "not_run"},
        "evidence_restriction": read_json(stage_dir("evidence_restriction") / "results.json") if (stage_dir("evidence_restriction") / "results.json").exists() else {"status": "not_run"},
        "analysis": read_json(stage_dir("analysis") / "results.json"),
        "study_scope": "preliminary candidate-mining study only unless data/annotated/pilot_adjudicated.jsonl exists and produces a frozen final benchmark",
        "limitations": [
            "The planned two-annotator plus adjudication workflow was not completed in this workspace, so the study is limited to a preliminary candidate-mining study rather than benchmark evaluation.",
            "No adjudicated obsolete/current labels were available, so all baselines and main comparisons were skipped instead of being run on invalid silver labels.",
            "The source pool remains below the proposal target, so any benchmark-size claim must stay downgraded until a larger collection and adjudicated labeling pass are completed.",
            "All numbers in this file were produced by code executed in this workspace.",
        ],
    }
    dump_json(ROOT / "results.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-results-per-category", type=int, default=120)
    parser.add_argument("--max-papers", type=int, default=1200)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--ablation-prompt-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--final-size", type=int, default=150)
    parser.add_argument("--split-seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEEDS[0])
    if args.stage == "environment_setup":
        capture_metadata(args)
    elif args.stage == "data_collection":
        collect_data(args)
    elif args.stage == "candidate_mining":
        mine_candidates(args)
    elif args.stage == "build_benchmark":
        build_benchmark(args)
    elif args.stage == "later_claim_wins":
        run_later_claim_wins(args)
    elif args.stage == "retrieval_baseline":
        run_retrieval_baseline(args)
    elif args.stage == "nli_baseline":
        run_nli_baseline(args)
    elif args.stage == "llm_baseline":
        if not args.llm_model:
            raise ValueError("--llm-model is required for llm_baseline")
        run_llm_baseline(args)
    elif args.stage == "main_experiment":
        run_main_experiment(args)
    elif args.stage == "ablation_mining":
        run_ablation_mining(args)
    elif args.stage == "prompt_ablation":
        run_prompt_ablation(args)
    elif args.stage == "evidence_restriction":
        run_evidence_restriction(args)
    elif args.stage == "analysis":
        run_analysis(args)
    elif args.stage == "aggregate":
        aggregate_results()
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
