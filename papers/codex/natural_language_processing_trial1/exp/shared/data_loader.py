import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset

from .utils import ROOT, SEEDS, ensure_dir, read_json, read_jsonl, truncate_by_words, write_json, write_jsonl


CUB_CONFIG = "qwen-1.5b-instruct"


def _render_cub_template(template: str, subject: str) -> str:
    text = template.replace("<subject-w-space>", subject + " ").replace("<subject>", subject)
    text = " ".join(text.split())
    return text.strip()


def load_cub_rows() -> List[Dict]:
    ds = load_dataset("copenlu/cub-counterfact", CUB_CONFIG)
    rows = []
    for split_name, split_ds in ds.items():
        for ex in split_ds:
            target_true = ex.get("target_true") or ""
            target_new = ex.get("target_new") or target_true
            rows.append(
                {
                    "dataset": "cub",
                    "orig_split": split_name,
                    "source_id": f"cub::{ex['id']}",
                    "context_type": ex["context_type"],
                    "question": _render_cub_template(ex["template"], ex["subject"]),
                    "context_question": _render_cub_template(ex["template"], ex["subject"]),
                    "context": ex["template_w_context"].replace("<subject-w-space>", ex["subject"] + " ").replace("<subject>", ex["subject"]).replace("<target>", target_new),
                    "subject": ex["subject"],
                    "gold_answers": [target_true],
                    "context_answers": [target_true if ex["context_type"] == "gold" else target_new],
                    "memory_answers": [target_true],
                    "benchmark_aliases": [target_true, target_new],
                    "source_metadata": ex,
                }
            )
    return rows


def load_whoqa_rows() -> List[Dict]:
    path = ROOT / "data/raw/WhoQA_repo/WhoQA.json"
    with path.open() as f:
        data = json.load(f)
    rows = []
    for ex in data:
        context_texts = []
        all_aliases = []
        first_context_aliases = []
        for idx, ctx in enumerate(ex["contexts"]):
            page_id = ctx.get("page_id", f"context_{idx}")
            text = truncate_by_words(ctx.get("candidate_texts", ""), 160)
            context_texts.append(f"[Context {idx}] {page_id}\n{text}")
            alias_groups = ex["answer_by_context"].get(str(idx), [])
            flattened = []
            for group in alias_groups:
                flattened.extend(group)
            all_aliases.extend(flattened)
            if idx == 0:
                first_context_aliases = flattened[:]
        rows.append(
            {
                "dataset": "whoqa",
                "orig_split": "eval",
                "source_id": f"whoqa::{ex['q_id']}",
                "question_type_id": ex["question_type_id"],
                "num_distinct_answers": ex["num_distinct_answers"],
                "question": ex["questions"][0],
                "context": "\n\n".join(context_texts),
                "gold_answers": first_context_aliases or all_aliases[:1],
                "context_answers": all_aliases,
                "memory_answers": first_context_aliases or all_aliases[:1],
                "benchmark_aliases": all_aliases,
                "candidate_entities": [ctx.get("page_id") for ctx in ex["contexts"]],
                "ambiguity_flags": {
                    "same_name_conflict": True,
                    "num_distinct_answers": ex["num_distinct_answers"],
                    "question_type_id": ex["question_type_id"],
                },
                "source_metadata": ex,
            }
        )
    return rows


def _cub_sample(rows: List[Dict], seed: int) -> Tuple[List[str], Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["context_type"]].append(row["source_id"])
    targets = {
        "gold": {"dev": 14, "test": 40},
        "edited": {"dev": 13, "test": 40},
        "irrelevant": {"dev": 13, "test": 40},
    }
    import random

    rng = random.Random(seed)
    selected = []
    stats = {"dataset": "cub", "seed": seed, "split_counts": {}, "context_counts": {}}
    for ctx_type, ids in grouped.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        dev_n = targets[ctx_type]["dev"]
        test_n = targets[ctx_type]["test"]
        chosen = ids[: dev_n + test_n]
        selected.extend(chosen)
        stats["split_counts"][ctx_type] = {"dev": dev_n, "test": test_n}
        stats["context_counts"][ctx_type] = dev_n + test_n
    return selected, stats


def _whoqa_bucket(row: Dict) -> str:
    n = row["num_distinct_answers"]
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    return "4+"


def _whoqa_sample(rows: List[Dict], seed: int) -> Tuple[List[str], Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_whoqa_bucket(row)].append(row["source_id"])
    import random

    rng = random.Random(seed)
    targets = {"2": {"dev": 20, "test": 60}, "3": {"dev": 10, "test": 30}, "4+": {"dev": 10, "test": 30}}
    selected = []
    stats = {"dataset": "whoqa", "seed": seed, "split_counts": {}, "bucket_counts": {}}
    for bucket, ids in grouped.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        dev_n = targets[bucket]["dev"]
        test_n = targets[bucket]["test"]
        chosen = ids[: dev_n + test_n]
        selected.extend(chosen)
        stats["split_counts"][bucket] = {"dev": dev_n, "test": test_n}
        stats["bucket_counts"][bucket] = dev_n + test_n
    return selected, stats


def build_candidate_pools() -> Dict:
    cub_rows = load_cub_rows()
    whoqa_rows = load_whoqa_rows()
    payload = {"candidate_pools": [], "final_seed": None, "selection_report": {}}
    for seed in SEEDS:
        cub_ids, cub_stats = _cub_sample(cub_rows, seed)
        whoqa_ids, whoqa_stats = _whoqa_sample(whoqa_rows, seed)
        cub_selected = [row for row in cub_rows if row["source_id"] in set(cub_ids)]
        whoqa_selected = [row for row in whoqa_rows if row["source_id"] in set(whoqa_ids)]
        whoqa_question_types = Counter(row["question_type_id"] for row in whoqa_selected)
        selection_score = (
            sum(abs((cub_stats["context_counts"].get(ctx, 0)) - (160 / 3.0)) for ctx in ["gold", "edited", "irrelevant"]),
            sum(abs(whoqa_stats["bucket_counts"].get(bucket, 0) - target) for bucket, target in {"2": 80, "3": 40, "4+": 40}.items()),
            -len(whoqa_question_types),
            seed,
        )
        payload["candidate_pools"].append(
            {
                "seed": seed,
                "cub_ids": cub_ids,
                "whoqa_ids": whoqa_ids,
                "balance_stats": {
                    "cub": cub_stats,
                    "whoqa": whoqa_stats,
                    "whoqa_question_type_counts": dict(sorted(whoqa_question_types.items())),
                },
                "selection_score": {
                    "cub_balance_l1": selection_score[0],
                    "whoqa_bucket_l1": selection_score[1],
                    "whoqa_question_type_diversity": len(whoqa_question_types),
                },
            }
        )
    ranked = sorted(
        payload["candidate_pools"],
        key=lambda pool: (
            pool["selection_score"]["cub_balance_l1"],
            pool["selection_score"]["whoqa_bucket_l1"],
            -pool["selection_score"]["whoqa_question_type_diversity"],
            pool["seed"],
        ),
    )
    payload["final_seed"] = ranked[0]["seed"]
    payload["selection_report"] = {
        "criterion": "Minimize stratification imbalance, maximize WhoQA question-type diversity, then break ties by seed.",
        "ranked_seeds": [pool["seed"] for pool in ranked],
    }
    return payload


def freeze_audited_ids() -> Dict:
    frozen_path = ROOT / "data/processed/frozen_audit_ids.json"
    pools = build_candidate_pools()
    write_json(ROOT / "data/interim/candidate_audit_pools.json", pools)
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if existing.get("seed") == pools["final_seed"]:
            if existing.get("selection_report") != pools.get("selection_report"):
                existing["selection_report"] = pools.get("selection_report", {})
                write_json(frozen_path, existing)
            return existing
    final = next(pool for pool in pools["candidate_pools"] if pool["seed"] == pools["final_seed"])
    frozen = {
        "seed": pools["final_seed"],
        "selection_report": pools["selection_report"],
        "dev": {"cub": final["cub_ids"][:40], "whoqa": final["whoqa_ids"][:40]},
        "test": {"cub": final["cub_ids"][40:160], "whoqa": final["whoqa_ids"][40:160]},
    }
    write_json(frozen_path, frozen)
    return frozen


def _cub_route_label(row: Dict) -> str:
    if row["context_type"] == "gold":
        return "context"
    return "memory"


def _whoqa_default_label(row: Dict) -> str:
    return "abstain"


def _whoqa_permissive_label(row: Dict) -> str:
    if row["num_distinct_answers"] == 2:
        return "memory"
    return "abstain"


def build_audited_subset() -> List[Dict]:
    audited_path = ROOT / "data/processed/audited_subset.jsonl"
    frozen = freeze_audited_ids()
    if audited_path.exists():
        existing = read_jsonl(audited_path)
        if existing:
            first_id = existing[0]["example_id"]
            dataset = first_id.split("::", 1)[0]
            split = first_id.rsplit("::", 1)[-1]
            source_id = "::".join(first_id.split("::")[:2])
            expected_first = frozen[split][dataset][0]
            if source_id == expected_first:
                return existing
    cub_map = {row["source_id"]: row for row in load_cub_rows()}
    whoqa_map = {row["source_id"]: row for row in load_whoqa_rows()}

    rows = []
    alias_additions = {}
    for split in ["dev", "test"]:
        for dataset in ["cub", "whoqa"]:
            ids = frozen[split][dataset]
            for idx, source_id in enumerate(ids):
                base = dict((cub_map if dataset == "cub" else whoqa_map)[source_id])
                if dataset == "cub":
                    default_label = _cub_route_label(base)
                    permissive_label = default_label
                    strict_label = default_label
                    audited_answers = base["gold_answers"] if default_label == "memory" else base["context_answers"]
                    alias_needed = len(set(audited_answers + base["benchmark_aliases"])) > 1
                    alias_note = "benchmark targets include true and edited aliases"
                else:
                    default_label = _whoqa_default_label(base)
                    permissive_label = _whoqa_permissive_label(base)
                    strict_label = "abstain"
                    audited_answers = base["memory_answers"]
                    alias_needed = len(base["benchmark_aliases"]) > len(set(base["benchmark_aliases"][:1]))
                    alias_note = "flattened context answer aliases from WhoQA"
                if alias_needed:
                    alias_additions[source_id] = {
                        "aliases": sorted(set(base["benchmark_aliases"]))[:12],
                        "justification": alias_note,
                    }
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "example_id": f"{source_id}::{split}",
                        "question": base["question"],
                        "context": truncate_by_words(base["context"], 512),
                        "benchmark_context_type": base.get("context_type", "multi_conflict"),
                        "gold_answers": base["gold_answers"],
                        "benchmark_aliases": sorted(set(base["benchmark_aliases"]))[:32],
                        "candidate_entities": base.get("candidate_entities", [base.get("subject", "")]),
                        "ambiguity_flags": base.get("ambiguity_flags", {}),
                        "source_metadata": base["source_metadata"],
                        "audited_label": default_label,
                        "policy_variant_label": {
                            "default": default_label,
                            "strict_abstain": strict_label,
                            "permissive_memory": permissive_label,
                        },
                        "audited_answers": audited_answers,
                        "support_note": "single-annotator rule-based audit executed inside experiment code",
                        "adjudication_note": "No external human double-annotation available in this workspace; agreement metrics are reported as unavailable.",
                        "annotator_confidence": 2 if dataset == "whoqa" else 3,
                        "alias_normalization_needed": alias_needed,
                    }
                )
    write_jsonl(audited_path, rows)
    write_json(ROOT / "data/processed/alias_tables.json", alias_additions)
    return rows


def write_audit_guidelines() -> None:
    text = """# TRIAD-Audit Local Execution Guidelines

This run uses the proposal's three route labels: `context`, `memory`, and `abstain`.

- `context`: the supplied context is sufficient and should be followed.
- `memory`: the supplied context is misleading or irrelevant, but the answer remains identifiable from memory.
- `abstain`: the item is conflict-heavy or ambiguous enough that a committed answer is unsafe.

Local limitation:

- This workspace does not include external human annotators.
- The audited subset is therefore produced by a single deterministic rule-based annotator implemented in code.
- Double-annotation agreement and Cohen's kappa are reported as unavailable instead of fabricated.

Sensitivity variants:

- `strict_abstain`: unresolved WhoQA conflicts remain abstain.
- `permissive_memory`: two-answer WhoQA conflicts are remapped to `memory` for sensitivity testing.
"""
    path = ROOT / "exp/audit/guidelines.md"
    ensure_dir(path.parent)
    path.write_text(text)
