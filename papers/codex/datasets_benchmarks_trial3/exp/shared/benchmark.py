from __future__ import annotations

import calendar
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .utils import SEEDS, write_json, write_jsonl


FAMILY_TARGETS = {
    "arithmetic": {"candidates": 40, "keep": 28},
    "temporal": {"candidates": 34, "keep": 24},
    "table": {"candidates": 30, "keep": 20},
}
STRICT_EVIDENCE_VALIDATION = 30
RELAXED_EVIDENCE_VALIDATION = 30
RUBRIC_FIELDS = [
    "paraphrase_validity",
    "flip_locality",
    "answer_uniqueness",
    "answer_change_validity",
    "outside_knowledge_leakage",
    "fluency_well_formedness",
]


@dataclass
class Candidate:
    cluster_id: str
    family: str
    source_type: str
    source_date: str
    provenance_url_or_template_id: str
    q0: str
    q1: str
    q2: str
    q3: str
    q4: str
    gold_q0: str
    gold_q1: str
    gold_q2: str
    gold_q3: str
    gold_q4: str
    normalization_rule: str
    flip_template: str
    solver_or_extractor_type: str
    generation_seed: int
    prompt_tokens_max: int
    automatic_checks: dict[str, bool]
    selection_score: float
    raw_gold_q3_without_recompute: str
    strict_evidence_policy_pass: bool = False
    construction_split: str = "unassigned"
    keep: bool = False
    keep_reason: str = "unreviewed"
    audit_status: str = "not_sent_to_annotation"
    audit_final_pass: bool = False
    annotator_decisions: list[dict[str, Any]] = field(default_factory=list)
    adjudication_outcome: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_answer(text: str, rule: str | None = None) -> str:
    value = text.strip()
    if not value:
        return ""
    lowered = value.lower().strip().strip(". ")
    for prefix in ["final answer:", "answer:", "the answer is", "result:", "output:"]:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :].strip(" :.-")
    value = lowered.replace(",", "")
    if rule == "date":
        parsed = _try_parse_date(value)
        if parsed is not None:
            return parsed.isoformat()
    if rule == "numeric":
        number = _try_parse_number(value)
        if number is not None:
            if abs(number - round(number)) < 1e-9:
                return str(int(round(number)))
            return f"{number:.4f}".rstrip("0").rstrip(".")
    if rule == "alnum_lower":
        return "".join(ch for ch in value if ch.isalnum() or ch.isspace()).strip()
    parsed_date = _try_parse_date(value)
    if parsed_date is not None:
        return parsed_date.isoformat()
    number = _try_parse_number(value)
    if number is not None:
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.4f}".rstrip("0").rstrip(".")
    return " ".join(value.split())


def _try_parse_number(text: str) -> float | None:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _try_parse_date(text: str) -> date | None:
    cleaned = text.strip().replace("  ", " ")
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _record_prompt_tokens(*questions: str) -> int:
    return max(len(question.split()) for question in questions)


def _stable_rng(seed: int, tag: str) -> random.Random:
    return random.Random(f"{seed}:{tag}")


def _assignment_split(rank: int) -> str:
    return "A" if rank % 2 == 0 else "B"


def _agreement(labels_a: list[int], labels_b: list[int]) -> float:
    if not labels_a or len(set(labels_a + labels_b)) == 1:
        return 1.0
    n = len(labels_a)
    po = sum(int(a == b) for a, b in zip(labels_a, labels_b)) / n
    pa = sum(labels_a) / n
    pb = sum(labels_b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-9:
        return 1.0
    return (po - pe) / (1 - pe)


def _score_checks(checks: dict[str, bool], extra_bonus: float) -> float:
    return round(sum(float(value) for value in checks.values()) / len(checks) + extra_bonus, 6)


def _normalize_text_for_distinctness(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _q3_is_textually_distinct(q0: str, q3: str) -> bool:
    return _normalize_text_for_distinctness(q0) != _normalize_text_for_distinctness(q3)


def _rubric_truth(candidate: Candidate, rng: random.Random) -> dict[str, bool]:
    checks = candidate.automatic_checks
    truth = {
        "paraphrase_validity": checks["paraphrase_preserves_answer"],
        "flip_locality": checks["local_edit"],
        "answer_uniqueness": checks["unique_answer"],
        "answer_change_validity": checks["flip_changes_answer"],
        "outside_knowledge_leakage": checks["self_contained"],
        "fluency_well_formedness": checks["prompt_under_180"] and candidate.extra.get("fluency_ok", True),
    }
    if candidate.extra.get("borderline_fluency") and rng.random() < 0.7:
        truth["fluency_well_formedness"] = False
    if candidate.extra.get("borderline_locality") and rng.random() < 0.7:
        truth["flip_locality"] = False
    return truth


def _annotate_candidate(candidate: Candidate) -> None:
    if not all(candidate.automatic_checks.values()):
        reason = next(key for key, value in candidate.automatic_checks.items() if not value)
        candidate.keep_reason = f"automatic_reject::{reason}"
        return

    truth_rng = _stable_rng(candidate.generation_seed, f"truth::{candidate.cluster_id}")
    truth = _rubric_truth(candidate, truth_rng)
    annotators = []
    disagreement_fields: list[str] = []
    for annotator_idx in (1, 2):
        rng = _stable_rng(candidate.generation_seed, f"annotator_{annotator_idx}::{candidate.cluster_id}")
        rubric = {}
        for field in RUBRIC_FIELDS:
            error_rate = 0.04 if truth[field] else 0.08
            rubric[field] = truth[field] if rng.random() >= error_rate else not truth[field]
        keep_recommendation = all(rubric.values())
        annotators.append(
            {
                "annotator_id": f"audit_r{annotator_idx}",
                "timestamp_utc": _utc_now(),
                "rubric": rubric,
                "recommend_keep": keep_recommendation,
            }
        )

    for field in RUBRIC_FIELDS:
        if annotators[0]["rubric"][field] != annotators[1]["rubric"][field]:
            disagreement_fields.append(field)
    if annotators[0]["recommend_keep"] != annotators[1]["recommend_keep"]:
        disagreement_fields.append("recommend_keep")

    final_rubric = truth.copy()
    candidate.annotator_decisions = annotators
    candidate.audit_final_pass = all(final_rubric.values())
    if disagreement_fields:
        candidate.adjudication_outcome = {
            "adjudicator_id": "audit_adj_1",
            "timestamp_utc": _utc_now(),
            "disagreement_fields": disagreement_fields,
            "final_rubric": final_rubric,
            "final_keep": candidate.audit_final_pass,
        }
        candidate.audit_status = (
            "accepted_after_adjudication" if candidate.audit_final_pass else "rejected_after_adjudication"
        )
    else:
        candidate.audit_status = "accepted_dual_pass" if candidate.audit_final_pass else "rejected_dual_fail"
    candidate.keep_reason = "audited_pass_not_selected" if candidate.audit_final_pass else "audit_reject"


def _arithmetic_candidate(rng: random.Random, seed: int, idx: int) -> Candidate:
    template = rng.choice(["sum", "diff", "prod"])
    a = rng.randint(3, 42)
    b = rng.randint(2, 18)
    c = rng.randint(1, 9)
    flip_b = max(1, b + rng.choice([-3, -2, -1, 1, 2, 3]))
    if template == "sum":
        base = a + b + c
        flip = a + flip_b + c
        q0 = f"Mira has {a} blue tokens, receives {b} more, and then finds {c} extra tokens. How many tokens does she have altogether?"
        q1 = f"After getting {b} additional tokens and finding {c} more, how many tokens does Mira have in total if she started with {a}?"
        q3 = f"Mira has {a} blue tokens, receives {flip_b} more, and then finds {c} extra tokens. How many tokens does she have altogether?"
    elif template == "diff":
        base = a - b + c
        flip = a - flip_b + c
        q0 = f"Mira starts with {a} tokens, gives away {b}, and later finds {c} tokens. How many tokens does she have now?"
        q1 = f"If Mira begins with {a} tokens, hands out {b}, and then recovers {c}, what is her final token count?"
        q3 = f"Mira starts with {a} tokens, gives away {flip_b}, and later finds {c} tokens. How many tokens does she have now?"
    else:
        base = a * b + c
        flip = a * flip_b + c
        q0 = f"A display has {a} shelves with {b} jars on each shelf, plus {c} spare jars beside it. How many jars are there in total?"
        q1 = f"There are {a} shelves, each holding {b} jars, and {c} more jars are off to the side. What is the total number of jars?"
        q3 = f"A display has {a} shelves with {flip_b} jars on each shelf, plus {c} spare jars beside it. How many jars are there in total?"
    q2 = f"{q0} Return digits only."
    q4 = f"{q0} Answer using a short sentence."
    checks = {
        "paraphrase_preserves_answer": True,
        "flip_changes_answer": normalize_answer(str(base), "numeric") != normalize_answer(str(flip), "numeric"),
        "unique_answer": True,
        "local_edit": True,
        "q3_textually_distinct": _q3_is_textually_distinct(q0, q3),
        "prompt_under_180": _record_prompt_tokens(q0, q1, q2, q3, q4) < 180,
        "solver_reproduces_gold": True,
        "self_contained": True,
    }
    fluency_ok = "digits only" not in q1.lower()
    borderline = idx % 11 == 0
    return Candidate(
        cluster_id=f"arith_s{seed}_{idx:03d}",
        family="arithmetic",
        source_type="procedural_template",
        source_date="2026-03-25",
        provenance_url_or_template_id=f"arith::{template}",
        q0=q0,
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        gold_q0=str(base),
        gold_q1=str(base),
        gold_q2=str(base),
        gold_q3=str(flip),
        gold_q4=str(base),
        normalization_rule="numeric",
        flip_template=f"quantity_replacement::{template}",
        solver_or_extractor_type="python_expression",
        generation_seed=seed,
        prompt_tokens_max=_record_prompt_tokens(q0, q1, q2, q3, q4),
        automatic_checks=checks,
        selection_score=_score_checks(checks, rng.random() * 0.08 + (0.0 if borderline else 0.04)),
        raw_gold_q3_without_recompute=str(base),
        extra={"fluency_ok": fluency_ok, "borderline_fluency": borderline and template == "prod"},
    )


def _random_date(rng: random.Random) -> date:
    year = rng.choice([2024, 2025, 2026])
    month = rng.randint(1, 12)
    day = rng.randint(1, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _temporal_candidate(rng: random.Random, seed: int, idx: int) -> Candidate:
    start = _random_date(rng)
    delta = rng.randint(3, 28)
    flip_delta = max(1, delta + rng.choice([-4, -2, -1, 1, 3, 5]))
    base = start + timedelta(days=delta)
    flip = start + timedelta(days=flip_delta)
    q0 = f"A workshop begins on {start.strftime('%B %d, %Y')} and lasts {delta} days. What date is {delta} days after the start?"
    q1 = f"The workshop starts on {start.strftime('%B %d, %Y')}. If you count forward {delta} days, which date do you reach?"
    q2 = f"{q0} Return the final date in YYYY-MM-DD format."
    q3 = q0.replace(f"lasts {delta} days", f"lasts {flip_delta} days", 1).replace(
        f"{delta} days after", f"{flip_delta} days after", 1
    )
    q4 = f"{q0} Respond using Month DD, YYYY."
    checks = {
        "paraphrase_preserves_answer": True,
        "flip_changes_answer": base != flip,
        "unique_answer": True,
        "local_edit": True,
        "q3_textually_distinct": _q3_is_textually_distinct(q0, q3),
        "prompt_under_180": _record_prompt_tokens(q0, q1, q2, q3, q4) < 180,
        "solver_reproduces_gold": True,
        "self_contained": True,
    }
    return Candidate(
        cluster_id=f"temp_s{seed}_{idx:03d}",
        family="temporal",
        source_type="procedural_template",
        source_date="2026-03-25",
        provenance_url_or_template_id="temporal::offset_date",
        q0=q0,
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        gold_q0=base.isoformat(),
        gold_q1=base.isoformat(),
        gold_q2=base.isoformat(),
        gold_q3=flip.isoformat(),
        gold_q4=base.strftime("%B %d, %Y"),
        normalization_rule="date",
        flip_template="interval_length_edit",
        solver_or_extractor_type="calendar_solver",
        generation_seed=seed,
        prompt_tokens_max=_record_prompt_tokens(q0, q1, q2, q3, q4),
        automatic_checks=checks,
        selection_score=_score_checks(checks, rng.random() * 0.08 + (0.03 if idx % 9 else -0.01)),
        raw_gold_q3_without_recompute=base.strftime("%B %d, %Y"),
        extra={"borderline_locality": idx % 13 == 0},
    )


def _table_answer(rows: list[dict[str, Any]], threshold: int, mode: str) -> str:
    filtered = [row for row in rows if row["score"] >= threshold]
    if mode == "count":
        return str(len(filtered))
    if not filtered:
        return "none"
    top = sorted(filtered, key=lambda row: (-row["score"], row["name"]))[0]
    return top["name"]


def _table_candidate(rng: random.Random, seed: int, idx: int) -> Candidate:
    names = rng.sample(["Ada", "Ben", "Cora", "Dax", "Eli", "Faye", "Gio", "Hana", "Iris", "Jude"], 5)
    rows = [{"name": name, "score": rng.randint(12, 98), "team": rng.choice(["red", "blue"])} for name in names]
    sorted_scores = sorted(row["score"] for row in rows)
    threshold = sorted_scores[rng.choice([1, 2, 3])]
    mode = rng.choice(["count", "best"])
    base = _table_answer(rows, threshold, mode)
    flip_choices = [value for value in range(10, 99) if _table_answer(rows, value, mode) != base]
    flip_threshold = rng.choice(flip_choices) if flip_choices else min(98, threshold + 7)
    flip = _table_answer(rows, flip_threshold, mode)
    table = "; ".join(f"{row['name']} scored {row['score']} for team {row['team']}" for row in rows)
    if mode == "count":
        q0 = f"Table rows: {table}. How many people have a score of at least {threshold}?"
        q1 = f"Using this table only, count the people whose score is {threshold} or higher: {table}"
        q3 = f"Table rows: {table}. How many people have a score of at least {flip_threshold}?"
        q4 = f"{q0} Answer with a short sentence."
    else:
        q0 = f"Table rows: {table}. Among the people with score at least {threshold}, who has the highest score?"
        q1 = f"Look only at this table: {table}. Which person has the top score among rows with score >= {threshold}?"
        q3 = f"Table rows: {table}. Among the people with score at least {flip_threshold}, who has the highest score?"
        q4 = f"{q0} Answer with the name only."
    q2 = f"{q0} Return only the final answer."
    checks = {
        "paraphrase_preserves_answer": True,
        "flip_changes_answer": normalize_answer(base, "alnum_lower") != normalize_answer(flip, "alnum_lower"),
        "unique_answer": normalize_answer(base, "alnum_lower") != "none" or mode == "count",
        "local_edit": True,
        "q3_textually_distinct": _q3_is_textually_distinct(q0, q3),
        "prompt_under_180": _record_prompt_tokens(q0, q1, q2, q3, q4) < 180,
        "solver_reproduces_gold": True,
        "self_contained": True,
    }
    return Candidate(
        cluster_id=f"table_s{seed}_{idx:03d}",
        family="table",
        source_type="procedural_template",
        source_date="2026-03-25",
        provenance_url_or_template_id=f"table::{mode}",
        q0=q0,
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        gold_q0=base,
        gold_q1=base,
        gold_q2=base,
        gold_q3=flip,
        gold_q4=base,
        normalization_rule="numeric" if mode == "count" else "alnum_lower",
        flip_template="filter_predicate_edit",
        solver_or_extractor_type="table_query",
        generation_seed=seed,
        prompt_tokens_max=_record_prompt_tokens(q0, q1, q2, q3, q4),
        automatic_checks=checks,
        selection_score=_score_checks(checks, rng.random() * 0.08 + (0.01 if mode == "count" else 0.03)),
        raw_gold_q3_without_recompute=base,
        extra={"mode": mode, "borderline_locality": idx % 10 == 0 and mode == "best"},
    )


def _strict_evidence_candidate(seed: int, idx: int) -> Candidate:
    rng = _stable_rng(seed, f"strict_evidence::{idx}")
    benchmark_name = rng.choice(["AtlasQA", "SignalBench", "VectorEval", "NovaCheck"])
    task_count = rng.randint(9, 21)
    flip_count = task_count + rng.choice([-3, -2, 2, 4])
    context = rng.choice(["32k", "64k", "128k"])
    snippet = (
        f"Versioned release note dated {rng.choice(['January 14, 2026', 'March 02, 2025', 'August 19, 2025'])}: "
        f"{benchmark_name} reports {task_count} evaluation tasks and supports a {context} context window."
    )
    q0 = f"Snippet: {snippet} How many evaluation tasks are reported?"
    q1 = f"Using only this snippet, what is the number of evaluation tasks? {snippet}"
    q2 = f"{q0} Return only the number."
    q3 = q0.replace(f"{task_count} evaluation tasks", f"{flip_count} evaluation tasks", 1)
    q4 = f"{q0} Answer with a full sentence."
    return Candidate(
        cluster_id=f"strict_evidence_s{seed}_{idx:03d}",
        family="evidence",
        source_type="versioned_release_note",
        source_date="2026-03-25",
        provenance_url_or_template_id="evidence::strict_release_note",
        q0=q0,
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        gold_q0=str(task_count),
        gold_q1=str(task_count),
        gold_q2=str(task_count),
        gold_q3=str(flip_count),
        gold_q4=str(task_count),
        normalization_rule="numeric",
        flip_template="numeric_slot_replacement",
        solver_or_extractor_type="exact_extraction",
        generation_seed=seed,
        prompt_tokens_max=_record_prompt_tokens(q0, q1, q2, q3, q4),
        automatic_checks={
            "paraphrase_preserves_answer": True,
            "flip_changes_answer": True,
            "unique_answer": True,
            "local_edit": True,
            "q3_textually_distinct": _q3_is_textually_distinct(q0, q3),
            "prompt_under_180": True,
            "solver_reproduces_gold": True,
            "self_contained": True,
        },
        selection_score=1.0 + rng.random() * 0.05,
        raw_gold_q3_without_recompute=str(task_count),
        strict_evidence_policy_pass=True,
        extra={"fluency_ok": True},
    )


def _relaxed_evidence_candidate(seed: int, idx: int) -> Candidate:
    rng = _stable_rng(seed, f"relaxed_evidence::{idx}")
    subject = rng.choice(["the project", "it", "the announcement", "the update"])
    first = rng.randint(8, 18)
    second = first + rng.choice([0, 1, 2])
    snippet = (
        f"Blog-style note from sometime in 2025: {subject} mentions {first} checks in one sentence. "
        f"A later sentence says the team expanded to {second} checks after revisions."
    )
    q0 = f"Snippet: {snippet} How many checks are reported?"
    q1 = f"From this snippet, what number of checks is given? {snippet}"
    q2 = f"{q0} Return only the number."
    q3 = q0.replace("How many checks are reported?", "How many checks are reported after revisions?", 1)
    q4 = f"{q0} Answer with a full sentence."
    unique = first != second
    fluent = idx % 5 != 0
    return Candidate(
        cluster_id=f"relaxed_evidence_s{seed}_{idx:03d}",
        family="evidence_relaxed",
        source_type="blog_post",
        source_date="2025-07-01",
        provenance_url_or_template_id="evidence::relaxed_blog_note",
        q0=q0,
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        gold_q0=str(first),
        gold_q1=str(first),
        gold_q2=str(first),
        gold_q3=str(second),
        gold_q4=str(first),
        normalization_rule="numeric",
        flip_template="question_slot_retarget",
        solver_or_extractor_type="non_release_validation",
        generation_seed=seed,
        prompt_tokens_max=_record_prompt_tokens(q0, q1, q2, q3, q4),
        automatic_checks={
            "paraphrase_preserves_answer": True,
            "flip_changes_answer": unique,
            "unique_answer": unique,
            "local_edit": idx % 4 != 0,
            "q3_textually_distinct": _q3_is_textually_distinct(q0, q3),
            "prompt_under_180": True,
            "solver_reproduces_gold": unique,
            "self_contained": idx % 3 != 0,
        },
        selection_score=0.6 + rng.random() * 0.05,
        raw_gold_q3_without_recompute=str(first),
        strict_evidence_policy_pass=False,
        extra={"fluency_ok": fluent, "borderline_fluency": not fluent},
    )


def _select_release(candidates: list[Candidate], family: str, mode: str) -> list[Candidate]:
    keep_target = FAMILY_TARGETS[family]["keep"]
    if mode == "audited":
        eligible = [candidate for candidate in candidates if all(candidate.automatic_checks.values()) and candidate.audit_final_pass]
    else:
        eligible = [candidate for candidate in candidates if all(candidate.automatic_checks.values())]
    ordered = sorted(eligible, key=lambda item: (-item.selection_score, item.cluster_id))
    return ordered[:keep_target]


def _build_audit_rows(candidates: list[Candidate]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations: list[dict[str, Any]] = []
    adjudications: list[dict[str, Any]] = []
    for candidate in candidates:
        for decision in candidate.annotator_decisions:
            annotations.append(
                {
                    "cluster_id": candidate.cluster_id,
                    "family": candidate.family,
                    "annotator_id": decision["annotator_id"],
                    "recommend_keep": decision["recommend_keep"],
                    **decision["rubric"],
                }
            )
        if candidate.adjudication_outcome is not None:
            adjudications.append(
                {
                    "cluster_id": candidate.cluster_id,
                    "family": candidate.family,
                    "adjudicator_id": candidate.adjudication_outcome["adjudicator_id"],
                    "final_keep": candidate.adjudication_outcome["final_keep"],
                    "disagreement_fields": ",".join(candidate.adjudication_outcome["disagreement_fields"]),
                }
            )
    return annotations, adjudications


def _agreement_summary(candidates: list[Candidate]) -> dict[str, Any]:
    annotated = [candidate for candidate in candidates if len(candidate.annotator_decisions) == 2]
    if not annotated:
        return {}
    summary: dict[str, Any] = {}
    for field in RUBRIC_FIELDS + ["recommend_keep"]:
        left = []
        right = []
        for candidate in annotated:
            if field == "recommend_keep":
                left.append(int(candidate.annotator_decisions[0]["recommend_keep"]))
                right.append(int(candidate.annotator_decisions[1]["recommend_keep"]))
            else:
                left.append(int(candidate.annotator_decisions[0]["rubric"][field]))
                right.append(int(candidate.annotator_decisions[1]["rubric"][field]))
        percent = sum(int(a == b) for a, b in zip(left, right)) / len(left)
        summary[field] = {"percent_agreement": percent, "cohen_kappa": _agreement(left, right)}
    return summary


def build_seed_benchmark(seed: int, out_dir) -> dict[str, Any]:
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    for idx in range(FAMILY_TARGETS["arithmetic"]["candidates"]):
        candidates.append(_arithmetic_candidate(rng, seed, idx))
    for idx in range(FAMILY_TARGETS["temporal"]["candidates"]):
        candidates.append(_temporal_candidate(rng, seed, idx))
    for idx in range(FAMILY_TARGETS["table"]["candidates"]):
        candidates.append(_table_candidate(rng, seed, idx))
    for candidate in candidates:
        _annotate_candidate(candidate)

    by_family: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.family, []).append(candidate)

    accepted: list[Candidate] = []
    auto_only_release: list[Candidate] = []
    auto_comparison_subset: list[dict[str, Any]] = []
    audited_comparison_subset: list[Candidate] = []
    recompute_validation: list[Candidate] = []
    no_recompute_validation: list[dict[str, Any]] = []

    for family in ["arithmetic", "temporal", "table"]:
        family_candidates = by_family[family]
        audited = _select_release(family_candidates, family, "audited")
        automatic = _select_release(family_candidates, family, "automatic")
        for rank, candidate in enumerate(audited):
            candidate.keep = True
            candidate.keep_reason = "accepted_core_release"
            candidate.construction_split = _assignment_split(rank)
            accepted.append(candidate)
        for rank, candidate in enumerate(automatic):
            copy_row = candidate.asdict()
            copy_row["construction_split"] = _assignment_split(rank)
            copy_row["keep_reason"] = "accepted_automatic_only_release"
            auto_only_release.append(copy_row)

        held_out_auto = [
            row
            for row in sorted(
                [
                    candidate.asdict()
                    for candidate in family_candidates
                    if all(candidate.automatic_checks.values()) and candidate.cluster_id not in {item.cluster_id for item in automatic}
                ],
                key=lambda item: (-item["selection_score"], item["cluster_id"]),
            )
        ][:2]
        for rank, row in enumerate(held_out_auto):
            row["construction_split"] = _assignment_split(rank)
            row["keep_reason"] = "heldout_automatic_only_subset"
            auto_comparison_subset.append(row)

        held_out_audited = [
            candidate
            for candidate in sorted(
                [row for row in family_candidates if row.audit_final_pass and row.cluster_id not in {item.cluster_id for item in audited}],
                key=lambda item: (-item.selection_score, item.cluster_id),
            )
        ][:4]
        audited_comparison_subset.extend(held_out_audited[:2])
        recompute_validation.extend(held_out_audited[2:4] if len(held_out_audited) >= 4 else held_out_audited[:2])

    for rank, candidate in enumerate(audited_comparison_subset):
        candidate.construction_split = _assignment_split(rank)
    for rank, candidate in enumerate(recompute_validation):
        candidate.construction_split = _assignment_split(rank)
        alt = candidate.asdict()
        alt["gold_q3"] = candidate.raw_gold_q3_without_recompute
        alt["keep_reason"] = "validation_no_recompute_labels"
        no_recompute_validation.append(alt)

    strict_evidence_candidates = [_strict_evidence_candidate(seed, idx) for idx in range(STRICT_EVIDENCE_VALIDATION)]
    relaxed_evidence_candidates = [_relaxed_evidence_candidate(seed, idx) for idx in range(RELAXED_EVIDENCE_VALIDATION)]
    for candidate in strict_evidence_candidates + relaxed_evidence_candidates:
        _annotate_candidate(candidate)
    for rank, candidate in enumerate(strict_evidence_candidates):
        candidate.construction_split = _assignment_split(rank)
    for rank, candidate in enumerate(relaxed_evidence_candidates):
        candidate.construction_split = _assignment_split(rank)
    strict_evidence_pool = [candidate.asdict() for candidate in strict_evidence_candidates]
    relaxed_evidence_pool = [candidate.asdict() for candidate in relaxed_evidence_candidates]

    candidate_rows = [candidate.asdict() for candidate in candidates]
    accepted_rows = [candidate.asdict() for candidate in accepted]
    annotations, adjudications = _build_audit_rows(candidates + strict_evidence_candidates + relaxed_evidence_candidates)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "candidates.jsonl", candidate_rows)
    write_json(out_dir / "accepted_clusters.json", accepted_rows)
    write_json(out_dir / "auto_only_release.json", auto_only_release)
    write_json(out_dir / "auto_comparison_subset.json", auto_comparison_subset)
    write_json(out_dir / "audited_comparison_subset.json", [candidate.asdict() for candidate in audited_comparison_subset])
    write_json(out_dir / "recompute_validation.json", [candidate.asdict() for candidate in recompute_validation])
    write_json(out_dir / "no_recompute_validation.json", no_recompute_validation)
    write_json(out_dir / "strict_evidence_validation.json", strict_evidence_pool)
    write_json(out_dir / "relaxed_evidence_validation.json", relaxed_evidence_pool)
    write_json(out_dir / "annotation_decisions.json", annotations)
    write_json(out_dir / "adjudications.json", adjudications)

    construction_stats: dict[str, dict[str, Any]] = {}
    for family in ["arithmetic", "temporal", "table"]:
        family_rows = [row for row in candidate_rows if row["family"] == family]
        auto_pass = [row for row in family_rows if all(row["automatic_checks"].values())]
        audited_pass = [row for row in family_rows if row["audit_final_pass"]]
        kept = [row for row in family_rows if row["keep"]]
        rejection_reasons: dict[str, int] = {}
        for row in family_rows:
            rejection_reasons[row["keep_reason"]] = rejection_reasons.get(row["keep_reason"], 0) + 1
        construction_stats[family] = {
            "candidate_count": len(family_rows),
            "automatic_pass_count": len(auto_pass),
            "audited_pass_count": len(audited_pass),
            "kept_count": len(kept),
            "keep_rate": len(kept) / max(1, len(family_rows)),
            "adjudication_rate": sum(int(row["adjudication_outcome"] is not None) for row in family_rows) / max(1, len(auto_pass)),
            "rejections": rejection_reasons,
        }

    summary = {
        "seed": seed,
        "seeds_all": SEEDS,
        "accepted_cluster_count": len(accepted_rows),
        "core_release_type": "synthetic_procedural_core_pilot",
        "human_audit_available": False,
        "synthetic_dual_audit_pipeline": True,
        "evidence_slice_status": "validation_only_not_core",
        "construction_stats": construction_stats,
        "agreement": _agreement_summary(candidates),
        "split_counts": {
            split: sum(1 for row in accepted_rows if row["construction_split"] == split)
            for split in ["A", "B"]
        },
        "validation_pool_sizes": {
            "auto_only_release": len(auto_only_release),
            "auto_comparison_subset": len(auto_comparison_subset),
            "audited_comparison_subset": len(audited_comparison_subset),
            "recompute_validation": len(recompute_validation),
            "strict_evidence_validation": len(strict_evidence_pool),
            "relaxed_evidence_validation": len(relaxed_evidence_pool),
        },
        "accepted_q3_textually_distinct_count": sum(
            int(row["automatic_checks"]["q3_textually_distinct"]) for row in accepted_rows
        ),
        "accepted_q3_gold_changed_count": sum(
            int(row["gold_q3"] != row["gold_q0"]) for row in accepted_rows
        ),
        "strict_evidence_gate_metrics": {
            "pilot_candidate_count": len(strict_evidence_candidates),
            "automatic_pass_count": sum(int(all(candidate.automatic_checks.values())) for candidate in strict_evidence_candidates),
            "dual_verified_count": sum(int(candidate.audit_final_pass) for candidate in strict_evidence_candidates),
            "agreement_answer_uniqueness": _agreement(
                [int(candidate.annotator_decisions[0]["rubric"]["answer_uniqueness"]) for candidate in strict_evidence_candidates],
                [int(candidate.annotator_decisions[1]["rubric"]["answer_uniqueness"]) for candidate in strict_evidence_candidates],
            ),
            "agreement_answer_change_validity": _agreement(
                [int(candidate.annotator_decisions[0]["rubric"]["answer_change_validity"]) for candidate in strict_evidence_candidates],
                [int(candidate.annotator_decisions[1]["rubric"]["answer_change_validity"]) for candidate in strict_evidence_candidates],
            ),
            "edited_evidence_fraction": 1.0,
            "borderline_fraction": sum(
                int(candidate.extra.get("borderline_fluency", False)) for candidate in strict_evidence_candidates
            ) / max(1, len(strict_evidence_candidates)),
            "passes_content_gates": False,
            "human_annotation_available": False,
            "passes_core_gate": False,
            "core_gate_failure_reasons": ["real_human_annotation_unavailable"],
        },
    }
    strict_metrics = summary["strict_evidence_gate_metrics"]
    strict_metrics["passes_content_gates"] = (
        strict_metrics["automatic_pass_count"] >= 20
        and strict_metrics["dual_verified_count"] >= 18
        and strict_metrics["agreement_answer_uniqueness"] >= 0.85
        and strict_metrics["agreement_answer_change_validity"] >= 0.85
        and strict_metrics["edited_evidence_fraction"] >= 0.5
        and strict_metrics["borderline_fraction"] < 0.1
    )
    if not strict_metrics["passes_content_gates"]:
        strict_metrics["core_gate_failure_reasons"].append("content_gates_failed")
    write_json(out_dir / "construction_summary.json", summary)
    return summary
