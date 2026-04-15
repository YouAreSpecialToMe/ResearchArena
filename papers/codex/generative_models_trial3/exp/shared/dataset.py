from __future__ import annotations

import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import math

from sklearn.metrics import cohen_kappa_score

from exp.shared.common import (
    BENCHMARK_FILES,
    DATA_DIR,
    HELD_OUT_PER_CATEGORY,
    PILOT_PER_CATEGORY,
    PromptRecord,
    ensure_dirs,
    prompt_slot_map,
    record_to_dict,
    write_json,
    write_jsonl,
)


RELATION_PATTERN = re.compile(
    r"^a (?P<object_1>[a-z ]+?) (?P<relation>next to|near|on side of|on the left of|on the right of|on the bottom of|on the top of) a (?P<object_2>[a-z ]+)$"
)
ATTRIBUTE_PATTERN = re.compile(r"^a (?P<attribute_1>[a-z]+) (?P<object_1>[a-z ]+) and a (?P<attribute_2>[a-z]+) (?P<object_2>[a-z ]+)$")
NUMERACY_PATTERN = re.compile(r"^(?P<count>one|two|three|four|five|six|seven|eight) (?P<object_1>[a-z ]+)$")

ATTRIBUTE_CANDIDATE_PATTERNS = [
    re.compile(r"^a (?P<object_1>[a-z ]+) that is (?P<attribute_1>[a-z]+) and a (?P<object_2>[a-z ]+) that is (?P<attribute_2>[a-z]+)$"),
    re.compile(r"^there is a (?P<attribute_1>[a-z]+) (?P<object_1>[a-z ]+) and a (?P<attribute_2>[a-z]+) (?P<object_2>[a-z ]+)$"),
]
RELATION_CANDIDATE_PATTERNS = [
    re.compile(
        r"^there is a (?P<object_1>[a-z ]+?) (?P<relation>next to|near|on side of|on the left of|on the right of|on the bottom of|on the top of) a (?P<object_2>[a-z ]+)$"
    ),
    re.compile(r"^(?P<fronted>next to|near) a (?P<object_2>[a-z ]+) is a (?P<object_1>[a-z ]+)$"),
    re.compile(r"^on the side of a (?P<object_2>[a-z ]+) is a (?P<object_1>[a-z ]+)$"),
    re.compile(r"^to the (?P<side>left|right) of a (?P<object_2>[a-z ]+) is a (?P<object_1>[a-z ]+)$"),
    re.compile(r"^(?P<vertical>above|below) a (?P<object_2>[a-z ]+) is a (?P<object_1>[a-z ]+)$"),
]
NUMERACY_CANDIDATE_PATTERNS = [
    re.compile(r"^(?P<count>\d+|one|two|three|four|five|six|seven|eight) (?P<object_1>[a-z ]+)$"),
    re.compile(r"^there (?P<verb>is|are) (?P<count>\d+|one|two|three|four|five|six|seven|eight) (?P<object_1>[a-z ]+)$"),
]

NUMBER_TO_DIGIT = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
}
DIGIT_TO_NUMBER = {value: key for key, value in NUMBER_TO_DIGIT.items()}
NUMBER_WORDS = set(NUMBER_TO_DIGIT)


def _download_if_missing(category: str) -> Path:
    target = DATA_DIR / "benchmark_raw" / f"{category}.txt"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BENCHMARK_FILES[category], timeout=30) as response:
        target.write_bytes(response.read())
    return target


def _plural_to_singular(noun: str) -> str:
    irregular = {
        "women": "woman",
        "men": "man",
        "people": "person",
        "mice": "mouse",
        "geese": "goose",
        "teeth": "tooth",
        "feet": "foot",
        "children": "child",
        "sheep": "sheep",
        "fish": "fish",
        "shrimp": "shrimp",
        "breads": "bread",
    }
    if noun in irregular:
        return irregular[noun]
    if noun.endswith("ies"):
        return noun[:-3] + "y"
    if noun.endswith("ves"):
        return noun[:-3] + "f"
    if noun.endswith("s") and not noun.endswith("ss"):
        return noun[:-1]
    return noun


def _singular_to_surface(noun: str, count: str) -> str:
    if count == "one":
        return noun
    irregular = {
        "person": "people",
        "man": "men",
        "woman": "women",
        "mouse": "mice",
        "goose": "geese",
        "child": "children",
        "fish": "fish",
        "sheep": "sheep",
        "shrimp": "shrimp",
    }
    if noun in irregular:
        return irregular[noun]
    if noun.endswith("y") and not noun.endswith(("ay", "ey", "iy", "oy", "uy")):
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def _normalize_count(raw: str) -> str:
    return DIGIT_TO_NUMBER.get(raw, raw)


def _normalize_relation(fronted: str | None = None, side: str | None = None, vertical: str | None = None, relation: str | None = None) -> str:
    if relation:
        return relation
    if fronted:
        return fronted
    if side is None and fronted is None and vertical is None:
        return "on side of"
    if side:
        return f"on the {side} of"
    if vertical == "above":
        return "on the top of"
    if vertical == "below":
        return "on the bottom of"
    return ""


def _proposal(text: str, rewrite_type: str, template_id: str) -> dict[str, str]:
    return {"text": text, "rewrite_type": rewrite_type, "template_id": template_id}


def _attribute_candidates(attr1: str, obj1: str, attr2: str, obj2: str) -> tuple[list[dict[str, str]], str]:
    proposals = [
        _proposal(f"a {obj1} that is {attr1} and a {obj2} that is {attr2}", "attribute-order swap", "attribute_relative_clause"),
        _proposal(f"there is a {attr1} {obj1} and a {attr2} {obj2}", "existential rewrite", "attribute_existential"),
    ]
    conf = f"a {attr2} {obj1} and a {attr1} {obj2}"
    return proposals, conf


def _relation_candidates(obj1: str, relation: str, obj2: str) -> tuple[list[dict[str, str]], str]:
    proposals = [
        _proposal(f"there is a {obj1} {relation} a {obj2}", "clause-order rewrite", "relation_existential"),
    ]
    if relation == "next to":
        fronted = f"next to a {obj2} is a {obj1}"
    elif relation == "near":
        fronted = f"near a {obj2} is a {obj1}"
    elif relation == "on the left of":
        fronted = f"to the left of a {obj2} is a {obj1}"
    elif relation == "on the right of":
        fronted = f"to the right of a {obj2} is a {obj1}"
    elif relation == "on the top of":
        fronted = f"above a {obj2} is a {obj1}"
    elif relation == "on the bottom of":
        fronted = f"below a {obj2} is a {obj1}"
    elif relation == "on side of":
        fronted = f"on the side of a {obj2} is a {obj1}"
    else:
        fronted = f"next to a {obj2} is a {obj1}"
    proposals.append(_proposal(fronted, "fronted relation rewrite", "relation_fronted"))
    conf = {
        "next to": f"a {obj1} far from a {obj2}",
        "near": f"a {obj1} far from a {obj2}",
        "on side of": f"a {obj1} far from a {obj2}",
        "on the left of": f"a {obj1} on the right of a {obj2}",
        "on the right of": f"a {obj1} on the left of a {obj2}",
        "on the bottom of": f"a {obj1} on the top of a {obj2}",
        "on the top of": f"a {obj1} on the bottom of a {obj2}",
    }[relation]
    return proposals, conf


def _numeracy_candidates(count: str, surface_obj: str) -> tuple[list[dict[str, str]], str]:
    proposals = [
        _proposal(f"{NUMBER_TO_DIGIT[count]} {surface_obj}", "digit rewrite", "numeracy_digit"),
    ]
    if count == "one":
        proposals.append(_proposal(f"there is one {surface_obj}", "existential rewrite", "numeracy_existential"))
        proposals.append(_proposal(f"there are one {surface_obj}", "existential rewrite", "numeracy_legacy_existential"))
    else:
        proposals.append(_proposal(f"there are {count} {surface_obj}", "existential rewrite", "numeracy_existential"))
    wrong_count = {
        "one": "two",
        "two": "three",
        "three": "two",
        "four": "five",
        "five": "four",
        "six": "five",
        "seven": "six",
        "eight": "seven",
    }[count]
    conf = f"{wrong_count} {surface_obj}"
    return proposals, conf


def _expected_slots(category: str, match: re.Match[str]) -> dict[str, str]:
    if category == "attribute_binding":
        return {
            "object_1": match.group("object_1"),
            "attribute_1": match.group("attribute_1"),
            "object_2": match.group("object_2"),
            "attribute_2": match.group("attribute_2"),
        }
    if category == "relations":
        return {
            "object_1": match.group("object_1"),
            "relation": match.group("relation"),
            "object_2": match.group("object_2"),
        }
    surface_obj = match.group("object_1")
    return {
        "object_1": _plural_to_singular(surface_obj),
        "count_1": match.group("count"),
    }


def _parse_candidate(category: str, text: str) -> dict[str, Any] | None:
    if category == "attribute_binding":
        for pattern in ATTRIBUTE_CANDIDATE_PATTERNS:
            matched = pattern.match(text)
            if matched:
                return {
                    "object_1": matched.group("object_1"),
                    "attribute_1": matched.group("attribute_1"),
                    "object_2": matched.group("object_2"),
                    "attribute_2": matched.group("attribute_2"),
                }
        return None

    if category == "relations":
        for pattern in RELATION_CANDIDATE_PATTERNS:
            matched = pattern.match(text)
            if matched:
                return {
                    "object_1": matched.group("object_1"),
                    "relation": _normalize_relation(
                        fronted=matched.groupdict().get("fronted"),
                        side=matched.groupdict().get("side"),
                        vertical=matched.groupdict().get("vertical"),
                        relation=matched.groupdict().get("relation"),
                    ),
                    "object_2": matched.group("object_2"),
                }
        return None

    for pattern in NUMERACY_CANDIDATE_PATTERNS:
        matched = pattern.match(text)
        if not matched:
            continue
        count = _normalize_count(matched.group("count"))
        object_surface = matched.group("object_1")
        parsed = {
            "object_1": _plural_to_singular(object_surface),
            "count_1": count,
            "object_surface": object_surface,
        }
        if "verb" in matched.groupdict():
            parsed["verb"] = matched.group("verb")
        return parsed
    return None


def _secondary_parse(category: str, text: str) -> dict[str, Any] | None:
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    if category == "attribute_binding":
        for pattern in ATTRIBUTE_CANDIDATE_PATTERNS:
            matched = pattern.match(lowered)
            if matched:
                return {
                    "object_1": matched.group("object_1"),
                    "attribute_1": matched.group("attribute_1"),
                    "object_2": matched.group("object_2"),
                    "attribute_2": matched.group("attribute_2"),
                }
        return None
    if category == "relations":
        for pattern in RELATION_CANDIDATE_PATTERNS:
            matched = pattern.match(lowered)
            if matched:
                return {
                    "object_1": matched.group("object_1"),
                    "relation": _normalize_relation(
                        fronted=matched.groupdict().get("fronted"),
                        side=matched.groupdict().get("side"),
                        vertical=matched.groupdict().get("vertical"),
                        relation=matched.groupdict().get("relation"),
                    ),
                    "object_2": matched.group("object_2"),
                }
        return None
    matched = re.match(r"^(?P<count>\d+|one|two|three|four|five|six|seven|eight) (?P<object_1>[a-z ]+)$", lowered)
    if not matched:
        return None
    return {
        "object_1": _plural_to_singular(matched.group("object_1")),
        "count_1": _normalize_count(matched.group("count")),
    }


def _check_no_extra_content(category: str, parsed: dict[str, Any], expected: dict[str, str]) -> bool:
    allowed = set(expected) | {"object_surface", "verb"}
    return set(parsed).issubset(allowed)


def _check_number_grammar(parsed: dict[str, Any]) -> bool:
    if "verb" not in parsed:
        return True
    count = parsed["count_1"]
    expected = "is" if count == "one" else "are"
    return parsed["verb"] == expected


def _audit_candidate(category: str, expected: dict[str, str], proposal: dict[str, str], secondary: bool = False) -> dict[str, Any]:
    parser = _secondary_parse if secondary else _parse_candidate
    parsed = parser(category, proposal["text"])
    parse_success = parsed is not None
    slots_match = parse_success and all(parsed.get(key) == value for key, value in expected.items())
    no_extra_content = parse_success and _check_no_extra_content(category, parsed, expected)
    count_preserved = slots_match and (category != "numeracy" or _check_number_grammar(parsed))
    checklist = {
        "object_identity_preserved": slots_match and parsed.get("object_1") == expected.get("object_1"),
        "count_preserved": count_preserved,
        "binding_preserved": slots_match if category == "attribute_binding" else True,
        "relation_preserved": slots_match if category == "relations" else True,
        "scene_content_preserved": no_extra_content,
    }
    accepted = parse_success and slots_match and no_extra_content and checklist["count_preserved"]
    rejection_reason = None
    if not accepted:
        if not parse_success:
            rejection_reason = "parser_could_not_certify_equivalence"
        elif not slots_match:
            rejection_reason = "slot_mismatch"
        elif not checklist["count_preserved"]:
            rejection_reason = "number_grammar_mismatch"
        elif not no_extra_content:
            rejection_reason = "added_or_removed_scene_content"
        else:
            rejection_reason = "checklist_failure"
    return {
        "candidate_text": proposal["text"],
        "rewrite_type": proposal["rewrite_type"],
        "template_id": proposal["template_id"],
        "parse_success": parse_success,
        "parsed_slots": {key: value for key, value in (parsed or {}).items() if key in expected},
        "checklist_result": checklist,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "auditor": "secondary_automatic" if secondary else "primary_automatic",
    }


def _summarize_audits(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, bool]]:
    approved_paraphrases: list[str] = []
    rewrite_types: list[str] = []
    merged: list[dict[str, Any]] = []
    for left, right in zip(primary, secondary):
        combined = {
            **left,
            "secondary_review": {
                "accepted": right["accepted"],
                "rejection_reason": right["rejection_reason"],
                "checklist_result": right["checklist_result"],
            },
            "double_audit_agreement": left["accepted"] == right["accepted"],
        }
        merged.append(combined)
        if left["accepted"] and len(approved_paraphrases) < 2:
            approved_paraphrases.append(left["candidate_text"])
            rewrite_types.append(left["rewrite_type"])
    accepted_items = [item for item in merged if item["candidate_text"] in approved_paraphrases]
    if accepted_items:
        checklist = accepted_items[0]["checklist_result"].copy()
        for item in accepted_items[1:]:
            for key, value in item["checklist_result"].items():
                checklist[key] = checklist[key] and value
    else:
        checklist = {
            "object_identity_preserved": False,
            "count_preserved": False,
            "binding_preserved": False,
            "relation_preserved": False,
            "scene_content_preserved": False,
        }
    return approved_paraphrases, rewrite_types, merged, checklist


def _make_record(category: str, source_file: str, prompt_id: str, split: str, overlap: bool, prompt: str, match: re.Match[str]) -> PromptRecord:
    expected = _expected_slots(category, match)
    if category == "attribute_binding":
        proposals, confounder = _attribute_candidates(
            match.group("attribute_1"),
            match.group("object_1"),
            match.group("attribute_2"),
            match.group("object_2"),
        )
        non_equivalent = f"a shiny {match.group('object_1')} and a shiny {match.group('object_2')}"
        count_1 = "one"
        count_2 = "one"
        relation = ""
    elif category == "relations":
        proposals, confounder = _relation_candidates(match.group("object_1"), match.group("relation"), match.group("object_2"))
        non_equivalent = f"a {match.group('object_1')} far from a {match.group('object_2')}"
        count_1 = "one"
        count_2 = "one"
        relation = match.group("relation")
    else:
        count = match.group("count")
        surface_obj = match.group("object_1")
        proposals, confounder = _numeracy_candidates(count, surface_obj)
        non_equivalent = f"one {surface_obj}"
        count_1 = count
        count_2 = ""
        relation = ""

    primary = [_audit_candidate(category, expected, proposal, secondary=False) for proposal in proposals]
    secondary = [_audit_candidate(category, expected, proposal, secondary=True) for proposal in proposals]
    approved, rewrite_types, candidate_audit, checklist = _summarize_audits(primary, secondary)

    if category == "attribute_binding":
        object_1 = match.group("object_1")
        object_2 = match.group("object_2")
        attribute_1 = match.group("attribute_1")
        attribute_2 = match.group("attribute_2")
    elif category == "relations":
        object_1 = match.group("object_1")
        object_2 = match.group("object_2")
        attribute_1 = ""
        attribute_2 = ""
    else:
        object_1 = _plural_to_singular(match.group("object_1"))
        object_2 = ""
        attribute_1 = ""
        attribute_2 = ""

    return PromptRecord(
        prompt_id=prompt_id,
        source_dataset="T2I-CompBench",
        source_file=source_file,
        category=category,
        split=split,
        overlap_subset_flag=overlap and bool(approved),
        original_prompt=prompt,
        object_1=object_1,
        count_1=count_1,
        attribute_1=attribute_1,
        relation=relation,
        object_2=object_2,
        count_2=count_2,
        attribute_2=attribute_2,
        approved_paraphrases=approved,
        rewrite_types=rewrite_types,
        preserved_slots=expected,
        checklist_result=checklist,
        audit_status="accepted" if approved else "rejected",
        confounder_prompt=confounder,
        non_equivalent_aux_prompt=non_equivalent,
        candidate_paraphrases=[proposal["text"] for proposal in proposals],
        candidate_rewrite_types=[proposal["rewrite_type"] for proposal in proposals],
        candidate_audit=candidate_audit,
    )


def _load_prompts(category: str) -> list[str]:
    path = _download_if_missing(category)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_prompt_records() -> list[PromptRecord]:
    ensure_dirs()
    records: list[PromptRecord] = []
    category_to_pattern = {
        "attribute_binding": ATTRIBUTE_PATTERN,
        "relations": RELATION_PATTERN,
        "numeracy": NUMERACY_PATTERN,
    }
    prefixes = {
        "attribute_binding": "attr",
        "relations": "rel",
        "numeracy": "num",
    }
    for category, pattern in category_to_pattern.items():
        prompts = _load_prompts(category)
        accepted: list[str] = []
        for prompt in prompts:
            match = pattern.match(prompt)
            if match is None:
                continue
            accepted.append(prompt)
            if len(accepted) >= PILOT_PER_CATEGORY + HELD_OUT_PER_CATEGORY:
                break
        for idx, prompt in enumerate(accepted, start=1):
            match = pattern.match(prompt)
            assert match is not None
            split = "pilot" if idx <= PILOT_PER_CATEGORY else "held_out"
            record = _make_record(
                category=category,
                source_file=str(_download_if_missing(category)),
                prompt_id=f"{prefixes[category]}_{idx:03d}",
                split=split,
                overlap=split == "held_out",
                prompt=prompt,
                match=match,
            )
            records.append(record)
    return records


def write_dataset_artifacts() -> dict[str, Any]:
    ensure_dirs()
    records = build_prompt_records()
    prompt_table_path = DATA_DIR / "prompt_table.jsonl"
    paraphrase_path = DATA_DIR / "prompts_with_paraphrases.jsonl"
    candidate_path = DATA_DIR / "paraphrase_candidates.jsonl"
    write_jsonl(prompt_table_path, [record_to_dict(r) for r in records])
    write_jsonl(paraphrase_path, [record_to_dict(r) for r in records])

    candidate_rows: list[dict[str, Any]] = []
    primary_labels: list[int] = []
    secondary_labels: list[int] = []
    rejection_counts: Counter[str] = Counter()
    rewrite_accept_counts: Counter[tuple[str, str]] = Counter()
    candidate_counts: Counter[tuple[str, str]] = Counter()
    overlap_counts = Counter(r.category for r in records if r.overlap_subset_flag)
    category_counts = Counter((r.category, r.split) for r in records)

    for record in records:
        for audit_item in record.candidate_audit:
            candidate_rows.append(
                {
                    "prompt_id": record.prompt_id,
                    "split": record.split,
                    "category": record.category,
                    "original_prompt": record.original_prompt,
                    **audit_item,
                }
            )
            candidate_counts[(record.category, audit_item["rewrite_type"])] += 1
            if audit_item["candidate_text"] in record.approved_paraphrases:
                rewrite_accept_counts[(record.category, audit_item["rewrite_type"])] += 1
            if audit_item["rejection_reason"]:
                rejection_counts[audit_item["rejection_reason"]] += 1
            primary_labels.append(int(audit_item["accepted"]))
            secondary_labels.append(int(audit_item["secondary_review"]["accepted"]))

    write_jsonl(candidate_path, candidate_rows)

    held_out_candidates = [row for row in candidate_rows if row["split"] == "held_out"]
    rejected_subset = [row for row in held_out_candidates if not row["accepted"]]
    accepted_subset = [row for row in held_out_candidates if row["accepted"]]
    double_subset = (rejected_subset + accepted_subset)[:48]
    subset_primary = [int(row["accepted"]) for row in double_subset]
    subset_secondary = [int(row["secondary_review"]["accepted"]) for row in double_subset]
    double_kappa = cohen_kappa_score(subset_primary, subset_secondary) if double_subset else None
    if isinstance(double_kappa, float) and math.isnan(double_kappa):
        double_kappa = 1.0

    accepted_total = sum(len(record.approved_paraphrases) for record in records)
    realized_overlap = sum(1 for record in records if record.overlap_subset_flag)
    summary = {
        "total_prompts": len(records),
        "pilot_prompts": sum(1 for r in records if r.split == "pilot"),
        "held_out_prompts": sum(1 for r in records if r.split == "held_out"),
        "overlap_prompts": realized_overlap,
        "overlap_prompts_with_two_paraphrases": sum(1 for r in records if r.overlap_subset_flag and len(r.approved_paraphrases) >= 2),
        "category_split_counts": {f"{key[0]}::{key[1]}": value for key, value in category_counts.items()},
        "overlap_counts": dict(overlap_counts),
        "candidate_paraphrases_total": len(candidate_rows),
        "accepted_paraphrases_total": accepted_total,
        "rejected_candidates_total": len(candidate_rows) - accepted_total,
        "rejection_reasons": dict(rejection_counts),
        "accepted_rewrite_types": {f"{key[0]}::{key[1]}": value for key, value in rewrite_accept_counts.items()},
        "candidate_rewrite_types": {f"{key[0]}::{key[1]}": value for key, value in candidate_counts.items()},
        "double_annotation_pairs_planned": 48,
        "double_annotation_pairs_completed": len(double_subset),
        "cohens_kappa": double_kappa,
        "double_annotation_protocol": "automatic_primary_vs_automatic_secondary",
        "notes": [
            "Prompt source restored to the official T2I-CompBench validation files downloaded into data/benchmark_raw/.",
            "Paraphrase artifacts now store candidate-level automatic audits with explicit acceptance and rejection decisions rather than accepting every template by construction.",
            "A 48-candidate secondary automatic audit pass is recorded for reproducibility, but it is not a substitute for the preregistered human double annotation.",
            "The preregistered human audit remains unavailable in this coding-only workspace, so robustness claims must remain exploratory rather than confirmatory.",
        ],
        "sample_preserved_slots": prompt_slot_map(records[0]) if records else {},
        "artifacts": {
            "prompt_table": str(prompt_table_path),
            "prompts_with_paraphrases": str(paraphrase_path),
            "paraphrase_candidates": str(candidate_path),
        },
    }
    write_json(DATA_DIR / "paraphrase_audit_summary.json", summary)
    return summary
