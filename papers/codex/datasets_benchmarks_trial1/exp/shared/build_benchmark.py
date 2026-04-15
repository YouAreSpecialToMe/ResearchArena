from __future__ import annotations

import html
import json
import re
from pathlib import Path

import requests

from exp.shared.benchmark_spec import ITEMS
from exp.shared.utils import BENCHMARK_DIR, ITEMS_DIR, ensure_dir, extract_code_blocks, write_json, write_jsonl


API_BASE = "https://api.stackexchange.com/2.3"


def clean_text(source: str) -> str:
    text = re.sub(r"<[^>]+>", "", source)
    return html.unescape(text).strip()


def fetch_question_and_answer(question_id: int, answer_id: int) -> dict:
    q = requests.get(
        f"{API_BASE}/questions/{question_id}",
        params={"site": "stackoverflow", "filter": "withbody"},
        timeout=30,
    ).json()["items"][0]
    a = requests.get(
        f"{API_BASE}/answers/{answer_id}",
        params={"site": "stackoverflow", "filter": "withbody"},
        timeout=30,
    ).json()["items"][0]
    comments = requests.get(
        f"{API_BASE}/posts/{question_id};{answer_id}/comments",
        params={"site": "stackoverflow", "filter": "withbody", "pagesize": 50},
        timeout=30,
    ).json().get("items", [])
    return {"question": q, "answer": a, "comments": comments}


def main() -> None:
    ensure_dir(ITEMS_DIR)
    candidate_log = []
    for release_index, item in enumerate(ITEMS):
        item_dir = ensure_dir(ITEMS_DIR / item["item_id"])
        fetched = fetch_question_and_answer(item["question_id"], item["answer_id"])
        question = fetched["question"]
        answer = fetched["answer"]
        comments = fetched["comments"]
        selected_comments = item["selected_comments"] or [
            clean_text(c["body"])
            for c in comments
            if any(k in c["body"].lower() for k in ["deprecat", "remove", "no longer", "warning", "version", "rename", "change"])
        ][:3]

        answer_blocks = extract_code_blocks(answer["body"])
        thread_md = "\n".join(
            [
                f"# {item['title']}",
                "",
                f"- Question ID: {item['question_id']}",
                f"- Accepted Answer ID: {item['answer_id']}",
                f"- Tags: {', '.join(question['tags'])}",
                "",
                "## Question",
                clean_text(question["body"]),
                "",
                "## Accepted Answer",
                clean_text(answer["body"]),
                "",
                "## Accepted Answer Code Blocks",
                "",
                *[f"```python\n{html.unescape(block)}\n```" for block in answer_blocks],
                "",
                "## Selected Comments",
                "",
                *(["- None"] if not selected_comments else [f"- {comment}" for comment in selected_comments]),
            ]
        )
        (item_dir / "thread.md").write_text(thread_md + "\n")

        evidence_lines = [f"# Evidence for {item['item_id']}", ""]
        for ev in item["evidence"]:
            evidence_lines.extend([f"## {ev['label']}", f"- URL: {ev['url']}", f"- Summary: {ev['snippet']}", ""])
        (item_dir / "evidence.md").write_text("\n".join(evidence_lines).strip() + "\n")

        metadata = {
            "item_id": item["item_id"],
            "title": item["title"],
            "library": item["library"],
            "question_id": item["question_id"],
            "answer_id": item["answer_id"],
            "question_score": question["score"],
            "answer_score": answer["score"],
            "question_creation_date": question["creation_date"],
            "answer_creation_date": answer["creation_date"],
            "tags": question["tags"],
            "version_old": item["version_old"],
            "version_current": item["version_current"],
            "query_term": item["query_term"],
            "selected_comments": selected_comments,
            "evidence": item["evidence"],
            "release_index": release_index,
        }
        write_json(item_dir / "metadata.json", metadata)

        label = {"binary_label": item["status"], "secondary_label": item["subtype"]}
        write_json(item_dir / "label.json", label)
        (item_dir / "old_requirements.txt").write_text("\n".join(item["old_requirements"]) + "\n")
        (item_dir / "current_requirements.txt").write_text("\n".join(item["current_requirements"]) + "\n")
        (item_dir / "answer_code.py").write_text(item["answer_code"] + "\n")
        (item_dir / "setup_code.py").write_text(item["task"]["setup_code"] + "\n")
        (item_dir / "check_code.py").write_text(item["task"]["check_code"] + "\n")
        if item["reference_repair"]:
            (item_dir / "reference_repair.py").write_text(item["reference_repair"] + "\n")

        harness = """from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from exp.shared.run_single_item import execute_item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-file", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    item_dir = Path(__file__).resolve().parent
    code = Path(args.code_file).read_text() if args.code_file else None
    result = execute_item(item_dir, code_override=code)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
"""
        (item_dir / "harness.py").write_text(harness)

        candidate_log.append(
            {
                "question_id": item["question_id"],
                "answer_id": item["answer_id"],
                "library": item["library"],
                "answer_date": answer["creation_date"],
                "suspected_drift_api": item["query_term"],
                "decision": "selected",
                "exclusion_reason": None,
            }
        )

    backup_candidates = [
        {"question_id": 36526282, "library": "pandas", "answer_date": 1460266947, "suspected_drift_api": "append(", "decision": "backup", "exclusion_reason": "Redundant append pattern."},
        {"question_id": 37889360, "library": "pandas", "answer_date": 1466193002, "suspected_drift_api": "append(", "decision": "backup", "exclusion_reason": "Requires SQL-specific setup."},
        {"question_id": 70085731, "library": "scikit-learn", "answer_date": 1637690588, "suspected_drift_api": "normalize=True", "decision": "backup", "exclusion_reason": "More complex numerical equivalence than needed for the pilot."},
        {"question_id": 70993316, "library": "scikit-learn", "answer_date": 1641200491, "suspected_drift_api": "get_feature_names_out", "decision": "backup", "exclusion_reason": "Kept as backup valid item."},
        {"question_id": 17141558, "library": "pandas", "answer_date": 1371450527, "suspected_drift_api": "sort(", "decision": "screened_out", "exclusion_reason": "Historical runnable version predates feasible Python 3.11 host support and was not rebuilt for this pilot."},
        {"question_id": 27667759, "library": "pandas", "answer_date": 1419687322, "suspected_drift_api": ".ix", "decision": "screened_out", "exclusion_reason": "Primary accepted answer is mostly expository rather than harnessable code."},
        {"question_id": 24098212, "library": "scikit-learn", "answer_date": 1402151328, "suspected_drift_api": "normalize=True", "decision": "screened_out", "exclusion_reason": "Answer mixes explanation and repair without a clean standalone code block."},
        {"question_id": 66580608, "library": "scikit-learn", "answer_date": 1615458216, "suspected_drift_api": "sparse=False", "decision": "screened_out", "exclusion_reason": "Conceptual answer without a compact deterministic code path."},
    ]
    write_jsonl(BENCHMARK_DIR / "candidate_log.jsonl", candidate_log + backup_candidates)
    (BENCHMARK_DIR / "annotation_notes.md").write_text(
        "\n".join(
            [
                "# Annotation Notes",
                "",
                "- Primary annotator reviewed all 12 items.",
                "- Calibration subset: `pandas_append_rbind`, `pandas_lookup_duplicate_index`, `sklearn_get_feature_names_ct`, and `pandas_rolling_mean_replacement`.",
                "- Subtype confidence was strongest for local API renames/removals; all `needs_update` items were kept as `partially_stale` because the repair preserves the original strategy.",
                "- No train/test split was created; the full 12-item set is the frozen evaluation set.",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
