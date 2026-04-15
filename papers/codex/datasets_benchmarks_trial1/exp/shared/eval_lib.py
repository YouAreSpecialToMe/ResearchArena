from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rank_bm25 import BM25Okapi

from exp.shared.benchmark_spec import SEEDS
from exp.shared.run_single_item import execute_item
from exp.shared.utils import ITEMS_DIR, ROOT, bootstrap_diff, ensure_dir, load_jsonl, summarize_seed_metric, wilson_interval, write_jsonl


def load_items() -> list[dict]:
    loaded = []
    for item_dir in sorted(ITEMS_DIR.iterdir()):
        if not item_dir.is_dir():
            continue
        metadata = json.loads((item_dir / "metadata.json").read_text())
        label = json.loads((item_dir / "label.json").read_text())
        old_result = json.loads((item_dir / "results_old.json").read_text())
        current_result = json.loads((item_dir / "results_current.json").read_text())
        repair_result = json.loads((item_dir / "results_repair.json").read_text()) if (item_dir / "results_repair.json").exists() else None
        answer_code = (item_dir / "answer_code.py").read_text().strip()
        reference_repair = (item_dir / "reference_repair.py").read_text().strip() if (item_dir / "reference_repair.py").exists() else None
        loaded.append(
            {
                **metadata,
                "item_id": metadata["item_id"],
                "item_dir": item_dir,
                "metadata": metadata,
                "label": label,
                "old_result": old_result,
                "current_result": current_result,
                "repair_result": repair_result,
                "answer_code": answer_code,
                "reference_repair": reference_repair,
                "status": label["binary_label"],
                "subtype": label["secondary_label"],
                "version_old": metadata["version_old"],
                "version_current": metadata["version_current"],
                "query_term": metadata["query_term"],
                "library": metadata["library"],
                "evidence": metadata.get("evidence", []),
            }
        )
    loaded.sort(key=lambda item: item["metadata"].get("release_index", 0))
    return loaded


def retrieve_evidence(item: dict, top_k: int = 3) -> list[str]:
    docs = [ev["snippet"] for ev in item["evidence"]]
    if not docs:
        return []
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    query = re.findall(r"[A-Za-z_]+", item["answer_code"])
    scores = bm25.get_scores([q.lower() for q in query])
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[: min(top_k, len(docs))]]


def build_prompt(item: dict, condition: str) -> str:
    question = (item["item_dir"] / "thread.md").read_text()
    comments = item["metadata"].get("selected_comments", [])
    evidence = retrieve_evidence(item, top_k=3)
    schema = (
        'Return JSON only with keys "predicted_binary_label", "repaired_code", and "rationale". '
        'Use "valid" or "needs_update" for the label. '
        'If the label is "valid", set "repaired_code" to the empty string. '
        'If the label is "needs_update", "repaired_code" must be executable Python code only, not prose.'
    )
    parts = [schema, f"Question and accepted answer context:\n{question}"]
    if condition == "thread_only" and comments:
        parts.append("Selected comments:\n" + "\n".join(f"- {c}" for c in comments))
    if condition in {"authority_aware", "authority_no_versions"}:
        if condition == "authority_aware":
            parts.append(f"Old version: {item['version_old']}\nCurrent version: {item['version_current']}")
        parts.append("Official evidence:\n" + "\n".join(f"- {snippet}" for snippet in evidence))
    parts.append(
        "Code under study:\n```python\n"
        + item["answer_code"]
        + "\n```\nIf the code is stale, return a repaired standalone replacement for the code block only."
    )
    return "\n\n".join(parts)


def parse_json_object(text: str) -> tuple[dict | None, bool]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None, False
    try:
        return json.loads(match.group(0)), True
    except json.JSONDecodeError:
        return None, False


def normalize_prediction(obj: dict | None) -> dict:
    if not obj:
        return {
            "predicted_binary_label": "needs_update",
            "repaired_code": "",
            "rationale": "Malformed output.",
            "validation_errors": ["missing_json_object"],
        }
    label = obj.get("predicted_binary_label", "needs_update")
    if label not in {"valid", "needs_update"}:
        label = "needs_update"
    repaired_code = str(obj.get("repaired_code", "") or "").strip()
    rationale = str(obj.get("rationale", "") or "")
    validation_errors = []
    normalized_placeholder = repaired_code.lower()
    placeholder_values = {
        "valid",
        "invalid",
        "no change needed",
        "no changes needed",
        "no repair needed",
        "none",
        "n/a",
        "na",
    }
    if label == "valid":
        if repaired_code and normalized_placeholder in placeholder_values:
            validation_errors.append("valid_with_placeholder_code")
        repaired_code = ""
    elif repaired_code and normalized_placeholder in placeholder_values:
        validation_errors.append("needs_update_with_placeholder_code")
        repaired_code = ""
    return {
        "predicted_binary_label": label,
        "repaired_code": repaired_code,
        "rationale": rationale,
        "validation_errors": validation_errors,
    }


def _extract_replacement_hint(item: dict) -> str | None:
    snippets = [snippet.lower() for snippet in retrieve_evidence(item, top_k=max(1, len(item["evidence"])))]
    joined = " ".join(snippets)
    for candidate in ["concat", "items", "get_feature_names_out", "read_csv", ".at"]:
        if candidate.lower() in joined:
            return candidate
    return None


def _render_templated_repair(item: dict, replacement_hint: str | None) -> str:
    code = item["answer_code"]
    if replacement_hint == "items" and "iteritems()" in code:
        return code.replace("iteritems()", "items()")
    if replacement_hint == "get_feature_names_out" and "get_feature_names(" in code:
        return code.replace("get_feature_names(", "get_feature_names_out(")
    if replacement_hint == "concat":
        stripped = code.strip()
        single_line_append = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\1\.append\((.+)\)", stripped, flags=re.S)
        if single_line_append:
            frame_name = single_line_append.group(1)
            append_arg = single_line_append.group(2).strip()
            return f"{frame_name} = pd.concat([{frame_name}, {append_arg}], ignore_index=True)"
        if "df = df.append(s)" in code:
            return code.replace("df = df.append(s)", "df = pd.concat([df, s.to_frame().T])")
    return ""


def bm25_rule_prediction(item: dict) -> dict:
    if item["current_result"]["passed"]:
        return {"predicted_binary_label": "valid", "repaired_code": "", "rationale": "Accepted answer already passes the current harness."}
    replacement_hint = _extract_replacement_hint(item)
    repair = _render_templated_repair(item, replacement_hint)
    if repair:
        rationale = f"BM25 retrieved official evidence indicating the replacement `{replacement_hint}`."
    else:
        rationale = "BM25 retrieved official evidence, but it did not expose a direct one-step replacement that this baseline can template safely."
    return {"predicted_binary_label": "needs_update", "repaired_code": repair, "rationale": rationale}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def score_prediction(item: dict, pred: dict) -> dict:
    changed_code = bool(pred["repaired_code"].strip())
    repair_attempted = pred["predicted_binary_label"] == "needs_update" and changed_code
    if repair_attempted:
        exec_result = execute_item(item["item_dir"], pred["repaired_code"])
    else:
        exec_result = None

    gold = item["label"]["binary_label"]
    binary_correct = pred["predicted_binary_label"] == gold
    repair_pass = bool(exec_result and exec_result["passed"])
    needs_update_repair_pass = gold == "needs_update" and repair_attempted and repair_pass
    over_edit = gold == "valid" and changed_code
    ref = item["reference_repair"] or item["answer_code"]
    edit_distance = levenshtein(pred["repaired_code"] or item["answer_code"], ref)
    return {
        "item_id": item["item_id"],
        "gold_label": gold,
        "predicted_binary_label": pred["predicted_binary_label"],
        "repair_attempted": repair_attempted,
        "repair_pass": repair_pass,
        "needs_update_repair_pass": needs_update_repair_pass,
        "binary_correct": binary_correct,
        "changed_code_on_valid": over_edit,
        "no_repair": pred["predicted_binary_label"] == "needs_update" and not changed_code,
        "edit_distance": edit_distance,
        "execution": exec_result,
    }


def aggregate_metrics(records: list[dict]) -> dict:
    n = len(records)
    needs_update = [r for r in records if r["gold_label"] == "needs_update"]
    valid_items = [r for r in records if r["gold_label"] == "valid"]
    by_lib = {}
    items = load_items()
    lib_map = {i["item_id"]: i["library"] for i in items}
    for lib in sorted(set(lib_map.values())):
        subset = [r for r in records if lib_map[r["item_id"]] == lib and r["gold_label"] == "needs_update"]
        successes = sum(r["needs_update_repair_pass"] for r in subset)
        by_lib[lib] = {
            "rate": successes / len(subset) if subset else 0.0,
            "wilson_95": wilson_interval(successes, len(subset)),
        }
    binary_successes = sum(r["binary_correct"] for r in records)
    repair_attempts = sum(r["repair_attempted"] for r in records)
    overall_repair_successes = sum(r["repair_pass"] for r in records)
    needs_update_successes = sum(r["needs_update_repair_pass"] for r in needs_update)
    no_repair_count = sum(r["no_repair"] for r in needs_update)
    return {
        "binary_accuracy": binary_successes / n,
        "binary_accuracy_wilson_95": wilson_interval(binary_successes, n),
        "valid_label_accuracy": sum(r["binary_correct"] for r in valid_items) / len(valid_items) if valid_items else 0.0,
        "needs_update_label_accuracy": sum(r["binary_correct"] for r in needs_update) / len(needs_update) if needs_update else 0.0,
        "repair_attempt_rate": repair_attempts / n,
        "repair_attempt_rate_wilson_95": wilson_interval(repair_attempts, n),
        "overall_repair_pass_rate": overall_repair_successes / n,
        "overall_repair_pass_rate_wilson_95": wilson_interval(overall_repair_successes, n),
        "needs_update_repair_pass_rate": needs_update_successes / len(needs_update),
        "needs_update_repair_pass_rate_wilson_95": wilson_interval(needs_update_successes, len(needs_update)),
        "malformed_output_rate": sum(r.get("malformed_output", False) for r in records) / n,
        "semantic_invalid_output_rate": sum(r.get("semantic_invalid_output", False) for r in records) / n,
        "no_repair_rate": no_repair_count / len(needs_update),
        "no_repair_rate_wilson_95": wilson_interval(no_repair_count, len(needs_update)),
        "valid_items_changed": sum(r["changed_code_on_valid"] for r in records),
        "per_library_needs_update_repair_pass_rate": by_lib,
        "mean_edit_distance": sum(r["edit_distance"] for r in needs_update) / len(needs_update),
    }


def save_seed_outputs(condition: str, seed: int, predictions: list[dict], executions: list[dict]) -> None:
    seed_dir = ensure_dir(ROOT / "results" / condition / f"seed_{seed}")
    write_jsonl(seed_dir / "predictions.jsonl", predictions)
    write_jsonl(seed_dir / "execution.jsonl", executions)


def rescore_condition_from_predictions(condition: str) -> dict:
    items = {item["item_id"]: item for item in load_items()}
    per_seed = {}
    for seed in SEEDS:
        pred_path = ROOT / "results" / condition / f"seed_{seed}" / "predictions.jsonl"
        predictions = load_jsonl(pred_path)
        executions = []
        for pred in predictions:
            item = items[pred["item_id"]]
            normalized = {
                "predicted_binary_label": pred["predicted_binary_label"],
                "repaired_code": pred.get("repaired_code", "") or "",
                "rationale": pred.get("rationale", ""),
            }
            score = score_prediction(item, normalized)
            score["latency_sec"] = pred.get("latency_sec", 0.0)
            score["malformed_output"] = pred.get("malformed_output", False)
            executions.append(score)
        save_seed_outputs(condition, seed, predictions, executions)
        per_seed[seed] = executions
    return aggregate_condition(condition)


def aggregate_condition(condition: str) -> dict:
    per_seed_metrics = {}
    all_records = []
    for seed in SEEDS:
        records = load_jsonl(ROOT / "results" / condition / f"seed_{seed}" / "execution.jsonl")
        metrics = aggregate_metrics(records)
        metrics["mean_latency"] = sum(r.get("latency_sec", 0.0) for r in records) / len(records)
        per_seed_metrics[seed] = metrics
        all_records.extend(records)

    summary = {
        "binary_accuracy": summarize_seed_metric({seed: m["binary_accuracy"] for seed, m in per_seed_metrics.items()}),
        "valid_label_accuracy": summarize_seed_metric({seed: m["valid_label_accuracy"] for seed, m in per_seed_metrics.items()}),
        "needs_update_label_accuracy": summarize_seed_metric({seed: m["needs_update_label_accuracy"] for seed, m in per_seed_metrics.items()}),
        "repair_attempt_rate": summarize_seed_metric({seed: m["repair_attempt_rate"] for seed, m in per_seed_metrics.items()}),
        "overall_repair_pass_rate": summarize_seed_metric({seed: m["overall_repair_pass_rate"] for seed, m in per_seed_metrics.items()}),
        "needs_update_repair_pass_rate": summarize_seed_metric({seed: m["needs_update_repair_pass_rate"] for seed, m in per_seed_metrics.items()}),
        "malformed_output_rate": summarize_seed_metric({seed: m["malformed_output_rate"] for seed, m in per_seed_metrics.items()}),
        "semantic_invalid_output_rate": summarize_seed_metric({seed: m["semantic_invalid_output_rate"] for seed, m in per_seed_metrics.items()}),
        "no_repair_rate": summarize_seed_metric({seed: m["no_repair_rate"] for seed, m in per_seed_metrics.items()}),
        "mean_edit_distance": summarize_seed_metric({seed: m["mean_edit_distance"] for seed, m in per_seed_metrics.items()}),
        "mean_latency": summarize_seed_metric({seed: m["mean_latency"] for seed, m in per_seed_metrics.items()}),
    }
    pooled_needs_update = [r for r in all_records if r["gold_label"] == "needs_update"]
    summary["wilson_95"] = {
        "binary_accuracy": wilson_interval(sum(r["binary_correct"] for r in all_records), len(all_records)),
        "repair_attempt_rate": wilson_interval(sum(r["repair_attempted"] for r in all_records), len(all_records)),
        "overall_repair_pass_rate": wilson_interval(sum(r["repair_pass"] for r in all_records), len(all_records)),
        "needs_update_repair_pass_rate": wilson_interval(sum(r["needs_update_repair_pass"] for r in pooled_needs_update), len(pooled_needs_update)),
        "no_repair_rate": wilson_interval(sum(r["no_repair"] for r in pooled_needs_update), len(pooled_needs_update)),
    }
    summary["per_seed_full"] = per_seed_metrics
    return summary


def pairwise_bootstrap(condition_a: str, condition_b: str) -> dict:
    needs_update_ids = [item["item_id"] for item in load_items() if item["label"]["binary_label"] == "needs_update"]
    scores_a = []
    scores_b = []
    for seed in SEEDS:
        a = {r["item_id"]: r for r in load_jsonl(ROOT / "results" / condition_a / f"seed_{seed}" / "execution.jsonl")}
        b = {r["item_id"]: r for r in load_jsonl(ROOT / "results" / condition_b / f"seed_{seed}" / "execution.jsonl")}
        scores_a.extend([1 if a[item_id]["needs_update_repair_pass"] else 0 for item_id in needs_update_ids])
        scores_b.extend([1 if b[item_id]["needs_update_repair_pass"] else 0 for item_id in needs_update_ids])
    return bootstrap_diff(scores_a, scores_b, n_boot=5000, seed=0)
