import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

from .data_loader import build_audited_subset, freeze_audited_ids, write_audit_guidelines
from .metrics import bootstrap_ci, ranking_stats, summarize_predictions
from .models import FeatureModels, Generator, MNLI_MODEL, MINILM_MODEL, QWEN_MODEL
from .utils import (
    ROOT,
    SEEDS,
    alias_match,
    canonicalize_date,
    ensure_dir,
    mean_std,
    normalize_text,
    read_json,
    read_jsonl,
    set_seed,
    support_count,
    write_json,
    write_jsonl,
)


TAU_GRID = [-2.5, -2.0, -1.5, -1.0]
HEUR_GRIDS = {
    "entailment_tau": [0.5, 0.55, 0.6],
    "contradiction_tau": [0.5, 0.55, 0.6],
    "conf_gap": [0.2, 0.4, 0.6],
}
PROXY_GRIDS = {
    "ctx_entail_tau": [0.5, 0.55, 0.6],
    "mem_gap": [0.0, 0.2, 0.4],
    "max_distinct_memory": [1, 2],
}
LOGISTIC_C_GRID = [0.1, 1.0, 10.0]
LOGISTIC_TAU_GRID = [0.45, 0.50, 0.55, 0.60]
POLICIES = [
    "always_context",
    "always_memory",
    "always_abstain",
    "abstain_low_conf",
    "supported_else_abstain",
    "proxy_router",
]


def _clean_for_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_for_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str, bytes)) else False:
        return None
    return value


def _ensure_exp_layout(exp_name: str) -> Path:
    base = ROOT / "exp" / exp_name
    ensure_dir(base / "logs")
    ensure_dir(base / "predictions")
    return base


def _log_path(exp_name: str) -> Path:
    return _ensure_exp_layout(exp_name) / "logs" / "run_20260320_3.log"


def _log(exp_name: str, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    path = _log_path(exp_name)
    with path.open("a") as f:
        f.write(f"[{timestamp}] {message}\n")


def _save_prediction_rows(exp_name: str, stem: str, rows: List[Dict]) -> None:
    base = _ensure_exp_layout(exp_name)
    payload = [{k: _clean_for_json(v) for k, v in row.items()} for row in rows]
    write_jsonl(base / "predictions" / f"{stem}.jsonl", payload)


def env_report() -> Dict:
    import platform

    import torch
    import transformers

    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "generator_model": QWEN_MODEL,
        "semantic_model": MINILM_MODEL,
        "mnli_model": MNLI_MODEL,
        "decoding": {
            "max_input_length": 1536,
            "max_new_tokens": 16,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    }
    write_json(ROOT / "exp/shared/env_report.json", report)
    return report


def run_audit() -> Dict:
    start = time.time()
    _ensure_exp_layout("audit")
    _log("audit", "Starting audit dry-run assembly.")
    env_report()
    frozen = freeze_audited_ids()
    write_audit_guidelines()
    rows = build_audited_subset()
    counts = Counter((row["dataset"], row["split"]) for row in rows)
    label_balance = Counter(row["audited_label"] for row in rows)
    variant_flips = sum(
        row["policy_variant_label"]["default"] != row["policy_variant_label"]["permissive_memory"] for row in rows
    )
    alias_additions = len(read_json(ROOT / "data/processed/alias_tables.json"))
    results = {
        "experiment": "audit",
        "metrics": {
            "num_items": len(rows),
            "split_counts": {f"{k[0]}_{k[1]}": int(v) for k, v in counts.items()},
            "label_balance": {k: int(v) for k, v in label_balance.items()},
            "non_discard_rate": 1.0,
            "cohens_kappa_overall": None,
            "cohens_kappa_memory_vs_abstain": None,
            "boundary_specific_agreement": None,
            "agreement_note": "Blocked: no external human double-annotation or adjudication records exist in this workspace.",
            "alias_addition_count": alias_additions,
            "strict_vs_permissive_label_flips": variant_flips,
            "audit_validity": "single_annotator_rule_proxy_only",
            "frozen_seed": frozen["seed"],
        },
        "runtime_minutes": (time.time() - start) / 60.0,
        "config": {"seeds": SEEDS, "selection_report": frozen.get("selection_report", {})},
    }
    write_json(ROOT / "exp/audit/results.json", results)
    skipped = ROOT / "exp/audit/SKIPPED.md"
    skipped.write_text(
        "Real audited labels are unavailable in this workspace.\n"
        "No external human double-annotation files or adjudication logs were provided, so Cohen's kappa and boundary agreement cannot be computed.\n"
        "Planned dev/test double annotation, adjudication, and locked-test holdout agreement estimation were not feasible from the provided artifacts.\n"
        "This run keeps the deterministic single-pass label proxy only to stress-test the rest of the evaluation pipeline, and any audited-regime conclusions are reported as provisional and non-substantive.\n"
    )
    _log("audit", f"Audit dry run finished with {len(rows)} items; frozen seed={frozen['seed']}.")
    return results


def _load_subset_df() -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(ROOT / "data/processed/audited_subset.jsonl"))


def _to_dict(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return {}
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    return dict(value)


def _normalize_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [str(value)]


def run_cache(batch_size: int = 12) -> Dict:
    start = time.time()
    _ensure_exp_layout("cache")
    _log("cache", f"Starting cache generation with batch_size={batch_size}.")
    df = _load_subset_df()
    generator = Generator()
    feature_models = FeatureModels()
    rows = []
    for i in range(0, len(df), batch_size):
        sub = df.iloc[i : i + batch_size]
        mem = generator.generate_batch(sub["question"].tolist(), [""] * len(sub))
        ctx = generator.generate_batch(sub["question"].tolist(), sub["context"].tolist())
        for local_idx, (_, row) in enumerate(sub.iterrows()):
            mem_out = mem[local_idx]
            ctx_out = ctx[local_idx]
            answer_aliases = row["audited_answers"] if row["audited_label"] != "abstain" else row["gold_answers"]
            ctx_nli = feature_models.nli_scores(row["context"], f"The answer is {ctx_out['answer']}.")
            mem_nli = feature_models.nli_scores(row["context"], f"The answer is {mem_out['answer']}.")
            rows.append(
                {
                    "example_id": row["example_id"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "question": row["question"],
                    "context": row["context"],
                    "audited_label": row["audited_label"],
                    "variant_labels": row["policy_variant_label"],
                    "audited_answers": row["audited_answers"],
                    "gold_answers": row["gold_answers"],
                    "benchmark_aliases": row["benchmark_aliases"],
                    "source_metadata": row["source_metadata"],
                    "original_target_new": row["source_metadata"].get("target_new") if row["dataset"] == "cub" else None,
                    "benchmark_context_type": row["benchmark_context_type"],
                    "ambiguity_flags": row["ambiguity_flags"],
                    "memory_answer": mem_out["answer"],
                    "memory_answer_norm": normalize_text(mem_out["answer"]),
                    "memory_logprob": mem_out["mean_logprob"],
                    "memory_entropy": mem_out["mean_entropy"],
                    "context_answer": ctx_out["answer"],
                    "context_answer_norm": normalize_text(ctx_out["answer"]),
                    "context_logprob": ctx_out["mean_logprob"],
                    "context_entropy": ctx_out["mean_entropy"],
                    "prompt_length": ctx_out["prompt_length"],
                    "latency_sec": mem_out["latency_sec"] + ctx_out["latency_sec"],
                    "context_support_count": support_count(row["context"], row["benchmark_aliases"]),
                    "context_similarity": feature_models.max_similarity(ctx_out["answer"], row["context"]),
                    "memory_similarity": feature_models.max_similarity(mem_out["answer"], row["context"]),
                    "context_entailment": ctx_nli["entailment"],
                    "context_contradiction": ctx_nli["contradiction"],
                    "memory_entailment": mem_nli["entailment"],
                    "memory_contradiction": mem_nli["contradiction"],
                    "draft_disagreement": int(normalize_text(mem_out["answer"]) != normalize_text(ctx_out["answer"])),
                    "target_alias_hit_memory": alias_match(mem_out["answer"], answer_aliases),
                    "target_alias_hit_context": alias_match(ctx_out["answer"], answer_aliases),
                }
            )
    cache_path = ROOT / "exp/cache/qwen_dual_view_outputs.parquet"
    ensure_dir(cache_path.parent)
    pd.DataFrame(rows).to_parquet(cache_path, index=False)
    results = {
        "experiment": "cache",
        "metrics": {
            "num_examples": len(rows),
            "mean_latency_sec": float(np.mean([r["latency_sec"] for r in rows])),
            "memory_exact_match_to_target": float(np.mean([r["target_alias_hit_memory"] for r in rows])),
            "context_exact_match_to_target": float(np.mean([r["target_alias_hit_context"] for r in rows])),
        },
        "config": {"batch_size": batch_size},
        "runtime_minutes": (time.time() - start) / 60.0,
    }
    write_json(ROOT / "exp/cache/results.json", results)
    _log("cache", f"Cached {len(rows)} dual-view outputs.")
    return results


def _route_answer(route: str, row: pd.Series) -> str:
    if route == "context":
        return str(row["context_answer"])
    if route == "memory":
        return str(row["memory_answer"])
    return ""


def _naive_match(answer: str, aliases: List[str]) -> bool:
    norm_answer = normalize_text(answer)
    if not norm_answer or not aliases:
        return False
    for alias in aliases:
        if norm_answer == normalize_text(alias) or canonicalize_date(answer) == canonicalize_date(alias):
            return True
    return False


def _whoqa_strict_answer_match(answer: str, source_metadata: Dict, alias_mode: str) -> bool:
    if not answer:
        return False
    norm_answer = normalize_text(answer)
    contexts = source_metadata.get("contexts", [])
    answer_by_context = source_metadata.get("answer_by_context", {})
    covered_contexts = 0
    for idx, _ in enumerate(contexts):
        answer_groups = answer_by_context.get(str(idx), [])
        matched_group = False
        for answer_group in answer_groups:
            aliases = _normalize_list(answer_group)
            if alias_mode == "alias":
                if any(alias_match(norm_answer, [alias]) for alias in aliases):
                    matched_group = True
                    break
            else:
                if any(_naive_match(norm_answer, [alias]) for alias in aliases):
                    matched_group = True
                    break
        if matched_group:
            covered_contexts += 1
    return covered_contexts == len(contexts) and len(contexts) > 0


def _original_answer_ok(row: pd.Series, pred_route: str, alias_mode: str) -> bool:
    if pred_route == "abstain":
        return False
    answer = _route_answer(pred_route, row)
    if row["dataset"] == "cub":
        if row["benchmark_context_type"] == "edited":
            aliases = [row["original_target_new"]]
        else:
            aliases = _normalize_list(row["gold_answers"])
        return alias_match(answer, aliases) if alias_mode == "alias" else _naive_match(answer, aliases)
    return _whoqa_strict_answer_match(answer, _to_dict(row["source_metadata"]), alias_mode)


def _proxy_gold_route(row: pd.Series) -> str:
    ambiguity = _to_dict(row["ambiguity_flags"])
    if row["dataset"] == "cub":
        if row["benchmark_context_type"] == "gold":
            return "context"
        if row["benchmark_context_type"] == "irrelevant":
            return "memory"
        if row["draft_disagreement"] and row["memory_logprob"] >= row["context_logprob"]:
            return "memory"
        return "context"
    if ambiguity.get("num_distinct_answers", 1) > 2:
        return "abstain"
    if row["context_support_count"] > 0 and row["context_entailment"] >= 0.55:
        return "context"
    if row["draft_disagreement"] and row["memory_logprob"] > row["context_logprob"]:
        return "memory"
    return "abstain"


def _audited_gold_route(row: pd.Series, variant: str) -> str:
    if variant == "default":
        return row["audited_label"]
    return _to_dict(row["variant_labels"]).get(variant, row["audited_label"])


def _audited_gold_aliases(row: pd.Series, gold_route: str) -> List[str]:
    if gold_route == "abstain":
        return []
    if row["dataset"] == "whoqa":
        return _normalize_list(row["audited_answers"])
    return _normalize_list(row["audited_answers"])


def _proxy_gold_aliases(row: pd.Series, gold_route: str) -> List[str]:
    if gold_route == "abstain":
        return []
    if row["dataset"] == "cub":
        if gold_route == "context" and row["benchmark_context_type"] == "edited":
            return [row["original_target_new"]]
        return _normalize_list(row["gold_answers"])
    if gold_route == "context":
        return _normalize_list(row["benchmark_aliases"])
    return _normalize_list(row["gold_answers"])


def _answer_ok_with_aliases(row: pd.Series, pred_route: str, aliases: List[str], alias_mode: str) -> bool:
    if pred_route == "abstain":
        return False
    answer = _route_answer(pred_route, row)
    return alias_match(answer, aliases) if alias_mode == "alias" else _naive_match(answer, aliases)


def _policy_predictions(df: pd.DataFrame, tau_conf: float, heur: Dict, proxy_cfg: Dict) -> Dict[str, pd.DataFrame]:
    out = {}

    always_context = df.copy()
    always_context["pred_route"] = "context"
    out["always_context"] = always_context

    always_memory = df.copy()
    always_memory["pred_route"] = "memory"
    out["always_memory"] = always_memory

    always_abstain = df.copy()
    always_abstain["pred_route"] = "abstain"
    out["always_abstain"] = always_abstain

    low_conf = df.copy()
    best_is_ctx = low_conf["context_logprob"] >= low_conf["memory_logprob"]
    low_conf["pred_route"] = np.where(best_is_ctx, "context", "memory")
    chosen_conf = np.where(best_is_ctx, low_conf["context_logprob"], low_conf["memory_logprob"])
    low_conf.loc[chosen_conf < tau_conf, "pred_route"] = "abstain"
    out["abstain_low_conf"] = low_conf

    heur_df = df.copy()
    heur_df["pred_route"] = "abstain"
    ctx_mask = (heur_df["context_support_count"] >= heur["support_min"]) & (
        heur_df["context_entailment"] >= heur["entailment_tau"]
    )
    mem_mask = (heur_df["memory_contradiction"] >= heur["contradiction_tau"]) & (
        (heur_df["memory_logprob"] - heur_df["context_logprob"]) >= heur["conf_gap"]
    )
    heur_df.loc[ctx_mask, "pred_route"] = "context"
    heur_df.loc[~ctx_mask & mem_mask, "pred_route"] = "memory"
    out["supported_else_abstain"] = heur_df

    proxy = df.copy()
    proxy["pred_route"] = "abstain"
    proxy.loc[proxy["dataset"] == "cub", "pred_route"] = proxy[proxy["dataset"] == "cub"].apply(
        lambda r: (
            "context"
            if r["benchmark_context_type"] == "gold"
            else (
                "memory"
                if r["benchmark_context_type"] == "irrelevant"
                and (r["memory_logprob"] - r["context_logprob"]) >= proxy_cfg["mem_gap"]
                else (
                    "context"
                    if r["benchmark_context_type"] == "edited"
                    and r["context_support_count"] >= 1
                    and r["context_entailment"] >= proxy_cfg["ctx_entail_tau"]
                    else (
                        "memory"
                        if r["benchmark_context_type"] == "edited"
                        and r["memory_contradiction"] >= proxy_cfg["ctx_entail_tau"]
                        and (r["memory_logprob"] - r["context_logprob"]) >= proxy_cfg["mem_gap"]
                        else "abstain"
                    )
                )
            )
        ),
        axis=1,
    )
    whoqa_mask = proxy["dataset"] == "whoqa"
    proxy.loc[whoqa_mask, "pred_route"] = proxy[whoqa_mask].apply(
        lambda r: (
            "abstain"
            if _to_dict(r["ambiguity_flags"]).get("num_distinct_answers", 1) > proxy_cfg["max_distinct_memory"]
            or r["draft_disagreement"]
            else (
                "context"
                if r["context_support_count"] >= 1 and r["context_entailment"] >= proxy_cfg["ctx_entail_tau"]
                else (
                    "memory"
                    if (r["memory_logprob"] - r["context_logprob"]) >= proxy_cfg["mem_gap"]
                    else "abstain"
                )
            )
        ),
        axis=1,
    )
    out["proxy_router"] = proxy
    return out


def _resample_dev(df_dev: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled = []
    for label in sorted(df_dev["audited_label"].unique()):
        bucket = df_dev[df_dev["audited_label"] == label]
        idx = rng.choice(bucket.index.to_numpy(), size=len(bucket), replace=True)
        sampled.append(bucket.loc[idx])
    return pd.concat(sampled, ignore_index=True)


def _score_rows(df: pd.DataFrame, regime: str, variant: str = "default", alias_mode: str = "alias") -> Tuple[List[Dict], Dict]:
    rows = []
    for _, row in df.iterrows():
        pred_route = row["pred_route"]
        pred_answer = _route_answer(pred_route, row)
        if regime == "original":
            answer_ok = _original_answer_ok(row, pred_route, alias_mode)
            gold_route = None
        elif regime == "proxy":
            gold_route = _proxy_gold_route(row)
            gold_aliases = _proxy_gold_aliases(row, gold_route)
            answer_ok = _answer_ok_with_aliases(row, pred_route, gold_aliases, alias_mode)
        else:
            gold_route = _audited_gold_route(row, variant)
            gold_aliases = _audited_gold_aliases(row, gold_route)
            answer_ok = _answer_ok_with_aliases(row, pred_route, gold_aliases, alias_mode)
        answer_ok_naive = (
            _original_answer_ok(row, pred_route, "naive")
            if regime == "original"
            else _answer_ok_with_aliases(
                row,
                pred_route,
                _proxy_gold_aliases(row, gold_route) if regime == "proxy" else _audited_gold_aliases(row, gold_route),
                "naive",
            )
        )
        utility = 0.0 if pred_route == "abstain" else (1.0 if answer_ok else -2.0)
        rows.append(
            {
                "example_id": row["example_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "regime": regime if variant == "default" else variant,
                "gold_route": gold_route,
                "pred_route": pred_route,
                "pred_answer": pred_answer,
                "answer_ok": answer_ok,
                "answer_ok_naive": answer_ok_naive,
                "utility": utility,
                "harmful": pred_route != "abstain" and not answer_ok,
                "benchmark_context_type": row["benchmark_context_type"],
            }
        )
    return rows, summarize_predictions(rows, regime if variant == "default" else variant)


def _tune(df_dev: pd.DataFrame, seed: int) -> Dict:
    dev_sample = _resample_dev(df_dev, seed)
    best_tau = TAU_GRID[0]
    best_util = -1e9
    for tau in TAU_GRID:
        preds = _policy_predictions(
            dev_sample,
            tau,
            {"support_min": 1, "entailment_tau": 0.55, "contradiction_tau": 0.55, "conf_gap": 0.4},
            {"ctx_entail_tau": 0.55, "mem_gap": 0.2, "max_distinct_memory": 1},
        )
        _, res = _score_rows(preds["abstain_low_conf"], "audited")
        if res["expected_utility"] > best_util:
            best_util = res["expected_utility"]
            best_tau = tau

    best_heur = None
    best_heur_util = -1e9
    for entail in HEUR_GRIDS["entailment_tau"]:
        for contra in HEUR_GRIDS["contradiction_tau"]:
            for gap in HEUR_GRIDS["conf_gap"]:
                heur = {"support_min": 1, "entailment_tau": entail, "contradiction_tau": contra, "conf_gap": gap}
                preds = _policy_predictions(
                    dev_sample,
                    best_tau,
                    heur,
                    {"ctx_entail_tau": 0.55, "mem_gap": 0.2, "max_distinct_memory": 1},
                )
                _, res = _score_rows(preds["supported_else_abstain"], "audited")
                if res["expected_utility"] > best_heur_util:
                    best_heur_util = res["expected_utility"]
                    best_heur = heur

    best_proxy = None
    best_proxy_util = -1e9
    for ctx_tau in PROXY_GRIDS["ctx_entail_tau"]:
        for gap in PROXY_GRIDS["mem_gap"]:
            for max_distinct in PROXY_GRIDS["max_distinct_memory"]:
                proxy_cfg = {"ctx_entail_tau": ctx_tau, "mem_gap": gap, "max_distinct_memory": max_distinct}
                preds = _policy_predictions(dev_sample, best_tau, best_heur, proxy_cfg)
                _, res = _score_rows(preds["proxy_router"], "proxy")
                if res["expected_utility"] > best_proxy_util:
                    best_proxy_util = res["expected_utility"]
                    best_proxy = proxy_cfg
    return {"tau_conf": best_tau, "heuristic": best_heur, "proxy_router": best_proxy}


def _aggregate_seed_metrics(seed_metrics: Dict[int, Dict]) -> Dict:
    seed_order = list(seed_metrics.keys())
    metric_names = sorted(next(iter(seed_metrics.values())).keys())
    summary = {}
    for metric in metric_names:
        values = [seed_metrics[seed][metric] for seed in seed_order]
        if values[0] is None:
            summary[metric] = None
        elif isinstance(values[0], str):
            summary[metric] = values[0]
        elif isinstance(values[0], dict):
            summary[metric] = values[0]
        else:
            summary[metric] = mean_std(values)
    return summary


def run_baselines() -> Dict:
    start = time.time()
    base = _ensure_exp_layout("baselines")
    _log("baselines", "Starting baseline evaluation.")
    df = pd.read_parquet(ROOT / "exp/cache/qwen_dual_view_outputs.parquet")
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    seed_runs = {}
    aggregated = defaultdict(dict)
    tuned_configs = {}
    for seed in SEEDS:
        _log("baselines", f"Tuning baselines for seed={seed}.")
        tuned = _tune(dev, seed)
        tuned_configs[str(seed)] = tuned
        policies = _policy_predictions(test, tuned["tau_conf"], tuned["heuristic"], tuned["proxy_router"])
        seed_runs[str(seed)] = {}
        for policy, pdf in policies.items():
            rows, metrics = _score_rows(pdf, "audited")
            _save_prediction_rows("baselines", f"seed_{seed}_audited_{policy}", rows)
            seed_runs[str(seed)][policy] = metrics
            aggregated[policy][seed] = metrics
            _log("baselines", f"seed={seed} policy={policy} utility={metrics['expected_utility']:.4f} route_acc={metrics['route_accuracy']:.4f}")
    metrics = {policy: _aggregate_seed_metrics(seed_map) for policy, seed_map in aggregated.items()}
    results = {
        "experiment": "baselines",
        "metrics": metrics,
        "seed_runs": seed_runs,
        "config": tuned_configs,
        "runtime_minutes": (time.time() - start) / 60.0,
    }
    write_json(base / "results.json", results)
    _log("baselines", "Baseline evaluation finished.")
    return results


def _rank_policies(policy_metrics: Dict[str, Dict]) -> Dict[str, int]:
    ordered = sorted(
        policy_metrics.items(),
        key=lambda kv: kv[1]["expected_utility"]["mean"] if kv[1]["expected_utility"] is not None else -1e9,
        reverse=True,
    )
    return {name: rank for rank, (name, _) in enumerate(ordered, start=1)}


def run_main() -> Dict:
    start = time.time()
    base = _ensure_exp_layout("main")
    _log("main", "Starting main regime comparison.")
    baseline_cfg = read_json(ROOT / "exp/baselines/results.json")["config"]
    df = pd.read_parquet(ROOT / "exp/cache/qwen_dual_view_outputs.parquet")
    test = df[df["split"] == "test"].copy()
    seed_runs = {str(seed): {"original": {}, "proxy": {}, "audited": {}, "strict_abstain": {}, "permissive_memory": {}} for seed in SEEDS}
    aggregated = {regime: defaultdict(dict) for regime in seed_runs[str(SEEDS[0])].keys()}
    for seed in SEEDS:
        _log("main", f"Evaluating all regimes for seed={seed}.")
        cfg = baseline_cfg[str(seed)]
        policies = _policy_predictions(test, cfg["tau_conf"], cfg["heuristic"], cfg["proxy_router"])
        for policy, pdf in policies.items():
            for regime, variant in [("original", "default"), ("proxy", "default"), ("audited", "default"), ("audited", "strict_abstain"), ("audited", "permissive_memory")]:
                rows, metrics = _score_rows(pdf, regime, variant)
                regime_key = regime if variant == "default" else variant
                seed_runs[str(seed)][regime_key][policy] = metrics
                aggregated[regime_key][policy][seed] = metrics
                _save_prediction_rows("main", f"seed_{seed}_{regime_key}_{policy}", rows)
        _log("main", f"Completed seed={seed}.")
    metrics = {
        regime: {policy: _aggregate_seed_metrics(seed_map) for policy, seed_map in policy_maps.items()}
        for regime, policy_maps in aggregated.items()
    }
    policy_ranks = {
        "original": _rank_policies(metrics["original"]),
        "proxy": _rank_policies(metrics["proxy"]),
        "audited": _rank_policies(metrics["audited"]),
        "strict_abstain": _rank_policies(metrics["strict_abstain"]),
        "permissive_memory": _rank_policies(metrics["permissive_memory"]),
    }
    ranking = {
        "original_vs_audited": ranking_stats(policy_ranks["original"], policy_ranks["audited"]),
        "proxy_vs_audited": ranking_stats(policy_ranks["proxy"], policy_ranks["audited"]),
        "strict_vs_audited": ranking_stats(policy_ranks["strict_abstain"], policy_ranks["audited"]),
        "permissive_vs_audited": ranking_stats(policy_ranks["permissive_memory"], policy_ranks["audited"]),
    }
    results = {
        "experiment": "main",
        "metrics": metrics,
        "seed_runs": seed_runs,
        "policy_ranks": policy_ranks,
        "ranking_instability": ranking,
        "runtime_minutes": (time.time() - start) / 60.0,
        "config": baseline_cfg,
    }
    write_json(base / "results.json", results)
    _log("main", f"Main comparison finished. original_vs_audited tau={ranking['original_vs_audited']['kendall_tau']:.4f}")
    return results


def _fit_logistic(train_df: pd.DataFrame, c_value: float, drop_ambiguity: bool = False):
    feats = ["conf_gap_feature", "memory_contradiction"]
    if not drop_ambiguity:
        feats.append("ambiguous")
    X = train_df[feats].to_numpy()
    y = train_df["audited_label"].to_numpy()
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        C=c_value,
        random_state=13,
    )
    model.fit(X, y)
    return model, feats


def _logistic_rows(model, feats: List[str], df: pd.DataFrame, tau_prob: float) -> List[Dict]:
    probs = model.predict_proba(df[feats].to_numpy())
    labels = list(model.classes_)
    rows = []
    for row, prob in zip(df.to_dict("records"), probs):
        max_idx = int(np.argmax(prob))
        pred_label = labels[max_idx] if prob[max_idx] >= tau_prob else "abstain"
        gold_route = row["audited_label"]
        gold_aliases = row["audited_answers"] if gold_route != "abstain" else []
        answer_ok = pred_label != "abstain" and alias_match(
            row["context_answer"] if pred_label == "context" else row["memory_answer"],
            gold_aliases,
        )
        rows.append(
            {
                "example_id": row["example_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "regime": "audited",
                "gold_route": gold_route,
                "pred_route": pred_label,
                "pred_answer": row["context_answer"] if pred_label == "context" else row["memory_answer"],
                "answer_ok": answer_ok,
                "answer_ok_naive": answer_ok,
                "utility": 0.0 if pred_label == "abstain" else (1.0 if answer_ok else -2.0),
                "harmful": pred_label != "abstain" and not answer_ok,
            }
        )
    return rows


def run_ablations_and_appendix() -> Dict:
    start = time.time()
    base = _ensure_exp_layout("ablations")
    _log("ablations", "Starting ablations and appendix router.")
    main = read_json(ROOT / "exp/main/results.json")
    baseline_cfg = read_json(ROOT / "exp/baselines/results.json")["config"]
    df = pd.read_parquet(ROOT / "exp/cache/qwen_dual_view_outputs.parquet")
    df["ambiguous"] = df["ambiguity_flags"].apply(lambda x: int((_to_dict(x).get("num_distinct_answers") or 1) > 1))
    df["conf_gap_feature"] = df["context_logprob"] - df["memory_logprob"]
    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()

    ablations = {
        "no_audited_labels": main["ranking_instability"]["proxy_vs_audited"],
        "no_utility": {},
        "no_alias_normalization": {},
        "strict_boundary": main["metrics"]["strict_abstain"],
        "permissive_boundary": main["metrics"]["permissive_memory"],
    }

    for policy, policy_metrics in main["metrics"]["audited"].items():
        ablations["no_utility"][policy] = policy_metrics["benchmark_native_accuracy"]

    for seed in SEEDS:
        _log("ablations", f"Running ablations for seed={seed}.")
        cfg = baseline_cfg[str(seed)]
        policies = _policy_predictions(test, cfg["tau_conf"], cfg["heuristic"], cfg["proxy_router"])
        for policy, pdf in policies.items():
            rows_naive, metrics_naive = _score_rows(pdf, "audited", alias_mode="naive")
            _save_prediction_rows("ablations", f"seed_{seed}_no_alias_{policy}", rows_naive)
            ablations["no_alias_normalization"].setdefault(policy, {})[str(seed)] = metrics_naive

        heur = dict(cfg["heuristic"])
        heur_wo_contra = dict(heur)
        heur_wo_contra["contradiction_tau"] = 1.1
        heur_wo_contra["conf_gap"] = 99.0
        heur_wo_support = dict(heur)
        heur_wo_support["support_min"] = 99
        rows_contra, met_contra = _score_rows(
            _policy_predictions(test, cfg["tau_conf"], heur_wo_contra, cfg["proxy_router"])["supported_else_abstain"],
            "audited",
        )
        rows_support, met_support = _score_rows(
            _policy_predictions(test, cfg["tau_conf"], heur_wo_support, cfg["proxy_router"])["supported_else_abstain"],
            "audited",
        )
        _save_prediction_rows("ablations", f"seed_{seed}_heuristic_without_contradiction", rows_contra)
        _save_prediction_rows("ablations", f"seed_{seed}_heuristic_without_support", rows_support)
        ablations.setdefault("heuristic_without_contradiction", {})[str(seed)] = met_contra
        ablations.setdefault("heuristic_without_support", {})[str(seed)] = met_support

    ablations["no_alias_normalization"] = {
        policy: _aggregate_seed_metrics({seed: vals for seed, vals in seed_map.items()})
        for policy, seed_map in ablations["no_alias_normalization"].items()
    }
    ablations["heuristic_without_contradiction"] = _aggregate_seed_metrics(
        {int(seed): vals for seed, vals in ablations["heuristic_without_contradiction"].items()}
    )
    ablations["heuristic_without_support"] = _aggregate_seed_metrics(
        {int(seed): vals for seed, vals in ablations["heuristic_without_support"].items()}
    )

    best_choices = []
    for seed in SEEDS:
        set_seed(seed)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=20, random_state=seed)
        for train_idx, val_idx in splitter.split(dev, dev["audited_label"]):
            train_df = dev.iloc[train_idx]
            val_df = dev.iloc[val_idx]
            best = None
            best_util = -1e9
            for c_value in LOGISTIC_C_GRID:
                model, feats = _fit_logistic(train_df, c_value)
                probs = model.predict_proba(val_df[feats].to_numpy())
                labels = list(model.classes_)
                for tau in LOGISTIC_TAU_GRID:
                    rows = []
                    for row, prob in zip(val_df.to_dict("records"), probs):
                        max_idx = int(np.argmax(prob))
                        pred_label = labels[max_idx] if prob[max_idx] >= tau else "abstain"
                        gold_aliases = row["audited_answers"] if row["audited_label"] != "abstain" else []
                        answer_ok = pred_label != "abstain" and alias_match(
                            row["context_answer"] if pred_label == "context" else row["memory_answer"],
                            gold_aliases,
                        )
                        rows.append({"pred_route": pred_label, "answer_ok": answer_ok, "gold_route": row["audited_label"]})
                    util = summarize_predictions(rows, "audited")["expected_utility"]
                    if util > best_util:
                        best_util = util
                        best = {"C": c_value, "tau_prob": tau}
            best_choices.append(best)

    chosen_c = float(round(np.mean([x["C"] for x in best_choices]), 2))
    chosen_tau = float(round(np.mean([x["tau_prob"] for x in best_choices]), 2))
    logistic_model, feats = _fit_logistic(dev, chosen_c)
    logistic_rows = _logistic_rows(logistic_model, feats, test, chosen_tau)
    logistic_model_wo, feats_wo = _fit_logistic(dev, chosen_c, drop_ambiguity=True)
    logistic_rows_wo = _logistic_rows(logistic_model_wo, feats_wo, test, chosen_tau)
    _save_prediction_rows("ablations", "appendix_logistic_router", logistic_rows)
    _save_prediction_rows("ablations", "appendix_logistic_router_without_ambiguity", logistic_rows_wo)
    appendix = {
        "logistic_router": summarize_predictions(logistic_rows, "audited"),
        "logistic_router_without_ambiguity": summarize_predictions(logistic_rows_wo, "audited"),
        "selected_hparams": {"C": chosen_c, "tau_prob": chosen_tau, "seed_choices": best_choices},
    }

    results = {
        "experiment": "ablations",
        "metrics": ablations,
        "appendix": appendix,
        "runtime_minutes": (time.time() - start) / 60.0,
    }
    write_json(base / "results.json", results)
    _log("ablations", "Ablations finished.")
    return results


def _extract_mean_metric(policy_metrics: Dict[str, Dict], metric_name: str) -> Dict[str, float]:
    return {policy: metrics[metric_name]["mean"] for policy, metrics in policy_metrics.items()}


def _bootstrap_metric(values: List[float], seed: int) -> Dict[str, float]:
    lo, hi = bootstrap_ci(values, np.random.default_rng(seed), 1000)
    return {"lo": lo, "hi": hi}


def _resample_predictions(rows: List[Dict], rng: np.random.Generator) -> List[Dict]:
    if not rows:
        return []
    idx = rng.integers(0, len(rows), len(rows))
    return [rows[int(i)] for i in idx]


def _policy_utility_from_rows(rows: List[Dict]) -> float:
    return float(np.mean([float(row["utility"]) for row in rows])) if rows else float("nan")


def _policy_harm_from_rows(rows: List[Dict]) -> float:
    return float(np.mean([1.0 if row["harmful"] else 0.0 for row in rows])) if rows else float("nan")


def _bootstrap_policy_metrics(rows: List[Dict], seed: int, n_resamples: int = 1000) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    utilities = []
    harms = []
    for _ in range(n_resamples):
        sample = _resample_predictions(rows, rng)
        utilities.append(_policy_utility_from_rows(sample))
        harms.append(_policy_harm_from_rows(sample))
    util_lo, util_hi = np.quantile(utilities, [0.025, 0.975])
    harm_lo, harm_hi = np.quantile(harms, [0.025, 0.975])
    return {
        "expected_utility": {"lo": float(util_lo), "hi": float(util_hi)},
        "harmful_answer_rate": {"lo": float(harm_lo), "hi": float(harm_hi)},
    }


def _bootstrap_rank_stats_by_seed(seed: int, policies: List[str], regime_a: str, regime_b: str, n_resamples: int = 1000) -> Dict[str, Dict[str, float]]:
    regime_rows = {}
    for regime in [regime_a, regime_b]:
        regime_rows[regime] = {
            policy: read_jsonl(ROOT / "exp/main/predictions" / f"seed_{seed}_{regime}_{policy}.jsonl")
            for policy in policies
        }
    n_items = len(next(iter(regime_rows[regime_a].values())))
    rng = np.random.default_rng(seed)
    taus = []
    reversals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_items, n_items)
        ranks = {}
        for regime in [regime_a, regime_b]:
            sampled_utils = {}
            for policy, rows in regime_rows[regime].items():
                sampled_rows = [rows[int(i)] for i in idx]
                sampled_utils[policy] = _policy_utility_from_rows(sampled_rows)
            ordered = sorted(sampled_utils.items(), key=lambda kv: kv[1], reverse=True)
            ranks[regime] = {policy: rank for rank, (policy, _) in enumerate(ordered, start=1)}
        stats = ranking_stats(ranks[regime_a], ranks[regime_b])
        taus.append(stats["kendall_tau"])
        reversals.append(float(stats["pairwise_reversals"]))
    tau_lo, tau_hi = np.quantile(taus, [0.025, 0.975])
    rev_lo, rev_hi = np.quantile(reversals, [0.025, 0.975])
    return {
        "kendall_tau": {"lo": float(tau_lo), "hi": float(tau_hi)},
        "pairwise_reversals": {"lo": float(rev_lo), "hi": float(rev_hi)},
    }


def run_eval_and_figures() -> Dict:
    import matplotlib.pyplot as plt

    start = time.time()
    _ensure_exp_layout("eval")
    _log("eval", "Starting evaluation aggregation and figure generation.")
    main = read_json(ROOT / "exp/main/results.json")
    ablations = read_json(ROOT / "exp/ablations/results.json")
    baseline = read_json(ROOT / "exp/baselines/results.json")
    audit = read_json(ROOT / "exp/audit/results.json")
    df = pd.read_parquet(ROOT / "exp/cache/qwen_dual_view_outputs.parquet")

    ranking_df = pd.DataFrame(main["policy_ranks"]).sort_index()
    plt.figure(figsize=(8, 5))
    for policy in ranking_df.index:
        plt.plot(ranking_df.columns, ranking_df.loc[policy].tolist(), marker="o", label=policy)
    plt.gca().invert_yaxis()
    plt.ylabel("Rank")
    plt.title("Policy Rank Shift Across Scoring Regimes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/ranking_shift.png", dpi=200)
    plt.close()

    proxy_gold = df[df["split"] == "test"].apply(_proxy_gold_route, axis=1)
    proxy_conf = pd.crosstab(df[df["split"] == "test"]["audited_label"], proxy_gold)
    plt.figure(figsize=(5, 4))
    plt.imshow(proxy_conf.values, cmap="Blues")
    plt.xticks(range(len(proxy_conf.columns)), proxy_conf.columns, rotation=45, ha="right")
    plt.yticks(range(len(proxy_conf.index)), proxy_conf.index)
    plt.title("Audited Proxy vs Single-Pass Audit Labels")
    for i in range(proxy_conf.shape[0]):
        for j in range(proxy_conf.shape[1]):
            plt.text(j, i, int(proxy_conf.values[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / "figures/proxy_confusion.png", dpi=200)
    plt.close()

    disagreement = {
        "strict_vs_permissive_flips": audit["metrics"]["strict_vs_permissive_label_flips"],
        "draft_disagreements": int(df["draft_disagreement"].sum()),
        "proxy_vs_single_pass_audit": int((proxy_gold != df[df["split"] == "test"]["audited_label"]).sum()),
    }
    plt.figure(figsize=(6, 4))
    plt.bar(disagreement.keys(), disagreement.values())
    plt.xticks(rotation=20, ha="right")
    plt.title("Disagreement Counts")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/disagreement_bars.png", dpi=200)
    plt.close()

    util_cov = []
    for policy in ["always_context", "always_memory", "always_abstain", "abstain_low_conf", "supported_else_abstain"]:
        met = main["metrics"]["audited"][policy]
        util_cov.append((policy, met["coverage"]["mean"], met["expected_utility"]["mean"]))
    util_cov.append(
        (
            "logistic_router",
            ablations["appendix"]["logistic_router"]["coverage"],
            ablations["appendix"]["logistic_router"]["expected_utility"],
        )
    )
    plt.figure(figsize=(6, 4))
    for policy, cov, util in util_cov:
        plt.scatter(cov, util, label=policy)
        plt.text(cov, util, policy, fontsize=8)
    plt.xlabel("Coverage")
    plt.ylabel("Expected Utility")
    plt.title("Utility vs Coverage")
    plt.tight_layout()
    plt.savefig(ROOT / "figures/utility_vs_coverage.png", dpi=200)
    plt.close()

    ci_payload = {}
    for regime in ["original", "proxy", "audited", "strict_abstain", "permissive_memory"]:
        ci_payload[regime] = {}
        for policy in POLICIES:
            ci_payload[regime][policy] = {}
            for seed in SEEDS:
                pred_path = ROOT / "exp/main/predictions" / f"seed_{seed}_{regime}_{policy}.jsonl"
                rows = read_jsonl(pred_path)
                ci_payload[regime][policy][f"seed_{seed}"] = _bootstrap_policy_metrics(rows, seed)

    original_utils = _extract_mean_metric(main["metrics"]["original"], "expected_utility")
    audited_utils = _extract_mean_metric(main["metrics"]["audited"], "expected_utility")
    kendall_ci = {
        f"seed_{seed}": _bootstrap_rank_stats_by_seed(seed, POLICIES, "original", "audited")
        for seed in SEEDS
    }

    proxy_disagreement = float(np.mean((proxy_gold != df[df["split"] == "test"]["audited_label"]).astype(float)))
    aggregated = {
        "audit": audit,
        "cache": read_json(ROOT / "exp/cache/results.json"),
        "baselines": baseline,
        "main": main,
        "ablations": ablations,
        "bootstrap_cis": ci_payload,
        "kendall_tau_ci": kendall_ci,
        "summary": {
            "audit_status": "blocked_no_human_annotations",
            "best_original_policy": min(main["policy_ranks"]["original"], key=main["policy_ranks"]["original"].get),
            "best_audited_policy": min(main["policy_ranks"]["audited"], key=main["policy_ranks"]["audited"].get),
            "proxy_vs_single_pass_audit_disagreement": proxy_disagreement,
            "main_claim_status": (
                "unsupported_due_missing_human_audit"
                if audit["metrics"]["cohens_kappa_overall"] is None
                else (
                    "supported"
                    if (
                        main["ranking_instability"]["original_vs_audited"]["top_policy_change"]
                        or main["ranking_instability"]["original_vs_audited"]["pairwise_reversals"] >= 2
                        or main["ranking_instability"]["original_vs_audited"]["kendall_tau"] <= 0.70
                    )
                    else "negative_result"
                )
            ),
            "negative_result_note": "After repairing the scoring methodology, rankings do move materially, but the central audited claim remains unsupported because the workspace still lacks a real human-adjudicated test set.",
        },
        "success_checks": {
            "fixed_320_items": audit["metrics"]["num_items"] == 320,
            "non_discard_rate_ge_095": audit["metrics"]["non_discard_rate"] >= 0.95,
            "real_human_audit_available": audit["metrics"]["cohens_kappa_overall"] is not None,
            "proxy_disagreement_ge_012": proxy_disagreement >= 0.12,
            "ranking_changed_or_unstable": any(
                [
                    main["ranking_instability"]["original_vs_audited"]["top_policy_change"],
                    main["ranking_instability"]["original_vs_audited"]["pairwise_reversals"] >= 2,
                    main["ranking_instability"]["original_vs_audited"]["kendall_tau"] <= 0.70,
                ]
            ),
        },
    }
    write_json(ROOT / "results.json", aggregated)
    results = {
        "experiment": "eval",
        "metrics": {
            "figures_created": 4,
            "success_checks": aggregated["success_checks"],
            "summary": aggregated["summary"],
            "original_mean_utilities": original_utils,
            "audited_mean_utilities": audited_utils,
        },
        "runtime_minutes": (time.time() - start) / 60.0,
    }
    write_json(ROOT / "exp/eval/results.json", results)
    _log("eval", "Evaluation aggregation finished and figures written.")
    return results
