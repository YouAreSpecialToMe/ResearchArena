from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .utils import append_log, exp_log_path, set_seed


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelBundle:
    retriever_tokenizer: Any
    retriever_model: Any
    reranker_tokenizer: Any
    reranker_model: Any
    verifier_tokenizer: Any
    verifier_model: Any
    device: str


@lru_cache(maxsize=1)
def load_bundle() -> ModelBundle:
    append_log(exp_log_path("pilot"), "Loading pretrained retriever, reranker, and verifier models.")
    device = _device()
    retriever_name = "intfloat/e5-base-v2"
    reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    verifier_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
    retriever_model = AutoModel.from_pretrained(retriever_name).to(device).eval()
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name).to(device).eval()
    verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_name)
    verifier_model = AutoModelForSequenceClassification.from_pretrained(verifier_name).to(device).eval()
    if device == "cuda":
        retriever_model.half()
        reranker_model.half()
        verifier_model.half()
    append_log(exp_log_path("pilot"), f"Finished model load on device={device}.")
    return ModelBundle(
        retriever_tokenizer,
        retriever_model,
        reranker_tokenizer,
        reranker_model,
        verifier_tokenizer,
        verifier_model,
        device,
    )


def mean_pool(last_hidden_state, attention_mask):
    masked = last_hidden_state * attention_mask.unsqueeze(-1)
    return masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1e-6)


@torch.no_grad()
def encode_texts(texts: list[str], prefix: str, batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    bundle = load_bundle()
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {t}" for t in texts[start : start + batch_size]]
        toks = bundle.retriever_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(bundle.device)
        hidden = bundle.retriever_model(**toks).last_hidden_state
        pooled = mean_pool(hidden, toks["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def rerank_pairs(claim: str, evidence_texts: list[str]) -> np.ndarray:
    if not evidence_texts:
        return np.zeros(0, dtype=np.float32)
    bundle = load_bundle()
    toks = bundle.reranker_tokenizer(
        [claim] * len(evidence_texts),
        evidence_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    ).to(bundle.device)
    logits = bundle.reranker_model(**toks).logits.squeeze(-1)
    return logits.float().cpu().numpy()


@torch.no_grad()
def verify_pairs(claims: list[str], evidences: list[str], batch_size: int = 24) -> np.ndarray:
    if not claims:
        return np.zeros((0, 3), dtype=np.float32)
    bundle = load_bundle()
    out = []
    for start in range(0, len(claims), batch_size):
        claim_batch = claims[start : start + batch_size]
        evidence_batch = evidences[start : start + batch_size]
        toks = bundle.verifier_tokenizer(
            claim_batch,
            evidence_batch,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        ).to(bundle.device)
        probs = torch.softmax(bundle.verifier_model(**toks).logits, dim=-1)
        out.append(probs.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def pick_label_indices(config) -> tuple[int, int, int]:
    id2label = {int(k): v for k, v in config.id2label.items()}
    entail = next(k for k, v in id2label.items() if "entail" in v.lower())
    contradiction = next(k for k, v in id2label.items() if "contrad" in v.lower())
    neutral = next(k for k, v in id2label.items() if "neutral" in v.lower())
    return entail, contradiction, neutral


def _support_dict(prob_row: np.ndarray, entail_idx: int, contradiction_idx: int, neutral_idx: int) -> dict[str, float]:
    return {
        "entailment": float(prob_row[entail_idx]),
        "contradiction": float(prob_row[contradiction_idx]),
        "neutral": float(prob_row[neutral_idx]),
    }


def _context_supports(
    claim_text: str,
    contexts: list[str],
    entail_idx: int,
    contradiction_idx: int,
    neutral_idx: int,
) -> list[dict[str, float]]:
    probs = verify_pairs([claim_text] * len(contexts), contexts)
    return [_support_dict(row, entail_idx, contradiction_idx, neutral_idx) for row in probs]


def _choose_greedy_bundle(prefix_supports: list[dict[str, float]], tau: float) -> list[int]:
    if not prefix_supports:
        return []
    for size, support in enumerate(prefix_supports, start=1):
        if support["entailment"] >= tau:
            return list(range(size))
    return list(range(len(prefix_supports)))


def _ordered_indices(similarities: np.ndarray, top_k: int) -> list[int]:
    if len(similarities) == 0:
        return []
    return [int(i) for i in np.argsort(-similarities)[:top_k]]


def _reranked_indices(claim_text: str, evidence_texts: list[str], similarities: np.ndarray) -> list[int]:
    retrieve = _ordered_indices(similarities, min(12, len(evidence_texts)))
    if not retrieve:
        return []
    rerank_scores = rerank_pairs(claim_text, [evidence_texts[i] for i in retrieve])
    rerank_local = np.argsort(-rerank_scores)[: min(6, len(retrieve))]
    return [int(retrieve[i]) for i in rerank_local]


def _prefix_contexts(reranked_texts: list[str], max_size: int = 3) -> list[str]:
    contexts = []
    for size in range(1, min(max_size, len(reranked_texts)) + 1):
        contexts.append(" ".join(reranked_texts[:size]))
    return contexts


def _rescore_after_removal(
    claim_text: str,
    evidence_texts: list[str],
    claim_sim: np.ndarray,
    removed_indices: list[int],
    tau: float,
    entail_idx: int,
    contradiction_idx: int,
    neutral_idx: int,
) -> tuple[float, list[int], float]:
    residual_indices = [i for i in range(len(evidence_texts)) if i not in set(removed_indices)]
    if not residual_indices:
        return 0.0, [], 0.0
    residual_sim = claim_sim[residual_indices]
    retrieve_local = _ordered_indices(residual_sim, min(12, len(residual_indices)))
    retrieve = [residual_indices[i] for i in retrieve_local]
    rerank_scores = rerank_pairs(claim_text, [evidence_texts[i] for i in retrieve])
    rerank_local = np.argsort(-rerank_scores)[: min(6, len(retrieve))]
    reranked = [int(retrieve[i]) for i in rerank_local]
    reranked_texts = [evidence_texts[i] for i in reranked]
    single_probs = verify_pairs([claim_text] * len(reranked_texts), reranked_texts)
    if len(single_probs) == 0:
        return 0.0, [], 0.0
    residual_best = float(single_probs[:, entail_idx].max())
    prefix_supports = _context_supports(
        claim_text,
        _prefix_contexts(reranked_texts),
        entail_idx,
        contradiction_idx,
        neutral_idx,
    )
    residual_bundle_local = _choose_greedy_bundle(prefix_supports, tau)
    residual_bundle = [reranked[i] for i in residual_bundle_local]
    residual_bundle_support = (
        float(prefix_supports[len(residual_bundle_local) - 1]["entailment"]) if residual_bundle_local else 0.0
    )
    return residual_best, residual_bundle, residual_bundle_support


def _compute_bundle_features(
    claim_text: str,
    tau: float,
    evidence_texts: list[str],
    evidence_emb: np.ndarray,
    evidence_doc_ids: list[str],
    claim_sim: np.ndarray,
    reranked_indices: list[int],
    entailments: np.ndarray,
    support_full: dict[str, float],
    support_no_context: dict[str, float],
    bundle_local: list[int],
    prefix_supports: list[dict[str, float]],
    entail_idx: int,
    contradiction_idx: int,
    neutral_idx: int,
) -> dict[str, Any]:
    smin_global = [reranked_indices[i] for i in bundle_local]
    smin_texts = [evidence_texts[i] for i in smin_global]
    smin_support = prefix_supports[len(bundle_local) - 1] if bundle_local else {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    alt_count = int((entailments >= tau).sum())
    doc_dispersion = len({evidence_doc_ids[idx] for idx in smin_global})

    contexts = []
    context_keys = []
    remove_local_texts = [text for idx, text in enumerate(evidence_texts) if idx not in set(smin_global)]
    contexts.append(" ".join(remove_local_texts))
    context_keys.append("remove_local")

    for pos in range(len(smin_texts)):
        subset = [text for j, text in enumerate(smin_texts) if j != pos]
        contexts.append(" ".join(subset))
        context_keys.append(f"drop_one_{pos}")

    reranked_outside = [idx for idx in reranked_indices if idx not in smin_global]
    low_support_candidates = [idx for idx in reranked_outside if entailments[reranked_indices.index(idx)] < tau]
    swap_contexts = []
    for local_pos, global_idx in zip(bundle_local, smin_global):
        if not low_support_candidates:
            continue
        sims = evidence_emb[low_support_candidates] @ evidence_emb[global_idx]
        replacement_idx = low_support_candidates[int(np.argmax(sims))]
        swapped = smin_texts.copy()
        swapped[local_pos] = evidence_texts[replacement_idx]
        swap_contexts.append(" ".join(swapped))
    contexts.extend(swap_contexts)
    context_keys.extend([f"swap_{i}" for i in range(len(swap_contexts))])

    scored_contexts = _context_supports(claim_text, contexts, entail_idx, contradiction_idx, neutral_idx) if contexts else []
    score_map = {key: scored_contexts[i] for i, key in enumerate(context_keys)}

    loo_scores = [score_map[f"drop_one_{pos}"]["entailment"] for pos in range(len(smin_texts))]
    redundancy = float(smin_support["entailment"] - max(loo_scores)) if loo_scores else 0.0
    remove_local_support = score_map.get("remove_local", {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0})
    swap_scores = [score_map[key]["entailment"] for key in context_keys if key.startswith("swap_")]

    residual_best, residual_bundle, residual_bundle_support = _rescore_after_removal(
        claim_text,
        evidence_texts,
        claim_sim,
        smin_global,
        tau,
        entail_idx,
        contradiction_idx,
        neutral_idx,
    )
    bundle_changed = int(bool(residual_bundle) and residual_bundle_support >= tau)

    return {
        "support_remove_support": remove_local_support["entailment"],
        "drop_full_to_remove_support": support_full["entailment"] - remove_local_support["entailment"],
        "support_smin": smin_support["entailment"],
        "smin_size": len(smin_global),
        "support_margin_smin_vs_best1": smin_support["entailment"] - float(entailments.max() if len(entailments) else 0.0),
        "candidate_above_tau": alt_count,
        "document_dispersion": doc_dispersion,
        "redundancy_proxy": redundancy,
        "drop_full_to_remove_local": support_full["entailment"] - remove_local_support["entailment"],
        "mean_drop_one": float(np.mean(smin_support["entailment"] - np.array(loo_scores))) if loo_scores else 0.0,
        "max_drop_one": float(np.max(smin_support["entailment"] - np.array(loo_scores))) if loo_scores else 0.0,
        "min_drop_one": float(np.min(smin_support["entailment"] - np.array(loo_scores))) if loo_scores else 0.0,
        "drop_swap_local": float(smin_support["entailment"] - np.mean(swap_scores)) if swap_scores else 0.0,
        "best_residual_support_after_local_removal": residual_best,
        "normalized_drop": float(
            (support_full["entailment"] - remove_local_support["entailment"]) / max(support_full["entailment"], 1e-6)
        ),
        "support_bundle_changed_after_removal": bundle_changed,
        "smin_indices": smin_global,
        "residual_bundle_indices": residual_bundle,
        "smin_expansion_entailments": [round(s["entailment"], 6) for s in prefix_supports],
    }


def build_claim_features(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    taus: list[float],
    max_claims: int | None = None,
    full_features: bool = True,
    stability_only: bool = False,
) -> pd.DataFrame:
    log_path = exp_log_path("pilot")
    evidence_group = evidence_df.groupby("source_id")
    rows = []
    source_cache: dict[str, dict[str, Any]] = {}
    claim_iter = claims_df.itertuples(index=False)
    if max_claims is not None:
        claim_iter = list(claim_iter)[:max_claims]

    bundle = load_bundle()
    entail_idx, contradiction_idx, neutral_idx = pick_label_indices(bundle.verifier_model.config)
    total = min(len(claims_df), max_claims or len(claims_df))
    append_log(
        log_path,
        f"Starting {'full' if full_features else ('stability' if stability_only else 'tau-search')} feature assembly for {total} claims across tau grid {taus}.",
    )

    for claim_no, claim in enumerate(tqdm(claim_iter, total=total, desc="features"), start=1):
        source_evidence = evidence_group.get_group(claim.example_id).sort_values(
            ["original_retrieval_rank", "sentence_position", "doc_id"]
        )
        if claim.example_id not in source_cache:
            evidence_texts = source_evidence["evidence_text"].tolist()
            evidence_emb = encode_texts(evidence_texts, "passage")
            source_cache[claim.example_id] = {
                "evidence_texts": evidence_texts,
                "evidence_emb": evidence_emb,
                "evidence_doc_ids": source_evidence["doc_id"].tolist(),
                "bm25": BM25Okapi([text.lower().split() for text in evidence_texts]),
                "source_evidence": source_evidence.reset_index(drop=True),
                "full_context_text": " ".join(evidence_texts),
            }
        cached = source_cache[claim.example_id]
        evidence_texts = cached["evidence_texts"]
        evidence_emb = cached["evidence_emb"]
        evidence_doc_ids = cached["evidence_doc_ids"]
        if not evidence_texts:
            continue

        bm25_scores = np.array(cached["bm25"].get_scores(claim.answer_sentence.lower().split()), dtype=float)
        top_bm25 = np.argsort(-bm25_scores)[: min(3, len(bm25_scores))]

        claim_emb = encode_texts([claim.answer_sentence], "query")[0]
        claim_sim = evidence_emb @ claim_emb
        reranked_indices = _reranked_indices(claim.answer_sentence, evidence_texts, claim_sim)
        reranked_texts = [evidence_texts[i] for i in reranked_indices]
        if not reranked_texts:
            continue
        single_probs = verify_pairs([claim.answer_sentence] * len(reranked_texts), reranked_texts)
        entailments = single_probs[:, entail_idx]
        contradictions = single_probs[:, contradiction_idx]
        neutrals = single_probs[:, neutral_idx]
        second_best = np.sort(entailments)[-2] if len(entailments) > 1 else 0.0

        prefix_supports = _context_supports(
            claim.answer_sentence,
            _prefix_contexts(reranked_texts),
            entail_idx,
            contradiction_idx,
            neutral_idx,
        )
        common = {
            "example_id": claim.example_id,
            "response_id": claim.response_id,
            "sentence_index": claim.sentence_index,
            "split": claim.split,
            "task_type": claim.task_type,
            "generator_family": claim.generator_family,
            "projected_all_label": claim.projected_all_label,
            "strict_label": claim.strict_label,
            "ambiguity_flag": claim.ambiguity_flag,
            "top1_bm25": float(bm25_scores[top_bm25[0]]) if len(top_bm25) > 0 else 0.0,
            "mean_top3_bm25": float(bm25_scores[top_bm25].mean()) if len(top_bm25) > 0 else 0.0,
            "top1_minus_top3mean": float(bm25_scores[top_bm25[0]] - bm25_scores[top_bm25].mean()) if len(top_bm25) > 0 else 0.0,
            "support_best_entailment": float(entailments.max()),
            "support_best_contradiction": float(contradictions[np.argmax(entailments)]),
            "support_best_neutral": float(neutrals[np.argmax(entailments)]),
            "support_margin_second_best": float(entailments.max() - second_best),
            "reranked_indices": reranked_indices,
        }
        if full_features or stability_only:
            support_full, support_no_context = _context_supports(
                claim.answer_sentence,
                [cached["full_context_text"], ""],
                entail_idx,
                contradiction_idx,
                neutral_idx,
            )
            common.update(
                {
                    "support_full": support_full["entailment"],
                    "support_full_contradiction": support_full["contradiction"],
                    "support_full_neutral": support_full["neutral"],
                    "support_no_context": support_no_context["entailment"],
                    "drop_full_to_no_context": support_full["entailment"] - support_no_context["entailment"],
                }
            )

        for tau in taus:
            bundle_local = _choose_greedy_bundle(prefix_supports, tau)
            row = {
                "tau": tau,
                **common,
                "support_smin": float(prefix_supports[len(bundle_local) - 1]["entailment"]) if bundle_local else 0.0,
                "smin_size": len(bundle_local),
                "smin_indices": [reranked_indices[i] for i in bundle_local],
            }
            if full_features:
                bundle_features = _compute_bundle_features(
                    claim.answer_sentence,
                    tau,
                    evidence_texts,
                    evidence_emb,
                    evidence_doc_ids,
                    claim_sim,
                    reranked_indices,
                    entailments,
                    support_full,
                    support_no_context,
                    bundle_local,
                    prefix_supports,
                    entail_idx,
                    contradiction_idx,
                    neutral_idx,
                )
                fixed_top2_features = {}
                fixed_top2_local = list(range(min(2, len(prefix_supports))))
                if fixed_top2_local:
                    fixed_top2_raw = _compute_bundle_features(
                        claim.answer_sentence,
                        tau,
                        evidence_texts,
                        evidence_emb,
                        evidence_doc_ids,
                        claim_sim,
                        reranked_indices,
                        entailments,
                        support_full,
                        support_no_context,
                        fixed_top2_local,
                        prefix_supports,
                        entail_idx,
                        contradiction_idx,
                        neutral_idx,
                    )
                    fixed_top2_features = {
                        f"top2_{key}": value
                        for key, value in fixed_top2_raw.items()
                        if key not in {"smin_indices", "residual_bundle_indices", "smin_expansion_entailments"}
                    }
                row.update(bundle_features)
                row.update(fixed_top2_features)
            elif stability_only:
                smin_global = [reranked_indices[i] for i in bundle_local]
                remove_local_texts = [text for idx, text in enumerate(evidence_texts) if idx not in set(smin_global)]
                remove_local_support = _context_supports(
                    claim.answer_sentence,
                    [" ".join(remove_local_texts)],
                    entail_idx,
                    contradiction_idx,
                    neutral_idx,
                )[0]
                row.update(
                    {
                        "support_full": support_full["entailment"],
                        "support_no_context": support_no_context["entailment"],
                        "drop_full_to_no_context": support_full["entailment"] - support_no_context["entailment"],
                        "drop_full_to_remove_local": support_full["entailment"] - remove_local_support["entailment"],
                    }
                )
            rows.append(row)
        if claim_no % 500 == 0:
            append_log(log_path, f"Processed {claim_no}/{total} claims.")
    append_log(log_path, f"Finished feature assembly for {total} claims.")
    return pd.DataFrame(rows)


def fit_logreg(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    c_values: list[float],
    seed: int,
):
    from .metrics import compute_metrics

    set_seed(seed)
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(train_x)
    x_va = scaler.transform(val_x)
    best = None
    for c in c_values:
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            random_state=seed,
            max_iter=400,
            solver="liblinear",
        )
        clf.fit(x_tr, train_y)
        val_prob = clf.predict_proba(x_va)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 19)
        local_best = None
        for threshold in thresholds:
            metrics = compute_metrics(val_y, val_prob, threshold)
            candidate = {"clf": clf, "scaler": scaler, "threshold": float(threshold), "C": float(c), "metrics": metrics}
            if local_best is None or (
                metrics["auprc"],
                metrics["macro_f1"],
            ) > (
                local_best["metrics"]["auprc"],
                local_best["metrics"]["macro_f1"],
            ):
                local_best = candidate
        if best is None or (
            local_best["metrics"]["auprc"],
            local_best["metrics"]["macro_f1"],
        ) > (
            best["metrics"]["auprc"],
            best["metrics"]["macro_f1"],
        ):
            best = local_best
    return best
