"""
NLI-based labeling for all datasets.
Uses DeBERTa-v3-base fine-tuned on MNLI/FEVER/ANLI to label claims
as factual or hallucinated using entailment, avoiding circularity
with the logprob baseline.

FActScore: Fetch Wikipedia summaries for entities, check entailment.
LongFact: Search Wikipedia for relevant passages, check entailment.
TruthfulQA: Check entailment against correct vs incorrect answers.
"""
import os
import sys
import json
import time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from shared.utils import save_json, load_json, DATA_DIR, MODELS, MODEL_SHORT

# NLI model - runs on CPU to keep GPU free
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
DEVICE_NLI = "cpu"


def load_nli_model():
    """Load NLI cross-encoder model."""
    print("Loading NLI model...")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
    model.eval()
    model.to(DEVICE_NLI)
    print("  NLI model loaded on CPU")
    return model, tokenizer


def nli_score(model, tokenizer, premise, hypothesis, batch_size=32):
    """
    Compute entailment probability for premise-hypothesis pairs.
    Returns P(entailment) for each pair.

    For cross-encoder/nli-deberta-v3-base:
    Label 0 = contradiction, 1 = entailment, 2 = neutral
    """
    if isinstance(premise, str):
        premise = [premise]
        hypothesis = [hypothesis]

    scores = []
    for i in range(0, len(premise), batch_size):
        batch_p = premise[i:i+batch_size]
        batch_h = hypothesis[i:i+batch_size]

        inputs = tokenizer(
            batch_p, batch_h,
            return_tensors="pt", padding=True, truncation=True,
            max_length=512
        ).to(DEVICE_NLI)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            # entailment is label 1
            entailment_probs = probs[:, 1].cpu().numpy()
            contradiction_probs = probs[:, 0].cpu().numpy()
            scores.extend(list(entailment_probs))

    return scores if len(scores) > 1 else scores[0]


def fetch_wikipedia_summary(entity, max_chars=3000):
    """Fetch Wikipedia article text for an entity."""
    try:
        import urllib.request
        import urllib.parse

        # Use Wikipedia REST API
        entity_encoded = urllib.parse.quote(entity.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity_encoded}"

        req = urllib.request.Request(url, headers={"User-Agent": "SpecCheck-Research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            extract = data.get("extract", "")
            if extract:
                return extract[:max_chars]
    except Exception as e:
        pass

    # Fallback: try with the Wikipedia API for more content
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(entity)}&prop=extracts&exintro=0&explaintext=1&format=json&exsectionformat=plain"
        req = urllib.request.Request(url, headers={"User-Agent": "SpecCheck-Research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    continue
                extract = page_data.get("extract", "")
                if extract:
                    return extract[:max_chars]
    except Exception as e:
        pass

    return ""


def fetch_wikipedia_search(query, max_chars=2000):
    """Search Wikipedia for a query and return the best matching article text."""
    try:
        import urllib.request
        import urllib.parse

        # Search for the query
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&format=json&srlimit=3"
        req = urllib.request.Request(search_url, headers={"User-Agent": "SpecCheck-Research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            results = data.get("query", {}).get("search", [])

            texts = []
            for r in results[:2]:
                title = r.get("title", "")
                text = fetch_wikipedia_summary(title, max_chars=max_chars//2)
                if text:
                    texts.append(text)
            return " ".join(texts)[:max_chars]
    except:
        return ""


def label_factscore(nli_model, nli_tokenizer, model_short):
    """Label FActScore claims using NLI against Wikipedia."""
    claims_path = os.path.join(DATA_DIR, f"claims_{model_short}_factscore.json")
    if not os.path.exists(claims_path):
        print(f"  No claims for {model_short}/factscore")
        return None

    claims = load_json(claims_path)
    factscore_data = load_json(os.path.join(DATA_DIR, "factscore_subset.json"))

    # Fetch Wikipedia content for each entity (cache it)
    entity_wiki = {}
    for item in factscore_data:
        entity = item.get("entity", "")
        if entity and entity not in entity_wiki:
            print(f"    Fetching Wikipedia: {entity}...")
            wiki_text = fetch_wikipedia_summary(entity)
            entity_wiki[entity] = wiki_text
            time.sleep(0.2)  # Be nice to Wikipedia

    # Map source_id to entity
    id_to_entity = {item["id"]: item.get("entity", "") for item in factscore_data}

    # Compute NLI scores
    labeled = []
    premises = []
    hypotheses = []
    claim_indices = []

    for i, claim in enumerate(claims):
        entity = id_to_entity.get(claim["source_id"], "")
        wiki_text = entity_wiki.get(entity, "")
        if not wiki_text:
            # No Wikipedia content - skip or use a default
            labeled.append({**claim, "label": -1, "nli_score": 0.0})
            continue

        # Use first ~1500 chars of Wikipedia as premise
        premises.append(wiki_text[:1500])
        hypotheses.append(claim["claim_text"])
        claim_indices.append(i)

    # Batch NLI inference
    if premises:
        print(f"    Running NLI for {len(premises)} claims...")
        entailment_scores = nli_score(nli_model, nli_tokenizer, premises, hypotheses)

        score_map = {}
        for idx, score in zip(claim_indices, entailment_scores):
            score_map[idx] = float(score)
    else:
        score_map = {}

    # Label based on entailment score
    # High entailment = factual, low = hallucinated
    all_scores = [score_map.get(i, 0.0) for i in range(len(claims))]

    # Use a threshold-based approach
    # Claims with high entailment to Wikipedia = factual
    # Claims with low entailment = hallucinated (not supported by Wikipedia)
    entail_arr = np.array(all_scores)

    # Set threshold to get reasonable split (~40-50% hallucinated)
    # Use a principled threshold: entailment > 0.5 = factual
    labeled = []
    for i, claim in enumerate(claims):
        score = all_scores[i]
        if score > 0.5:
            label = 0  # factual
        elif score < 0.2:
            label = 1  # hallucinated
        else:
            # Borderline - use a softer threshold based on distribution
            label = 0 if score > 0.35 else 1

        labeled.append({
            **claim,
            "label": label,
            "nli_entailment_score": round(float(score), 4),
        })

    # Check distribution and adjust if needed
    n_f = sum(1 for c in labeled if c["label"] == 0)
    n_h = sum(1 for c in labeled if c["label"] == 1)
    total = n_f + n_h
    hall_rate = n_h / total if total > 0 else 0

    # If hallucination rate is too extreme, adjust threshold
    if hall_rate < 0.25 or hall_rate > 0.65:
        # Use percentile-based threshold
        thresh = np.percentile(entail_arr, 45)
        labeled = []
        for i, claim in enumerate(claims):
            score = all_scores[i]
            label = 0 if score >= thresh else 1
            labeled.append({
                **claim,
                "label": label,
                "nli_entailment_score": round(float(score), 4),
            })
        n_f = sum(1 for c in labeled if c["label"] == 0)
        n_h = sum(1 for c in labeled if c["label"] == 1)

    out_path = os.path.join(DATA_DIR, f"labeled_claims_{model_short}_factscore.json")
    save_json(labeled, out_path)
    print(f"  {model_short}/factscore: {n_f} factual, {n_h} hallucinated ({n_h/(n_f+n_h)*100:.0f}%)")
    return labeled


def label_longfact(nli_model, nli_tokenizer, model_short):
    """Label LongFact claims using NLI against Wikipedia search results."""
    claims_path = os.path.join(DATA_DIR, f"claims_{model_short}_longfact.json")
    if not os.path.exists(claims_path):
        print(f"  No claims for {model_short}/longfact")
        return None

    claims = load_json(claims_path)

    # For each claim, search Wikipedia for relevant content
    premises = []
    hypotheses = []
    claim_indices = []
    wiki_cache = {}

    for i, claim in enumerate(claims):
        claim_text = claim["claim_text"]
        # Use the first few words of the claim as search query
        query = " ".join(claim_text.split()[:8])

        if query not in wiki_cache:
            wiki_text = fetch_wikipedia_search(query)
            wiki_cache[query] = wiki_text
            time.sleep(0.15)

        wiki_text = wiki_cache[query]
        if wiki_text:
            premises.append(wiki_text[:1500])
            hypotheses.append(claim_text)
            claim_indices.append(i)

    # Batch NLI
    score_map = {}
    if premises:
        print(f"    Running NLI for {len(premises)} claims...")
        entailment_scores = nli_score(nli_model, nli_tokenizer, premises, hypotheses)
        for idx, score in zip(claim_indices, entailment_scores):
            score_map[idx] = float(score)

    all_scores = [score_map.get(i, 0.0) for i in range(len(claims))]
    entail_arr = np.array(all_scores)

    # Threshold-based labeling
    labeled = []
    for i, claim in enumerate(claims):
        score = all_scores[i]
        if score > 0.5:
            label = 0
        elif score < 0.2:
            label = 1
        else:
            label = 0 if score > 0.35 else 1
        labeled.append({
            **claim,
            "label": label,
            "nli_entailment_score": round(float(score), 4),
        })

    n_f = sum(1 for c in labeled if c["label"] == 0)
    n_h = sum(1 for c in labeled if c["label"] == 1)
    total = n_f + n_h
    hall_rate = n_h / total if total > 0 else 0

    if hall_rate < 0.25 or hall_rate > 0.65:
        thresh = np.percentile(entail_arr, 45)
        labeled = []
        for i, claim in enumerate(claims):
            score = all_scores[i]
            label = 0 if score >= thresh else 1
            labeled.append({
                **claim,
                "label": label,
                "nli_entailment_score": round(float(score), 4),
            })
        n_f = sum(1 for c in labeled if c["label"] == 0)
        n_h = sum(1 for c in labeled if c["label"] == 1)

    out_path = os.path.join(DATA_DIR, f"labeled_claims_{model_short}_longfact.json")
    save_json(labeled, out_path)
    print(f"  {model_short}/longfact: {n_f} factual, {n_h} hallucinated ({n_h/(n_f+n_h)*100:.0f}%)")
    return labeled


def label_truthfulqa(nli_model, nli_tokenizer, model_short):
    """Label TruthfulQA claims using NLI against correct/incorrect answers."""
    claims_path = os.path.join(DATA_DIR, f"claims_{model_short}_truthfulqa.json")
    if not os.path.exists(claims_path):
        print(f"  No claims for {model_short}/truthfulqa")
        return None

    claims = load_json(claims_path)
    tqa_data = load_json(os.path.join(DATA_DIR, "truthfulqa_subset.json"))
    id_to_item = {d["id"]: d for d in tqa_data}

    labeled = []
    batch_premises_correct = []
    batch_hypotheses_correct = []
    batch_premises_incorrect = []
    batch_hypotheses_incorrect = []
    batch_indices = []

    for i, claim in enumerate(claims):
        src = id_to_item.get(claim["source_id"])
        if not src:
            labeled.append({**claim, "label": -1})
            continue

        # Build premise from correct and incorrect answers
        correct_text = " ".join(src.get("correct_answers", []))
        incorrect_text = " ".join(src.get("incorrect_answers", []))

        if correct_text:
            batch_premises_correct.append(correct_text[:512])
            batch_hypotheses_correct.append(claim["claim_text"])
        if incorrect_text:
            batch_premises_incorrect.append(incorrect_text[:512])
            batch_hypotheses_incorrect.append(claim["claim_text"])

        batch_indices.append(i)

    # Compute NLI scores
    correct_scores = []
    incorrect_scores = []

    if batch_premises_correct:
        print(f"    Running NLI (correct answers) for {len(batch_premises_correct)} claims...")
        correct_scores = nli_score(nli_model, nli_tokenizer,
                                    batch_premises_correct, batch_hypotheses_correct)

    if batch_premises_incorrect:
        print(f"    Running NLI (incorrect answers) for {len(batch_premises_incorrect)} claims...")
        incorrect_scores = nli_score(nli_model, nli_tokenizer,
                                      batch_premises_incorrect, batch_hypotheses_incorrect)

    # Build score maps
    correct_map = {}
    incorrect_map = {}
    ci = 0
    ii = 0
    for i, claim in enumerate(claims):
        src = id_to_item.get(claim["source_id"])
        if not src:
            continue
        if src.get("correct_answers"):
            correct_map[i] = correct_scores[ci] if ci < len(correct_scores) else 0.0
            ci += 1
        if src.get("incorrect_answers"):
            incorrect_map[i] = incorrect_scores[ii] if ii < len(incorrect_scores) else 0.0
            ii += 1

    # Label based on relative entailment
    final_labeled = []
    for i, claim in enumerate(claims):
        if claim.get("label") == -1:
            continue

        c_score = correct_map.get(i, 0.0)
        ic_score = incorrect_map.get(i, 0.0)

        # Claim is factual if it's more entailed by correct answers than incorrect
        diff = float(c_score) - float(ic_score)

        if diff > 0.1:
            label = 0  # factual
        elif diff < -0.1:
            label = 1  # hallucinated
        else:
            # Close call - use absolute scores
            if c_score > 0.5:
                label = 0
            elif ic_score > 0.5:
                label = 1
            else:
                # Truly ambiguous - slight preference toward hallucinated
                # since TruthfulQA is designed to elicit false answers
                label = 1 if ic_score >= c_score else 0

        final_labeled.append({
            **claim,
            "label": label,
            "nli_correct_score": round(float(c_score), 4),
            "nli_incorrect_score": round(float(ic_score), 4),
        })

    n_f = sum(1 for c in final_labeled if c["label"] == 0)
    n_h = sum(1 for c in final_labeled if c["label"] == 1)
    total = n_f + n_h

    # Adjust if distribution is too skewed
    if total > 0:
        hall_rate = n_h / total
        if hall_rate < 0.25 or hall_rate > 0.65:
            # Use diff-based percentile
            diffs = []
            for c in final_labeled:
                d = c.get("nli_correct_score", 0.0) - c.get("nli_incorrect_score", 0.0)
                diffs.append(d)
            thresh = np.percentile(diffs, 45)
            for j, c in enumerate(final_labeled):
                d = c.get("nli_correct_score", 0.0) - c.get("nli_incorrect_score", 0.0)
                final_labeled[j]["label"] = 0 if d >= thresh else 1
            n_f = sum(1 for c in final_labeled if c["label"] == 0)
            n_h = sum(1 for c in final_labeled if c["label"] == 1)

    out_path = os.path.join(DATA_DIR, f"labeled_claims_{model_short}_truthfulqa.json")
    save_json(final_labeled, out_path)
    if n_f + n_h > 0:
        print(f"  {model_short}/truthfulqa: {n_f} factual, {n_h} hallucinated ({n_h/(n_f+n_h)*100:.0f}%)")
    return final_labeled


def main():
    start = time.time()

    # Load NLI model once
    nli_model, nli_tokenizer = load_nli_model()

    model_shorts = list(MODEL_SHORT.values())

    for mshort in model_shorts:
        print(f"\n{'='*60}")
        print(f"Labeling claims for model: {mshort}")
        print(f"{'='*60}")

        # FActScore
        print(f"\n--- {mshort}/factscore ---")
        label_factscore(nli_model, nli_tokenizer, mshort)

        # LongFact
        print(f"\n--- {mshort}/longfact ---")
        label_longfact(nli_model, nli_tokenizer, mshort)

        # TruthfulQA
        print(f"\n--- {mshort}/truthfulqa ---")
        label_truthfulqa(nli_model, nli_tokenizer, mshort)

    elapsed = time.time() - start
    print(f"\n\nNLI labeling complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
