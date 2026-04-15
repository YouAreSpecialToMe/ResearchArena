"""Data loading and preparation for C2UD experiments."""
import json
import os
import random
import numpy as np
from datasets import load_dataset


def load_and_sample_datasets(data_dir, n_per_dataset=500, seed=42):
    """Load NQ, TriviaQA, and PopQA, sample n examples each."""
    os.makedirs(data_dir, exist_ok=True)
    rng = random.Random(seed)
    datasets_info = {}

    # Natural Questions (open domain)
    print("Loading Natural Questions...")
    nq = load_dataset("nq_open", split="validation")
    indices = rng.sample(range(len(nq)), min(n_per_dataset, len(nq)))
    nq_data = []
    for idx in indices:
        ex = nq[idx]
        nq_data.append({
            "question": ex["question"],
            "gold_answers": ex["answer"],
            "dataset": "nq",
            "query_id": f"nq_{idx}"
        })
    datasets_info["nq"] = nq_data
    print(f"  NQ: {len(nq_data)} examples")

    # TriviaQA
    print("Loading TriviaQA...")
    tqa = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    indices = rng.sample(range(len(tqa)), min(n_per_dataset, len(tqa)))
    tqa_data = []
    for idx in indices:
        ex = tqa[idx]
        answers = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        answers = list(set(answers))
        tqa_data.append({
            "question": ex["question"],
            "gold_answers": answers,
            "dataset": "triviaqa",
            "query_id": f"tqa_{idx}"
        })
    datasets_info["triviaqa"] = tqa_data
    print(f"  TriviaQA: {len(tqa_data)} examples")

    # PopQA
    print("Loading PopQA...")
    popqa = load_dataset("akariasai/PopQA", split="test")
    indices = rng.sample(range(len(popqa)), min(n_per_dataset, len(popqa)))
    popqa_data = []
    for idx in indices:
        ex = popqa[idx]
        answers = [ex["possible_answers"]] if isinstance(ex["possible_answers"], str) else ex["possible_answers"]
        # PopQA possible_answers is a list stored as string
        if isinstance(answers[0], str) and answers[0].startswith("["):
            import ast
            try:
                answers = ast.literal_eval(answers[0])
            except:
                answers = [answers[0]]
        popqa_data.append({
            "question": ex["question"],
            "gold_answers": answers if isinstance(answers, list) else [answers],
            "dataset": "popqa",
            "query_id": f"popqa_{idx}"
        })
    datasets_info["popqa"] = popqa_data
    print(f"  PopQA: {len(popqa_data)} examples")

    # Save
    for name, data in datasets_info.items():
        path = os.path.join(data_dir, f"{name}_data.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return datasets_info


def prepare_retrieval_corpus(data_dir, n_passages=500000, seed=42):
    """Download Wikipedia passages for BM25 retrieval."""
    corpus_path = os.path.join(data_dir, "wiki_corpus.json")
    if os.path.exists(corpus_path):
        print("Loading cached Wikipedia corpus...")
        with open(corpus_path) as f:
            return json.load(f)

    print("Loading Wikipedia passages for retrieval...")
    # Try multiple Wikipedia dataset variants
    wiki = None
    for wiki_name in ["wikimedia/wikipedia", "wikipedia"]:
        for wiki_config in ["20231101.en", "20220301.en", "en"]:
            try:
                wiki = load_dataset(wiki_name, wiki_config, split="train", streaming=True)
                print(f"  Using {wiki_name}/{wiki_config}")
                break
            except Exception as e:
                continue
        if wiki is not None:
            break

    if wiki is None:
        # Fallback: generate passages from NQ/TriviaQA context or use simple text
        print("  Wikipedia not available, using NQ passages as corpus fallback...")
        nq = load_dataset("nq_open", split="train")
        passages = []
        for i, ex in enumerate(nq):
            q = ex["question"]
            for ans in ex["answer"]:
                passages.append({"text": f"{q} {ans}", "title": "nq", "id": f"nq_{i}"})
            if len(passages) >= n_passages:
                break
        passages = passages[:n_passages]
        with open(corpus_path, "w") as f:
            json.dump(passages, f)
        print(f"  Corpus: {len(passages)} passages")
        return passages

    rng = random.Random(seed)
    passages = []
    # Stream and collect passages
    for i, article in enumerate(wiki):
        if i >= n_passages:
            break
        text = article["text"]
        # Split into ~100 word chunks
        words = text.split()
        for j in range(0, len(words), 100):
            chunk = " ".join(words[j:j+100])
            if len(chunk) > 50:
                passages.append({
                    "text": chunk,
                    "title": article["title"],
                    "id": f"wiki_{i}_{j//100}"
                })
        if len(passages) >= n_passages:
            break

    passages = passages[:n_passages]
    rng.shuffle(passages)

    with open(corpus_path, "w") as f:
        json.dump(passages, f)
    print(f"  Corpus: {len(passages)} passages")
    return passages


def retrieve_passages_bm25(questions, corpus, top_k=5):
    """BM25 retrieval for a list of questions."""
    from rank_bm25 import BM25Okapi
    import re

    print(f"Building BM25 index over {len(corpus)} passages...")
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    results = []
    for i, q in enumerate(questions):
        if i % 100 == 0:
            print(f"  Retrieving {i}/{len(questions)}...")
        tokenized_q = q.lower().split()
        scores = bm25.get_scores(tokenized_q)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved = [corpus[idx]["text"] for idx in top_indices]
        results.append(retrieved)

    return results


def prepare_irrelevant_passages(corpus, n_pool=1000, n_per_query=5, seed=42):
    """Prepare pool of irrelevant passages for control condition."""
    rng = random.Random(seed + 1000)  # different seed
    # Sample from end of corpus (less likely to overlap with retrieved)
    pool_start = max(0, len(corpus) - n_pool * 2)
    pool = corpus[pool_start:pool_start + n_pool * 2]
    rng.shuffle(pool)
    pool = pool[:n_pool]
    return [p["text"] for p in pool]
