"""Stage 1: Download datasets, set up retrieval, preprocess data."""
import os
import sys
import json
import random
import time
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEEDS = [42, 43, 44]
N_QUESTIONS = 500
TOP_K = 10  # retrieve top-10, use top-5 for main experiments


def download_and_subsample():
    """Download NQ, TriviaQA, PopQA and subsample."""
    print("=== Downloading datasets ===")

    # Natural Questions (open)
    print("Loading NQ...")
    nq = load_dataset("nq_open", split="validation", trust_remote_code=True)
    print(f"  NQ validation: {len(nq)} questions")

    # TriviaQA
    print("Loading TriviaQA...")
    tqa = load_dataset("trivia_qa", "unfiltered", split="validation", trust_remote_code=True)
    print(f"  TriviaQA validation: {len(tqa)} questions")

    # PopQA
    print("Loading PopQA...")
    popqa = load_dataset("akariasai/PopQA", split="test", trust_remote_code=True)
    print(f"  PopQA: {len(popqa)} questions")

    datasets_raw = {
        "nq": [(q["question"], q["answer"]) for q in nq],
        "triviaqa": [(q["question"], q["answer"]["aliases"] if q["answer"]["aliases"] else [q["answer"]["value"]]) for q in tqa],
        "popqa": [(q["question"], [q["possible_answers"]] if isinstance(q["possible_answers"], str) else q["possible_answers"],
                    q.get("s_pop", 0.0)) for q in popqa],
    }

    # Subsample per seed
    for dataset_name, data in datasets_raw.items():
        for seed in SEEDS:
            rng = random.Random(seed)
            indices = rng.sample(range(len(data)), min(N_QUESTIONS, len(data)))
            samples = []
            for idx in indices:
                item = data[idx]
                if dataset_name == "popqa":
                    question, answers, s_pop = item
                    if isinstance(answers, str):
                        answers = [answers]
                    samples.append({
                        "question": question,
                        "gold_answers": answers,
                        "s_pop": float(s_pop) if s_pop else 0.0
                    })
                elif dataset_name == "nq":
                    question, answers = item
                    if isinstance(answers, str):
                        answers = [answers]
                    samples.append({
                        "question": question,
                        "gold_answers": answers
                    })
                else:  # triviaqa
                    question, answers = item
                    if isinstance(answers, str):
                        answers = [answers]
                    samples.append({
                        "question": question,
                        "gold_answers": answers
                    })

            path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"  Saved {len(samples)} samples to {path}")


def build_retrieval_corpus():
    """Build BM25 index from Wikipedia passages."""
    corpus_path = os.path.join(DATA_DIR, "wiki_corpus.json")
    index_path = os.path.join(DATA_DIR, "bm25_index.json")

    if os.path.exists(corpus_path):
        print("Wikipedia corpus already exists, skipping download.")
        return

    print("=== Building Wikipedia retrieval corpus ===")
    print("Loading Wikipedia passages from HuggingFace (wiki_dpr)...")

    # Load a subset of Wikipedia passages for BM25
    # Use wiki_dpr which has pre-chunked 100-word passages
    try:
        wiki = load_dataset(
            "wiki_dpr", "psgs_w100.nq.exact",
            split="train",
            trust_remote_code=True
        )
        print(f"  Loaded {len(wiki)} passages")
        # Take a manageable subset (2M passages)
        max_passages = 2_000_000
        corpus = []
        for i, passage in enumerate(wiki):
            if i >= max_passages:
                break
            corpus.append({
                "id": passage.get("id", i),
                "title": passage.get("title", ""),
                "text": passage.get("text", "")
            })
    except Exception as e:
        print(f"  wiki_dpr failed: {e}")
        print("  Falling back to wikipedia simple subset...")
        try:
            wiki = load_dataset("wikipedia", "20220301.simple", split="train", trust_remote_code=True)
            print(f"  Loaded {len(wiki)} articles, chunking...")
            corpus = []
            for article in wiki:
                text = article["text"]
                title = article["title"]
                # Chunk into ~100 word passages
                words = text.split()
                for j in range(0, len(words), 100):
                    chunk = " ".join(words[j:j+100])
                    if len(chunk.split()) >= 20:
                        corpus.append({
                            "id": len(corpus),
                            "title": title,
                            "text": chunk
                        })
                if len(corpus) >= 500_000:
                    break
        except Exception as e2:
            print(f"  Wikipedia fallback also failed: {e2}")
            print("  Creating minimal synthetic corpus for testing...")
            corpus = [{"id": i, "title": f"Doc {i}", "text": f"This is document {i}."} for i in range(1000)]

    print(f"  Corpus size: {len(corpus)} passages")
    with open(corpus_path, "w") as f:
        json.dump(corpus, f)
    print(f"  Saved corpus to {corpus_path}")


def retrieve_passages():
    """Run BM25 retrieval for all questions."""
    corpus_path = os.path.join(DATA_DIR, "wiki_corpus.json")

    print("=== Loading corpus for BM25 retrieval ===")
    with open(corpus_path) as f:
        corpus = json.load(f)

    texts = [p["text"] for p in corpus]
    titles = [p["title"] for p in corpus]
    print(f"  Tokenizing {len(texts)} passages for BM25...")
    tokenized_corpus = [doc.lower().split() for doc in texts]
    print("  Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("  BM25 index built.")

    # Retrieve for all dataset/seed combinations
    for dataset_name in ["nq", "triviaqa", "popqa"]:
        for seed in SEEDS:
            data_path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}.json")
            out_path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}_retrieved.json")

            if os.path.exists(out_path):
                print(f"  Skipping {out_path} (already exists)")
                continue

            with open(data_path) as f:
                samples = json.load(f)

            print(f"  Retrieving for {dataset_name} seed {seed} ({len(samples)} questions)...")
            for i, sample in enumerate(samples):
                query_tokens = sample["question"].lower().split()
                scores = bm25.get_scores(query_tokens)
                top_indices = np.argsort(scores)[-TOP_K:][::-1]
                passages = []
                for idx in top_indices:
                    passages.append({
                        "title": titles[idx],
                        "text": texts[idx],
                        "score": float(scores[idx])
                    })
                sample["retrieved_passages"] = passages

                if (i + 1) % 100 == 0:
                    print(f"    {i+1}/{len(samples)} done")

            with open(out_path, "w") as f:
                json.dump(samples, f, indent=2)
            print(f"  Saved to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    download_and_subsample()
    build_retrieval_corpus()
    retrieve_passages()
    elapsed = time.time() - t0
    print(f"\n=== Data preparation complete in {elapsed/60:.1f} minutes ===")
