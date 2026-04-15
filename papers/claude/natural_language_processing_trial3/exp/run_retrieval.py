"""Run TF-IDF retrieval for all questions using the Wikipedia corpus."""
import os
import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEEDS = [42, 43, 44]
DATASETS = ["nq", "triviaqa", "popqa"]
TOP_K = 10
MAX_PASSAGES = 200_000


def main():
    t0 = time.time()
    corpus_path = os.path.join(DATA_DIR, "wiki_corpus.json")
    print("Loading corpus...")
    with open(corpus_path) as f:
        corpus = json.load(f)

    corpus = corpus[:MAX_PASSAGES]
    texts = [p["text"] for p in corpus]
    titles = [p["title"] for p in corpus]
    print(f"Using {len(texts)} passages")

    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words='english',
        dtype=np.float32
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}, built in {(time.time()-t0):.1f}s")

    # Collect all queries first for batch processing
    all_queries = []
    query_map = []  # (dataset, seed, idx)
    for dataset_name in DATASETS:
        for seed in SEEDS:
            data_path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}.json")
            with open(data_path) as f:
                samples = json.load(f)
            for i, s in enumerate(samples):
                all_queries.append(s["question"])
                query_map.append((dataset_name, seed, i))

    print(f"Total queries: {len(all_queries)}")
    print("Computing query TF-IDF vectors...")
    query_vectors = vectorizer.transform(all_queries)

    print("Computing similarities (batch)...")
    # Process in chunks to manage memory
    chunk_size = 500
    all_top_indices = []
    all_top_scores = []
    for start in range(0, len(all_queries), chunk_size):
        end = min(start + chunk_size, len(all_queries))
        sim = (query_vectors[start:end] @ tfidf_matrix.T).toarray()
        for row in sim:
            top_idx = np.argsort(row)[-TOP_K:][::-1]
            all_top_indices.append(top_idx)
            all_top_scores.append(row[top_idx])
        print(f"  Processed {end}/{len(all_queries)}")

    # Now assemble results per dataset/seed
    loaded_data = {}
    for dataset_name in DATASETS:
        for seed in SEEDS:
            data_path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}.json")
            with open(data_path) as f:
                loaded_data[(dataset_name, seed)] = json.load(f)

    print("Assembling retrieved passages...")
    for qi, (dataset_name, seed, idx) in enumerate(query_map):
        top_idx = all_top_indices[qi]
        top_scores = all_top_scores[qi]
        passages = []
        for rank, (pidx, score) in enumerate(zip(top_idx, top_scores)):
            passages.append({
                "title": titles[pidx],
                "text": texts[pidx],
                "score": float(score)
            })
        loaded_data[(dataset_name, seed)][idx]["retrieved_passages"] = passages

    # Save
    for dataset_name in DATASETS:
        for seed in SEEDS:
            out_path = os.path.join(DATA_DIR, f"{dataset_name}_seed{seed}_retrieved.json")
            with open(out_path, "w") as f:
                json.dump(loaded_data[(dataset_name, seed)], f, indent=2)
            print(f"  Saved {out_path}")

    total = time.time() - t0
    print(f"\nAll retrieval done in {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
